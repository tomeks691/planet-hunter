import logging
import shutil
import threading
import time
from pathlib import Path
from planet_hunter import db
from planet_hunter.models import (
    AnalysisResult, Classification, QueueSource,
)
from planet_hunter.pipeline.fetcher import fetch_star_info, fetch_lightcurves, get_sector_numbers, get_cadence_seconds
from planet_hunter.pipeline.cleaner import clean_lightcurve, stitch_lightcurves
from planet_hunter.pipeline.periodogram import (
    run_bls, run_iterative_bls, check_sinusoid, check_secondary_eclipse, check_odd_even,
)
from planet_hunter.pipeline.classifier import classify
from planet_hunter.pipeline.properties import compute_properties
from planet_hunter.pipeline.plots import generate_all_plots
from planet_hunter.config import SNR_MINIMUM

# Requeue RUNNING items that look stale (worker crash/restart leftovers)
STUCK_RUNNING_HOURS = 6
STUCK_SWEEP_INTERVAL_SECONDS = 300

log = logging.getLogger(__name__)


def _pick_best_signal(lc, signals):
    """From iterative BLS results, pick the most planet-like signal.

    Prefers signals with depth < 5% (not EB) and high SNR.
    Falls back to highest-SNR signal.
    """
    from planet_hunter.config import DEPTH_MAX_PLANET, SECONDARY_ECLIPSE_RATIO

    candidates = []
    for sig in signals:
        secondary = check_secondary_eclipse(lc, sig.period, sig.t0, sig.depth)
        is_eb_like = sig.depth > DEPTH_MAX_PLANET or secondary > SECONDARY_ECLIPSE_RATIO
        candidates.append((sig, is_eb_like))

    # Prefer non-EB signals
    non_eb = [(s, eb) for s, eb in candidates if not eb]
    if non_eb:
        return max(non_eb, key=lambda x: x[0].snr)[0]
    # All look like EBs, return highest SNR
    return max(candidates, key=lambda x: x[0].snr)[0]


def _clear_lightkurve_cache(tic_id: int) -> None:
    """Remove cached FITS files for a given TIC ID from lightkurve cache."""
    cache_dir = Path("/root/.lightkurve/cache/mastDownload")
    if not cache_dir.exists():
        return
    padded = f"{tic_id:016d}"
    cleared = 0
    for subdir in cache_dir.iterdir():
        if not subdir.is_dir():
            continue
        for entry in subdir.iterdir():
            if padded in entry.name:
                try:
                    shutil.rmtree(entry)
                    cleared += 1
                except Exception as e:
                    log.warning("Cache cleanup failed for %s: %s", entry, e)
    log.info("TIC %d: cache cleared (%d entries removed)", tic_id, cleared)


def run_pipeline(tic_id: int) -> AnalysisResult:
    """Full pipeline for a single TIC ID. Returns AnalysisResult."""
    result = AnalysisResult(tic_id=tic_id, classification=Classification.PROCESSING)

    # 1. Fetch star info
    star = fetch_star_info(tic_id)
    db.upsert_star(star)

    # 2. Fetch light curves
    raw_curves = fetch_lightcurves(tic_id)
    if not raw_curves:
        result.classification = Classification.ERROR
        result.error_message = "No light curves found on MAST"
        return result

    # 3. Clean each sector
    cleaned_curves = []
    sector_numbers = []
    total_points = 0
    for lc in raw_curves:
        sector = get_sector_numbers(lc)
        cadence = get_cadence_seconds(lc)
        cleaned = clean_lightcurve(lc)
        if cleaned is not None:
            cleaned_curves.append(cleaned)
            total_points += len(cleaned.flux)
            if sector is not None:
                sector_numbers.append(sector)
            log.info("Sector %s: %.0fs cadence, %d pts after cleaning",
                     sector, cadence, len(cleaned.flux))
    log.info("Total: %d sectors, %d data points", len(cleaned_curves), total_points)

    if not cleaned_curves:
        result.classification = Classification.ERROR
        result.error_message = "All light curves rejected during cleaning"
        return result

    result.sectors_checked = len(cleaned_curves)

    # 4. Stitch all sectors
    stitched = stitch_lightcurves(cleaned_curves)
    if stitched is None:
        result.classification = Classification.ERROR
        result.error_message = "Failed to stitch light curves"
        return result

    # 5. Iterative BLS - find up to 3 signals
    all_signals = run_iterative_bls(stitched, n_signals=3)
    if not all_signals:
        result.classification = Classification.NOISE
        result.error_message = "BLS found no periodic signal"
        return result

    # Pick the best signal: prefer the one most likely to be a planet
    # (passes basic EB checks and has reasonable depth)
    bls = _pick_best_signal(stitched, all_signals)

    result.period = bls.period
    result.depth = bls.depth
    result.snr = bls.snr
    result.duration = bls.duration

    # 6. Diagnostic checks
    sin_ratio = check_sinusoid(stitched, bls.period)
    result.sinusoid_better = sin_ratio < 1.0

    result.secondary_depth = check_secondary_eclipse(stitched, bls.period, bls.t0, bls.depth)
    result.odd_even_sigma = check_odd_even(stitched, bls.period, bls.t0)

    # 7. Multi-sector validation
    if result.snr >= SNR_MINIMUM and len(cleaned_curves) > 1:
        sectors_with_signal = 0
        sample = cleaned_curves[:8]
        for sector_lc in sample:
            sector_bls = run_bls(sector_lc)
            if sector_bls is not None and sector_bls.snr >= SNR_MINIMUM * 0.5:
                ratio = sector_bls.period / bls.period
                if abs(ratio - 1.0) < 0.10 or abs(ratio - 0.5) < 0.10 or abs(ratio - 2.0) < 0.10:
                    sectors_with_signal += 1
        result.sectors_checked = len(sample)
        result.sectors_detected = sectors_with_signal

    # 8. Classify
    result.classification = classify(result)

    # 9. Compute planet properties (if candidate)
    if result.classification in (Classification.PLANET_CANDIDATE, Classification.MANUAL_REVIEW):
        compute_properties(result, star)

    # 10. Generate plots
    try:
        plot_paths = generate_all_plots(tic_id, stitched, bls, result.classification.value)
        result.plot_lightcurve = plot_paths.get("plot_lightcurve")
        result.plot_periodogram = plot_paths.get("plot_periodogram")
        result.plot_phase_fold = plot_paths.get("plot_phase_fold")
        result.plot_diagnostic = plot_paths.get("plot_diagnostic")
    except Exception as e:
        log.error("Plot generation failed for TIC %d: %s", tic_id, e)

    return result


class PipelineRunner:
    """Background thread that processes the queue."""

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("Pipeline runner started")

    def stop(self):
        self._stop_event.set()
        log.info("Pipeline runner stop requested")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self):
        last_stuck_sweep = 0.0

        while not self._stop_event.is_set():
            now = time.time()
            if now - last_stuck_sweep >= STUCK_SWEEP_INTERVAL_SECONDS:
                fixed = db.requeue_stuck_running(hours=STUCK_RUNNING_HOURS)
                if fixed > 0:
                    log.warning(
                        "Recovered %d stale RUNNING queue item(s) -> QUEUED",
                        fixed,
                    )
                last_stuck_sweep = now

            item = db.next_in_queue()
            if item is None:
                time.sleep(2)
                continue

            tic_id = item["tic_id"]
            queue_id = item["id"]
            source = item.get("source")
            log.info("Processing TIC %d (queue #%d)", tic_id, queue_id)

            try:
                result = run_pipeline(tic_id)

                # Mark as KNOWN_PLANET only when we actually detected a periodic signal.
                if db.is_known_planet(tic_id) and result.period is not None:
                    result.classification = Classification.KNOWN_PLANET

                # For ML training queue: skip persisting rows without period.
                # This keeps training data clean (usable examples only).
                if source == QueueSource.ML_TRAINING.value and result.period is None:
                    db.finish_queue_item(queue_id, failed=False)
                    log.info(
                        "TIC %d skipped (ML_TRAINING, no period detected)",
                        tic_id,
                    )
                    continue

                analysis_id = db.insert_analysis(result)
                db.finish_queue_item(queue_id, failed=False)
                log.info(
                    "TIC %d done -> %s (analysis #%d)",
                    tic_id, result.classification.value, analysis_id,
                )
            except Exception as e:
                log.error("Pipeline failed for TIC %d: %s", tic_id, e, exc_info=True)
                error_result = AnalysisResult(
                    tic_id=tic_id,
                    classification=Classification.ERROR,
                    error_message=str(e),
                )
                db.insert_analysis(error_result)
                db.finish_queue_item(queue_id, failed=True)
            finally:
                _clear_lightkurve_cache(tic_id)
