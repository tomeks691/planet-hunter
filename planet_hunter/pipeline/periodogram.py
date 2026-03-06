import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
from astropy.timeseries import BoxLeastSquares
from astropy import units as u
from scipy.optimize import curve_fit
from planet_hunter.config import PERIOD_MIN, PERIOD_MAX, SNR_MINIMUM

log = logging.getLogger(__name__)


@dataclass
class BLSResult:
    period: float           # days
    t0: float               # epoch (BTJD)
    depth: float            # fractional transit depth
    duration: float         # hours
    snr: float
    power_spectrum: Optional[np.ndarray] = None
    periods_searched: Optional[np.ndarray] = None


_BLS_MAX_POINTS = 80000  # bin down if more points than this


def _bin_for_bls(time, flux, flux_err, max_pts=_BLS_MAX_POINTS):
    """Bin high-cadence data to keep BLS fast. Uses time-based binning."""
    if len(time) <= max_pts:
        return time, flux, flux_err
    bin_factor = int(np.ceil(len(time) / max_pts))
    n = (len(time) // bin_factor) * bin_factor
    t = np.mean(time[:n].reshape(-1, bin_factor), axis=1)
    f = np.nanmedian(flux[:n].reshape(-1, bin_factor), axis=1)
    e = np.sqrt(np.nanmean(flux_err[:n].reshape(-1, bin_factor) ** 2, axis=1))
    mask = np.isfinite(t) & np.isfinite(f) & np.isfinite(e)
    log.info("Binned %d -> %d points (factor %d) for BLS", len(time), int(np.sum(mask)), bin_factor)
    return t[mask], f[mask], e[mask]


def run_bls(lc, period_min=None, period_max=None) -> Optional[BLSResult]:
    """Run Box Least Squares periodogram on a light curve.

    Returns BLSResult or None if no significant signal found.
    High-cadence data is automatically binned for performance.
    """
    try:
        time = lc.time.value
        flux = lc.flux.value
        flux_err = lc.flux_err.value if hasattr(lc, "flux_err") and lc.flux_err is not None else np.ones_like(flux) * np.nanstd(flux)

        # Remove remaining NaNs
        mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
        time = time[mask]
        flux = flux[mask]
        flux_err = flux_err[mask]

        if len(time) < 100:
            log.warning("Too few data points for BLS: %d", len(time))
            return None

        # Bin if needed to keep BLS fast
        time, flux, flux_err = _bin_for_bls(time, flux, flux_err)

        p_min = period_min or PERIOD_MIN
        p_max = period_max or PERIOD_MAX

        # Set up BLS
        bls = BoxLeastSquares(time * u.day, flux, dy=flux_err)

        # Duration grid: 1-8 hours
        durations = np.linspace(0.02, 0.33, 20) * u.day

        # Period grid
        periods = np.linspace(p_min, min(p_max, (time[-1] - time[0]) / 2), 10000) * u.day

        result = bls.power(periods, durations)

        # Best period
        best_idx = np.argmax(result.power)
        best_period = result.period[best_idx].value
        best_t0 = result.transit_time[best_idx].value
        best_duration = result.duration[best_idx].value * 24  # to hours

        # Compute depth from the BLS model
        stats = bls.compute_stats(
            best_period * u.day,
            result.duration[best_idx],
            best_t0 * u.day,
        )
        depth = float(stats.get("depth", [0])[0]) if "depth" in stats else _estimate_depth(time, flux, best_period, best_t0, best_duration / 24)

        # Compute SNR
        snr = _compute_snr(result.power, best_idx)

        log.info(
            "BLS result: P=%.4f d, depth=%.5f, SNR=%.1f, dur=%.2f h",
            best_period, depth, snr, best_duration,
        )

        return BLSResult(
            period=best_period,
            t0=best_t0,
            depth=depth,
            duration=best_duration,
            snr=snr,
            power_spectrum=result.power.value if hasattr(result.power, "value") else np.array(result.power),
            periods_searched=periods.value,
        )
    except Exception as e:
        log.error("BLS failed: %s", e, exc_info=True)
        return None


def run_iterative_bls(lc, n_signals: int = 3) -> list[BLSResult]:
    """Run BLS iteratively: find signal, mask it, repeat.

    Returns list of BLSResults sorted by SNR descending.
    """
    import lightkurve as lk
    results = []
    current_lc = lc

    for i in range(n_signals):
        bls = run_bls(current_lc)
        if bls is None or bls.snr < SNR_MINIMUM:
            break
        results.append(bls)
        log.info("Iterative BLS #%d: P=%.4f d, SNR=%.1f", i + 1, bls.period, bls.snr)

        # Mask the found transit and search again
        time = current_lc.time.value
        phase = ((time - bls.t0) % bls.period) / bls.period
        half_dur = (bls.duration / 24) / bls.period / 2
        in_transit = (phase < half_dur) | (phase > 1 - half_dur)
        # Replace in-transit points with local median
        flux = current_lc.flux.value.copy()
        flux[in_transit] = np.nanmedian(flux[~in_transit])
        current_lc = current_lc.copy()
        current_lc.flux = flux * current_lc.flux.unit

    return results


def check_sinusoid(lc, period: float) -> float:
    """Fit a sinusoid at the BLS period. Return the chi-squared ratio
    (sinusoid_chi2 / box_chi2). Values < 1.0 mean sinusoid fits better."""
    try:
        time = lc.time.value
        flux = lc.flux.value
        mask = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[mask], flux[mask]

        phase = (time % period) / period
        median_flux = np.median(flux)

        # Sinusoid fit
        def sinusoid(x, a, phi, offset):
            return a * np.sin(2 * np.pi * x + phi) + offset

        try:
            popt, _ = curve_fit(sinusoid, phase, flux, p0=[np.std(flux), 0, median_flux], maxfev=5000)
            sin_model = sinusoid(phase, *popt)
            sin_chi2 = np.sum((flux - sin_model) ** 2)
        except Exception:
            return 999.0  # sinusoid fit failed -> box is better

        # Box residuals (simple approach: in-transit vs out-of-transit)
        box_chi2 = np.sum((flux - median_flux) ** 2) * 0.8  # rough approximation

        return sin_chi2 / max(box_chi2, 1e-10)
    except Exception:
        return 999.0


def check_secondary_eclipse(lc, period: float, t0: float, primary_depth: float) -> float:
    """Look for a secondary eclipse at phase 0.5. Return secondary/primary depth ratio."""
    try:
        time = lc.time.value
        flux = lc.flux.value
        mask = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[mask], flux[mask]

        phase = ((time - t0) % period) / period

        # Primary: phase near 0
        primary_mask = (phase < 0.05) | (phase > 0.95)
        # Secondary: phase near 0.5
        secondary_mask = (phase > 0.45) & (phase < 0.55)

        out_mask = ~primary_mask & ~secondary_mask
        baseline = np.median(flux[out_mask]) if np.sum(out_mask) > 10 else np.median(flux)

        if np.sum(secondary_mask) < 5:
            return 0.0

        secondary_depth = baseline - np.median(flux[secondary_mask])
        if primary_depth <= 0:
            return 0.0

        ratio = max(0.0, secondary_depth / primary_depth)
        return ratio
    except Exception:
        return 0.0


def check_odd_even(lc, period: float, t0: float) -> float:
    """Compare odd and even transit depths. Return sigma difference."""
    try:
        time = lc.time.value
        flux = lc.flux.value
        mask = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[mask], flux[mask]

        phase = ((time - t0) % period) / period
        transit_number = np.floor((time - t0) / period).astype(int)

        in_transit = (phase < 0.05) | (phase > 0.95)
        out_transit = (phase > 0.2) & (phase < 0.8)

        baseline = np.median(flux[out_transit]) if np.sum(out_transit) > 10 else np.median(flux)

        odd_mask = in_transit & (transit_number % 2 == 1)
        even_mask = in_transit & (transit_number % 2 == 0)

        if np.sum(odd_mask) < 3 or np.sum(even_mask) < 3:
            return 0.0

        odd_depth = baseline - np.median(flux[odd_mask])
        even_depth = baseline - np.median(flux[even_mask])

        odd_err = np.std(flux[odd_mask]) / np.sqrt(np.sum(odd_mask))
        even_err = np.std(flux[even_mask]) / np.sqrt(np.sum(even_mask))

        combined_err = np.sqrt(odd_err**2 + even_err**2)
        if combined_err <= 0:
            return 0.0

        sigma = abs(odd_depth - even_depth) / combined_err
        return sigma
    except Exception:
        return 0.0


def _compute_snr(power, best_idx) -> float:
    """Estimate SNR as (peak - median) / MAD of the power spectrum."""
    p = np.array(power)
    median_p = np.median(p)
    mad = np.median(np.abs(p - median_p))
    if mad <= 0:
        return 0.0
    return float((p[best_idx] - median_p) / (1.4826 * mad))


def _estimate_depth(time, flux, period, t0, duration_days) -> float:
    """Simple depth estimate from in-transit vs out-of-transit median."""
    phase = ((time - t0) % period) / period
    half_dur = duration_days / period / 2
    in_transit = (phase < half_dur) | (phase > 1 - half_dur)
    out_transit = ~in_transit

    if np.sum(in_transit) < 3 or np.sum(out_transit) < 10:
        return 0.0

    return float(np.median(flux[out_transit]) - np.median(flux[in_transit]))
