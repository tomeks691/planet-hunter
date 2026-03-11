import logging
from typing import Optional
import lightkurve as lk
import numpy as np
from astroquery.mast import Catalogs
from planet_hunter.models import StarInfo

log = logging.getLogger(__name__)


def fetch_star_info(tic_id: int) -> StarInfo:
    """Query TIC catalog for stellar parameters."""
    log.info("Fetching TIC %d catalog info", tic_id)
    try:
        result = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC", radius=0.001)
        if result is None or len(result) == 0:
            return StarInfo(tic_id=tic_id)
        row = result[0]
        return StarInfo(
            tic_id=tic_id,
            ra=_safe_float(row.get("ra")),
            dec=_safe_float(row.get("dec")),
            tmag=_safe_float(row.get("Tmag")),
            radius=_safe_float(row.get("rad")),
            teff=_safe_float(row.get("Teff")),
        )
    except Exception as e:
        log.warning("TIC catalog query failed for %d: %s", tic_id, e)
        return StarInfo(tic_id=tic_id)


def fetch_lightcurves(tic_id: int) -> list:
    """Download all available TESS light curves, keeping the best per sector.

    For each sector, prefers the highest-cadence SPOC product (20s > 2min).
    Falls back to other authors (QLP, TESS-SPOC, etc.) if SPOC is unavailable.
    """
    log.info("Searching MAST for TIC %d light curves (all sources)", tic_id)
    search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
    if search is None or len(search) == 0:
        log.warning("No light curves found for TIC %d", tic_id)
        return []

    log.info("Found %d light curve products for TIC %d", len(search), tic_id)

    # Download individually to skip broken/unsupported products
    by_sector: dict[int, list] = {}
    for i in range(len(search)):
        try:
            log.info("TIC %d: downloading product %d/%d", tic_id, i + 1, len(search))
            lc = search[i].download()
            if lc is None or not hasattr(lc, "flux") or lc.flux is None:
                continue
            sector = get_sector_numbers(lc)
            if sector is None:
                continue
            by_sector.setdefault(sector, []).append(lc)
        except Exception as e:
            log.warning("Skipping product %d/%d: %s", i + 1, len(search), e)
            continue

    if not by_sector:
        log.warning("No usable light curves for TIC %d", tic_id)
        return []

    # Author priority: SPOC > TESS-SPOC > QLP > others
    author_rank = {"SPOC": 0, "TESS-SPOC": 1, "QLP": 2}

    curves = []
    for sector in sorted(by_sector):
        candidates = by_sector[sector]
        # Score: lower is better (best author, then shortest cadence)
        def _score(lc):
            meta = getattr(lc, "meta", {})
            author = str(meta.get("AUTHOR", ""))
            cadence = meta.get("TIMEDEL", 1) * 86400  # days -> seconds
            rank = author_rank.get(author, 99)
            return (rank, cadence)

        best = min(candidates, key=_score)
        meta = getattr(best, "meta", {})
        cadence = meta.get("TIMEDEL", 0) * 86400
        author = meta.get("AUTHOR", "?")
        log.info(
            "Sector %d: selected %s %.0fs cadence (%d points)",
            sector, author, cadence, len(best.flux),
        )
        curves.append(best)

    log.info("Selected %d sector light curves for TIC %d", len(curves), tic_id)
    return curves


def get_sector_numbers(lc) -> Optional[int]:
    """Extract sector number from a light curve object."""
    meta = getattr(lc, "meta", {})
    sector = meta.get("SECTOR")
    return int(sector) if sector is not None else None


def get_cadence_seconds(lc) -> float:
    """Return cadence of a light curve in seconds."""
    meta = getattr(lc, "meta", {})
    timedel = meta.get("TIMEDEL")
    if timedel is not None:
        return float(timedel) * 86400  # days -> seconds
    # Estimate from timestamps
    if hasattr(lc, "time") and len(lc.time) > 1:
        dt = np.nanmedian(np.diff(lc.time.value)) * 86400
        return float(dt)
    return 120.0  # default assumption


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None
