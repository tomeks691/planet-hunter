import logging
import numpy as np
from typing import Optional

log = logging.getLogger(__name__)


def _flatten_window(cadence_seconds: float) -> int:
    """Compute flatten window_length (~1.25 days) scaled to cadence."""
    target_days = 1.25
    pts = int(target_days * 86400 / cadence_seconds)
    # Must be odd and at least 101
    pts = max(pts, 101)
    if pts % 2 == 0:
        pts += 1
    return pts


def clean_lightcurve(lc):
    """Normalize, remove upward outliers, and flatten with a cadence-aware window.

    Returns the cleaned light curve or None if insufficient data remains.
    """
    from planet_hunter.pipeline.fetcher import get_cadence_seconds

    try:
        lc = lc.remove_nans()
        lc = lc.normalize()

        # Only clip upward spikes - keep dips (transits/eclipses)
        lc = lc.remove_outliers(sigma_upper=5.0, sigma_lower=50.0)

        # Flatten with a window scaled to cadence (~1.25 days)
        cadence = get_cadence_seconds(lc)
        window = _flatten_window(cadence)
        lc = lc.flatten(window_length=window)

        if lc is None or len(lc.flux) < 100:
            log.warning("Light curve has < 100 points after cleaning")
            return None

        return lc

    except Exception as e:
        log.warning("Error cleaning light curve: %s", e)
        return None


def stitch_lightcurves(curves: list):
    """Stitch multiple sector light curves into one.

    Each curve should already be cleaned individually.
    Returns the stitched light curve or None.
    """
    if not curves:
        return None
    if len(curves) == 1:
        return curves[0]

    try:
        import lightkurve as lk
        collection = lk.LightCurveCollection(curves)
        stitched = collection.stitch()
        stitched = stitched.remove_nans().normalize()
        return stitched
    except Exception as e:
        log.warning("Error stitching light curves: %s", e)
        longest = max(curves, key=lambda c: len(c.flux))
        return longest
