import logging
import random
import urllib.request
from planet_hunter import db

log = logging.getLogger(__name__)

# Cache: sector -> list of TIC IDs (refreshed when exhausted)
_sector_cache: dict[int, list[int]] = {}
_toi_tic_ids: set[int] = set()  # All known TOI TIC IDs (loaded once)
_toi_loaded: bool = False
_MAX_SECTOR = 96


def _load_all_tois() -> None:
    """Load all known TOI TIC IDs from ExoFOP (once)."""
    global _toi_tic_ids, _toi_loaded
    if _toi_loaded:
        return
    try:
        log.info("Loading all TOI TIC IDs from ExoFOP...")
        url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=60)
        data = resp.read().decode()
        # Parse CSV - TIC ID is first column
        for line in data.split("\n")[1:]:  # Skip header
            parts = line.split(",")
            if len(parts) >= 1:
                try:
                    tic_id = int(parts[0])
                    _toi_tic_ids.add(tic_id)
                except (ValueError, IndexError):
                    continue
        log.info("Loaded %d known TOI TIC IDs", len(_toi_tic_ids))
        _toi_loaded = True
    except Exception as e:
        log.error("Failed to load TOI list: %s", e)
        _toi_loaded = True  # Don't retry on error


def _is_known_toi(tic_id: int) -> bool:
    """Check if TIC ID is already a known TOI."""
    _load_all_tois()
    return tic_id in _toi_tic_ids


def _query_sector_targets(sector: int) -> list[int]:
    """Query MAST for all TIC IDs with TESS SPOC data in a given sector."""
    try:
        from astroquery.mast import Observations

        log.info("Querying MAST for SPOC targets in sector %d", sector)
        obs = Observations.query_criteria(
            obs_collection="TESS",
            dataproduct_type="timeseries",
            provenance_name="SPOC",
            sequence_number=sector,
            t_exptime=[20, 130],
        )
        if obs is None or len(obs) == 0:
            return []

        tic_ids = set()
        for row in obs:
            try:
                tic_ids.add(int(row["target_name"]))
            except (ValueError, TypeError):
                continue

        log.info("Sector %d: found %d unique TIC IDs", sector, len(tic_ids))
        return list(tic_ids)
    except Exception as e:
        log.error("MAST query failed for sector %d: %s", sector, e)
        return []


def _already_analyzed(tic_id: int) -> bool:
    return db.get_latest_analysis(tic_id) is not None


def find_fresh_targets(count: int = 5) -> list[int]:
    """Pick random TIC IDs from MAST that have SPOC data.

    Randomly selects a TESS sector, queries MAST for all SPOC targets,
    and picks unanalyzed ones.
    """
    # Try up to 3 random sectors to find fresh targets
    for _ in range(3):
        sector = random.randint(1, _MAX_SECTOR)

        # Use cache if available
        if sector not in _sector_cache or not _sector_cache[sector]:
            targets = _query_sector_targets(sector)
            if not targets:
                continue
            random.shuffle(targets)
            _sector_cache[sector] = targets

        # Filter out already analyzed
        pool = [t for t in _sector_cache[sector] if not _already_analyzed(t)]

        if not pool:
            log.info("Sector %d: all targets already analyzed, trying another", sector)
            _sector_cache.pop(sector, None)
            continue

        # Filter out known TOIs - only keep unknown targets
        picked = []
        checked = 0
        for tic_id in pool:
            if len(picked) >= count:
                break
            checked += 1
            if not _is_known_toi(tic_id):
                picked.append(tic_id)
                log.info("TIC %d: NOT a known TOI - adding to queue", tic_id)
            else:
                log.debug("TIC %d: known TOI - skipping", tic_id)
            # Remove from cache regardless
            if tic_id in _sector_cache[sector]:
                _sector_cache[sector].remove(tic_id)

        if not picked:
            log.info("Sector %d: checked %d targets, all are known TOIs", sector, checked)
            continue

        log.info("Picked %d NEW (non-TOI) targets from sector %d (pool: %d remaining)",
                 len(picked), sector, len(_sector_cache[sector]))
        return picked

    log.warning("Could not find fresh targets after 3 sector attempts")
    return []
