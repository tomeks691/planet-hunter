#!/usr/bin/env python3
"""
import_known_planets.py - jednorazowy skrypt do zasilenia bazy danymi treningowymi ML.

Pobiera potwierdzone planety z NASA Exoplanet Archive, zapisuje ich TIC IDs
do tabeli known_planet_tids, a następnie wrzuca do kolejki te które nie mają
jeszcze pełnej analizy (z features BLS).

Uruchomienie (na Beelinku, poza kontenerem):
    docker exec planet-hunter python3 /app/import_known_planets.py
"""

import csv
import io
import logging
import sqlite3
import urllib.request
from pathlib import Path

DB_PATH = Path("/app/data/planet_hunter.db")

NASA_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=SELECT+tic_id,pl_name+FROM+ps"
    "+WHERE+tran_flag=1"
    "+AND+tic_id+IS+NOT+NULL"
    "+AND+pl_orbper+IS+NOT+NULL"
    "+AND+pl_trandep+IS+NOT+NULL"
    "+AND+pl_trandur+IS+NOT+NULL"
    "+AND+st_teff+IS+NOT+NULL"
    "+AND+st_rad+IS+NOT+NULL"
    "+AND+sy_tmag+IS+NOT+NULL"
    "+AND+default_flag=1"
    "&format=csv"
)

ML_TRAINING_PRIORITY = 3  # między MANUAL=1 a AUTO=5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def get_conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def fetch_nasa_planets() -> list[tuple[int, str]]:
    """Pobierz TIC IDs potwierdzonych planet tranzytowych z NASA API."""
    log.info("Pobieranie danych z NASA Exoplanet Archive...")
    req = urllib.request.Request(NASA_URL, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=60)
    data = resp.read().decode()

    records = []
    reader = csv.DictReader(io.StringIO(data))
    for row in reader:
        raw_tic = row.get("tic_id", "").strip()
        pl_name = row.get("pl_name", "").strip()
        if not raw_tic:
            continue
        # Format: "TIC 467929202" lub samo "467929202"
        raw_tic = raw_tic.replace("TIC", "").strip()
        try:
            tic_id = int(raw_tic)
            records.append((tic_id, pl_name))
        except ValueError:
            log.warning("Nie udało się sparsować TIC ID: %r", raw_tic)

    log.info("Pobrano %d rekordów z NASA", len(records))
    return records


def insert_known_planet_tids(conn, records: list[tuple[int, str]]) -> int:
    conn.executemany(
        "INSERT OR IGNORE INTO known_planet_tids (tic_id, pl_name) VALUES (?,?)",
        records,
    )
    count = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    return count


def has_good_analysis(conn, tic_id: int) -> bool:
    """Zwraca True jeśli gwiazda ma już analizę z kompletnymi features BLS."""
    row = conn.execute(
        "SELECT period FROM analyses WHERE tic_id=? AND period IS NOT NULL ORDER BY id DESC LIMIT 1",
        (tic_id,),
    ).fetchone()
    return row is not None


def is_queued_or_running(conn, tic_id: int) -> bool:
    row = conn.execute(
        "SELECT id FROM queue WHERE tic_id=? AND status IN ('QUEUED','RUNNING')",
        (tic_id,),
    ).fetchone()
    return row is not None


def enqueue_for_ml(conn, tic_id: int) -> bool:
    """Dodaj do kolejki z priorytetem ML_TRAINING. Zwraca True jeśli dodano."""
    if is_queued_or_running(conn, tic_id):
        return False
    conn.execute(
        "INSERT INTO queue (tic_id, source, priority, status) VALUES (?,?,?,?)",
        (tic_id, "ML_TRAINING", ML_TRAINING_PRIORITY, "QUEUED"),
    )
    return True


def main():
    log.info("=== import_known_planets.py start ===")

    # 1. Pobierz z NASA
    records = fetch_nasa_planets()
    if not records:
        log.error("Brak danych z NASA - przerywam")
        return

    conn = get_conn()

    # 2. Zapisz do known_planet_tids
    inserted = insert_known_planet_tids(conn, records)
    total_known = conn.execute("SELECT COUNT(*) FROM known_planet_tids").fetchone()[0]
    log.info("Dodano %d nowych do known_planet_tids (łącznie: %d)", inserted, total_known)

    # 3. Sprawdź które nie mają jeszcze dobrej analizy → wrzuć do kolejki
    queued = 0
    skipped_has_analysis = 0
    skipped_already_queued = 0

    tic_ids = [r[0] for r in records]
    for tic_id in tic_ids:
        if has_good_analysis(conn, tic_id):
            skipped_has_analysis += 1
            continue
        if enqueue_for_ml(conn, tic_id):
            queued += 1
        else:
            skipped_already_queued += 1

    conn.commit()
    conn.close()

    log.info(
        "Gotowe: %d dodano do kolejki, %d pominięto (ma analizę), %d pominięto (już w kolejce)",
        queued, skipped_has_analysis, skipped_already_queued,
    )
    log.info("=== koniec ===")


if __name__ == "__main__":
    main()
