#!/usr/bin/env python3
"""
retry_errors.py - uruchamiany raz dziennie przez crona.

Logika:
- Pobiera wszystkie analizy z classification='ERROR'
- Jeśli dana gwiazda miala TYLKO 1 blad -> dodaje do kolejki ponownie
- Jesli dana gwiazda miala 2 lub wiecej bledow -> usuwa wszystkie jej bledy z bazy
"""

import sqlite3
import sys
import logging
from datetime import datetime
from pathlib import Path

DB_PATH = Path("/app/data/planet_hunter.db")

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


def retry_errors():
    conn = get_conn()

    # Policz bledy per gwiazda
    rows = conn.execute("""
        SELECT tic_id, COUNT(*) as error_count
        FROM analyses
        WHERE classification = 'ERROR'
        GROUP BY tic_id
    """).fetchall()

    retry_count = 0
    delete_count = 0
    skipped_count = 0

    for row in rows:
        tic_id = row["tic_id"]
        error_count = row["error_count"]

        if error_count >= 2:
            # Drugi blad - usun wszystkie bledy tej gwiazdy
            conn.execute("""
                DELETE FROM analyses
                WHERE tic_id = ? AND classification = 'ERROR'
            """, (tic_id,))
            delete_count += 1
            log.info(f"TIC {tic_id} - usunieto {error_count} bledow (2+ proby)")

        else:
            # Pierwszy blad - sprawdz czy juz nie jest w kolejce
            existing = conn.execute("""
                SELECT id FROM queue
                WHERE tic_id = ? AND status IN ('QUEUED', 'RUNNING')
            """, (tic_id,)).fetchone()

            if existing:
                skipped_count += 1
                log.debug(f"TIC {tic_id} - juz w kolejce, pomijam")
                continue

            # Dodaj do kolejki i od razu usun blad z analyses
            conn.execute("""
                INSERT INTO queue (tic_id, source, priority, status)
                VALUES (?, 'AUTO', 5, 'QUEUED')
            """, (tic_id,))
            conn.execute("""
                DELETE FROM analyses
                WHERE tic_id = ? AND classification = 'ERROR'
            """, (tic_id,))
            retry_count += 1
            log.info(f"TIC {tic_id} - dodano do kolejki i usunieto blad")

    conn.commit()
    conn.close()

    log.info(
        f"Gotowe - retry: {retry_count}, usunieto: {delete_count}, "
        f"pominieto (juz w kolejce): {skipped_count}"
    )
    return retry_count, delete_count


if __name__ == "__main__":
    log.info("=== retry_errors.py start ===")
    retry, deleted = retry_errors()
    log.info(f"=== koniec: {retry} wrocilo do kolejki, {deleted} usunieto ===")
