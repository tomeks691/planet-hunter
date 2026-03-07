#!/usr/bin/env python3
"""
build_training_db.py - buduje ml_training.db z danych w planet_hunter.db.

Uruchomić PO przetworzeniu kolejki ML_TRAINING przez pipeline.

Uruchomienie:
    docker exec planet-hunter python3 /app/build_training_db.py

Wynik: /app/data/ml_training.db
"""

import json
import logging
import sqlite3
from pathlib import Path

DB_PATH = Path("/app/data/planet_hunter.db")
ML_DB_PATH = Path("/app/data/ml_training.db")
BLACKLIST_PATH = Path("/app/data/ml/label_blacklist_v1.json")

VALID_LABELS = {"KNOWN_PLANET", "FALSE_POSITIVE", "ECLIPSING_BINARY", "NOISE"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def build():
    log.info("=== build_training_db.py start ===")
    log.info("Źródło: %s", DB_PATH)
    log.info("Wynik:  %s", ML_DB_PATH)

    src = sqlite3.connect(str(DB_PATH))
    src.row_factory = sqlite3.Row

    # Pobierz dane: analyses JOIN stars, tylko rekordy z features BLS
    rows = src.execute("""
        SELECT
            a.tic_id,
            a.classification                                    AS label,
            a.period,
            a.depth,
            a.snr,
            a.duration,
            a.secondary_depth,
            a.odd_even_sigma,
            a.sinusoid_better,
            CASE
                WHEN a.sectors_checked > 0
                THEN CAST(a.sectors_detected AS REAL) / a.sectors_checked
                ELSE NULL
            END                                                 AS sectors_ratio,
            s.tmag,
            s.teff,
            s.radius                                            AS star_radius,
            a.planet_radius,
            a.equilibrium_temp
        FROM analyses a
        LEFT JOIN stars s ON a.tic_id = s.tic_id
        WHERE a.classification IN ('KNOWN_PLANET','FALSE_POSITIVE','ECLIPSING_BINARY','NOISE')
          AND a.period IS NOT NULL
        ORDER BY a.classification, a.id
    """).fetchall()
    src.close()

    log.info("Pobrano %d rekordów z planet_hunter.db", len(rows))

    # Optional blacklist cleanup for suspicious KNOWN_PLANET labels
    blacklisted_tics = set()
    if BLACKLIST_PATH.exists():
        try:
            payload = json.loads(BLACKLIST_PATH.read_text(encoding="utf-8"))
            blacklisted_tics = {int(t) for t in payload.get("tic_ids", [])}
            if blacklisted_tics:
                before = len(rows)
                rows = [
                    r for r in rows
                    if not (r["label"] == "KNOWN_PLANET" and int(r["tic_id"]) in blacklisted_tics)
                ]
                log.warning(
                    "Blacklist applied: removed %d KNOWN_PLANET row(s) by TIC id",
                    before - len(rows),
                )
        except Exception as e:
            log.error("Failed to apply blacklist %s: %s", BLACKLIST_PATH, e)

    # Statystyki per klasa
    from collections import Counter
    counts = Counter(r["label"] for r in rows)
    for label, cnt in sorted(counts.items()):
        log.info("  %-20s %d", label, cnt)

    # Zbuduj ml_training.db
    if ML_DB_PATH.exists():
        ML_DB_PATH.unlink()
        log.info("Usunięto starą %s", ML_DB_PATH)

    dst = sqlite3.connect(str(ML_DB_PATH))
    dst.execute("""
        CREATE TABLE training_data (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            tic_id           INTEGER NOT NULL,
            label            TEXT NOT NULL,
            -- BLS features
            period           REAL,
            depth            REAL,
            snr              REAL,
            duration         REAL,
            -- Pipeline diagnostics
            secondary_depth  REAL,
            odd_even_sigma   REAL,
            sinusoid_better  INTEGER,
            sectors_ratio    REAL,
            -- Stellar
            tmag             REAL,
            teff             REAL,
            star_radius      REAL,
            -- Derived
            planet_radius    REAL,
            equilibrium_temp REAL
        )
    """)

    dst.executemany("""
        INSERT INTO training_data (
            tic_id, label,
            period, depth, snr, duration,
            secondary_depth, odd_even_sigma, sinusoid_better, sectors_ratio,
            tmag, teff, star_radius,
            planet_radius, equilibrium_temp
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        (
            r["tic_id"], r["label"],
            r["period"], r["depth"], r["snr"], r["duration"],
            r["secondary_depth"], r["odd_even_sigma"], r["sinusoid_better"], r["sectors_ratio"],
            r["tmag"], r["teff"], r["star_radius"],
            r["planet_radius"], r["equilibrium_temp"],
        )
        for r in rows
    ])

    dst.commit()

    # Weryfikacja
    total = dst.execute("SELECT COUNT(*) FROM training_data").fetchone()[0]
    log.info("Zapisano %d rekordów do ml_training.db", total)
    for row in dst.execute("SELECT label, COUNT(*) as cnt FROM training_data GROUP BY label ORDER BY cnt DESC").fetchall():
        log.info("  %-20s %d", row[0], row[1])

    # Sprawdź pokrycie features (% non-NULL)
    log.info("Pokrycie features (non-NULL %):")
    features = ["period","depth","snr","duration","secondary_depth","odd_even_sigma",
                "sinusoid_better","sectors_ratio","tmag","teff","star_radius",
                "planet_radius","equilibrium_temp"]
    for feat in features:
        pct = dst.execute(
            f"SELECT ROUND(100.0 * SUM(CASE WHEN {feat} IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) FROM training_data"
        ).fetchone()[0]
        log.info("  %-20s %s%%", feat, pct)

    dst.close()
    log.info("=== koniec ===")


if __name__ == "__main__":
    build()
