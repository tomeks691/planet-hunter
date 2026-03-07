#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_training_db as b


def create_source_db(path: Path):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tic_id INTEGER,
            classification TEXT,
            period REAL,
            depth REAL,
            snr REAL,
            duration REAL,
            secondary_depth REAL,
            odd_even_sigma REAL,
            sinusoid_better INTEGER,
            sectors_checked INTEGER,
            sectors_detected INTEGER,
            planet_radius REAL,
            equilibrium_temp REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE stars (
            tic_id INTEGER PRIMARY KEY,
            tmag REAL,
            teff REAL,
            radius REAL
        )
        """
    )

    rows = [
        (111, "KNOWN_PLANET", 5.0, 0.01, 20.0, 2.0, 0.001, 0.2, 0, 3, 2, 1.0, 500.0),
        (222, "KNOWN_PLANET", 6.0, 0.02, 18.0, 3.0, 0.002, 0.3, 0, 3, 2, 1.2, 600.0),
        (333, "FALSE_POSITIVE", 7.0, 0.03, 10.0, 3.5, 0.01, 5.0, 0, 3, 1, 0.0, 0.0),
    ]
    cur.executemany(
        """
        INSERT INTO analyses (
            tic_id, classification, period, depth, snr, duration,
            secondary_depth, odd_even_sigma, sinusoid_better,
            sectors_checked, sectors_detected, planet_radius, equilibrium_temp
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )

    stars = [(111, 10.0, 5500.0, 1.0), (222, 11.0, 5600.0, 1.1), (333, 12.0, 5700.0, 1.2)]
    cur.executemany("INSERT INTO stars (tic_id, tmag, teff, radius) VALUES (?,?,?,?)", stars)

    conn.commit()
    conn.close()


def main():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        src = td_path / "src.db"
        out = td_path / "ml.db"
        bl = td_path / "blacklist.json"

        create_source_db(src)
        bl.write_text(json.dumps({"tic_ids": [222]}), encoding="utf-8")

        b.DB_PATH = src
        b.ML_DB_PATH = out
        b.BLACKLIST_PATH = bl

        b.build()

        conn = sqlite3.connect(out)
        cur = conn.cursor()
        total = cur.execute("SELECT COUNT(*) FROM training_data").fetchone()[0]
        kp = cur.execute("SELECT COUNT(*) FROM training_data WHERE label='KNOWN_PLANET'").fetchone()[0]
        banned = cur.execute("SELECT COUNT(*) FROM training_data WHERE tic_id=222").fetchone()[0]
        conn.close()

        assert total == 2, total
        assert kp == 1, kp
        assert banned == 0, banned

    print("build_training_db blacklist check passed")


if __name__ == "__main__":
    main()
