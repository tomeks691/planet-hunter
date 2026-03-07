#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planet_hunter import db
from planet_hunter.models import Classification


def assert_has_columns(conn: sqlite3.Connection, table: str, required: set[str]):
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    cols = {r[1] for r in rows}
    missing = required - cols
    assert not missing, f"Missing columns in {table}: {missing}"


def create_legacy_analyses_schema(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE stars (
            tic_id      INTEGER PRIMARY KEY,
            ra          REAL,
            dec         REAL,
            tmag        REAL,
            radius      REAL,
            teff        REAL,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE analyses (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            tic_id            INTEGER NOT NULL,
            classification    TEXT NOT NULL DEFAULT 'PENDING',
            period            REAL,
            depth             REAL,
            snr               REAL,
            duration          REAL,
            planet_radius     REAL,
            equilibrium_temp  REAL,
            sectors_checked   INTEGER DEFAULT 0,
            sectors_detected  INTEGER DEFAULT 0,
            sinusoid_better   INTEGER DEFAULT 0,
            secondary_depth   REAL,
            odd_even_sigma    REAL,
            plot_lightcurve   TEXT,
            plot_periodogram  TEXT,
            plot_phase_fold   TEXT,
            plot_diagnostic   TEXT,
            review_notes      TEXT,
            error_message     TEXT,
            created_at        TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (tic_id) REFERENCES stars(tic_id)
        );
        """
    )
    conn.commit()
    conn.close()


def main():
    with tempfile.TemporaryDirectory() as td:
        db_file = Path(td) / "tmp_ci_ml.db"

        # 1) migration test: existing legacy schema should gain ML columns
        create_legacy_analyses_schema(db_file)
        db.DB_PATH = db_file
        db.init_db()

        conn = sqlite3.connect(db_file)
        assert_has_columns(conn, "analyses", {"ml_planet_score", "ml_model_version", "ml_decision_source"})

        # 2) insert fixture rows for filters/snapshot tests
        conn.execute("INSERT INTO stars (tic_id) VALUES (1001)")
        conn.execute("INSERT INTO stars (tic_id) VALUES (1002)")
        conn.execute("INSERT INTO stars (tic_id) VALUES (1003)")

        conn.execute(
            """
            INSERT INTO analyses (
                tic_id, classification, period, depth, snr, duration,
                ml_planet_score, ml_model_version, ml_decision_source
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (1001, Classification.PLANET_CANDIDATE.value, 4.1, 0.01, 15.0, 2.2, 0.91, "v1.0-best", "two_stage"),
        )
        conn.execute(
            """
            INSERT INTO analyses (
                tic_id, classification, period, depth, snr, duration,
                ml_planet_score, ml_model_version, ml_decision_source
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (1002, Classification.MANUAL_REVIEW.value, 5.3, 0.02, 10.0, 2.8, 0.49, "v1.0-best", "two_stage"),
        )
        conn.execute(
            """
            INSERT INTO analyses (
                tic_id, classification, period, depth, snr, duration,
                ml_planet_score, ml_model_version, ml_decision_source
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (1003, Classification.FALSE_POSITIVE.value, 7.0, 0.03, 9.0, 3.0, None, None, "rules"),
        )
        conn.commit()
        conn.close()

        # 3) list_analyses filter checks
        high = db.list_analyses(classification="PLANET_CANDIDATE", min_ml_score=0.8, limit=20)
        assert len(high) == 1 and high[0]["tic_id"] == 1001, high

        v1 = db.list_analyses(model_version="v1.0-best", limit=20)
        assert len(v1) == 2, v1

        # 4) snapshot checks
        snap = db.ml_monitor_snapshot(hours=24)
        assert snap["total"] >= 3, snap
        assert snap["ml_scored"] >= 2, snap
        assert snap["planet_candidate"] >= 1, snap
        assert snap["manual_review"] >= 1, snap
        assert any(v["model_version"] == "v1.0-best" for v in snap["versions"]), snap

    print("ML DB migration/filter/snapshot checks passed")


if __name__ == "__main__":
    main()
