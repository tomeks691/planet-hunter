#!/usr/bin/env python3
import logging
import os
import sqlite3
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

PH_SQLITE_PATH = Path(os.getenv("PH_SQLITE_PATH", "/app/data/planet_hunter.db"))
ML_SQLITE_PATH = Path(os.getenv("ML_SQLITE_PATH", "/app/data/ml_training.db"))
DATABASE_URL = os.getenv("DATABASE_URL")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sqlite-to-postgres")


def sqlite_conn(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {path}")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def pg_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is required")
    return psycopg2.connect(DATABASE_URL)


def table_exists_sqlite(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def fetch_all(conn: sqlite3.Connection, table: str):
    return conn.execute(f"SELECT * FROM {table}").fetchall()


def ensure_pg_schema(pg):
    with pg.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS stars (
                tic_id      BIGINT PRIMARY KEY,
                ra          DOUBLE PRECISION,
                dec         DOUBLE PRECISION,
                tmag        DOUBLE PRECISION,
                radius      DOUBLE PRECISION,
                teff        DOUBLE PRECISION,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS analyses (
                id                 BIGSERIAL PRIMARY KEY,
                tic_id             BIGINT NOT NULL REFERENCES stars(tic_id),
                classification     TEXT NOT NULL DEFAULT 'PENDING',
                period             DOUBLE PRECISION,
                depth              DOUBLE PRECISION,
                snr                DOUBLE PRECISION,
                duration           DOUBLE PRECISION,
                planet_radius      DOUBLE PRECISION,
                equilibrium_temp   DOUBLE PRECISION,
                sectors_checked    INTEGER DEFAULT 0,
                sectors_detected   INTEGER DEFAULT 0,
                sinusoid_better    INTEGER DEFAULT 0,
                secondary_depth    DOUBLE PRECISION,
                odd_even_sigma     DOUBLE PRECISION,
                plot_lightcurve    TEXT,
                plot_periodogram   TEXT,
                plot_phase_fold    TEXT,
                plot_diagnostic    TEXT,
                review_notes       TEXT,
                error_message      TEXT,
                ml_planet_score    DOUBLE PRECISION,
                ml_model_version   TEXT,
                ml_decision_source TEXT,
                created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS queue (
                id         BIGSERIAL PRIMARY KEY,
                tic_id     BIGINT NOT NULL,
                source     TEXT NOT NULL DEFAULT 'MANUAL',
                priority   INTEGER NOT NULL DEFAULT 1,
                status     TEXT NOT NULL DEFAULT 'QUEUED',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS known_planet_tids (
                tic_id BIGINT PRIMARY KEY,
                pl_name TEXT,
                source TEXT DEFAULT 'NASA'
            );

            CREATE TABLE IF NOT EXISTS training_data (
                id               BIGSERIAL PRIMARY KEY,
                tic_id           BIGINT NOT NULL,
                label            TEXT NOT NULL,
                period           DOUBLE PRECISION,
                depth            DOUBLE PRECISION,
                snr              DOUBLE PRECISION,
                duration         DOUBLE PRECISION,
                secondary_depth  DOUBLE PRECISION,
                odd_even_sigma   DOUBLE PRECISION,
                sinusoid_better  INTEGER,
                sectors_ratio    DOUBLE PRECISION,
                tmag             DOUBLE PRECISION,
                teff             DOUBLE PRECISION,
                star_radius      DOUBLE PRECISION,
                planet_radius    DOUBLE PRECISION,
                equilibrium_temp DOUBLE PRECISION
            );
            """
        )
    pg.commit()


def copy_rows(pg, table: str, rows, columns: list[str], conflict: str):
    if not rows:
        log.info("%s: no rows to copy", table)
        return 0

    values = [tuple(r[c] for c in columns) for r in rows]
    insert_sql = (
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES %s "
        f"ON CONFLICT {conflict} DO NOTHING"
    )
    with pg.cursor() as cur:
        execute_values(cur, insert_sql, values, page_size=1000)
    pg.commit()

    with pg.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        total = int(cur.fetchone()[0])
    log.info("%s: copied batch=%d, total_in_pg=%d", table, len(rows), total)
    return len(rows)


def sync_sequence(pg, table: str, pk_col: str = "id"):
    if table not in {"analyses", "queue", "training_data"}:
        return
    with pg.cursor() as cur:
        cur.execute(
            f"""
            SELECT setval(
                pg_get_serial_sequence('{table}', '{pk_col}'),
                COALESCE((SELECT MAX({pk_col}) FROM {table}), 1),
                true
            )
            """
        )
    pg.commit()


def main():
    log.info("Starting migration SQLite -> PostgreSQL")
    log.info("PH sqlite: %s", PH_SQLITE_PATH)
    log.info("ML sqlite: %s", ML_SQLITE_PATH)

    ph = sqlite_conn(PH_SQLITE_PATH)
    ml = sqlite_conn(ML_SQLITE_PATH) if ML_SQLITE_PATH.exists() else None
    pg = pg_conn()

    try:
        ensure_pg_schema(pg)

        tables = [
            ("stars", ph, ["tic_id", "ra", "dec", "tmag", "radius", "teff", "created_at"], "(tic_id)"),
            (
                "analyses",
                ph,
                [
                    "id",
                    "tic_id",
                    "classification",
                    "period",
                    "depth",
                    "snr",
                    "duration",
                    "planet_radius",
                    "equilibrium_temp",
                    "sectors_checked",
                    "sectors_detected",
                    "sinusoid_better",
                    "secondary_depth",
                    "odd_even_sigma",
                    "plot_lightcurve",
                    "plot_periodogram",
                    "plot_phase_fold",
                    "plot_diagnostic",
                    "review_notes",
                    "error_message",
                    "ml_planet_score",
                    "ml_model_version",
                    "ml_decision_source",
                    "created_at",
                ],
                "(id)",
            ),
            (
                "queue",
                ph,
                ["id", "tic_id", "source", "priority", "status", "created_at"],
                "(id)",
            ),
            ("known_planet_tids", ph, ["tic_id", "pl_name", "source"], "(tic_id)"),
        ]

        total_rows = 0
        for table, src, cols, conflict in tables:
            if not table_exists_sqlite(src, table):
                log.warning("Skipping missing table in sqlite: %s", table)
                continue
            rows = fetch_all(src, table)
            total_rows += copy_rows(pg, table, rows, cols, conflict)

        if ml and table_exists_sqlite(ml, "training_data"):
            rows = fetch_all(ml, "training_data")
            total_rows += copy_rows(
                pg,
                "training_data",
                rows,
                [
                    "id",
                    "tic_id",
                    "label",
                    "period",
                    "depth",
                    "snr",
                    "duration",
                    "secondary_depth",
                    "odd_even_sigma",
                    "sinusoid_better",
                    "sectors_ratio",
                    "tmag",
                    "teff",
                    "star_radius",
                    "planet_radius",
                    "equilibrium_temp",
                ],
                "(id)",
            )
        else:
            log.info("training_data: source table missing, skipped")

        for tbl in ("analyses", "queue", "training_data"):
            sync_sequence(pg, tbl, "id")

        log.info("Migration finished. Copied rows (attempted inserts): %d", total_rows)
    finally:
        ph.close()
        if ml:
            ml.close()
        pg.close()


if __name__ == "__main__":
    main()
