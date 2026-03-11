import json
import sqlite3
import time
from typing import Optional

import numpy as np

from planet_hunter.config import DATABASE_URL, DB_PATH
from planet_hunter.models import (
    AnalysisResult,
    Classification,
    QueueItem,
    QueueSource,
    QueueStatus,
    StarInfo,
)

try:
    import psycopg2
    import psycopg2.extras
    from psycopg2 import OperationalError as PgOperationalError
except Exception:  # pragma: no cover - optional in sqlite-only runs
    psycopg2 = None
    PgOperationalError = Exception


def _py(val):
    """Convert numpy types to Python native for DB adapters."""
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return float(val.item()) if val.size == 1 else None
    return val


def _is_sqlite() -> bool:
    return DATABASE_URL.startswith("sqlite")


def _sqlite_path_from_url(url: str) -> str:
    return url.replace("sqlite:///", "", 1)


def _to_sql_params(params):
    return tuple(_py(v) for v in (params or ()))


def _execute_pg_with_retry(cur, sql: str, params=None):
    retries = (0.1, 0.3, 1.0)
    last_err = None
    for i, delay in enumerate((0.0, *retries)):
        if delay:
            time.sleep(delay)
        try:
            cur.execute(sql, _to_sql_params(params))
            return
        except PgOperationalError as e:
            last_err = e
            if i == len(retries):
                raise
    if last_err:
        raise last_err


def get_conn():
    if _is_sqlite():
        db_path = _sqlite_path_from_url(DATABASE_URL) if DATABASE_URL.startswith("sqlite:///") else str(DB_PATH)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    if psycopg2 is None:
        raise RuntimeError("PostgreSQL URL configured but psycopg2 is not installed")

    conn = psycopg2.connect(
        DATABASE_URL,
        connect_timeout=10,
        application_name="planet-hunter",
    )
    conn.autocommit = False
    return conn


def _fetchone(conn, sql: str, params=None):
    if _is_sqlite():
        return conn.execute(sql, _to_sql_params(params)).fetchone()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        _execute_pg_with_retry(cur, sql, params)
        return cur.fetchone()


def _fetchall(conn, sql: str, params=None):
    if _is_sqlite():
        return conn.execute(sql, _to_sql_params(params)).fetchall()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        _execute_pg_with_retry(cur, sql, params)
        return cur.fetchall()


def _execute(conn, sql: str, params=None):
    if _is_sqlite():
        return conn.execute(sql, _to_sql_params(params))
    with conn.cursor() as cur:
        _execute_pg_with_retry(cur, sql, params)
        return cur


def init_db():
    conn = get_conn()
    if _is_sqlite():
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS stars (
                tic_id      INTEGER PRIMARY KEY,
                ra          REAL,
                dec         REAL,
                tmag        REAL,
                radius      REAL,
                teff        REAL,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS analyses (
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
                ml_planet_score   REAL,
                ml_model_version  TEXT,
                ml_decision_source TEXT,
                created_at        TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (tic_id) REFERENCES stars(tic_id)
            );

            CREATE TABLE IF NOT EXISTS queue (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                tic_id    INTEGER NOT NULL,
                source    TEXT NOT NULL DEFAULT 'MANUAL',
                priority  INTEGER NOT NULL DEFAULT 1,
                status    TEXT NOT NULL DEFAULT 'QUEUED',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_queue_status_priority
                ON queue(status, priority, created_at);
            CREATE INDEX IF NOT EXISTS idx_analyses_tic
                ON analyses(tic_id);
            CREATE INDEX IF NOT EXISTS idx_analyses_class
                ON analyses(classification);

            CREATE TABLE IF NOT EXISTS known_planet_tids (
                tic_id  INTEGER PRIMARY KEY,
                pl_name TEXT,
                source  TEXT DEFAULT 'NASA'
            );
            """
        )
    else:
        _execute(
            conn,
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
                tic_id  BIGINT PRIMARY KEY,
                pl_name TEXT,
                source  TEXT DEFAULT 'NASA'
            );

            CREATE INDEX IF NOT EXISTS idx_queue_status_priority
                ON queue(status, priority, created_at);
            CREATE INDEX IF NOT EXISTS idx_analyses_tic
                ON analyses(tic_id);
            CREATE INDEX IF NOT EXISTS idx_analyses_class
                ON analyses(classification);
            """,
        )

    # Lightweight migrations for existing DBs
    if _is_sqlite():
        for sql in [
            "ALTER TABLE analyses ADD COLUMN ml_planet_score REAL",
            "ALTER TABLE analyses ADD COLUMN ml_model_version TEXT",
            "ALTER TABLE analyses ADD COLUMN ml_decision_source TEXT",
        ]:
            try:
                _execute(conn, sql)
            except sqlite3.OperationalError:
                pass
    else:
        for sql in [
            "ALTER TABLE analyses ADD COLUMN IF NOT EXISTS ml_planet_score DOUBLE PRECISION",
            "ALTER TABLE analyses ADD COLUMN IF NOT EXISTS ml_model_version TEXT",
            "ALTER TABLE analyses ADD COLUMN IF NOT EXISTS ml_decision_source TEXT",
        ]:
            _execute(conn, sql)

    _execute(conn, "CREATE INDEX IF NOT EXISTS idx_analyses_ml_score ON analyses(ml_planet_score)")
    _execute(conn, "CREATE INDEX IF NOT EXISTS idx_analyses_model_version ON analyses(ml_model_version)")

    conn.commit()
    conn.close()


# ---------- Stars ----------

def upsert_star(star: StarInfo):
    conn = get_conn()
    sql = """
        INSERT INTO stars (tic_id, ra, dec, tmag, radius, teff)
        VALUES ({p},{p},{p},{p},{p},{p})
        ON CONFLICT(tic_id) DO UPDATE SET
            ra=excluded.ra, dec=excluded.dec, tmag=excluded.tmag,
            radius=excluded.radius, teff=excluded.teff
    """
    p = "?" if _is_sqlite() else "%s"
    _execute(
        conn,
        sql.format(p=p),
        (star.tic_id, star.ra, star.dec, star.tmag, star.radius, star.teff),
    )
    conn.commit()
    conn.close()


def get_star(tic_id: int) -> Optional[StarInfo]:
    conn = get_conn()
    p = "?" if _is_sqlite() else "%s"
    row = _fetchone(conn, f"SELECT * FROM stars WHERE tic_id={p}", (tic_id,))
    conn.close()
    if not row:
        return None
    return StarInfo(
        tic_id=row["tic_id"],
        ra=row["ra"],
        dec=row["dec"],
        tmag=row["tmag"],
        radius=row["radius"],
        teff=row["teff"],
    )


# ---------- Analyses ----------

def insert_analysis(result: AnalysisResult) -> int:
    conn = get_conn()
    p = "?" if _is_sqlite() else "%s"
    returning = "" if _is_sqlite() else " RETURNING id"
    sql = f"""
        INSERT INTO analyses (
            tic_id, classification, period, depth, snr, duration,
            planet_radius, equilibrium_temp, sectors_checked, sectors_detected,
            sinusoid_better, secondary_depth, odd_even_sigma,
            plot_lightcurve, plot_periodogram, plot_phase_fold, plot_diagnostic,
            review_notes, error_message,
            ml_planet_score, ml_model_version, ml_decision_source
        ) VALUES ({','.join([p]*22)}){returning}
    """
    params = (
        result.tic_id,
        result.classification.value,
        _py(result.period),
        _py(result.depth),
        _py(result.snr),
        _py(result.duration),
        _py(result.planet_radius),
        _py(result.equilibrium_temp),
        _py(result.sectors_checked),
        _py(result.sectors_detected),
        int(result.sinusoid_better),
        _py(result.secondary_depth),
        _py(result.odd_even_sigma),
        result.plot_lightcurve,
        result.plot_periodogram,
        result.plot_phase_fold,
        result.plot_diagnostic,
        result.review_notes,
        result.error_message,
        _py(result.ml_planet_score),
        result.ml_model_version,
        result.ml_decision_source,
    )
    if _is_sqlite():
        cur = _execute(conn, sql, params)
        row_id = cur.lastrowid
    else:
        row = _fetchone(conn, sql, params)
        row_id = int(row["id"])
    conn.commit()
    conn.close()
    return row_id


def update_analysis(analysis_id: int, **kwargs):
    if not kwargs:
        return
    conn = get_conn()
    p = "?" if _is_sqlite() else "%s"
    sets = []
    vals = []
    for k, v in kwargs.items():
        sets.append(f"{k}={p}")
        if isinstance(v, Classification):
            v = v.value
        elif isinstance(v, bool):
            v = int(v)
        vals.append(v)
    vals.append(analysis_id)
    _execute(conn, f"UPDATE analyses SET {', '.join(sets)} WHERE id={p}", vals)
    conn.commit()
    conn.close()


def get_analysis(analysis_id: int) -> Optional[dict]:
    conn = get_conn()
    p = "?" if _is_sqlite() else "%s"
    row = _fetchone(conn, f"SELECT * FROM analyses WHERE id={p}", (analysis_id,))
    conn.close()
    return dict(row) if row else None


def get_latest_analysis(tic_id: int) -> Optional[dict]:
    conn = get_conn()
    p = "?" if _is_sqlite() else "%s"
    row = _fetchone(
        conn,
        f"SELECT * FROM analyses WHERE tic_id={p} ORDER BY id DESC LIMIT 1",
        (tic_id,),
    )
    conn.close()
    return dict(row) if row else None


def list_analyses(
    classification: Optional[str] = None,
    limit: int = 50,
    min_ml_score: Optional[float] = None,
    model_version: Optional[str] = None,
) -> list[dict]:
    conn = get_conn()
    p = "?" if _is_sqlite() else "%s"
    where = []
    vals: list = []

    if classification:
        where.append(f"classification={p}")
        vals.append(classification)
    if min_ml_score is not None:
        where.append("ml_planet_score IS NOT NULL")
        where.append(f"ml_planet_score>={p}")
        vals.append(float(min_ml_score))
    if model_version:
        where.append(f"ml_model_version={p}")
        vals.append(model_version)

    sql = "SELECT * FROM analyses"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY id DESC LIMIT {p}"
    vals.append(limit)

    rows = _fetchall(conn, sql, tuple(vals))
    conn.close()
    return [dict(r) for r in rows]


def count_by_classification() -> dict[str, int]:
    conn = get_conn()
    rows = _fetchall(
        conn,
        "SELECT classification, COUNT(*) as cnt FROM analyses GROUP BY classification",
    )
    conn.close()
    return {r["classification"]: int(r["cnt"] or 0) for r in rows}


def ml_monitor_snapshot(hours: int = 24) -> dict:
    conn = get_conn()
    if _is_sqlite():
        row = _fetchone(
            conn,
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN ml_planet_score IS NOT NULL THEN 1 ELSE 0 END) AS ml_scored,
                SUM(CASE WHEN classification='PLANET_CANDIDATE' THEN 1 ELSE 0 END) AS planet_candidate,
                SUM(CASE WHEN classification='MANUAL_REVIEW' THEN 1 ELSE 0 END) AS manual_review,
                AVG(ml_planet_score) AS avg_score,
                SUM(CASE WHEN ml_planet_score >= 0.80 THEN 1 ELSE 0 END) AS high_conf_planet
            FROM analyses
            WHERE datetime(created_at) >= datetime('now', ?)
            """,
            (f"-{int(hours)} hours",),
        )

        versions = _fetchall(
            conn,
            """
            SELECT COALESCE(ml_model_version, 'N/A') AS model_version, COUNT(*) AS cnt
            FROM analyses
            WHERE datetime(created_at) >= datetime('now', ?)
            GROUP BY COALESCE(ml_model_version, 'N/A')
            ORDER BY cnt DESC
            """,
            (f"-{int(hours)} hours",),
        )
    else:
        row = _fetchone(
            conn,
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN ml_planet_score IS NOT NULL THEN 1 ELSE 0 END) AS ml_scored,
                SUM(CASE WHEN classification='PLANET_CANDIDATE' THEN 1 ELSE 0 END) AS planet_candidate,
                SUM(CASE WHEN classification='MANUAL_REVIEW' THEN 1 ELSE 0 END) AS manual_review,
                AVG(ml_planet_score) AS avg_score,
                SUM(CASE WHEN ml_planet_score >= 0.80 THEN 1 ELSE 0 END) AS high_conf_planet
            FROM analyses
            WHERE created_at >= (CURRENT_TIMESTAMP - (%s * INTERVAL '1 hour'))
            """,
            (int(hours),),
        )

        versions = _fetchall(
            conn,
            """
            SELECT COALESCE(ml_model_version, 'N/A') AS model_version, COUNT(*) AS cnt
            FROM analyses
            WHERE created_at >= (CURRENT_TIMESTAMP - (%s * INTERVAL '1 hour'))
            GROUP BY COALESCE(ml_model_version, 'N/A')
            ORDER BY cnt DESC
            """,
            (int(hours),),
        )
    conn.close()

    return {
        "hours": hours,
        "total": int(row["total"] or 0),
        "ml_scored": int(row["ml_scored"] or 0),
        "planet_candidate": int(row["planet_candidate"] or 0),
        "manual_review": int(row["manual_review"] or 0),
        "high_conf_planet": int(row["high_conf_planet"] or 0),
        "avg_score": float(row["avg_score"]) if row["avg_score"] is not None else None,
        "versions": [
            {"model_version": r["model_version"], "count": int(r["cnt"] or 0)} for r in versions
        ],
    }


# ---------- Queue ----------

def enqueue(tic_id: int, source: QueueSource, priority: int) -> int:
    conn = get_conn()
    p = "?" if _is_sqlite() else "%s"
    existing = _fetchone(
        conn,
        f"SELECT id FROM queue WHERE tic_id={p} AND status IN ('QUEUED','RUNNING')",
        (tic_id,),
    )
    if existing:
        conn.close()
        return int(existing["id"])

    if _is_sqlite():
        cur = _execute(
            conn,
            f"INSERT INTO queue (tic_id, source, priority, status) VALUES ({p},{p},{p},{p})",
            (tic_id, source.value, priority, QueueStatus.QUEUED.value),
        )
        row_id = cur.lastrowid
    else:
        row = _fetchone(
            conn,
            f"INSERT INTO queue (tic_id, source, priority, status) VALUES ({p},{p},{p},{p}) RETURNING id",
            (tic_id, source.value, priority, QueueStatus.QUEUED.value),
        )
        row_id = int(row["id"])

    conn.commit()
    conn.close()
    return row_id


def next_in_queue() -> Optional[dict]:
    conn = get_conn()
    row = _fetchone(
        conn,
        "SELECT * FROM queue WHERE status='QUEUED' ORDER BY priority ASC, created_at ASC LIMIT 1",
    )
    if row:
        p = "?" if _is_sqlite() else "%s"
        _execute(conn, f"UPDATE queue SET status='RUNNING' WHERE id={p}", (row["id"],))
        conn.commit()
    conn.close()
    return dict(row) if row else None


def finish_queue_item(queue_id: int, failed: bool = False):
    conn = get_conn()
    status = QueueStatus.FAILED.value if failed else QueueStatus.DONE.value
    p = "?" if _is_sqlite() else "%s"
    _execute(conn, f"UPDATE queue SET status={p} WHERE id={p}", (status, queue_id))
    conn.commit()
    conn.close()


def list_queue(limit: int = 50) -> list[dict]:
    conn = get_conn()
    p = "?" if _is_sqlite() else "%s"
    rows = _fetchall(
        conn,
        f"SELECT * FROM queue ORDER BY status='RUNNING' DESC, priority ASC, created_at ASC LIMIT {p}",
        (limit,),
    )
    conn.close()
    return [dict(r) for r in rows]


# ---------- Known Planets ----------

def is_known_planet(tic_id: int) -> bool:
    conn = get_conn()
    p = "?" if _is_sqlite() else "%s"
    r = _fetchone(conn, f"SELECT 1 FROM known_planet_tids WHERE tic_id={p}", (tic_id,))
    conn.close()
    return r is not None


def add_known_planet_tids(records: list) -> int:
    """records = [(tic_id, pl_name), ...]. Returns number of newly inserted rows."""
    if not records:
        return 0
    conn = get_conn()
    p = "?" if _is_sqlite() else "%s"
    inserted = 0
    if _is_sqlite():
        for rec in records:
            cur = _execute(
                conn,
                f"INSERT OR IGNORE INTO known_planet_tids (tic_id, pl_name) VALUES ({p},{p})",
                rec,
            )
            if cur.rowcount and cur.rowcount > 0:
                inserted += 1
    else:
        for rec in records:
            with conn.cursor() as cur:
                _execute_pg_with_retry(
                    cur,
                    "INSERT INTO known_planet_tids (tic_id, pl_name) VALUES (%s,%s) ON CONFLICT (tic_id) DO NOTHING",
                    rec,
                )
                if cur.rowcount and cur.rowcount > 0:
                    inserted += 1

    conn.commit()
    conn.close()
    return inserted


def queue_stats() -> dict:
    conn = get_conn()
    row = _fetchone(
        conn,
        """
        SELECT
            SUM(CASE WHEN status='QUEUED' THEN 1 ELSE 0 END) as queued,
            SUM(CASE WHEN status='RUNNING' THEN 1 ELSE 0 END) as running,
            SUM(CASE WHEN status='DONE' THEN 1 ELSE 0 END) as done,
            SUM(CASE WHEN status='FAILED' THEN 1 ELSE 0 END) as failed
        FROM queue
        """,
    )
    conn.close()
    return {
        "queued": int((row["queued"] or 0) if row else 0),
        "running": int((row["running"] or 0) if row else 0),
        "done": int((row["done"] or 0) if row else 0),
        "failed": int((row["failed"] or 0) if row else 0),
    }


def count_active_by_source(
    source: QueueSource,
    running_max_age_minutes: int | None = None,
) -> int:
    """Count active queue items for a source.

    By default counts QUEUED + RUNNING. If running_max_age_minutes is provided,
    stale RUNNING rows older than that threshold are excluded.
    """
    conn = get_conn()
    p = "?" if _is_sqlite() else "%s"
    if running_max_age_minutes is None:
        row = _fetchone(
            conn,
            f"""
            SELECT COUNT(*) AS cnt
            FROM queue
            WHERE source={p} AND status IN ('QUEUED', 'RUNNING')
            """,
            (source.value,),
        )
    else:
        if _is_sqlite():
            row = _fetchone(
                conn,
                f"""
                SELECT COUNT(*) AS cnt
                FROM queue
                WHERE source={p}
                  AND (
                    status='QUEUED'
                    OR (
                      status='RUNNING'
                      AND datetime(created_at) >= datetime('now', {p})
                    )
                  )
                """,
                (source.value, f"-{int(running_max_age_minutes)} minutes"),
            )
        else:
            row = _fetchone(
                conn,
                f"""
                SELECT COUNT(*) AS cnt
                FROM queue
                WHERE source={p}
                  AND (
                    status='QUEUED'
                    OR (
                      status='RUNNING'
                      AND created_at >= (CURRENT_TIMESTAMP - ({p} * INTERVAL '1 minute'))
                    )
                  )
                """,
                (source.value, int(running_max_age_minutes)),
            )
    conn.close()
    return int((row["cnt"] if row else 0) or 0)


def requeue_stuck_running(
    minutes: int = 360,
    source: QueueSource | None = None,
) -> int:
    """Reset stale RUNNING queue items back to QUEUED.

    Uses created_at as a proxy for running age.
    """
    conn = get_conn()
    changed = 0
    p = "?" if _is_sqlite() else "%s"

    if _is_sqlite():
        params = [f"-{int(minutes)} minutes"]
        sql = f"""
            UPDATE queue
            SET status='QUEUED'
            WHERE status='RUNNING'
              AND datetime(created_at) < datetime('now', {p})
        """
        if source is not None:
            sql += f" AND source={p}"
            params.append(source.value)
        cur = _execute(conn, sql, tuple(params))
        changed = cur.rowcount if cur.rowcount is not None else 0
    else:
        params = [int(minutes)]
        sql = f"""
            UPDATE queue
            SET status='QUEUED'
            WHERE status='RUNNING'
              AND created_at < (CURRENT_TIMESTAMP - ({p} * INTERVAL '1 minute'))
        """
        if source is not None:
            sql += f" AND source={p}"
            params.append(source.value)
        with conn.cursor() as cur:
            _execute_pg_with_retry(cur, sql, tuple(params))
            changed = cur.rowcount if cur.rowcount is not None else 0

    conn.commit()
    conn.close()
    return changed
