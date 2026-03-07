import sqlite3
import json
import numpy as np
from typing import Optional
from planet_hunter.config import DB_PATH
from planet_hunter.models import (
    StarInfo, AnalysisResult, QueueItem,
    Classification, QueueSource, QueueStatus,
)


def _py(val):
    """Convert numpy types to Python native for sqlite3."""
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return float(val.item()) if val.size == 1 else None
    return val


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
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
    """)

    # Lightweight migrations for existing DBs
    for sql in [
        "ALTER TABLE analyses ADD COLUMN ml_planet_score REAL",
        "ALTER TABLE analyses ADD COLUMN ml_model_version TEXT",
        "ALTER TABLE analyses ADD COLUMN ml_decision_source TEXT",
    ]:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass

    conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_ml_score ON analyses(ml_planet_score)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_model_version ON analyses(ml_model_version)")

    conn.commit()
    conn.close()


# ---------- Stars ----------

def upsert_star(star: StarInfo):
    conn = get_conn()
    conn.execute("""
        INSERT INTO stars (tic_id, ra, dec, tmag, radius, teff)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(tic_id) DO UPDATE SET
            ra=excluded.ra, dec=excluded.dec, tmag=excluded.tmag,
            radius=excluded.radius, teff=excluded.teff
    """, (star.tic_id, star.ra, star.dec, star.tmag, star.radius, star.teff))
    conn.commit()
    conn.close()


def get_star(tic_id: int) -> Optional[StarInfo]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM stars WHERE tic_id=?", (tic_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return StarInfo(
        tic_id=row["tic_id"], ra=row["ra"], dec=row["dec"],
        tmag=row["tmag"], radius=row["radius"], teff=row["teff"],
    )


# ---------- Analyses ----------

def insert_analysis(result: AnalysisResult) -> int:
    conn = get_conn()
    cur = conn.execute("""
        INSERT INTO analyses (
            tic_id, classification, period, depth, snr, duration,
            planet_radius, equilibrium_temp, sectors_checked, sectors_detected,
            sinusoid_better, secondary_depth, odd_even_sigma,
            plot_lightcurve, plot_periodogram, plot_phase_fold, plot_diagnostic,
            review_notes, error_message,
            ml_planet_score, ml_model_version, ml_decision_source
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        result.tic_id, result.classification.value,
        _py(result.period), _py(result.depth), _py(result.snr), _py(result.duration),
        _py(result.planet_radius), _py(result.equilibrium_temp),
        _py(result.sectors_checked), _py(result.sectors_detected),
        int(result.sinusoid_better),
        _py(result.secondary_depth), _py(result.odd_even_sigma),
        result.plot_lightcurve, result.plot_periodogram,
        result.plot_phase_fold, result.plot_diagnostic,
        result.review_notes, result.error_message,
        _py(result.ml_planet_score), result.ml_model_version, result.ml_decision_source,
    ))
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def update_analysis(analysis_id: int, **kwargs):
    conn = get_conn()
    sets = []
    vals = []
    for k, v in kwargs.items():
        sets.append(f"{k}=?")
        if isinstance(v, Classification):
            v = v.value
        elif isinstance(v, bool):
            v = int(v)
        vals.append(v)
    vals.append(analysis_id)
    conn.execute(f"UPDATE analyses SET {', '.join(sets)} WHERE id=?", vals)
    conn.commit()
    conn.close()


def get_analysis(analysis_id: int) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_latest_analysis(tic_id: int) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM analyses WHERE tic_id=? ORDER BY id DESC LIMIT 1",
        (tic_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def list_analyses(
    classification: Optional[str] = None,
    limit: int = 50,
    min_ml_score: Optional[float] = None,
    model_version: Optional[str] = None,
) -> list[dict]:
    conn = get_conn()
    where = []
    vals: list = []

    if classification:
        where.append("classification=?")
        vals.append(classification)
    if min_ml_score is not None:
        where.append("ml_planet_score IS NOT NULL")
        where.append("ml_planet_score>=?")
        vals.append(float(min_ml_score))
    if model_version:
        where.append("ml_model_version=?")
        vals.append(model_version)

    sql = "SELECT * FROM analyses"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    vals.append(limit)

    rows = conn.execute(sql, tuple(vals)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_by_classification() -> dict[str, int]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT classification, COUNT(*) as cnt FROM analyses GROUP BY classification"
    ).fetchall()
    conn.close()
    return {r["classification"]: r["cnt"] for r in rows}


def ml_monitor_snapshot(hours: int = 24) -> dict:
    conn = get_conn()
    row = conn.execute(
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
    ).fetchone()

    versions = conn.execute(
        """
        SELECT COALESCE(ml_model_version, 'N/A') AS model_version, COUNT(*) AS cnt
        FROM analyses
        WHERE datetime(created_at) >= datetime('now', ?)
        GROUP BY COALESCE(ml_model_version, 'N/A')
        ORDER BY cnt DESC
        """,
        (f"-{int(hours)} hours",),
    ).fetchall()
    conn.close()

    return {
        "hours": hours,
        "total": int(row["total"] or 0),
        "ml_scored": int(row["ml_scored"] or 0),
        "planet_candidate": int(row["planet_candidate"] or 0),
        "manual_review": int(row["manual_review"] or 0),
        "high_conf_planet": int(row["high_conf_planet"] or 0),
        "avg_score": float(row["avg_score"]) if row["avg_score"] is not None else None,
        "versions": [{"model_version": r["model_version"], "count": int(r["cnt"] or 0)} for r in versions],
    }


# ---------- Queue ----------

def enqueue(tic_id: int, source: QueueSource, priority: int) -> int:
    conn = get_conn()
    # skip if already queued
    existing = conn.execute(
        "SELECT id FROM queue WHERE tic_id=? AND status IN ('QUEUED','RUNNING')",
        (tic_id,),
    ).fetchone()
    if existing:
        conn.close()
        return existing["id"]
    cur = conn.execute(
        "INSERT INTO queue (tic_id, source, priority, status) VALUES (?,?,?,?)",
        (tic_id, source.value, priority, QueueStatus.QUEUED.value),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def next_in_queue() -> Optional[dict]:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM queue WHERE status='QUEUED' ORDER BY priority ASC, created_at ASC LIMIT 1"
    ).fetchone()
    if row:
        conn.execute("UPDATE queue SET status='RUNNING' WHERE id=?", (row["id"],))
        conn.commit()
    conn.close()
    return dict(row) if row else None


def finish_queue_item(queue_id: int, failed: bool = False):
    conn = get_conn()
    status = QueueStatus.FAILED.value if failed else QueueStatus.DONE.value
    conn.execute("UPDATE queue SET status=? WHERE id=?", (status, queue_id))
    conn.commit()
    conn.close()


def list_queue(limit: int = 50) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM queue ORDER BY status='RUNNING' DESC, priority ASC, created_at ASC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- Known Planets ----------

def is_known_planet(tic_id: int) -> bool:
    conn = get_conn()
    r = conn.execute("SELECT 1 FROM known_planet_tids WHERE tic_id=?", (tic_id,)).fetchone()
    conn.close()
    return r is not None


def add_known_planet_tids(records: list) -> int:
    """records = [(tic_id, pl_name), ...]. Returns number of newly inserted rows."""
    conn = get_conn()
    conn.executemany(
        "INSERT OR IGNORE INTO known_planet_tids (tic_id, pl_name) VALUES (?,?)",
        records,
    )
    count = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    conn.close()
    return count


def queue_stats() -> dict:
    conn = get_conn()
    row = conn.execute("""
        SELECT
            SUM(CASE WHEN status='QUEUED' THEN 1 ELSE 0 END) as queued,
            SUM(CASE WHEN status='RUNNING' THEN 1 ELSE 0 END) as running,
            SUM(CASE WHEN status='DONE' THEN 1 ELSE 0 END) as done,
            SUM(CASE WHEN status='FAILED' THEN 1 ELSE 0 END) as failed
        FROM queue
    """).fetchone()
    conn.close()
    return dict(row)


def count_active_by_source(
    source: QueueSource,
    running_max_age_minutes: int | None = None,
) -> int:
    """Count active queue items for a source.

    By default counts QUEUED + RUNNING. If running_max_age_minutes is provided,
    stale RUNNING rows older than that threshold are excluded.
    """
    conn = get_conn()
    if running_max_age_minutes is None:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM queue
            WHERE source=? AND status IN ('QUEUED', 'RUNNING')
            """,
            (source.value,),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM queue
            WHERE source=?
              AND (
                status='QUEUED'
                OR (
                  status='RUNNING'
                  AND datetime(created_at) >= datetime('now', ?)
                )
              )
            """,
            (source.value, f"-{int(running_max_age_minutes)} minutes"),
        ).fetchone()
    conn.close()
    return int(row[0] or 0)


def requeue_stuck_running(
    minutes: int = 360,
    source: QueueSource | None = None,
) -> int:
    """Reset stale RUNNING queue items back to QUEUED.

    Uses created_at as a proxy for running age.
    """
    conn = get_conn()
    params: tuple = (f"-{int(minutes)} minutes",)
    sql = """
        UPDATE queue
        SET status='QUEUED'
        WHERE status='RUNNING'
          AND datetime(created_at) < datetime('now', ?)
    """
    if source is not None:
        sql += " AND source=?"
        params = (f"-{int(minutes)} minutes", source.value)

    cur = conn.execute(sql, params)
    changed = cur.rowcount if cur.rowcount is not None else 0
    conn.commit()
    conn.close()
    return changed
