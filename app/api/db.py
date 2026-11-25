import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from .config import settings


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    path = db_path or settings.db_path
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                request_time TEXT NOT NULL,
                bar_time TEXT NOT NULL,
                bar_close_ms INTEGER NOT NULL,
                expires_at TEXT NOT NULL,
                expires_at_ms INTEGER NOT NULL,
                p_up REAL NOT NULL,
                p_flat REAL NOT NULL,
                p_down REAL NOT NULL,
                action TEXT NOT NULL,
                entry_price REAL NOT NULL,
                model_version TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                exit_price REAL,
                realized_return REAL,
                net_return REAL,
                resolved_at TEXT,
                features_json TEXT,
                UNIQUE(pair, bar_time)
            );
            """
        )
        # Migrate databases created before features_json existed (CREATE IF NOT EXISTS won't add it).
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(predictions)")}
        if "features_json" not in columns:
            conn.execute("ALTER TABLE predictions ADD COLUMN features_json TEXT")
        # Online learning: whether this resolved prediction was sent back to the model service as a labeled sample.
        if "feedback_sent" not in columns:
            conn.execute("ALTER TABLE predictions ADD COLUMN feedback_sent INTEGER NOT NULL DEFAULT 0")
        # WAIT is not a trade, so it has no right/wrong outcome (older rows were marked 'incorrect').
        conn.execute(
            "UPDATE predictions SET status = 'neutral' WHERE action = 'WAIT' AND status != 'pending'"
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_predictions_status_expires
            ON predictions (status, expires_at_ms);
            """
        )


def get_prediction_by_bar(
    pair: str, bar_time: str, db_path: str | None = None
) -> dict | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE pair = ? AND bar_time = ?",
            (pair.upper(), bar_time),
        ).fetchone()
        return dict(row) if row else None


def insert_prediction(record: dict, db_path: str | None = None) -> dict:
    record = {"features_json": None, **record}
    with get_conn(db_path) as conn:
        try:
            cursor = conn.execute(
                """
                INSERT INTO predictions (
                    pair, request_time, bar_time, bar_close_ms,
                    expires_at, expires_at_ms, p_up, p_flat, p_down,
                    action, entry_price, model_version, schema_version, features_json, status
                ) VALUES (
                    :pair, :request_time, :bar_time, :bar_close_ms,
                    :expires_at, :expires_at_ms, :p_up, :p_flat, :p_down,
                    :action, :entry_price, :model_version, :schema_version, :features_json, 'pending'
                )
                """,
                record,
            )
            row_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            # Handles race condition if two identical bar requests arrive simultaneously
            existing = conn.execute(
                "SELECT * FROM predictions WHERE pair = ? AND bar_time = ?",
                (record["pair"], record["bar_time"]),
            ).fetchone()
            return dict(existing)

        row = conn.execute(
            "SELECT * FROM predictions WHERE id = ?", (row_id,)
        ).fetchone()
        return dict(row)


def list_predictions(
    pair: str | None = None, limit: int = 50, db_path: str | None = None
) -> list[dict]:
    with get_conn(db_path) as conn:
        if pair:
            rows = conn.execute(
                """
                SELECT * FROM predictions
                WHERE pair = ?
                ORDER BY request_time DESC
                LIMIT ?
                """,
                (pair.upper(), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM predictions
                ORDER BY request_time DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def get_due_pending_predictions(
    now_ms: int, db_path: str | None = None
) -> list[dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM predictions
            WHERE status = 'pending' AND expires_at_ms <= ?
            ORDER BY expires_at_ms ASC
            """,
            (now_ms,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_prediction_outcome(
    prediction_id: int,
    exit_price: float,
    realized_return: float,
    net_return: float,
    status: str,
    db_path: str | None = None,
) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """
            UPDATE predictions
            SET exit_price = ?,
                realized_return = ?,
                net_return = ?,
                status = ?,
                resolved_at = ?
            WHERE id = ?
            """,
            (exit_price, realized_return, net_return, status, _utc_now_iso(), prediction_id),
        )


def get_unsent_feedback(limit: int = 500, db_path: str | None = None) -> list[dict]:
    """Resolved predictions with stored features that the model service has not received yet."""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, pair, bar_time, features_json, realized_return FROM predictions
            WHERE status != 'pending' AND features_json IS NOT NULL AND realized_return IS NOT NULL
              AND feedback_sent = 0
            ORDER BY expires_at_ms ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def mark_feedback_sent(prediction_ids: list[int], db_path: str | None = None) -> None:
    with get_conn(db_path) as conn:
        conn.executemany("UPDATE predictions SET feedback_sent = 1 WHERE id = ?", [(i,) for i in prediction_ids])


def get_stats(db_path: str | None = None) -> dict:
    with get_conn(db_path) as conn:
        totals = conn.execute(
            """
            SELECT
                COUNT(*) AS total_predictions,
                SUM(CASE WHEN status != 'pending' THEN 1 ELSE 0 END) AS resolved_predictions,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending_predictions
            FROM predictions
            """
        ).fetchone()

        rows = conn.execute(
            """
            SELECT
                action,
                COUNT(*) AS total,
                SUM(CASE WHEN status != 'pending' THEN 1 ELSE 0 END) AS resolved,
                SUM(CASE WHEN status = 'correct' THEN 1 ELSE 0 END) AS profitable_count,
                AVG(CASE WHEN status != 'pending' THEN net_return END) AS avg_net_return,
                AVG(CASE WHEN status != 'pending' THEN realized_return END) AS avg_realized_return
            FROM predictions
            GROUP BY action
            """
        ).fetchall()

        signals = conn.execute(
            """
            SELECT
                COUNT(*) AS resolved_signals,
                AVG(CASE WHEN net_return > 0 THEN 1.0 ELSE 0.0 END) AS signal_win_rate,
                AVG(net_return) AS signal_avg_net_return
            FROM predictions
            WHERE status != 'pending' AND action IN ('BUY', 'SELL')
            """
        ).fetchone()

    by_action_map = {r["action"]: dict(r) for r in rows}
    by_action = []
    for act in ("BUY", "SELL", "WAIT"):
        data = by_action_map.get(act)
        if not data:
            by_action.append(
                {
                    "action": act,
                    "total": 0,
                    "resolved": 0,
                    "profitable_count": 0,
                    "win_rate": None,
                    "avg_net_return": None,
                    "avg_realized_return": None,
                }
            )
        else:
            resolved = data["resolved"] or 0
            profitable = data["profitable_count"] or 0
            by_action.append(
                {
                    "action": act,
                    "total": data["total"],
                    "resolved": resolved,
                    "profitable_count": profitable,
                    # WAIT is not a trade, so a win rate is meaningless for it
                    "win_rate": (profitable / resolved) if (resolved > 0 and act != "WAIT") else None,
                    "avg_net_return": data["avg_net_return"],
                    "avg_realized_return": data["avg_realized_return"],
                }
            )

    return {
        "total_predictions": totals["total_predictions"] or 0,
        "resolved_predictions": totals["resolved_predictions"] or 0,
        "pending_predictions": totals["pending_predictions"] or 0,
        "resolved_signals": signals["resolved_signals"] or 0,
        "signal_win_rate": signals["signal_win_rate"],
        "signal_avg_net_return": signals["signal_avg_net_return"],
        "by_action": by_action,
    }