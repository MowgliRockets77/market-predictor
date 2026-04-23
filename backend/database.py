import os
import sqlite3
from pathlib import Path

_DB_PATH = Path(os.getenv("SQLITE_PATH", "./market.db"))

def _get_conn():
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

async def init_db():
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_time        TEXT NOT NULL,
                ticker           TEXT NOT NULL,
                prediction_date  TEXT NOT NULL,
                target_date      TEXT NOT NULL,
                prob_up          REAL NOT NULL,
                prediction       TEXT NOT NULL,
                recommendation   TEXT NOT NULL,
                price            REAL,
                ret5             REAL,
                ret20            REAL,
                holdout_acc      REAL,
                actual           TEXT,
                correct          INTEGER,
                created_at       TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_pred_ticker ON predictions(ticker);
            CREATE INDEX IF NOT EXISTS idx_pred_target ON predictions(target_date);
            CREATE INDEX IF NOT EXISTS idx_pred_scan   ON predictions(scan_time);
        """)
        conn.commit()

def save_prediction(scan_time, ticker, pred_date, target_date,
                    prob_up, prediction, recommendation,
                    price=None, ret5=None, ret20=None, holdout_acc=None):
    with _get_conn() as conn:
        exists = conn.execute(
            "SELECT id FROM predictions WHERE ticker=? AND target_date=?",
            (ticker, target_date)
        ).fetchone()
        if not exists:
            conn.execute("""
                INSERT INTO predictions
                (scan_time,ticker,prediction_date,target_date,
                 prob_up,prediction,recommendation,price,ret5,ret20,holdout_acc,actual,correct)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL,NULL)
            """, (scan_time, ticker, pred_date, target_date,
                  prob_up, prediction, recommendation,
                  price, ret5, ret20, holdout_acc))
            conn.commit()

def update_actuals_for_ticker(ticker, close_series):
    idx_map = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(close_series.index)}
    with _get_conn() as conn:
        pending = conn.execute(
            "SELECT id,target_date,prediction FROM predictions WHERE ticker=? AND actual IS NULL",
            (ticker,)
        ).fetchall()
        for row in pending:
            td = row["target_date"]
            if td in idx_map:
                pos = idx_map[td]
                if pos > 0:
                    actual  = "UP" if close_series.iloc[pos] > close_series.iloc[pos-1] else "DOWN"
                    correct = 1 if actual == row["prediction"] else 0
                    conn.execute(
                        "UPDATE predictions SET actual=?,correct=? WHERE id=?",
                        (actual, correct, row["id"])
                    )
        conn.commit()

def fetch_all_predictions(limit=500):
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]

def fetch_latest_scan():
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT scan_time FROM predictions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return []
        scan_time = row["scan_time"]
        rows = conn.execute(
            "SELECT * FROM predictions WHERE scan_time=? ORDER BY prob_up DESC",
            (scan_time,)
        ).fetchall()
    return [dict(r) for r in rows]

def fetch_ticker_predictions(ticker, limit=100):
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE ticker=? ORDER BY id DESC LIMIT ?",
            (ticker, limit)
        ).fetchall()
    return [dict(r) for r in rows]

def fetch_accuracy_stats():
    with _get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        resolved = conn.execute(
            "SELECT COUNT(*),AVG(correct) FROM predictions WHERE actual IS NOT NULL"
        ).fetchone()
        by_ticker = conn.execute("""
            SELECT ticker,
                   COUNT(*) as predictions,
                   ROUND(AVG(correct)*100,1) as accuracy,
                   MAX(prediction_date) as last_date
            FROM predictions
            WHERE actual IS NOT NULL
            GROUP BY ticker
            ORDER BY accuracy DESC
        """).fetchall()
        last_scan = conn.execute(
            "SELECT scan_time FROM predictions ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return {
        "total_predictions": total,
        "resolved": resolved[0] or 0,
        "overall_accuracy": round((resolved[1] or 0) * 100, 1),
        "last_scan": last_scan[0] if last_scan else None,
        "by_ticker": [dict(r) for r in by_ticker],
    }

def fetch_top_picks(limit=10):
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT scan_time FROM predictions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return []
        scan_time = row["scan_time"]
        rows = conn.execute("""
            SELECT * FROM predictions
            WHERE scan_time=? AND recommendation='BUY'
            ORDER BY prob_up DESC
            LIMIT ?
        """, (scan_time, limit)).fetchall()
    return [dict(r) for r in rows]
