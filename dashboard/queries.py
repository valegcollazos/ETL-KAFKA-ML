import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "predictions.db"

COLS = "id, country, year, actual, predicted, abs_error, alert, alert_reason"
EMPTY = pd.DataFrame(columns=COLS.split(", "))


def _ready():
    """True si la BD existe y ya tiene la tabla (evita crear un archivo vacío)."""
    if not DB_PATH.exists():
        return False
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='predictions'"
        ).fetchone() is not None


def load_summary():
    empty = {"total": 0, "alerts": 0, "mae": None,
             "error_alto": 0, "bienestar_bajo": 0}
    if not _ready():
        return empty
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        total, alerts, mae, e_alto, b_bajo = conn.execute("""
            SELECT COUNT(*),
                   COALESCE(SUM(alert), 0),
                   AVG(abs_error),
                   COALESCE(SUM(alert_reason LIKE '%error_alto%'), 0),
                   COALESCE(SUM(alert_reason LIKE '%bienestar_bajo%'), 0)
            FROM predictions""").fetchone()
    return {"total": total, "alerts": alerts, "mae": mae,
            "error_alto": e_alto, "bienestar_bajo": b_bajo}


def load_recent(n=40):
    """Últimos n eventos, en orden de llegada."""
    if not _ready():
        return EMPTY.copy()
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        df = pd.read_sql_query(
            f"SELECT {COLS} FROM predictions ORDER BY id DESC LIMIT ?",
            conn, params=(n,))
    return df.iloc[::-1].reset_index(drop=True)


def load_alerts(n=8):
    """Últimas n alertas, la más reciente primero."""
    if not _ready():
        return EMPTY.copy()
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        return pd.read_sql_query(
            f"SELECT {COLS} FROM predictions WHERE alert = 1 "
            "ORDER BY id DESC LIMIT ?", conn, params=(n,))