import sqlite3
from pathlib import Path

DB_PATH = Path("data/predictions.db")


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")  # permite leer (Streamlit) mientras se escribe
    return conn


def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            received_at TEXT NOT NULL,
            country TEXT,
            year INTEGER,
            gdp REAL,
            social_support REAL,
            life_expectancy REAL,
            freedom REAL,
            corruption REAL,
            generosity REAL,
            actual REAL,
            predicted REAL,
            abs_error REAL,
            alert INTEGER NOT NULL DEFAULT 0,
            alert_reason TEXT,
            kafka_partition INTEGER,
            kafka_offset INTEGER,
            UNIQUE (kafka_partition, kafka_offset)
        )
    """)
    conn.commit()


def insert_prediction(conn, row):
    """INSERT OR IGNORE: si el mismo mensaje se reprocesa, no se duplica."""
    cols = ", ".join(row)
    marks = ", ".join(f":{k}" for k in row)
    conn.execute(f"INSERT OR IGNORE INTO predictions ({cols}) VALUES ({marks})", row)
    conn.commit()