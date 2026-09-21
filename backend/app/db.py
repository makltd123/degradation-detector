"""SQLite-слой: схема и тонкие хелперы (соединение на операцию)."""
import sqlite3
from contextlib import contextmanager

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS historical_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    article_id INTEGER NOT NULL REFERENCES kb_articles(id),
    kind TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    day_label TEXT NOT NULL,
    config_preset TEXT NOT NULL,
    config_json TEXT NOT NULL,
    config_description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    error TEXT,
    avg_judge_score REAL,
    avg_similarity REAL,
    pct_low_factual REAL,
    degradation_detected INTEGER,
    degraded_axes TEXT,
    detection_json TEXT
);
CREATE TABLE IF NOT EXISTS test_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    mutation_type TEXT NOT NULL,
    source_query TEXT NOT NULL,
    question TEXT NOT NULL,
    golden_question TEXT NOT NULL,
    golden_answer TEXT NOT NULL,
    answer TEXT,
    model_used TEXT,
    factual_correctness INTEGER,
    completeness INTEGER,
    relevance INTEGER,
    similarity REAL,
    judge_comment TEXT
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with get_conn() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript(SCHEMA)


def execute(sql, params=()):
    with get_conn() as c:
        cur = c.execute(sql, params)
        return cur.lastrowid


def fetchall(sql, params=()):
    with get_conn() as c:
        return c.execute(sql, params).fetchall()


def fetchone(sql, params=()):
    with get_conn() as c:
        return c.execute(sql, params).fetchone()


def count(table):
    row = fetchone(f"SELECT COUNT(*) AS n FROM {table}")
    return row["n"]
