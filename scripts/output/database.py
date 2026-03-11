"""SQLite experiment database for FreeClimber.

Stores all analysis results for cross-experiment queries,
longitudinal tracking, and historical comparisons.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.expanduser('~/.freeclimber/experiments.db')


def get_db_path() -> str:
    db_dir = os.path.dirname(DEFAULT_DB_PATH)
    os.makedirs(db_dir, exist_ok=True)
    return DEFAULT_DB_PATH


def init_db(db_path: str = None) -> sqlite3.Connection:
    """Initialize the experiment database, creating tables if needed."""
    if db_path is None:
        db_path = get_db_path()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            config_json TEXT,
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER REFERENCES experiments(id),
            video_path TEXT NOT NULL,
            video_name TEXT,
            n_frames INTEGER,
            fps REAL,
            width INTEGER,
            height INTEGER,
            quality_score REAL,
            quality_level TEXT,
            processed_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS slopes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER REFERENCES videos(id),
            vial_id TEXT,
            vial_num INTEGER,
            first_frame INTEGER,
            last_frame INTEGER,
            slope REAL,
            intercept REAL,
            r_value REAL,
            p_value REAL,
            std_err REAL,
            quality_score REAL,
            quality_level TEXT
        );

        CREATE TABLE IF NOT EXISTS fly_tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER REFERENCES videos(id),
            particle_id INTEGER,
            vial_num INTEGER,
            n_frames INTEGER,
            climbing_speed REAL,
            start_latency INTEGER,
            max_height REAL,
            path_straightness REAL,
            hesitation_count INTEGER,
            horizontal_drift REAL,
            track_completeness REAL
        );

        CREATE TABLE IF NOT EXISTS stats_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER REFERENCES experiments(id),
            test_name TEXT,
            statistic REAL,
            p_value REAL,
            effect_size REAL,
            effect_size_name TEXT,
            post_hoc_json TEXT,
            computed_at TEXT DEFAULT (datetime('now'))
        );
    """)

    conn.commit()
    return conn


def save_experiment(conn: sqlite3.Connection, name: str,
                    config: dict = None, notes: str = '') -> int:
    """Create a new experiment record. Returns experiment ID."""
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO experiments (name, date, config_json, notes) VALUES (?, ?, ?, ?)",
        (name, datetime.now().isoformat(), json.dumps(config or {}), notes)
    )
    conn.commit()
    return cursor.lastrowid


def save_video(conn: sqlite3.Connection, experiment_id: int,
               video_path: str, n_frames: int = 0, fps: float = 0,
               width: int = 0, height: int = 0,
               quality_score: float = None, quality_level: str = None) -> int:
    """Save a video record. Returns video ID."""
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO videos (experiment_id, video_path, video_name, n_frames, fps,
           width, height, quality_score, quality_level)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (experiment_id, video_path, os.path.basename(video_path),
         n_frames, fps, width, height, quality_score, quality_level)
    )
    conn.commit()
    return cursor.lastrowid


def save_slopes(conn: sqlite3.Connection, video_id: int,
                slopes_df: pd.DataFrame):
    """Save slope results for a video."""
    for _, row in slopes_df.iterrows():
        conn.execute(
            """INSERT INTO slopes (video_id, vial_id, slope, intercept,
               r_value, p_value, std_err, first_frame, last_frame)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (video_id,
             row.get('vial_ID', ''),
             row.get('slope'),
             row.get('intercept'),
             row.get('r_value'),
             row.get('p_value'),
             row.get('std_err'),
             row.get('first_frame'),
             row.get('last_frame'))
        )
    conn.commit()


def save_fly_tracks(conn: sqlite3.Connection, video_id: int,
                    metrics_df: pd.DataFrame):
    """Save per-fly tracking metrics."""
    for _, row in metrics_df.iterrows():
        conn.execute(
            """INSERT INTO fly_tracks (video_id, particle_id, vial_num,
               n_frames, climbing_speed, start_latency, max_height,
               path_straightness, hesitation_count, horizontal_drift,
               track_completeness)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (video_id,
             int(row.get('particle', 0)),
             int(row.get('vial', 0)),
             int(row.get('n_frames', 0)),
             row.get('climbing_speed'),
             row.get('start_latency'),
             row.get('max_height'),
             row.get('path_straightness'),
             row.get('hesitation_count'),
             row.get('horizontal_drift'),
             row.get('track_completeness'))
        )
    conn.commit()


def query_experiments(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get all experiments with summary info."""
    return pd.read_sql_query("""
        SELECT e.id, e.name, e.date, e.notes,
               COUNT(DISTINCT v.id) as n_videos,
               AVG(v.quality_score) as avg_quality
        FROM experiments e
        LEFT JOIN videos v ON v.experiment_id = e.id
        GROUP BY e.id
        ORDER BY e.date DESC
    """, conn)


def query_slopes(conn: sqlite3.Connection, experiment_id: int = None) -> pd.DataFrame:
    """Get all slopes, optionally filtered by experiment."""
    query = """
        SELECT s.*, v.video_name, v.video_path, e.name as experiment_name
        FROM slopes s
        JOIN videos v ON v.id = s.video_id
        JOIN experiments e ON e.id = v.experiment_id
    """
    if experiment_id:
        query += f" WHERE e.id = {experiment_id}"
    return pd.read_sql_query(query, conn)
