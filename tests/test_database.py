"""Tests for output/database.py — SQLite experiment database."""


import pytest
from output.database import (
    init_db,
    query_experiments,
    query_slopes,
    save_experiment,
    save_slopes,
    save_video,
)


@pytest.fixture
def conn():
    c = init_db(':memory:')
    yield c
    c.close()


class TestInitDB:
    def test_creates_tables(self, conn):
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert 'experiments' in tables
        assert 'videos' in tables
        assert 'slopes' in tables
        assert 'fly_tracks' in tables
        assert 'stats_results' in tables


class TestExperimentCRUD:
    def test_save_and_query(self, conn):
        eid = save_experiment(conn, 'Test Experiment', config={'vials': 3}, notes='test')
        assert eid > 0
        df = query_experiments(conn)
        assert len(df) == 1
        assert df.iloc[0]['name'] == 'Test Experiment'

    def test_multiple_experiments(self, conn):
        save_experiment(conn, 'Exp 1')
        save_experiment(conn, 'Exp 2')
        df = query_experiments(conn)
        assert len(df) == 2


class TestVideoAndSlopes:
    def test_save_video_and_slopes(self, conn, sample_slopes_df):
        eid = save_experiment(conn, 'Test')
        vid = save_video(conn, eid, '/path/to/video.h264', n_frames=300, fps=30)
        assert vid > 0
        save_slopes(conn, vid, sample_slopes_df)
        df = query_slopes(conn, experiment_id=eid)
        assert len(df) == len(sample_slopes_df)
        assert 'slope' in df.columns

    def test_query_slopes_all(self, conn, sample_slopes_df):
        eid = save_experiment(conn, 'Test')
        vid = save_video(conn, eid, '/path/video.mp4')
        save_slopes(conn, vid, sample_slopes_df)
        df = query_slopes(conn)
        assert len(df) == len(sample_slopes_df)
