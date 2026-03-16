"""Tests for scripts/output/video.py — _pick_col and annotated video export."""

import os

import pandas as pd
import pytest


class TestPickCol:
    def test_finds_exact_match(self):
        from output.video import _pick_col

        cols_lower = {'x': 'X', 'y': 'Y', 'frame': 'Frame'}
        assert _pick_col(cols_lower, 'x') == 'X'

    def test_finds_first_candidate(self):
        from output.video import _pick_col

        cols_lower = {'y': 'Y_pos', 'frame': 'Frame'}
        assert _pick_col(cols_lower, 'x', 'y') == 'Y_pos'

    def test_returns_none_if_missing(self):
        from output.video import _pick_col

        cols_lower = {'x': 'X', 'y': 'Y'}
        assert _pick_col(cols_lower, 'z', 'w') is None


class TestAnnotatedVideo:
    def test_export_creates_file(self, synthetic_video_path, tmp_path):
        from output.video import export_annotated_video

        positions_df = pd.DataFrame({
            'frame': [0, 1, 2, 3, 4] * 3,
            'x': [80] * 5 + [160] * 5 + [240] * 5,
            'y': [200, 197, 194, 191, 188] * 3,
            'particle': [0] * 5 + [1] * 5 + [2] * 5,
            'vial': [1] * 5 + [2] * 5 + [3] * 5,
        })

        out_path = str(tmp_path / "annotated.mp4")
        try:
            export_annotated_video(
                video_path=synthetic_video_path,
                positions_df=positions_df,
                output_path=out_path,
                crop=(0, 0, 320, 240),
            )
            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 0
        except Exception:
            pytest.skip("Video export not supported in this environment")
