"""Tests for analysis/quality.py — confidence scoring."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from analysis.quality import score_vial, score_video


class TestScoreVial:
    def test_high_quality(self):
        result = score_vial(r_value=0.97, fly_count=10, track_completeness=0.95)
        assert result['level'] == 'high'
        assert result['score'] >= 0.7

    def test_low_quality(self):
        result = score_vial(r_value=0.5, fly_count=1, track_completeness=0.3)
        assert result['level'] == 'low'
        assert len(result['warnings']) > 0

    def test_medium_quality(self):
        result = score_vial(r_value=0.88, fly_count=4, track_completeness=0.6)
        assert result['level'] in ('medium', 'high')


class TestScoreVideo:
    def test_basic(self, sample_slopes_df):
        result = score_video(sample_slopes_df)
        assert 'overall_score' in result
        assert 'overall_level' in result
        assert 'per_vial' in result
        assert result['overall_level'] in ('high', 'medium', 'low')
