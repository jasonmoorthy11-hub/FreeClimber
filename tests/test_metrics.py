"""Tests for analysis/metrics.py — per-fly and population metrics."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from analysis.metrics import compute_per_fly_metrics, compute_population_metrics, climbing_index


class TestPerFlyMetrics:
    def test_basic_computation(self, sample_tracking_df):
        result = compute_per_fly_metrics(sample_tracking_df)
        assert len(result) > 0
        assert 'climbing_speed' in result.columns
        assert 'path_straightness' in result.columns
        assert 'track_completeness' in result.columns

    def test_no_particle_column(self):
        df = pd.DataFrame({'x': [1, 2], 'y': [3, 4], 'frame': [0, 1]})
        result = compute_per_fly_metrics(df)
        assert len(result) == 0

    def test_completeness_range(self, sample_tracking_df):
        result = compute_per_fly_metrics(sample_tracking_df)
        assert (result.track_completeness >= 0).all()
        assert (result.track_completeness <= 1).all()

    def test_straightness_range(self, sample_tracking_df):
        result = compute_per_fly_metrics(sample_tracking_df)
        assert (result.path_straightness >= 0).all()
        assert (result.path_straightness <= 1.01).all()  # small float tolerance


class TestPopulationMetrics:
    def test_with_slopes(self, sample_slopes_df):
        metrics = compute_population_metrics(pd.DataFrame(), slopes_df=sample_slopes_df)
        assert 'mean_speed' in metrics
        assert 'median_speed' in metrics

    def test_fly_count(self, sample_tracking_df):
        metrics = compute_population_metrics(sample_tracking_df)
        assert 'fly_count_per_vial' in metrics


class TestClimbingIndex:
    def test_basic(self, sample_tracking_df):
        result = climbing_index(sample_tracking_df)
        assert isinstance(result, dict)
        for vial, idx in result.items():
            assert 0 <= idx <= 100
