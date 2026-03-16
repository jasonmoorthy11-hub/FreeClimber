"""Tests for analysis/metrics.py — per-fly and population metrics."""


import numpy as np
import pandas as pd
import pytest
from analysis.metrics import climbing_index, compute_per_fly_metrics, compute_population_metrics


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

    def test_auc_computed(self, sample_tracking_df):
        result = compute_per_fly_metrics(sample_tracking_df)
        assert 'auc' in result.columns

    def test_auc_displacement_based(self):
        """A climbing fly should have higher AUC than a stationary fly."""
        rows_climb = [{'particle': 0, 'frame': f, 'x': 100, 'y': 100 - f * 2, 'vial': 1}
                      for f in range(50)]
        rows_still = [{'particle': 1, 'frame': f, 'x': 200, 'y': 100, 'vial': 1}
                      for f in range(50)]
        df = pd.DataFrame(rows_climb + rows_still)
        result = compute_per_fly_metrics(df)
        climber = result[result.particle == 0].iloc[0]
        sitter = result[result.particle == 1].iloc[0]
        assert climber.auc > sitter.auc


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
        for _vial, idx in result.items():
            assert 0 <= idx <= 100
