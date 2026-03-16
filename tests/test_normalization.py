"""Tests for analysis/normalization.py — normalization methods."""


import numpy as np
import pandas as pd
import pytest
from analysis.normalization import adjust_for_fly_count, batch_zscore, normalize_to_control


class TestNormalizeToControl:
    def test_control_becomes_100(self, sample_slopes_df):
        result = normalize_to_control(sample_slopes_df, control_vials=['wt_m_1', 'wt_m_2', 'wt_m_3'])
        assert 'normalized_slope' in result.columns
        control = result[result['vial_ID'].isin(['wt_m_1', 'wt_m_2', 'wt_m_3'])]
        assert abs(control['normalized_slope'].mean() - 100.0) < 1.0

    def test_preserves_original(self, sample_slopes_df):
        result = normalize_to_control(sample_slopes_df, control_vials=['wt_m_1'])
        assert 'slope' in result.columns
        assert (result['slope'] == sample_slopes_df['slope']).all()


class TestBatchZscore:
    def test_zscore_properties(self, sample_slopes_df):
        result = batch_zscore(sample_slopes_df)
        assert 'zscore_slope' in result.columns
        z = result['zscore_slope']
        assert abs(z.mean()) < 0.1
        assert abs(z.std() - 1.0) < 0.3  # small sample tolerance

    def test_constant_values(self):
        df = pd.DataFrame({'slope': [5.0, 5.0, 5.0], 'vial_ID': ['a', 'b', 'c']})
        result = batch_zscore(df)
        assert (result['zscore_slope'] == 0.0).all()


class TestAdjustForFlyCount:
    def test_equal_counts_no_change(self, sample_slopes_df):
        counts = {vid: 10 for vid in sample_slopes_df['vial_ID']}
        result = adjust_for_fly_count(sample_slopes_df, fly_counts=counts)
        assert 'adjusted_slope' in result.columns
        np.testing.assert_array_almost_equal(
            result['adjusted_slope'].values, sample_slopes_df['slope'].values, decimal=3
        )

    def test_unequal_counts_adjusts(self, sample_slopes_df):
        counts = {'wt_m_1': 20, 'wt_m_2': 10, 'wt_m_3': 10, 'wt_m_all': 10}
        result = adjust_for_fly_count(sample_slopes_df, fly_counts=counts)
        # vial with 20 flies should be adjusted down
        row_20 = result[result['vial_ID'] == 'wt_m_1'].iloc[0]
        assert row_20['adjusted_slope'] < row_20['slope']
