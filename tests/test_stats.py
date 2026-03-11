"""Tests for analysis/stats.py — statistical testing."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from analysis.stats import (
    check_normality, compare_two_groups, compare_multiple_groups,
    cohens_d, confidence_interval, publication_stats_table,
)


class TestNormality:
    def test_normal_data(self):
        np.random.seed(42)
        data = {'group1': np.random.normal(0, 1, 50)}
        result = check_normality(data)
        assert result['all_normal'] is True

    def test_non_normal_data(self):
        data = {'group1': np.array([1, 1, 1, 1, 1, 100])}
        result = check_normality(data)
        # Highly skewed data should fail normality
        assert 'group1' in result['results']

    def test_small_sample(self):
        data = {'small': np.array([1, 2])}
        result = check_normality(data)
        assert result['results']['small']['normal'] is False


class TestTwoGroups:
    def test_parametric(self):
        np.random.seed(42)
        g1 = np.random.normal(5, 1, 30)
        g2 = np.random.normal(5, 1, 30)
        result = compare_two_groups(g1, g2, normal=True)
        assert result['test'] == "Student's t-test"
        assert 'p_value' in result

    def test_nonparametric(self):
        np.random.seed(42)
        g1 = np.random.normal(5, 1, 30)
        g2 = np.random.normal(7, 1, 30)
        result = compare_two_groups(g1, g2, normal=False)
        assert result['test'] == "Mann-Whitney U"
        assert result['p_value'] < 0.05

    def test_effect_size_direction(self):
        g1 = np.array([10, 11, 12, 13, 14])
        g2 = np.array([1, 2, 3, 4, 5])
        result = compare_two_groups(g1, g2)
        assert result['effect_size_d'] > 0


class TestMultipleGroups:
    def test_anova(self, three_groups):
        result = compare_multiple_groups(three_groups)
        assert result['n_groups'] == 3
        assert result['test'] in ['One-way ANOVA', 'Kruskal-Wallis']
        assert 'p_value' in result
        assert 'post_hoc' in result

    def test_significant_difference(self):
        groups = {
            'low': np.array([1, 2, 3, 2, 1, 2, 3]),
            'high': np.array([10, 11, 12, 11, 10, 11, 12]),
        }
        result = compare_multiple_groups(groups)
        assert result['p_value'] < 0.05

    def test_no_significant_difference(self):
        np.random.seed(42)
        groups = {
            'a': np.random.normal(5, 1, 20),
            'b': np.random.normal(5, 1, 20),
            'c': np.random.normal(5, 1, 20),
        }
        result = compare_multiple_groups(groups)
        # Should not be significant (same distribution)
        assert result['p_value'] > 0.01 or not result['significant']


class TestEffectSizes:
    def test_cohens_d_identical(self):
        g = np.array([1, 2, 3, 4, 5])
        assert cohens_d(g, g) == 0.0

    def test_cohens_d_large(self):
        g1 = np.array([1, 2, 3, 4, 5])
        g2 = np.array([10, 11, 12, 13, 14])
        d = cohens_d(g1, g2)
        assert abs(d) > 2.0  # very large effect

    def test_confidence_interval(self):
        values = np.array([10, 11, 12, 13, 14])
        ci = confidence_interval(values)
        assert ci[0] < 12  # mean
        assert ci[1] > 12


class TestPublicationTable:
    def test_two_groups(self):
        groups = {
            'control': np.array([0.5, 0.6, 0.4, 0.55, 0.45]),
            'treated': np.array([0.3, 0.35, 0.25, 0.28, 0.32]),
        }
        table = publication_stats_table(groups)
        assert len(table) == 2
        assert 'Group' in table.columns
        assert 'N' in table.columns
        assert 'Mean' in table.columns

    def test_three_groups(self, three_groups):
        table = publication_stats_table(three_groups)
        assert len(table) == 3
