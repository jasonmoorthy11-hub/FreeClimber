"""Tests for analysis/stats.py — statistical testing."""


import numpy as np
import pytest
from analysis.stats import (
    check_normality,
    cohens_d,
    compare_multiple_groups,
    compare_two_groups,
    confidence_interval,
    correct_pvalues,
    dunnett_vs_control,
    publication_stats_table,
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


class TestPValueCorrections:
    def test_holm_bonferroni(self):
        pvals = [0.01, 0.04, 0.03, 0.005]
        adjusted = correct_pvalues(pvals, method='holm')
        assert len(adjusted) == 4
        # All adjusted values >= raw values
        for raw, adj in zip(pvals, adjusted):
            assert adj >= raw
        # All <= 1.0
        assert all(p <= 1.0 for p in adjusted)

    def test_benjamini_hochberg(self):
        pvals = [0.01, 0.04, 0.03, 0.005]
        adjusted = correct_pvalues(pvals, method='bh')
        assert len(adjusted) == 4
        assert all(p <= 1.0 for p in adjusted)
        # Smallest raw p should remain significant
        assert adjusted[3] < 0.05

    def test_holm_with_all_significant(self):
        pvals = [0.001, 0.002, 0.003]
        adjusted = correct_pvalues(pvals, method='holm')
        assert all(p < 0.05 for p in adjusted)

    def test_empty_pvalues(self):
        assert correct_pvalues([], method='holm') == []
        assert correct_pvalues([], method='bh') == []


class TestDunnett:
    def test_dunnett_vs_control(self, three_groups):
        result = dunnett_vs_control(three_groups, control_name='wildtype')
        assert len(result) == 2
        for comp in result:
            assert comp['vs_control'] == 'wildtype'
            assert 'p_value' in comp
            assert 'effect_size_d' in comp
            assert 'significance' in comp

    def test_invalid_control(self, three_groups):
        with pytest.raises(ValueError):
            dunnett_vs_control(three_groups, control_name='nonexistent')

    def test_only_control_group(self):
        groups = {'control': np.array([1, 2, 3, 4, 5])}
        result = dunnett_vs_control(groups, control_name='control')
        assert result == []
