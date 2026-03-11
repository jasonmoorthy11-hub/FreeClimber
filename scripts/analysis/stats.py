"""Statistical testing for FreeClimber.

Automated parametric vs non-parametric test selection with post-hoc
comparisons and effect sizes. Publication-ready output.
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging
from itertools import combinations

logger = logging.getLogger(__name__)


def test_normality(groups: dict, alpha: float = 0.05) -> dict:
    """Shapiro-Wilk normality test per group.

    Args:
        groups: {group_name: array_of_values}
        alpha: significance threshold

    Returns:
        dict with keys: results (per-group), all_normal (bool)
    """
    results = {}
    all_normal = True
    for name, values in groups.items():
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        if len(values) < 3:
            results[name] = {'statistic': np.nan, 'p_value': np.nan, 'normal': False, 'n': len(values)}
            all_normal = False
            continue
        stat, p = stats.shapiro(values)
        is_normal = p > alpha
        if not is_normal:
            all_normal = False
        results[name] = {'statistic': round(stat, 4), 'p_value': round(p, 4), 'normal': is_normal, 'n': len(values)}
    return {'results': results, 'all_normal': all_normal}


def compare_two_groups(group1: np.ndarray, group2: np.ndarray,
                       normal: bool = None, alpha: float = 0.05) -> dict:
    """Compare two groups with appropriate test.

    Auto-selects Student's t-test (normal) or Mann-Whitney U (non-normal).
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1, g2 = g1[~np.isnan(g1)], g2[~np.isnan(g2)]

    if normal is None:
        norm1 = stats.shapiro(g1)[1] > alpha if len(g1) >= 3 else False
        norm2 = stats.shapiro(g2)[1] > alpha if len(g2) >= 3 else False
        normal = norm1 and norm2

    if normal:
        stat, p = stats.ttest_ind(g1, g2)
        test_name = "Student's t-test"
    else:
        stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        test_name = "Mann-Whitney U"

    d = cohens_d(g1, g2)
    sig = _significance_label(p)

    return {
        'test': test_name,
        'statistic': round(float(stat), 4),
        'p_value': round(float(p), 6),
        'effect_size_d': round(d, 4),
        'significance': sig,
        'n1': len(g1),
        'n2': len(g2),
    }


def compare_multiple_groups(groups: dict, normal: bool = None, alpha: float = 0.05) -> dict:
    """Compare 3+ groups with ANOVA or Kruskal-Wallis, plus post-hoc tests.

    Args:
        groups: {group_name: array_of_values}
        normal: force parametric (True) or non-parametric (False), or auto-detect (None)

    Returns:
        dict with omnibus test, post_hoc comparisons, effect sizes
    """
    group_names = list(groups.keys())
    arrays = [np.asarray(v, dtype=float) for v in groups.values()]
    arrays = [a[~np.isnan(a)] for a in arrays]

    if normal is None:
        norm_test = test_normality(groups, alpha)
        normal = norm_test['all_normal']

    # Omnibus test
    if normal:
        stat, p = stats.f_oneway(*arrays)
        test_name = "One-way ANOVA"
        # Eta-squared effect size
        grand_mean = np.concatenate(arrays).mean()
        ss_between = sum(len(a) * (a.mean() - grand_mean)**2 for a in arrays)
        ss_total = sum(np.sum((a - grand_mean)**2) for a in arrays)
        effect_size = round(ss_between / ss_total, 4) if ss_total > 0 else 0.0
        effect_name = "eta_squared"
    else:
        stat, p = stats.kruskal(*arrays)
        test_name = "Kruskal-Wallis"
        # Epsilon-squared effect size
        n_total = sum(len(a) for a in arrays)
        effect_size = round((float(stat) - len(arrays) + 1) / (n_total - len(arrays)), 4) if n_total > len(arrays) else 0.0
        effect_name = "epsilon_squared"

    result = {
        'test': test_name,
        'statistic': round(float(stat), 4),
        'p_value': round(float(p), 6),
        'effect_size': effect_size,
        'effect_size_name': effect_name,
        'significant': p < alpha,
        'n_groups': len(groups),
        'parametric': normal,
    }

    # Post-hoc tests (only if omnibus is significant)
    if p < alpha:
        if normal:
            result['post_hoc'] = _tukey_hsd(groups)
            result['post_hoc_method'] = "Tukey HSD"
        else:
            result['post_hoc'] = _dunns_test(groups)
            result['post_hoc_method'] = "Dunn's test (Bonferroni)"
    else:
        result['post_hoc'] = []
        result['post_hoc_method'] = "N/A (omnibus not significant)"

    return result


def _tukey_hsd(groups: dict) -> list:
    """Tukey HSD post-hoc test for all pairwise comparisons."""
    group_names = list(groups.keys())
    arrays = [np.asarray(v, dtype=float) for v in groups.values()]
    arrays = [a[~np.isnan(a)] for a in arrays]

    try:
        result = stats.tukey_hsd(*arrays)
    except AttributeError:
        # scipy < 1.8 fallback: manual pairwise t-tests with Bonferroni
        return _pairwise_ttest_bonferroni(groups)

    comparisons = []
    for i, j in combinations(range(len(group_names)), 2):
        p = float(result.pvalue[i][j])
        d = cohens_d(arrays[i], arrays[j])
        comparisons.append({
            'group1': group_names[i],
            'group2': group_names[j],
            'p_value': round(p, 6),
            'effect_size_d': round(d, 4),
            'significance': _significance_label(p),
        })
    return comparisons


def _pairwise_ttest_bonferroni(groups: dict) -> list:
    """Fallback pairwise t-tests with Bonferroni correction."""
    group_names = list(groups.keys())
    arrays = {k: np.asarray(v, dtype=float)[~np.isnan(np.asarray(v, dtype=float))]
              for k, v in groups.items()}
    n_comparisons = len(list(combinations(group_names, 2)))

    comparisons = []
    for g1, g2 in combinations(group_names, 2):
        _, p = stats.ttest_ind(arrays[g1], arrays[g2])
        p_adj = min(p * n_comparisons, 1.0)
        d = cohens_d(arrays[g1], arrays[g2])
        comparisons.append({
            'group1': g1,
            'group2': g2,
            'p_value': round(p_adj, 6),
            'effect_size_d': round(d, 4),
            'significance': _significance_label(p_adj),
        })
    return comparisons


def _dunns_test(groups: dict) -> list:
    """Dunn's test with Bonferroni correction for non-parametric post-hoc."""
    try:
        import scikit_posthocs as sp
        group_names = list(groups.keys())
        data = []
        labels = []
        for name, values in groups.items():
            arr = np.asarray(values, dtype=float)
            arr = arr[~np.isnan(arr)]
            data.extend(arr.tolist())
            labels.extend([name] * len(arr))

        df = pd.DataFrame({'value': data, 'group': labels})
        result = sp.posthoc_dunn(df, val_col='value', group_col='group', p_adjust='bonferroni')

        comparisons = []
        for g1, g2 in combinations(group_names, 2):
            p = float(result.loc[g1, g2])
            a1 = np.asarray(groups[g1], dtype=float)
            a2 = np.asarray(groups[g2], dtype=float)
            d = cohens_d(a1[~np.isnan(a1)], a2[~np.isnan(a2)])
            comparisons.append({
                'group1': g1,
                'group2': g2,
                'p_value': round(p, 6),
                'effect_size_d': round(d, 4),
                'significance': _significance_label(p),
            })
        return comparisons
    except ImportError:
        logger.warning("scikit-posthocs not installed; falling back to pairwise Mann-Whitney with Bonferroni")
        return _pairwise_mannwhitney_bonferroni(groups)


def _pairwise_mannwhitney_bonferroni(groups: dict) -> list:
    """Fallback pairwise Mann-Whitney with Bonferroni correction."""
    group_names = list(groups.keys())
    arrays = {k: np.asarray(v, dtype=float)[~np.isnan(np.asarray(v, dtype=float))]
              for k, v in groups.items()}
    n_comparisons = len(list(combinations(group_names, 2)))

    comparisons = []
    for g1, g2 in combinations(group_names, 2):
        _, p = stats.mannwhitneyu(arrays[g1], arrays[g2], alternative='two-sided')
        p_adj = min(p * n_comparisons, 1.0)
        d = cohens_d(arrays[g1], arrays[g2])
        comparisons.append({
            'group1': g1,
            'group2': g2,
            'p_value': round(p_adj, 6),
            'effect_size_d': round(d, 4),
            'significance': _significance_label(p_adj),
        })
    return comparisons


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size for two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple:
    """95% CI on the mean."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 2:
        return (np.nan, np.nan)
    mean = np.mean(values)
    se = stats.sem(values)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (round(mean - h, 4), round(mean + h, 4))


def _significance_label(p: float) -> str:
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


def publication_stats_table(groups: dict, metric_name: str = 'slope') -> pd.DataFrame:
    """Generate a publication-format statistics summary table.

    Columns: Group | N | Mean +/- SEM | 95% CI | Test Statistic | p-value | Effect Size | Sig
    """
    rows = []
    for name, values in groups.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        ci = confidence_interval(arr)
        rows.append({
            'Group': name,
            'N': len(arr),
            'Mean': round(np.mean(arr), 4) if len(arr) > 0 else np.nan,
            'SEM': round(stats.sem(arr), 4) if len(arr) > 1 else np.nan,
            'CI_lower': ci[0],
            'CI_upper': ci[1],
        })

    df = pd.DataFrame(rows)

    # Add omnibus test results
    n_groups = len(groups)
    if n_groups == 2:
        keys = list(groups.keys())
        result = compare_two_groups(
            np.asarray(groups[keys[0]], dtype=float),
            np.asarray(groups[keys[1]], dtype=float),
        )
        df['Test'] = result['test']
        df['Statistic'] = result['statistic']
        df['p_value'] = result['p_value']
        df['Effect_Size'] = result['effect_size_d']
        df['Significance'] = result['significance']
    elif n_groups >= 3:
        result = compare_multiple_groups(groups)
        df['Test'] = result['test']
        df['Statistic'] = result['statistic']
        df['p_value'] = result['p_value']
        df['Effect_Size'] = result['effect_size']
        df['Significance'] = 'sig' if result['significant'] else 'ns'

    return df
