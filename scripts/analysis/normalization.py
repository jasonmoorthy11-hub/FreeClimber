"""Normalization methods for cross-experiment comparisons.

Control referencing, batch correction, fly count adjustment.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def normalize_to_control(slopes_df: pd.DataFrame, control_vials: list,
                         metric_col: str = 'slope') -> pd.DataFrame:
    """Normalize all vial speeds as percentage of control mean.

    Args:
        slopes_df: DataFrame with slope data
        control_vials: list of vial_ID values to use as control/wild-type
        metric_col: column to normalize

    Returns:
        DataFrame with added 'normalized_<metric>' column
    """
    df = slopes_df.copy()
    control_mask = df['vial_ID'].isin(control_vials)
    control_mean = df.loc[control_mask, metric_col].mean()

    if control_mean == 0 or np.isnan(control_mean):
        logger.warning("Control mean is zero or NaN; normalization skipped")
        df[f'normalized_{metric_col}'] = np.nan
    else:
        df[f'normalized_{metric_col}'] = (df[metric_col] / control_mean * 100).round(2)

    return df


def batch_zscore(slopes_df: pd.DataFrame, batch_col: str = None,
                 metric_col: str = 'slope') -> pd.DataFrame:
    """Z-score normalization within experimental batch for cross-day comparisons.

    Args:
        slopes_df: DataFrame with slope data
        batch_col: column identifying batch (e.g., date). If None, treats all as one batch.
        metric_col: column to z-score

    Returns:
        DataFrame with added 'zscore_<metric>' column
    """
    df = slopes_df.copy()

    if batch_col and batch_col in df.columns:
        df[f'zscore_{metric_col}'] = df.groupby(batch_col)[metric_col].transform(
            lambda x: ((x - x.mean()) / x.std()).round(4) if x.std() > 0 else 0.0
        )
    else:
        mean = df[metric_col].mean()
        std = df[metric_col].std()
        if std > 0:
            df[f'zscore_{metric_col}'] = ((df[metric_col] - mean) / std).round(4)
        else:
            df[f'zscore_{metric_col}'] = 0.0

    return df


def adjust_for_fly_count(slopes_df: pd.DataFrame, fly_counts: dict,
                         metric_col: str = 'slope') -> pd.DataFrame:
    """Normalize per-fly metrics when vial counts differ significantly.

    Args:
        slopes_df: DataFrame with slope data
        fly_counts: {vial_ID: median_fly_count}
        metric_col: column to adjust

    Returns:
        DataFrame with added 'adjusted_<metric>' column
    """
    df = slopes_df.copy()

    counts = pd.Series(fly_counts)
    median_count = counts.median()

    if median_count == 0:
        logger.warning("Median fly count is zero; adjustment skipped")
        df[f'adjusted_{metric_col}'] = df[metric_col]
        return df

    # Adjust = raw * (median_count / vial_count) — brings all to same effective n
    adjustments = {}
    for vid, count in fly_counts.items():
        adjustments[vid] = median_count / count if count > 0 else 1.0

    df[f'adjusted_{metric_col}'] = df.apply(
        lambda row: round(row[metric_col] * adjustments.get(row['vial_ID'], 1.0), 4),
        axis=1
    )

    return df
