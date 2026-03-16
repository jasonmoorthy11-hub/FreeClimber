"""Quality scoring for FreeClimber analysis results.

Assigns confidence scores to per-vial and per-video results so
non-technical users know when to trust results vs re-run.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def score_vial(r_value: float, fly_count: float, track_completeness: float = 1.0) -> dict:
    """Score a single vial's result quality.

    Args:
        r_value: regression coefficient from local linear regression
        fly_count: median number of flies detected per frame
        track_completeness: mean track completeness (0-1) if individual tracking enabled

    Returns:
        dict with 'score' (0-1), 'level' (high/medium/low), 'warnings' (list)
    """
    warnings = []
    score = 1.0

    # R-value scoring
    r = abs(r_value) if not np.isnan(r_value) else 0.0
    if r < 0.7:
        score -= 0.4
        warnings.append(f'Low R-value ({r:.2f}): noisy or no climbing detected')
    elif r < 0.85:
        score -= 0.2
        warnings.append(f'Moderate R-value ({r:.2f}): some noise in tracking')

    # Fly count scoring
    if fly_count < 3:
        score -= 0.3
        warnings.append(f'Very few flies detected ({fly_count:.0f}): check ROI/threshold')
    elif fly_count < 5:
        score -= 0.1
        warnings.append(f'Low fly count ({fly_count:.0f})')

    # Track completeness scoring
    if track_completeness < 0.5:
        score -= 0.2
        warnings.append(f'Low track completeness ({track_completeness:.0%})')
    elif track_completeness < 0.8:
        score -= 0.1

    score = max(0.0, min(1.0, score))

    if score >= 0.7:
        level = 'high'
    elif score >= 0.4:
        level = 'medium'
    else:
        level = 'low'

    return {'score': round(score, 2), 'level': level, 'warnings': warnings}


def score_video(slopes_df: pd.DataFrame, df_filtered: pd.DataFrame = None) -> dict:
    """Score overall video quality from slopes and tracking data.

    Returns:
        dict with 'overall_score', 'overall_level', 'per_vial' scores, 'warnings'
    """
    per_vial = {}
    all_warnings = []

    for _, row in slopes_df.iterrows():
        vial_id = row.get('vial_ID', row.name)
        r = row.get('r_value', 0.0)

        # Compute fly count for this vial if tracking data available
        fly_count = 5  # default assumption
        completeness = 1.0
        if df_filtered is not None and 'vial' in df_filtered.columns:
            vial_num = row.name if isinstance(row.name, int) else 0
            vial_data = df_filtered[df_filtered.vial == vial_num] if vial_num > 0 else df_filtered
            if len(vial_data) > 0 and 'frame' in vial_data.columns:
                fly_count = vial_data.groupby('frame').size().median()
            if 'track_completeness' in vial_data.columns:
                completeness = vial_data.track_completeness.mean()

        vial_score = score_vial(r, fly_count, completeness)
        per_vial[str(vial_id)] = vial_score
        all_warnings.extend(
            [f"Vial {vial_id}: {w}" for w in vial_score['warnings']]
        )

    # Overall score = mean of per-vial scores
    scores = [v['score'] for v in per_vial.values()]
    overall = np.mean(scores) if scores else 0.0

    if overall >= 0.7:
        level = 'high'
    elif overall >= 0.4:
        level = 'medium'
    else:
        level = 'low'

    return {
        'overall_score': round(overall, 2),
        'overall_level': level,
        'per_vial': per_vial,
        'warnings': all_warnings,
    }
