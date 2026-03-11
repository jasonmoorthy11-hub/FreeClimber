"""Per-fly and population metrics for FreeClimber.

Requires individual tracking (particle column in DataFrame).
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import linregress

logger = logging.getLogger(__name__)


def compute_per_fly_metrics(df: pd.DataFrame, frame_rate: int = 30,
                            pixel_to_cm: float = 1.0,
                            convert_to_cm_sec: bool = False) -> pd.DataFrame:
    """Compute per-fly metrics from linked tracking data.

    Args:
        df: DataFrame with columns: particle, frame, x, y, vial
        frame_rate: video fps
        pixel_to_cm: calibration factor
        convert_to_cm_sec: whether to convert units

    Returns:
        DataFrame with one row per fly, columns for each metric
    """
    if 'particle' not in df.columns:
        logger.warning("No 'particle' column — individual tracking not available")
        return pd.DataFrame()

    conversion = pixel_to_cm / frame_rate if convert_to_cm_sec else 1.0
    results = []

    for pid, track in df.groupby('particle'):
        track = track.sort_values('frame')
        n_frames = len(track)

        if n_frames < 3:
            continue

        vial = track.vial.mode().iloc[0] if 'vial' in track.columns else 0
        dt = np.diff(track.frame.values).astype(float)
        dx = np.diff(track.x.values)
        dy = np.diff(track.y.values)

        # Instantaneous speeds
        speeds = np.sqrt(dx**2 + dy**2) / np.maximum(dt, 1)
        vy = dy / np.maximum(dt, 1)  # vertical velocity per step

        # Climbing speed (mean vertical velocity)
        climbing_speed = np.mean(vy) * conversion

        # Start latency: frames until first sustained upward movement (3+ consecutive)
        upward = vy > 0
        start_latency = _find_sustained_start(upward, threshold=3)

        # Maximum height reached
        max_height = track.y.max()
        if convert_to_cm_sec:
            max_height = max_height / pixel_to_cm

        # Path straightness: net vertical / total path length
        net_vertical = abs(track.y.iloc[-1] - track.y.iloc[0])
        total_path = np.sum(np.sqrt(dx**2 + dy**2))
        straightness = net_vertical / total_path if total_path > 0 else 0.0

        # Hesitation count: pauses where speed < threshold for 3+ frames
        speed_threshold = np.median(speeds) * 0.1 if len(speeds) > 0 else 0
        hesitations = _count_hesitations(speeds, speed_threshold, min_duration=3)

        # Horizontal drift: std of x position
        x_drift = track.x.std()
        if convert_to_cm_sec:
            x_drift = x_drift / pixel_to_cm

        # Track completeness
        total_possible = track.frame.max() - track.frame.min() + 1
        completeness = n_frames / total_possible if total_possible > 0 else 0.0

        # AUC: area under climb displacement curve (upward movement from start)
        time_vals = track.frame.values.astype(float)
        if convert_to_cm_sec:
            time_vals = time_vals / frame_rate
        # Displacement from starting position (positive = upward in video coords where y=0 is top)
        y_vals = (track.y.values[0] - track.y.values).astype(float)
        if convert_to_cm_sec:
            y_vals = y_vals / pixel_to_cm
        _trapz = getattr(np, 'trapezoid', np.trapz)
        auc = float(_trapz(y_vals, time_vals)) if len(time_vals) > 1 else 0.0

        results.append({
            'particle': pid,
            'vial': int(vial),
            'n_frames': n_frames,
            'climbing_speed': round(climbing_speed, 4),
            'start_latency': start_latency,
            'max_height': round(max_height, 2),
            'path_straightness': round(straightness, 4),
            'hesitation_count': hesitations,
            'horizontal_drift': round(x_drift, 4),
            'track_completeness': round(completeness, 4),
            'mean_speed': round(np.mean(speeds) * conversion, 4),
            'auc': round(auc, 4),
        })

    return pd.DataFrame(results)


def compute_population_metrics(df: pd.DataFrame, slopes_df: pd.DataFrame = None,
                               frame_rate: int = 30, pixel_to_cm: float = 1.0,
                               convert_to_cm_sec: bool = False) -> dict:
    """Compute population-level analytics.

    Returns dict of metric_name -> value or DataFrame.
    """
    metrics = {}

    if slopes_df is not None and 'slope' in slopes_df.columns:
        slopes = slopes_df.slope.dropna().values

        metrics['mean_speed'] = round(float(np.mean(slopes)), 4)
        metrics['median_speed'] = round(float(np.median(slopes)), 4)
        metrics['speed_std'] = round(float(np.std(slopes)), 4)
        metrics['p25'] = round(float(np.percentile(slopes, 25)), 4)
        metrics['p75'] = round(float(np.percentile(slopes, 75)), 4)

    # Fly count per vial per frame
    if 'vial' in df.columns and 'frame' in df.columns:
        counts = df.groupby(['vial', 'frame']).size().reset_index(name='count')
        median_counts = counts.groupby('vial')['count'].median()
        metrics['fly_count_per_vial'] = median_counts.to_dict()

    return metrics


def climbing_index(df: pd.DataFrame, threshold_height: float = None,
                   at_frame: int = None) -> dict:
    """Compute climbing index: % flies above threshold height at given frame.

    Args:
        df: DataFrame with y, frame, vial columns
        threshold_height: y-position threshold (default: 50% of max y)
        at_frame: frame to evaluate (default: last frame)

    Returns:
        dict: {vial: climbing_index_percentage}
    """
    if threshold_height is None:
        threshold_height = df.y.max() * 0.5
    if at_frame is None:
        at_frame = df.frame.max()

    frame_df = df[df.frame == at_frame]
    result = {}

    for vial, vdf in frame_df.groupby('vial'):
        n_total = len(vdf)
        n_above = len(vdf[vdf.y >= threshold_height])
        result[int(vial)] = round(n_above / n_total * 100, 1) if n_total > 0 else 0.0

    return result


def _find_sustained_start(upward: np.ndarray, threshold: int = 3) -> int:
    """Find first index where threshold consecutive True values occur."""
    count = 0
    for i, v in enumerate(upward):
        if v:
            count += 1
            if count >= threshold:
                return i - threshold + 1
        else:
            count = 0
    return len(upward)  # never sustained


def _count_hesitations(speeds: np.ndarray, threshold: float, min_duration: int = 3) -> int:
    """Count number of pauses (speed below threshold for min_duration+ frames)."""
    count = 0
    run = 0
    for s in speeds:
        if s < threshold:
            run += 1
        else:
            if run >= min_duration:
                count += 1
            run = 0
    if run >= min_duration:
        count += 1
    return count
