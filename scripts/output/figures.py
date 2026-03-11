"""Publication-ready figure generation for FreeClimber.

Colorblind-safe palettes, journal formatting, 300 DPI export.
"""

import logging

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)

# Okabe-Ito colorblind-safe palette (8 colors)
OKABE_ITO = [
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # green
    '#F0E442',  # yellow
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#CC79A7',  # pink
    '#000000',  # black
]


def get_color_palette(n: int) -> list:
    """Get a colorblind-safe color palette with n colors."""
    if n <= len(OKABE_ITO):
        return OKABE_ITO[:n]
    # Extend with viridis for larger sets
    cmap = plt.cm.viridis
    return [cmap(i / (n - 1)) for i in range(n)]


def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def bar_chart_with_points(groups: dict, ylabel: str = 'Climbing Speed',
                          title: str = '', significance: dict = None,
                          ax=None) -> plt.Axes:
    """Grouped bar chart with individual data points + error bars.

    Args:
        groups: {group_name: array_of_values}
        ylabel: y-axis label
        title: plot title
        significance: dict of {(group1, group2): '***'} for bracket annotations
        ax: matplotlib axes (creates new if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    colors = get_color_palette(len(groups))
    names = list(groups.keys())
    x_positions = np.arange(len(names))

    for i, (_name, values) in enumerate(groups.items()):
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        mean = np.mean(values)
        sem = np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0

        # Bar
        ax.bar(i, mean, yerr=sem, color=colors[i], alpha=0.7,
               capsize=4, edgecolor='black', linewidth=0.5)

        # Individual data points (jittered)
        jitter = np.random.normal(0, 0.06, len(values))
        ax.scatter(np.full_like(values, i) + jitter, values,
                   color=colors[i], edgecolor='black', linewidth=0.3,
                   s=20, zorder=3, alpha=0.8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(names, rotation=45 if len(names) > 4 else 0, ha='right')
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Add significance brackets
    if significance:
        _add_significance_brackets(ax, names, significance)

    return ax


def box_swarm_plot(groups: dict, ylabel: str = 'Climbing Speed',
                   title: str = '', ax=None) -> plt.Axes:
    """Box + swarm plots showing full distribution."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    colors = get_color_palette(len(groups))
    names = list(groups.keys())
    data = [np.asarray(v, dtype=float)[~np.isnan(np.asarray(v, dtype=float))] for v in groups.values()]

    bp = ax.boxplot(data, labels=names, patch_artist=True, widths=0.5,
                    medianprops={'color': 'black', 'linewidth': 1.5})

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # Swarm overlay
    for i, (values, color) in enumerate(zip(data, colors)):
        jitter = np.random.normal(0, 0.04, len(values))
        ax.scatter(np.full_like(values, i + 1) + jitter, values,
                   color=color, edgecolor='black', linewidth=0.3,
                   s=15, zorder=3, alpha=0.8)

    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    return ax


def trajectory_plot(df_filtered: pd.DataFrame, vials: int, color_list: list = None,
                    convert_to_cm_sec: bool = False, frame_rate: int = 30,
                    pixel_to_cm: float = 1.0, ax=None) -> plt.Axes:
    """Climbing trajectory: mean y-position vs time, colored by vial."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    if color_list is None:
        color_list = get_color_palette(vials)

    for v in range(1, vials + 1):
        vdf = df_filtered[df_filtered.vial == v]
        if len(vdf) == 0:
            continue

        x = vdf.groupby('frame').frame.mean()
        y = vdf.groupby('frame').y.mean()

        if convert_to_cm_sec:
            x = x / frame_rate
            y = y / pixel_to_cm

        ax.plot(x, y, color=color_list[v - 1], label=f'Vial {v}', alpha=0.8)

    ax.legend(loc='best', frameon=False, fontsize='small')
    ax.set_xlabel('Time (s)' if convert_to_cm_sec else 'Frame')
    ax.set_ylabel('Mean y-position (cm)' if convert_to_cm_sec else 'Mean y-position (px)')
    ax.set_title('Climbing Trajectory')
    ax.set_ylim(bottom=0)

    return ax


def speed_distribution(groups: dict, ylabel: str = 'Density',
                       title: str = 'Speed Distribution', ax=None) -> plt.Axes:
    """KDE curves per genotype, overlaid."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    colors = get_color_palette(len(groups))

    for (name, values), color in zip(groups.items(), colors):
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        if len(values) < 2:
            continue
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(values)
        x_range = np.linspace(values.min() - values.std(), values.max() + values.std(), 200)
        ax.plot(x_range, kde(x_range), color=color, label=name, linewidth=1.5)
        ax.fill_between(x_range, kde(x_range), alpha=0.2, color=color)

    ax.set_xlabel('Speed')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize='small')

    return ax


def _add_significance_brackets(ax, group_names, significance):
    """Add significance bracket annotations to a bar chart."""
    y_max = ax.get_ylim()[1]
    y_step = y_max * 0.08
    current_y = y_max * 1.05

    for (g1, g2), label in significance.items():
        if label == 'ns':
            continue
        try:
            x1 = group_names.index(g1)
            x2 = group_names.index(g2)
        except ValueError:
            continue

        ax.plot([x1, x1, x2, x2], [current_y, current_y + y_step * 0.3,
                current_y + y_step * 0.3, current_y],
                color='black', linewidth=0.8)
        ax.text((x1 + x2) / 2, current_y + y_step * 0.35, label,
                ha='center', va='bottom', fontsize=9)
        current_y += y_step

    ax.set_ylim(top=current_y + y_step)


def per_fly_trajectory_overlay(df: pd.DataFrame, first_frame: np.ndarray = None,
                                vials: int = 3, ax=None) -> plt.Axes:
    """Plot all tracked particles' paths overlaid on first video frame, color-coded by vial."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    if first_frame is not None:
        ax.imshow(first_frame, cmap='gray', alpha=0.5)

    if 'particle' not in df.columns:
        ax.set_title("No individual tracking data")
        return ax

    colors = get_color_palette(vials)
    for _pid, track in df.groupby('particle'):
        track = track.sort_values('frame')
        vial = int(track.vial.mode().iloc[0]) if 'vial' in track.columns else 1
        color = colors[(vial - 1) % len(colors)]
        ax.plot(track.x.values, track.y.values, color=color, alpha=0.4, linewidth=0.8)

    legend_elements = [Line2D([0], [0], color=colors[i], label=f'Vial {i+1}')
                       for i in range(min(vials, len(colors)))]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize='small')
    ax.set_title('Per-Fly Trajectory Overlay')
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    return ax


def per_fly_metrics_heatmap(metrics_df: pd.DataFrame, ax=None) -> plt.Axes:
    """Heatmap of fly_id x metric using imshow."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    numeric_cols = [c for c in metrics_df.columns
                    if c not in ('particle', 'vial') and metrics_df[c].dtype in ['float64', 'int64', 'float32']]
    if not numeric_cols:
        ax.set_title("No numeric metrics")
        return ax

    data = metrics_df[numeric_cols].values
    # Normalize each column to 0-1 for display
    with np.errstate(invalid='ignore'):
        mins = np.nanmin(data, axis=0)
        maxs = np.nanmax(data, axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1
        normed = (data - mins) / ranges

    im = ax.imshow(normed, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=8)
    fly_labels = metrics_df['particle'].values if 'particle' in metrics_df.columns else range(len(metrics_df))
    ax.set_yticks(range(len(fly_labels)))
    ax.set_yticklabels(fly_labels, fontsize=7)
    ax.set_ylabel('Fly ID')
    ax.set_title('Per-Fly Metrics Heatmap')
    plt.colorbar(im, ax=ax, label='Normalized value')
    return ax


def batch_comparison(results: dict, ylabel: str = 'Mean Climbing Speed',
                     title: str = 'Batch Comparison', ax=None) -> plt.Axes:
    """Grouped bar chart comparing mean speed across multiple videos/genotypes.

    Args:
        results: {video_or_genotype_name: array_of_slopes}
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    colors = get_color_palette(len(results))
    names = list(results.keys())
    means = [np.nanmean(np.asarray(v, dtype=float)) for v in results.values()]
    sems = [np.nanstd(np.asarray(v, dtype=float)) / np.sqrt(len(v)) if len(v) > 1 else 0
            for v in results.values()]

    x = np.arange(len(names))
    ax.bar(x, means, yerr=sems, color=colors, alpha=0.7, capsize=4,
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


def save_figure(fig, path: str, formats: list = None):
    """Save figure in multiple formats.

    Supported formats: 'png', 'svg', 'eps', 'pdf'.
    """
    if formats is None:
        formats = ['png']

    for fmt in formats:
        out_path = path.rsplit('.', 1)[0] + '.' + fmt
        fig.savefig(out_path, format=fmt, dpi=300, bbox_inches='tight',
                    transparent=(fmt in ('pdf', 'svg')))
        logger.info(f'Saved: {out_path}')
