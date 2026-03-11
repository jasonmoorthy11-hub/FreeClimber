"""Tests for output/figures.py — figure generation."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from output.figures import (
    get_color_palette, setup_publication_style,
    bar_chart_with_points, box_swarm_plot, trajectory_plot,
    speed_distribution, per_fly_trajectory_overlay,
    per_fly_metrics_heatmap, batch_comparison, save_figure,
)


@pytest.fixture
def groups():
    np.random.seed(42)
    return {
        'wildtype': np.random.normal(0.5, 0.1, 15),
        'mutant_a': np.random.normal(0.3, 0.12, 12),
        'mutant_b': np.random.normal(0.4, 0.08, 14),
    }


@pytest.fixture
def metrics_df():
    return pd.DataFrame({
        'particle': [0, 1, 2, 3, 4],
        'vial': [1, 1, 2, 2, 3],
        'n_frames': [50, 45, 48, 42, 50],
        'climbing_speed': [0.45, 0.32, 0.51, 0.28, 0.39],
        'start_latency': [3, 5, 2, 8, 4],
        'max_height': [120.5, 98.3, 135.2, 85.1, 110.7],
        'path_straightness': [0.85, 0.72, 0.91, 0.65, 0.78],
        'hesitation_count': [1, 3, 0, 4, 2],
        'horizontal_drift': [5.2, 8.1, 3.4, 12.5, 6.7],
        'track_completeness': [1.0, 0.9, 0.96, 0.84, 1.0],
        'mean_speed': [2.1, 1.8, 2.4, 1.5, 2.0],
        'auc': [45.2, 32.1, 52.8, 25.4, 40.1],
    })


class TestColorPalette:
    def test_n_zero(self):
        assert get_color_palette(0) == []

    def test_n_one(self):
        colors = get_color_palette(1)
        assert len(colors) == 1

    def test_n_within_okabe(self):
        colors = get_color_palette(8)
        assert len(colors) == 8
        assert all(isinstance(c, str) for c in colors)

    def test_n_exceeds_okabe(self):
        colors = get_color_palette(15)
        assert len(colors) == 15


class TestBarChart:
    def test_returns_axes(self, groups):
        ax = bar_chart_with_points(groups, ylabel='Speed')
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    def test_with_significance(self, groups):
        sig = {('wildtype', 'mutant_a'): '***', ('wildtype', 'mutant_b'): '*'}
        ax = bar_chart_with_points(groups, significance=sig)
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    def test_with_provided_axes(self, groups):
        fig, ax = plt.subplots()
        result = bar_chart_with_points(groups, ax=ax)
        assert result is ax
        plt.close('all')


class TestBoxSwarm:
    def test_returns_axes(self, groups):
        ax = box_swarm_plot(groups)
        assert isinstance(ax, plt.Axes)
        plt.close('all')


class TestTrajectory:
    def test_returns_axes(self, sample_tracking_df):
        ax = trajectory_plot(sample_tracking_df, vials=3)
        assert isinstance(ax, plt.Axes)
        plt.close('all')


class TestSpeedDistribution:
    def test_returns_axes(self, groups):
        ax = speed_distribution(groups)
        assert isinstance(ax, plt.Axes)
        plt.close('all')


class TestPerFlyTrajectoryOverlay:
    def test_returns_axes(self, sample_tracking_df):
        ax = per_fly_trajectory_overlay(sample_tracking_df, vials=3)
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    def test_no_particle_column(self):
        df = pd.DataFrame({'x': [1], 'y': [2], 'frame': [0]})
        ax = per_fly_trajectory_overlay(df, vials=1)
        assert isinstance(ax, plt.Axes)
        plt.close('all')


class TestHeatmap:
    def test_returns_axes(self, metrics_df):
        ax = per_fly_metrics_heatmap(metrics_df)
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    def test_no_numeric_cols(self):
        df = pd.DataFrame({'particle': [0, 1], 'vial': [1, 2]})
        ax = per_fly_metrics_heatmap(df)
        assert isinstance(ax, plt.Axes)
        plt.close('all')


class TestBatchComparison:
    def test_returns_axes(self, groups):
        ax = batch_comparison(groups)
        assert isinstance(ax, plt.Axes)
        plt.close('all')


class TestSaveFigure:
    def test_save_png(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        out = str(tmp_path / 'test.png')
        save_figure(fig, out, formats=['png'])
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        plt.close('all')

    def test_save_svg(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        out = str(tmp_path / 'test.svg')
        save_figure(fig, out, formats=['svg'])
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        plt.close('all')

    def test_save_pdf(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        out = str(tmp_path / 'test.pdf')
        save_figure(fig, out, formats=['pdf'])
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        plt.close('all')

    def test_save_multiple_formats(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        out = str(tmp_path / 'test.png')
        save_figure(fig, out, formats=['png', 'svg'])
        assert os.path.exists(str(tmp_path / 'test.png'))
        assert os.path.exists(str(tmp_path / 'test.svg'))
        plt.close('all')


class TestPublicationStyle:
    def test_sets_rcparams(self):
        setup_publication_style()
        assert plt.rcParams['figure.dpi'] == 300
        assert plt.rcParams['axes.spines.top'] is False
