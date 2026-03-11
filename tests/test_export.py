"""Tests for output/export.py — data export formats."""

import os
import sys
import tempfile
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from output.export import (
    export_slopes_csv, export_tidy_csv, export_prism_csv, export_per_fly_tracks,
)


class TestSlopesCSV:
    def test_round_trip(self, sample_slopes_df):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            export_slopes_csv(sample_slopes_df, path)
            loaded = pd.read_csv(path)
            assert list(loaded.columns) == list(sample_slopes_df.columns)
            assert len(loaded) == len(sample_slopes_df)
        finally:
            os.unlink(path)


class TestTidyCSV:
    def test_long_format(self, sample_slopes_df):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            export_tidy_csv(sample_slopes_df, path, experiment='test_exp')
            loaded = pd.read_csv(path)
            assert 'metric_name' in loaded.columns
            assert 'metric_value' in loaded.columns
            assert 'experiment' in loaded.columns
        finally:
            os.unlink(path)


class TestPrismCSV:
    def test_grouped_format(self, sample_slopes_df):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            export_prism_csv(sample_slopes_df, path, group_col='geno')
            loaded = pd.read_csv(path)
            assert 'wt' in loaded.columns
        finally:
            os.unlink(path)


class TestPerFlyTracks:
    def test_with_particles(self, sample_tracking_df):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            export_per_fly_tracks(sample_tracking_df, path)
            loaded = pd.read_csv(path)
            assert 'fly_id' in loaded.columns
        finally:
            os.unlink(path)

    def test_without_particles(self):
        df = pd.DataFrame({'x': [1], 'y': [2], 'frame': [0]})
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            export_per_fly_tracks(df, path)
            assert not os.path.exists(path) or os.path.getsize(path) == 0
        finally:
            if os.path.exists(path):
                os.unlink(path)
