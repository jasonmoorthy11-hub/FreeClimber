"""Shared test fixtures for FreeClimber tests."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def example_config_path():
    """Path to the example .cfg file."""
    return os.path.join(os.path.dirname(__file__), '..', 'example', 'example.cfg')


@pytest.fixture
def example_slopes_path():
    """Path to the example .slopes.csv file."""
    return os.path.join(os.path.dirname(__file__), '..', 'example', 'w1118_m_2_1.slopes.csv')


@pytest.fixture
def sample_config():
    """A valid config dict for testing."""
    return {
        'x': 100, 'y': 136, 'w': 1112, 'h': 380,
        'check_frame': 100, 'blank_0': 0, 'blank_n': 145,
        'crop_0': 0, 'crop_n': 145,
        'threshold': 'auto', 'diameter': 7, 'minmass': 100,
        'maxsize': 11, 'ecc_low': 0.05, 'ecc_high': 0.58,
        'vials': 3, 'window': 50, 'pixel_to_cm': 46.0,
        'frame_rate': 29, 'vial_id_vars': 2,
        'outlier_TB': 1.0, 'outlier_LR': 3.0,
        'naming_convention': 'geno_sex_day_rep',
        'path_project': './example',
        'file_suffix': 'h264',
        'convert_to_cm_sec': True,
        'trim_outliers': True,
    }


@pytest.fixture
def tmp_cfg_file(sample_config):
    """Create a temporary .cfg file."""
    from config import save_config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
        save_config(f.name, sample_config)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_slopes_df():
    """Sample slopes DataFrame for testing."""
    return pd.DataFrame({
        'vial_ID': ['wt_m_1', 'wt_m_2', 'wt_m_3', 'wt_m_all'],
        'first_frame': [10, 12, 8, 10],
        'last_frame': [60, 62, 58, 60],
        'slope': [0.45, 0.52, 0.38, 0.44],
        'intercept': [5.2, 4.8, 6.1, 5.3],
        'r_value': [0.95, 0.97, 0.89, 0.93],
        'p_value': [0.001, 0.0005, 0.01, 0.002],
        'std_err': [0.02, 0.015, 0.04, 0.025],
        'quality': ['good', 'good', 'low_r', 'good'],
        'geno': ['wt', 'wt', 'wt', 'wt'],
    })


@pytest.fixture
def sample_tracking_df():
    """Sample per-frame tracking DataFrame with particles."""
    np.random.seed(42)
    rows = []
    for particle in range(5):
        for frame in range(50):
            rows.append({
                'particle': particle,
                'frame': frame,
                'x': 100 + particle * 200 + np.random.normal(0, 5),
                'y': frame * 3 + np.random.normal(0, 2),
                'vial': particle % 3 + 1,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def three_groups():
    """Three sample groups for statistical testing."""
    np.random.seed(42)
    return {
        'wildtype': np.random.normal(0.5, 0.1, 15),
        'mutant_a': np.random.normal(0.3, 0.12, 12),
        'mutant_b': np.random.normal(0.4, 0.08, 14),
    }
