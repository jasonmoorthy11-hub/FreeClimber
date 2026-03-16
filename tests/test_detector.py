"""Tests for detector.py — scientific accuracy and core logic.

Uses synthetic data to test Phase 1 fixes without needing a real video.
"""

import numpy as np
import pandas as pd
import pytest


class MockDetector:
    """Minimal mock with just enough state to test individual methods."""

    def __init__(self, **kwargs):
        defaults = {
            'debug': False, 'vials': 3, 'diameter': 7, 'frame_rate': 30,
            'crop_0': 0, 'crop_n': 100, 'window': 50,
            'blank_0': 0, 'blank_n': 100, 'check_frame': 50,
            'h': 380, 'w': 1112,
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


def _get_detector_class():
    from detector import detector
    return detector


class TestFindThreshold:
    """1A: find_threshold must return a signal VALUE, not a bin index."""

    def test_returns_signal_value_not_index(self):
        det_cls = _get_detector_class()
        mock = MockDetector()
        # Bimodal distribution: peak near 50 and peak near 150
        signal = np.concatenate([
            np.random.normal(50, 10, 200),
            np.random.normal(150, 10, 200),
        ])
        threshold = det_cls.find_threshold(mock, signal, bins=40)
        # Threshold should be in the signal range, not a small bin index
        assert threshold > 20, f"Threshold {threshold} looks like a bin index, not a signal value"
        assert threshold < 200, f"Threshold {threshold} out of expected signal range"

    def test_single_peak_returns_value(self):
        det_cls = _get_detector_class()
        mock = MockDetector()
        signal = np.random.normal(100, 15, 500)
        threshold = det_cls.find_threshold(mock, signal, bins=40)
        # Should be near the peak
        assert 50 < threshold < 150

    def test_no_peaks_returns_median(self):
        det_cls = _get_detector_class()
        mock = MockDetector()
        # Uniform distribution — unlikely to have prominent peaks
        signal = np.linspace(0, 100, 10)
        threshold = det_cls.find_threshold(mock, signal, bins=3)
        # With very few bins, may or may not find peaks, but should return a sensible value
        assert 0 <= threshold <= 100


class TestCheckVariableFormats:
    """1C, 1D: diameter oddness and window int type."""

    def test_even_diameter_gets_corrected(self):
        mock = MockDetector(diameter=6)
        det_cls = _get_detector_class()
        det_cls.check_variable_formats(mock)
        assert mock.diameter == 7
        assert mock.diameter % 2 == 1

    def test_odd_diameter_unchanged(self):
        mock = MockDetector(diameter=9)
        det_cls = _get_detector_class()
        det_cls.check_variable_formats(mock)
        assert mock.diameter == 9

    def test_window_is_int_after_resize(self):
        mock = MockDetector(crop_0=0, crop_n=100, window=200)
        det_cls = _get_detector_class()
        det_cls.check_variable_formats(mock)
        assert isinstance(mock.window, int)
        assert mock.window == 80  # int(100 * 0.8)

    def test_negative_frame_rate_corrected(self):
        mock = MockDetector(frame_rate=-5)
        det_cls = _get_detector_class()
        det_cls.check_variable_formats(mock)
        assert mock.frame_rate == 1

    def test_blank_clamped_to_crop(self):
        mock = MockDetector(crop_0=10, crop_n=90, blank_0=5, blank_n=95)
        det_cls = _get_detector_class()
        det_cls.check_variable_formats(mock)
        assert mock.blank_0 == 10
        assert mock.blank_n == 90


class TestInvertY:
    """1E: invert_y should use self.h (ROI height), not data max."""

    def test_uses_roi_height(self):
        det_cls = _get_detector_class()
        mock = MockDetector(h=400)
        spots = pd.DataFrame({'y': [0, 100, 200, 400]})
        inv_y = det_cls.invert_y(mock, spots)
        expected = pd.Series([400, 300, 200, 0])
        pd.testing.assert_series_equal(inv_y, expected, check_names=False)

    def test_inversion_independent_of_data_max(self):
        det_cls = _get_detector_class()
        mock = MockDetector(h=500)
        spots = pd.DataFrame({'y': [10, 20]})
        inv_y = det_cls.invert_y(mock, spots)
        # With h=500, inversion is abs(y - 500), not abs(y - 20)
        assert inv_y.iloc[0] == 490
        assert inv_y.iloc[1] == 480


class TestSpecifyPathsDetails:
    """1F: file extension handling should work for any extension length."""

    def test_mp4_extension(self, tmp_path):
        det_cls = _get_detector_class()
        mock = MockDetector(
            path_project=None,
            naming_convention='geno_sex_day_rep',
            vial_id_vars=2,
            vial_color_map=lambda x: (x, x, x, 1),
        )
        import matplotlib.cm as cm
        mock.vial_color_map = cm.viridis
        video = tmp_path / "wt_m_1_1.mp4"
        video.touch()
        det_cls.specify_paths_details(mock, str(video))
        assert mock.name == "wt_m_1_1"

    def test_h264_extension(self, tmp_path):
        det_cls = _get_detector_class()
        mock = MockDetector(
            path_project=None,
            naming_convention='geno_sex_day_rep',
            vial_id_vars=2,
            vial_color_map=lambda x: (x, x, x, 1),
        )
        import matplotlib.cm as cm
        mock.vial_color_map = cm.viridis
        video = tmp_path / "wt_m_1_1.h264"
        video.touch()
        det_cls.specify_paths_details(mock, str(video))
        assert mock.name == "wt_m_1_1"

    def test_avi_extension(self, tmp_path):
        det_cls = _get_detector_class()
        mock = MockDetector(
            path_project=None,
            naming_convention='geno_sex_day_rep',
            vial_id_vars=2,
            vial_color_map=lambda x: (x, x, x, 1),
        )
        import matplotlib.cm as cm
        mock.vial_color_map = cm.viridis
        video = tmp_path / "wt_m_1_1.avi"
        video.touch()
        det_cls.specify_paths_details(mock, str(video))
        assert mock.name == "wt_m_1_1"


class TestCropAndGrayscale:
    """1G: x_max and y_max should map correctly to shape dimensions."""

    def test_defaults_match_video_dimensions(self):
        det_cls = _get_detector_class()
        mock = MockDetector(crop_0=0, crop_n=5)
        # shape: (frames, height, width, channels) = (10, 480, 640, 3)
        video = np.random.randint(0, 255, (10, 480, 640, 3), dtype=np.uint8)
        result = det_cls.crop_and_grayscale(mock, video)
        assert result.shape == (5, 480, 640)

    def test_crop_region(self):
        det_cls = _get_detector_class()
        mock = MockDetector(crop_0=2, crop_n=5)
        video = np.random.randint(0, 255, (10, 480, 640, 3), dtype=np.uint8)
        result = det_cls.crop_and_grayscale(mock, video, x=100, x_max=200, y=50, y_max=150)
        assert result.shape == (3, 100, 100)
