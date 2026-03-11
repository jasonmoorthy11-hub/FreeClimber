"""Tests for controller profile management."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from gui.controller import AnalysisController


@pytest.fixture
def ctrl(tmp_path):
    c = AnalysisController()
    c.PROFILES_DIR = str(tmp_path / 'profiles')
    return c


class TestProfileRoundTrip:
    def test_save_and_load(self, ctrl, sample_config):
        ctrl.save_profile('test_profile', sample_config)
        loaded = ctrl.load_profile('test_profile')
        for key in ['diameter', 'vials', 'frame_rate']:
            assert loaded[key] == sample_config[key]

    def test_list_empty(self, ctrl):
        assert ctrl.list_profiles() == []

    def test_list_populated(self, ctrl, sample_config):
        ctrl.save_profile('alpha', sample_config)
        ctrl.save_profile('beta', sample_config)
        profiles = ctrl.list_profiles()
        assert profiles == ['alpha', 'beta']

    def test_delete(self, ctrl, sample_config):
        ctrl.save_profile('to_delete', sample_config)
        assert 'to_delete' in ctrl.list_profiles()
        ctrl.delete_profile('to_delete')
        assert 'to_delete' not in ctrl.list_profiles()


class TestProfileValidation:
    def test_rejects_path_traversal(self, ctrl, sample_config):
        with pytest.raises(ValueError):
            ctrl.save_profile('../../etc/passwd', sample_config)

    def test_rejects_slash(self, ctrl, sample_config):
        with pytest.raises(ValueError):
            ctrl.save_profile('foo/bar', sample_config)

    def test_rejects_backslash(self, ctrl, sample_config):
        with pytest.raises(ValueError):
            ctrl.save_profile('foo\\bar', sample_config)

    def test_rejects_empty(self, ctrl, sample_config):
        with pytest.raises(ValueError):
            ctrl.save_profile('', sample_config)

    def test_rejects_dotdot(self, ctrl, sample_config):
        with pytest.raises(ValueError):
            ctrl.load_profile('..')
