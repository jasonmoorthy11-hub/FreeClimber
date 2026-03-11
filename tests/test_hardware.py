"""Tests for hardware modules in mock mode (runs on Mac without Pi)."""

import os
import tempfile

import pytest

from scripts.hardware.motor import RINGMotor, DEFAULT_PUL_PIN, DEFAULT_DIR_PIN
from scripts.hardware.camera import RINGCamera
from scripts.workflow import RINGWorkflow


# --- Motor tests ---

class TestRINGMotor:
    def test_init_mock(self):
        m = RINGMotor(mock=True)
        assert m.mock is True
        assert m.pul_pin == DEFAULT_PUL_PIN
        assert m.dir_pin == DEFAULT_DIR_PIN
        assert m.current_position == 0
        assert m._h is None

    def test_init_auto_mock_on_mac(self):
        m = RINGMotor()
        assert m.mock is True

    def test_rotate(self):
        m = RINGMotor(mock=True)
        m.rotate(90, direction='cw')
        expected_steps = int(m.steps_per_rev * 90 / 360)
        assert m.current_position == expected_steps

    def test_rotate_ccw(self):
        m = RINGMotor(mock=True)
        m.rotate(90, direction='ccw')
        expected_steps = int(m.steps_per_rev * 90 / 360)
        assert m.current_position == -expected_steps

    def test_flip_180(self):
        m = RINGMotor(mock=True)
        m.flip_180()
        expected_steps = int(m.steps_per_rev * 180 / 360)
        assert m.current_position == expected_steps

    def test_tap_sequence(self):
        m = RINGMotor(mock=True)
        m.tap_sequence(n_taps=3, interval=0.01)
        # Taps are symmetric (cw then ccw), so position should return to 0
        assert m.current_position == 0

    def test_home(self):
        m = RINGMotor(mock=True)
        m.rotate(90, direction='cw')
        assert m.current_position != 0
        m.home()
        assert m.current_position == 0

    def test_home_already_at_zero(self):
        m = RINGMotor(mock=True)
        m.home()  # should be a no-op
        assert m.current_position == 0

    def test_cleanup(self):
        m = RINGMotor(mock=True)
        m.cleanup()  # should not raise
        assert m._h is None

    def test_custom_pins(self):
        m = RINGMotor(pul_pin=22, dir_pin=23, mock=True)
        assert m.pul_pin == 22
        assert m.dir_pin == 23

    def test_custom_steps_per_rev(self):
        m = RINGMotor(steps_per_rev=3200, mock=True)
        m.rotate(360, direction='cw')
        assert m.current_position == 3200


# --- Camera tests ---

class TestRINGCamera:
    def test_init_mock(self):
        c = RINGCamera(mock=True)
        assert c.mock is True
        assert c._camera is None

    def test_init_auto_mock_on_mac(self):
        c = RINGCamera()
        assert c.mock is True

    def test_get_info(self):
        c = RINGCamera(mock=True)
        info = c.get_info()
        assert info['model'] == 'Mock Camera'
        assert info['available'] is False
        assert 'resolution' in info

    def test_record_mock(self):
        c = RINGCamera(mock=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'test.mp4')
            c.record(duration=1, output_path=path)
            # Mock mode doesn't create a file, just logs

    def test_capture_still_mock(self):
        c = RINGCamera(mock=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'test.png')
            c.capture_still(output_path=path)

    def test_close(self):
        c = RINGCamera(mock=True)
        c.close()
        assert c._camera is None

    def test_close_idempotent(self):
        c = RINGCamera(mock=True)
        c.close()
        c.close()  # should not raise


# --- Workflow tests ---

class TestRINGWorkflow:
    def test_init_defaults(self):
        wf = RINGWorkflow()
        assert wf.motor is None
        assert wf.camera is None
        assert wf.config == {}
        assert wf.session_dir is None
        assert wf.results == []
        assert os.path.isdir(wf.output_dir)

    def test_init_custom_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'data')
            wf = RINGWorkflow(output_dir=out)
            assert wf.output_dir == out
            assert os.path.isdir(out)

    def test_create_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            wf = RINGWorkflow(output_dir=tmp)
            session = wf._create_session()
            assert os.path.isdir(session)
            assert session.startswith(tmp)
            assert 'session_' in session

    def test_preflight_no_hardware(self):
        wf = RINGWorkflow()
        status = wf.preflight_check()
        assert status['motor'] is False
        assert status['camera'] is False
        assert status['disk_space'] is True
        assert status['config'] is False

    def test_preflight_with_mock_hardware(self):
        m = RINGMotor(mock=True)
        c = RINGCamera(mock=True)
        wf = RINGWorkflow(motor=m, camera=c, config={'test': True})
        status = wf.preflight_check()
        assert status['motor'] is True
        assert status['camera'] is True
        assert status['config'] is True

    def test_tap_no_motor(self):
        wf = RINGWorkflow()
        wf.tap(n_taps=3)  # should not raise, just log warning

    def test_tap_with_mock_motor(self):
        m = RINGMotor(mock=True)
        wf = RINGWorkflow(motor=m)
        wf.tap(n_taps=2, interval=0.01)
        assert m.current_position == 0

    def test_record_no_camera(self):
        wf = RINGWorkflow()
        result = wf.record(duration=1)
        assert result is None

    def test_record_with_mock_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = RINGCamera(mock=True)
            wf = RINGWorkflow(camera=c, output_dir=tmp)
            path = wf.record(duration=1)
            assert path is not None
            assert path.endswith('.mp4')

    def test_settle_short(self):
        wf = RINGWorkflow()
        progress = []
        wf.settle(duration=0.1, callback=lambda r: progress.append(r))
        # Should complete quickly without error

    def test_analyze_missing_video(self):
        wf = RINGWorkflow()
        result = wf.analyze('/nonexistent/video.mp4')
        assert result == {}
