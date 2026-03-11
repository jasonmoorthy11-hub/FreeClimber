"""Automated RING assay workflow orchestrator.

Coordinates motor, camera, and FreeClimber analysis for
fully automated RING assay experiments.
"""

import os
import sys
import time
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


class RINGWorkflow:
    """Orchestrates the full RING assay: settle -> tap -> record -> analyze -> report.

    Works on Pi with real hardware, or in mock mode for testing.
    """

    def __init__(self, motor=None, camera=None, config: dict = None,
                 output_dir: str = None):
        self.motor = motor
        self.camera = camera
        self.config = config or {}
        self.output_dir = output_dir or os.path.expanduser('~/ring_assay_data')
        os.makedirs(self.output_dir, exist_ok=True)

        self.session_dir = None
        self.results = []

    def _create_session(self) -> str:
        """Create a new session directory."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(self.output_dir, f'session_{timestamp}')
        os.makedirs(self.session_dir, exist_ok=True)
        logger.info(f"Session created: {self.session_dir}")
        return self.session_dir

    def preflight_check(self) -> dict:
        """Verify all components are ready before starting assay.

        Returns dict with status of each component.
        """
        status = {
            'motor': False,
            'camera': False,
            'disk_space': False,
            'config': False,
        }

        # Motor check
        if self.motor is not None:
            status['motor'] = True
            logger.info("Motor: OK")
        else:
            logger.warning("Motor: not available")

        # Camera check
        if self.camera is not None:
            info = self.camera.get_info()
            status['camera'] = info.get('available', False) or self.camera.mock
            logger.info(f"Camera: {'OK' if status['camera'] else 'FAIL'} ({info.get('model', 'unknown')})")
        else:
            logger.warning("Camera: not available")

        # Disk space check (need at least 1GB)
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.output_dir)
            gb_free = free / (1024**3)
            status['disk_space'] = gb_free > 1.0
            logger.info(f"Disk: {gb_free:.1f} GB free {'(OK)' if status['disk_space'] else '(LOW)'}")
        except Exception:
            status['disk_space'] = True  # assume OK if can't check

        # Config check
        status['config'] = bool(self.config)
        logger.info(f"Config: {'OK' if status['config'] else 'missing'}")

        return status

    def settle(self, duration: float = 60, callback=None):
        """Wait for flies to settle (countdown display).

        Args:
            duration: seconds to wait
            callback: optional function(remaining_seconds) for progress updates
        """
        logger.info(f"Settling: waiting {duration}s for flies to settle")
        start = time.time()
        while time.time() - start < duration:
            remaining = int(duration - (time.time() - start))
            if callback:
                callback(remaining)
            time.sleep(1)
        logger.info("Settle period complete")

    def tap(self, n_taps: int = 3, interval: float = 0.5):
        """Execute tapping protocol to dislodge flies."""
        if self.motor is None:
            logger.warning("No motor available; skipping tap")
            return
        logger.info(f"Tapping: {n_taps} taps")
        self.motor.tap_sequence(n_taps=n_taps, interval=interval)

    def record(self, duration: float = 10, resolution: tuple = (1920, 1080),
               fps: int = 30) -> str:
        """Record climbing video.

        Returns path to recorded video.
        """
        if self.camera is None:
            logger.warning("No camera available; skipping recording")
            return None

        if self.session_dir is None:
            self._create_session()

        timestamp = datetime.now().strftime('%H%M%S')
        video_path = os.path.join(self.session_dir, f'climbing_{timestamp}.mp4')
        self.camera.record(duration=duration, output_path=video_path,
                           resolution=resolution, fps=fps)
        return video_path

    def analyze(self, video_path: str, config: dict = None) -> dict:
        """Run FreeClimber analysis on recorded video.

        Returns results dict with slopes, quality scores, etc.
        """
        if video_path is None or not os.path.exists(video_path):
            logger.error(f"Video not found: {video_path}")
            return {}

        cfg = config or self.config
        logger.info(f"Analyzing: {video_path}")

        try:
            from config import save_config
            cfg_path = video_path.rsplit('.', 1)[0] + '.cfg'
            save_config(cfg_path, cfg, video_file=video_path)

            from detector import detector
            d = detector(video_file=video_path, config_file=cfg_path)
            d.step_1(gui=False)
            d.step_2()
            d.step_3(gui=False)
            d.step_4()
            d.step_5()
            d.step_6(gui=False)
            d.step_7()

            return {
                'video_path': video_path,
                'slopes_path': d.path_slope,
                'slopes_df': d.df_slopes,
                'success': True,
            }
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {'video_path': video_path, 'success': False, 'error': str(e)}

    def run_assay(self, settle_time: float = 60, n_taps: int = 3,
                  record_duration: float = 10, progress_callback=None) -> dict:
        """Run a complete RING assay cycle.

        1. Settle -> 2. Tap -> 3. Record -> 4. Analyze -> 5. Return results
        """
        self._create_session()
        logger.info("Starting RING assay")

        # 1. Settle
        self.settle(settle_time, callback=progress_callback)

        # 2. Tap
        self.tap(n_taps)

        # 3. Record
        video_path = self.record(record_duration)

        # 4. Analyze
        result = self.analyze(video_path)
        self.results.append(result)

        logger.info(f"Assay complete. Success: {result.get('success', False)}")
        return result

    def repeat(self, n_trials: int = 3, rest_interval: float = 60, **kwargs) -> list:
        """Run multiple assay trials with rest periods between them."""
        all_results = []
        for i in range(n_trials):
            logger.info(f"Trial {i+1}/{n_trials}")
            result = self.run_assay(**kwargs)
            all_results.append(result)

            if i < n_trials - 1:
                logger.info(f"Resting {rest_interval}s before next trial")
                time.sleep(rest_interval)

        return all_results
