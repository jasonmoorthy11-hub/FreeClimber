"""Framework-agnostic controller bridging the GUI and FreeClimber detector engine.

All analysis calls go through this controller so that the GUI never touches
detector internals directly.  Threading is handled here — the GUI just provides
a progress callback.
"""

import os
import sys
import threading
import queue
import logging

# Bootstrap sys.path so bare imports work from any launch location
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AnalysisController:
    """Mediator between any GUI front-end and the detector engine."""

    def __init__(self):
        self.detector = None
        self.video_path = None
        self.config = {}
        self._cancel = threading.Event()
        self._worker = None
        self.slopes_df = None
        self.positions_df = None

    # ------------------------------------------------------------------
    # Video loading
    # ------------------------------------------------------------------
    def load_video(self, path: str, params: dict | None = None) -> dict:
        """Load a video file and return metadata.

        Returns dict with keys: n_frames, width, height, fps, first_frame, last_frame
        """
        import cv2
        from detector import detector

        self.video_path = path
        variables = self._params_to_variables(params or self.config)
        self.detector = detector(path, gui=True, variables=variables)

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        meta = {
            'n_frames': self.detector.n_frames,
            'width': self.detector.width,
            'height': self.detector.height,
            'fps': fps if fps > 0 else 30,
            'first_frame': self.detector.image_stack[0] if len(self.detector.image_stack) > 0 else None,
            'last_frame': self.detector.image_stack[-1] if len(self.detector.image_stack) > 1 else None,
        }
        return meta

    # ------------------------------------------------------------------
    # Parameter testing (runs steps 1-7, shows diagnostics)
    # ------------------------------------------------------------------
    def test_parameters(self, params: dict, axes=None) -> dict:
        """Run the detector pipeline on current video with given params.

        Args:
            params: parameter dict
            axes: optional list of 6 matplotlib axes for diagnostic plots

        Returns:
            dict with 'slopes_df', 'positions_df', 'paths'
        """
        if self.detector is None:
            raise RuntimeError("Load a video first")

        variables = self._params_to_variables(params)
        self.config = params.copy()
        self.detector.parameter_testing(variables, axes)

        result = {'paths': {}}
        # Try to load outputs
        try:
            base = getattr(self.detector, 'path_noext', None)
            if base is None:
                proj = getattr(self.detector, 'path_project', None)
                name = getattr(self.detector, 'name', None)
                if proj and name:
                    base = os.path.join(str(proj), str(name))

            if base:
                slopes_path = base + '.slopes.csv'
                if os.path.exists(slopes_path):
                    self.slopes_df = pd.read_csv(slopes_path)
                    result['slopes_df'] = self.slopes_df
                    result['paths']['slopes'] = slopes_path

                for suffix, key in [('.raw.csv', 'raw'), ('.filter.csv', 'filter')]:
                    p = base + suffix
                    if os.path.exists(p):
                        result['paths'][key] = p
                        if key == 'filter':
                            self.positions_df = pd.read_csv(p)
                            result['positions_df'] = self.positions_df
        except Exception as e:
            logger.warning("Could not load outputs: %s", e)

        return result

    # ------------------------------------------------------------------
    # Full analysis in background thread
    # ------------------------------------------------------------------
    def run_analysis(self, params: dict, progress_callback=None, done_callback=None):
        """Run full analysis in a background thread.

        progress_callback(step, total, message) is called from the worker thread.
        done_callback(result_or_exception) is called when complete.
        """
        self._cancel.clear()

        def _worker():
            try:
                result = self.test_parameters(params)
                if done_callback:
                    done_callback(result)
            except Exception as e:
                logger.error("Analysis failed: %s", e)
                if done_callback:
                    done_callback(e)

        self._worker = threading.Thread(target=_worker, daemon=True)
        self._worker.start()

    def cancel_analysis(self):
        self._cancel.set()

    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def get_slopes(self) -> pd.DataFrame | None:
        return self.slopes_df

    def get_positions(self) -> pd.DataFrame | None:
        return self.positions_df

    def export_results(self, fmt: str, path: str):
        """Export results in the given format.

        fmt: 'csv', 'tidy', 'prism', 'excel', 'tracks'
        """
        if self.slopes_df is None:
            raise RuntimeError("No results to export — run analysis first")

        from output.export import (
            export_slopes_csv, export_tidy_csv, export_prism_csv,
            export_per_fly_tracks,
        )

        if fmt == 'csv':
            export_slopes_csv(self.slopes_df, path)
        elif fmt == 'tidy':
            export_tidy_csv(self.slopes_df, path)
        elif fmt == 'prism':
            export_prism_csv(self.slopes_df, path, group_col='geno')
        elif fmt == 'tracks' and self.positions_df is not None:
            export_per_fly_tracks(self.positions_df, path)
        else:
            raise ValueError(f"Unknown format: {fmt}")

    # ------------------------------------------------------------------
    # Config save/load
    # ------------------------------------------------------------------
    def save_config(self, path: str, params: dict):
        from config import save_config
        save_config(path, params)

    def load_config(self, path: str) -> dict:
        from config import load_config
        params = load_config(path)
        self.config = params
        return params

    # ------------------------------------------------------------------
    # Config profiles
    # ------------------------------------------------------------------
    PROFILES_DIR = os.path.expanduser('~/.freeclimber/profiles')

    def save_profile(self, name: str, params: dict):
        """Save params as a named profile."""
        from config import save_config
        os.makedirs(self.PROFILES_DIR, exist_ok=True)
        save_config(os.path.join(self.PROFILES_DIR, f'{name}.cfg'), params)

    def load_profile(self, name: str) -> dict:
        """Load a named profile."""
        from config import load_config
        path = os.path.join(self.PROFILES_DIR, f'{name}.cfg')
        params = load_config(path)
        self.config = params
        return params

    def list_profiles(self) -> list[str]:
        """Return list of saved profile names."""
        if not os.path.isdir(self.PROFILES_DIR):
            return []
        return sorted(
            f[:-4] for f in os.listdir(self.PROFILES_DIR) if f.endswith('.cfg')
        )

    def delete_profile(self, name: str):
        path = os.path.join(self.PROFILES_DIR, f'{name}.cfg')
        if os.path.exists(path):
            os.remove(path)

    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------
    def run_batch(self, video_paths: list[str], params: dict,
                  progress_callback=None) -> pd.DataFrame:
        """Run pipeline on multiple videos, return combined results.

        progress_callback(index, total, path, status) is called per video.
        Returns combined slopes DataFrame with a 'source_video' column.
        """
        from config import save_config
        from detector import detector

        all_results = []
        statuses = []

        for i, video_path in enumerate(video_paths):
            if progress_callback:
                progress_callback(i, len(video_paths), video_path, 'processing')
            try:
                cfg_path = video_path.rsplit('.', 1)[0] + '.cfg'
                save_config(cfg_path, params, video_file=video_path)

                d = detector(video_file=video_path, config_file=cfg_path)
                d.step_1(gui=False)
                d.step_2()
                d.step_3(gui=False)
                d.step_4()
                d.step_5()
                d.step_6(gui=False)
                d.step_7()

                df = d.df_slopes.copy()
                df['source_video'] = os.path.basename(video_path)
                all_results.append(df)
                statuses.append({'path': video_path, 'success': True, 'error': None})
            except Exception as e:
                logger.error("Batch failed for %s: %s", video_path, e)
                statuses.append({'path': video_path, 'success': False, 'error': str(e)})

            if progress_callback:
                status = 'done' if statuses[-1]['success'] else 'error'
                progress_callback(i + 1, len(video_paths), video_path, status)

        combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        self.slopes_df = combined
        self._batch_statuses = statuses
        return combined

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _params_to_variables(params: dict) -> list[str]:
        """Convert a params dict to the variable list format detector expects."""
        variables = []
        for key, val in params.items():
            if isinstance(val, str):
                variables.append(f'{key}="{val}"')
            elif isinstance(val, bool):
                variables.append(f'{key}={val}')
            else:
                variables.append(f'{key}={val}')
        return variables
