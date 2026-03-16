"""Framework-agnostic controller bridging the GUI and FreeClimber detector engine.

All analysis calls go through this controller so that the GUI never touches
detector internals directly.  Threading is handled here — the GUI just provides
a progress callback.
"""
from __future__ import annotations

import logging
import os
import queue
import sys
import threading

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
        self.quality_scores = None
        self.population_metrics = None
        self.per_fly_metrics = None
        self.climbing_index = None
        self.first_frame = None

    # ------------------------------------------------------------------
    # Video loading
    # ------------------------------------------------------------------
    def load_video(self, path: str, params: dict = None) -> dict:
        """Load a video file and return metadata.

        Returns dict with keys: n_frames, width, height, fps, first_frame, last_frame
        """
        from detector import detector

        self.video_path = path
        variables = self._params_to_variables(params or self.config)
        self.detector = detector(path, gui=True, variables=variables)

        fps = getattr(self.detector, 'frame_rate', 30)

        first = self.detector.image_stack[0] if len(self.detector.image_stack) > 0 else None
        last = self.detector.image_stack[-1] if len(self.detector.image_stack) > 1 else None
        self.first_frame = first

        meta = {
            'n_frames': self.detector.n_frames,
            'width': self.detector.width,
            'height': self.detector.height,
            'fps': fps if fps > 0 else 30,
            'first_frame': first,
            'last_frame': last,
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
            dict with 'slopes_df', 'positions_df', 'paths', plus
            'quality', 'population_metrics', 'per_fly_metrics', 'climbing_index',
            'has_individual_tracking', 'first_frame'
        """
        if self.detector is None:
            raise RuntimeError("Load a video first")

        variables = self._params_to_variables(params)
        self.config = params.copy()
        self.detector.parameter_testing(variables, axes)

        result = {'paths': {}}
        # Try to load outputs — first from in-memory DataFrame, then from disk
        try:
            # The detector stores slopes in self.df_slopes after step_7
            if hasattr(self.detector, 'df_slopes') and self.detector.df_slopes is not None:
                self.slopes_df = self.detector.df_slopes
                result['slopes_df'] = self.slopes_df

            # The detector stores filtered positions in self.df_filtered
            if hasattr(self.detector, 'df_filtered') and self.detector.df_filtered is not None:
                self.positions_df = self.detector.df_filtered
                result['positions_df'] = self.positions_df

            # Also record file paths if they exist
            base = getattr(self.detector, 'name_nosuffix', None)
            if base:
                for suffix, key in [('.slopes.csv', 'slopes'), ('.raw.csv', 'raw'), ('.filtered.csv', 'filter')]:
                    p = base + suffix
                    if os.path.exists(p):
                        result['paths'][key] = p
                        # Fall back to disk if not already loaded
                        if key == 'slopes' and 'slopes_df' not in result:
                            self.slopes_df = pd.read_csv(p)
                            result['slopes_df'] = self.slopes_df
                        elif key == 'filter' and 'positions_df' not in result:
                            self.positions_df = pd.read_csv(p)
                            result['positions_df'] = self.positions_df
        except Exception as e:
            logger.warning("Could not load outputs: %s", e)

        # --- Post-analysis: wire up quality, metrics, per-fly ---
        self._compute_post_analysis(params, result)

        # --- Auto-save to database ---
        self._save_to_database(params, result)

        return result

    def _compute_post_analysis(self, params: dict, result: dict):
        """Compute quality scores, population metrics, per-fly metrics after pipeline."""
        slopes_df = result.get('slopes_df')
        positions_df = result.get('positions_df')

        # Quality scoring
        if slopes_df is not None:
            try:
                from analysis.quality import score_video
                self.quality_scores = score_video(slopes_df, positions_df)
                result['quality'] = self.quality_scores
            except Exception as e:
                logger.warning("Quality scoring failed: %s", e)

        # Population metrics
        if positions_df is not None and slopes_df is not None:
            try:
                from analysis.metrics import compute_population_metrics
                self.population_metrics = compute_population_metrics(
                    positions_df, slopes_df,
                    frame_rate=params.get('frame_rate', 30),
                    pixel_to_cm=params.get('pixel_to_cm', 1.0),
                    convert_to_cm_sec=params.get('convert_to_cm_sec', False),
                )
                result['population_metrics'] = self.population_metrics
            except Exception as e:
                logger.warning("Population metrics failed: %s", e)

        # Individual tracking data
        has_tracking = getattr(self.detector, 'has_individual_tracking', False)
        result['has_individual_tracking'] = has_tracking
        result['first_frame'] = self.first_frame

        # Raw tracking data (with particle column) for per-fly plots
        raw_df = getattr(self.detector, 'df_big', None)
        if raw_df is not None and 'particle' in raw_df.columns:
            result['raw_tracking_df'] = raw_df

        if has_tracking and positions_df is not None:
            # Per-fly metrics
            try:
                from analysis.metrics import compute_per_fly_metrics
                raw = getattr(self.detector, 'df_big', positions_df)
                if 'particle' in raw.columns:
                    self.per_fly_metrics = compute_per_fly_metrics(
                        raw,
                        frame_rate=params.get('frame_rate', 30),
                        pixel_to_cm=params.get('pixel_to_cm', 1.0),
                        convert_to_cm_sec=params.get('convert_to_cm_sec', False),
                    )
                    if len(self.per_fly_metrics) > 0:
                        result['per_fly_metrics'] = self.per_fly_metrics
            except Exception as e:
                logger.warning("Per-fly metrics failed: %s", e)

            # Climbing index
            try:
                from analysis.metrics import climbing_index
                raw = getattr(self.detector, 'df_big', positions_df)
                if 'y' in raw.columns and 'vial' in raw.columns:
                    self.climbing_index = climbing_index(raw)
                    result['climbing_index'] = self.climbing_index
            except Exception as e:
                logger.warning("Climbing index failed: %s", e)

    def _save_to_database(self, params: dict, result: dict):
        """Auto-save analysis results to SQLite database."""
        try:
            from output.database import (
                init_db,
                save_experiment,
                save_fly_tracks,
                save_slopes,
                save_video,
            )
            conn = init_db()
            exp_name = getattr(self.detector, 'name', 'unnamed')
            exp_id = save_experiment(conn, exp_name,
                                     config=params,
                                     notes="Auto-saved from GUI analysis")
            quality = self.quality_scores or {}
            vid_id = save_video(conn, exp_id,
                                video_path=self.video_path or '',
                                n_frames=getattr(self.detector, 'n_frames', 0),
                                fps=getattr(self.detector, 'frame_rate', 0),
                                width=getattr(self.detector, 'width', 0),
                                height=getattr(self.detector, 'height', 0),
                                quality_score=quality.get('overall_score', None),
                                quality_level=quality.get('overall_level', None))
            if self.slopes_df is not None:
                save_slopes(conn, vid_id, self.slopes_df)
            if self.positions_df is not None and 'particle' in self.positions_df.columns:
                save_fly_tracks(conn, vid_id, self.positions_df)
            conn.commit()
            conn.close()
            logger.info("Results saved to database (experiment=%s, video=%s)", exp_id, vid_id)
        except Exception as e:
            logger.warning("Database save failed: %s", e)

        try:
            from output.provenance import save_provenance
            save_provenance(self.video_path, params, self.slopes_df)
        except Exception as e:
            logger.warning("Provenance save failed: %s", e)

    # ------------------------------------------------------------------
    # Pipeline-only (no plotting) for threaded analysis
    # ------------------------------------------------------------------
    def run_pipeline_only(self, params: dict, progress_callback=None) -> dict:
        """Run detector steps 1-7 without plotting. Safe for background threads.

        progress_callback(step, total, message) is called per step.
        Returns result dict (same as test_parameters but without axes).
        """
        if self.detector is None:
            raise RuntimeError("Load a video first")

        variables = self._params_to_variables(params)
        self.config = params.copy()
        self.detector.load_for_gui(variables=variables)

        steps = [
            (self.detector.step_1, {'gui': False}, "Background subtraction"),
            (self.detector.step_2, {}, "Spot detection"),
            (self.detector.step_3, {'gui': False}, "Spot metrics"),
            (self.detector.step_4, {}, "Vial assignment"),
            (self.detector.step_5, {}, "Outlier removal"),
            (self.detector.step_6, {'gui': False}, "Linear regression"),
            (self.detector.step_7, {}, "Slope calculation"),
        ]

        for i, (fn, kwargs, msg) in enumerate(steps):
            if self._cancel.is_set():
                raise RuntimeError("Analysis cancelled")
            if progress_callback:
                progress_callback(i, len(steps), f"Step {i+1}/7: {msg}...")
            fn(**kwargs)

        if progress_callback:
            progress_callback(7, 7, "Computing metrics...")

        result = {'paths': {}}
        if hasattr(self.detector, 'df_slopes') and self.detector.df_slopes is not None:
            self.slopes_df = self.detector.df_slopes
            result['slopes_df'] = self.slopes_df
        if hasattr(self.detector, 'df_filtered') and self.detector.df_filtered is not None:
            self.positions_df = self.detector.df_filtered
            result['positions_df'] = self.positions_df

        self._compute_post_analysis(params, result)
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
                result = self.run_pipeline_only(params, progress_callback)
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
    def get_slopes(self) -> pd.DataFrame:
        return self.slopes_df

    def get_positions(self) -> pd.DataFrame:
        return self.positions_df

    def export_results(self, fmt: str, path: str):
        """Export results in the given format.

        fmt: 'csv', 'tidy', 'prism', 'excel', 'tracks'
        """
        if self.slopes_df is None:
            raise RuntimeError("No results to export — run analysis first")

        from output.export import (
            export_per_fly_tracks,
            export_prism_csv,
            export_slopes_csv,
            export_tidy_csv,
        )

        if fmt == 'csv':
            export_slopes_csv(self.slopes_df, path)
        elif fmt == 'tidy':
            export_tidy_csv(self.slopes_df, path)
        elif fmt == 'prism':
            export_prism_csv(self.slopes_df, path, group_col='geno')
        elif fmt == 'excel':
            from output.export import export_excel
            export_excel(
                self.slopes_df, path,
                per_fly_df=self.per_fly_metrics,
                params=self.config,
            )
        elif fmt == 'tracks' and self.positions_df is not None:
            export_per_fly_tracks(self.positions_df, path)
        elif fmt == 'pdf':
            from output.reports import generate_pdf_report
            generate_pdf_report(
                self.slopes_df,
                figures={},
                params=self.config,
                output_path=path,
            )
        elif fmt == 'html':
            from output.reports import generate_html_report
            generate_html_report(
                self.slopes_df,
                params=self.config,
                output_path=path,
            )
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

    @staticmethod
    def _validate_profile_name(name: str):
        if not name or '/' in name or '\\' in name or '..' in name:
            raise ValueError(f"Invalid profile name: {name!r}")

    def save_profile(self, name: str, params: dict):
        """Save params as a named profile."""
        self._validate_profile_name(name)
        from config import save_config
        os.makedirs(self.PROFILES_DIR, exist_ok=True)
        save_config(os.path.join(self.PROFILES_DIR, f'{name}.cfg'), params)

    def load_profile(self, name: str) -> dict:
        """Load a named profile."""
        self._validate_profile_name(name)
        from config import load_config
        path = os.path.join(self.PROFILES_DIR, f'{name}.cfg')
        params = load_config(path)
        self.config = params
        return params

    def list_profiles(self) -> list:
        """Return list of saved profile names."""
        if not os.path.isdir(self.PROFILES_DIR):
            return []
        return sorted(
            f[:-4] for f in os.listdir(self.PROFILES_DIR) if f.endswith('.cfg')
        )

    def delete_profile(self, name: str):
        self._validate_profile_name(name)
        path = os.path.join(self.PROFILES_DIR, f'{name}.cfg')
        if os.path.exists(path):
            os.remove(path)

    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------
    def run_batch(self, video_paths: list, params: dict,
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
    def _params_to_variables(params: dict) -> list:
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
