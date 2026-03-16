"""Integration test: run full pipeline on example video, verify regression."""

import os

import numpy as np
import pandas as pd
import pytest

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'example')
KNOWN_SLOPES = {
    'w1118_m_1': 1.3968,
    'w1118_m_2': 1.3187,
    'w1118_m_3': 1.2233,
}
SLOPE_TOLERANCE = 0.15
MIN_R_VALUE = 0.70


def _find_example_video():
    for ext in ['h264', 'mp4', 'mov', 'avi']:
        path = os.path.join(EXAMPLE_DIR, f'w1118_m_2_1.{ext}')
        if os.path.exists(path):
            return path
    return None


class TestSyntheticPipeline:
    """Run detector pipeline on synthetic video — no real video needed."""

    def test_pipeline_produces_slopes(self, synthetic_video_path, tmp_path):
        from config import save_config
        from detector import detector

        cfg = {
            'x': 0, 'y': 0, 'w': 320, 'h': 240,
            'check_frame': 0, 'blank_0': 0, 'blank_n': 9,
            'crop_0': 0, 'crop_n': 9,
            'threshold': 'auto', 'diameter': 5, 'minmass': 10,
            'maxsize': 15, 'ecc_low': 0.0, 'ecc_high': 1.0,
            'vials': 3, 'window': 5, 'pixel_to_cm': 46.0,
            'frame_rate': 30, 'vial_id_vars': 2,
            'outlier_TB': 5.0, 'outlier_LR': 5.0,
            'naming_convention': 'geno_sex_day_rep',
            'path_project': str(tmp_path),
            'file_suffix': 'mp4',
            'convert_to_cm_sec': False,
            'trim_outliers': False,
        }
        cfg_path = str(tmp_path / 'synthetic.cfg')
        save_config(cfg_path, cfg, video_file=synthetic_video_path)

        d = detector(video_file=synthetic_video_path, config_file=cfg_path)
        d.step_1(gui=False)
        d.step_2()
        d.step_3(gui=False)
        d.step_4()
        d.step_5()
        d.step_6(gui=False)
        d.step_7()

        assert hasattr(d, 'df_slopes')
        assert len(d.df_slopes) >= 1


@pytest.mark.slow
class TestFullPipeline:
    """Run detector pipeline on example video, check slopes match known-good values."""

    def test_pipeline_regression(self, tmp_path):
        from config import load_config, save_config
        from detector import detector

        video_path = _find_example_video()
        if video_path is None:
            pytest.skip("No example video found")

        cfg_path = os.path.join(EXAMPLE_DIR, 'example.cfg')
        params = load_config(cfg_path)
        params['path_project'] = str(tmp_path)

        tmp_cfg = str(tmp_path / 'test.cfg')
        save_config(tmp_cfg, params, video_file=video_path)

        d = detector(video_file=video_path, config_file=tmp_cfg)
        d.step_1(gui=False)
        d.step_2()
        d.step_3(gui=False)
        d.step_4()
        d.step_5()
        d.step_6(gui=False)
        d.step_7()

        assert hasattr(d, 'df_slopes')
        slopes_df = d.df_slopes
        assert len(slopes_df) >= 3

        for _, row in slopes_df.iterrows():
            vid = str(row.iloc[0])
            if vid.endswith('_all'):
                continue
            if vid in KNOWN_SLOPES:
                assert abs(row['slope'] - KNOWN_SLOPES[vid]) < SLOPE_TOLERANCE, \
                    f"Slope for {vid}: {row['slope']} vs expected {KNOWN_SLOPES[vid]}"
            if 'r_value' in slopes_df.columns:
                assert row['r_value'] > MIN_R_VALUE, \
                    f"r_value for {vid}: {row['r_value']}"

    def test_known_slopes_csv_exists(self):
        slopes_path = os.path.join(EXAMPLE_DIR, 'w1118_m_2_1.slopes.csv')
        if not os.path.exists(slopes_path):
            pytest.skip("No known-good slopes CSV")
        df = pd.read_csv(slopes_path)
        assert len(df) >= 3
        assert 'slope' in df.columns

    def test_analysis_modules_on_real_output(self):
        slopes_path = os.path.join(EXAMPLE_DIR, 'w1118_m_2_1.slopes.csv')
        if not os.path.exists(slopes_path):
            pytest.skip("No known-good slopes CSV")

        from analysis.metrics import compute_population_metrics
        from analysis.stats import compare_multiple_groups, publication_stats_table

        slopes_df = pd.read_csv(slopes_path)
        non_all = slopes_df[~slopes_df.iloc[:, 0].astype(str).str.endswith('_all')]

        if len(non_all) >= 2 and 'slope' in non_all.columns:
            groups = {}
            for _, row in non_all.iterrows():
                groups[str(row.iloc[0])] = [row['slope']]

            metrics = compute_population_metrics(pd.DataFrame(), slopes_df=slopes_df)
            assert 'mean_speed' in metrics

    def test_export_roundtrip(self, tmp_path):
        slopes_path = os.path.join(EXAMPLE_DIR, 'w1118_m_2_1.slopes.csv')
        if not os.path.exists(slopes_path):
            pytest.skip("No known-good slopes CSV")

        from output.export import export_slopes_csv, export_tidy_csv

        slopes_df = pd.read_csv(slopes_path)

        csv_out = str(tmp_path / 'slopes.csv')
        export_slopes_csv(slopes_df, csv_out)
        assert os.path.exists(csv_out)
        reloaded = pd.read_csv(csv_out)
        assert len(reloaded) == len(slopes_df)

        tidy_out = str(tmp_path / 'tidy.csv')
        export_tidy_csv(slopes_df, tidy_out)
        assert os.path.exists(tidy_out)
