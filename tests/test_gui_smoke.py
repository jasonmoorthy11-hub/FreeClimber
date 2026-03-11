"""Smoke tests for the customtkinter GUI — imports, controller, param collection."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


class TestController:
    def test_import(self):
        from gui.controller import AnalysisController
        ctrl = AnalysisController()
        assert ctrl.detector is None
        assert ctrl.video_path is None

    def test_params_to_variables(self):
        from gui.controller import AnalysisController
        params = {'x': 100, 'diameter': 7, 'threshold': 'auto', 'trim_outliers': True}
        variables = AnalysisController._params_to_variables(params)
        assert 'x=100' in variables
        assert 'diameter=7' in variables
        assert 'threshold="auto"' in variables
        assert 'trim_outliers=True' in variables

    def test_no_results_before_analysis(self):
        from gui.controller import AnalysisController
        ctrl = AnalysisController()
        assert ctrl.get_slopes() is None
        assert ctrl.get_positions() is None

    def test_export_raises_without_results(self):
        from gui.controller import AnalysisController
        ctrl = AnalysisController()
        with pytest.raises(RuntimeError, match="No results"):
            ctrl.export_results('csv', '/tmp/test.csv')


class TestExportDispatch:
    def test_export_csv_with_slopes(self, sample_slopes_df, tmp_path):
        from gui.controller import AnalysisController
        ctrl = AnalysisController()
        ctrl.slopes_df = sample_slopes_df
        out = str(tmp_path / 'test.csv')
        ctrl.export_results('csv', out)
        import pandas as pd
        df = pd.read_csv(out)
        assert len(df) == len(sample_slopes_df)

    def test_export_unknown_format_raises(self, sample_slopes_df):
        from gui.controller import AnalysisController
        ctrl = AnalysisController()
        ctrl.slopes_df = sample_slopes_df
        import pytest as pt
        with pt.raises(ValueError):
            ctrl.export_results('unknown', '/tmp/test.csv')


class TestGUIImport:
    def test_app_module_imports(self):
        """Verify the GUI app module can be imported without starting tkinter."""
        # We can't instantiate the app in headless CI, but importing should work
        import gui
        from gui.controller import AnalysisController
        assert AnalysisController is not None
