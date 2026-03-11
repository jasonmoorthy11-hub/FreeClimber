"""Tests for output/reports.py — report generation."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from output.reports import generate_methods_paragraph


class TestMethodsParagraph:
    def test_returns_string(self, sample_config):
        text = generate_methods_paragraph(sample_config)
        assert isinstance(text, str)
        assert len(text) > 50

    def test_contains_key_terms(self, sample_config):
        text = generate_methods_paragraph(sample_config)
        assert 'FreeClimber' in text
        assert str(sample_config['frame_rate']) in text
        assert str(sample_config['vials']) in text
        assert str(sample_config['diameter']) in text

    def test_contains_version(self, sample_config):
        text = generate_methods_paragraph(sample_config)
        assert 'v2.0' in text


class TestHTMLReport:
    @pytest.fixture(autouse=True)
    def _check_plotly(self):
        pytest.importorskip("plotly")

    def test_generates_html_file(self, sample_slopes_df, tmp_path):
        from output.reports import generate_html_report
        out = str(tmp_path / 'report.html')
        generate_html_report(sample_slopes_df, output_path=out)
        assert os.path.exists(out)
        content = open(out).read()
        assert '<html>' in content.lower()

    def test_html_with_stats(self, sample_slopes_df, tmp_path):
        from output.reports import generate_html_report
        stats = {'test': 'ANOVA', 'p_value': 0.01, 'significant': True,
                 'effect_size': 0.45, 'effect_size_name': 'eta_squared'}
        out = str(tmp_path / 'report_stats.html')
        generate_html_report(sample_slopes_df, stats_result=stats, output_path=out)
        content = open(out).read()
        assert 'ANOVA' in content

    def test_html_with_experiment_name(self, sample_slopes_df, tmp_path):
        from output.reports import generate_html_report
        out = str(tmp_path / 'report_named.html')
        generate_html_report(sample_slopes_df, output_path=out, experiment_name='Test Experiment')
        content = open(out).read()
        assert 'Test Experiment' in content
