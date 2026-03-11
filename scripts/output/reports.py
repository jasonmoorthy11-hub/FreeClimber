"""Report generation for FreeClimber — PDF and interactive HTML.

PDF: Jinja2 template -> HTML -> WeasyPrint PDF
HTML: Plotly interactive standalone page
"""
from __future__ import annotations

import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_pdf_report(
    slopes_df: pd.DataFrame,
    stats_result: dict | None = None,
    figures: dict | None = None,
    params: dict | None = None,
    output_path: str = 'report.pdf',
    experiment_name: str = '',
):
    """Generate a PDF report.

    Args:
        slopes_df: slopes DataFrame
        stats_result: output from compare_multiple_groups or compare_two_groups
        figures: dict of {name: matplotlib.figure.Figure} to embed
        params: analysis parameters used
        output_path: where to save the PDF
        experiment_name: experiment identifier for the title page
    """
    try:
        from jinja2 import Template
    except ImportError:
        logger.error("jinja2 not installed — pip install jinja2")
        raise ImportError("PDF reports require jinja2: pip install jinja2")

    try:
        from weasyprint import HTML as WeasyprintHTML
    except ImportError:
        logger.error("weasyprint not installed — pip install weasyprint")
        raise ImportError("PDF reports require weasyprint: pip install weasyprint")

    import base64
    from io import BytesIO

    # Convert figures to base64 PNGs
    fig_images = {}
    if figures:
        for name, fig in figures.items():
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            fig_images[name] = base64.b64encode(buf.read()).decode('utf-8')

    # Build stats summary text
    stats_lines = []
    if stats_result:
        stats_lines.append(f"Test: {stats_result.get('test', 'N/A')}")
        stats_lines.append(f"Statistic: {stats_result.get('statistic', 'N/A')}")
        stats_lines.append(f"p-value: {stats_result.get('p_value', 'N/A')}")
        if 'effect_size' in stats_result:
            stats_lines.append(f"Effect size ({stats_result.get('effect_size_name', '')}): {stats_result['effect_size']}")
        if 'post_hoc' in stats_result and stats_result['post_hoc']:
            stats_lines.append(f"Post-hoc: {stats_result.get('post_hoc_method', '')}")
            for comp in stats_result['post_hoc']:
                stats_lines.append(
                    f"  {comp['group1']} vs {comp['group2']}: "
                    f"p={comp['p_value']} {comp['significance']}, d={comp['effect_size_d']}"
                )

    # Parameters summary
    params_lines = []
    if params:
        for k, v in sorted(params.items()):
            params_lines.append(f"{k} = {v}")

    methods_text = generate_methods_paragraph(params) if params else ''

    html_content = _PDF_TEMPLATE.render(
        experiment_name=experiment_name or 'FreeClimber Analysis',
        date=datetime.now().strftime('%Y-%m-%d %H:%M'),
        version='2.0.0',
        slopes_html=slopes_df.to_html(index=False, classes='table', float_format=lambda x: f'{x:.4f}'),
        fig_images=fig_images,
        stats_lines=stats_lines,
        params_lines=params_lines,
        methods_text=methods_text,
        n_vials=len(slopes_df),
    )

    WeasyprintHTML(string=html_content).write_pdf(output_path)
    logger.info(f"PDF report saved: {output_path}")


def generate_html_report(
    slopes_df: pd.DataFrame,
    stats_result: dict | None = None,
    groups: dict | None = None,
    params: dict | None = None,
    output_path: str = 'report.html',
    experiment_name: str = '',
):
    """Generate an interactive HTML report using Plotly.

    Standalone HTML file — no server needed.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        logger.error("plotly not installed — pip install plotly")
        raise ImportError("HTML reports require plotly: pip install plotly")

    # Find slope column
    slope_col = None
    for c in slopes_df.columns:
        if any(kw in c.lower() for kw in ['slope', 'velocity', 'speed']):
            slope_col = c
            break
    if slope_col is None and len(slopes_df.columns) >= 2:
        slope_col = slopes_df.columns[1]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Climbing Speed', 'Speed Distribution', 'Data Table', 'Statistics'),
        specs=[[{"type": "bar"}, {"type": "histogram"}],
               [{"type": "table"}, {"type": "table"}]],
    )

    # Bar chart
    if slope_col:
        y_vals = pd.to_numeric(slopes_df[slope_col], errors='coerce').values
        labels = slopes_df.iloc[:, 0].astype(str).values if len(slopes_df.columns) > 0 else [str(i) for i in range(len(slopes_df))]
        fig.add_trace(
            go.Bar(x=labels, y=y_vals, name='Speed',
                   marker_color='#1f6aa5'),
            row=1, col=1
        )

        # Histogram
        fig.add_trace(
            go.Histogram(x=y_vals, name='Distribution', nbinsx=20,
                         marker_color='#56B4E9'),
            row=1, col=2
        )

    # Data table
    fig.add_trace(
        go.Table(
            header=dict(values=list(slopes_df.columns), fill_color='#444444',
                        font=dict(color='white', size=11)),
            cells=dict(
                values=[slopes_df[c].round(4) if slopes_df[c].dtype in ['float64', 'float32'] else slopes_df[c]
                        for c in slopes_df.columns],
                fill_color='#333333',
                font=dict(color='white', size=10),
            )
        ),
        row=2, col=1
    )

    # Stats table
    stats_rows = [['Metric', 'Value']]
    if stats_result:
        stats_rows.append(['Test', str(stats_result.get('test', 'N/A'))])
        stats_rows.append(['p-value', str(stats_result.get('p_value', 'N/A'))])
        stats_rows.append(['Significant', str(stats_result.get('significant', 'N/A'))])
        if 'effect_size' in stats_result:
            stats_rows.append([stats_result.get('effect_size_name', 'Effect size'),
                              str(stats_result['effect_size'])])
    else:
        stats_rows.append(['', 'No statistics computed'])

    fig.add_trace(
        go.Table(
            header=dict(values=['Metric', 'Value'], fill_color='#444444',
                        font=dict(color='white')),
            cells=dict(
                values=[[r[0] for r in stats_rows[1:]], [r[1] for r in stats_rows[1:]]],
                fill_color='#333333',
                font=dict(color='white'),
            )
        ),
        row=2, col=2
    )

    fig.update_layout(
        title=f"{experiment_name or 'FreeClimber Analysis'} — {datetime.now().strftime('%Y-%m-%d')}",
        template='plotly_dark',
        height=900,
        showlegend=False,
    )

    pio.write_html(fig, output_path, include_plotlyjs=True, full_html=True)
    logger.info(f"HTML report saved: {output_path}")


# Jinja2 PDF template
_PDF_TEMPLATE_STR = """
<!DOCTYPE html>
<html>
<head>
<style>
    body { font-family: Helvetica, Arial, sans-serif; margin: 40px; color: #333; font-size: 11pt; }
    h1 { color: #1f6aa5; border-bottom: 2px solid #1f6aa5; padding-bottom: 8px; }
    h2 { color: #444; margin-top: 24px; }
    .table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 10pt; }
    .table th { background: #1f6aa5; color: white; padding: 6px 10px; text-align: left; }
    .table td { border-bottom: 1px solid #ddd; padding: 5px 10px; }
    .table tr:nth-child(even) { background: #f9f9f9; }
    .figure { text-align: center; margin: 16px 0; }
    .figure img { max-width: 100%; }
    .stats { background: #f5f5f5; padding: 12px; border-radius: 4px; font-family: monospace; font-size: 10pt; }
    .params { background: #f0f5fa; padding: 12px; border-radius: 4px; font-family: monospace; font-size: 9pt; }
    .footer { margin-top: 30px; color: #888; font-size: 9pt; text-align: center; border-top: 1px solid #ddd; padding-top: 8px; }
</style>
</head>
<body>
    <h1>{{ experiment_name }}</h1>
    <p>Generated: {{ date }} | FreeClimber v{{ version }} | {{ n_vials }} vials</p>

    <h2>Results</h2>
    {{ slopes_html }}

    {% for name, img_b64 in fig_images.items() %}
    <div class="figure">
        <h3>{{ name }}</h3>
        <img src="data:image/png;base64,{{ img_b64 }}" />
    </div>
    {% endfor %}

    {% if stats_lines %}
    <h2>Statistical Analysis</h2>
    <div class="stats">
        {% for line in stats_lines %}
        {{ line }}<br>
        {% endfor %}
    </div>
    {% endif %}

    {% if methods_text %}
    <h2>Methods</h2>
    <div class="stats">
        {{ methods_text }}
    </div>
    {% endif %}

    {% if params_lines %}
    <h2>Parameters</h2>
    <div class="params">
        {% for line in params_lines %}
        {{ line }}<br>
        {% endfor %}
    </div>
    {% endif %}

    <div class="footer">
        Generated by FreeClimber v{{ version }} &mdash;
        Based on Spierer et al. (2021) J Exp Biol, doi:10.1242/jeb.229377
    </div>
</body>
</html>
"""

try:
    from jinja2 import Template
    _PDF_TEMPLATE = Template(_PDF_TEMPLATE_STR)
except ImportError:
    _PDF_TEMPLATE = None


def generate_methods_paragraph(params: dict) -> str:
    """Auto-generate a publication-ready methods paragraph from config parameters."""
    vials = params.get('vials', '?')
    diameter = params.get('diameter', '?')
    minmass = params.get('minmass', '?')
    window = params.get('window', '?')
    frame_rate = params.get('frame_rate', '?')
    pixel_to_cm = params.get('pixel_to_cm', '?')
    bg = params.get('background_method', 'temporal_median')
    tracking = 'with individual fly tracking' if params.get('individual_tracking') else 'without individual fly tracking'

    threshold = params.get('threshold', 'auto')
    threshold_desc = "Otsu's method" if threshold == 'auto' else f"a fixed threshold of {threshold}"

    crop_0 = params.get('crop_0', '?')
    crop_n = params.get('crop_n', '?')

    text = (
        f"Climbing velocity was measured using FreeClimber v2.0 (Spierer et al., 2021). "
        f"Videos were recorded at {frame_rate} fps and analyzed from frame {crop_0} to {crop_n}. "
        f"Background subtraction was performed using the {bg} method. "
        f"Fly positions were detected using particle detection (diameter={diameter} px, "
        f"minimum mass={minmass}), with binarization via {threshold_desc}. "
        f"The field of view was divided into {vials} vertical strips (vials). "
        f"Climbing speed was estimated by local linear regression over a {window}-frame "
        f"sliding window, {tracking}. "
        f"Spatial calibration: {pixel_to_cm} pixels/cm."
    )
    return text
