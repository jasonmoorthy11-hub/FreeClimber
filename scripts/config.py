"""Safe configuration parser for FreeClimber.

Replaces unsafe config loading with ast.literal_eval (safe string-to-value).
Backward compatible with existing .cfg files (same key=value format).
"""

import ast
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Known parameters: name -> (type, default, min, max, description)
KNOWN_PARAMS = {
    'x':                    (int,   0,      0,    None,  'ROI left x-position (pixels)'),
    'y':                    (int,   0,      0,    None,  'ROI top y-position (pixels)'),
    'w':                    (int,   0,      0,    None,  'ROI width (pixels)'),
    'h':                    (int,   0,      0,    None,  'ROI height (pixels)'),
    'check_frame':          (int,   0,      0,    None,  'Frame to display for parameter check'),
    'blank_0':              (int,   0,      0,    None,  'First frame for background computation'),
    'blank_n':              (int,   0,      0,    None,  'Last frame for background computation'),
    'crop_0':               (int,   0,      0,    None,  'First frame to analyze'),
    'crop_n':               (int,   0,      0,    None,  'Last frame to analyze'),
    'threshold':            (None,  'auto', None, None,  'Signal threshold ("auto" or numeric)'),
    'diameter':             (int,   7,      1,    None,  'Expected spot diameter (odd integer)'),
    'minmass':              (int,   100,    0,    None,  'Minimum spot mass for detection'),
    'maxsize':              (int,   11,     1,    None,  'Maximum spot diameter'),
    'ecc_low':              (float, 0.0,    0.0,  1.0,   'Minimum eccentricity (circularity filter)'),
    'ecc_high':             (float, 1.0,    0.0,  1.0,   'Maximum eccentricity (circularity filter)'),
    'vials':                (int,   1,      1,    None,  'Number of vials in video'),
    'window':               (int,   50,     1,    None,  'Sliding window size for local linear regression'),
    'pixel_to_cm':          (float, 1.0,    0.0,  None,  'Pixels per centimeter calibration'),
    'frame_rate':           (int,   30,     1,    None,  'Video frame rate (fps)'),
    'vial_id_vars':         (int,   2,      1,    None,  'Number of naming convention vars for vial ID'),
    'outlier_TB':           (float, 1.0,    0.0,  None,  'Top/bottom outlier trim sensitivity'),
    'outlier_LR':           (float, 3.0,    0.0,  None,  'Left/right outlier trim sensitivity'),
    'naming_convention':    (str,   'geno_sex_day_rep', None, None, 'Underscore-separated naming pattern'),
    'path_project':         (str,   '',     None, None,  'Path to project folder'),
    'file_suffix':          (str,   'h264', None, None,  'Video file extension'),
    'convert_to_cm_sec':    (bool,  False,  None, None,  'Convert final slope to cm/sec'),
    'trim_outliers':        (bool,  False,  None, None,  'Trim outlier points at edges'),
    'background_method':    (str,   'temporal_median', None, None, 'Background subtraction method: temporal_median, mog2, or running_average'),
    'individual_tracking':  (bool,  False,  None, None,  'Enable per-fly individual tracking via tp.link'),
}


def _parse_value(raw_value: str):
    """Safely parse a config value string.

    Uses ast.literal_eval which only allows literals (strings, numbers,
    tuples, lists, dicts, booleans, None) — no arbitrary code.
    Falls back to stripped string if parsing fails.
    """
    stripped = raw_value.strip()

    if stripped in ('True', 'False', 'None'):
        return ast.literal_eval(stripped)

    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return stripped


def _coerce_type(key: str, value, expected_type):
    """Coerce a parsed value to the expected type for a known parameter."""
    if expected_type is None:
        return value

    if expected_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes')
        return bool(value)

    if expected_type is int:
        if isinstance(value, (int, float)):
            return int(value)
        try:
            return int(float(value))
        except (ValueError, TypeError):
            logger.warning("Could not convert %s=%r to int", key, value)
            return value

    if expected_type is float:
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning("Could not convert %s=%r to float", key, value)
            return value

    if expected_type is str:
        return str(value)

    return value


def load_config(filepath: str) -> dict:
    """Load a .cfg file safely. Returns dict of parameter name -> value.

    Backward compatible with original FreeClimber .cfg format (key=value lines).
    Comments (#) and blank lines are skipped.
    """
    params = {}

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, 'r') as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()

            if not line or line.startswith('#'):
                continue

            if '=' not in line:
                logger.warning("Config line %d: no '=' found, skipping: %s", line_num, line)
                continue

            key, _, raw_value = line.partition('=')
            key = key.strip()
            raw_value = raw_value.strip()

            if not key:
                logger.warning("Config line %d: empty key, skipping", line_num)
                continue

            value = _parse_value(raw_value)

            if key in KNOWN_PARAMS:
                expected_type = KNOWN_PARAMS[key][0]
                value = _coerce_type(key, value, expected_type)
            else:
                logger.warning("Unknown config parameter: %s (keeping anyway)", key)

            params[key] = value

    return params


def save_config(filepath: str, params: dict, video_file: str = ''):
    """Write parameters to a .cfg file in the original FreeClimber format."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filepath, 'w') as f:
        f.write('## FreeClimber ##\n')
        if video_file:
            f.write(f'## Generated from file: {video_file}\n')
        f.write(f'##     @ {now}\n')
        f.write('##\n')
        f.write('## Analysis parameters:\n')

        for key, value in params.items():
            if isinstance(value, str):
                f.write(f'{key}="{value}"\n')
            elif isinstance(value, bool):
                f.write(f'{key}={value}\n')
            else:
                f.write(f'{key}={value}\n')


def validate_config(params: dict) -> list:
    """Validate configuration parameters. Returns list of error messages (empty = valid)."""
    errors = []

    for key, value in params.items():
        if key not in KNOWN_PARAMS:
            continue

        expected_type, default, min_val, max_val, desc = KNOWN_PARAMS[key]

        if expected_type is not None and not isinstance(value, expected_type):
            if not (expected_type in (int, float) and isinstance(value, (int, float))):
                errors.append(f"{key}: expected {expected_type.__name__}, got {type(value).__name__}")
                continue

        if min_val is not None and isinstance(value, (int, float)):
            if value < min_val:
                errors.append(f"{key}: value {value} below minimum {min_val}")

        if max_val is not None and isinstance(value, (int, float)):
            if value > max_val:
                errors.append(f"{key}: value {value} above maximum {max_val}")

    if 'blank_0' in params and 'crop_0' in params:
        if params['blank_0'] < params['crop_0']:
            errors.append(f"blank_0 ({params['blank_0']}) < crop_0 ({params['crop_0']})")

    if 'blank_n' in params and 'crop_n' in params:
        if params['blank_n'] > params['crop_n']:
            errors.append(f"blank_n ({params['blank_n']}) > crop_n ({params['crop_n']})")

    if 'diameter' in params:
        d = params['diameter']
        if isinstance(d, int) and d % 2 == 0:
            errors.append(f"diameter ({d}) must be odd; will be auto-corrected to {d+1}")

    return errors


def apply_config(obj, params: dict):
    """Safely apply config parameters as attributes on an object.

    Uses setattr() instead of arbitrary code execution.
    """
    for key, value in params.items():
        setattr(obj, key, value)


def config_to_variable_list(params: dict) -> list:
    """Convert a params dict back to the 'key=value' string list format
    used by the GUI's update_variables() method.
    """
    variables = []
    for key, value in params.items():
        if isinstance(value, str):
            variables.append(f'{key}="{value}"')
        else:
            variables.append(f'{key}={value}')
    return variables


def parse_variable_list(variables: list) -> dict:
    """Parse a list of 'key=value' strings into a dict safely."""
    params = {}
    for item in variables:
        item = item.strip()
        if not item or item.startswith(('#', '\n', '\t')):
            continue

        if '=' not in item:
            logger.warning("Variable has no '=': %s", item)
            continue

        key, _, raw_value = item.partition('=')
        key = key.strip()
        raw_value = raw_value.strip()

        if not key:
            continue

        value = _parse_value(raw_value)

        if key in KNOWN_PARAMS:
            expected_type = KNOWN_PARAMS[key][0]
            value = _coerce_type(key, value, expected_type)

        params[key] = value

    return params
