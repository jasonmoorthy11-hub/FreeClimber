"""Tests for config.py — safe parsing, validation, backward compatibility."""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from config import (
    load_config, save_config, validate_config,
    apply_config, parse_variable_list, _parse_value,
    KNOWN_PARAMS,
)


class TestParseValue:
    def test_integer(self):
        assert _parse_value('42') == 42

    def test_float(self):
        assert _parse_value('.05') == 0.05
        assert _parse_value('3.14') == 3.14

    def test_string_quoted(self):
        assert _parse_value('"hello"') == 'hello'
        assert _parse_value('"auto"') == 'auto'

    def test_bool(self):
        assert _parse_value('True') is True
        assert _parse_value('False') is False

    def test_unquoted_string(self):
        assert _parse_value('h264') == 'h264'

    def test_malicious_import(self):
        result = _parse_value('__import__("os").system("echo pwned")')
        assert isinstance(result, str)

    def test_malicious_open(self):
        result = _parse_value('open("/etc/passwd").read()')
        assert isinstance(result, str)

    def test_malicious_lambda(self):
        result = _parse_value('lambda: None')
        assert isinstance(result, str)

    def test_malicious_subprocess(self):
        result = _parse_value('__import__("subprocess").call(["rm","-rf","/"])')
        assert isinstance(result, str)


class TestLoadConfig:
    def test_load_example(self, example_config_path):
        if not os.path.exists(example_config_path):
            pytest.skip("Example config not found")
        params = load_config(example_config_path)
        assert params['x'] == 100
        assert params['vials'] == 3
        assert params['threshold'] == 'auto'
        assert params['convert_to_cm_sec'] is True
        assert isinstance(params['ecc_low'], float)

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_config('/nonexistent/path.cfg')

    def test_comments_skipped(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
            f.write('## Comment\n')
            f.write('# Another comment\n')
            f.write('x=42\n')
            f.write('\n')
            f.write('y=10\n')
            f.name
        try:
            params = load_config(f.name)
            assert params == {'x': 42, 'y': 10}
        finally:
            os.unlink(f.name)

    def test_round_trip(self, sample_config):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
            path = f.name
        try:
            save_config(path, sample_config)
            loaded = load_config(path)
            for key in sample_config:
                assert loaded[key] == sample_config[key], f"Mismatch for {key}"
        finally:
            os.unlink(path)


class TestValidateConfig:
    def test_valid_config(self, sample_config):
        errors = validate_config(sample_config)
        assert len(errors) == 0

    def test_even_diameter(self):
        errors = validate_config({'diameter': 6})
        assert any('diameter' in e and 'odd' in e for e in errors)

    def test_negative_frame_rate(self):
        errors = validate_config({'frame_rate': -1})
        assert any('frame_rate' in e for e in errors)

    def test_blank_range(self):
        errors = validate_config({'blank_0': 50, 'crop_0': 100})
        assert any('blank_0' in e for e in errors)


class TestApplyConfig:
    def test_apply(self):
        class Obj:
            pass
        obj = Obj()
        apply_config(obj, {'x': 10, 'name': 'test'})
        assert obj.x == 10
        assert obj.name == 'test'


class TestParseVariableList:
    def test_basic(self):
        variables = ['x=100', 'y=200', 'name="hello"', 'flag=True']
        params = parse_variable_list(variables)
        assert params['x'] == 100
        assert params['y'] == 200
        assert params['name'] == 'hello'
        assert params['flag'] is True

    def test_skip_comments(self):
        variables = ['# comment', 'x=5', '', '\n']
        params = parse_variable_list(variables)
        assert params == {'x': 5}

    def test_malicious_input(self):
        variables = ['x=__import__("os").system("echo pwned")']
        params = parse_variable_list(variables)
        # Should be treated as string, not executed
        assert isinstance(params['x'], str)
