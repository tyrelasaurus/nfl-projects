import os
import tempfile
import yaml
import pytest

from ncaa_model.config_manager import NCAAConfigManager, ConfigurationError


def write_yaml(tmpdir, content):
    path = os.path.join(tmpdir, 'config.yaml')
    with open(path, 'w') as f:
        yaml.safe_dump(content, f)
    return path


def test_load_basic_config():
    with tempfile.TemporaryDirectory() as td:
        cfg = {
            'model': {'home_field_advantage': 2.5},
            'logging': {'level': 'INFO'},
        }
        path = write_yaml(td, cfg)
        mgr = NCAAConfigManager(config_path=path)
        loaded = mgr.load_config()
        assert loaded.model.home_field_advantage == 2.5


def test_invalid_yaml_dict_raises():
    with tempfile.TemporaryDirectory() as td:
        # Write a list instead of dict
        path = os.path.join(td, 'config.yaml')
        with open(path, 'w') as f:
            f.write('- a\n- b\n')
        mgr = NCAAConfigManager(config_path=path)
        with pytest.raises(ConfigurationError):
            mgr.load_config()


def test_environment_overrides_merge():
    with tempfile.TemporaryDirectory() as td:
        cfg = {
            'model': {'home_field_advantage': 2.0},
            'environments': {
                'production': {'model': {'home_field_advantage': 3.0}}
            }
        }
        path = write_yaml(td, cfg)
        mgr = NCAAConfigManager(config_path=path, environment='production')
        loaded = mgr.load_config()
        assert loaded.model.home_field_advantage == 3.0

