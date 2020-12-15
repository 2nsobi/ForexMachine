from pathlib import Path
from typing import Optional
import yaml

PACKAGE_ROOT = Path(__file__).parent.resolve()


def make_data_directory() -> None:
    (PACKAGE_ROOT / f'Data/DataWithIndicators/').mkdir(parents=True, exist_ok=True)
    (PACKAGE_ROOT / f'Data/RawData/').mkdir(parents=True, exist_ok=True)
    (PACKAGE_ROOT / f'Data/TrainingData/').mkdir(parents=True, exist_ok=True)
    (PACKAGE_ROOT / f'Data/.NumpyData/').mkdir(parents=True, exist_ok=True)


def yaml_to_dict(yaml_path: Optional[str] = (PACKAGE_ROOT / 'Config/main_config.yml')) -> dict:
    # if get_indicators is run from command line and yaml path is not give argparse will pass None as yaml_path
    if not yaml_path:
        yaml_path = PACKAGE_ROOT / 'Config/main_config.yml'

    # yaml.add_constructor('!datetime',datetime.datetime)

    with open(yaml_path, 'r') as configYAML:
        yaml_dict = yaml.safe_load(configYAML)
    return yaml_dict


def string_to_type(string: str) -> type:
    types = {
        'str': str,
        'int': int,
        'float': float,
        'bool': bool
    }
    return type(types[string])
