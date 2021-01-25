from pathlib import Path
from typing import Optional
import logging
import linecache
import inspect


PACKAGE_ROOT_DIR = Path(__file__).parent.resolve()
TICKS_DATA_DIR = PACKAGE_ROOT_DIR / './PackageData/TicksData'
MODEL_FILES_DIR = PACKAGE_ROOT_DIR / './PackageData/ModelFiles'
LIVE_TRADE_FILES_DIR = PACKAGE_ROOT_DIR / './PackageData/.LiveTradingFiles'
TEMP_DIR = PACKAGE_ROOT_DIR / './PackageData/.temp'


class Logger:
    __logger_instance = None

    @staticmethod
    def get_instance():
        if Logger.__logger_instance is None:
            logging.basicConfig(format=('%(filename)s: '
                                        '%(levelname)s: '
                                        '%(funcName)s(): '
                                        '%(lineno)d:\t'
                                        '%(message)s')
                                )
            Logger.__logger_instance = logging.getLogger(__name__)
        return Logger.__logger_instance


def create_folder(path):
    path = Path(path)
    if not path.is_dir():
        path.mkdir(parents=True)
    return path.resolve()


def get_ticks_data_dir():
    create_folder(TICKS_DATA_DIR)
    return TICKS_DATA_DIR


def get_model_files_dir():
    create_folder(MODEL_FILES_DIR)
    return MODEL_FILES_DIR


def get_live_trade_files_dir():
    create_folder(LIVE_TRADE_FILES_DIR)
    return LIVE_TRADE_FILES_DIR


def get_temp_dir():
    create_folder(TEMP_DIR)
    return TEMP_DIR
