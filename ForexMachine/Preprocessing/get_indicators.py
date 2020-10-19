import ta
import pandas as pd
import argparse as ap
from pathlib import PurePath
import yaml
from typing import Optional
from ForexMachine.util import PACKAGE_ROOT, yaml_to_dict
from datetime import datetime


def save_data_with_indicators(df: pd.DataFrame, filename: Optional[str] = None) -> None:
    if not filename:
        filename = 'no_name_data_with_indicators'

    filepath = PACKAGE_ROOT.parent / f'Data/DataWithIndicators/{filename}.csv'
    df.to_csv(filepath)


def add_indicators_to_raw(filepath: str, config: dict, broker: Optional[str] = 'tw',
                          save_to_disk: Optional[bool] = False,
                          file_save_name: Optional[str] = None) -> pd.DataFrame:

    # edit column names for raw data from different brokers
    raw_data = None
    column_headers = []

    if broker == 'tw':
        column_headers.extend(('Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'))
        raw_data = pd.read_csv(filepath, names=column_headers)

    add_datetimes(raw_data)

    current_model = config['current_model']
    indicators = config[current_model]['indicators']

    indicator_functions = {
        'ichimoku': lambda df, **kwargs: add_ichimoku_cloud(df, **kwargs),
        'rsi': lambda df, **kwargs: add_rsi(df, **kwargs)
    }

    for indicator in indicators:
        indicator_functions[indicator](raw_data, **indicators[indicator])

    if save_to_disk:
        if file_save_name:
            raw_data.to_csv(PACKAGE_ROOT.parent / f'Data/DataWithIndicators/{file_save_name}.csv')
        else:
            raw_data_file_name = PurePath(filepath).name
            if '.' in raw_data_file_name:
                x = raw_data_file_name[::-1].find('.')
                no_ext = raw_data_file_name[:len(raw_data_file_name) - x - 1]

            raw_data.to_csv(PACKAGE_ROOT.parent / f'Data/DataWithIndicators/{no_ext}-{current_model}-model.csv')

    return raw_data


def add_datetimes(raw_data: pd.DataFrame) -> None:
    datetimes = list(map(lambda row: datetime.strptime(f'{row.Date} {row.Time}', '%Y.%m.%d %H:%M'),
                         raw_data.itertuples()))
    raw_data['datetime'] = datetimes
    from collections import namedtuple
    x = namedtuple('ichimoku_features', 'is_price_above_cb_lines is_price_above_cloud'
                                        'is_price_inside_cloud is_price_below_cloud cloud_top cloud_bottom')


# is just copy of Close column 26 periods in the past on default settings
def add_chikou_span(raw_data: pd.DataFrame, delay_periods: int = 26) -> None:
    c_span = raw_data['Close'].to_list()[delay_periods:]
    c_span.extend([None] * delay_periods)
    raw_data['chikou_span'] = c_span


def add_ichimoku_cloud(df: pd.DataFrame, chikou_period: int = 26, tenkan_period: int = 9, kijun_period: int = 26,
                       senkou_b_period: int = 52) -> None:
    indicator = ta.trend.IchimokuIndicator(high=df['High'], low=df['Low'], n1=tenkan_period, n2=kijun_period,
                                           n3=senkou_b_period, visual=False)
    df['trend_ichimoku_conv'] = indicator.ichimoku_conversion_line()
    df['trend_ichimoku_base'] = indicator.ichimoku_base_line()
    df['trend_ichimoku_a'] = indicator.ichimoku_a()
    df['trend_ichimoku_b'] = indicator.ichimoku_b()

    indicator = ta.trend.IchimokuIndicator(high=df['High'], low=df['Low'], n1=tenkan_period, n2=kijun_period,
                                           n3=senkou_b_period, visual=True)
    df['trend_visual_ichimoku_a'] = indicator.ichimoku_a()
    df['trend_visual_ichimoku_b'] = indicator.ichimoku_b()

    # Chikou Span trendline of Ichimoku is not apart of 'ta' package
    add_chikou_span(df, chikou_period)


def add_rsi(df: pd.DataFrame, periods: int = 14) -> None:
    df['momentum_rsi'] = ta.momentum.RSIIndicator(close=df['Close'], n=periods).rsi()


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('filepath')
    parser.add_argument('-b', '--broker', dest='broker', default='tw')
    parser.add_argument('-s', '--save', dest='save_to_disk', action='store_true')
    parser.add_argument('-n', '--name', dest='file_save_name')
    parser.add_argument('-c', '--config', dest='config_path')
    args = parser.parse_args()

    filepath = args.filepath
    broker = args.broker
    save_to_disk = args.save_to_disk
    file_save_name = args.file_save_name
    config_path = args.config_path

    config = yaml_to_dict()  # config = yaml_to_dict(config_path)
    # print(config)

    add_indicators_to_raw(filepath=filepath, config=config, broker=broker, save_to_disk=save_to_disk,
                          file_save_name=file_save_name)
