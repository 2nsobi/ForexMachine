import ta
import pandas as pd
import argparse as ap
from pathlib import PurePath
import yaml
from typing import Optional
from ForexMachine import PROJECT_ROOT

def add_indicators_to_raw(filepath: str, config: dict, broker: Optional[str] = 'tw',
                          save_to_disk: Optional[bool] = False,
                          file_save_name: Optional[str] = None) -> pd.DataFrame:
    # edit column names for raw data from different brokers
    raw_data = None
    column_headers = []

    if broker == 'tw':
        column_headers.extend(('Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'))
        raw_data = pd.read_csv(filepath, names=column_headers)

    current_model = config['current_model']
    indicators = config[current_model]['indicators']

    indicator_functions = {
        'ichimoku': lambda: add_ichimoku_cloud(raw_data),
        'rsi': lambda: add_rsi(raw_data)
    }

    for indicator in indicators:
        indicator_functions[indicator]()

    if save_to_disk:
        if file_save_name:
            raw_data.to_csv(PROJECT_ROOT / f'Data/DataWithIndicators/{file_save_name}.csv')
        else:
            raw_data_file_name = PurePath(filepath).name
            if '.' in raw_data_file_name:
                x = raw_data_file_name[::-1].find('.')
                no_ext = raw_data_file_name[:len(raw_data_file_name) - x - 1]

            raw_data.to_csv(PROJECT_ROOT / f'Data/DataWithIndicators/{no_ext}-{current_model}-model.csv')

    return raw_data


# is just copy of Close column 26 periods in the past on default settings
def add_chikou_span(raw_data: pd.DataFrame, delay_periods: int = 26) -> None:
    c_span = raw_data['Close'].to_list()[delay_periods:]
    c_span.extend([None] * delay_periods)
    raw_data['chikou_span'] = c_span


def add_ichimoku_cloud(df: pd.DataFrame, delay_periods: int = 26, n1: int = 9, n2: int = 26, n3: int = 52) -> None:
    indicator = ta.trend.IchimokuIndicator(high=df['High'], low=df['Low'], n1=n1, n2=n2, n3=n3, visual=False)
    df['trend_ichimoku_conv'] = indicator.ichimoku_conversion_line()
    df['trend_ichimoku_base'] = indicator.ichimoku_base_line()
    df['trend_ichimoku_a'] = indicator.ichimoku_a()
    df['trend_ichimoku_b'] = indicator.ichimoku_b()

    indicator = ta.trend.IchimokuIndicator(high=df['High'], low=df['Low'], n1=n1, n2=n2, n3=n3, visual=True)
    df['trend_visual_ichimoku_a'] = indicator.ichimoku_a()
    df['trend_visual_ichimoku_b'] = indicator.ichimoku_b()

    # Chikou Span trendline of Ichimoku is not apart of 'ta' package
    add_chikou_span(df, delay_periods)


def add_rsi(df: pd.DataFrame, n: int = 14) -> None:
    df['momentum_rsi'] = ta.momentum.RSIIndicator(close=df['Close'], n=n).rsi()


if __name__ == '__main__':
    with open((PROJECT_ROOT / 'Config/main_config.yml'), 'r') as configYAML:
        config = yaml.safe_load(configYAML)

    parser = ap.ArgumentParser()
    parser.add_argument('filepath')
    parser.add_argument('-b', '--broker', dest='broker', default='tw')
    parser.add_argument('-s', '--save', dest='save_to_disk', action='store_true')
    parser.add_argument('-n', '--name', dest='file_save_name')
    args = parser.parse_args()

    filepath = args.filepath
    broker = args.broker
    save_to_disk = args.save_to_disk
    file_save_name = args.file_save_name
    add_indicators_to_raw(filepath=filepath, config=config, broker=broker, save_to_disk=save_to_disk,
                          file_save_name=file_save_name)
