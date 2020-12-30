import ta
import pandas as pd
import argparse as ap
from pathlib import PurePath, Path
from typing import Optional
from ForexMachine.util import PACKAGE_ROOT, yaml_to_dict
import MetaTrader5 as mt5
import os
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import numpy as np

def save_data_with_indicators(df: pd.DataFrame, filename: Optional[str] = None,
                              save_indices: Optional[bool] = False) -> None:
    if not filename:
        filename = 'no_name_data_with_indicators'

    filepath = PACKAGE_ROOT.parent / f'Data/DataWithIndicators/{filename}.csv'
    df.to_csv(filepath, index=save_indices)

    return filepath


def add_indicators_to_raw(filepath: str, config: dict, has_headers: Optional[bool] = False,
                          save_to_disk: Optional[bool] = False, file_save_name: Optional[str] = None,
                          datetime_col: Optional[int] = -1, index_col: Optional[int] = None) -> pd.DataFrame:
    # edit column names for raw data from different brokers
    if not has_headers:
        raw_data = pd.read_csv(filepath, index_col=index_col,
                               names=('Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'))
    else:
        raw_data = pd.read_csv(filepath, index_col=index_col)

    if datetime_col == -1:
        add_datetimes(raw_data)
    else:
        raw_data = raw_data.astype({datetime_col: 'datetime64[ns]'})

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
            raw_data.to_csv(PACKAGE_ROOT.parent / f'Data/DataWithIndicators/{file_save_name}.csv', index=False)
        else:
            raw_data_file_name = PurePath(filepath).name
            if '.' in raw_data_file_name:
                x = raw_data_file_name[::-1].find('.')
                no_ext = raw_data_file_name[:len(raw_data_file_name) - x - 1]

            raw_data.to_csv(PACKAGE_ROOT.parent / f'Data/DataWithIndicators/{no_ext}-{current_model}-model.csv',
                            index=False)

    return raw_data


def add_datetimes(raw_data: pd.DataFrame) -> None:
    datetimes = list(map(lambda row: datetime.strptime(f'{row.Date} {row.Time}', '%Y.%m.%d %H:%M'),
                         raw_data.itertuples()))
    raw_data['datetime'] = datetimes


# is just copy of Close column 26 periods in the past on default settings
def add_chikou_span(raw_data: pd.DataFrame, delay_periods: int = 26) -> None:
    c_span = raw_data['Close'].to_list()[delay_periods:]
    c_span.extend([None] * delay_periods)
    raw_data['chikou_span'] = c_span


def add_ichimoku_cloud(df: pd.DataFrame, chikou_period: int = 26, tenkan_period: int = 9, kijun_period: int = 26,
                       senkou_b_period: int = 52) -> None:
    indicator = ta.trend.IchimokuIndicator(high=df['High'], low=df['Low'], window1=tenkan_period, window2=kijun_period,
                                           window3=senkou_b_period, visual=False)
    df['trend_ichimoku_conv'] = indicator.ichimoku_conversion_line()
    df['trend_ichimoku_base'] = indicator.ichimoku_base_line()
    df['trend_ichimoku_a'] = indicator.ichimoku_a()
    df['trend_ichimoku_b'] = indicator.ichimoku_b()

    indicator = ta.trend.IchimokuIndicator(high=df['High'], low=df['Low'], window1=tenkan_period, window2=kijun_period,
                                           window3=senkou_b_period, visual=True)
    df['trend_visual_ichimoku_a'] = indicator.ichimoku_a()
    df['trend_visual_ichimoku_b'] = indicator.ichimoku_b()

    # Chikou Span trendline of Ichimoku is not apart of 'ta' package
    add_chikou_span(df, chikou_period)


def add_rsi(df: pd.DataFrame, periods: int = 14) -> None:
    df['momentum_rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=periods).rsi()


def download_mt5_data(symbol, resolution, start_time, end_time, mt5_initialized=False, filepath=None):
    time_frames = {
        'm1': mt5.TIMEFRAME_M1,
        'm2': mt5.TIMEFRAME_M2,
        'm3': mt5.TIMEFRAME_M3,
        'm4': mt5.TIMEFRAME_M4,
        'm5': mt5.TIMEFRAME_M5,
        'm6': mt5.TIMEFRAME_M6,
        'm10': mt5.TIMEFRAME_M10,
        'm12': mt5.TIMEFRAME_M12,
        'm15': mt5.TIMEFRAME_M15,
        'm20': mt5.TIMEFRAME_M20,
        'm30': mt5.TIMEFRAME_M30,
        'h1': mt5.TIMEFRAME_H1,
        'h2': mt5.TIMEFRAME_H2,
        'h3': mt5.TIMEFRAME_H3,
        'h4': mt5.TIMEFRAME_H4,
        'h6': mt5.TIMEFRAME_H6,
        'h8': mt5.TIMEFRAME_H8,
        'h12': mt5.TIMEFRAME_H12,
        'w1': mt5.TIMEFRAME_W1,
        'mn1': mt5.TIMEFRAME_MN1,
    }

    if resolution.lower() not in time_frames:
        print(f'"{resolution}" is not a valid chart time frame')
        return
    resolution = resolution.lower()
    time_frame = time_frames[resolution]

    if not mt5_initialized:
        res = mt5.initialize(portable=True)
        if not res:
            print('failed to initialize MT5 terminal')
            return

    start_time = datetime.fromisoformat(start_time)
    start_time = datetime(start_time.year, start_time.month, start_time.day, hour=start_time.hour,
                          minute=start_time.minute, second=start_time.second, tzinfo=timezone.utc)
    cur_start_time = start_time

    end_time = datetime.fromisoformat(end_time)
    end_time = datetime(end_time.year, end_time.month, end_time.day, hour=end_time.hour, minute=end_time.minute,
                        second=end_time.second, tzinfo=timezone.utc)

    dt_save_form = '%Y-%m-%dT%H;%M%Z'
    if not filepath:
        filepath = PACKAGE_ROOT.parent / f'Data/RawData/mt5_{symbol}_{resolution}_ticks_' \
                                             f'{start_time.strftime(dt_save_form)}_to_{end_time.strftime(dt_save_form)}.csv'
    else:
        filepath = Path(filepath).resolve()

    if not filepath.parent.is_dir():
        print(f'will be unable to save to {filepath} because {filepath.parent} does not exist, returning')
        return

    ticks_cache_path = Path(PACKAGE_ROOT.parent / f'Data/.cache/mt5_{symbol}_{resolution}_ticks_'
                                                  f'{start_time.strftime(dt_save_form)}_to_{end_time.strftime(dt_save_form)}.csv')

    if not ticks_cache_path.parent.is_dir():
        print(f'{ticks_cache_path.parent} does not exist, returning')
        return

    if ticks_cache_path.is_file():
        ticks_df = pd.read_csv(ticks_cache_path, index_col=None)
        print(f'loaded {len(ticks_df)} rows of tick data from {ticks_cache_path}')
    else:
        time_skip = timedelta(days=365)
        retries = 0
        files_to_merge = []
        while cur_start_time < end_time:
            cur_end_time = cur_start_time + time_skip
            cur_end_time = min((cur_end_time - cur_start_time, cur_end_time), (end_time - cur_start_time, end_time),
                               key=lambda tup: tup[0])[1]

            cur_ticks_batch = mt5.copy_rates_range(symbol, time_frame, cur_start_time, cur_end_time)
            if cur_ticks_batch is not None:
                if len(cur_ticks_batch) == 0:
                    print(f'no tick data returned by mt5 terminal for time range of {cur_start_time} '
                          f'to {cur_end_time}')

            res = mt5.last_error()
            if res[0] == 1:
                print(f'successfully retrieved {len(cur_ticks_batch)} rows of tick data '
                      f'for time range of {cur_start_time} to {cur_end_time}')
            # "No IPC connection" error
            elif res[0] == -10004:
                print(f'{res} lost connection to mt5 terminal')

                if retries < 3:
                    print('retrying...')
                    if not mt5.initialize(portable=True):
                        print('failed to initialize MT5 terminal')
                        return
                    retries += 1
                    continue
            # any other mt5 error: https://www.mql5.com/en/docs/integration/python_metatrader5/mt5lasterror_py
            else:
                print(f'failed to retrieve tick data from MT5 terminal for {symbol} {resolution} data for '
                      f'time range of {cur_start_time} to {cur_end_time}')
                print('error from mt5:')
                print(res)
                if retries < 3:
                    retries += 1
                    print('retrying...')
                    continue

            # temporarily save data from mt5 terminal because it can sometimes run out of memory if RAM is close to max
            if cur_ticks_batch is not None and len(cur_ticks_batch) > 0:
                temp_cache_path = Path(PACKAGE_ROOT.parent / f'Data/.cache/temp_mt5_{symbol}_{resolution}_ticks_'
                                                             f'{cur_start_time.strftime(dt_save_form)}_to_{cur_end_time.strftime(dt_save_form)}.npy')
                np.save(temp_cache_path, cur_ticks_batch)
                files_to_merge.append(temp_cache_path)

            cur_start_time += time_skip
            retries = 0

        ticks = None
        if len(files_to_merge) > 0:
            ticks = np.load(files_to_merge[0])
            if len(files_to_merge) > 1:
                print('starting to concatenate all downloaded data...')
                for i, file_path in enumerate(files_to_merge[1:]):
                    ticks_to_append = np.load(file_path)
                    ticks = np.append(ticks, ticks_to_append, axis=0)
                    print(f'concatenated {i + 2}/{len(files_to_merge)} downloaded datasets')

            for file_path in files_to_merge:
                os.remove(file_path)
        else:
            print('no tick data retrieved, done.')
            return

        ticks = ticks.tolist()
        # row[1:-1] should get all tick data besides the date and 'real_volume' column
        formatted_ticks = [(datetime.utcfromtimestamp(row[0]), *row[1:-1]) for row in ticks]
        ticks_df = pd.DataFrame(formatted_ticks, columns=['datetime', 'Open', 'High', 'Low',
                                                          'Close', 'Volume', 'spread'])

        # remove any duplicate datetimes (most likely caused by using timedelta)
        ticks_df.drop_duplicates(subset=['datetime'], inplace=True)

        ticks_df.to_csv(ticks_cache_path, index=False)
        print(f'saved {len(ticks_df)} rows of tick data to {ticks_cache_path}')

    ticks_df.to_csv(filepath, index=False)
    print(f'saved {ticks_df.shape[0]} rows of {symbol} {resolution} tick data to {filepath}, done.')

    if not mt5_initialized:
        mt5.shutdown()

    return filepath


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('filepath')
    parser.add_argument('-s', '--save', dest='save_to_disk', action='store_true')
    parser.add_argument('-hh', '--headers', dest='has_headers', action='store_false')
    parser.add_argument('-dc', '--date_col', dest='datetime_col', type=int, default=-1)
    parser.add_argument('-ic', '--idx_col', dest='index_col', type=int, default=None)
    parser.add_argument('-n', '--name', dest='file_save_name')
    parser.add_argument('-c', '--config', dest='config_path')
    args = parser.parse_args()

    filepath = args.filepath
    save_to_disk = args.save_to_disk
    has_headers = args.has_headers
    datetime_col = args.datetime_col
    index_col = args.index_col
    file_save_name = args.file_save_name
    config_path = args.config_path

    config = yaml_to_dict()  # config = yaml_to_dict(config_path)
    print(config)

    add_indicators_to_raw(filepath=filepath, config=config, save_to_disk=save_to_disk, file_save_name=file_save_name,
                          has_headers=has_headers, datetime_col=datetime_col, index_col=index_col)
