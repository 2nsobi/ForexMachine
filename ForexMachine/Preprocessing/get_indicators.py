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
import time
import re
import numpy as np
import shutil


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


def download_mt5_data(symbol, resolution, start_time, end_time, mt5_initialized=False, filepath=None, overwrite=False):
    match = re.match(r'(\w)(\d+)', resolution, re.ASCII)
    if not match or len(match.group(0)) != len(resolution):
        print(f'"{resolution}" is not a valid resolution')
        return

    time_period = match.group(1).lower()
    sub_periods_per_tick = int(match.group(2))
    sub_period_count = 0
    time_period_funcs = {
        'm': lambda x: x.minute,
        'h': lambda x: x.hour,
        'd': lambda x: x.day,
        'w': lambda x: x.weekday() // 6 + sub_period_count
    }
    if time_period not in time_period_funcs or sub_periods_per_tick < 1:
        print(f'"{resolution}" is not a valid resolution')
        return
    get_sub_period = time_period_funcs[time_period]

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
    default_filepath = PACKAGE_ROOT.parent / f'Data/RawData/mt5_{symbol}_{resolution}_ticks_' \
                                             f'{start_time.strftime(dt_save_form)}_to_{end_time.strftime(dt_save_form)}.csv'
    if not filepath:
        filepath = default_filepath
    else:
        filepath = Path(filepath).resolve()

    if not filepath.parent.is_dir():
        print(f'will be unable to save to {filepath} because {filepath.parent} does not exist, returning')
        return

    if default_filepath.is_file() and filepath == default_filepath and not overwrite:
        print(f'{symbol} {resolution} tick data already saved at {default_filepath}')
    elif default_filepath.is_file() and filepath != default_filepath and not overwrite:
        shutil.copy(default_filepath, filepath)
        print(f'copied already saved {symbol} {resolution} tick data from {default_filepath} to {filepath}')
    else:
        ticks_numpy = Path(PACKAGE_ROOT.parent / f'Data/.NumpyData/mt5_{symbol}_{resolution}_ticks_'
                                                 f'{start_time.strftime(dt_save_form)}_to_{end_time.strftime(dt_save_form)}.npy')

        if not ticks_numpy.parent.is_dir():
            print(f'{ticks_numpy.parent} does not exist, returning')
            return

        if ticks_numpy.is_file():
            ticks = np.load(ticks_numpy)
            print(f'loaded {len(ticks)} rows of tick data from {ticks_numpy}')
        else:
            ticks = None
            a_year = timedelta(days=365)
            while cur_start_time < end_time:
                cur_end_time = cur_start_time + a_year
                cur_end_time = min((cur_end_time - cur_start_time, cur_end_time), (end_time - cur_start_time, end_time),
                                   key=lambda tup: tup[0])[1]

                # "The first call of CopyTicks() initiates synchronization of the symbol's tick database stored
                # on the hard disk. CopyTicks() is the mql C function that is called from inside copy_ticks_range()
                # probably, so just call it 3 times to be safe
                for i in range(3):
                    cur_ticks_batch = mt5.copy_ticks_range(symbol, cur_start_time, cur_end_time, mt5.COPY_TICKS_ALL)
                    if cur_ticks_batch is not None:
                        if len(cur_ticks_batch) == 0:
                            print(
                                f'no tick data returned by mt5 terminal for time range of {cur_start_time} to {cur_end_time}')
                        break
                    time.sleep(0.5)

                res = mt5.last_error()
                if res[0] == 1:
                    print(f'successfully retrieved {len(cur_ticks_batch)} rows of tick data '
                          f'for time range of {cur_start_time} to {cur_end_time}')
                # "No IPC connection" error
                elif res[0] == -10004:
                    print(f'{res} lost connection to mt5 terminal, retrying...')
                    if not mt5.initialize(portable=True):
                        print('failed to initialize MT5 terminal')
                        return
                    continue
                else:
                    print(f'{res} failed to download tick data for time range of {cur_start_time} to {cur_end_time}')

                if cur_ticks_batch is not None and len(cur_ticks_batch) > 0:
                    if ticks is None:
                        ticks = cur_ticks_batch
                    else:
                        # the first axis is the outer most grouping (rows when working with a 2d array)
                        ticks = np.append(ticks, cur_ticks_batch, axis=0)

                cur_start_time += a_year

            np.save(ticks_numpy, arr=ticks)
            print(f'saved {len(ticks)} rows of tick data to {ticks_numpy}')

            if not mt5_initialized:
                mt5.shutdown()

        # for d in [(datetime.utcfromtimestamp(ticks[i][0]).isoformat(), ticks[i][1]) for i in range(20)]:
        #     print(d)

        def utc_dt_no_seconds(d):
            return datetime(d.year, d.month, d.day, hour=d.hour, minute=d.minute, tzinfo=timezone.utc)

        column_idx = {name: i for i, name in enumerate(ticks.dtype.names)}
        last_sub_period = None
        last_dt = datetime.utcfromtimestamp(ticks[0][column_idx['time']])
        new_ticks = []
        last_open = last_high = last_low = last_bid = ticks[0][column_idx['bid']]

        zero_open = True
        if last_open != 0:
            zero_open = False

        print('starting parsing all tick data...')
        last_percent = 0

        for i, tick in enumerate(ticks):
            dt = datetime.utcfromtimestamp(tick[column_idx['time']])
            sub_period = get_sub_period(dt)
            bid = tick[column_idx['bid']]

            if last_sub_period is not None and sub_period != last_sub_period:
                sub_period_count += 1

                if sub_period_count % sub_periods_per_tick == 0:
                    if not (last_open == last_bid == 0):
                        new_ticks.append((utc_dt_no_seconds(last_dt), last_open, last_high, last_low, last_bid))

                    last_open = last_high = last_low = last_bid = bid
                    if last_open == 0:
                        zero_open = True

                    last_dt = dt

            if bid != 0:
                if zero_open:
                    last_open = last_high = last_low = last_bid = bid
                    zero_open = False
                else:
                    if bid > last_high:
                        last_high = bid
                    if bid < last_low:
                        last_low = bid
                    last_bid = bid

            last_sub_period = sub_period

            percent_done = int(((i + 1) / len(ticks) * 100))
            if percent_done != last_percent and percent_done % 10 == 0:
                print(f'done parsing {percent_done}% of tick data')
            last_percent = percent_done

        new_ticks.append((utc_dt_no_seconds(last_dt), last_open, last_high, last_low, last_bid))

        data = pd.DataFrame(new_ticks, columns=['datetime', 'Open', 'High', 'Low', 'Close'])
        data.to_csv(filepath, index=False)

        print(f'saved {data.shape[0]} rows of {symbol} {resolution} tick data to {filepath}')

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
