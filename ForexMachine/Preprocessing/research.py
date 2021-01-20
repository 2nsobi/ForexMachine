import ta
import pandas as pd
from pathlib import Path
from typing import Optional
import MetaTrader5 as mt5
import os
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import numpy as np
from ForexMachine import util
from collections import namedtuple
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

TRAIN_DATA_START_ISO, TRAIN_DATA_END_ISO = '2011-01-01', '2020-10-01'


class FeatuteGenerator:
    def __init__(self, data, feature_indices):
        self.data = data
        self.feature_indices = feature_indices
        self.cross_lengths = {}
        self.temporal_features = namedtuple('temporal_features', 'quarter year month day day_of_week hour minute')
        self.safe_start_idx = self._end_of_missing_data_idx(['chikou_span_visual'])

    def get_temporal_features(self, i):
        dt = self.data[i][self.feature_indices['datetime']]

        features = self.temporal_features(quarter=dt.quarter, year=dt.year, month=dt.month, day=dt.day,
                                          day_of_week=dt.dayofweek, hour=dt.hour, minute=dt.minute)
        return features

    def get_ichimoku_features(self, i, cross_length_limit=1):
        is_price_above_cb_lines = None
        is_price_above_cloud = None
        is_price_inside_cloud = None
        is_price_below_cloud = None

        # cross signals represented as tuples: (bullish strength, bearish strength, cross length)
        # - cross signal strength indicated by 0, 1, 2, 3 for none, weak, neutral, strong
        #    or just 0, 1, 3 for none, weak, strong
        # - cross length is just the number of ticks the cross occured over
        tk_cross = (0, 0, 0)
        tk_price_cross = (0, 0, 0)
        senkou_cross = (0, 0, 0)
        chikou_cross = (0, 0, 0)
        cloud_breakout_bull = False
        cloud_breakout_bear = False

        close = self.feature_indices['Close']
        trend_visual_ichimoku_a = self.feature_indices['trend_visual_ichimoku_a']
        trend_visual_ichimoku_b = self.feature_indices['trend_visual_ichimoku_b']
        trend_ichimoku_a = self.feature_indices['trend_ichimoku_a']
        trend_ichimoku_b = self.feature_indices['trend_ichimoku_b']
        trend_ichimoku_conv = self.feature_indices['trend_ichimoku_conv']
        trend_ichimoku_base = self.feature_indices['trend_ichimoku_base']
        chikou_span = self.feature_indices['chikou_span']

        cloud_top, cloud_bottom = self._get_top_and_bottom_line_idx(trend_visual_ichimoku_a, trend_visual_ichimoku_b, i)

        if not pd.isna(self.data[i][trend_ichimoku_conv]) and not pd.isna(self.data[i][trend_ichimoku_base]):
            if self.data[i][close] > self.data[i][trend_ichimoku_conv] \
                    and self.data[i][close] > self.data[i][trend_ichimoku_base]:
                is_price_above_cb_lines = True
            else:
                is_price_above_cb_lines = False

            if self._is_line_between_region(close, cloud_top, cloud_bottom, i):
                is_price_inside_cloud = True
                is_price_above_cloud = False
                is_price_below_cloud = False
            else:
                is_price_inside_cloud = False
                if self.data[i][close] <= self.data[i][cloud_bottom]:
                    is_price_above_cloud = False
                    is_price_below_cloud = True
                else:
                    is_price_above_cloud = True
                    is_price_below_cloud = False

        ### check for crosses

        if i >= self.safe_start_idx:

            ### tk cross

            cross, length, top_line_i, bottom_line_i = \
                self._get_cross_and_length('tk_cross', trend_ichimoku_conv, trend_ichimoku_base, i)

            # price cross clean through both tk region (cross == 2), or price cross through both
            # tk region over limited amout of ticks (cross == 3 and length <= cross_length_limit)
            if cross == 2 \
                    or (cross == 3 and length <= cross_length_limit):

                # bullish
                if top_line_i == trend_ichimoku_conv:
                    if self._is_line_between_region(top_line_i, cloud_top, cloud_bottom, i) \
                            and self._is_line_between_region(bottom_line_i, cloud_top, cloud_bottom, i):
                        tk_cross = (2, 0, length)
                    elif self.data[i][bottom_line_i] >= self.data[i][cloud_top]:
                        tk_cross = (3, 0, length)
                    else:
                        tk_cross = (1, 0, length)
                # bearish
                elif top_line_i == trend_ichimoku_base:
                    if self._is_line_between_region(top_line_i, cloud_top, cloud_bottom, i) \
                            and self._is_line_between_region(bottom_line_i, cloud_top, cloud_bottom, i):
                        tk_cross = (0, 2, length)
                    elif self.data[i][top_line_i] <= self.data[i][cloud_bottom]:
                        tk_cross = (0, 3, length)
                    else:
                        tk_cross = (0, 1, length)
                else:
                    print('weird 5:', self.data[i][self.feature_indices['datetime']])

            ### tk price cross

            cross_res = self._get_cross_and_length_regions('tk_price_cross', trend_ichimoku_conv, trend_ichimoku_base,
                                                           close, close, i)
            cross, length, first_line, second_line, third_line, fourth_line = cross_res

            if cross == 2 or (cross == 3 and length <= cross_length_limit):

                # "It’s a noise zone when price is in the Cloud"
                #  https://www.tradeciety.com/the-complete-ichimoku-trading-guide-how-to-use-the-ichimoku-indicator/

                # bullish
                if first_line == close:
                    if self.data[i][close] >= self.data[i][cloud_top]:
                        tk_price_cross = (3, 0, length)
                    elif self.data[i][close] <= self.data[i][cloud_bottom]:
                        tk_price_cross = (1, 0, length)
                # bearish
                elif fourth_line == close:
                    if self.data[i][close] >= self.data[i][cloud_top]:
                        tk_price_cross = (0, 1, length)
                    elif self.data[i][close] <= self.data[i][cloud_bottom]:
                        tk_price_cross = (0, 3, length)
            elif cross == 3 and length > cross_length_limit:
                print(
                    f'cross type = {cross}, cross length = {length}, {self.data[i][self.feature_indices["datetime"]]}')

            ### cloud (senkou) cross

            # As the Senkou Spans are projected forward, the cross that triggers this signal will be 26 days ahead of the
            # price and, hence, the actual date that the signal occurs.  The strength of the signal is determined by the
            # relationship of the price on the date of the signal (not the trigger) to the Kumo (Cloud)
            # - https://www.ichimokutrader.com/signals.html

            cross, length, top_line_i, bottom_line_i = \
                self._get_cross_and_length('cloud_cross', trend_ichimoku_a, trend_ichimoku_b, i)

            if cross == 2 \
                    or (cross == 3 and length <= cross_length_limit):

                # bullish
                if top_line_i == trend_ichimoku_a:
                    if self._is_line_between_region(close, cloud_top, cloud_bottom, i):
                        senkou_cross = (2, 0, length)
                    elif self.data[i][close] >= self.data[i][cloud_top]:
                        senkou_cross = (3, 0, length)
                    else:
                        senkou_cross = (1, 0, length)
                # bearish
                elif top_line_i == trend_ichimoku_b:
                    if self._is_line_between_region(close, cloud_top, cloud_bottom, i):
                        senkou_cross = (0, 2, length)
                    elif self.data[i][close] <= self.data[i][cloud_bottom]:
                        senkou_cross = (0, 3, length)
                    else:
                        senkou_cross = (0, 1, length)
                else:
                    print('weird 55:', self.data[i][self.feature_indices['datetime']])

            ### chikou span cross

            # Note (1) that the Chikou Span must be rising when it crosses to above the price for a bull signal
            # and falling when it crosses to below for a bear signal; just crossing the price alone is not
            # sufficient to trigger the signal. (2) As the Chikou Span is the closing price shifted into the past,
            # the cross that triggers this signal will be 26 days behind the price and, hence, the actual date
            # that the signal occurs.The strength of the signal is determined by the relationship of the price
            # on the date of the signal (not the trigger) to the Kumo (Cloud).
            # - https://www.ichimokutrader.com/signals.html

            # remember the chikou_span at this point is just the price 26 (or whatever chikou/senkou projection is) ticks ago
            cross, length, top_line_i, bottom_line_i = \
                self._get_cross_and_length('chikou_cross', chikou_span, close, i)

            if cross == 2 \
                    or (cross == 3 and length <= cross_length_limit):
                # bullish
                if top_line_i == close:
                    if self._is_line_between_region(close, cloud_top, cloud_bottom, i):
                        chikou_cross = (2, 0, length)
                    elif self.data[i][close] > self.data[i][cloud_top]:
                        chikou_cross = (3, 0, length)
                    else:
                        chikou_cross = (1, 0, length)
                # bearish
                elif top_line_i == chikou_span:
                    if self._is_line_between_region(close, cloud_top, cloud_bottom, i):
                        chikou_cross = (0, 2, length)
                    elif self.data[i][close] < self.data[i][cloud_bottom]:
                        chikou_cross = (0, 3, length)
                    else:
                        chikou_cross = (0, 1, length)
                else:
                    print('weird 6:', self.data[i][self.feature_indices['datetime']])

            ### kumo (cloud) breakout

            cross_res = self._get_cross_and_length_regions('kumo_breakout', cloud_top, cloud_bottom,
                                                           close, close, i)
            cross, length, first_line, second_line, third_line, fourth_line = cross_res

            # The Kumo Breakout signal occurs when the price leaves or crosses the Kumo (Cloud), which is why
            # we also want to check for if cross == 4 (end of overlap but not a cross)
            # - https://www.ichimokutrader.com/signals.html
            if cross == 2 or cross == 3 or cross == 4:
                # bullish
                if first_line == close:
                    cloud_breakout_bull = True
                # bearish
                elif fourth_line == close:
                    cloud_breakout_bear = True

        features = {
            'is_price_above_cb_lines': is_price_above_cb_lines,
            'is_price_above_cloud': is_price_above_cloud,
            'is_price_inside_cloud': is_price_inside_cloud,
            'is_price_below_cloud': is_price_below_cloud,
            'cloud_top': cloud_top,
            'cloud_bottom': cloud_bottom,
            'tk_cross': tk_cross,
            'tk_price_cross': tk_price_cross,
            'senkou_cross': senkou_cross,
            'chikou_cross': chikou_cross,
            'cloud_breakout_bull': cloud_breakout_bull,
            'cloud_breakout_bear': cloud_breakout_bear
        }

        return features

    def _end_of_missing_data_idx(self, exluded_features):
        exluded_features = set(exluded_features)
        safe_idx = None

        for i in range(len(self.data)):
            nan_in_row = False

            for feature in self.feature_indices:
                if feature in exluded_features:
                    continue

                feature_i = self.feature_indices[feature]
                if isinstance(self.data[i][feature_i], float) and np.isnan(self.data[i][feature_i]):
                    safe_idx = None
                    nan_in_row = True
                    break

            if not nan_in_row and not safe_idx:
                safe_idx = i

        # add 1 because we are looking for crosses and need to look back one tick in order to do so
        return safe_idx + 1

    def _get_top_and_bottom_line_idx(self, line1_i, line2_i, i):
        """
        line1_i is top if line values are equal
        """
        top_line_i = line1_i
        bottom_line_i = line2_i
        if self.data[i][line1_i] < self.data[i][line2_i]:
            top_line_i = line2_i
            bottom_line_i = line1_i
        return top_line_i, bottom_line_i

    def _is_line_between_region(self, target_line_i, top_line_i, bottom_line_i, i):
        if self.data[i][target_line_i] > self.data[i][bottom_line_i] \
                and self.data[i][target_line_i] < self.data[i][top_line_i]:
            return True
        return False

    def _get_cross_and_length_regions(self, cross_name, r1_line1, r1_line2, r2_line1, r2_line2, i):
        """
        cross type can be: no cross '=' (0), start of overlap '>' (1), full cross 'X' (2), end of cross '<' (3),
            or end of overlap w/ no cross (4)
        """

        old_r1_top, old_r1_bot = self._get_top_and_bottom_line_idx(r1_line1, r1_line2, i - 1)
        old_r2_top, old_r2_bot = self._get_top_and_bottom_line_idx(r2_line1, r2_line2, i - 1)

        r1_top, r1_bot = self._get_top_and_bottom_line_idx(r1_line1, r1_line2, i)
        r2_top, r2_bot = self._get_top_and_bottom_line_idx(r2_line1, r2_line2, i)

        # defines lines from top to bottom between both regions
        sorted_regions_lines = sorted([(line, self.data[i][line]) for line in [r1_top, r1_bot, r2_top, r2_bot]],
                                      key=lambda line_tuple: line_tuple[1], reverse=True)
        first_line, second_line, third_line, fourth_line = sorted_regions_lines

        ### check for no cross

        old_top_region_bot = None

        # region 1 is fully above region 2
        if self.data[i - 1][old_r1_bot] > self.data[i - 1][old_r2_top]:
            old_top_region_bot = old_r1_bot
            if self.data[i][r1_bot] > self.data[i][r2_top]:
                return 0, 0, first_line[0], second_line[0], third_line[0], fourth_line[0]
        # region 2 is fully above region 1
        elif self.data[i - 1][old_r2_bot] > self.data[i - 1][old_r1_top]:
            old_top_region_bot = old_r2_bot
            if self.data[i][r2_bot] > self.data[i][r1_top]:
                return 0, 0, first_line[0], second_line[0], third_line[0], fourth_line[0]

        ### check for full cross

        # region 1 crossed to below region 2
        if self.data[i - 1][old_r1_bot] > self.data[i - 1][old_r2_top] \
                and self.data[i][r1_top] < self.data[i][r2_bot]:
            return 2, 0, first_line[0], second_line[0], third_line[0], fourth_line[0]

        # region 2 crossed to below region 1
        elif self.data[i - 1][old_r2_bot] > self.data[i - 1][old_r1_top] \
                and self.data[i][r2_top] < self.data[i][r1_bot]:
            return 2, 0, first_line[0], second_line[0], third_line[0], fourth_line[0]

        ### check for start of overlap

        top_region_top = top_region_bot = bot_region_top = bot_region_bot = None
        # region 1 is highest
        if self.data[i][r1_top] > self.data[i][r2_top]:
            top_region_top = r1_top
            top_region_bot = r1_bot
            bot_region_top = r2_top
            bot_region_bot = r2_bot
        # region 2 is highest
        else:
            top_region_top = r2_top
            top_region_bot = r2_bot
            bot_region_top = r1_top
            bot_region_bot = r1_bot

        # checking for start of overlap
        if cross_name not in self.cross_lengths:
            # if the bottom line of the top region is still not defined then just consider no cross
            if not old_top_region_bot:
                return 0, 0, first_line[0], second_line[0], third_line[0], fourth_line[0]
            else:
                # one region is beginning to intertwine or completely swallow the other, regardless this counts
                # as the start of an overlap
                if self.data[i][bot_region_top] <= self.data[i][top_region_top] \
                        and self.data[i][bot_region_top] >= self.data[i][top_region_bot]:
                    self.cross_lengths[cross_name] = (0, old_top_region_bot)
                    return 1, 0, first_line[0], second_line[0], third_line[0], fourth_line[0]
                print('weird 11:', self.data[i][self.feature_indices['datetime']])
        else:
            # check for continuation of overlap
            if self.data[i][bot_region_top] <= self.data[i][top_region_top] \
                    and self.data[i][bot_region_top] >= self.data[i][top_region_bot]:
                self.cross_lengths[cross_name] = (
                    self.cross_lengths[cross_name][0] + 1, self.cross_lengths[cross_name][1])
                return 0, self.cross_lengths[cross_name][0], first_line[0], second_line[0], third_line[0], fourth_line[
                    0]
            # otherwise, 1 region must be completely above the other
            else:
                original_top_region_bot = self.cross_lengths[cross_name][1]
                res = None

                # check for end of cross
                if original_top_region_bot != top_region_bot and original_top_region_bot != top_region_top:
                    res = 3
                # otherwise, end of overlap w/ no cross
                else:
                    res = 4

                cross_length = self.cross_lengths[cross_name][0]
                del self.cross_lengths[cross_name]
                return res, cross_length, first_line[0], second_line[0], third_line[0], fourth_line[0]

    def _get_cross_and_length(self, cross_name, line_index1, line_index2, index):
        """
        cross type can be: no cross '=' (0), start of overlap '>' (1), full cross 'X' (2), end of cross '<' (3),
            or end of overlap w/ no cross (4)
        """

        # remember that if lines are of equal values the first line argument to _get_top_and_bottom_line_idx()
        # will be returned as the top line
        old_top_line_i, old_bottom_line_i = self._get_top_and_bottom_line_idx(line_index1, line_index2, index - 1)
        top_line_i, bottom_line_i = self._get_top_and_bottom_line_idx(line_index1, line_index2, index)

        ## check for no cross

        if self.data[index][line_index1] != self.data[index][line_index2] \
                and self.data[index - 1][line_index1] != self.data[index - 1][line_index2] \
                and old_top_line_i == top_line_i \
                and bottom_line_i == old_bottom_line_i:
            return 0, 0, top_line_i, bottom_line_i

        ## check for full cross

        if old_top_line_i != top_line_i \
                and self.data[index - 1][old_top_line_i] > self.data[index - 1][old_bottom_line_i] \
                and self.data[index][old_top_line_i] < self.data[index][old_bottom_line_i]:
            return 2, 0, top_line_i, bottom_line_i

        ##check for start of overlap

        if cross_name not in self.cross_lengths:
            if self.data[index][line_index1] == self.data[index][line_index2]:
                self.cross_lengths[cross_name] = (0, old_top_line_i, self.data[index][self.feature_indices['datetime']])
                return 1, 0, top_line_i, bottom_line_i
            print('weird 1:', self.data[index][self.feature_indices['datetime']])
        else:

            ## check for continuation of overlap

            if self.data[index][line_index1] == self.data[index][line_index2]:
                self.cross_lengths[cross_name] = (self.cross_lengths[cross_name][0] + 1,
                                                  self.cross_lengths[cross_name][1], self.cross_lengths[cross_name][2])
                return 0, self.cross_lengths[cross_name][0], top_line_i, bottom_line_i
            else:
                cross_old_top_line_i = self.cross_lengths[cross_name][1]
                res = None

                ## check for end of cross

                if cross_old_top_line_i != top_line_i:
                    res = 3
                # otherwise, end of overlap w/ no cross
                else:
                    res = 4

                cross_length = self.cross_lengths[cross_name][0]
                del self.cross_lengths[cross_name]
                return res, cross_length, top_line_i, bottom_line_i


def add_indicators_to_raw(filepath: str, indicators_info: dict, has_headers: Optional[bool] = True,
                          headers: Optional[list] = None, datetime_col: Optional[str] = None,
                          index_col: Optional[int] = None) -> pd.DataFrame:
    if has_headers:
        raw_data = pd.read_csv(filepath, index_col=index_col)
    else:
        if headers is None:
            headers = ('Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume')
        raw_data = pd.read_csv(filepath, index_col=index_col, names=headers)

    if datetime_col is None:
        add_datetimes(raw_data)
    else:
        raw_data = raw_data.astype({datetime_col: 'datetime64[ns]'})

    indicator_functions = {
        'ichimoku': lambda df, **kwargs: add_ichimoku_cloud(df, **kwargs),
        'rsi': lambda df, **kwargs: add_rsi(df, **kwargs)
    }

    for indicator in indicators_info:
        indicator_functions[indicator](raw_data, **indicators_info[indicator])

    return raw_data


def add_datetimes(raw_data: pd.DataFrame) -> None:
    datetimes = list(map(lambda row: datetime.strptime(f'{row.Date} {row.Time}', '%Y.%m.%d %H:%M'),
                         raw_data.itertuples()))
    raw_data['datetime'] = datetimes


# is just copy of Close column 26 periods in the past on default settings
def add_chikou_span(raw_data: pd.DataFrame, delay_periods: int = 26) -> None:
    c_span = [None] * delay_periods
    c_span.extend(raw_data['Close'].to_list()[:-delay_periods])
    raw_data['chikou_span'] = c_span  # this will just be the value of close @ i - delay_periods

    c_span_vis = raw_data['Close'].to_list()[delay_periods:]
    c_span_vis.extend([None] * delay_periods)
    raw_data['chikou_span_visual'] = c_span_vis


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


def download_mt5_data(symbol, resolution, start_time, end_time, mt5_initialized=False, filepath=None,
                      cache_path=None):
    timeframes = {
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

    if resolution.lower() not in timeframes:
        print(f'"{resolution}" is not a valid chart time frame')
        return
    resolution = resolution.lower()
    time_frame = timeframes[resolution]

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

    if cache_path is None:
        cache_path = util.get_ticks_data_dir()
    else:
        cache_path = Path(cache_path).resolve()

    dt_save_form = '%Y-%m-%dT%H;%M%Z'
    default_filepath = cache_path / f'mt5_{symbol}_{resolution}_ticks_{start_time.strftime(dt_save_form)}' \
                                    f'_to_{end_time.strftime(dt_save_form)}.csv'

    if not default_filepath.parent.is_dir():
        print(f'{default_filepath.parent} does not exist, returning')
        return

    if filepath is None:
        filepath = default_filepath
    else:
        filepath = Path(filepath).resolve()

    if not filepath.parent.is_dir():
        print(f'will be unable to save to {filepath} because {filepath.parent} does not exist, returning')
        return

    if default_filepath.is_file():
        ticks_df = pd.read_csv(default_filepath, index_col=None)
        print(f'loaded {len(ticks_df)} rows of tick data from {default_filepath}')
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
                temp_cache_path = util.get_temp_dir() / f'temp_mt5_{symbol}_{resolution}_ticks_' \
                                                        f'{cur_start_time.strftime(dt_save_form)}_to_{cur_end_time.strftime(dt_save_form)}.npy'
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
        formatted_ticks = [(datetime.fromtimestamp(row[0], tz=timezone.utc), *row[1:-1]) for row in ticks]
        ticks_df = pd.DataFrame(formatted_ticks, columns=['datetime', 'Open', 'High', 'Low',
                                                          'Close', 'Volume', 'spread'])

        # remove any duplicate datetimes (most likely caused by using timedelta)
        ticks_df.drop_duplicates(subset=['datetime'], inplace=True)

        ticks_df.to_csv(default_filepath, index=False)
        print(f'saved {len(ticks_df)} rows of tick data to {default_filepath}')

    if default_filepath != filepath:
        ticks_df.to_csv(filepath, index=False)
        print(f'saved {ticks_df.shape[0]} rows of {symbol} {resolution} tick data to {filepath}')

    if not mt5_initialized:
        mt5.shutdown()

    return filepath


def add_ichimoku_features(df, inplace=False, negative_bears=True, include_most_recent_feats=False):
    ### temporal features

    quarters = []
    days_of_week = []
    months = []
    days = []
    minutes = []
    hours = []
    years = []

    ### ichimoku features

    is_price_above_cb_lines = []
    is_price_above_cloud = []
    is_price_inside_cloud = []
    is_price_below_cloud = []
    cloud_breakout_bull = []
    cloud_breakout_bear = []
    ticks_since_cloud_breakout_bull = []
    ticks_since_cloud_breakout_bear = []

    first_kumo_breakout_bull = False
    first_kumo_breakout_bear = False

    # names of each cross type
    cross_names = ['tk_cross', 'tk_price_cross', 'senkou_cross', 'chikou_cross']
    # dict to hold similar features of each cross type
    crosses_dict = {}
    for name in cross_names:
        crosses_dict[name] = {
            'most_recent_bull_strength': [],
            'most_recent_bear_strength': [],
            'bull_strength': [],
            'bear_strength': [],
            'ticks_since_bull': [],
            'ticks_since_bear': [],
            'most_recent_bull_length': [],
            'most_recent_bear_length': [],
            'bull_length': [],
            'bear_length': [],
            'first_bull': False,
            'first_bear': False
        }

    data = df.to_numpy()
    feature_indices = {df.columns[i]: i for i in range(len(df.columns))}

    fg = FeatuteGenerator(data, feature_indices)
    for i in range(len(data)):
        # get temporal features signals
        temporal_features = fg.get_temporal_features(i)
        quarters.append(temporal_features.quarter)
        days_of_week.append(temporal_features.day_of_week)
        months.append(temporal_features.month)
        days.append(temporal_features.day)
        minutes.append(temporal_features.minute)
        hours.append(temporal_features.hour)
        years.append(temporal_features.year)

        # get ichimoku signals
        ichimoku_features = fg.get_ichimoku_features(i, cross_length_limit=np.Inf)
        is_price_above_cb_lines.append(ichimoku_features['is_price_above_cb_lines'])
        is_price_above_cloud.append(ichimoku_features['is_price_above_cloud'])
        is_price_inside_cloud.append(ichimoku_features['is_price_inside_cloud'])
        is_price_below_cloud.append(ichimoku_features['is_price_below_cloud'])

        # handle kumo breakout
        cloud_breakout_bull.append(ichimoku_features['cloud_breakout_bull'])
        cloud_breakout_bear.append(ichimoku_features['cloud_breakout_bear'])

        if ichimoku_features['cloud_breakout_bull']:
            first_kumo_breakout_bull = True
        if ichimoku_features['cloud_breakout_bear']:
            first_kumo_breakout_bear = True

        if first_kumo_breakout_bull:
            if ichimoku_features['cloud_breakout_bull']:
                ticks_since_cloud_breakout_bull.append(0)
            else:
                ticks_since_cloud_breakout_bull.append(ticks_since_cloud_breakout_bull[-1] + 1)
        else:
            ticks_since_cloud_breakout_bull.append(None)

        if first_kumo_breakout_bear:
            if ichimoku_features['cloud_breakout_bear']:
                ticks_since_cloud_breakout_bear.append(0)
            else:
                ticks_since_cloud_breakout_bear.append(ticks_since_cloud_breakout_bear[-1] + 1)
        else:
            ticks_since_cloud_breakout_bear.append(None)

        # handle other ichimoku cloud crosses
        for cross_name in crosses_dict:
            cross_dict = crosses_dict[cross_name]

            bull_strength, bear_strength, cross_length = ichimoku_features[cross_name]

            if bull_strength > 0:
                cross_dict['first_bull'] = True
            if bear_strength > 0:
                cross_dict['first_bear'] = True

            if cross_dict['first_bull']:
                if bull_strength > 0:
                    cross_dict['most_recent_bull_strength'].append(bull_strength)
                    cross_dict['bull_strength'].append(bull_strength)
                    cross_dict['ticks_since_bull'].append(0)
                    cross_dict['most_recent_bull_length'].append(cross_length)
                    cross_dict['bull_length'].append(cross_length)
                else:
                    cross_dict['most_recent_bull_strength'].append(cross_dict['most_recent_bull_strength'][-1])
                    cross_dict['bull_strength'].append(0)
                    cross_dict['ticks_since_bull'].append(cross_dict['ticks_since_bull'][-1] + 1)
                    cross_dict['most_recent_bull_length'].append(cross_dict['most_recent_bull_length'][-1])
                    cross_dict['bull_length'].append(0)
            else:
                cross_dict['most_recent_bull_strength'].append(None)
                cross_dict['bull_strength'].append(None)
                cross_dict['ticks_since_bull'].append(None)
                cross_dict['most_recent_bull_length'].append(None)
                cross_dict['bull_length'].append(None)

            if cross_dict['first_bear']:
                if bear_strength > 0:
                    if negative_bears:
                        bear_strength *= -1
                    cross_dict['most_recent_bear_strength'].append(bear_strength)
                    cross_dict['bear_strength'].append(bear_strength)
                    cross_dict['ticks_since_bear'].append(0)
                    cross_dict['most_recent_bear_length'].append(cross_length)
                    cross_dict['bear_length'].append(cross_length)
                else:
                    cross_dict['most_recent_bear_strength'].append(cross_dict['most_recent_bear_strength'][-1])
                    cross_dict['bear_strength'].append(0)
                    cross_dict['ticks_since_bear'].append(cross_dict['ticks_since_bear'][-1] + 1)
                    cross_dict['most_recent_bear_length'].append(cross_dict['most_recent_bear_length'][-1])
                    cross_dict['bear_length'].append(0)
            else:
                cross_dict['most_recent_bear_strength'].append(None)
                cross_dict['bear_strength'].append(None)
                cross_dict['ticks_since_bear'].append(None)
                cross_dict['most_recent_bear_length'].append(None)
                cross_dict['bear_length'].append(None)

    if not inplace:
        df = df.copy()

    df['quarter'] = quarters
    df['day_of_week'] = days_of_week
    df['month'] = months
    df['day'] = days
    df['minute'] = minutes
    df['hour'] = hours
    df['year'] = years
    df['is_price_above_cb_lines'] = is_price_above_cb_lines
    df['is_price_above_cloud'] = is_price_above_cloud
    df['is_price_inside_cloud'] = is_price_inside_cloud
    df['is_price_below_cloud'] = is_price_below_cloud
    df['cloud_breakout_bull'] = cloud_breakout_bull
    df['cloud_breakout_bear'] = cloud_breakout_bear

    if include_most_recent_feats:
        df['ticks_since_cloud_breakout_bull'] = ticks_since_cloud_breakout_bull
        df['ticks_since_cloud_breakout_bear'] = ticks_since_cloud_breakout_bear

    for cross_name in crosses_dict:
        df[f'{cross_name}_bull_strength'] = crosses_dict[cross_name]['bull_strength']
        df[f'{cross_name}_bear_strength'] = crosses_dict[cross_name]['bear_strength']
        df[f'{cross_name}_bull_length'] = crosses_dict[cross_name]['bull_length']
        df[f'{cross_name}_bear_length'] = crosses_dict[cross_name]['bear_length']

        if include_most_recent_feats:
            df[f'{cross_name}_most_recent_bull_strength'] = crosses_dict[cross_name]['most_recent_bull_strength']
            df[f'{cross_name}_most_recent_bear_strength'] = crosses_dict[cross_name]['most_recent_bear_strength']
            df[f'{cross_name}_ticks_since_bull'] = crosses_dict[cross_name]['ticks_since_bull']
            df[f'{cross_name}_ticks_since_bear'] = crosses_dict[cross_name]['ticks_since_bear']
            df[f'{cross_name}_most_recent_bull_length'] = crosses_dict[cross_name]['most_recent_bull_length']
            df[f'{cross_name}_most_recent_bear_length'] = crosses_dict[cross_name]['most_recent_bear_length']

    return df


def disregard_rows_with_missing_data(x_df, y_df=None, ignored_x_cols=None, ignored_y_cols=None, seperate_chunks=False):
    if y_df is not None:
        if x_df.shape[0] != y_df.shape[0]:
            print(f'x_df (rows={x_df.shape[0]}) and y_df (rows={y_df.shape[0]}) do not have the same number of rows')
            return

    if ignored_x_cols:
        ignored_x_cols = set([x_df.columns.get_loc(col_name) for col_name in ignored_x_cols])
    else:
        ignored_x_cols = set()

    if y_df is not None:
        if ignored_y_cols:
            ignored_y_cols = set([y_df.columns.get_loc(col_name) for col_name in ignored_y_cols])
        else:
            ignored_y_cols = set()

    wanted_data = []
    cur_x_data = []
    cur_y_data = []

    x_data = x_df.to_numpy()
    if y_df is not None:
        y_data = y_df.to_numpy()

    for i in range(len(x_data)):
        missing_data = False
        for j in range(len(x_data[i])):
            if j not in ignored_x_cols and pd.isnull(x_data[i][j]):
                missing_data = True
                break

        if y_df is not None and not missing_data:
            for j in range(len(y_data[i])):
                if j not in ignored_y_cols and pd.isnull(y_data[i][j]):
                    missing_data = True
                    break

        if not missing_data:
            cur_x_data.append(x_data[i])
            if y_df is not None:
                cur_y_data.append(y_data[i])
        elif seperate_chunks and len(cur_x_data) > 0:
            if y_df is not None:
                wanted_data.append(
                    (pd.DataFrame(cur_x_data, columns=x_df.columns), pd.DataFrame(cur_y_data, columns=y_df.columns)))
            else:
                wanted_data.append((pd.DataFrame(cur_x_data, columns=x_df.columns), None))
            cur_x_data = []
            cur_y_data = []

    if len(cur_x_data) > 0:
        if y_df is not None:
            wanted_data.append(
                (pd.DataFrame(cur_x_data, columns=x_df.columns), pd.DataFrame(cur_y_data, columns=y_df.columns)))
        else:
            wanted_data.append((pd.DataFrame(cur_x_data, columns=x_df.columns), None))

    return wanted_data


def dummy_and_remove_features(data_df, categories_dict={}, cols_to_remove=[], include_defaults=True,
                              keep_datetime=False):
    if include_defaults:
        cd = {
            'quarter': [1, 2, 3, 4],
            'day_of_week': [0, 1, 2, 3, 4]
        }

        cols = {'spread', 'momentum_rsi', 'month', 'day', 'minute', 'hour', 'year', 'chikou_span_visual', 'chikou_span',
                'tk_cross_bull_length', 'tk_cross_bear_length',
                'tk_price_cross_bull_length', 'tk_price_cross_bear_length',
                'senkou_cross_bull_length', 'senkou_cross_bear_length',
                'chikou_cross_bull_length', 'chikou_cross_bear_length',
                'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', }
        # 'trend_ichimoku_base','trend_ichimoku_conv', 'trend_ichimoku_a', 'trend_ichimoku_b'}

        if not keep_datetime:
            cols.add('datetime')

        cd.update(categories_dict)
        categories_dict = cd
        cols_to_remove = set(cols_to_remove) | cols  # prios keys/vals from 2nd arg of | (or) operand
        data_df_cols_set = set(data_df.columns)

        categories_dict = {key: categories_dict[key] for key in categories_dict if key in data_df_cols_set}
        cols_to_remove = [col for col in cols_to_remove if col in data_df_cols_set]

    if len(categories_dict) > 0:
        catagorical_cols = list(categories_dict.keys())
        categories = list(categories_dict.values())

        cols_to_dummy = data_df[catagorical_cols]
        cols_to_dummy_vals = cols_to_dummy.to_numpy()

        dummy_enc = OneHotEncoder(categories=categories, drop='first')
        dummied_vals = dummy_enc.fit_transform(cols_to_dummy_vals).toarray()
        dummy_col_names = dummy_enc.get_feature_names(catagorical_cols)

        dummied_cols_df = pd.DataFrame(dummied_vals, columns=dummy_col_names, index=data_df.index)
        data_df = pd.concat((data_df, dummied_cols_df), axis=1)

        cols_to_remove.extend(catagorical_cols)

    data_df = data_df.drop(cols_to_remove, axis=1)

    return data_df


def convert_class_labels(y_df, to_ints=True, labels_dict=None, to_numpy=False):
    if to_ints:
        if labels_dict is None:
            unique_labels = np.unique(y_df.to_numpy())
            labels_to_int = {unique_labels[i]: i for i in range(len(unique_labels))}
        else:
            labels_to_int = {labels_dict[i]: i for i in labels_dict}

        new_labels = []
        for v in y_df.to_numpy():
            v = v[0]
            new_labels.append(labels_to_int[v])

        if not to_numpy:
            new_labels = pd.DataFrame(new_labels, columns=y_df.columns)
        else:
            new_labels = np.array(new_labels)

        if labels_dict is None:
            labels_dict = {labels_to_int[label]: label for label in labels_to_int}

        return new_labels, labels_dict
    else:
        new_labels = []
        for v in y_df.to_numpy():
            v = v[0]
            new_labels.append(labels_dict[v])

        if not to_numpy:
            new_labels = pd.DataFrame(new_labels, columns=y_df.columns)
        else:
            new_labels = np.array(new_labels)
        return new_labels, labels_dict


def error_rate(y_true_df, y_pred_df):
    if y_true_df.shape[0] != y_pred_df.shape[0]:
        print(
            f'y_true_df (rows={y_true_df.shape[0]}) and y_pred_df (rows={y_pred_df.shape[0]}) do not have the same number of rows')
        return
    d1, d2 = y_true_df.to_numpy(), y_pred_df.to_numpy()
    wrong_indices = []
    for i in range(len(d1)):
        if d1[i] != d2[i]:
            wrong_indices.append(i)
    return len(wrong_indices) / len(d1), wrong_indices


def no_missing_data_idx_range(df, early_ending_cols=[]):
    early_ending_cols = [df.columns.get_loc(col_name) for col_name in early_ending_cols]
    early_ending_cols = set(early_ending_cols)
    data = df.to_numpy()
    start_idx = None
    end_idx = None
    for i in range(len(data)):
        missing_data = False
        for j in range(len(data[i])):
            if j not in early_ending_cols and pd.isnull(data[i][j]):
                missing_data = True
                start_idx = None
                end_idx = None
                break
            elif j in early_ending_cols and pd.isnull(data[i][j]):
                if start_idx and not end_idx:
                    end_idx = i - 1
        if not start_idx and not missing_data:
            start_idx = i
    if not end_idx:
        end_idx = len(data) - 1
    return start_idx, end_idx


def normalize_data(df, train_data, groups=None, normalization_terms=None):
    df = df.copy(deep=True)

    if train_data:
        if not groups:
            print(f'groups must be specified if train_data is true')
            return None

        normalization_terms = {}
        normalized = set()

        for group in groups:
            min_value = min(df[group].min())
            max_value = max(df[group].max())
            dict_val = (min_value, max_value)

            for col in group:
                if min_value != max_value:
                    df[col] = (df[col] - min_value) / (max_value - min_value)
                elif min_value > 1 or min_value < 0:
                    df[col] = [0] * df.shape[0]

                normalization_terms[col] = dict_val
                normalization_terms[df.columns.get_loc(col)] = dict_val

            normalized = normalized.union(group)

        for col in df:
            if col not in normalized and df[col].dtype != bool and pd.api.types.is_numeric_dtype(df[col].dtype):
                min_value = df[col].min()
                max_value = df[col].max()
                dict_val = (min_value, max_value)

                if min_value != max_value:
                    df[col] = (df[col] - min_value) / (max_value - min_value)
                elif min_value > 1 or min_value < 0:
                    df[col] = [0] * df.shape[0]

                normalization_terms[col] = dict_val
                normalization_terms[df.columns.get_loc(col)] = dict_val

    else:
        if not normalization_terms:
            print(f'normalization_terms must be specified if train_data is false')
            return None

        for col in df:
            if col in normalization_terms:
                min_value, max_value = normalization_terms[col]

                if min_value != max_value:
                    df[col] = (df[col] - min_value) / (max_value - min_value)
                elif min_value > 1 or min_value < 0:
                    df[col] = [0] * df.shape[0]

    return df, normalization_terms


def normalize_data_list(row, normalization_terms):
    new_row = []
    for col_i in range(len(row)):
        if col_i in normalization_terms:
            min_value, max_value = normalization_terms[col_i]

            if min_value != max_value:
                normalized = (row[col_i] - min_value) / (max_value - min_value)
            elif min_value > 1 or min_value < 0:
                normalized = 0

            new_row.append(normalized)
        else:
            new_row.append(row[col_i])
    return new_row


def apply_perc_change(df, cols, limit=None):
    df = df.copy(deep=True)
    for col in cols:
        df[col] = df[col].pct_change(limit=limit)
    return df


def apply_perc_change_list(last_row, cur_row, cols_set):
    new_row = []
    for col in range(len(last_row)):
        if col in cols_set:
            pc = (cur_row[col] / last_row[col]) - 1
            new_row.append(pc)
        else:
            new_row.append(cur_row[col])
    return new_row


def apply_moving_avg(df, cols, window):
    df = df.copy(deep=True)
    df[cols] = df[cols].rolling(window).mean()
    return df


def apply_moving_avg_q(q, cols_set):
    n_rows, n_cols = len(q), len(q[0])
    new_row = []
    for col in range(n_cols):
        if col in cols_set:
            avg = sum([row[col] for row in q]) / n_rows
            new_row.append(avg)
        else:
            new_row.append(q[-1][col])
    return new_row


def missing_labels_preprocess(x_df, y_df, y_col):
    if y_df is not None:
        res = disregard_rows_with_missing_data(x_df, pd.DataFrame(y_df[y_col]))
    else:
        res = disregard_rows_with_missing_data(x_df, None)
    x_df, y_df = res[0]
    x_df = dummy_and_remove_features(x_df)
    return x_df, y_df


def potention_profits(decisons_true, decisons_pred, decisons_true_profits):
    if decisons_true.shape[0] != decisons_pred.shape[0] != decisons_true_profits.shape[0]:
        print(f'decisons_true (rows={decisons_true.shape[0]}), decisons_pred (rows={decisons_pred.shape[0]}), and '
              f'decisons_true_profits (rows={decisons_true_profits.shape[0]}) do not have the same number of rows')
        return
    d1, d2, d3 = decisons_true.to_numpy(), decisons_pred.to_numpy(), decisons_true_profits.to_numpy()
    potential_profits = 0
    for i in range(len(d1)):
        if d1[i][0] == d2[i][0]:
            potential_profits += d3[i][0]
    return potential_profits


def get_profit(close_price, open_price, pip_value, pip_resolution, in_quote_currency):
    pips = (close_price - open_price) / pip_resolution
    # calculates profit in the quote currency (right side currency of currency pair) by default
    profit = pip_value * pips  # can be negative
    if not in_quote_currency:
        profit /= close_price
    return profit


# reference: https://www.mql5.com/en/articles/4830
def get_margin(trades, buy_label, sell_label, contract_size, leverage, tradersway_commodity, in_quote_currency,
               hedged_margin, trade_indices=None):
    # *_trade_tups: (lots, open price)

    buy_lots = 0
    sell_lots = 0
    hedged_volume_margin = 0
    uncovered_volume_margin = 0

    multiplier = 1
    if contract_size > hedged_margin:
        multiplier = contract_size / hedged_margin

    if in_quote_currency:
        buy_price_lots = 0
        sell_price_lots = 0

        if trade_indices is not None:
            for trade_i in trade_indices:
                trade = trades[trade_i]
                decision_label = trade['decision_label']
                if decision_label == buy_label:
                    buy_lots += trade['lots']
                    buy_price_lots += trade['lots'] * trade['open_price']
                elif decision_label == sell_label:
                    sell_lots += trade['lots']
                    sell_price_lots += trade['lots'] * trade['open_price']
        else:
            for trade_i in trades:
                trade = trades[trade_i]
                decision_label = trade['decision_label']
                if decision_label == buy_label:
                    buy_lots += trade['lots']
                    buy_price_lots += trade['lots'] * trade['open_price']
                elif decision_label == sell_label:
                    sell_lots += trade['lots']
                    sell_price_lots += trade['lots'] * trade['open_price']

        total_lots = buy_lots + sell_lots
        wap = (buy_price_lots + sell_price_lots) / total_lots  # weighted average price

        # calculate uncovered volume margin
        if buy_lots > sell_lots:
            uncovered_lots = buy_lots - sell_lots
            uncovered_wap = buy_price_lots / buy_lots
            uncovered_volume_margin = uncovered_wap * uncovered_lots * contract_size / leverage
        elif buy_lots < sell_lots:
            uncovered_lots = sell_lots - buy_lots
            uncovered_wap = sell_price_lots / sell_lots
            uncovered_volume_margin = uncovered_wap * uncovered_lots * contract_size / leverage

        # calculate hedged volume margin
        hedged_volume_margin = wap * min(buy_lots, sell_lots) * contract_size / multiplier / leverage
    else:
        if trade_indices is not None:
            for trade_i in trade_indices:
                trade = trades[trade_i]
                decision_label = trade['decision_label']
                if decision_label == buy_label:
                    buy_lots += trade['lots']
                elif decision_label == sell_label:
                    sell_lots += trade['lots']
        else:
            for trade_i in trades:
                trade = trades[trade_i]
                decision_label = trade['decision_label']
                if decision_label == buy_label:
                    buy_lots += trade['lots']
                elif decision_label == sell_label:
                    sell_lots += trade['lots']

        # calculate uncovered volume margin
        if buy_lots > sell_lots:
            uncovered_lots = buy_lots - sell_lots
            uncovered_volume_margin = uncovered_lots * contract_size / leverage
        elif buy_lots < sell_lots:
            uncovered_lots = sell_lots - buy_lots
            uncovered_volume_margin = uncovered_lots * contract_size / leverage

        # calculate hedged volume margin
        hedged_volume_margin = min(buy_lots, sell_lots) * contract_size / multiplier / leverage

    margin = hedged_volume_margin + uncovered_volume_margin
    if tradersway_commodity:
        margin *= 2  # idk
    return margin


def convert_to_tensor(data):
    tensor = tf.convert_to_tensor(data, dtype=tf.float32)
    return tensor


def get_best_batch_size(data_len, min_bs, max_bs):
    best_bs = 0
    best_size = 0
    for i in range(data_len):
        bs = min_bs
        cur_len = data_len - i
        while cur_len % bs != 0 and bs < max_bs:
            bs += 1
        if cur_len % bs == 0:
            best_size = cur_len
            best_bs = bs
            break
    return best_bs, best_size


def get_split_lstm_data(preprocessed_data_df, ma_window, seq_len, split_percents=None, ma_cols=None, pc_cols=None,
                        normalization_groups=None,
                        min_batch_size=1000, max_batch_size=3500, just_train=False, print_info=True,
                        fully_divisible_batch_sizes=False, batch_size=1024,
                        buy_sell_labels_df=None, apply_pct_change=True):
    feature_names = preprocessed_data_df.columns
    if buy_sell_labels_df is not None:
        if buy_sell_labels_df.shape[0] != preprocessed_data_df.shape[0]:
            print(f'buy_sell_labels_df (shape={buy_sell_labels_df.shape}) does not have the same '
                  f'number of rows as preprocessed_data_df (shape={preprocessed_data_df.shape})')
            return
        buy_sell_label_name = buy_sell_labels_df.name
        preprocessed_data_df = pd.concat((preprocessed_data_df, buy_sell_labels_df), axis=1)

    if not ma_cols:
        ma_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    if not pc_cols:
        pc_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                   'trend_ichimoku_base', 'trend_ichimoku_conv',
                   'trend_ichimoku_a', 'trend_ichimoku_b']

    if not normalization_groups:
        normalization_groups = [['Open', 'High', 'Low', 'Close'],  # prices
                                ['trend_ichimoku_base', 'trend_ichimoku_conv'],  # ichi conv & base lines
                                ['trend_ichimoku_a', 'trend_ichimoku_b'],  # ichi cloud lines
                                ['tk_cross_bull_strength', 'tk_cross_bear_strength',  # tk cross strength
                                 'tk_price_cross_bull_strength', 'tk_price_cross_bear_strength',
                                 # tk price cross strength
                                 'senkou_cross_bull_strength', 'senkou_cross_bear_strength',  # semkou cross strength
                                 'chikou_cross_bull_strength', 'chikou_cross_bear_strength']]  # chikou cross strength

    col_to_idx = {col_name: preprocessed_data_df.columns.get_loc(col_name) for col_name in preprocessed_data_df.columns}

    # apply moving average to data to reduce bias (just learns centre of data) due to rugged raw price data that might look like a random walk to model
    # (but since we are now predicting a mov avg of the price it is less representative of the outliers which are still important)
    if ma_window is not None:
        preprocessed_data_df = apply_moving_avg(preprocessed_data_df, cols=ma_cols, window=ma_window)
        preprocessed_data_df.dropna(how='any', axis=0, inplace=True,
                                    subset=feature_names)  # drop any NA rows due to applying moving average

    # apply percentage change to data to make it so data is more stationary (past data more related to future data) since
    # the price data is typically strictly increasing or decreasing over the whole distribution
    if apply_pct_change:
        preprocessed_data_df = apply_perc_change(preprocessed_data_df, cols=pc_cols)
        preprocessed_data_df.dropna(how='any', axis=0, inplace=True,
                                    subset=feature_names)  # drop any NA rows due to applying percentage change

    if buy_sell_labels_df is not None:
        buy_sell_labels_df = preprocessed_data_df[buy_sell_label_name]
        preprocessed_data_df = preprocessed_data_df[feature_names]

    # normalize data for improved model training performance
    all_train_df, all_train_normalization_terms = normalize_data(preprocessed_data_df, train_data=True,
                                                                 groups=normalization_groups)

    # all training data
    all_train = all_train_df.to_numpy()

    if buy_sell_labels_df is not None:
        decision_to_int = {'buy': 1, 'sell': 0}
        buy_sell_labels = buy_sell_labels_df.to_numpy()

    all_x_train, all_y_train = [], []
    for i in range(seq_len, len(all_train)):
        if buy_sell_labels_df is None:
            all_x_train.append(all_train[i - seq_len:i])
            all_y_train.append(all_train[i][col_to_idx['Close']])
        else:
            decision = buy_sell_labels[i]
            if decision is not None:
                all_x_train.append(all_train[i - seq_len:i])
                all_y_train.append(decision_to_int[decision])
    all_x_train, all_y_train = np.array(all_x_train), np.array(all_y_train)

    if fully_divisible_batch_sizes:
        final_batch_size, final_train_data_size = get_best_batch_size(len(all_x_train), min_batch_size, max_batch_size)
    else:
        final_batch_size, final_train_data_size = batch_size, len(all_x_train)

    all_x_train_len_orig = len(all_x_train)
    all_x_train, all_y_train = all_x_train[-final_train_data_size:], all_y_train[-final_train_data_size:]

    # split data
    if not just_train:
        if sum(split_percents) > 1:
            print(f'sum of split_percents {split_percents} should not exceed 1')
            return

        len_data = preprocessed_data_df.shape[0]
        # only need to pass validation and test split percentages and the rest will be used as training data
        val_p, test_p = split_percents
        # only need to define num rows for validation and test data split so that they remain constant w/ respect to the size of preprocessed_data_df
        # so that when plotting data of diffrernt moving averages they line up consistatnly
        val_len, test_len = int(len_data * val_p), int(len_data * test_p)

        test_data_df = preprocessed_data_df.iloc[-test_len:]
        val_data_df = preprocessed_data_df.iloc[-(val_len + test_len):-test_len]
        train_data_df = preprocessed_data_df.iloc[:-(val_len + test_len)]

        if buy_sell_labels_df is not None:
            test_buy_sell_labels = buy_sell_labels_df.iloc[-test_len:].to_numpy()
            val_buy_sell_labels = buy_sell_labels_df.iloc[-(val_len + test_len):-test_len].to_numpy()
            train_buy_sell_labels = buy_sell_labels_df.iloc[:-(val_len + test_len)].to_numpy()

        train_data_df, normalization_terms = normalize_data(train_data_df, train_data=True, groups=normalization_groups)
        val_data_df = normalize_data(val_data_df, train_data=False, normalization_terms=normalization_terms)[0]
        test_data_df = normalize_data(test_data_df, train_data=False, normalization_terms=normalization_terms)[0]

        train_data = train_data_df.to_numpy()
        val_data = val_data_df.to_numpy()
        test_data = test_data_df.to_numpy()

        # training data
        x_train, y_train = [], []
        for i in range(seq_len, len(train_data)):
            if buy_sell_labels_df is None:
                x_train.append(train_data[i - seq_len:i])
                y_train.append(train_data[i][col_to_idx['Close']])
            else:
                decision = train_buy_sell_labels[i]
                if decision is not None:
                    x_train.append(train_data[i - seq_len:i])
                    y_train.append(decision_to_int[decision])
        x_train, y_train = np.array(x_train), np.array(y_train)

        # validation data
        x_val, y_val = [], []
        for i in range(seq_len, len(val_data)):
            if buy_sell_labels_df is None:
                x_val.append(val_data[i - seq_len:i])
                y_val.append(val_data[i][col_to_idx['Close']])
            else:
                decision = val_buy_sell_labels[i]
                if decision is not None:
                    x_val.append(val_data[i - seq_len:i])
                    y_val.append(decision_to_int[decision])
        x_val, y_val = np.array(x_val), np.array(y_val)

        # test data
        x_test, y_test = [], []
        for i in range(seq_len, len(test_data)):
            if buy_sell_labels_df is None:
                x_test.append(test_data[i - seq_len:i])
                y_test.append(test_data[i][col_to_idx['Close']])
            else:
                decision = test_buy_sell_labels[i]
                if decision is not None:
                    x_test.append(test_data[i - seq_len:i])
                    y_test.append(decision_to_int[decision])
        x_test, y_test = np.array(x_test), np.array(y_test)

        if fully_divisible_batch_sizes:
            eval_batch_size, eval_train_data_size = get_best_batch_size(len(x_train), min_batch_size, max_batch_size)
        else:
            eval_batch_size, eval_train_data_size = batch_size, len(x_train)

        x_train_len_orig = len(x_train)
        x_train, y_train = x_train[-eval_train_data_size:], y_train[-eval_train_data_size:]

        if print_info:
            print('------------------------------------------------------')
            print(f'data w/ moving average window of {ma_window} info:\n')
            print(f'batch size for evaluation: {eval_batch_size}')
            print(f'training data size reduction for evaulation: {x_train_len_orig} -> {eval_train_data_size}')
            print(f'batch size for final training: {final_batch_size}')
            print(
                f'training data size reduction for final training: {all_x_train_len_orig} -> {final_train_data_size}\n')
            print(f'training data shape: x={x_train.shape}, y={y_train.shape}')
            print(f'validation data shape: x={x_val.shape}, y={y_val.shape}')
            print(f'test data shape: x={x_test.shape}, y={y_test.shape}')
            print(f'all train data shape: x={all_x_train.shape}, y={all_y_train.shape}')
            print('------------------------------------------------------')

        data_dict = {
            'ma_window': ma_window,
            'eval_batch_size': eval_batch_size,
            'final_batch_size': final_batch_size,
            'train_data_df': train_data_df,
            'val_data_df': val_data_df,
            'test_data_df': test_data_df,
            'all_train_df': all_train_df,
            'train_data_np': (x_train, y_train),
            'val_data_np': (x_val, y_val),
            'test_data_np': (x_test, y_test),
            'sub_train_normalization_terms': normalization_terms,
            'all_train_data_np': (all_x_train, all_y_train),
            'all_train_normalization_terms': all_train_normalization_terms
        }
    else:
        if print_info:
            print('------------------------------------------------------')
            print(f'data w/ moving average window of {ma_window} info:\n')
            print(f'batch size for final training: {final_batch_size}')
            print(
                f'training data size reduction for final training: {all_x_train_len_orig} -> {final_train_data_size}\n')
            print(f'all train data shape: x={all_x_train.shape}, y={all_y_train.shape}')
            print('------------------------------------------------------')

        data_dict = {
            'ma_window': ma_window,
            'final_batch_size': final_batch_size,
            'all_train_df': all_train_df,
            'all_train_data_np': (all_x_train, all_y_train),
            'all_train_normalization_terms': all_train_normalization_terms
        }
    return data_dict
