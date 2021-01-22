from collections import deque
import MetaTrader5 as mt5
from ForexMachine import util
from ForexMachine.Preprocessing.research import FeatureGenerator
from sklearn.preprocessing import OneHotEncoder
from datetime import timedelta, timezone
from pandas import Timestamp
import numpy as np
from typing import Optional

logger = util.Logger.get_instance()


class LiveFeatureGenerator:
    def __init__(self, indicators, features, utc_offset, custom_settings=None, q_size=None):
        self.indicators_list = indicators
        self.features_list = features
        self.utc_offset = utc_offset
        self.q_size = q_size
        self.indicators = {}
        self.features = {}
        self.custom_settings = custom_settings
        self.data_q = deque()
        self.feature_indices = {}
        self.temporal_enc = None
        self.temporal_feature_names = None
        self.research_feature_generator = None
        self.all_indicators_filled_idx = None
        self.all_features_filled_idx = None
        self.ichi_cross_names = None
        self.lfg_setup = False
        self.default_settings = {
            'ichimoku': {
                'tenkan_period': 9,
                'kijun_period': 26,
                'chikou_period': 26,
                'senkou_b_period': 52
            },
            'ichimoku_signals': {
                'negative_bears': True
            }
        }
        self.available_indicators_funcs = {
            'ichimoku': self.add_ichimoku_indicator
        }
        self.available_features_funcs = {
            'ichimoku_signals': self.add_ichimoku_signals
        }
        self.temporal_categories = {
            'quarter': [1, 2, 3, 4],
            'day_of_week': [0, 1, 2, 3, 4, 6]
        }

    def setup_live_feature_generator(self, first_bars, first_bar_headers=None):
        if first_bar_headers is None:
            first_bar_headers = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'spread', 'real_volume']
        if len(first_bar_headers) != len(first_bars[0]):
            logger.error(f'The length of first_bar_headers ({first_bar_headers}, len={len(first_bar_headers)}) does'
                         f'not equal the number of elements in each row of first_bars (len={len(first_bars[0])})')
            return False
        self.feature_indices = {first_bar_headers[i]: i for i in range(len(first_bar_headers))}

        if self.q_size is None:
            self.q_size = len(first_bars)

        for indicator in self.indicators_list:
            if indicator not in self.available_indicators_funcs:
                logger.error(f'"{indicator}" is not an available indicator')
                return False
            indicator_dict = {}
            for setting in self.default_settings[indicator]:
                value = self.default_settings[indicator][setting]
                if self.custom_settings is not None \
                        and indicator in self.custom_settings and setting in self.custom_settings[indicator]:
                    value = self.custom_settings[indicator][setting]
                indicator_dict[setting] = value
            indicator_dict['func'] = self.available_indicators_funcs[indicator]
            indicator_dict['processed_bars'] = 0
            self.indicators[indicator] = indicator_dict

        for feature in self.features_list:
            if feature not in self.available_features_funcs:
                logger.error(f'"{feature}" is not an available feature')
                return False
            feature_dict = {}
            for setting in self.default_settings[feature]:
                value = self.default_settings[feature][setting]
                if self.custom_settings is not None \
                        and feature in self.custom_settings and setting in self.custom_settings[feature]:
                    value = self.custom_settings[feature][setting]
                feature_dict[setting] = value
            feature_dict['func'] = self.available_features_funcs[feature]
            feature_dict['processed_bars'] = 0
            self.features[feature] = feature_dict

        self.lfg_setup = True
        return True

    def process_first_bars(self, first_bars, first_bar_headers=None):
        if not self.lfg_setup:
            self.setup_live_feature_generator(first_bars=first_bars, first_bar_headers=first_bar_headers)

        for bar in first_bars:
            self.data_q.append(list(bar))

        for i in range(len(self.data_q)):
            # add temporal features by default
            self.add_temporal_features(i)

            # calculate indicators first
            for indicator in self.indicators:
                self.indicators[indicator]['func'](i)

            # check and update oldest idx of data_q that has no None
            # values in indicator cols, should eventually equal 0
            if self.all_indicators_filled_idx is None:
                has_none = False
                for elem in self.data_q[i]:
                    if elem is None:
                        has_none = True
                        break
                if not has_none:
                    self.all_indicators_filled_idx = i

            # then generate other features because they can depend in indicators
            for feature in self.features:
                self.features[feature]['func'](i)

            # check and update oldest idx of data_q that has no None
            # values in all feature cols, should eventually equal 0
            if self.all_features_filled_idx is None:
                has_none = False
                for elem in self.data_q[i]:
                    if elem is None:
                        has_none = True
                        break
                if not has_none:
                    self.all_features_filled_idx = i

        return True

    def process_new_bar(self, new_bar, first_bar_headers=None):
        if not self.lfg_setup:
            self.setup_live_feature_generator(first_bars=[new_bar], first_bar_headers=first_bar_headers)

        self.data_q.append(list(new_bar))
        while len(self.data_q) > self.q_size:
            self.data_q.popleft()
            if self.all_indicators_filled_idx is not None and self.all_indicators_filled_idx > 0:
                self.all_indicators_filled_idx -= 1
            if self.all_features_filled_idx is not None and self.all_features_filled_idx > 0:
                self.all_features_filled_idx -= 1
        data_idx = len(self.data_q) - 1

        # add temporal features by default
        self.add_temporal_features(data_idx)

        # calculate indicators first
        for indicator in self.indicators:
            self.indicators[indicator]['func'](data_idx)

        # check and update oldest idx of data_q that has no None values in indicator cols, should eventually equal 0
        if self.all_indicators_filled_idx is None:
            has_none = False
            for elem in self.data_q[data_idx]:
                if elem is None:
                    has_none = True
                    break
            if not has_none:
                self.all_indicators_filled_idx = data_idx

        # then generate other features because they can depend in indicators
        for feature in self.features:
            self.features[feature]['func'](data_idx)

        # check and update oldest idx of data_q that has no None
        # values in all feature cols, should eventually equal 0
        if self.all_features_filled_idx is None:
            has_none = False
            for elem in self.data_q[i]:
                if elem is None:
                    has_none = True
                    break
            if not has_none:
                self.all_features_filled_idx = i

        return True

    def add_temporal_features(self, data_idx):
        row = self.data_q[data_idx]
        if self.temporal_enc is None:
            self.temporal_enc = OneHotEncoder(categories=[self.temporal_categories['quarter'],
                                                          self.temporal_categories['day_of_week']], drop='first')
            # fit the sklearn OneHotEncoder with any timestamp to avoid having to fit for each encoder transform()
            timestamp = 1581552000
            pandas_dt = Timestamp(timestamp, unit='s', tzinfo=timezone.utc)
            quarter_and_day = (pandas_dt.quarter, pandas_dt.dayofweek)
            self.temporal_enc.fit([quarter_and_day])
            self.temporal_feature_names = self.temporal_enc.get_feature_names(('quarter', 'day_of_week'))

        pandas_dt = Timestamp(row[self.feature_indices['timestamp']],
                              unit='s', tzinfo=timezone.utc) - timedelta(hours=self.utc_offset)
        quarter_and_day = (pandas_dt.quarter, pandas_dt.dayofweek)
        temporal_feats = self.temporal_enc.transform([quarter_and_day]).toarray()[0]

        for i in range(len(self.temporal_feature_names)):
            temporal_feat_name = self.temporal_feature_names[i]
            temporal_feat_val = temporal_feats[i]
            if temporal_feat_name not in self.feature_indices:
                self.feature_indices[temporal_feat_name] = len(self.feature_indices)
            row.append(temporal_feat_val)

    def add_ichimoku_indicator(self, data_idx):
        row = self.data_q[data_idx]

        if 'longest_period' not in self.indicators['ichimoku']:
            self.indicators['ichimoku']['longest_period'] = max(self.indicators['ichimoku']['tenkan_period'],
                                                                self.indicators['ichimoku']['kijun_period'],
                                                                self.indicators['ichimoku']['chikou_period'],
                                                                self.indicators['ichimoku']['senkou_b_period'])

        if 'high_q' not in self.indicators['ichimoku']:
            self.indicators['ichimoku']['high_q'] = deque()
            self.indicators['ichimoku']['low_q'] = deque()
        high_q = self.indicators['ichimoku']['high_q']
        low_q = self.indicators['ichimoku']['low_q']
        high_q.append(row[self.feature_indices['High']])
        low_q.append(row[self.feature_indices['Low']])
        if len(high_q) > self.indicators['ichimoku']['longest_period']:
            high_q.popleft()
            low_q.popleft()

        tenken_conv = None
        tenkan_period = self.indicators['ichimoku']['tenkan_period']
        if len(high_q) >= tenkan_period:
            period_high = max([high_q[-i] for i in range(1, tenkan_period + 1)])
            period_low = min([low_q[-i] for i in range(1, tenkan_period + 1)])
            tenken_conv = (period_high + period_low) / 2

        kijun_base = None
        kijun_period = self.indicators['ichimoku']['kijun_period']
        if len(high_q) >= kijun_period:
            period_high = max([high_q[-i] for i in range(1, kijun_period + 1)])
            period_low = min([low_q[-i] for i in range(1, kijun_period + 1)])
            kijun_base = (period_high + period_low) / 2

        senkou_a = None
        if tenken_conv and kijun_base:
            senkou_a = (tenken_conv + kijun_base) / 2

        senkou_b = None
        senkou_b_period = self.indicators['ichimoku']['senkou_b_period']
        if len(high_q) >= senkou_b_period:
            period_high = max([high_q[-i] for i in range(1, senkou_b_period + 1)])
            period_low = min([low_q[-i] for i in range(1, senkou_b_period + 1)])
            senkou_b = (period_high + period_low) / 2

        chikou_span = None
        senkou_a_visual = None
        senkou_b_visual = None
        chikou_period = self.indicators['ichimoku']['chikou_period']
        if data_idx >= chikou_period:
            chikou_span = self.data_q[data_idx - chikou_period][self.feature_indices['Close']]

            if 'senkou_a' in self.feature_indices:
                senkou_a_visual = self.data_q[data_idx - chikou_period][self.feature_indices['senkou_a']]
            if 'senkou_b' in self.feature_indices:
                senkou_b_visual = self.data_q[data_idx - chikou_period][self.feature_indices['senkou_b']]

        for key, val in [('tenken_conv', tenken_conv), ('kijun_base', kijun_base), ('senkou_a', senkou_a),
                         ('senkou_b', senkou_b), ('chikou_span', chikou_span), ('senkou_a_visual', senkou_a_visual),
                         ('senkou_b_visual', senkou_b_visual)]:
            if key not in self.feature_indices:
                self.feature_indices[key] = len(self.feature_indices)
            row.append(val)

        self.indicators['ichimoku']['processed_bars'] += 1

    def add_ichimoku_signals(self, data_idx):
        row = self.data_q[data_idx]

        if self.research_feature_generator is None:
            self.research_feature_generator = FeatureGenerator(self.data_q, self.feature_indices, live_trading=True)
            self.ichi_cross_names = ['tk_cross', 'tk_price_cross', 'senkou_cross', 'chikou_cross']

        is_price_above_cb_lines = None
        is_price_above_cloud = None
        is_price_inside_cloud = None
        is_price_below_cloud = None
        cloud_breakout_bull = None
        cloud_breakout_bear = None

        # dict to hold similar features of each cross type
        cross_feature_dict = {}
        for cross_name in self.ichi_cross_names:
            cross_feature_dict[cross_name] = {
                'bull_strength': None,
                'bear_strength': None,
                'bull_length': None,
                'bear_length': None,
            }

        # need to be 1 bar ahead of last row with completely fill indicators because crosses are
        # identified by looking one row back
        if self.all_indicators_filled_idx is not None and data_idx > self.all_indicators_filled_idx + 1:
            ichimoku_features = self.research_feature_generator.get_ichimoku_features(data_idx,
                                                                                      cross_length_limit=np.Inf)
            is_price_above_cb_lines = ichimoku_features['is_price_above_cb_lines']
            is_price_above_cloud = ichimoku_features['is_price_above_cloud']
            is_price_inside_cloud = ichimoku_features['is_price_inside_cloud']
            is_price_below_cloud = ichimoku_features['is_price_below_cloud']
            cloud_breakout_bull = ichimoku_features['cloud_breakout_bull']
            cloud_breakout_bear = ichimoku_features['cloud_breakout_bear']

            negative_bears = self.features['ichimoku_signals']['negative_bears']
            for cross_name in self.ichi_cross_names:
                bull_strength, bear_strength, cross_length = ichimoku_features[cross_name]
                bull_length = 0
                bear_length = 0
                if bull_strength > bear_strength:
                    bull_length = cross_length
                else:
                    bear_length = cross_length
                if negative_bears:
                    bear_strength *= -1
                cross_feature_dict[cross_name]['bull_strength'] = bull_strength
                cross_feature_dict[cross_name]['bear_strength'] = bear_strength
                cross_feature_dict[cross_name]['bull_length'] = bull_length
                cross_feature_dict[cross_name]['bear_length'] = bear_length

        for key, val in [('is_price_above_cb_lines', is_price_above_cb_lines),
                         ('is_price_above_cloud', is_price_above_cloud),
                         ('is_price_inside_cloud', is_price_inside_cloud),
                         ('is_price_below_cloud', is_price_below_cloud),
                         ('cloud_breakout_bull', cloud_breakout_bull), ('cloud_breakout_bear', cloud_breakout_bear)]:
            if key not in self.feature_indices:
                self.feature_indices[key] = len(self.feature_indices)
            row.append(val)

        for cross_name in cross_feature_dict:
            for cross_feature in cross_feature_dict[cross_name]:
                key = f'{cross_name}_{cross_feature}'
                if key not in self.feature_indices:
                    self.feature_indices[key] = len(self.feature_indices)
                row.append(cross_feature_dict[cross_name][cross_feature])

        self.features['ichimoku_signals']['processed_bars'] += 1


if __name__ == '__main__':
    import time
    import pandas as pd
    from ForexMachine.Preprocessing import research

    mt5.initialize()

    start_bar = 1
    num_bars = 400

    rates = mt5.copy_rates_from_pos('EURUSD', mt5.TIMEFRAME_H1, start_bar, num_bars)

    custom_settings = {
        'ichimoku': {
            'tenkan_period': 9,
            'kijun_period': 30,
            'chikou_period': 30,
            'senkou_b_period': 60
        }
    }

    lfg = LiveFeatureGenerator(indicators=['ichimoku'], features=['ichimoku_signals'], utc_offset=2,
                               custom_settings=custom_settings, q_size=num_bars)

    s = time.time()
    lfg.process_first_bars(rates)
    d = (time.time() - s) * 1000

    # for bar in rates:
    #     lfg.process_new_bar(bar)

    # for i, row in enumerate(lfg.data_q):
    #     print(i, len(row), row)
    # print(lfg.feature_indices)
    # print(d)

    # new_bar = mt5.copy_rates_from_pos('EURUSD', mt5.TIMEFRAME_H1, 0, 1)[0]
    #
    # lfg.process_new_bar(new_bar)
    #
    # for i, row in enumerate(lfg.data_q):
    #     print(i, len(row), row)

    lfg_df = pd.DataFrame(lfg.data_q, columns=list(lfg.feature_indices.keys()))
    lfg_df.to_csv('lfg_df.csv')
    print(lfg_df)

    from_dt = Timestamp(rates[0][0], unit='s', tzinfo=timezone.utc)
    to_dt = Timestamp(rates[-1][0], unit='s', tzinfo=timezone.utc)
    print(from_dt.isoformat())
    print(to_dt.isoformat())

    indicators_info = {
        'ichimoku': custom_settings['ichimoku'],
        'rsi': {
            'periods': 14
        }
    }

    tick_data_filepath = research.download_mt5_data('EURUSD', 'H1', from_dt.isoformat(), to_dt.isoformat())
    data_with_indicators = research.add_indicators_to_raw(filepath=tick_data_filepath,
                                                          indicators_info=indicators_info,
                                                          datetime_col='datetime')
    train_data = research.add_ichimoku_features(data_with_indicators)
    train_data.to_csv('train_data.csv')
    print(train_data)

    lfg_df_cols = set(lfg_df.columns)
    train_data_cols = set(train_data.columns)

    non_matching = 0
    non_matching_cols = set()
    for col in lfg_df_cols:
        if col in train_data_cols:
            for i in range(lfg_df.shape[0]):
                lfg_df_val = lfg_df.iloc[i][lfg_df.columns.get_loc(col)]
                if not pd.isnull(lfg_df_val):
                    train_data_val = train_data.iloc[i][train_data.columns.get_loc(col)]
                    matching = True
                    if isinstance(lfg_df_val, (np.floating, float)):
                        if not np.isclose(lfg_df_val, train_data_val):
                            matching = False
                    else:
                        if lfg_df_val != train_data_val:
                            matching = False
                    if not matching:
                        non_matching_cols.add(col)
                        print(f'non-matching data in col "{col}" and row {i}, rows:')
                        print(f'lfg row: {list(lfg_df.iloc[i])}')
                        print(f'train_data row: {list(train_data.iloc[i])}')
                        print(f'lfg val: {lfg_df_val}')
                        print(f'train_data val: {train_data_val}\n')
                        non_matching += 1
    print(f'non_matching: {non_matching}')
    print(f'non_matching_cols: {non_matching_cols}')
    print(lfg_df.shape)
    print(train_data.shape)

    mt5.shutdown()
