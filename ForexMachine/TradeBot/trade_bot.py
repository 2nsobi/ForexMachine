import MetaTrader5 as mt5
import multiprocessing as mltp
from datetime import datetime, timezone, timedelta
import getpass
import time
from ForexMachine import util
from collections import deque
from ForexMachine.Preprocessing import research, live_trading
from typing import Optional
import pickle
import tensorflow as tf
import pandas as pd
import xgboost as xgb
import numpy as np

logger = util.Logger.get_instance()

timeframes = {
    'M1': (mt5.TIMEFRAME_M1, 60 * 1),
    'M2': (mt5.TIMEFRAME_M2, 60 * 2),
    'M3': (mt5.TIMEFRAME_M3, 60 * 3),
    'M4': (mt5.TIMEFRAME_M4, 60 * 4),
    'M5': (mt5.TIMEFRAME_M5, 60 * 5),
    'M6': (mt5.TIMEFRAME_M6, 60 * 6),
    'M10': (mt5.TIMEFRAME_M10, 60 * 10),
    'M12': (mt5.TIMEFRAME_M12, 60 * 12),
    'M15': (mt5.TIMEFRAME_M15, 60 * 15),
    'M20': (mt5.TIMEFRAME_M20, 60 * 20),
    'M30': (mt5.TIMEFRAME_M30, 60 * 30),
    'H1': (mt5.TIMEFRAME_H1, 60 * 60 * 1),
    'H2': (mt5.TIMEFRAME_H2, 60 * 60 * 2),
    'H3': (mt5.TIMEFRAME_H3, 60 * 60 * 3),
    'H4': (mt5.TIMEFRAME_H4, 60 * 60 * 4),
    'H6': (mt5.TIMEFRAME_H6, 60 * 60 * 6),
    'H8': (mt5.TIMEFRAME_H8, 60 * 60 * 8),
    'H12': (mt5.TIMEFRAME_H12, 60 * 60 * 12),
    'W1': (mt5.TIMEFRAME_W1, 60 * 60 * 24 * 7),
}


class TradeBot:
    def __init__(self):
        self.strats = {}
        self.mt5_login_info = {}

    def init_mt5(self):
        # get MT5 terminal initialization and login info from stdin
        self.mt5_login_info['trade_server'] = input(
            'Please enter MT5 server name (i.e. "MetaQuotes-Demo", "TradersWay-Live"): ')
        while True:
            try:
                login_id = int(input('Please enter login ID: '))
                self.mt5_login_info['login_id'] = login_id
                break
            except ValueError:
                print('Invalid login ID, ID must be interpretable as an integer')
        self.mt5_login_info['password'] = getpass.getpass(prompt='Please enter password: ')

    def run_strategy(self, strategy, strategy_kwargs={}, name_suffix=None):
        if name_suffix is not None:
            strat_name = f'{strategy.name}_{name_suffix}'
            if strat_name in self.strats:
                logger.error('Strategy already running with name "{strat_name}"')
                return
        else:
            strat_name = strategy.name
            i = 1
            while strat_name in self.strats:
                strat_name = f'{strategy.name}_{i}'
                i += 1
        parent_conn, child_conn = mltp.Pipe(duplex=True)
        exit_event = mltp.Event()
        proc = mltp.Process(target=run_trade_strategy, args=(strategy, strategy_kwargs, strat_name, child_conn,
                                                             self.mt5_login_info, exit_event))
        self.strats[strat_name] = {
            'parent_connection': parent_conn,
            'child_connection': child_conn,
            'process': proc,
            'strategy_kwargs': strategy_kwargs,
            'exit_event': exit_event
        }
        proc.start()
        return strat_name

    def stop_strategy(self, strat_name):
        self.strats[strat_name]['exit_event'].set()
        self.strats[strat_name]['process'].join()
        print(f'Stopped strategy with name: {strat_name}')


class TradeStrategy:
    def __init__(self):
        self.name = None
        self.proc_conn = None
        self.mt5_terminal_info = None
        self.mt5_account_info = None
        self.mt5_login_info = {}
        self.symbol = None
        self.timeframe = None
        self.mt5_timeframe = None
        self.data = deque()
        self.update_delta = None  # in secs
        self.bar_buffer_size = 200  # size of bar q
        self.utc_offset = None
        self.exit_event = None
        self.last_completed_bar_timestamp = None
        self.lfg = None
        self.max_concurrent_trades = np.inf
        self.indicators = []
        self.features = []
        self.custom_settings = {}
        self.trades = {}
        self.process_immediately = False

    def setup_trade_strategy(self, strat_name, proc_conn, mt5_login_info, exit_event):
        self.name = strat_name
        self.proc_conn = proc_conn
        self.mt5_login_info = mt5_login_info
        self.mt5_timeframe, self.update_delta = timeframes[self.timeframe]
        self.exit_event = exit_event

        if not mt5.initialize(login=self.mt5_login_info['login_id'], password=self.mt5_login_info['password'],
                              server=self.mt5_login_info['trade_server'], portable=True):
            logger.error(f'Failed to initialize MT5 terminal for {self.name} strategy, error:\n{mt5.last_error()}\n')
            return False
        else:
            self.mt5_terminal_info = mt5.terminal_info()
            self.mt5_account_info = mt5.account_info()
            print(f'Successfully initialized MT5 terminal for {self.name} strategy'
                  f'\n{self.mt5_terminal_info}\n{self.mt5_account_info}\n')

        self.utc_offset = self.get_server_utc_offset()
        if self.utc_offset is None:
            return False

        return True

    def get_server_utc_offset(self):
        latest_bar = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1, 0, 1)
        if latest_bar is None:
            logger.error(f'Nothing returned by copy_rates_from_pos(), mt5 error:\n{mt5.last_error()}')
            return
        latest_bar = latest_bar[0]

        server_current_hour_bar_dt = datetime.fromtimestamp(latest_bar[0], tz=timezone.utc)
        current_utc_dt = datetime.now(tz=timezone.utc)

        # just make sure both datetimes only go to hour resolution
        server_current_hour_bar_dt = datetime(year=server_current_hour_bar_dt.year,
                                              month=server_current_hour_bar_dt.month,
                                              day=server_current_hour_bar_dt.day, hour=server_current_hour_bar_dt.hour,
                                              tzinfo=timezone.utc)
        current_utc_hour_dt = datetime(year=current_utc_dt.year, month=current_utc_dt.month, day=current_utc_dt.day,
                                       hour=current_utc_dt.hour, tzinfo=timezone.utc)

        hour_offset = round((server_current_hour_bar_dt - current_utc_hour_dt).total_seconds() / (60 * 60))
        return hour_offset

    def get_next_update_delta(self, latest_bar_timestamp):
        next_bar_dt = (datetime.fromtimestamp(latest_bar_timestamp, tz=timezone.utc)
                       - timedelta(hours=self.utc_offset) + timedelta(seconds=self.update_delta))
        update_delta = (next_bar_dt - datetime.now(tz=timezone.utc)).total_seconds()
        return update_delta

    def run(self):
        first_bars = mt5.copy_rates_from_pos(self.symbol, self.mt5_timeframe, 0, self.bar_buffer_size + 1)
        if first_bars is None:
            logger.error(f'Nothing returned by copy_rates_from_pos(), mt5 error:\n{mt5.last_error()}')
            return False

        self.lfg = live_trading.LiveFeatureGenerator(indicators=self.indicators, features=self.features,
                                                     utc_offset=self.utc_offset, custom_settings=self.custom_settings)

        first_completed_bars = first_bars[1:]
        if not self.lfg.process_first_bars(first_completed_bars):
            logger.error(f'Live feature generator unable to process first bars')
            return False
        self.process_first_bars(self.lfg.data_q, self.lfg.all_features_filled_idx)

        if self.process_immediately:
            print(f'Immediate start: {self.name} strategy starting to process {self.symbol} '
                  f'{self.timeframe} price bars')
            self.process_new_data(self.lfg.data_q, self.lfg.feature_indices)

        next_update_delta = self.get_next_update_delta(first_bars[-1][0])
        if next_update_delta > 0:
            if not self.process_immediately:
                print(f'{self.name} strategy will begin processing {self.symbol} {self.timeframe} price bars in '
                      f'{next_update_delta} seconds')
            self.exit_event.wait(next_update_delta)
        else:
            if not self.process_immediately:
                print(f'{self.name} strategy has begun processing {self.symbol} {self.timeframe} price bars')

        self.last_completed_bar_timestamp = first_bars[-2][0]
        while not self.exit_event.is_set():
            latest_bars = mt5.copy_rates_from_pos(self.symbol, self.mt5_timeframe, 0, 2)
            if latest_bars is None:
                logger.error(f'Nothing returned by copy_rates_from_pos(), mt5 error:\n{mt5.last_error()}')
                return False

            while latest_bars[-2][0] == self.last_completed_bar_timestamp:
                self.exit_event.wait(0.5)
                latest_bars = mt5.copy_rates_from_pos(self.symbol, self.mt5_timeframe, 0, 2)
                if latest_bars is None:
                    logger.error(f'Nothing returned by copy_rates_from_pos(), mt5 error:\n{mt5.last_error()}')
                    return False

            if not self.lfg.process_new_bar(latest_bars[-2]):
                logger.error(f'Live feature generator unable to process new bar:\n{latest_bars[-2]}')
                return False
            self.process_new_data(self.lfg.data_q, self.lfg.feature_indices)

            next_update_delta = self.get_next_update_delta(latest_bars[-1][0])
            if next_update_delta > 0:
                self.exit_event.wait(next_update_delta)

            self.last_completed_bar_timestamp = latest_bars[-2][0]
        return True

    def get_reordered_feature_row(self, row, features):
        row = [row[self.lfg.feature_indices[feat_name]] for feat_name in features]
        return row

    def open_trade(self):
        if len(self.trades) < self.max_concurrent_trades:
            print('opening trade')
        return True

    def finish_up(self):
        mt5.shutdown()
        return True

    def process_first_bars(self, first_bars_q, all_features_filled_idx):
        pass

    def process_new_data(self, data_q, feature_indices):
        pass


class IchiCloudStrategy(TradeStrategy):
    name = f'ichi-cloud'

    def __init__(self, symbol='EURUSD', timeframe='H1', fast_ma_window=7, lots_per_trade=0.2, lstm_seq_len=128,
                 ichi_settings=(9, 30, 60), profit_noise_percent=0.0016, fast_ma_model_path=None, xgb_model_path=None,
                 fast_ma_diff_thresh=0.01, decision_prob_diff_thresh=0.5, bar_buffer_size=200, model_files_path=None,
                 train_data_start_iso=research.TRAIN_DATA_START_ISO, train_data_end_iso=research.TRAIN_DATA_END_ISO,
                 ma_cols=None, pc_cols=None, normalization_groups=None, open_trade_sigs=None, models_features_names=None,
                 max_concurrent_trades=np.inf, process_immediately=False):
        self.symbol = symbol.upper()
        self.timeframe = timeframe.upper()
        self.lots_per_trade = lots_per_trade
        self.tenkan_period, self.kijun_period, self.senkou_b_period = ichi_settings
        self.profit_noise_percent = profit_noise_percent
        self.fast_ma_diff_thresh = fast_ma_diff_thresh
        self.decision_prob_diff_thresh = decision_prob_diff_thresh
        self.fast_ma_window = fast_ma_window
        self.fast_ma_model_path = fast_ma_model_path
        self.bar_buffer_size = bar_buffer_size
        self.lstm_seq_len = lstm_seq_len
        self.xgb_model_path = xgb_model_path
        self.model_files_path = model_files_path
        self.normalization_groups = normalization_groups
        self.open_trade_sigs = open_trade_sigs
        self.ma_cols = ma_cols
        self.pc_cols = pc_cols
        self.models_features_names = models_features_names
        self.max_concurrent_trades = max_concurrent_trades
        self.process_immediately = process_immediately
        self.ma_cols_idx_set = None
        self.pc_cols_idx_set = None
        self.xgb_labels_dict = {1: 'buy', 0: 'sell'}
        self.indicators = ['ichimoku']
        self.features = ['ichimoku_signals']
        self.xgb_model_perc_chngs_q = deque()
        self.custom_settings = {
            'ichimoku': {
                'tenkan_period': self.tenkan_period,
                'kijun_period': self.kijun_period,
                'chikou_period': self.kijun_period,
                'senkou_b_period': self.senkou_b_period
            }
        }

        if self.ma_cols is None:
            self.ma_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if self.pc_cols is None:
            self.pc_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                            'kijun_base', 'tenken_conv', 'senkou_a', 'senkou_b']

        if self.normalization_groups is None:
            self.normalization_groups = [['Open', 'High', 'Low', 'Close'],
                                         ['kijun_base', 'tenken_conv'],
                                         ['senkou_a', 'senkou_b'],
                                         ['tk_cross_bull_strength', 'tk_cross_bear_strength',
                                          'tk_price_cross_bull_strength', 'tk_price_cross_bear_strength',
                                          'senkou_cross_bull_strength', 'senkou_cross_bear_strength',
                                          'chikou_cross_bull_strength', 'chikou_cross_bear_strength']]
        if self.open_trade_sigs is None:
            self.open_trade_sigs = ['cloud_breakout_bull', 'cloud_breakout_bear',
                                    'tk_cross_bull_strength', 'tk_cross_bear_strength',
                                    'tk_price_cross_bull_strength', 'tk_price_cross_bear_strength',
                                    'senkou_cross_bull_strength', 'senkou_cross_bear_strength',
                                    'chikou_cross_bull_strength', 'chikou_cross_bear_strength']

        if self.models_features_names is None:
            self.models_features_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'tenken_conv', 'kijun_base',
                                           'senkou_a', 'senkou_b', 'is_price_above_cb_lines',
                                           'is_price_above_cloud', 'is_price_inside_cloud', 'is_price_below_cloud',
                                           'cloud_breakout_bull', 'cloud_breakout_bear', 'tk_cross_bull_strength',
                                           'tk_cross_bear_strength', 'tk_price_cross_bull_strength',
                                           'tk_price_cross_bear_strength', 'senkou_cross_bull_strength',
                                           'senkou_cross_bear_strength', 'chikou_cross_bull_strength',
                                           'chikou_cross_bear_strength', 'quarter_2', 'quarter_3', 'quarter_4',
                                           'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 'day_of_week_4']

        if self.model_files_path is None:
            self.model_files_path = util.get_model_files_dir()

        if self.fast_ma_model_path is None:
            self.fast_ma_model_path = self.model_files_path / f'final_{self.symbol}-{self.timeframe}_Bi-LSTM' \
                                                              f'_{fast_ma_window}-ma_{self.tenkan_period}-' \
                                                              f'{self.kijun_period}-{self.senkou_b_period}-ichi.hdf5'

        self.fast_ma_model = tf.keras.models.load_model(self.fast_ma_model_path)

        if self.xgb_model_path is None:
            self.xgb_model_path = self.model_files_path / f'{self.symbol}-{self.timeframe}_0.01-min_profit_0.2-lots_right-cur_side' \
                                                          f'_{self.tenkan_period}-{self.kijun_period}-{self.senkou_b_period}' \
                                                          f'-cb-tk-tkp-sen-chi-ichi_xgb_classifier.json'

        self.xgb_decision_predictor = xgb.Booster()
        self.xgb_decision_predictor.load_model(self.xgb_model_path)

        self.norm_terms = self.get_normalization_terms(train_data_start_iso, train_data_end_iso)

    def get_normalization_terms(self, train_data_start_iso, train_data_end_iso):
        norm_terms_filepath = self.model_files_path / f'{self.name}_normalization_terms.pickle'

        if norm_terms_filepath.is_file():
            with open(norm_terms_filepath, 'rb') as ntfp:
                norm_terms = pickle.load(ntfp)
            print(f'loaded normalization terms for strategy with name "{self.name}" from {norm_terms_filepath}')
        else:
            indicators_info = {
                'ichimoku': {
                    'tenkan_period': self.tenkan_period,
                    'kijun_period': self.kijun_period,
                    'chikou_period': self.kijun_period,
                    'senkou_b_period': self.senkou_b_period
                }
            }
            tick_data_filepath = research.download_mt5_data(self.symbol, self.timeframe, train_data_start_iso,
                                                            train_data_end_iso)
            print(tick_data_filepath)
            data_with_indicators = research.add_indicators_to_raw(filepath=tick_data_filepath,
                                                                  indicators_info=indicators_info,
                                                                  datetime_col='datetime')
            train_data = research.add_ichimoku_features(data_with_indicators)

            start_idx, end_idx = research.no_missing_data_idx_range(train_data,
                                                                    early_ending_cols=['chikou_span_visual'])
            train_data = train_data.iloc[start_idx:end_idx + 1]
            train_data = research.dummy_and_remove_features(train_data)

            fast_ma_data = research.get_split_lstm_data(train_data, ma_window=self.fast_ma_window, min_batch_size=1000,
                                                        seq_len=self.lstm_seq_len, split_percents=(0, 0),
                                                        fully_divisible_batch_sizes=True, max_batch_size=2000,
                                                        normalization_groups=self.normalization_groups,
                                                        pc_cols=self.pc_cols, ma_cols=self.ma_cols, print_info=False)
            norm_terms = fast_ma_data['all_train_normalization_terms']

            with open(norm_terms_filepath, 'wb') as ntfp:
                pickle.dump(norm_terms, ntfp)
            print(f'generated normalization terms for strategy with name "{self.name}" from {train_data_start_iso} to '
                  f'{train_data_end_iso} and saved pickle file to {norm_terms_filepath}')

        return norm_terms

    def process_new_data(self, data_q, feature_indices):
        if self.ma_cols_idx_set is None:
            self.ma_cols_idx_set = {feature_indices[feat_name] for feat_name in self.ma_cols}
            self.pc_cols_idx_set = {feature_indices[feat_name] for feat_name in self.pc_cols}

        open_trade = False
        for sig in self.open_trade_sigs:
            sig_i = feature_indices[sig]
            if data_q[-1][sig_i] != 0:
                open_trade = True
                break

        if open_trade or 1:
            xgb_model_input_row = research.apply_perc_change_list(data_q[-2], data_q[-1], cols_set=self.pc_cols_idx_set)
            xgb_model_input_row = self.get_reordered_feature_row(xgb_model_input_row, features=self.models_features_names)
            xgb_model_input = pd.DataFrame([xgb_model_input_row], columns=self.models_features_names)
            xgb_model_input = xgb.DMatrix(xgb_model_input)
            decision_prob = self.xgb_decision_predictor.predict(xgb_model_input)[0]
            decision_label = round(decision_prob)
            decision_prob_diff = abs(decision_label - decision_prob)

            print(f'{decision_prob}: decision_prob')
            print(f'{decision_label}: decision_label')
            print(f'{decision_prob_diff}: decision_prob_diff')
            print(f'{self.decision_prob_diff_thresh}: self.decision_prob_diff_thresh')
            if decision_prob_diff <= self.decision_prob_diff_thresh:
                self.open_trade()


def run_trade_strategy(strategy, strategy_kwargs, strat_name, proc_conn, mt5_login_info, exit_event):
    strategy = strategy(**strategy_kwargs)

    if not strategy.setup_trade_strategy(strat_name, proc_conn, mt5_login_info, exit_event):
        return False

    if not strategy.run():
        return False

    if not strategy.finish_up():
        logger.error('Unable to properly finish up strategy')
        return False

    return True


if __name__ == '__main__':
    # mt5.initialize()
    #
    # rates = mt5.copy_rates_from_pos('EURUSD', 16385, 0, 200)
    # print(rates)
    # print(type(rates))
    # print(type(rates[0]))
    # print(type(rates[0][0]))
    # mt5.shutdown()

    # print()
    # for bar in rates:
    #     print(datetime.utcnow().isoformat())
    #     print(datetime.now(tz=timezone.utc).isoformat())
    #     print(datetime.fromtimestamp(bar[0], tz=timezone.utc).isoformat())
    #     print(datetime.utcfromtimestamp(bar[0]).isoformat(), bar)
    #     print()

    tb = TradeBot()
    tb.init_mt5()

    strategy_kwargs = {
        'timeframe': 'm1',
        'process_immediately': False
    }
    name1 = tb.run_strategy(IchiCloudStrategy, strategy_kwargs=strategy_kwargs, name_suffix='1')

    time.sleep(120)

    tb.stop_strategy(name1)
