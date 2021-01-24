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
import pandas as pd
import xgboost as xgb
import numpy as np
import sys
from pathlib import Path
import configparser
import subprocess
import os

import tensorflow as tf

# print(f'is GPU available for TF: {tf.test.is_gpu_available()}')
#
# gpu_devices = tf.config.list_physical_devices('GPU')
# print(f'GPU devices: {gpu_devices}')
#
# all_devices = tf.config.list_physical_devices()
# print(f'all devices: {all_devices}')
#
# if len(gpu_devices) > 0:
#     for device in gpu_devices:
#         tf.config.experimental.set_memory_growth(device, True)

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

    def init_mt5(self, forex_machine_config_path=None, enable_mt5_auto_trading=False):
        # set mt5 terminal config so that auto trading is enabled
        mt5_terminal_path = None
        if forex_machine_config_path is None:
            forex_machine_config_path = Path('./forex_machine_config.ini')
        else:
            forex_machine_config_path = Path(forex_machine_config_path)
        if not forex_machine_config_path.is_file():
            print(f'No ForexMachine config .ini file found with name {forex_machine_config_path}'
                  f'\nPlease enter login info on terminal')
        else:
            config_parser = configparser.ConfigParser()
            config_parser.read(forex_machine_config_path)
            if 'mt5_terminal_path' in config_parser['common']:
                mt5_terminal_path = config_parser['common']['mt5_terminal_path']
            if 'trade_server' in config_parser['common']:
                self.mt5_login_info['trade_server'] = config_parser['common']['trade_server']
            if 'login_id' in config_parser['common']:
                try:
                    self.mt5_login_info['login_id'] = int(config_parser['common']['login_id'])
                except ValueError:
                    print('Invalid login ID, ID must be interpretable as an integer')
            if 'password' in config_parser['common']:
                self.mt5_login_info['password'] = config_parser['common']['password']

        if enable_mt5_auto_trading:
            if mt5_terminal_path is None:
                print('MT5 terminal path not found in ForexMachine config, '
                      'please make sure auto trading is already enabled')
            else:
                mt5_terminal_path = Path(mt5_terminal_path).resolve()
                if not mt5_terminal_path.is_file():
                    logger.error(f'MT5 terminal not found at {mt5_terminal_path}')
                    return False
                else:
                    proc = mltp.Process(target=enable_mt5_auto_trading_func, args=(mt5_terminal_path,))
                    proc.start()
                    if os.name == 'nt':  # windows
                        stdout = ''
                        retries = 0
                        while 'terminal64.exe' not in stdout and retries < 60:
                            subproc = subprocess.run(args=['tasklist'], encoding='utf-8',
                                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                            if subproc.returncode != 0:
                                logger.error('Unable to run "tasklist" command and see if MT5 terminal is running')
                                proc.terminate()
                                return False
                            stdout = subproc.stdout
                            time.sleep(1)
                            retries += 1
                        if 'terminal64.exe' not in stdout:
                            logger.error(f'MT5 terminal not detected as running based on "tasklist" command '
                                         f'after {retries} retries')
                            proc.terminate()
                            return False
                    else:  # unix
                        logger.error('Enabling auto trading for MT5 terminal not implemented for unix yet :(')
                        return False
                    proc.terminate()

        # get MT5 terminal initialization and login info from stdin
        if 'trade_server' not in self.mt5_login_info:
            self.mt5_login_info['trade_server'] = input(
                'Please enter MT5 server name (i.e. "MetaQuotes-Demo", "TradersWay-Live"): ')
        if 'login_id' not in self.mt5_login_info:
            while True:
                try:
                    login_id = int(input('Please enter login ID: '))
                    self.mt5_login_info['login_id'] = login_id
                    break
                except ValueError:
                    print('Invalid login ID, ID must be interpretable as an integer')
        if 'password' not in self.mt5_login_info:
            self.mt5_login_info['password'] = getpass.getpass(prompt='Please enter password: ')

        return True

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
        self.ea_id = None
        self.proc_conn = None
        self.mt5_terminal_info = None
        self.mt5_account_info = None
        self.mt5_login_info = {}
        self.symbol = None
        self.timeframe = None
        self.mt5_timeframe = None
        self.data = deque()
        self.update_delta = None  # in secs
        self.bar_buffer_size = 300  # size of bar q
        self.utc_offset = None
        self.exit_event = None
        self.last_completed_bar_timestamp = None
        self.lfg = None
        self.max_concurrent_trades = np.inf
        self.indicators = []
        self.features = []
        self.custom_settings = {}
        self.trades = None
        self.process_immediately = False
        self.check_if_market_is_closed = True
        self.lots_per_trade = None
        self.trade_decision_strings = None

    def setup_trade_strategy_process(self, strat_name, proc_conn, mt5_login_info, exit_event):
        self.name = strat_name
        self.ea_id = hash(self.name) % ((sys.maxsize + 1) * 2)
        self.proc_conn = proc_conn
        self.mt5_login_info = mt5_login_info
        self.mt5_timeframe, self.update_delta = timeframes[self.timeframe]
        self.exit_event = exit_event
        self.trades = {}
        self.trade_decision_strings = {1: 'buy', 0: 'sell'}

        if not hasattr(self, 'check_if_market_is_closed'):
            self.check_if_market_is_closed = True

        if not mt5.initialize(login=self.mt5_login_info['login_id'], password=self.mt5_login_info['password'],
                              server=self.mt5_login_info['trade_server']):
            logger.error(f'Failed to initialize MT5 terminal for {self.name} strategy, error:\n{mt5.last_error()}\n')
            return False
        else:
            self.mt5_terminal_info = mt5.terminal_info()
            self.mt5_account_info = mt5.account_info()
            print(f'Successfully initialized MT5 terminal for {self.name} strategy'
                  f'\nMT5 terminal info: {self.mt5_terminal_info}\nMT5 account info: {self.mt5_account_info}')

        market_closed = False
        if self.check_if_market_is_closed:
            market_closed = self.is_market_closed()

        if market_closed:
            self.sleep_till_market_open()

        self.utc_offset = self.get_server_utc_offset()
        if self.utc_offset is None:
            return False

        if self.symbol is not None and self.lots_per_trade is None:
            symbol_info = mt5.symbol_info(self.symbol)
            self.lots_per_trade = symbol_info.volume_step

        self.mt5_initialized()

        return True

    def sleep_till_market_open(self):
        # markets should be open by 10:00PM GMT Sunday
        # source: http://www.forex-internet.com/forexhours.htm
        open_dt = datetime.now(tz=timezone.utc)
        while open_dt.weekday() != 6:
            open_dt += timedelta(days=1)
        open_dt = datetime(year=open_dt.year, month=open_dt.month, day=open_dt.day, hour=22, minute=1,
                           tzinfo=timezone.utc)

        time_now = datetime.now(tz=timezone.utc)
        sleep_time = (open_dt - time_now).total_seconds()
        print(f'Sleeping {self.name} strategy until {self.symbol} market is open on Sunday at 10:00 PM GMT '
              f'in {sleep_time / 60 / 60} hours, goodnight')
        self.exit_event.wait(sleep_time)

        while True and not self.exit_event.is_set():
            if not self.is_market_closed():
                break
            self.exit_event.wait(60 * 60)
            print(f'{self.symbol} market appears to still be closed at {datetime.now(tz=timezone.utc)},'
                  f' sleeping {self.name} strategy for another hour')

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

        #######yo
        hour_offset = 2
        return hour_offset

    def get_next_update_delta(self, latest_bar_timestamp):
        next_bar_dt = (datetime.fromtimestamp(latest_bar_timestamp, tz=timezone.utc)
                       - timedelta(hours=self.utc_offset) + timedelta(seconds=self.update_delta))
        update_delta = (next_bar_dt - datetime.now(tz=timezone.utc)).total_seconds()
        return update_delta

    def run(self):
        if self.exit_event.is_set():
            return True

        first_bars = mt5.copy_rates_from_pos(self.symbol, self.mt5_timeframe, 0, self.bar_buffer_size + 1)
        if first_bars is None:
            logger.error(f'Nothing returned by copy_rates_from_pos(), mt5 error:\n{mt5.last_error()}')
            return False

        self.lfg = live_trading.LiveFeatureGenerator(indicators=self.indicators, features=self.features,
                                                     utc_offset=self.utc_offset, custom_settings=self.custom_settings)

        first_completed_bars = first_bars[:-1]
        if not self.lfg.process_first_bars(first_completed_bars):
            logger.error(f'Live feature generator unable to process first bars')
            return False
        self.process_first_bars(self.lfg.data_q, self.lfg.feature_indices, self.lfg.all_features_filled_idx)

        if self.exit_event.is_set():
            return True

        if self.process_immediately:
            print(f'Immediate start: {self.name} strategy starting to process {self.symbol} '
                  f'{self.timeframe} price bars')
            self.process_new_data(self.lfg.data_q, self.lfg.feature_indices, True)

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

            retries = 0
            while latest_bars[-2][0] == self.last_completed_bar_timestamp and retries < 120:
                self.exit_event.wait(0.5)
                latest_bars = mt5.copy_rates_from_pos(self.symbol, self.mt5_timeframe, 0, 2)
                if latest_bars is None:
                    logger.error(f'Nothing returned by copy_rates_from_pos(), mt5 error:\n{mt5.last_error()}')
                    return False
                retries += 1

            if latest_bars[-2][0] == self.last_completed_bar_timestamp:
                market_closed = self.is_market_closed()
                if market_closed:
                    self.sleep_till_market_open()
                    continue

            if not self.lfg.process_new_bar(latest_bars[-2]):
                logger.error(f'Live feature generator unable to process new bar:\n{latest_bars[-2]}')
                return False
            self.process_new_data(self.lfg.data_q, self.lfg.feature_indices, False)

            next_update_delta = self.get_next_update_delta(latest_bars[-1][0])
            if next_update_delta > 0:
                self.exit_event.wait(next_update_delta)

            self.last_completed_bar_timestamp = latest_bars[-2][0]
        return True

    def is_market_closed(self):
        print(f'Checking if {self.symbol} market is open...')
        last_timestamp = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, 1)[0][0]
        for _ in range(2):
            self.exit_event.wait(70)
            timestamp = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, 1)[0][0]
            if timestamp != last_timestamp:
                print(f'{self.symbol} market does not seem to be closed')
                return False
            last_timestamp = timestamp
        print(f'{self.symbol} market seems to be closed')
        return True

    def get_reordered_feature_row(self, row, features):
        row = [row[self.lfg.feature_indices[feat_name]] for feat_name in features]
        return row

    def open_trade(self, trade_type_int, additional_dict=None):
        if len(self.trades) < self.max_concurrent_trades:
            symbol_info_tick = mt5.symbol_info_tick(self.symbol)
            open_price = symbol_info_tick.ask if trade_type_int == 1 else symbol_info_tick.bid
            mt5_order_type = mt5.ORDER_TYPE_BUY if trade_type_int == 1 else mt5.ORDER_TYPE_SELL

            trade_req = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': self.symbol,
                'volume': self.lots,
                'type': mt5_order_type,
                'type_filling': mt5.ORDER_FILLING_FOK,
                'type_time': mt5.ORDER_TIME_GTC,
                'price': open_price,
                'deviation': 5,
                'magic': self.ea_id,
                'comment': f'ForexMachine trade open from {self.name} strategy'
            }
            trade_resp = mt5.order_send(trade_req)

            if trade_resp.retcode == mt5.TRADE_RETCODE_DONE or trade_resp.retcode == mt5.TRADE_RETCODE_DONE \
                    or trade_resp.retcode == mt5.TRADE_RETCODE_DONE_PARTIAL:
                # 'mt5_open_timestamp' not accounting for the UTC offset of mt5 terminal (usually 2 hours)
                self.trades[trade_resp.order] = {
                    'trade_request': trade_req,
                    'trade_response': trade_resp,
                    'look_to_close': False,
                    'mt5_open_timestamp': symbol_info_tick.time
                }
                if additional_dict is not None:
                    self.trades[trade_resp.order].update(additional_dict)
                print(f'{self.name} STRATEGY SUCCESSFULLY PLACED {self.trade_decision_strings[trade_type_int]} ORDER, '
                      f'RETCODE: {trade_resp.retcode}, COMMENT: {trade_resp.comment}')
                return True
            else:
                print(f'{self.name} STRATEGY FAILED TO PLACE {self.trade_decision_strings[trade_type_int]} ORDER, '
                      f'RETCODE: {trade_resp.retcode}, COMMENT: {trade_resp.comment}')
        return False

    def close_trade(self, order_ticket):
        trade_dict = self.trades[order_ticket]
        trade_type_int = 1 if trade_dict['trade_request']['type'] == mt5.ORDER_TYPE_BUY else 0
        symbol_info_tick = mt5.symbol_info_tick(self.symbol)
        close_price = symbol_info_tick.bid if trade_type_int == 1 else symbol_info_tick.ask
        mt5_order_type = mt5.ORDER_TYPE_SELL if trade_type_int == 1 else mt5.ORDER_TYPE_BUY

        trade_req = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': self.symbol,
            'volume': trade_dict['trade_response'].volume,
            'type': mt5_order_type,
            'type_filling': mt5.ORDER_FILLING_FOK,
            'type_time': mt5.ORDER_TIME_GTC,
            'price': close_price,
            'deviation': 10,
            'magic': self.ea_id,
            'comment': f'ForexMachine trade close from {self.name} strategy'
        }
        trade_resp = mt5.order_send(trade_req)

        position_info = mt5.positions_get(ticket=order_ticket)
        if trade_resp.retcode == mt5.TRADE_RETCODE_DONE or trade_resp.retcode == mt5.TRADE_RETCODE_DONE:
            print(f'{self.name} STRATEGY SUCCESSFULLY ClOSED {self.trade_decision_strings[trade_type_int]} ORDER, '
                  f'PROFIT: {position_info.profit}, MT5 OPEN TIMESTAMP: {trade_dict["mt5_open_timestamp"]}, '
                  f'RETCODE: {trade_resp.retcode}, COMMENT: {trade_resp.comment}')
            del self.trades[order_ticket]
            return True
        else:
            print(f'{self.name} STRATEGY FAILED TO CLOSE {self.trade_decision_strings[trade_type_int]} ORDER, '
                  f'PROFIT: {position_info.profit}, MT5 OPEN TIMESTAMP: {trade_dict["mt5_open_timestamp"]}, '
                  f'RETCODE: {trade_resp.retcode}, COMMENT: {trade_resp.comment}\nRECOMMEND CLOSING MANUALLY')
        return False

    def finish_up(self):
        mt5.shutdown()
        return True

    def mt5_initialized(self):
        pass

    def process_first_bars(self, data_q, feature_indices, all_features_filled_idx):
        pass

    def process_new_data(self, data_q, feature_indices, processing_immediately):
        pass


class IchiCloudStrategy(TradeStrategy):
    name = f'ichi-cloud'

    def __init__(self, symbol='EURUSD', timeframe='H1', fast_ma_window=7, lots_per_trade=0.2, lstm_seq_len=128,
                 ichi_settings=(9, 30, 60), profit_noise_percent=0.0012, fast_ma_model_path=None, xgb_model_path=None,
                 fast_ma_diff_threshold=0.01, decision_prob_diff_thresh=0.5, bar_buffer_size=300, model_files_path=None,
                 train_data_start_iso=research.TRAIN_DATA_START_ISO, train_data_end_iso=research.TRAIN_DATA_END_ISO,
                 ma_cols=None, pc_cols=None, normalization_groups=None, open_trade_sigs=None, tf_force_cpu=False,
                 models_features_names=None, max_concurrent_trades=np.inf, process_immediately=False,
                 profit_in_quote_currency=True, pip_resolution=None, check_if_market_is_closed=False):
        self.symbol = symbol.upper()
        self.timeframe = timeframe.upper()
        self.lots_per_trade = lots_per_trade
        self.tenkan_period, self.kijun_period, self.senkou_b_period = ichi_settings
        self.profit_noise_percent = profit_noise_percent
        self.fast_ma_diff_threshold = fast_ma_diff_threshold
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
        self.profit_in_quote_currency = profit_in_quote_currency  # if false, consider profit in base currency
        self.pip_resolution = pip_resolution
        self.check_if_market_is_closed = check_if_market_is_closed
        self.tf_force_cpu = tf_force_cpu
        self.ma_cols_idx_set = None
        self.pc_cols_idx_set = None
        self.profit_noise = None
        self.pip_value = None
        self.indicators = ['ichimoku']
        self.features = ['ichimoku_signals']
        self.xgb_model_perc_chngs_q = deque()
        self.fast_ma_model_preds_q = deque()
        self.fast_ma_model_rows_q = deque()
        self.fast_ma_model_seq_q = deque()
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
            self.xgb_model_path = self.model_files_path / f'{self.symbol}-{self.timeframe}_0.0082-min_profit_0.2-lots_right-cur_side' \
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

    def mt5_initialized(self):
        symbol_info = mt5.symbol_info('EURUSD')
        self.profit_noise = self.profit_noise_percent * self.lots_per_trade * symbol_info.trade_contract_size
        if self.pip_resolution is None:
            self.pip_resolution = 10 ** -(symbol_info.digits - 1)
        self.pip_value = symbol_info.trade_contract_size * self.lots_per_trade * self.pip_resolution

    def process_first_bars(self, data_q, feature_indices, all_features_filled_idx):
        if self.ma_cols_idx_set is None:
            self.ma_cols_idx_set = {feature_indices[feat_name] for feat_name in self.ma_cols}
            self.pc_cols_idx_set = {feature_indices[feat_name] for feat_name in self.pc_cols}

        for i in range(all_features_filled_idx + self.fast_ma_window - 1, len(data_q)):
            fast_ma_model_input_row = research.apply_moving_avg_q(data_q, self.ma_cols_idx_set,
                                                                  window=self.fast_ma_window,
                                                                  reverse_start_idx=len(data_q) - i)
            self.fast_ma_model_rows_q.append(fast_ma_model_input_row)
            if len(self.fast_ma_model_rows_q) > 2:
                self.fast_ma_model_rows_q.popleft()

            if len(self.fast_ma_model_rows_q) > 1:
                fast_ma_perc_chng = research.apply_perc_change_list(self.fast_ma_model_rows_q[-2],
                                                                    self.fast_ma_model_rows_q[-1],
                                                                    self.pc_cols_idx_set)
                fast_ma_normalized_perc_chng = research.normalize_data_list(fast_ma_perc_chng, self.norm_terms,
                                                                            self.lfg.feature_indices_keys)
                fast_ma_normalized_perc_chng = self.get_reordered_feature_row(fast_ma_normalized_perc_chng,
                                                                              features=self.models_features_names)
                self.fast_ma_model_seq_q.append(fast_ma_normalized_perc_chng)
                if len(self.fast_ma_model_seq_q) > self.lstm_seq_len:
                    self.fast_ma_model_seq_q.popleft()

                if len(self.fast_ma_model_seq_q) == self.lstm_seq_len:
                    fast_ma_pred = self.fast_ma_model.predict(np.array([self.fast_ma_model_seq_q]))[0][0]
                    self.fast_ma_model_preds_q.append(fast_ma_pred)
                    # only really need 2 preds in self.fast_ma_model_preds_q for current strategy
                    # implementation but whatever
                    if len(self.fast_ma_model_preds_q) > self.fast_ma_window:
                        self.fast_ma_model_preds_q.popleft()

    def process_new_data(self, data_q, feature_indices, processing_immediately):
        # if processing_immediately is True then the most recent bars were already
        # processed in self.process_first_bars() so no need to process most recent bar
        if not processing_immediately:
            fast_ma_model_input_row = research.apply_moving_avg_q(data_q, self.ma_cols_idx_set,
                                                                  window=self.fast_ma_window)
            self.fast_ma_model_rows_q.append(fast_ma_model_input_row)
            # since self.process_first_bars() should have been ran before this method we
            # do not have to check if the len of self.fast_ma_model_rows_q is greater than 2 so just pop
            self.fast_ma_model_rows_q.popleft()
            fast_ma_perc_chng = research.apply_perc_change_list(self.fast_ma_model_rows_q[-2],
                                                                self.fast_ma_model_rows_q[-1],
                                                                self.pc_cols_idx_set)
            fast_ma_normalized_perc_chng = research.normalize_data_list(fast_ma_perc_chng, self.norm_terms,
                                                                        self.lfg.feature_indices_keys)
            fast_ma_normalized_perc_chng = self.get_reordered_feature_row(fast_ma_normalized_perc_chng,
                                                                          features=self.models_features_names)
            self.fast_ma_model_seq_q.append(fast_ma_normalized_perc_chng)
            if len(self.fast_ma_model_seq_q) > self.lstm_seq_len:
                self.fast_ma_model_seq_q.popleft()

            fast_ma_pred = self.fast_ma_model.predict(np.array([self.fast_ma_model_seq_q]))[0][0]
            self.fast_ma_model_preds_q.append(fast_ma_pred)
            if len(self.fast_ma_model_preds_q) > self.fast_ma_window:
                self.fast_ma_model_preds_q.popleft()

        if len(self.trades) > 0:
            symbol_info_tick = mt5.symbol_info_tick(self.symbol)
            for trade_order_ticket in self.trades:
                trade_dict = self.trades[trade_order_ticket]
                trade_type = 1 if trade_dict['trade_request']['type'] == mt5.ORDER_TYPE_BUY else 0
                close_price = symbol_info_tick.ask if trade_type == 0 \
                    else symbol_info_tick.bid
                scaled_profit_noise = self.profit_noise if not self.profit_in_quote_currency \
                    else self.profit_noise * close_price

                profit = research.get_profit(close_price, trade_dict['trade_response'].price, self.pip_value,
                                             self.pip_resolution, in_quote_currency=self.profit_in_quote_currency)

                if abs(profit) >= scaled_profit_noise:
                    trade_dict['look_to_close'] = True

                if trade_dict['look_to_close']:
                    fast_ma_pred_diff = self.fast_ma_model_preds_q[-1] - self.fast_ma_model_preds_q[-2]
                    if abs(fast_ma_pred_diff) >= self.fast_ma_diff_threshold:
                        # (MA pct_change is decreasing on a long trade)
                        # or (MA pct_change is increasing on a short trade)
                        if (fast_ma_pred_diff < 0 and trade_type == 1) \
                                or (fast_ma_pred_diff > 0 and trade_type == 0):
                            self.close_trade(trade_dict['trade_response'].order)

        open_trade = False
        for sig in self.open_trade_sigs:
            sig_i = feature_indices[sig]
            if data_q[-1][sig_i] != 0:
                open_trade = True
                break

        if open_trade or 1:
            xgb_model_input_row = research.apply_perc_change_list(data_q[-2], data_q[-1], cols_set=self.pc_cols_idx_set)
            xgb_model_input_row = self.get_reordered_feature_row(xgb_model_input_row,
                                                                 features=self.models_features_names)
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
                self.open_trade(decision_label)


def run_trade_strategy(strategy, strategy_kwargs, strat_name, proc_conn, mt5_login_info, exit_event):
    strategy = strategy(**strategy_kwargs)

    if not strategy.setup_trade_strategy_process(strat_name, proc_conn, mt5_login_info, exit_event):
        return False

    if hasattr(strategy, 'tf_force_cpu') and strategy.tf_force_cpu:
        with tf.device('/CPU:0'):
            if not strategy.run():
                return False
    else:
        if not strategy.run():
            return False

    if not strategy.finish_up():
        logger.error('Unable to properly finish up strategy')
        return False

    return True


def enable_mt5_auto_trading_func(mt5_terminal_path):
    live_trading_files_dir = util.get_live_trade_files_dir()
    mt5_config_path = live_trading_files_dir / 'ForexMachine_mt5_config.ini'
    if not mt5_config_path.is_file():
        with open(mt5_config_path, 'w') as mt5_config_fp:
            mt5_config_fp.write('[Experts]'
                                '\nAllowLiveTrading=1'
                                '\nEnabled=1'
                                '\nAccount=0'
                                '\nProfile=0')

    # this function should not return because this will launch the mt5 terminal program which stays alive until closed
    subprocess.run(f'"{mt5_terminal_path}" "/config:{mt5_config_path}"')


if __name__ == '__main__':
    # mt5.initialize()
    #
    # ts = mt5.copy_rates_from_pos('EURUSD', mt5.TIMEFRAME_M1, 0, 300)[0][0]
    #
    # print('yo 2', (datetime.fromtimestamp(ts, tz=timezone.utc)
    #                - timedelta(hours=2)))

    # symbol = 'EURUSD'
    # symbol_info_tick = mt5.symbol_info_tick(symbol)
    # symbol_info = mt5.symbol_info(symbol)
    # lots_per_trade = 0.2
    # pip_resolution = 0.0001
    # open_price = 1.34257
    #
    # trade_req = {
    #     'action': mt5.TRADE_ACTION_DEAL,
    #     'symbol': symbol,
    #     'volume': 0.1,
    #     'type': mt5.ORDER_TYPE_BUY,
    #     'type_filling': mt5.ORDER_FILLING_FOK,
    #     'type_time': mt5.ORDER_TIME_GTC,
    #     'price': symbol_info_tick.ask,
    #     'magic': -5151554997339335449 % ((sys.maxsize + 1) * 2),
    #     'comment': 'ForexMachine trade open'
    # }
    #
    # s = time.time()
    # trade_resp = mt5.order_send(trade_req)
    # print(f'time to open trade: {time.time()-s} sec')
    # print(trade_resp)
    # print(symbol_info_tick)

    # pip_value = symbol_info.trade_contract_size * lots_per_trade * pip_resolution
    #
    # s = time.time_ns()
    # p1 = research.get_profit(symbol_info_tick.ask, open_price, pip_value, pip_resolution, in_quote_currency=True)
    # print(f'time to get profit 1 {p1}: {time.time_ns() - s} ns')
    #
    # s = time.time()
    # p2 = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, 'EURUSD', lots_per_trade, open_price, symbol_info_tick.ask)
    # print(f'time to get profit 2 {p2}: {time.time_ns() - s} ns')
    #
    # print(symbol_info_tick.ask)
    # print(10 ** -(symbol_info.digits - 1))
    # print(symbol_info.trade_contract_size * lots_per_trade * (10 ** -(symbol_info.digits - 1)))
    #
    # mt5.shutdown()

    tb = TradeBot()

    if not tb.init_mt5(enable_mt5_auto_trading=True):
        quit()

    strategy_kwargs = {
        'timeframe': 'h1',
        'process_immediately': True,
        'check_if_market_is_closed': False,
        'tf_force_cpu': True,
        'bar_buffer_size': 1440
    }

    name1 = tb.run_strategy(IchiCloudStrategy, strategy_kwargs=strategy_kwargs, name_suffix='1')
    # name2 = tb.run_strategy(IchiCloudStrategy, strategy_kwargs=strategy_kwargs, name_suffix='2')

    time.sleep(185)

    tb.stop_strategy(name1)
    # tb.stop_strategy(name2)
