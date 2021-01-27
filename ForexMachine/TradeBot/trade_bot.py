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
from pathlib import Path
import configparser
import subprocess
import os
import hashlib

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

TIMEFRAMES = {
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
TRADE_DECISION_STRINGS = {1: 'buy', 0: 'sell'}


class TradeBot:
    def __init__(self, debug_mode=False):
        self.strats = {}
        self.mt5_login_info = {}
        self.debug_mode = debug_mode
        if self.debug_mode:
            logger.setLevel(level=util.LOGGER_LEVELS['DEBUG'])

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
                'Please enter MT5 broker server name (i.e. "MetaQuotes-Demo", "TradersWay-Live"): ')
        if 'login_id' not in self.mt5_login_info:
            while True:
                try:
                    login_id = int(input('Please enter MT5 account login ID: '))
                    self.mt5_login_info['login_id'] = login_id
                    break
                except ValueError:
                    print('Invalid login ID, ID must be interpretable as an integer')
        if 'password' not in self.mt5_login_info:
            self.mt5_login_info['password'] = getpass.getpass(prompt='Please enter MT5 account password: ')

        return True

    def run_strategy(self, strategy, strategy_kwargs={}, base_strategy_kwargs={}, name=None):
        if name is None:
            name = strategy.default_name
        if name in self.strats:
            if self.strats[name]['process'].is_alive():
                logger.error(f'Strategy already running with name: {name}')
                return
            else:
                del self.strats[name]

        parent_conn, child_conn = mltp.Pipe(duplex=True)
        exit_event = mltp.Event()
        debug_exit_event = mltp.Event()
        proc = mltp.Process(target=run_trade_strategy, args=(strategy, strategy_kwargs, base_strategy_kwargs,
                                                             name, child_conn, exit_event, self.mt5_login_info,
                                                             self.debug_mode, debug_exit_event))
        self.strats[name] = {
            'parent_connection': parent_conn,
            'child_connection': child_conn,
            'process': proc,
            'strategy_kwargs': strategy_kwargs,
            'exit_event': exit_event,
            'debug_exit_event': debug_exit_event
        }
        proc.start()
        return name

    def send_command(self, strat_name, cmd, args=()):
        if not isinstance(args, tuple):
            logger.error(f'args must be a tuple: {args}')
            return False
        self.strats[strat_name]['parent_connection'].send((cmd, args))
        self.strats[strat_name]['debug_exit_event'].set()
        return True

    def stop_strategy(self, name):
        if name not in self.strats:
            logger.error(f'No strategy running with name: {name}')
            return False
        self.strats[name]['exit_event'].set()
        self.strats[name]['debug_exit_event'].set()
        self.strats[name]['process'].join()
        del self.strats[name]
        print(f'Stopped strategy with name: {name}')
        return True


class TradeStrategy:
    default_name = 'default_strategy'

    __name = None
    __proc_conn = None
    __mt5_terminal_info = None
    __mt5_account_info = None
    __mt5_timeframe = None
    __update_delta = None  # in secs
    __exit_event = None
    __debug_exit_event = None
    __last_completed_bar_timestamp = None
    __lfg = None
    __trades = {}
    __pip_resolution = None
    __pip_value = None
    __trades_filepath = None
    __commands = None

    def __init__(self, symbol=None, timeframe=None, ea_id=None, mt5_login_info=None, bar_buffer_size=300,
                 detect_utc_offset=False, utc_offset=2, max_concurrent_trades=np.inf, indicators=[], features=[],
                 custom_feature_settings={}, process_immediately=False, check_if_market_is_closed=True,
                 lots_per_trade=None, debug_mode=None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.ea_id = ea_id
        self.mt5_login_info = mt5_login_info
        self.bar_buffer_size = bar_buffer_size  # size of bar q
        self.detect_utc_offset = detect_utc_offset
        self.utc_offset = utc_offset  # utc offset of mt5 terminal, +2 hours for me on east coast of states but maybe diff for others
        self.max_concurrent_trades = max_concurrent_trades
        self.indicators = indicators
        self.features = features
        self.custom_feature_settings = custom_feature_settings
        self.process_immediately = process_immediately
        self.check_if_market_is_closed = check_if_market_is_closed
        self.lots_per_trade = lots_per_trade
        self.debug_mode = debug_mode

    def get_strat_name(self):
        return self.__name

    def get_terminal_info(self):
        return self.__mt5_terminal_info

    def get_account_info(self):
        return self.__mt5_account_info

    def get_trades(self):
        return self.__trades

    def get_lfg_feature_indices_keys(self):
        return self.__lfg.feature_indices_keys

    def dump_data_q(self, path=None):
        data_q_df = pd.DataFrame(self.__lfg.data_q, columns=self.get_lfg_feature_indices_keys())
        if path is None:
            path = f'./{self.__name}_strategy_data_q.csv'
        data_q_df.to_csv(path)
        print(f'Dumped data queue of {self.__name} strategy to {Path(path).resolve()}')

    def setup_trade_strategy_process(self, strat_name, proc_conn, exit_event, mt5_login_info, base_strategy_kwargs,
                                     debug_mode, debug_exit_event):
        self.__name = strat_name
        self.__proc_conn = proc_conn
        self.__exit_event = exit_event
        self.__debug_exit_event = debug_exit_event
        self.mt5_login_info = mt5_login_info

        # sync up default class attributes with any child instances
        default_strat_attrs = TradeStrategy().__dict__
        for key, val in default_strat_attrs.items():
            if key not in self.__dict__:
                self.__dict__[key] = val
                if key in base_strategy_kwargs:
                    self.__dict__[key] = base_strategy_kwargs[key]

        self.symbol = self.symbol.upper()
        self.timeframe = self.timeframe.upper()
        self.__mt5_timeframe, self.__update_delta = TIMEFRAMES[self.timeframe]
        self.lots_per_trade = float(self.lots_per_trade)

        # set debug mode if not specified in base_strategy_kwargs
        if self.debug_mode is None:
            self.debug_mode = debug_mode
        if self.debug_mode:
            logger.setLevel(level=util.LOGGER_LEVELS['DEBUG'])
            logger.debug(f'{self.__name} strategy is in debug mode')

        # set up commands dict
        self.__commands = {
            'dump_data_q': self.dump_data_q
        }

        # use bad 8-byte hash on name of strat to get EA ID if not specified
        if self.ea_id is None:
            self.ea_id = int(hashlib.blake2b(bytes(self.__name, encoding='utf-8'), digest_size=8).hexdigest(), base=16)

        # login and initialize connection to mt5 terminal
        if not mt5.initialize(login=self.mt5_login_info['login_id'], password=self.mt5_login_info['password'],
                              server=self.mt5_login_info['trade_server']):
            logger.error(f'Failed to initialize MT5 terminal for {self.__name} strategy, error:\n{mt5.last_error()}\n')
            return False
        else:
            self.__mt5_terminal_info = mt5.terminal_info()
            if self.__mt5_terminal_info is None:
                logger.error(f'Call to mt5.terminal_info() returned None, error:\n{mt5.last_error()}')
                return False
            self.__mt5_account_info = mt5.account_info()
            if self.__mt5_account_info is None:
                logger.error(f'Call to mt5.account_info() returned None, error:\n{mt5.last_error()}')
                return False
            print(f'Successfully initialized MT5 terminal for {self.__name} strategy'
                  f'\nMT5 terminal info: {self.__mt5_terminal_info}\nMT5 account info: {self.__mt5_account_info}')

        # load in trades from disk if strategy had been previously shutdown w/ open ones
        self.__trades_filepath = util.get_live_trade_files_dir() / f'{self.__name}_{self.ea_id}_trades.pickle'
        if self.__trades_filepath.is_file():
            with open(self.__trades_filepath, 'rb') as trades_fp:
                self.__trades = pickle.load(trades_fp)

            closed_trades = []
            for order_ticket in self.__trades:
                positions_info = mt5.positions_get(ticket=order_ticket)
                if positions_info is None:
                    logger.error(f'Call to mt5.positions_get(ticket={order_ticket}) '
                                 f'returned None, error:\n{mt5.last_error()}')
                    return False
                if len(positions_info) == 0:
                    closed_trades.append(order_ticket)
            for order_ticket in closed_trades:
                del self.__trades[order_ticket]

        if len(self.__trades) > 0:
            print(f'{self.__name} strategy loaded in {len(self.__trades)} active trades\nTrades: {self.__trades}')

        # get info about currency pair
        if self.symbol is not None:
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f'Call to mt5.symbol_info({self.symbol}) returned None, error:\n{mt5.last_error()}')
                return False
            self.__pip_resolution = 10 ** -(symbol_info.digits - 1)
            if self.lots_per_trade is None:
                self.lots_per_trade = symbol_info.volume_step
            self.__pip_value = symbol_info.trade_contract_size * self.lots_per_trade * self.__pip_resolution

        # check if market is closed if specified to, and sleep if so
        market_closed = False
        if self.check_if_market_is_closed:
            market_closed = self.is_market_closed()
        if market_closed:
            self.sleep_till_market_open()

        # get utc offset of mt5 terminal once market is open if it isn't already
        if self.detect_utc_offset:
            self.utc_offset = self.get_server_utc_offset()
        if self.utc_offset is None:
            logger.error('UTC offset is not None')
            return False

        self.strategy_process_setup()

        return True

    def calculate_profit(self, close_price, open_price, in_quote_currency):
        return research.get_profit(close_price, open_price, self.__pip_value, self.__pip_resolution, in_quote_currency)

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
        print(f'Sleeping {self.__name} strategy until {self.symbol} market is open on Sunday at 10:00 PM GMT '
              f'in {sleep_time / 60 / 60} hours, goodnight')
        self.__exit_event.wait(sleep_time)

        while True and not self.__exit_event.is_set():
            if not self.is_market_closed():
                break
            self.__exit_event.wait(60 * 60)
            print(f'{self.symbol} market appears to still be closed at {datetime.now(tz=timezone.utc)},'
                  f' sleeping {self.__name} strategy for another hour')

    def get_server_utc_offset(self):
        latest_bar = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1, 0, 1)
        if latest_bar is None:
            logger.error(f'Call to mt5.copy_rates_from_pos({self.symbol, mt5.TIMEFRAME_H1, 0, 1}) '
                         f'returned None, error:\n{mt5.last_error()}')
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
                       - timedelta(hours=self.utc_offset) + timedelta(seconds=self.__update_delta))
        update_delta = (next_bar_dt - datetime.now(tz=timezone.utc)).total_seconds()
        return update_delta

    def debug_wait(self, duration):
        start_time = time.time()
        time_elapsed = 0

        while duration > time_elapsed:
            self.__debug_exit_event.wait(duration-time_elapsed)

            # if there is a command sent from parent TradeBot proc call command
            if self.__proc_conn.poll():
                self.__debug_exit_event.clear()  # reset event flag

                try:
                    cmd, args = self.__proc_conn.recv()
                    func = self.__commands[cmd]
                    func(*args)
                except Exception as e:
                    logger.error(f'Failed to call "{cmd}" with args {args} on {self.__name} strategy, exception: {e}')

                time_elapsed += time.time() - start_time
                start_time = time.time()
            # otherwise just exit
            else:
                break

    def run(self):
        if self.__exit_event.is_set():
            return True

        first_bars = mt5.copy_rates_from_pos(self.symbol, self.__mt5_timeframe, 0, self.bar_buffer_size + 1)
        if first_bars is None:
            logger.error(f'Call to mt5.copy_rates_from_pos('
                         f'{self.symbol, self.__mt5_timeframe, 0, self.bar_buffer_size + 1}) '
                         f'returned None, error:\n{mt5.last_error()}')
            return False

        self.__lfg = live_trading.LiveFeatureGenerator(indicators=self.indicators, features=self.features,
                                                       utc_offset=self.utc_offset,
                                                       custom_settings=self.custom_feature_settings)
        first_completed_bars = first_bars[:-1]
        if not self.__lfg.process_first_bars(first_completed_bars):
            logger.error(f'Live feature generator unable to process first bars')
            return False
        self.process_first_bars(self.__lfg.data_q, self.__lfg.feature_indices, self.__lfg.all_features_filled_idx)

        if self.__exit_event.is_set():
            return True

        if self.process_immediately:
            print(f'Immediate start: {self.__name} strategy starting to process {self.symbol} '
                  f'{self.timeframe} price bars')
            self.process_new_data(self.__lfg.data_q, self.__lfg.feature_indices, True)

        next_update_delta = self.get_next_update_delta(first_bars[-1][0])
        if next_update_delta > 0:
            print(f'{self.__name} strategy will begin processing {self.symbol} {self.timeframe} price bars'
                  f' {"again " if self.process_immediately else ""}in {next_update_delta / 60} minutes')
            if self.debug_mode:
                self.debug_wait(next_update_delta)
            else:
                self.__exit_event.wait(next_update_delta)
        else:
            if not self.process_immediately:
                print(f'{self.__name} strategy has begun processing {self.symbol} {self.timeframe} price bars')

        self.__last_completed_bar_timestamp = first_bars[-2][0]
        while not self.__exit_event.is_set():
            latest_bars = mt5.copy_rates_from_pos(self.symbol, self.__mt5_timeframe, 0, 2)
            if latest_bars is None:
                logger.error(f'Call to mt5.copy_rates_from_pos({self.symbol, self.__mt5_timeframe, 0, 2})'
                             f' returned None, error:\n{mt5.last_error()}')
                return False

            retries = 0
            while latest_bars[-2][0] == self.__last_completed_bar_timestamp and retries < 120:
                self.__exit_event.wait(0.2)
                latest_bars = mt5.copy_rates_from_pos(self.symbol, self.__mt5_timeframe, 0, 2)
                if latest_bars is None:
                    logger.error(f'Call to mt5.copy_rates_from_pos({self.symbol, self.__mt5_timeframe, 0, 2})'
                                 f' returned None, error:\n{mt5.last_error()}')
                    return False
                retries += 1

            if latest_bars[-2][0] == self.__last_completed_bar_timestamp:
                market_closed = self.is_market_closed()
                if market_closed:
                    self.sleep_till_market_open()
                    continue

            if not self.__lfg.process_new_bar(latest_bars[-2]):
                logger.error(f'Live feature generator unable to process new bar:\n{latest_bars[-2]}')
                return False
            self.process_new_data(self.__lfg.data_q, self.__lfg.feature_indices, False)

            next_update_delta = self.get_next_update_delta(latest_bars[-1][0])
            if next_update_delta > 0:
                if self.debug_mode:
                    logger.debug(f'{self.__name} strategy will begin processing {self.symbol} {self.timeframe} '
                                 f'price bars again in {next_update_delta / 60} minutes')
                    self.debug_wait(next_update_delta)
                else:
                    self.__exit_event.wait(next_update_delta)

            self.__last_completed_bar_timestamp = latest_bars[-2][0]
        return True

    # returns False by default if error with MetaTrader5 package
    def is_market_closed(self):
        print(f'Checking if {self.symbol} market is open...')
        last_m1_bar = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, 1)
        if last_m1_bar is None:
            logger.error(f'Call to mt5.copy_rates_from_pos({self.symbol, mt5.TIMEFRAME_M1, 0, 1})'
                         f' returned None, error:\n{mt5.last_error()}')
            return False
        for _ in range(2):
            self.__exit_event.wait(70)
            latest_m1_bar = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, 1)
            if last_m1_bar is None:
                logger.error(f'Call to mt5.copy_rates_from_pos({self.symbol, mt5.TIMEFRAME_M1, 0, 1})'
                             f' returned None, error:\n{mt5.last_error()}')
                return False
            if last_m1_bar[0][0] != latest_m1_bar[0][0]:
                print(f'{self.symbol} market does not seem to be closed')
                return False
            last_m1_bar = latest_m1_bar
        print(f'{self.symbol} market seems to be closed')
        return True

    def get_reordered_feature_row(self, row, features):
        row = [row[self.__lfg.feature_indices[feat_name]] for feat_name in features]
        return row

    def open_trade(self, trade_type_int, custom_trade_req=None, additional_dict=None, log_msg=None):
        if len(self.__trades) < self.max_concurrent_trades:
            symbol_info_tick = mt5.symbol_info_tick(self.symbol)
            if symbol_info_tick is None:
                logger.error(f'Call to mt5.symbol_info_tick({self.symbol}) returned None, error:\n{mt5.last_error()}')
                return False

            open_price = symbol_info_tick.ask if trade_type_int == 1 else symbol_info_tick.bid
            mt5_order_type = mt5.ORDER_TYPE_BUY if trade_type_int == 1 else mt5.ORDER_TYPE_SELL

            trade_req = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': self.symbol,
                'volume': self.lots_per_trade,
                'type': mt5_order_type,
                'type_filling': mt5.ORDER_FILLING_FOK,
                'type_time': mt5.ORDER_TIME_GTC,
                'price': open_price,
                'deviation': 10,
                'magic': self.ea_id,
                'comment': 'ForexMachine trade open'  # there is a length limit for comments
            }

            # args sent to call_mt5_func() will not work if the arg is a dictionary
            trade_resp = mt5.order_send(trade_req)
            if trade_resp is None:
                logger.error(f'Call to mt5.order_send({trade_req}) returned None, error:\n{mt5.last_error()}')
                return False

            if custom_trade_req is not None:
                trade_req.update(custom_trade_req)

            if trade_resp.retcode == mt5.TRADE_RETCODE_DONE or trade_resp.retcode == mt5.TRADE_RETCODE_DONE \
                    or trade_resp.retcode == mt5.TRADE_RETCODE_DONE_PARTIAL:
                # 'mt5_open_timestamp' not accounting for the UTC offset of mt5 terminal (usually 2 hours)
                # also unable to pickle named tuples (trade_resp) because its a class attribute so convert it to dict,
                # explanation: https://stackoverflow.com/a/4678982/10276720
                trade_resp_dict = trade_resp._asdict()
                trade_resp_dict['request'] = trade_resp_dict['request']._asdict()
                self.__trades[trade_resp.order] = {
                    'trade_request': trade_req,
                    'trade_response': trade_resp_dict,
                    'look_to_close': False,
                    'mt5_open_timestamp': symbol_info_tick.time,
                    'log_msg': log_msg
                }
                if additional_dict is not None:
                    self.__trades[trade_resp.order].update(additional_dict)
                print(f'{self.__name} STRATEGY SUCCESSFULLY PLACED {TRADE_DECISION_STRINGS[trade_type_int]} ORDER,'
                      f' RETCODE: {trade_resp.retcode}, COMMENT: {trade_resp.comment}')
                return True
            else:
                print(f'{self.__name} STRATEGY FAILED TO PLACE {TRADE_DECISION_STRINGS[trade_type_int]} ORDER, '
                      f'RETCODE: {trade_resp.retcode}, COMMENT: {trade_resp.comment}')
        else:
            print(f'{self.__name} strategy unable to place any more trades, '
                  f'already at max open trade limit of {self.max_concurrent_trades} trades')
        return False

    def close_trade(self, order_ticket):
        symbol_info_tick = mt5.symbol_info_tick(self.symbol)
        if symbol_info_tick is None:
            logger.error(f'Call to mt5.symbol_info_tick({self.symbol}) returned None, error:\n{mt5.last_error()}')
            return False

        positions_info = mt5.positions_get(ticket=order_ticket)
        if positions_info is None:
            logger.error(f'Call to mt5.positions_get(ticket={order_ticket}) returned None, error:\n{mt5.last_error()}')
            return False
        if len(positions_info) == 0:
            logger.error(f'No trade was found with order ticket number: {order_ticket}, recommend '
                         f'closing trade manually if it\'s believed to still be open')
            del self.__trades[order_ticket]
            return False

        trade_dict = self.__trades[order_ticket]
        trade_type_int = 1 if trade_dict['trade_request']['type'] == mt5.ORDER_TYPE_BUY else 0
        close_price = symbol_info_tick.bid if trade_type_int == 1 else symbol_info_tick.ask
        mt5_order_type = mt5.ORDER_TYPE_SELL if trade_type_int == 1 else mt5.ORDER_TYPE_BUY

        trade_req = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': self.symbol,
            'position': order_ticket,
            'volume': trade_dict['trade_response']['volume'],
            'type': mt5_order_type,
            'type_filling': mt5.ORDER_FILLING_FOK,
            'type_time': mt5.ORDER_TIME_GTC,
            'price': close_price,
            'deviation': 10,
            'magic': self.ea_id,
            'comment': 'ForexMachine trade close'
        }

        # args sent to call_mt5_func() will not work if the arg is a dictionary
        trade_resp = mt5.order_send(trade_req)
        if trade_resp is None:
            logger.error(f'Call to mt5.order_send({trade_req}) returned None, error:\n{mt5.last_error()}')
            return False

        if trade_resp.retcode == mt5.TRADE_RETCODE_DONE or trade_resp.retcode == mt5.TRADE_RETCODE_DONE:
            print(f'{self.__name} STRATEGY SUCCESSFULLY ClOSED {TRADE_DECISION_STRINGS[trade_type_int]} ORDER, '
                  f'PROFIT: {positions_info[0].profit}, MT5 OPEN TIMESTAMP: {trade_dict["mt5_open_timestamp"]}, '
                  f'RETCODE: {trade_resp.retcode}, COMMENT: {trade_resp.comment}')
            del self.__trades[order_ticket]
            return True
        else:
            print(f'{self.__name} STRATEGY FAILED TO CLOSE {TRADE_DECISION_STRINGS[trade_type_int]} ORDER, '
                  f'PROFIT: {positions_info[0].profit}, MT5 OPEN TIMESTAMP: {trade_dict["mt5_open_timestamp"]}, '
                  f'RETCODE: {trade_resp.retcode}, COMMENT: {trade_resp.comment}\nRECOMMEND CLOSING MANUALLY')
        return False

    def finish_up(self):
        closed_trades = []
        for order_ticket in self.__trades:
            positions_info = mt5.positions_get(ticket=order_ticket)
            if positions_info is None:
                logger.error(f'Call to mt5.positions_get(ticket={order_ticket}) '
                             f'returned None, error:\n{mt5.last_error()}')
                break
            if len(positions_info) == 0:
                closed_trades.append(order_ticket)
        for order_ticket in closed_trades:
            del self.__trades[order_ticket]
        mt5.shutdown()
        return True

    # when strategy process is set up and running and MT5 connection is ready
    def strategy_process_setup(self):
        pass

    def process_first_bars(self, data_q, feature_indices, all_features_filled_idx):
        pass

    def process_new_data(self, data_q, feature_indices, processing_immediately):
        pass

    def __del__(self):
        if self.__trades_filepath is not None:
            if len(self.__trades) > 0:
                with open(self.__trades_filepath, 'wb') as trades_fp:
                    pickle.dump(self.__trades, trades_fp)
            elif self.__trades_filepath.is_file():
                os.remove(self.__trades_filepath)


class IchiCloudStrategy(TradeStrategy):
    default_name = f'ichi-cloud'

    def __init__(self, symbol='EURUSD', timeframe='H1', fast_ma_window=7, lots_per_trade=0.2, lstm_seq_len=128,
                 ichi_settings=(9, 30, 60), profit_noise_percent=0.0012, fast_ma_model_path=None, xgb_model_path=None,
                 fast_ma_diff_threshold=0.01, decision_prob_diff_thresh=0.5, bar_buffer_size=300, model_files_path=None,
                 train_data_start_iso=research.TRAIN_DATA_START_ISO, train_data_end_iso=research.TRAIN_DATA_END_ISO,
                 ma_cols=None, pc_cols=None, normalization_groups=None, open_trade_sigs=None, tf_force_cpu=False,
                 models_features_names=None, max_concurrent_trades=np.inf, profit_in_quote_currency=True):
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
        self.profit_in_quote_currency = profit_in_quote_currency  # if false, consider profit in base currency
        self.tf_force_cpu = tf_force_cpu
        self.train_data_start_iso = train_data_start_iso
        self.train_data_end_iso = train_data_end_iso
        self.ma_cols_idx_set = None
        self.pc_cols_idx_set = None
        self.profit_noise = None
        self.norm_terms = None
        self.strat_name = None
        self.indicators = ['ichimoku']
        self.features = ['ichimoku_signals']
        self.xgb_model_perc_chngs_q = deque()
        self.fast_ma_model_preds_q = deque()
        self.fast_ma_model_rows_q = deque()
        self.fast_ma_model_seq_q = deque()
        self.custom_feature_settings = {
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

    def get_normalization_terms(self, train_data_start_iso, train_data_end_iso):
        norm_terms_filepath = self.model_files_path / f'{self.strat_name}_normalization_terms.pickle'

        if norm_terms_filepath.is_file():
            with open(norm_terms_filepath, 'rb') as ntfp:
                norm_terms = pickle.load(ntfp)
            print(f'loaded normalization terms for strategy with name "{self.strat_name}" from {norm_terms_filepath}')
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
                                                            train_data_end_iso, mt5_initialized=True)
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
            print(f'generated normalization terms for strategy with name "{self.strat_name}" from '
                  f'{train_data_start_iso} to {train_data_end_iso} and saved pickle file to {norm_terms_filepath}')

        return norm_terms

    def strategy_process_setup(self):
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logger.error(f'Call to mt5.symbol_info({self.symbol}) returned None, error:\n{mt5.last_error()}')
        self.profit_noise = self.profit_noise_percent * self.lots_per_trade * symbol_info.trade_contract_size

        self.strat_name = self.get_strat_name()
        self.norm_terms = self.get_normalization_terms(self.train_data_start_iso, self.train_data_end_iso)

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
                                                                            self.get_lfg_feature_indices_keys())
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

        trades = self.get_trades()
        trades_to_close = []
        if len(trades) > 0:
            symbol_info_tick = mt5.symbol_info_tick(self.symbol)
            if symbol_info_tick is None:
                logger.error(f'Call to mt5.symbol_info_tick({self.symbol}) returned None, error:\n{mt5.last_error()}')
                return
            for order_ticket in trades:
                trade_dict = trades[order_ticket]
                trade_type = 1 if trade_dict['trade_request']['type'] == mt5.ORDER_TYPE_BUY else 0
                close_price = symbol_info_tick.ask if trade_type == 0 \
                    else symbol_info_tick.bid
                scaled_profit_noise = self.profit_noise if not self.profit_in_quote_currency \
                    else self.profit_noise * close_price

                profit = self.calculate_profit(close_price, trade_dict['trade_response']['price'],
                                               in_quote_currency=self.profit_in_quote_currency)

                if abs(profit) >= scaled_profit_noise:
                    trade_dict['look_to_close'] = True

                if trade_dict['look_to_close']:
                    fast_ma_pred_diff = self.fast_ma_model_preds_q[-1] - self.fast_ma_model_preds_q[-2]
                    if abs(fast_ma_pred_diff) >= self.fast_ma_diff_threshold:
                        # (MA pct_change is decreasing on a long trade)
                        # or (MA pct_change is increasing on a short trade)
                        if (fast_ma_pred_diff < 0 and trade_type == 1) \
                                or (fast_ma_pred_diff > 0 and trade_type == 0):
                            trades_to_close.append(trade_dict['trade_response']['order'])

        for order_ticket in trades_to_close:
            self.close_trade(order_ticket)

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
                                                                        self.get_lfg_feature_indices_keys())
            fast_ma_normalized_perc_chng = self.get_reordered_feature_row(fast_ma_normalized_perc_chng,
                                                                          features=self.models_features_names)
            self.fast_ma_model_seq_q.append(fast_ma_normalized_perc_chng)
            if len(self.fast_ma_model_seq_q) > self.lstm_seq_len:
                self.fast_ma_model_seq_q.popleft()

            fast_ma_pred = self.fast_ma_model.predict(np.array([self.fast_ma_model_seq_q]))[0][0]
            self.fast_ma_model_preds_q.append(fast_ma_pred)
            if len(self.fast_ma_model_preds_q) > self.fast_ma_window:
                self.fast_ma_model_preds_q.popleft()

        trades = self.get_trades()
        trades_to_close = []
        if len(trades) > 0:
            symbol_info_tick = mt5.symbol_info_tick(self.symbol)
            if symbol_info_tick is None:
                logger.error(f'Call to mt5.symbol_info_tick({self.symbol}) returned None, error:\n{mt5.last_error()}')
                return
            for order_ticket in trades:
                trade_dict = trades[order_ticket]
                trade_type = 1 if trade_dict['trade_request']['type'] == mt5.ORDER_TYPE_BUY else 0
                close_price = symbol_info_tick.ask if trade_type == 0 \
                    else symbol_info_tick.bid
                scaled_profit_noise = self.profit_noise if not self.profit_in_quote_currency \
                    else self.profit_noise * close_price

                profit = self.calculate_profit(close_price, trade_dict['trade_response']['price'],
                                               in_quote_currency=self.profit_in_quote_currency)

                if abs(profit) >= scaled_profit_noise:
                    trade_dict['look_to_close'] = True

                if trade_dict['look_to_close']:
                    fast_ma_pred_diff = self.fast_ma_model_preds_q[-1] - self.fast_ma_model_preds_q[-2]
                    if abs(fast_ma_pred_diff) >= self.fast_ma_diff_threshold:
                        # (MA pct_change is decreasing on a long trade)
                        # or (MA pct_change is increasing on a short trade)
                        if (fast_ma_pred_diff < 0 and trade_type == 1) \
                                or (fast_ma_pred_diff > 0 and trade_type == 0):
                            trades_to_close.append(trade_dict['trade_response']['order'])

        for order_ticket in trades_to_close:
            self.close_trade(order_ticket)

        open_trade_causes = []
        for sig in self.open_trade_sigs:
            sig_i = feature_indices[sig]
            if data_q[-1][sig_i] != 0:
                open_trade_causes.append(sig)

        if len(open_trade_causes) > 0:
            xgb_model_input_row = research.apply_perc_change_list(data_q[-2], data_q[-1], cols_set=self.pc_cols_idx_set)
            xgb_model_input_row = self.get_reordered_feature_row(xgb_model_input_row,
                                                                 features=self.models_features_names)
            xgb_model_input = pd.DataFrame([xgb_model_input_row], columns=self.models_features_names)
            xgb_model_input = xgb.DMatrix(xgb_model_input)
            decision_prob = self.xgb_decision_predictor.predict(xgb_model_input)[0]
            decision_label = round(decision_prob)
            decision_prob_diff = abs(decision_label - decision_prob)

            if decision_prob_diff <= self.decision_prob_diff_thresh:
                log_msg = f'{self.strat_name} strategy detected {open_trade_causes} ichimoku signals' \
                          f'\nWill attempt to open a {TRADE_DECISION_STRINGS[decision_label]} trade based ' \
                          f'on classifier probability of {decision_prob}'
                print(log_msg)
                self.open_trade(decision_label, log_msg=log_msg)


def run_trade_strategy(strategy, strategy_kwargs, base_strategy_kwargs, strat_name, proc_conn, exit_event,
                       mt5_login_info, debug_mode, debug_exit_event):
    strategy = strategy(**strategy_kwargs)

    if not strategy.setup_trade_strategy_process(strat_name=strat_name, proc_conn=proc_conn, exit_event=exit_event,
                                                 mt5_login_info=mt5_login_info,
                                                 base_strategy_kwargs=base_strategy_kwargs,
                                                 debug_mode=debug_mode, debug_exit_event=debug_exit_event):
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
    # mt5.initialize(login=38050173, password='kmmiv4ud', server='MetaQuotes-Demo')
    #
    # symbol = "EURUSD"
    #
    # symbol_tick_info = mt5.symbol_info_tick(symbol)
    # trade_req = {
    #     'action': mt5.TRADE_ACTION_DEAL,
    #     'symbol': symbol,
    #     'volume': float(0.01),
    #     'type': mt5.ORDER_TYPE_SELL,
    #     'type_filling': mt5.ORDER_FILLING_FOK,
    #     'type_time': mt5.ORDER_TIME_GTC,
    #     'price': symbol_tick_info.bid,
    #     'deviation': 5,
    #     'magic': 123456,
    #     'comment': f'auto trade test trade open'
    # }
    # trade_resp = mt5.order_send(trade_req)
    # print(type(trade_resp))
    # print(mt5.last_error())
    # print(trade_resp)
    #
    # positions_info = mt5.positions_get(ticket=trade_resp.order)
    # print(positions_info)
    # print(positions_info[0].profit)
    #
    # symbol_tick_info = mt5.symbol_info_tick(symbol)
    # trade_req = {
    #     'action': mt5.TRADE_ACTION_DEAL,
    #     'symbol': symbol,
    #     'position': trade_resp.order,
    #     'volume': float(0.01),
    #     'type': mt5.ORDER_TYPE_BUY,
    #     'type_filling': mt5.ORDER_FILLING_FOK,
    #     'type_time': mt5.ORDER_TIME_GTC,
    #     'price': symbol_tick_info.ask,
    #     'deviation': 5,
    #     'magic': 123456,
    #     'comment': f'auto trade test trade close'
    # }
    # trade_resp = mt5.order_send(trade_req)
    # print(type(trade_resp))
    # print(mt5.last_error())
    # print(trade_resp)
    #
    # positions_info = mt5.positions_get(ticket=trade_resp.order)
    # print(positions_info)
    # print(positions_info[0].profit)
    #
    # quit()

    # date_from = datetime(2021,1,8)
    # date_to = datetime(2021,1,24)
    # orders = mt5.history_orders_get(date_from,date_to)
    # print(orders)
    # print(len(orders))
    #
    # print(mt5.positions_get(ticket=9790141))
    # print(mt5.last_error())
    # print(mt5.orders_get(ticket=9790141))
    # print(mt5.last_error())
    #
    # print(mt5.positions_get(ticket=9789032))
    # print(mt5.last_error())
    # print(mt5.orders_get(ticket=9789032))
    # print(mt5.last_error())
    #
    # print(mt5.positions_get(ticket=12312312))
    # print(mt5.last_error())
    # print(mt5.orders_get(ticket=12312312))
    # print(mt5.last_error())

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

    tb = TradeBot(debug_mode=True)

    if not tb.init_mt5(enable_mt5_auto_trading=True):
        quit()

    ichi_strategy_kwargs = {
        'tf_force_cpu': True,
    }

    base_strategy_kwargs = {
        'process_immediately': True,
        'check_if_market_is_closed': False,
    }

    name1 = tb.run_strategy(IchiCloudStrategy, strategy_kwargs=ichi_strategy_kwargs,
                            base_strategy_kwargs=base_strategy_kwargs)
    # name2 = tb.run_strategy(IchiCloudStrategy, strategy_kwargs=strategy_kwargs, name_suffix='2')

    time.sleep(30)
    tb.send_command(name1, 'dump_data_q')

    time.sleep(60)

    tb.stop_strategy(name1)
    # tb.stop_strategy(name2)
