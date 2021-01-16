import MetaTrader5 as mt5
import multiprocessing as mltp
from datetime import datetime, timedelta
import os
import getpass
import time
from pathlib import Path
from ForexMachine import util
from collections import deque

"""
thoughts:
- 1 TradeBot brain that handles multiple TradeStrategys
- 1 process per TradeStrategy instance
- when TradeStrategy process ends it will save json file of its current state (open trades, their majic numbers, etc.)
  and then when TradeStrategy process starts again will load in that json 
"""
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

    def run_strategy(self, strategy, name=None):
        if name is not None:
            strat = strategy(name=name)
        else:
            strat = strategy()
        name = strat.name
        parent_conn, child_conn = mltp.Pipe(duplex=True)
        proc = mltp.Process(target=run_trade_strategy, args=(strat, child_conn, self.mt5_login_info))
        self.strats[name] = {
            'parent_connection': parent_conn,
            'child_connection': child_conn,
            'process': proc,
            'strategy_instance': strat
        }
        proc.start()
        return name

    def stop_strategy(self, strat_name):
        self.strats[strat_name]['parent_connection'].send('stop')
        self.strats[strat_name]['process'].join()
        print(f'Stopped {self.name} strategy.')


class TradeStrategy:
    def __init__(self):
        self.name = None
        self.proc_conn = None
        self.mt5_terminal_info = None
        self.mt5_account_info = None
        self.mt5_login_info = {}
        self.symbol = None
        self.timeframe = None
        self.data = deque()
        self.update_delta = 0.5  # in sec

    def setup_trade_strategy(self, proc_conn, mt5_login_info):
        self.proc_conn = proc_conn
        self.mt5_login_info = mt5_login_info

        if not mt5.initialize(login=self.mt5_login_info['login_id'], password=self.mt5_login_info['password'],
                              server=self.mt5_login_info['trade_server'], portable=True):
            print(f'Failed to initialize MT5 terminal for {self.name} strategy, error:\n{mt5.last_error()}\n')
        else:
            self.mt5_terminal_info = mt5.terminal_info()
            self.mt5_account_info = mt5.account_info()
            print(f'Successfully initialized MT5 terminal for {self.name} strategy'
                  f'\n{self.mt5_terminal_info}\n{self.mt5_account_info}\n')

    def run(self):
        while True:
            rates = mt5.copy_rates_from_pos('EURUSD', mt5.TIMEFRAME_H1, 0, 2)
            for bar in rates:
                print(datetime.utcfromtimestamp(bar[0]), bar, '\n')
            msg = None
            if self.proc_conn.poll():
                msg = self.proc_conn.recv()

            if msg is not None and msg == 'stop':
                break
            time.sleep(0.5)
        return 0


class IchiCloudStrategy(TradeStrategy):
    def __init__(self, name='ichi-cloud', symbol='EURUSD', timeframe='H1', fast_ma_window=7, lots_per_trade=0.2,
                 ichi_settings=(8, 22, 24), profit_noise_percent=0.004, fast_ma_model_path=None, xgb_model_path=None,
                 fast_ma_diff_thresh=0.05, decision_prob_diff_thresh=0.5):
        self.name = name
        self.symbol = symbol.capitalize()
        self.timeframe = timeframe.capitalize()
        self.lots_per_trade = lots_per_trade
        self.tenkan_period, self.kijun_period, self.senkou_b_period = ichi_settings
        self.profit_noise_percent = profit_noise_percent
        self.fast_ma_diff_thresh = fast_ma_diff_thresh
        self.decision_prob_diff_thresh = decision_prob_diff_thresh
        self.fast_ma_window = fast_ma_window
        if fast_ma_model_path is None:
            fast_ma_model_path = f'./final_{self.symbol}-{self.timeframe}_Bi-LSTM_7-ma_8-22-44-ichi.hdf5'
        if xgb_model_path is None:
            xgb_model_path = f'./{self.symbol}-{self.timeframe}_0.004-min_profit_0.2-lots_right-' \
                             f'cur_side_8-22-24-cb-tk-tkp-sen-chi-ichi_xgb_classifier.json'
        self.fast_ma_model_path = fast_ma_model_path
        self.xgb_model_path = xgb_model_path


def run_trade_strategy(strategy, proc_conn, mt5_login_info):
    strategy.setup_trade_strategy(proc_conn, mt5_login_info)
    strategy.run()


if __name__ == '__main__':
    # mt5.initialize()
    # rates = mt5.copy_rates_from_pos('EURUSD', mt5.TIMEFRAME_H1, 0, 2)
    # mt5.shutdown()
    #
    # for bar in rates:
    #     print(datetime.utcfromtimestamp(bar[0]), bar)

    tb = TradeBot()
    tb.init_mt5()
    name1 = tb.run_strategy(IchiCloudStrategy, name=1)
    name2 = tb.run_strategy(IchiCloudStrategy, name=2)
    time.sleep(5)
    tb.stop_strategy(name1)
    time.sleep(2)
    tb.stop_strategy(name2)
