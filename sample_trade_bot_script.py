import time
from ForexMachine import TradeBot, IchiCloudStrategy

tb = TradeBot(debug_mode=True)

tb.init_mt5(enable_mt5_auto_trading=True)

ichi_strategy_kwargs = {'tf_force_cpu': True}
base_strategy_kwargs = {'process_immediately': True, 'check_if_market_is_closed': False}

ichi_strat_name = tb.run_strategy(IchiCloudStrategy, strategy_kwargs=ichi_strategy_kwargs, base_strategy_kwargs=base_strategy_kwargs)

# run for a month
time.sleep(60*60*24*30)

# save contents of the strategy's feature queue to a csv file
tb.send_command(ichi_strat_name, 'dump_data_q', args=('./ichi-cloud_strategy_data_q.csv',))

# run for another month
time.sleep(60*60*24*30)

tb.stop_strategy(ichi_strat_name)
