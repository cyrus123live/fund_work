import ModelTools
import sys
import yfinance as yf

run_directory = 'runs/2024-10-15-21-37-59'

if len(sys.argv) > 1:
    run_directory = sys.argv[1]

history = ModelTools.combine_trade_window_histories(run_directory)

ModelTools.print_parameters(run_directory)
ModelTools.print_stats_from_history(history)
ModelTools.plot_history(history)
