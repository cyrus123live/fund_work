import ModelTools
import sys
import yfinance as yf

# run_directory = 'runs/2024-10-22-16-44-22' # First bitcoin test, one month
# run_directory = 'runs/2024-10-22-16-58-27' # 2020-2024 slaughterfest after 2022
run_directory = 'runs/2024-10-23-20-08-41' # 2023-2024 only killed it but so did underlying asset


if len(sys.argv) > 1:
    run_directory = sys.argv[1]

history = ModelTools.combine_trade_window_histories(run_directory)

ModelTools.print_parameters(run_directory)
ModelTools.print_stats_from_history(history)
ModelTools.plot_history(history)
