from stable_baselines3 import PPO
from TradingEnv import TradingEnv
import StockData
import ModelTools

import math
import pandas as pd
import datetime as dt
import multiprocessing
import os
from ModelTools import make_dir
import sys
import random
import json

def main():

    '''
    Loops 3 months long:
    - Train from 9-6 months ago -> 20 PPOs, ordered by validation (6-3 months ago) score
        - Train each one just until overfitting by testing each one between rounds of training
    - Trade 3-0 months ago with best from training
    '''

    multiprocessing.set_start_method('spawn')
    manager = multiprocessing.Manager()
    multiprocessing_cores = 1
    if len(sys.argv) > 1:
        multiprocessing_cores = int(sys.argv[1])

    parameters = {
        "starting_month": "2024-1",
        "ending_month": "2024-6",
        "train_months": 3,
        "test_months": 3,
        "trade_months": 3,
        "num_ppo": 12,
        "num_a2c": 12,
        "training_rounds_per_contender": 3,
        "starting_cash": 1000000,
        "ent_coef": 0.01,
        "buy_action_space": "discrete",
        "sell_action_space": "discrete",
        "t": "minutely"
    }

    cash = parameters["starting_cash"]

    # Instatiate run folder and parameters file
    run_start_time = dt.datetime.now()
    run_folder_name = "runs/" + run_start_time.strftime('%Y-%m-%d-%H-%M-%S')
    ModelTools.make_dir(run_folder_name)
    logger = ModelTools.Logger(run_folder_name)
    with open(f"{run_folder_name}/parameters.json", 'w') as f:
        json.dump(parameters, f)

    # Main Loop
    starting_month = dt.datetime(year=int(parameters["starting_month"].split("-")[0]), month=int(parameters["starting_month"].split("-")[1]), day=1)
    ending_month = dt.datetime(year=int(parameters["ending_month"].split("-")[0]), month=int(parameters["ending_month"].split("-")[1]), day=1)
    total_months = math.ceil((ending_month - starting_month).days / 30.44)
    for trade_window_start in [starting_month + pd.DateOffset(months=i) for i in range(0, total_months, parameters["trade_months"])]:

        trade_window_start_time = dt.datetime.now()

        # Figure out monthly windows for trading, testing, and training
        train_window_start = trade_window_start - pd.DateOffset(months=parameters["test_months"] + parameters["train_months"])
        train_window_end = trade_window_start - pd.DateOffset(months=parameters["test_months"], days=1)
        validation_window_start = train_window_start + pd.DateOffset(months=parameters["train_months"])
        validation_window_end = trade_window_start - pd.DateOffset(days=1)
        trade_window_end = trade_window_start + pd.DateOffset(months=parameters["trade_months"]) - pd.DateOffset(days=1)
        
        # Printout dates
        logger.print_out(f"\nStarting round with trading window [{trade_window_start.strftime('%Y-%m-%d')}, {trade_window_end.strftime('%Y-%m-%d')}],")
        if parameters["test_months"] == 0:
            logger.print_out("Validation window is equal to training window,")
        else:
            logger.print_out(f"Validation window [{validation_window_start.strftime('%Y-%m-%d')}, {validation_window_end.strftime('%Y-%m-%d')}],")
        logger.print_out(f"Training window [{train_window_start.strftime('%Y-%m-%d')}, {train_window_end.strftime('%Y-%m-%d')}]\n")

        # Get train, test, and trade data
        train_data = StockData.get_consecutive_months(starting_month=train_window_start, num_months=parameters["train_months"], t=parameters["t"])
        if parameters["test_months"] == 0:
            test_data = train_data
        else:
            test_data = StockData.get_consecutive_months(starting_month=validation_window_start, num_months=parameters["test_months"], t=parameters["t"])
        trade_data = StockData.get_consecutive_months(starting_month=trade_window_start, num_months=parameters["trade_months"], t=parameters["t"])

        # Instantiate trade window folder
        trade_window_folder_name = f"{run_folder_name}/{trade_window_start.strftime('%Y-%m-%d')}"
        make_dir(trade_window_folder_name + "/models")

        logger.print_out(f"Finished initializing, beginning to train contenders.\n")
        training_start_time = dt.datetime.now()

        # Train our contenders using multiprocessing 
        processes = []
        contenders = manager.list()
        for i in range(int(parameters["num_a2c"]) + int(parameters["num_ppo"])):
            seed = int(random.random() * 100000)
            if i < int(parameters["num_a2c"]):
                contender_name = f"{trade_window_folder_name}/models/A2C_{i}"
                p = multiprocessing.Process(target=ModelTools.train, args=("A2C", seed, train_data, test_data, parameters, contender_name, contenders, logger))
            else:
                contender_name = f"{trade_window_folder_name}/models/PPO_{i}"
                p = multiprocessing.Process(target=ModelTools.train, args=("PPO", seed, train_data, test_data, parameters, contender_name, contenders, logger))
            p.start()
            processes.append(p)

            if len(processes) >= multiprocessing_cores:
                for p in processes:
                    p.join()
                processes = []
        for p in processes:
            p.join()
        contenders = list(contenders)
        contenders.sort(key=lambda x: x['score'], reverse=True)

        logger.print_out(f"\nFinished training contenders in {(dt.datetime.now() - training_start_time).seconds} seconds.\n")

        # Print contenders for debugging
        for p in contenders:
            logger.print_out(p)

        # Get best contender and trade with them
        logger.print_out(f"\nStarting trading with model with score {contenders[0]['score']:.2f}")
        trade_window_history = ModelTools.test_model(PPO.load(contenders[0]['model']), trade_data, parameters, cash)
        ModelTools.write_history_to_file(trade_window_history, f"{trade_window_folder_name}/trade_window_history")

        # Update running balance
        cash = trade_window_history.iloc[-1]['portfolio_value'] # Assume bot sells off all stocks at end of 3 month period
        logger.print_out(f"- Finished trading, new running cash total: {cash:.2f}")

        logger.print_out(f"\nFinished date window, total time: {(dt.datetime.now() - trade_window_start_time).seconds} seconds.\n\n")

    combined_history = ModelTools.combine_trade_window_histories(run_folder_name)

    logger.print_out(f"\nFinished run in {(dt.datetime.now() - run_start_time).seconds} seconds.\n")

    ModelTools.print_stats_from_history(combined_history)

if __name__ == "__main__":
    main()