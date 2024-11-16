from stable_baselines3 import PPO
from stable_baselines3 import A2C
from TradingEnv import TradingEnv
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import datetime as dt
import StockData
import logging
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
import json
import csv
import math
import random
import numpy as np
import torch
import os

class Logger():
    def __init__(self, run_folder_name):
        self.run_folder_name = run_folder_name

    def print_out(self, s):
        print(s)
        with open(f"{self.run_folder_name}/run.log", 'a') as f:
            f.write(str(s) + "\n")
        

def print_parameters(run_folder_name):
    print("\nrun " + run_folder_name)
    try:
        print("\nParameters:\n")
        with open(f"{run_folder_name}/parameters.json", 'r') as f:
            # print(json.dumps(json.loads(f.read()), indent=1))
            data = json.loads(f.read())
        for d in data:
            print(f"{d}: {data[d]}")
        print("\n")
    except Exception as e:
        print("Parameters File not found")


def combine_trade_window_histories(run_folder_name):

    combined_history = pd.DataFrame()

    folders = [d for d in os.listdir(f"{run_folder_name}") if "20" in d]
    folders.sort(key=lambda x: (int(x.split("-")[0]), int(x.split("-")[1])))

    for f in folders:
        try:
            combined_history = pd.concat([combined_history, read_history_from_file(f"{run_folder_name}/{f}/trade_window_history")])
        except: 
            continue

    write_history_to_file(combined_history, f"{run_folder_name}/run_history")
    return combined_history

def make_dir(name):
    os.makedirs(name, exist_ok=True)
    os.chmod(name, 0o755)

def read_history_from_file(name):
    return pd.read_csv(name + ".csv", parse_dates=['timestamp'], index_col='timestamp')

def write_history_to_file(history, name="test"):
    history.to_csv(name + ".csv")


def get_cumulative_and_annual_returns(history, col):

    num_years = (history.index[-1].year - history.index[0].year + ((history.index[-1].month + 1) / 12) - (history.index[0].month / 12))

    cumulative_return = history.iloc[-1][col] / history.iloc[0][col] - 1
    annual_return = math.pow(1 + cumulative_return, (1/num_years)) - 1

    return cumulative_return, annual_return


def get_sharpe_and_volatility(history, col):

    # Plus one to last month because it ends at last day
    # num_years = (history.index[-1] - history.index[0]).days / 365.25 # by the day
    num_years = (history.index[-1].year - history.index[0].year + ((history.index[-1].month + 1) / 12) - (history.index[0].month / 12))
    trading_minutes_per_year = history.shape[0] / num_years

    history = history
    history['returns'] = history[col].pct_change(1)
    mean_return = history['returns'].mean()
    std_return = history['returns'].std()

    history.dropna(inplace=True)

    sharpe = (mean_return / std_return) * np.sqrt(trading_minutes_per_year)
    volatility = std_return * np.sqrt(trading_minutes_per_year)

    return sharpe, volatility


def get_max_drawdown(history, col):

    rolling_max = history[col].cummax()
    drawdown = (history[col] - rolling_max) / rolling_max
    return drawdown.min()


def print_stats_from_history(history):

    print(f"\nRun Statistics for run [{history.index[0]}, {history.index[-1]}]:\n")
    
    sharpe, volatility = get_sharpe_and_volatility(history, 'close')
    cumulative, annual = get_cumulative_and_annual_returns(history, 'close')
    max_drawdown = get_max_drawdown(history, 'close')

    print(f"Buy and Hold Strategy:\n")
    print(f"- Cumulative return: {cumulative * 100:.2f}%")
    print(f"- Annual return: {annual * 100:.2f}%")
    print(f"- Annual volatility: {volatility * 100:.2f}%")
    print(f"- Sharpe ratio: {sharpe:.2f}")
    print(f"- Max drawdown: {max_drawdown * 100:.2f}%")
    
    sharpe, volatility = get_sharpe_and_volatility(history, 'portfolio_value')
    cumulative, annual = get_cumulative_and_annual_returns(history, 'portfolio_value')
    max_drawdown = get_max_drawdown(history, 'portfolio_value')

    print(f"\n\nTest Strategy:\n")
    print(f"- Cumulative return: {cumulative * 100:.2f}%")
    print(f"- Annual return: {annual * 100:.2f}%")
    print(f"- Annual volatility: {volatility * 100:.2f}%")
    print(f"- Sharpe ratio: {sharpe:.2f}")
    print(f"- Max drawdown: {max_drawdown * 100:.2f}%")

    print("\n")

def plot_history(history):
    to_plot = pd.DataFrame(index=history.index)
    to_plot['close'] = history["close"] / history.iloc[0]["close"]
    to_plot['portfolio'] = history["portfolio_value"] / history.iloc[0]["portfolio_value"]

    figure = plt.figure()
    p = figure.add_subplot()

    p.plot(to_plot['close'], label="Stock Movement")
    p.plot(to_plot['portfolio'], label="Portfolio Value")
    # [p.axvline(x = i, color = 'b') for i in pd.date_range(history.index[0], history.index[-1], freq='QS')]
    p.legend()

    plt.show()


# Returns a history dataframe using TradingEnv
def test_model(model, test_data, parameters, cash = 0):
    if cash==0:
        cash = parameters['starting_cash']

    test_env = Monitor(TradingEnv(test_data, parameters, cash))
    obs, info = test_env.reset()
    history = [test_env.render()]
    for i in range(test_data.shape[0] - 1):

        action = model.predict(obs, deterministic=True)[0]
        obs, reward, terminated, truncated, info = test_env.step(action)
        history.append(test_env.render())

    return pd.DataFrame(history, index=test_data.index)

    # "PPO", seed, train_data, test_data, parameters, contender_name, contenders, logger
def train(model_type, seed, train_data, test_data, parameters, contender_name, contenders, logger):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    train_env = Monitor(TradingEnv(train_data, parameters, parameters['starting_cash']))
    if model_type == "A2C":
        model = A2C("MlpPolicy", train_env, verbose=0, seed=seed, ent_coef=parameters["ent_coef"])
    else:
        model = PPO("MlpPolicy", train_env, verbose=0, seed=seed, ent_coef=parameters["ent_coef"])

    return train_model(model, train_data, test_data, parameters["training_rounds_per_contender"], contender_name, contenders, logger, parameters)

def train_model(model, train_data, test_data, training_rounds_per_contender, contender_name, contenders, logger, parameters):

    model = model
    best_model = model
    best_score = 0

    logger.print_out("Started training a model")
    
    for i in range(training_rounds_per_contender):

        model.learn(total_timesteps=train_data.shape[0] - 1, progress_bar=False, reset_num_timesteps=False)
        score = test_model(model, test_data, parameters).iloc[-1]["portfolio_value"] # For now measuring score of model with just ending portfolio value, TODO: make this sharpe or sortino
        if score > best_score:
            best_model = model
            best_score = score

        # logger.print_out(f"    - Ended training round {i + 1}/{training_rounds_per_contender} with score {score:.2f}")

    if not len(contenders) > 1 or round(float(best_score), 2) > max([c["score"] for c in contenders]):
        best_model.save(contender_name)
    contenders.append({
        "model": contender_name,
        "score": round(float(best_score), 2)
    })

    logger.print_out(f"- Ended training with score {best_score}")