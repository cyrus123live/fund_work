from stable_baselines3 import A2C
from stable_baselines3 import PPO
from TradingEnv import TradingEnv
import StockData
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
import time
import datetime
import requests
import os
import sqlite3 as sql
from dotenv import load_dotenv
from pathlib import Path
import pytz
import pandas as pd

CASH_DIVISOR = 100
CASH_SUBTRACTOR = 91970 # Try to work with just 10
STARTING_CASH = 92070 - CASH_SUBTRACTOR
EXAMPLE_CLOSE = 580
MODEL_NAME = "PPO_109"


load_dotenv()
api_key = os.getenv("API_KEY")
api_secret_key = os.getenv("API_SECRET_KEY")

def get_cash():
    headers = {"accept": "application/json", "APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret_key}
    response = requests.get("https://paper-api.alpaca.markets/v2/account", headers=headers)

    return float(response.json()["cash"]) - CASH_SUBTRACTOR

def get_position():
    headers = {"accept": "application/json", "APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret_key}
    response = requests.get("https://paper-api.alpaca.markets/v2/positions", headers=headers)
    return response.json()

def cancel_all():
    headers = {"accept": "application/json", "APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret_key}
    response = requests.delete("https://paper-api.alpaca.markets/v2/orders", headers=headers)
    return response.json()

def get_position_quantity():
    if len(get_position()) > 0:
        return float(get_position()[0]['qty'])
    else:
        return 0

def get_position_value():
    if len(get_position()) > 0:
        return float(get_position()[0]['market_value'])
    else:
        return 0

def make_order(qty, buy_or_sell, price):

    headers = {"accept": "application/json", "APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret_key}
    params = {
        'symbol': 'SPY',
        'qty': str(qty),
        'side': buy_or_sell, # "buy" or "sell"
        'type': 'limit',
        'limit_price': str(price),
        'extended_hours': True,
        'time_in_force': 'day' # Note: experiment with this
    }

    response = requests.post("https://paper-api.alpaca.markets/v2/orders", headers=headers, json=params)
    return response.json()

def sell_all(price):
    return make_order(get_position_quantity(), "sell", price)

def buy_all(price, cash):
    # to_buy = cash / data.iloc[-1]["Close"]
    to_buy = cash / price
    return make_order(to_buy, 'buy', price)

def buy(qty, price):
    return make_order(qty, "buy", price)

def write_df_to_csv(file_name, data_dict):
    df = pd.DataFrame(data_dict)
    if not os.path.isfile(file_name):
        df.to_csv(file_name, mode="a", index=False)
    else:
        df.to_csv(file_name, mode="a", index=False, header=False)

def add_to_stockdata_csv(folder_name, data_dict):
    write_df_to_csv(f"/root/RLTrader/csv/{folder_name}/stockdata.csv", data_dict)

def add_to_minutely_csv(folder_name, data_dict):
    write_df_to_csv(f"/root/RLTrader/csv/{folder_name}/minutely.csv", data_dict)

def add_to_daily_csv(data_dict):
    write_df_to_csv(f"/root/RLTrader/csv/daily.csv", data_dict)
    

def main():

    folder_name = datetime.datetime.now().strftime("%Y-%m-%d")

    os.makedirs(f"/root/RLTrader/csv/{folder_name}", exist_ok=True)

    # Parameters
    model = PPO.load("/root/RLTrader/models/" + MODEL_NAME)
    k = STARTING_CASH / EXAMPLE_CLOSE
    held = get_position_quantity()
    cash = get_cash()
    start_time = datetime.datetime.now()

    missed_trades = 0
    total_trades = 0
    starting_cash = cash
    starting_held = held

    print(f"Starting trader session, cash: {cash:.2f}, held: {held:.2f}\n")
    
    while True:

        # try:

        time.sleep(1)
        current_time = datetime.datetime.now()

        # Weekend
        if current_time.weekday() == 5 or current_time.weekday() == 6:
            print("It is the weekend, ending trader session.")
            quit()

        # Too late (next day UTC = 8pm New York)
        if current_time.day != start_time.day:
            print("Trading day over, ending trader session.")
            data = StockData.get_current_data()
            add_to_daily_csv([{
                "Start Time": start_time,
                "First Close": data["Close"].iloc[0],
                "First Held": starting_held,
                "First Cash": starting_cash,
                "End Time": datetime.datetime.now().timestamp(),
                "Last Close": data["Close"].iloc[-1],
                "Last Held": held,
                "Last Cash": cash,
                "Total Trades": total_trades,
                "Missed Trades": missed_trades
            }])
            print("Trading day ended successfully.")
            quit()

        # Too early (8am UTC = 4 am New York)
        if current_time.hour < 8:
            continue

        if current_time.second == 50: 
            try:
                data = StockData.get_current_data()
            except Exception as e:
                print("Error in getting current data:", e)
                continue

            if data.shape[0] == 0:
                print("No data...")
                continue

            obs = np.array(data[["Close_Normalized", "Change_Normalized", "D_HL_Normalized"]].iloc[-1].tolist() + [held / k, cash / STARTING_CASH])
            row = data.iloc[-1]
            pre_trade_cash = cash
            pre_trade_held = held

            # print(obs)

            action = model.predict(obs, deterministic=True)[0][0]
            bought = False
            missed_buy = False
            sold = False
            missed_sell = False

            if action < 0 and held > 0:
                total_trades += 1
                sold = True
                print(f"{current_time.strftime('%Y-%m-%d %H:%M')} Executing sell at price {round(data['Close'].iloc[-1], 2)}")
                sell_all(round(data['Close'].iloc[-1], 2))
            elif action > 0 and cash > 1:
                total_trades += 1
                bought = True
                print(f"{current_time.strftime('%Y-%m-%d %H:%M')} Executing buy all ({cash / round(data['Close'].iloc[-1], 2):.2f}) at price {round(data['Close'].iloc[-1], 2)}, with cash: {cash:.2f}")
                buy_all(round(data['Close'].iloc[-1], 2), cash)
            else:
                print(f"{current_time.strftime('%Y-%m-%d %H:%M')} Holding at price {round(data['Close'].iloc[-1], 2)}")

            time.sleep(55)
            cancel_output = cancel_all()
            if len(cancel_output) > 0: # cancel orders if not made in 25 seconds, so that we can get up to date info and safely move to next minute
                missed_trades += 1 
                if bought: missed_buy = True
                if sold: missed_sell = True
                print("** Missed Trade **")
            time.sleep(2)

            # Update csv
            cash = get_cash()
            held = get_position_quantity()

            add_to_minutely_csv(folder_name, [{
                "Time": datetime.datetime.now().timestamp(),
                "Close": row["Close"],
                "Action": float(action), 
                "Cash": pre_trade_cash,
                "Held": pre_trade_held,
                "Resulting Cash": cash, 
                "Resulting Held": held,
                "Bought": bought,
                "Sold": sold,
                "Missed Buy": missed_buy,
                "Missed Sell": missed_sell,
                "Obs Held": obs[len(obs) - 2],
                "Obs Cash": obs[len(obs) - 1]
            }])
            add_to_stockdata_csv(folder_name, [data.iloc[-1].to_dict()])
            print(f"{current_time.strftime('%Y-%m-%d %H:%M')} Ended Trade. Cash: {cash:.2f}, Held: {held:.2f}\n\n")

        # except Exception as e:
        #     print("Failure in loop:", e)

if __name__ == "__main__":
    main()