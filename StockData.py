import yfinance as yf
from datetime import datetime
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import random
import requests
import datetime as dt
from dotenv import load_dotenv
import os

# Technical Indicators by Chatgpt --------------------

def calculate_williams_r(data, window=14):
    highest_high = data['High'].rolling(window=window).max()
    lowest_low = data['Low'].rolling(window=window).min()
    williams_r = -100 * (highest_high - data['Close']) / (highest_high - lowest_low)
    return williams_r
def calculate_cmf(data, window=20):
    mf_multiplier = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
    mf_volume = mf_multiplier * data['Volume']
    cmf = mf_volume.rolling(window=window).sum() / data['Volume'].rolling(window=window).sum()
    return cmf
def calculate_cci(data, window=20):
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    sma_tp = tp.rolling(window=window).mean()
    mean_dev = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    return cci
def calculate_obv(data):
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i - 1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i - 1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    data['OBV'] = obv
    return data['OBV']
def calculate_stochastic_oscillator(data, k_window=14, d_window=3):
    lowest_low = data['Low'].rolling(window=k_window).min()
    highest_high = data['High'].rolling(window=k_window).max()
    percent_k = 100 * (data['Close'] - lowest_low) / (highest_high - lowest_low)
    percent_d = percent_k.rolling(window=d_window).mean()
    return percent_k, percent_d
def calculate_atr(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = high_low.combine(high_close, max).combine(low_close, max)
    atr = tr.rolling(window=window).mean()
    return atr
def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return sma, upper_band, lower_band
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
def calculate_macd(data, slow=26, fast=12, signal=9):
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_adx(data, window=14):
    # Calculate True Range (TR)
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = high_low.combine(high_close, max).combine(low_close, max)

    # Calculate Directional Movement (+DM, -DM)
    plus_dm = data['High'].diff()
    minus_dm = data['Low'].diff()

    # Assign positive and negative directional movements
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    # Smooth the True Range, +DM, and -DM
    atr = tr.rolling(window=window).mean()
    smoothed_plus_dm = plus_dm.rolling(window=window).mean()
    smoothed_minus_dm = abs(minus_dm.rolling(window=window).mean())

    # Calculate +DI and -DI
    plus_di = 100 * (smoothed_plus_dm / atr)
    minus_di = 100 * (smoothed_minus_dm / atr)

    # Calculate the Directional Index (DX)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))

    # Calculate the ADX by smoothing the DX
    adx = dx.rolling(window=window).mean()

    return adx


# ---------------------------------------------------- 


def process_data(data, daily = True):

    processed_data = pd.DataFrame(index=data.index)

    processed_data["Close"] = data["close"]
    processed_data["Change"] = data["close"].diff()
    processed_data["D_HL"] = data["high"] - data["low"]
    
    # Technical indicators by Chatgpt ----- 
    # processed_data["High"] = data["high"]
    # processed_data["Low"] = data["low"]
    # processed_data["Open"] = data["open"]
    # processed_data["Volume"] = data["volume"]

    # processed_data['SMA_20'] = processed_data['Close'].rolling(window=20).mean()
    # processed_data['EMA_20'] = processed_data['Close'].ewm(span=20, adjust=False).mean()
    # processed_data['RSI'] = calculate_rsi(processed_data)
    # processed_data['ATR'] = calculate_atr(processed_data)
    # processed_data['MACD'], processed_data['MACD_Signal'], _ = calculate_macd(processed_data)
    # processed_data['Bollinger_Mid'], processed_data['Bollinger_Upper'], processed_data['Bollinger_Lower'] = calculate_bollinger_bands(processed_data)
    # processed_data['CCI'] = calculate_cci(processed_data)
    # processed_data['Williams_%R'] = calculate_williams_r(processed_data)
    # processed_data['CMF'] = calculate_cmf(processed_data)
    # processed_data['OBV'] = calculate_obv(processed_data)
    # processed_data['ADX'] = calculate_adx(processed_data)
    #  --------------------------

    for feature in processed_data.columns:
        # Calculate rolling mean and std
        rolling_mean = processed_data[feature].rolling(window=20).mean()
        rolling_std = processed_data[feature].rolling(window=20).std()

        # Normalize the feature
        processed_data[f'{feature}_Normalized'] = (processed_data[feature] - rolling_mean) / rolling_std

        # Min-Max Scaling to range -1 to 1 using rolling window 
        # rolling_min = processed_data[f'{feature}_Normalized'].rolling(window=20).min()
        # rolling_max = processed_data[f'{feature}_Normalized'].rolling(window=20).max()
        # processed_data[f'{feature}_Scaled'] = -1 + 2 * (processed_data[f'{feature}_Normalized'] - rolling_min) / (rolling_max - rolling_min)

    processed_data.dropna(inplace=True)

    return processed_data

def get_min_max_values():
    historical_data = get_consecutive_months(dt.datetime(year=2000, month=1, day=1), 120) # get 2000-2010 data 
    for c in historical_data.columns:
        print(c)
        print("Max: ", historical_data[c].max()) 
        print("Min: ", historical_data[c].min())
    

def get_month_csv(year, month):

    year = int("20" + str(year))

    data = data = pd.read_csv(f"btc_data.csv", index_col="timestamp")
    data.index = pd.to_datetime(data.index)
    year_data = data[data.index.year == year]
    return year_data[year_data.index.month == month]

    folder_name = "spy_data"

    data = pd.read_csv(f"{folder_name}/20{year:02d}-{month:02d}.csv", index_col="timestamp").iloc[::-1]
    data.index = pd.to_datetime(data.index)

    return data

def get_month(year, month):
    return process_data(get_month_csv(year, month))

def get_month_hourly(year, month):
    return process_data(
        get_month_csv(year, month).resample('h').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        ).dropna()
    )

def get_month_daily(year, month):
    frames = []
    frames.append(
        get_month_csv(year, month).resample('D').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum')
        ).dropna().iloc[::-1]
    )
    i_year = year
    i_month = month
    for i in range(5):
        i_month -= 1
        if i_month < 1:
            i_month = 12
            i_year -= 1
        frames.append(
            get_month_csv(i_year, i_month).resample('D').agg(
            open=('open', 'first'),
            high=('high', 'max'),
            low=('low', 'min'),
            close=('close', 'last'),
            volume=('volume', 'sum')
            ).dropna().iloc[::-1]
        )
    output = process_data(pd.concat(frames).iloc[::-1], True)
    return output[output.index.month == month]

def get_random_month_not_2008():
    year = random.randint(0, 23)
    while year == 8:
        year = random.randint(0, 23)
    return get_month(year, random.randint(1, 12))

def get_random_month_2020s():
    return get_month(random.randint(20, 23), random.randint(1, 12))

def get_random_month_2020s_hourly():
    return get_month_hourly(random.randint(20, 23), random.randint(1, 12))

def get_random_month():
    return get_month(random.randint(0, 23), random.randint(1, 12))

def get_test_data():
    frames = []
    for i in range(1, 9):
        frames.append(get_month(24, i))
    return pd.concat(frames)

def get_random_train_data(num_months):
    frames = []
    for i in range(num_months):
        frames.append(get_random_month())
    return pd.concat(frames)


def get_consecutive_months(starting_month, num_months, t="minutely"):
    frames = []
    for i in range(num_months):
        if t == "daily":
            frames.append(get_month_daily((starting_month + pd.DateOffset(months=i)).year % 100, (starting_month + pd.DateOffset(months=i)).month))   
        else: 
            frames.append(get_month((starting_month + pd.DateOffset(months=i)).year % 100, (starting_month + pd.DateOffset(months=i)).month))
    return pd.concat(frames)

def get_year(year):
    frames = []
    for i in range(1, 13):
        frames.append(get_month(year, i))
    return pd.concat(frames)

def get_year_hourly(year):
    frames = []
    for i in range(1, 13):
        frames.append(get_month_hourly(year, i))
    return pd.concat(frames)

def get_pre_2020():
    frames = []
    for i in range(0, 20):
        frames.append(get_year(i))
    return pd.concat(frames)

def get_pre_2020_hourly():
    frames = []
    for i in range(0, 20):
        frames.append(get_year_hourly(i))
    return pd.concat(frames)

def get_day(year, month, day):
    month = get_month(year, month)
    return month[month.index.day == day]

def get_current_data():

    # Note: provides a week of data
    prices = yf.Ticker("SPY").history(period='max', interval='1m', prepost=True)
    prices["close"] = prices["Close"]
    prices["low"] = prices["Low"]
    prices["high"] = prices["High"]

    prices = prices[prices.index.day == int(dt.datetime.today().strftime("%d"))] # Get current date data

    return process_data(prices)
    
def get_current_alpaca():

    load_dotenv()
    api_key = os.getenv("API_KEY")
    api_secret_key = os.getenv("API_SECRET_KEY")

    url = "https://data.alpaca.markets/v2/stocks/bars?symbols=spy&timeframe=1Min&limit=3000&adjustment=raw&feed=sip&sort=asc"
    headers = {"accept": "application/json", "APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret_key}

    response = requests.get(url, headers=headers) 

    data = pd.DataFrame.from_dict(response.json()['bars']['SPY'])

    data['time'] = pd.to_datetime(data['t'], format='%Y-%m-%dT%H:%M:%SZ')
    data['time'] = data['time'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    data = data.set_index('time')

    data['close'] = data['c']
    data['high'] = data['h']
    data['low'] = data['l']

    return process_data(data)