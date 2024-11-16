import requests
import datetime as dt
import pandas as pd
import time

def parse_and_save(data, header = False):
    minute_data = pd.DataFrame(data)
    minute_data["time"] = minute_data[0] / 1000.0
    minute_data.index = [dt.datetime.utcfromtimestamp(minute_data.iloc[i]["time"]) for i in range(len(minute_data))]
    parsed_data = pd.DataFrame(index=minute_data.index)
    parsed_data.index.name = "timestamp"
    parsed_data["open"] = minute_data[1]
    parsed_data["close"] = minute_data[4]
    parsed_data["high"] = minute_data[2]
    parsed_data["low"] = minute_data[3]
    parsed_data["volume"] = minute_data[5]
    parsed_data = parsed_data.head(parsed_data.shape[0] -1)
    parsed_data.to_csv("btc_data.csv", mode='a', header=header)
    print(parsed_data)

url = 'https://api.binance.com/api/v3/klines'

day = 0

parse_and_save(requests.get(url, params={
    'symbol': 'BTCUSDT',
    'startTime': int(dt.datetime.timestamp(dt.datetime(year=2020, month=1, day=1) + dt.timedelta(hours=(24 * day))) * 1000),
    'endTime': int(dt.datetime.timestamp(dt.datetime(year=2020, month=1, day=1) + dt.timedelta(hours=(24 * day) + 12)) * 1000),
    'interval': '1m',
    'limit': '1000' 
}).json(), True)

parse_and_save(requests.get(url, params={
    'symbol': 'BTCUSDT',
    'startTime': int(dt.datetime.timestamp(dt.datetime(year=2020, month=1, day=1) + dt.timedelta(hours=(24 * day) + 12)) * 1000),
    'endTime': int(dt.datetime.timestamp(dt.datetime(year=2020, month=1, day=1) + dt.timedelta(hours=(24 * day) + 24)) * 1000),
    'interval': '1m',
    'limit': '1000' 
}).json())

for day in range(1, (dt.datetime(year=2024, month=9, day=30) - dt.datetime(year=2020, month=1, day=1)).days):

    parse_and_save(requests.get(url, params={
        'symbol': 'BTCUSDT',
        'startTime': int(dt.datetime.timestamp(dt.datetime(year=2020, month=1, day=1) + dt.timedelta(hours=(24 * day))) * 1000),
        'endTime': int(dt.datetime.timestamp(dt.datetime(year=2020, month=1, day=1) + dt.timedelta(hours=(24 * day) + 12)) * 1000),
        'interval': '1m',
        'limit': '1000' 
    }).json())
    
    parse_and_save(requests.get(url, params={
        'symbol': 'BTCUSDT',
        'startTime': int(dt.datetime.timestamp(dt.datetime(year=2020, month=1, day=1) + dt.timedelta(hours=(24 * day) + 12)) * 1000),
        'endTime': int(dt.datetime.timestamp(dt.datetime(year=2020, month=1, day=1) + dt.timedelta(hours=(24 * day) + 24)) * 1000),
        'interval': '1m',
        'limit': '1000' 
    }).json())

    time.sleep(1)