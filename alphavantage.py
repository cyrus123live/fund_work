import requests
import random
import pandas as pd
import os

api_key = os.getenv('alpha_api_key')
# url = f'https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol=USD&to_symbol=YEN&interval=1min&apikey={api_key}'
# https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords=vfv&apikey={api_key}

ticker = "SH"
os.makedirs(f"stock_data/{ticker}/", exist_ok=True)
for year in range(0, 24):
    for month in range(1, 13):
        time = f"20{year:02d}-{month:02d}"
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval=1min&adjusted=false&extended_hours=true&apikey={api_key}&datatype=csv&outputsize=full&month={time}'
        data = requests.get(url)
        decoded_content = data.content.decode('utf-8')
        # Open the file for writing
        if decoded_content:
            with open(f'stock_data/{ticker}/{time}.csv', 'w') as f:
                f.write(decoded_content)
                
        
for year in range(24, 25):
    for month in range(1, 9):
        time = f"20{year:02d}-{month:02d}"
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval=1min&adjusted=false&extended_hours=true&apikey={api_key}&datatype=csv&outputsize=full&month={time}'
        data = requests.get(url)
        decoded_content = data.content.decode('utf-8')
        # Open the file for writing
        if decoded_content:
            with open(f'stock_data/{ticker}/{time}.csv', 'w') as f:
                f.write(decoded_content)