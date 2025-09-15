import json
import os
import pandas as pd
from matplotlib import pyplot as plt
from config import symbol, month_val
from parse import create_json


def plot_json():
    file_path = 'data/output' + symbol + month_val + '.json'
    if os.path.exists(file_path):
        with open(file_path) as json_file:
            data = json.load(json_file)
    else:
        create_json()
    with open('data/output' + symbol + month_val + '.json') as json_file:
        json_data = json.load(json_file)
    time_series = json_data['Time Series (60min)']
    dates = list(time_series.keys())
    closing_prices = [float(time_series[date]['4. close']) for date in dates]
    # opening_prices = [float(time_series[date]['1. open']) for date in dates]

    dates = [pd.to_datetime(date) for date in dates]

    plt.figure(figsize=(12, 6))
    plt.plot(dates, closing_prices, color='blue', label='Closing Price', linestyle='-', linewidth=2.5)
    # plt.plot(dates, opening_prices, linestyle='--', color='green', label='Opening Price')
    plt.title(symbol + ' Stock Price - ' + month_val)
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.xticks(rotation=45)
    plt.legend(
        ['Closing Price'],
        loc='upper right'
    )
    # plt.legend(['Opening Price'], loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_json()
