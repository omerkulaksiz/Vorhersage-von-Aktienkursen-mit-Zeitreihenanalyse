import json
import pandas as pd
import requests
import os


def parse_json(m, s):
    api_key = os.getenv("API_KEY")
    function = "TIME_SERIES_INTRADAY"
    url = f'https://www.alphavantage.co/query?function={function}&symbol={s}&interval=60min&outputsize=full&month={m}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    return data


def get_df(data):
    time_series_data = data['Time Series (60min)']
    df = pd.DataFrame.from_dict(time_series_data, orient='index')
    df = df[['4. close']]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.astype(float)
    return df


print(get_df(parse_json('2021-03', 'AAPL')))


def create_json(m, s):
    a_raw = parse_json(m, s)
    file_name = "output" + s + m + ".json"
    file_path = os.path.join('data', file_name)
    with open(file_path, "w") as json_file:
        json.dump(a_raw, json_file, indent=4)
