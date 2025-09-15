import os
import json
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from config import inputval
from parse import create_json, get_df
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

inval = inputval()

def _adf_pvalue(x):
    res = adfuller(x, autolag='AIC')
    return res[1]

def _kpss_pvalue(x, regression='c'):
    # regression='c' for level-stationary; 'ct' for trend-stationary series
    stat, pvalue, lags, crit = kpss(x, regression=regression, nlags='auto')
    return pvalue

def choose_d(series, max_d=2, regression='c', adf_alpha=0.05, kpss_alpha=0.05):
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    last_details = {}
    for d in range(0, max_d + 1):
        x = s if d == 0 else np.diff(s, n=d)
        adf_p = _adf_pvalue(x)
        try:
            kpss_p = _kpss_pvalue(x, regression=regression)
        except Exception:
            kpss_p = np.nan
        last_details = {'d': d, 'adf_p': adf_p, 'kpss_p': kpss_p}
        adf_ok = adf_p < adf_alpha
        kpss_ok = (np.isnan(kpss_p) or kpss_p >= kpss_alpha)
        if adf_ok and kpss_ok:
            return d, last_details
    return max_d, last_details


def predict(df, forecast_steps=None):
    closing_prices = df['4. close']
    if forecast_steps is None:
        forecast_steps = len(closing_prices)

    # Decide differencing order d based on stationarity tests
    d, details = choose_d(closing_prices.values, max_d=2, regression='ct')
    print(f"Chosen d={d} (ADF p={details['adf_p']:.4f}, KPSS p={details['kpss_p'] if not np.isnan(details['kpss_p']) else 'NA'})")

    model = ARIMA(closing_prices, order=(30, d, 0))
    model_fit = model.fit()

    # Use levels to get predictions on original scale; align with differencing
    start = d  # first in-sample prediction available after d differences
    end = len(closing_prices) - 1
    predictions = model_fit.predict(start=start, end=end, typ='levels')

    predictions.index = closing_prices.index[start:]

    return closing_prices[start:], predictions


def plot_prediction():
    file_path = 'data/output' + inval[1] + inval[0] + '.json'

    if not os.path.exists(file_path):
        create_json(inval[0], inval[1])

    with open(file_path) as json_file:
        json_data = json.load(json_file)

    df = get_df(json_data)
    actual, predicted = predict(df)
    rmse = np.sqrt(mean_squared_error(actual.values, predicted.values))
    print(f'RMSE: {rmse:.4f}')
    plt.figure(figsize=(14, 7))
    plt.plot(actual.index, actual.values, label='Actual Closing Price', linewidth=2, color='blue')
    plt.plot(predicted.index, predicted.values, label='ARIMA Prediction', linestyle='--', color='green')

    plt.title(f'{inval[1]} Stock Price vs ARIMA Prediction - {inval[0]} (RMSE: {rmse:.4f})')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_prediction()
