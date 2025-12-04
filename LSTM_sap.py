import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from sap_parse import parse_json, get_df




def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


tf.random.set_seed(7)

# 1) load SAP data and build time series of act_qty
dataframe = get_df(parse_json())
print("Historical time series (act_qty by month):")
print(dataframe)

# 2) to numpy and normalize
dataset = dataframe.values.astype("float32")
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

# Save basic stats to keep forecasts in a realistic range
hist_min = float(dataset.min())
hist_max = float(dataset.max())

# 3) train / test split (to check how good the model is on known data)
train_size = int(len(dataset_scaled) * 0.67)
train, test = dataset_scaled[0:train_size, :], dataset_scaled[train_size:, :]

# 4) supervised samples
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# 5) reshape to [samples, time steps, features]
trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))

# 6) LSTM model
model = Sequential()
model.add(LSTM(8, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=0)

# 7) in-sample predictions (to see if they are at least reasonable)
trainPredict = model.predict(trainX, verbose=0)
testPredict = model.predict(testX, verbose=0)

# 8) invert scaling
trainPredict = scaler.inverse_transform(trainPredict)
trainY_inv = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY_inv = scaler.inverse_transform([testY])

# 9) error on train/test
trainScore = np.sqrt(mean_squared_error(trainY_inv[0], trainPredict[:, 0]))
testScore = np.sqrt(mean_squared_error(testY_inv[0], testPredict[:, 0]))
print(f"\nTrain Score: {trainScore:.2f} RMSE")
print(f"Test Score: {testScore:.2f} RMSE\n")

# Show last few actual vs predicted points to see quality
print("Last few known months")
test_start_idx = len(trainPredict) + (look_back * 2) + 1
test_dates = dataframe.index[test_start_idx:test_start_idx + len(testPredict)]
for date, pred, actual in zip(test_dates[-5:], testPredict[-5:, 0], testY_inv[0][-5:]):
    print(
        f"{date.year}-{date.month:02d}: "
        f"predicted={pred:.2f}, actual={actual:.2f}"
    )

# 10) 12-step-ahead forecast for 2025 (recursive, but constrained)
future_steps = 12

# Start from the last look_back observations in the scaled series
last_input = dataset_scaled[-look_back:, 0]  # shape: (look_back,)
current_input = last_input.reshape((1, 1, look_back))

future_scaled = []
for _ in range(future_steps):
    next_scaled = model.predict(current_input, verbose=0)[0, 0]
    future_scaled.append(next_scaled)
    # update window
    current_input = np.roll(current_input, -1, axis=2)
    current_input[0, 0, -1] = next_scaled

future_scaled_arr = np.array(future_scaled).reshape(-1, 1)
future_values = scaler.inverse_transform(future_scaled_arr).flatten()

# 11) apply simple domain constraints to avoid absurd predictions
growth_factor = 1.20  # allow up to +20% above historical max
upper_bound = hist_max * growth_factor
lower_bound = max(0.0, hist_min)  # act_qty should not be negative

future_values_clipped = np.clip(future_values, lower_bound, upper_bound)

# 12) print 12-month forecast for 2025
print("\nforecast for 2025")
year = 2025
for i, (raw_val, clipped_val) in enumerate(zip(future_values, future_values_clipped), start=1):
    month = i  # 1..12 → Jan..Dec
    print(
        f"{year}-{month:02d}: "
        f"raw_pred={raw_val:.2f}, clipped_pred={clipped_val:.2f}"
    )