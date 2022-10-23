import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from collections import deque
import random
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Bidirectional, CuDNNLSTM, CuDNNGRU
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam, rmsprop_v2
from matplotlib import pyplot as plt
import yfinance as yf

# Global variables and hyper parameters
SEQ_LEN = 60
FUTURE_PREDICT_PERIOD = 3
CRYPTO_TO_PREDICT = "LTC-USD"
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PREDICT_PERIOD}-PRED-{int(time.time())}"
DROPOUT_RATE = 0.2
LSTM_ACTIVATION = 'tanh'

# Classification criteria for crypto movement


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

# Scale/Normalize the data, then balance the data to have even amounts for buys and sells


def preprocess(df_in):
    df_in = df_in.drop('future', axis=1)

    for col in df_in.columns:
        if col != "trigger":
            df_in[col] = df_in[col].pct_change()
            df_in.dropna(inplace=True)
            df_in[col] = preprocessing.scale(df_in[col].values)
    df_in.dropna(inplace=True)
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for row in df_in.values:
        prev_days.append([val for val in row[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), row[-1]])
    random.shuffle(sequential_data)
    buys = []
    sells = []

    for sequence, trigger in sequential_data:
        if trigger == 0:
            sells.append([sequence, trigger])
        elif trigger == 1:
            buys.append([sequence, trigger])
    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]
    sequential_data = buys+sells
    random.shuffle(sequential_data)
    X = []
    y = []

    for sequence, trigger in sequential_data:
        X.append(sequence)
        y.append(trigger)

    return np.array(X), y

# Data Acquisition and Formatting
main_df = pd.DataFrame()
cryptos = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]

for crypto in cryptos:
    raw_data = f"D:/crypto_data2/{crypto}.csv"

    df = pd.read_csv(raw_data, names=["time", "low", "high", "open", "close", "volume"], )
    df.rename(columns={"close": f"{crypto}_close", "volume": f"{crypto}_volume"}, inplace=True)

    df.set_index("time", inplace=True)
    df = df[[f"{crypto}_close", f"{crypto}_volume"]]
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)
main_df['future'] = main_df[f"{CRYPTO_TO_PREDICT}_close"].shift(-FUTURE_PREDICT_PERIOD)

main_df['trigger'] = list(map(classify, main_df[f"{CRYPTO_TO_PREDICT}_close"], main_df["future"]))
# print(main_df[[f"{CRYPTO_TO_PREDICT}_close", "future", "trigger"]].head(10))

times = sorted(main_df.index.values)
final_5pt = times[-int(0.05 * len(times))]

validation_main_df = main_df[(main_df.index >= final_5pt)]
main_df = main_df[(main_df.index < final_5pt)]

train_x, train_y = preprocess(main_df)
val_x, val_y = preprocess(validation_main_df)

print(f"train data: {len(train_x)} validation: {len(val_x)}")
print(f"Buys: {train_y.count(1)} dont buys: {train_y.count(0)}")
print(f"Validation Buys: {val_y.count(1)} dont buys: {val_y.count(0)}")
# Make the model
model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(DROPOUT_RATE))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(DROPOUT_RATE))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=False))
model.add(Dropout(DROPOUT_RATE))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(DROPOUT_RATE))

model.add(Dense(2, activation='softmax'))

opt = Adam(learning_rate=0.001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

train_x = np.array(train_x)
train_y = np.array(train_y)
val_x = np.array(val_x)
val_y = np.array(val_y)

history = model.fit(train_x, train_y, batch_size=64, epochs=10, validation_data=(val_x, val_y), verbose=1)

# Plot the model loss and accuracy
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6),)

ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_ylabel('Loss')

ax2.plot(history.history['accuracy'], label='Training Accuracy')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax2.set_ylabel('Accuracy')

plt.legend()
plt.tight_layout()
plt.show()

# Web Scraper ?
