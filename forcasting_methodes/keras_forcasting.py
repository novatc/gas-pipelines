from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras

from sklearn.model_selection import train_test_split

def to_datetime(date_string):
    date_format = "%b %Y"
    return datetime.strptime(date_string, date_format)

def keras_forcast(df: pd.DataFrame):
    name = df.columns[0]
    df.index = df.iloc[:, 0].apply(to_datetime)

    # turn the index into timestampes
    df["date"] = df.index.map(datetime.timestamp)

    # let x be the co2 and the date_float column
    x = df[["co2", "date"]].values
    y = df["ch4"].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # create the model
    model = keras.Sequential()
    model.add(keras.layers.Dense(12, input_dim=2, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(1, activation='linear'))
    # compile the model
    model.compile(loss='mse', optimizer='adam')

    # fit the model on the training data
    model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=0)
    predictions = model.predict(x_test)
    performance = model.evaluate(x_test, y_test, verbose=0)

    print(f"Keras Model performance: {performance}")
