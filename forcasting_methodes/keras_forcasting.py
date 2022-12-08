from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras

from sklearn.model_selection import train_test_split

def to_datetime(date_string):
    date_format = "%b %Y"
    return datetime.strptime(date_string, date_format)

def keras_forcast(df: pd.DataFrame):
    df_copy = df.copy()
    df_copy.reset_index(inplace=True)
    name = df_copy.columns[0]
    df_copy.index = df_copy.iloc[:, 0].apply(to_datetime)

    # turn the index into timestampes
    df_copy["date"] = df_copy.index.map(datetime.timestamp)

    # let x be the co2 and the date_float column
    x = df_copy[["co2", "date"]].values
    y = df_copy["ch4"].values

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

    plot_forcast(predictions,y_test,x_test[:, 1], name)

def plot_forcast(predictions, accurate_values,time, name):
    date = [datetime.fromtimestamp(x) for x in time]

    fig, ax = plt.subplots()
    ax.scatter(date, predictions, label="predictions", color="red")
    ax.scatter(date, accurate_values, label="actual", color="blue")

    plt.xlabel("Time")
    plt.ylabel("CH4 emissions")
    plt.title("Predicted output over time")
    plt.legend()
    plt.savefig('clean/images/keras/' + name + '.png')
    plt.clf()
