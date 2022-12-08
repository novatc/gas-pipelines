from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def to_datetime(date_string):
    date_format = "%b %Y"
    return datetime.strptime(date_string, date_format)

def linear_regression(df: pd.DataFrame):
    df.reset_index(inplace=True)
    name = df.columns[0]

    df.index = df.iloc[:, 0].apply(to_datetime)

    # turn the index into timestampes
    df["date"] = df.index.map(datetime.timestamp)

    # let x be the co2 and the date_float column
    x = df[["co2", "date"]].values
    y = df["ch4"].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = LinearRegression()
    model.fit(x_train, y_train)
    slope = model.coef_[0]
    intercept = model.intercept_
    y_pred = model.predict(x_test)
    print(f"y = {slope}x + {intercept}")
    score = model.score(x_test, y_test)
    print(f"LR Model performance: {score}")

    plot_lr(y_pred, y_test, x_test[:, 1], name)

def plot_lr(predictions: list, acctual: list, time, name):

    date = [datetime.fromtimestamp(x) for x in time]

    fig, ax = plt.subplots()
    ax.scatter(date, predictions, label="predictions", color="red")
    ax.scatter(date, acctual, label="actual", color="blue")

    plt.xlabel("CH4 emissions")
    plt.ylabel("Time")
    plt.title("Predicted output over time")

    plt.legend()
    plt.savefig('clean/images/lr/' + name + '_lr.png')
    plt.clf()

def plot_f(slop, intercept):
    x = np.linspace(0, 10, 100)
    y = slop * x + intercept
    plt.plot(x, y)
    plt.show()
