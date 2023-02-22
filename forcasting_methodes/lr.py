from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mlflow import tracking


def to_datetime(date_string):
    date_format = "%b %Y"
    return datetime.strptime(date_string, date_format)

def linear_regression(df: pd.DataFrame):

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
    model = LinearRegression()
    model.fit(x_train, y_train)
    slope = model.coef_[0]
    intercept = model.intercept_
    y_pred = model.predict(x_test)
    print(f"y = {slope}x + {intercept}")
    score = model.score(x_test, y_test)
    print(f"LR Model performance: {score}")


    plot_lr(y_pred, y_test, x_test[:, 1], name)
    # put the predictions in a dataframe
    df_pred = pd.DataFrame(y_pred, columns=["predictions"])
    df_pred["actual"] = y_test
    df_pred["date"] = x_test[:, 1]
    df_pred["date"] = df_pred["date"].apply(datetime.fromtimestamp)
    df_pred.set_index("date", inplace=True)
    df_pred.to_csv("clean/forcasting/lr_prediction.csv")
    return df_pred

def linear_regression_co2_dash(data: pd.DataFrame, predict_steps=100):
    # Convert the date column to a pandas datetime object
    data['time'] = pd.to_datetime(data.index, format='%b %Y')

    # Sort the data by date
    data = data.sort_values('time')

    # Create a linear regression model using scikit-learn
    X = data['time'].values.reshape(-1, 1)
    y = data['co2'].values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)

    # Predict the next 10 steps
    next_steps = predict_steps
    last_date = data['time'].iloc[-1]
    next_dates = pd.date_range(start=last_date, periods=next_steps + 1, freq='MS')[1:]
    next_X = pd.DataFrame(next_dates, columns=['time']).reset_index(drop=True)
    next_X['index'] = next_X.index
    next_y = model.predict(next_X['index'].values.reshape(-1, 1))

    # Add the predicted values to the original DataFrame
    next_data = pd.DataFrame({'time': next_dates, 'co2': next_y.ravel()})
    data = pd.concat([data, next_data])

    return data

def linear_regression_ch4_dash(data: pd.DataFrame, predict_steps=50):
    # Convert the date column to a pandas datetime object
    data['time'] = pd.to_datetime(data.index, format='%b %Y')

    # Sort the data by date
    data = data.sort_values('time')

    # Create a linear regression model using scikit-learn
    X = data['time'].values.reshape(-1, 1)
    y = data['ch4'].values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)

    # Predict the next 10 steps
    next_steps = predict_steps
    last_date = data['time'].iloc[-1]
    next_dates = pd.date_range(start=last_date, periods=next_steps + 1, freq='MS')[1:]
    next_X = pd.DataFrame(next_dates, columns=['time']).reset_index(drop=True)
    next_X['index'] = next_X.index
    next_y = model.predict(next_X['index'].values.reshape(-1, 1))

    # Add the predicted values to the original DataFrame
    next_data = pd.DataFrame({'time': next_dates, 'ch4': next_y.ravel()})
    data = pd.concat([data, next_data])

    return data

def plot_lr(predictions: list, acctual: list, time, name):

    date = [datetime.fromtimestamp(x) for x in time]

    fig, ax = plt.subplots()
    ax.scatter(date, predictions, label="predictions", color="red")
    ax.scatter(date, acctual, label="actual", color="blue")

    plt.xlabel("Time")
    plt.ylabel("CH4 emissions")
    plt.title("Predicted output over time")

    plt.legend()
    plt.savefig('clean/images/lr/' + name + '_lr.png')
    plt.clf()

