import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from plot_service.plotting import plot_arma


def linear_regression(df: pd.DataFrame):
    model = LinearRegression()
    X = df.drop('ch4', axis=1)
    Y = df.drop('co2', axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    model.fit(X_train, Y_train)
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    prediction = model.predict(X_test)


def SARIMA_co2(forcasting: pd.DataFrame):
    df = forcasting.copy()
    warnings.filterwarnings("ignore")
    df.drop(columns='ch4', inplace=True)
    train = df[:-12]
    test = df.tail(12)

    y = train['co2']
    SARIMAmodel = sm.tsa.SARIMAX(y, order=(0,1,0), seasonal_order=(1, 0, 0, 12), enforce_stationarity=False)
    SARIMAmodel = SARIMAmodel.fit()

    y_pred_SARIMA = SARIMAmodel.get_forecast(test.shape[0])
    y_pred_df_SARIMA = y_pred_SARIMA.conf_int(alpha=0.05)
    y_pred_df_SARIMA["Predictions"] = SARIMAmodel.predict(start=y_pred_df_SARIMA.index[0],
                                                          end=y_pred_df_SARIMA.index[-1])
    y_pred_df_SARIMA.index = test.index
    y_pred_out_SARIMA = y_pred_df_SARIMA["Predictions"]

    SARIMA_rmse = np.sqrt(mean_squared_error(test["co2"].values, y_pred_df_SARIMA["Predictions"]))

    # put test and prediction in one dataframe for plotting
    test["Predictions"] = y_pred_df_SARIMA["Predictions"]
    print(test.columns)



    # plot_arma(df.index.name, train, test, y_pred_out_SARIMA, f"clean/images/forcasting/{df.index.name}_sarima_co2.jpg",
    #           "SARIMA")
    return test


def ARIMA_co2(forcasting: pd.DataFrame):
    df = forcasting.copy()
    warnings.filterwarnings("ignore")
    df.drop(columns='ch4', inplace=True)
    train = df[:-12]
    test = df.tail(12)

    y = train['co2']
    ARIMAmodel = ARIMA(y, order=(5, 4, 3))
    ARIMAmodel = ARIMAmodel.fit()

    y_pred_ARIMA = ARIMAmodel.get_forecast(test.shape[0])
    y_pred_df_ARIMA = y_pred_ARIMA.conf_int(alpha=0.05)
    y_pred_df_ARIMA["Predictions"] = ARIMAmodel.predict(start=y_pred_df_ARIMA.index[0], end=y_pred_df_ARIMA.index[-1])
    y_pred_df_ARIMA.index = test.index
    y_pred_out_ARIMA = y_pred_df_ARIMA["Predictions"]

    ARIMA_rmse = np.sqrt(mean_squared_error(test["co2"].values, y_pred_df_ARIMA["Predictions"]))
    print("ARIMA RMSE: ", ARIMA_rmse)

    plot_arma(df.index.name, train, test, y_pred_out_ARIMA, f"clean/images/forcasting/{df.index.name}_arima_co2.jpg",
              "ARIMA")


def ARMA_co2(forcasting: pd.DataFrame):
    df = forcasting.copy()
    warnings.filterwarnings("ignore")
    df.drop(columns='ch4', inplace=True)
    train = df[:-12]
    test = df.tail(12)

    y = train['co2']
    ARMAmodel = SARIMAX(y, order = (1, 0, 1))
    ARMAmodel = ARMAmodel.fit()

    y_pred = ARMAmodel.get_forecast(test.shape[0])
    y_pred_df = y_pred.conf_int(alpha=0.05)
    y_pred_df["Predictions"] = ARMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
    y_pred_df.index = test.index
    y_pred_out_ARMA = y_pred_df["Predictions"]

    arma_rmse = np.sqrt(mean_squared_error(test["co2"].values, y_pred_df["Predictions"]))
    print("ARMA RMSE: ", arma_rmse)

    plot_arma(df.index.name, train, test, y_pred_out_ARMA, f"clean/images/forcasting/{df.index.name}_arma_co2.jpg",
              "ARMA")
