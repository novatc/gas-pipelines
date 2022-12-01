import numpy as np
import pandas as pd
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
    slope  = model.coef_[0][0]
    intercept = model.intercept_[0]
    prediction = model.predict(X_test)

def arma_co2(df: pd.DataFrame):
    df.drop(columns='ch4', inplace=True)
    train = df[:-12]
    test = df.tail(12)

    y = train['co2']
    ARMAmodel = SARIMAX(y, order=(1, 0, 1))
    ARMAmodel = ARMAmodel.fit()

    y_pred = ARMAmodel.get_forecast(test.shape[0])
    y_pred_df = y_pred.conf_int(alpha=0.05)
    y_pred_df["Predictions"] = ARMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
    y_pred_df.index = test.index
    y_pred_out = y_pred_df["Predictions"]

    arma_rmse = np.sqrt(mean_squared_error(test["co2"].values, y_pred_df["Predictions"]))
    print("ARMA RMSE: ", arma_rmse)

    ARIMAmodel = ARIMA(y, order=(5, 4, 3))
    ARIMAmodel = ARIMAmodel.fit()

    y_pred = ARIMAmodel.get_forecast(test.shape[0])
    y_pred_df_ARIMA = y_pred.conf_int(alpha=0.05)
    y_pred_df_ARIMA["Predictions"] = ARIMAmodel.predict(start=y_pred_df_ARIMA.index[0], end=y_pred_df.index[-1])
    y_pred_df_ARIMA.index = test.index
    y_pred_out_ARIMA = y_pred_df_ARIMA["Predictions"]

    ARIMA_rmse = np.sqrt(mean_squared_error(test["co2"].values, y_pred_df_ARIMA["Predictions"]))
    print("ARIMA RMSE: ", ARIMA_rmse)

    SARIMAXmodel = SARIMAX(y, order=(5, 4, 2), seasonal_order=(2, 2, 2, 12))
    SARIMAXmodel = SARIMAXmodel.fit()

    y_pred = SARIMAXmodel.get_forecast(test.shape[0])
    y_pred_df_SARIMAX = y_pred.conf_int(alpha=0.05)
    y_pred_df_SARIMAX["Predictions"] = SARIMAXmodel.predict(start=y_pred_df_SARIMAX.index[0], end=y_pred_df.index[-1])
    y_pred_df_SARIMAX.index = test.index
    y_pred_out_SARIMAX = y_pred_df_SARIMAX["Predictions"]

    SARIMAX_rmse = np.sqrt(mean_squared_error(test["co2"].values, y_pred_df_SARIMAX["Predictions"]))
    print("RMSE: ", SARIMAX_rmse)


    plot_arma(train, test, y_pred_out, "clean/images/forcasting/arma_co2.jpg", "ARMA")
    plot_arma(train, test, y_pred_out_ARIMA, "clean/images/forcasting/arima_co2.jpg", "ARIMA")
    plot_arma(train, test, y_pred_out_SARIMAX, "clean/images/forcasting/sarimax_co2.jpg", "SARIMAX")