import numpy as np
import pandas as pd
import requests


def get_data_for_forecast_page(model="arch"):
    forecast_response = requests.get(f"http://backend:8000/api/forecast/{model}/")
    historical_data_response = requests.get("http://backend:8000/api/get_btc_usdt_data/")

    if forecast_response.status_code == 200:
        forecast_data = forecast_response.json()
    else:
        raise Exception(f"Ошибка: Получен статус: {forecast_response.status_code}")

    if historical_data_response.status_code == 200:
        historical_data = historical_data_response.json()
    else:
        raise Exception(f"Ошибка: Получен статус: {historical_data_response.status_code}")

    df_now = pd.DataFrame(forecast_data["now"])
    df_forecast = pd.DataFrame(forecast_data["forecast"])
    df_btc_usdt = pd.DataFrame(historical_data["result"])
    df_btc_usdt = df_btc_usdt.rename(columns={"date": "ds"})
    df_btc_usdt["unique_id"] = 1

    df_now["ds"] = pd.to_datetime(df_now['ds'])
    df_forecast["ds"] = pd.to_datetime(df_forecast['ds'])
    df_btc_usdt["ds"] = pd.to_datetime(df_btc_usdt['ds'])

    df_now["ds"] += pd.Timedelta(hours=3)
    df_forecast["ds"] += pd.Timedelta(hours=3)
    df_btc_usdt["ds"] += pd.Timedelta(hours=3)

    last_close_price = df_btc_usdt["close"].iloc[-1]
    df_forecast_norm = df_forecast.copy()

    if model in ["arch", "garch"]:
        df_forecast_norm.iloc[:, 1] = last_close_price * (1 + df_forecast_norm.iloc[:, 1] / 100).cumprod()
        df_forecast_norm.iloc[:, 2] = last_close_price * (1 + df_forecast_norm.iloc[:, 2] / 100).cumprod()
        df_forecast_norm.iloc[:, 3] = last_close_price * (1 + df_forecast_norm.iloc[:, 3] / 100).cumprod()
        df_forecast_norm.iloc[:, 4] = last_close_price * (1 + df_forecast_norm.iloc[:, 4] / 100).cumprod()
        df_forecast_norm.iloc[:, 5] = last_close_price * (1 + df_forecast_norm.iloc[:, 5] / 100).cumprod()
        df_forecast_norm.iloc[:, 6] = last_close_price * (1 + df_forecast_norm.iloc[:, 6] / 100).cumprod()
        df_forecast_norm.iloc[:, 7] = last_close_price * (1 + df_forecast_norm.iloc[:, 7] / 100).cumprod()
        df_forecast_norm.iloc[:, 8] = last_close_price * (1 + df_forecast_norm.iloc[:, 8] / 100).cumprod()
        df_forecast_norm.iloc[:, 9] = last_close_price * (1 + df_forecast_norm.iloc[:, 9] / 100).cumprod()
        df_forecast_norm.iloc[:, 10] = last_close_price * (1 + df_forecast_norm.iloc[:, 10] / 100).cumprod()
        df_forecast_norm.iloc[:, 11] = last_close_price * (1 + df_forecast_norm.iloc[:, 11] / 100).cumprod()
    elif model in ["knn", "svr", "lightgbmregressor"]:
        df_forecast_norm.iloc[:, 2] = last_close_price * (1 + df_forecast_norm.iloc[:, 2] / 100).cumprod()
        df_forecast_norm.iloc[:, 3] = last_close_price * (1 + df_forecast_norm.iloc[:, 3] / 100).cumprod()
        df_forecast_norm.iloc[:, 4] = last_close_price * (1 + df_forecast_norm.iloc[:, 4] / 100).cumprod()
        df_forecast_norm.iloc[:, 5] = last_close_price * (1 + df_forecast_norm.iloc[:, 5] / 100).cumprod()
        df_forecast_norm.iloc[:, 6] = last_close_price * (1 + df_forecast_norm.iloc[:, 6] / 100).cumprod()
        df_forecast_norm.iloc[:, 7] = last_close_price * (1 + df_forecast_norm.iloc[:, 7] / 100).cumprod()
        df_forecast_norm.iloc[:, 8] = last_close_price * (1 + df_forecast_norm.iloc[:, 8] / 100).cumprod()
        df_forecast_norm.iloc[:, 9] = last_close_price * (1 + df_forecast_norm.iloc[:, 9] / 100).cumprod()
        df_forecast_norm.iloc[:, 10] = last_close_price * (1 + df_forecast_norm.iloc[:, 10] / 100).cumprod()
        df_forecast_norm.iloc[:, 11] = last_close_price * (1 + df_forecast_norm.iloc[:, 11] / 100).cumprod()
        df_forecast_norm.iloc[:, 12] = last_close_price * (1 + df_forecast_norm.iloc[:, 12] / 100).cumprod()

    return df_now, df_forecast, df_forecast_norm, df_btc_usdt

