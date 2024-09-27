import datetime
import dill


def make_forecast_with_arch():
    with open("ml_models/arch.dill", "rb") as file:
        model = dill.load(file)

    forecast = forecasts = model.predict(48, level=[95, 90])
    forecasts["unique_id"] = 1
    forecast = forecast.iloc[24:]
    forecast = forecast.reset_index(drop=True)

    return forecast


def make_forecast_with_garch():
    with open("ml_models/garch.dill", "rb") as file:
        model = dill.load(file)

    forecast = forecasts = model.predict(48, level=[95, 90])
    forecasts["unique_id"] = 1
    forecast = forecast.iloc[24:]
    forecast = forecast.reset_index(drop=True)

    return forecast


def make_forecast_with_svr():
    with open("ml_models/svr.dill", "rb") as file:
        model = dill.load(file)

    forecast = forecasts = model.predict(48, level=[95, 90])
    forecasts["unique_id"] = 1
    forecast = forecast.iloc[24:]
    forecast = forecast.reset_index(drop=True)

    return forecast


def make_forecast_with_lgbmregressor():
    with open("ml_models/lgbmregressor.dill", "rb") as file:
        model = dill.load(file)


    forecast = forecasts = model.predict(48, level=[95, 90])
    forecasts["unique_id"] = 1
    forecast = forecast.iloc[24:]
    forecast = forecast.reset_index(drop=True)

    return forecast


def make_forecast_with_knn():
    with open("ml_models/knn.dill", "rb") as file:
        model = dill.load(file)

    forecast = forecasts = model.predict(48, level=[95, 90])
    forecasts["unique_id"] = 1
    forecast = forecast.iloc[24:]
    forecast = forecast.reset_index(drop=True)

    return forecast
