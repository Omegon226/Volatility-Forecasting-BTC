import datetime
import dill


def make_forecast_with_arch():
    with open("ml_models/arch.dill", "rb") as file:
        model = dill.load(file)

    forecast = forecasts = model.predict(48, level=[99, 95, 90, 75, 50])
    forecasts["unique_id"] = 1
    forecast = forecast.iloc[24:]
    forecast = forecast.reset_index(drop=True)

    return forecast


def make_forecast_with_garch():
    with open("ml_models/garch.dill", "rb") as file:
        model = dill.load(file)

    forecast = forecasts = model.predict(48, level=[99, 95, 90, 75, 50])
    forecasts["unique_id"] = 1
    forecast = forecast.iloc[24:]
    forecast = forecast.reset_index(drop=True)

    return forecast


def make_forecast_with_svr():
    with open("ml_models/svr.dill", "rb") as file:
        model = dill.load(file)

    forecast = forecasts = model.predict(48, level=[99, 95, 90, 75, 50])
    forecasts["unique_id"] = 1
    forecast = forecast.iloc[24:]
    forecast = forecast.reset_index(drop=True)

    return forecast


def make_forecast_with_lgbmregressor():
    with open("ml_models/lgbmregressor.dill", "rb") as file:
        model = dill.load(file)


    forecast = forecasts = model.predict(48, level=[99, 95, 90, 75, 50])
    forecasts["unique_id"] = 1
    forecast = forecast.iloc[24:]
    forecast = forecast.reset_index(drop=True)

    return forecast


def make_forecast_with_knn():
    with open("ml_models/knn.dill", "rb") as file:
        model = dill.load(file)

    forecast = forecasts = model.predict(48, level=[99, 95, 90, 75, 50])
    forecasts["unique_id"] = 1
    forecast = forecast.iloc[24:]
    forecast = forecast.reset_index(drop=True)

    return forecast


def make_forecast_with_nlinear():
    with open("ml_models/nlinear.dill", "rb") as file:
        model = dill.load(file)

    forecast = model.predict()
    forecast = forecast.reset_index()
    current_columns = forecast.columns.tolist()
    current_columns[2] = current_columns[2].split("-")[0]
    forecast.columns = current_columns
    forecast = forecast.iloc[24:]

    return forecast


def make_forecast_with_dlinear():
    with open("ml_models/dlinear.dill", "rb") as file:
        model = dill.load(file)

    forecast = model.predict()
    forecast = forecast.reset_index()
    current_columns = forecast.columns.tolist()
    current_columns[2] = current_columns[2].split("-")[0]
    forecast.columns = current_columns
    forecast = forecast.iloc[24:]

    return forecast


def make_forecast_with_kan():
    with open("ml_models/kan.dill", "rb") as file:
        model = dill.load(file)

    forecast = model.predict()
    forecast = forecast.reset_index()
    current_columns = forecast.columns.tolist()
    current_columns[2] = current_columns[2].split("-")[0]
    forecast.columns = current_columns
    forecast = forecast.iloc[24:]

    return forecast


def make_forecast_with_nbeats():
    with open("ml_models/nbeats.dill", "rb") as file:
        model = dill.load(file)

    forecast = model.predict()
    forecast = forecast.reset_index()
    current_columns = forecast.columns.tolist()
    current_columns[2] = current_columns[2].split("-")[0]
    forecast.columns = current_columns
    forecast = forecast.iloc[24:]

    return forecast


def make_forecast_with_lstm():
    with open("ml_models/lstm.dill", "rb") as file:
        model = dill.load(file)

    forecast = model.predict()
    forecast = forecast.reset_index()
    current_columns = forecast.columns.tolist()
    current_columns[2] = current_columns[2].split("-")[0]
    forecast.columns = current_columns
    forecast = forecast.iloc[24:]

    return forecast
