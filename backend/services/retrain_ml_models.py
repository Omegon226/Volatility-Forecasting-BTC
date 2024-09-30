import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import dill

from statsforecast import StatsForecast
from statsforecast.models import ARCH, GARCH
from mlforecast import MLForecast
from mlforecast.utils import PredictionIntervals
from neuralforecast import NeuralForecast
from neuralforecast.models import NLinear, DLinear, KAN, NBEATS, LSTM
from neuralforecast.losses.pytorch import MQLoss
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import lightgbm as lgb
from datasetsforecast import losses
from utilsforecast.plotting import plot_series
import mlflow

from db.get_data_to_retrain_ml_models import get_all_data_for_ml_models


mlflow.set_tracking_uri("http://mlflow:5000")  # Для прода
#mlflow.set_tracking_uri("http://localhost:5000")    # Для отладки
mlflow.set_experiment("btc-usdt_volatility_experiment")


def retrain_all_models():
    try:
        ohlcv = get_all_data_for_ml_models(count=5000)

        # Создание выборки для обучения модели
        train_df_300 = pd.DataFrame(
            columns=["ds", "y", "unique_id"]
        )
        train_df_300["ds"] = ohlcv["date"].iloc[-324:-24]
        train_df_300["y"] = ohlcv["close_pct_change"].iloc[-324:-24]
        train_df_300["unique_id"] = 1
        train_df_300 = train_df_300.reset_index(drop=True)

        train_df_1000 = pd.DataFrame(
            columns=["ds", "y", "unique_id"]
        )
        train_df_1000["ds"] = ohlcv["date"].iloc[-1024:-24]
        train_df_1000["y"] = ohlcv["close_pct_change"].iloc[-1024:-24]
        train_df_1000["unique_id"] = 1
        train_df_1000 = train_df_1000.reset_index(drop=True)

        train_df_5000 = pd.DataFrame(
            columns=["ds", "y", "unique_id"]
        )
        train_df_5000["ds"] = ohlcv["date"].iloc[-5024:-24]
        train_df_5000["y"] = ohlcv["close_pct_change"].iloc[-5024:-24]
        train_df_5000["unique_id"] = 1
        train_df_5000 = train_df_5000.reset_index(drop=True)

        # Создание выборки для тестирования модели
        test_df = pd.DataFrame(
            columns=["ds", "y", "unique_id"]
        )
        test_df["ds"] = ohlcv["date"].iloc[-24:]
        test_df["y"] = ohlcv["close_pct_change"].iloc[-24:]
        test_df["unique_id"] = 1
        test_df = test_df.reset_index(drop=True)

        retrain_arch_model(train_df_300, test_df)
        retrain_garch_model(train_df_300, test_df)
        retrain_svr_model(train_df_300, test_df)
        retrain_lgbmregressor(train_df_300, test_df)
        retrain_knn(train_df_300, test_df)
        retrain_nlinear(train_df_5000, test_df)
        retrain_dlinear(train_df_5000, test_df)
        retrain_kan(train_df_5000, test_df)
        retrain_nbeats(train_df_5000, test_df)
        retrain_lstm(train_df_5000, test_df)

    except Exception as error:
        raise error


def retrain_arch_model(train_df, test_df):
    with mlflow.start_run(run_name=f'ARCH_{str(datetime.datetime.now())}') as run:
        # Лучшие параметры полученные в исследовании
        params = {
            "p": 84,
        }

        # Сохранение тегов
        mlflow.set_tag("model_name", "ARCH")
        mlflow.set_tag("model_type", "regression")
        # Сохранение параметров
        mlflow.log_params(params)

        # Создание модели
        model = StatsForecast(
            models=[ARCH(**params)],
            freq='h',
            n_jobs=-1
        )

        # Обучение моедли
        model.fit(train_df)

        # Прогнозирование для test датасета
        forecasts = model.forecast(48, level=[99, 95, 90, 75, 50])
        forecasts["unique_id"] = 1

        # Рассчёт метрик
        rmse = losses.rmse(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        mse = losses.mse(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        mae = losses.mae(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        smape = losses.smape(test_df["y"].values, forecasts.iloc[:, 2].values[:24])

        # Сохранение метрик
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("SMAPE", smape)

        # Сохранение визуализации
        fig = plot_series(
            pd.concat([train_df, test_df]),
            forecasts_df=forecasts,
            engine='matplotlib',
            level=[99, 95, 90, 75, 50],
        )
        fig.savefig('forecast.png', bbox_inches='tight')
        plt.close()
        mlflow.log_artifact("forecast.png", "forecast")

        # Сохранение модели
        with open("ml_models/arch.dill", "wb") as file:
            dill.dump(model, file)


def retrain_garch_model(train_df, test_df):
    with mlflow.start_run(run_name=f'GARCH_{str(datetime.datetime.now())}') as run:
        # Лучшие параметры полученные в исследовании
        params = {
            "p": 92,
            "q": 24
        }

        # Сохранение тегов
        mlflow.set_tag("model_name", "GARCH")
        mlflow.set_tag("model_type", "regression")
        # Сохранение параметров
        mlflow.log_params(params)

        # Создание модели
        model = StatsForecast(
            models=[GARCH(**params)],
            freq='h',
            n_jobs=-1
        )

        # Обучение моедли
        model.fit(train_df)

        # Прогнозирование для test датасета
        forecasts = model.forecast(48, level=[99, 95, 90, 75, 50])
        forecasts["unique_id"] = 1

        # Рассчёт метрик
        rmse = losses.rmse(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        mse = losses.mse(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        mae = losses.mae(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        smape = losses.smape(test_df["y"].values, forecasts.iloc[:, 2].values[:24])

        # Сохранение метрик
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("SMAPE", smape)

        # Сохранение визуализации
        fig = plot_series(
            pd.concat([train_df, test_df]),
            forecasts_df=forecasts,
            engine='matplotlib',
            level=[99, 95, 90, 75, 50],
        )
        fig.savefig('forecast.png', bbox_inches='tight')
        plt.close()
        mlflow.log_artifact("forecast.png", "forecast")

        # Сохранение модели
        with open("ml_models/garch.dill", "wb") as file:
            dill.dump(model, file)


def retrain_svr_model(train_df, test_df):
    with mlflow.start_run(run_name=f'SVR_{str(datetime.datetime.now())}') as run:
        # Лучшие параметры полученные в исследовании
        params = {
            "kernel": "poly",
            "degree": 3,
            "C": 0.9702637495163653
        }

        # Сохранение тегов
        mlflow.set_tag("model_name", "SVR")
        mlflow.set_tag("model_type", "regression")
        # Сохранение параметров
        mlflow.log_params(params)

        # Создание модели
        model = MLForecast(
            models=[SVR(**params)],
            freq='h',
            lags=list(range(1, 24, 1)),
        )

        # Обучение моедли
        model.fit(train_df, prediction_intervals=PredictionIntervals(n_windows=5, h=48))

        # Прогнозирование для test датасета
        # forecasts = model.predict(24, new_df=test_df)
        forecasts = model.predict(48, level=[99, 95, 90, 75, 50])
        forecasts["unique_id"] = 1

        # Рассчёт метрик
        rmse = losses.rmse(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        mse = losses.mse(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        mae = losses.mae(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        smape = losses.smape(test_df["y"].values, forecasts.iloc[:, 2].values[:24])

        # Сохранение метрик
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("SMAPE", smape)

        # Сохранение визуализации
        fig = plot_series(
            pd.concat([train_df, test_df]),
            forecasts_df=forecasts,
            engine='matplotlib',
            level=[99, 95, 90, 75, 50],
        )
        fig.savefig('forecast.png', bbox_inches='tight')
        plt.close()
        mlflow.log_artifact("forecast.png", "forecast")

        # Сохранение модели
        with open("ml_models/svr.dill", "wb") as file:
            dill.dump(model, file)


def retrain_lgbmregressor(train_df, test_df):
    with mlflow.start_run(run_name=f'LightGBM_{str(datetime.datetime.now())}') as run:
        # Лучшие параметры полученные в исследовании
        params = {
            "n_estimators": 382,
            "boosting_type": "dart",
            "num_leaves": 93,
            "max_depth": 7,
            "learning_rate": 0.04001572844964948,
            "verbose": -1
        }

        # Сохранение тегов
        mlflow.set_tag("model_name", "LightGBM")
        mlflow.set_tag("model_type", "regression")
        # Сохранение параметров
        mlflow.log_params(params)

        # Создание модели
        model = MLForecast(
            models=[lgb.LGBMRegressor(**params)],
            freq='h',
            lags=list(range(1, 17, 1)),
        )

        # Обучение моедли
        model.fit(train_df, prediction_intervals=PredictionIntervals(n_windows=5, h=48))

        # Прогнозирование для test датасета
        #forecasts = model.predict(24, new_df=test_df)
        forecasts = model.predict(48, level=[99, 95, 90, 75, 50])
        forecasts["unique_id"] = 1

        # Рассчёт метрик
        rmse = losses.rmse(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        mse = losses.mse(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        mae = losses.mae(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        smape = losses.smape(test_df["y"].values, forecasts.iloc[:, 2].values[:24])

        # Сохранение метрик
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("SMAPE", smape)

        # Сохранение визуализации
        fig = plot_series(
            pd.concat([train_df, test_df]),
            forecasts_df=forecasts,
            engine='matplotlib',
            level=[99, 95, 90, 75, 50],
        )
        fig.savefig('forecast.png', bbox_inches='tight')
        plt.close()
        mlflow.log_artifact("forecast.png", "forecast")

        # Сохранение модели
        with open("ml_models/lgbmregressor.dill", "wb") as file:
            dill.dump(model, file)


def retrain_knn(train_df, test_df):
    with mlflow.start_run(run_name=f'KNN_{str(datetime.datetime.now())}') as run:
        # Лучшие параметры полученные в исследовании
        params = {
            "n_neighbors": 27,
            "weights": 'uniform',
            "leaf_size": 34
        }

        # Сохранение тегов
        mlflow.set_tag("model_name", "KNN")
        mlflow.set_tag("model_type", "regression")
        # Сохранение параметров
        mlflow.log_params(params)

        # Создание модели
        model = MLForecast(
            models=[KNeighborsRegressor(**params)],
            freq='h',
            lags=list(range(1, 24, 1)),
        )

        # Обучение моедли
        model.fit(train_df, prediction_intervals=PredictionIntervals(n_windows=5, h=48))

        # Прогнозирование для test датасета
        #forecasts = model.predict(24, new_df=test_df)
        forecasts = model.predict(48, level=[99, 95, 90, 75, 50])
        forecasts["unique_id"] = 1

        # Рассчёт метрик
        rmse = losses.rmse(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        mse = losses.mse(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        mae = losses.mae(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        smape = losses.smape(test_df["y"].values, forecasts.iloc[:, 2].values[:24])

        # Сохранение метрик
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("SMAPE", smape)

        # Сохранение визуализации
        fig = plot_series(
            pd.concat([train_df, test_df]),
            forecasts_df=forecasts,
            engine='matplotlib',
            level=[99, 95, 90, 75, 50],
        )
        fig.savefig('forecast.png', bbox_inches='tight')
        plt.close()
        mlflow.log_artifact("forecast.png", "forecast")

        # Сохранение модели
        with open("ml_models/knn.dill", "wb") as file:
            dill.dump(model, file)


def retrain_nlinear(train_df, test_df):
    with mlflow.start_run(run_name=f'NLinear_{str(datetime.datetime.now())}') as run:
        # Лучшие параметры полученные в исследовании
        params = {
            "h": 48,
            "loss": MQLoss(level=[99, 95, 90, 75, 50]),
            "random_seed": 1,
            "input_size": 190,
            "max_steps": 3852,
            "batch_size": 73,
            "learning_rate": 0.013543424178956737,
        }

        # Сохранение тегов
        mlflow.set_tag("model_name", "NLinear")
        mlflow.set_tag("model_type", "regression")
        # Сохранение параметров
        mlflow.log_params(params)

        # Создание модели
        model = NeuralForecast(models=[NLinear(**params)], freq='h')

        # Обучение моедли
        model.fit(train_df)

        # Прогнозирование для test датасета
        forecasts = model.predict()
        forecasts = forecasts.reset_index()
        current_columns = forecasts.columns.tolist()
        current_columns[2] = current_columns[2].split("-")[0]
        forecasts.columns = current_columns

        # Рассчёт метрик
        rmse = losses.rmse(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        mse = losses.mse(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        mae = losses.mae(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        smape = losses.smape(test_df["y"].values, forecasts.iloc[:, 2].values[:24])

        # Сохранение метрик
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("SMAPE", smape)

        # Сохранение визуализации
        fig = plot_series(
            pd.concat([train_df.iloc[-300:], test_df]),
            forecasts_df=forecasts,
            engine='matplotlib',
            level=[99, 95, 90, 75, 50],
        )
        fig.savefig('forecast.png', bbox_inches='tight')
        plt.close()
        mlflow.log_artifact("forecast.png", "forecast")

        # Сохранение модели
        with open("ml_models/nlinear.dill", "wb") as file:
            dill.dump(model, file)


def retrain_dlinear(train_df, test_df):
    with mlflow.start_run(run_name=f'DLinear_{str(datetime.datetime.now())}') as run:
        # Лучшие параметры полученные в исследовании
        params = {
            "h": 48,
            "loss": MQLoss(level=[99, 95, 90, 75, 50]),
            "random_seed": 1,
            "input_size": 382,
            "max_steps": 3441,
            "batch_size": 103,
            "learning_rate": 0.03553664865176262,
        }

        # Сохранение тегов
        mlflow.set_tag("model_name", "DLinear")
        mlflow.set_tag("model_type", "regression")
        # Сохранение параметров
        mlflow.log_params(params)

        # Создание модели
        model = NeuralForecast(models=[DLinear(**params)], freq='h')

        # Обучение моедли
        model.fit(train_df)

        # Прогнозирование для test датасета
        forecasts = model.predict()
        forecasts = forecasts.reset_index()
        current_columns = forecasts.columns.tolist()
        current_columns[2] = current_columns[2].split("-")[0]
        forecasts.columns = current_columns

        # Рассчёт метрик
        rmse = losses.rmse(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        mse = losses.mse(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        mae = losses.mae(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        smape = losses.smape(test_df["y"].values, forecasts.iloc[:, 2].values[:24])

        # Сохранение метрик
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("SMAPE", smape)

        # Сохранение визуализации
        fig = plot_series(
            pd.concat([train_df.iloc[-300:], test_df]),
            forecasts_df=forecasts,
            engine='matplotlib',
            level=[99, 95, 90, 75, 50],
        )
        fig.savefig('forecast.png', bbox_inches='tight')
        plt.close()
        mlflow.log_artifact("forecast.png", "forecast")

        # Сохранение модели
        with open("ml_models/dlinear.dill", "wb") as file:
            dill.dump(model, file)


def retrain_kan(train_df, test_df):
    with mlflow.start_run(run_name=f'KAN_{str(datetime.datetime.now())}') as run:
        # Лучшие параметры полученные в исследовании
        params = {
            "h": 48,
            "loss": MQLoss(level=[99, 95, 90, 75, 50]),
            "random_seed": 1,
            "input_size": 480,
            "max_steps": 662,
            "batch_size": 108,
            "learning_rate": 0.008924943318328573
        }

        # Сохранение тегов
        mlflow.set_tag("model_name", "KAN")
        mlflow.set_tag("model_type", "regression")
        # Сохранение параметров
        mlflow.log_params(params)

        # Создание модели
        model = NeuralForecast(models=[KAN(**params)], freq='h')

        # Обучение моедли
        model.fit(train_df)

        # Прогнозирование для test датасета
        forecasts = model.predict()
        forecasts = forecasts.reset_index()
        current_columns = forecasts.columns.tolist()
        current_columns[2] = current_columns[2].split("-")[0]
        forecasts.columns = current_columns

        # Рассчёт метрик
        rmse = losses.rmse(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        mse = losses.mse(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        mae = losses.mae(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        smape = losses.smape(test_df["y"].values, forecasts.iloc[:, 2].values[:24])

        # Сохранение метрик
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("SMAPE", smape)

        # Сохранение визуализации
        fig = plot_series(
            pd.concat([train_df.iloc[-300:], test_df]),
            forecasts_df=forecasts,
            engine='matplotlib',
            level=[99, 95, 90, 75, 50],
        )
        fig.savefig('forecast.png', bbox_inches='tight')
        plt.close()
        mlflow.log_artifact("forecast.png", "forecast")

        # Сохранение модели
        with open("ml_models/kan.dill", "wb") as file:
            dill.dump(model, file)


def retrain_nbeats(train_df, test_df):
    with mlflow.start_run(run_name=f'NBEATS_{str(datetime.datetime.now())}') as run:
        # Лучшие параметры полученные в исследовании
        params = {
            "h": 48,
            "loss": MQLoss(level=[99, 95, 90, 75, 50]),
            "random_seed": 1,
            "input_size": 455,
            "max_steps": 400,
            "batch_size": 55,
            "learning_rate": 0.2696231301904332,
        }

        # Сохранение тегов
        mlflow.set_tag("model_name", "NBEATS")
        mlflow.set_tag("model_type", "regression")
        # Сохранение параметров
        mlflow.log_params(params)

        # Создание модели
        model = NeuralForecast(models=[NBEATS(**params)], freq='h')

        # Обучение моедли
        model.fit(train_df)

        # Прогнозирование для test датасета
        forecasts = model.predict()
        forecasts = forecasts.reset_index()
        current_columns = forecasts.columns.tolist()
        current_columns[2] = current_columns[2].split("-")[0]
        forecasts.columns = current_columns

        # Рассчёт метрик
        rmse = losses.rmse(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        mse = losses.mse(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        mae = losses.mae(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        smape = losses.smape(test_df["y"].values, forecasts.iloc[:, 2].values[:24])

        # Сохранение метрик
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("SMAPE", smape)

        # Сохранение визуализации
        fig = plot_series(
            pd.concat([train_df.iloc[-300:], test_df]),
            forecasts_df=forecasts,
            engine='matplotlib',
            level=[99, 95, 90, 75, 50],
        )
        fig.savefig('forecast.png', bbox_inches='tight')
        plt.close()
        mlflow.log_artifact("forecast.png", "forecast")

        # Сохранение модели
        with open("ml_models/nbeats.dill", "wb") as file:
            dill.dump(model, file)


def retrain_lstm(train_df, test_df):
    with mlflow.start_run(run_name=f'LSTM_{str(datetime.datetime.now())}') as run:
        # Лучшие параметры полученные в исследовании
        params = {
            "h": 48,
            "loss": MQLoss(level=[99, 95, 90, 75, 50]),
            "random_seed": 1,
            "input_size": 105,
            "max_steps": 200,
            "batch_size": 2,
            "learning_rate": 0.09490673043669288,
        }

        # Сохранение тегов
        mlflow.set_tag("model_name", "LSTM")
        mlflow.set_tag("model_type", "regression")
        # Сохранение параметров
        mlflow.log_params(params)

        # Создание модели
        model = NeuralForecast(models=[LSTM(**params)], freq='h')

        # Обучение моедли
        model.fit(train_df)

        # Прогнозирование для test датасета
        forecasts = model.predict()
        forecasts = forecasts.reset_index()
        current_columns = forecasts.columns.tolist()
        current_columns[2] = current_columns[2].split("-")[0]
        forecasts.columns = current_columns

        # Рассчёт метрик
        rmse = losses.rmse(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        mse = losses.mse(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        mae = losses.mae(test_df["y"].values, forecasts.iloc[:, 2].values[:24])
        smape = losses.smape(test_df["y"].values, forecasts.iloc[:, 2].values[:24])

        # Сохранение метрик
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("SMAPE", smape)

        # Сохранение визуализации
        fig = plot_series(
            pd.concat([train_df.iloc[-300:], test_df]),
            forecasts_df=forecasts,
            engine='matplotlib',
            level=[99, 95, 90, 75, 50],
        )
        fig.savefig('forecast.png', bbox_inches='tight')
        plt.close()
        mlflow.log_artifact("forecast.png", "forecast")

        # Сохранение модели
        with open("ml_models/lstm.dill", "wb") as file:
            dill.dump(model, file)
