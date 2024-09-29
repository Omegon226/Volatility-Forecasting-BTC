import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import dill

from statsforecast import StatsForecast
from statsforecast.models import ARCH, GARCH
from mlforecast import MLForecast
from mlforecast.utils import PredictionIntervals
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
        ohlcv = get_all_data_for_ml_models()

        # Создание выборки для обучения модели
        train_df = pd.DataFrame(
            columns=["ds", "y", "unique_id"]
        )
        train_df["ds"] = ohlcv["date"].iloc[-324:-24]
        train_df["y"] = ohlcv["close_pct_change"].iloc[-324:-24]
        train_df["unique_id"] = 1
        train_df = train_df.reset_index(drop=True)

        # Создание выборки для тестирования модели
        test_df = pd.DataFrame(
            columns=["ds", "y", "unique_id"]
        )
        test_df["ds"] = ohlcv["date"].iloc[-24:]
        test_df["y"] = ohlcv["close_pct_change"].iloc[-24:]
        test_df["unique_id"] = 1
        test_df = test_df.reset_index(drop=True)

        retrain_arch_model(train_df, test_df)
        retrain_garch_model(train_df, test_df)
        retrain_svr_model(train_df, test_df)
        retrain_lgbmregressor(train_df, test_df)
        retrain_knn(train_df, test_df)

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
