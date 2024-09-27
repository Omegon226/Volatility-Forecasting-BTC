from fastapi import APIRouter, HTTPException

from services.forecast import (
    make_forecast_with_arch,
    make_forecast_with_garch,
    make_forecast_with_svr,
    make_forecast_with_lgbmregressor,
    make_forecast_with_knn
)
from db.get_data_to_retrain_ml_models import get_pct_returns


router = APIRouter(prefix="/forecast")


@router.post("/arch/")
def forecast_with_arch():
    df_now = get_pct_returns(300)
    df_forecast = make_forecast_with_arch()
    return {
        "now": df_now.to_dict(orient='records'),
        "forecast": df_forecast.to_dict(orient='records'),
    }


@router.post("/garch/")
def forecast_with_garch():
    df_now = get_pct_returns(300)
    df_forecast = make_forecast_with_garch()
    return {
        "now": df_now.to_dict(orient='records'),
        "forecast": df_forecast.to_dict(orient='records'),
    }


@router.post("/svr/")
def forecast_with_svr():
    df_now = get_pct_returns(300)
    df_forecast = make_forecast_with_svr()
    return {
        "now": df_now.to_dict(orient='records'),
        "forecast": df_forecast.to_dict(orient='records'),
    }


@router.post("/lightgbmregressor/")
def forecast_with_lightgbmregressor():
    df_now = get_pct_returns(300)
    df_forecast = make_forecast_with_lgbmregressor()
    return {
        "now": df_now.to_dict(orient='records'),
        "forecast": df_forecast.to_dict(orient='records'),
    }

@router.post("/knn/")
def forecast_with_knn():
    df_now = get_pct_returns(300)
    df_forecast = make_forecast_with_knn()
    return {
        "now": df_now.to_dict(orient='records'),
        "forecast": df_forecast.to_dict(orient='records'),
    }
