from fastapi import FastAPI
import logging

from core.init_app import init_ml_models_dir, init_retrain_all_models
from api.endponts.forecast import router as forecast_router
from api.endponts.retrain_ml_models import router as retrain_ml_models_router


app = FastAPI(title="BTC-USDT Volatility forecast")
app.include_router(forecast_router, prefix="/api", tags=["ml"])
app.include_router(retrain_ml_models_router, prefix="/api", tags=["ml"])

# Подготовка приложения к работе
init_ml_models_dir()
init_retrain_all_models()


@app.get("/")
def read_root():
    return {"Hello": "World"}


if __name__ == "__main__":
    # При запуске через отладчик PyCharm (и др. IDE) или через консоль файла main.py
    logging.info("Запуск backend компонента произведён через отладчик")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
else:
    # При запуске через команду Uvicorn (пример: python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000)
    logging.info("Запуск backend компонента произведён через команду python -m uvicorn")
