import os

from services.retrain_ml_models import retrain_all_models


def init_ml_models_dir():
    if not os.path.isdir("ml_models"):
        os.mkdir("ml_models")


def init_retrain_all_models():
    retrain_all_models()
