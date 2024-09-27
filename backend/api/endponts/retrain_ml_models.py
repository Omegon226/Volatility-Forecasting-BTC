from fastapi import APIRouter, HTTPException

from services.retrain_ml_models import retrain_all_models


router = APIRouter()


@router.get("/retrain_ml_models/")
def forecast():
    try:
        retrain_all_models()
        return {"result": "Success"}
    except Exception as error:
        raise HTTPException(status_code=500, detail=error)
