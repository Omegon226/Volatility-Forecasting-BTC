from fastapi import APIRouter, HTTPException

from services.get_btc_usdt_data import get_btc_usdt_data_from_influxdb


router = APIRouter()


@router.get("/get_btc_usdt_data/")
def get_btc_usdt_data(count_of_points: int = 300):
    try:
        df = get_btc_usdt_data_from_influxdb(count_of_points)
        return {"result": df.to_dict(orient='records')}
    except Exception as error:
        raise HTTPException(status_code=500, detail=error)
