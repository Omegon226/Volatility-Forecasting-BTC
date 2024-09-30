from db.get_data_to_retrain_ml_models import get_btc_usdt_data


def get_btc_usdt_data_from_influxdb(count_of_points):
    df = get_btc_usdt_data(count_of_points)

    return df
