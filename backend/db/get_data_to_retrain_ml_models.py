from influxdb_client import InfluxDBClient
from influxdb_client.client.flux_table import FluxTable
import os


# Параметры подключения
#INFLUX_URL = "http://localhost:8010"  # Для отладки
INFLUX_URL = "http://influxdb:8086"    # Для прода
INFLUX_TOKEN = "admin"
INFLUX_BUCKET = "admin"
INFLUX_ORG = "admin"


def get_all_data_for_ml_models():
    # Инициализация клиента
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)

    # Определение Flux запроса
    flux_query = f'''
    from(bucket: "admin")
      |> range(start: -324h)  
      |> filter(fn: (r) => r["_measurement"] == "btc_usdt")
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''

    # Выполнение запроса
    query_api = client.query_api()
    df = query_api.query_data_frame(flux_query)
    df = df.drop(columns=["result", "table", "_start", "_stop", "_measurement", "symbol"])
    df = df.rename(columns={"_time": "date"})

    # Обработка данных
    df = df.drop_duplicates(subset=['date'], keep='last')
    df = df.fillna(0)

    # Закрытие клиента
    client.close()

    return df


def get_pct_returns(count=300):
    # Инициализация клиента
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)

    # Определение Flux запроса
    flux_query = f'''
        from(bucket: "admin")
          |> range(start: -{count}h)  
          |> filter(fn: (r) => r["_measurement"] == "btc_usdt")
          |> filter(fn: (r) => r["_field"] == "close_pct_change")
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''

    # Выполнение запроса
    query_api = client.query_api()
    df = query_api.query_data_frame(flux_query)
    df = df.drop(columns=["result", "table", "_start", "_stop", "_measurement", "symbol"])
    df = df.rename(columns={"_time": "ds"})

    # Обработка данных
    df = df.drop_duplicates(subset=['ds'], keep='last')
    df = df.fillna(0)
    df["unique_id"] = 1

    # Закрытие клиента
    client.close()

    return df
