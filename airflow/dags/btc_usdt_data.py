from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import ccxt
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import os
import logging
import pandas as pd

# Параметры DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
} 

# Инициализация DAG
with DAG(
    dag_id='btc_usdt_to_influxdb',
    default_args=default_args,
    description='Получение 100 часовых свечей BTC-USDT, трансформируем их и сохранение в InfluxDB',
    schedule_interval='@hourly',  # Запуск DAG каждый час
    start_date=days_ago(1),
    catchup=False,
) as dag:

    def extract_btc_usdt_data(**kwargs):
        symbol = 'BTC/USDT'  # Формат символа в ccxt
        timeframe = '1h'
        limit = 100

        try:
            exchange = ccxt.okx()

            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

            logging.info(ohlcv)

            # Преобразуем данные в нужный формат
            processed_data = []
            for entry in ohlcv:
                candle = {
                    'timestamp': entry[0],  # В миллисекундах
                    'open': float(entry[1]),
                    'high': float(entry[2]),
                    'low': float(entry[3]),
                    'close': float(entry[4]),
                    'volume': float(entry[5]),
                }
                processed_data.append(candle)

            # Передаем данные в следующий таск через XCom
            kwargs['ti'].xcom_push(key='btc_usdt_data', value=processed_data)

            logging.info(kwargs)

        except Exception as error:
            raise Exception(f"Ошибка при получении данных с OKX: {error}")


    def transform_btc_usdt_data(**kwargs):
        try:
            # Получение данных из XCom
            ti = kwargs['ti']
            data = ti.xcom_pull(key='btc_usdt_data', task_ids='extract_btc_usdt_data')

            logging.info(data)

            if not data:
                raise ValueError("Нет данных для преобразования.")

            # Преобразование данных в DataFrame
            df = pd.DataFrame(data)
            # Конвертация метки времени из миллисекунд в datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Сортировка по времени
            df = df.sort_values('timestamp')

            # Расчет процентного изменения закрытия
            df['close_pct_change'] = df['close'].pct_change() * 100  # В процентах

            # Замена NaN на 0 для первого значения
            df['close_pct_change'].fillna(0, inplace=True)

            # Преобразование 'timestamp' в строковый формат ISO
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')

            # Преобразование обратно в список словарей
            transformed_data = df.to_dict(orient='records')

            logging.info(transformed_data)

            # Передача преобразованных данных следующей задаче через XCom
            ti.xcom_push(key='btc_usdt_data', value=transformed_data)

        except Exception as e:
            raise Exception(f"Ошибка при преобразовании данных: {e}")



    def load_btc_usdt_data_to_influxdb(**kwargs):
        """
        Функция для сохранения данных в InfluxDB.
        """
        try:
            # Получение данных из XCom
            ti = kwargs['ti']
            data = ti.xcom_pull(key='btc_usdt_data', task_ids='transform_btc_usdt_data')

            logging.info(data)

            if not data:
                raise ValueError("Нет данных для сохранения в InfluxDB.")

            # Параметры подключения к InfluxDB
            INFLUXDB_URL = "http://influxdb:8086"
            INFLUXDB_TOCKEN = "admin"
            INFLUXDB_BUCKET = "admin"
            INFLUXDB_ORG = "admin"

            # Инициализация клиента InfluxDB
            client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOCKEN, org=INFLUXDB_ORG)
            write_api = client.write_api(write_options=SYNCHRONOUS)

            # Запись данных
            points = []
            for point in data:
                timestamp_ms = point['timestamp']
                point_influx = (
                    Point("btc_usdt")
                    .tag("symbol", "BTC-USDT")
                    .field("open", point['open'])
                    .field("high", point['high'])
                    .field("low", point['low'])
                    .field("close", point['close'])
                    .field("volume", point['volume'])
                    .field("close_pct_change", point['close_pct_change'])
                    .time(timestamp_ms, WritePrecision.MS)
                )
                points.append(point_influx)

            write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=points)

            client.close()

        except Exception as e:
            raise Exception(f"Ошибка при сохранении данных в InfluxDB: {e}")


    # Определение задач
    extract_data = PythonOperator(
        task_id='extract_btc_usdt_data',
        python_callable=extract_btc_usdt_data,
        provide_context=True,
    )

    transform_data = PythonOperator(
        task_id='transform_btc_usdt_data',
        python_callable=transform_btc_usdt_data,
        provide_context=True,
    )

    load_data = PythonOperator(
        task_id='load_btc_usdt_data_to_influxdb',
        python_callable=load_btc_usdt_data_to_influxdb,
        provide_context=True,
    )

    # Установка зависимостей
    extract_data >> transform_data >> load_data