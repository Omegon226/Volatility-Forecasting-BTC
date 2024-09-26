import os
import sys
import time
from datetime import datetime
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import ccxt
import pandas as pd
import warnings


warnings.filterwarnings('ignore')

# Биржа из которой будут браться данные с помощью CCXT
EXCHANGE = ccxt.okx()
# Инструмент в формате символа для обработки
SYMBOL = "BTC/USDT"
# Таймфрейм свеч
TIMEFRAME = "1h"
# Начальная дата
#FROM_TS = EXCHANGE.parse8601('2018-01-10 00:00:00')
FROM_TS = EXCHANGE.parse8601('2023-01-01 00:00:00')


def fetch_ohlcv_data(exchange, symbol, timeframe, since, limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=since, limit=limit)
        return ohlcv
    except Exception as e:
        print(f"Ошибка при получении данных OHLCV: {e}")
        sys.exit(1)

def write_to_influx(write_api, bucket, org, data_frame):
    try:
        for index, row in data_frame.iterrows():
            point = Point("btc_usdt") \
                .tag("symbol", "BTC-USDT") \
                .field("open", float(row['open'])) \
                .field("high", float(row['high'])) \
                .field("low", float(row['low'])) \
                .field("close", float(row['close'])) \
                .field("volume", float(row['volume'])) \
                .field("close_pct_change", float(row['close_pct_change'])) \
                .time(row['timestamp'])
            write_api.write(bucket=bucket, org=org, record=point)
        print(f"Записано {len(data_frame)} свечей в InfluxDB.")
    except Exception as e:
        print(f"Ошибка при записи данных в InfluxDB: {e}")
        sys.exit(1)

def main():
    # Считываем переменные окружения
    INFLUXDB_URL = os.getenv('INFLUXDB_URL', 'http://influxdb:8086')
    INFLUXDB_TOKEN = os.getenv('INFLUXDB_TOKEN', 'admin')
    INFLUXDB_BUCKET = os.getenv('INFLUXDB_BUCKET', 'admin')
    INFLUXDB_ORG = os.getenv('INFLUXDB_ORG', 'admin')

    print(INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_BUCKET, INFLUXDB_ORG)

    # Инициализируем клиента InfluxDB
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    # Инициализируем обменник CCXT
    exchange = ccxt.okx()
    exchange.enableRateLimit = True  # Включаем ограничение скорости

    # Парсим начальную дату
    from_ts = FROM_TS

    # Список для хранения данных
    ohlcv_list = []
    counter = 0

    # Цикл для получения данных
    while True:
        ohlcv = fetch_ohlcv_data(exchange, SYMBOL, TIMEFRAME, since=from_ts, limit=100)
        if not ohlcv:
            print("Нет новых данных для загрузки.")
            break

        ohlcv_list.extend(ohlcv)

        # Преобразуем данные в DataFrame для расчета процентного изменения
        df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x / 1000))
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Вычисляем процентное изменение цены закрытия
        df['close_pct_change'] = df['close'].pct_change() * 100  # В процентах
        # Замена NaN на 0 для первого значения
        df['close_pct_change'].fillna(0, inplace=True)

        # Записываем последние 100 записей в InfluxDB
        # Избегаем повторной записи ранее записанных данных
        latest_data = df.iloc[-100:].copy()

        #print(latest_data.head())
        
        write_to_influx(write_api, INFLUXDB_BUCKET, INFLUXDB_ORG, latest_data)

        print(f"Последняя запись: {exchange.iso8601(ohlcv[-1][0])}")

        # Обновляем время для следующего запроса
        from_ts = ohlcv[-2][0]  # Добавляем 1 мс, чтобы избежать дубликатов

        print(counter)
        counter += 1
        
        # Если получено меньше лимита, значит данных больше нет
        if len(ohlcv) < 10:
            break

        # Ограничение по времени (опционально)
        # Например, чтобы не превышать API лимиты
        time.sleep(exchange.rateLimit / 1000)

    # Закрываем клиента InfluxDB
    client.close()
    print("Загрузка данных завершена.")

if __name__ == "__main__":
    main()