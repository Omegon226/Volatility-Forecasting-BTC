from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.models import BaseOperator
from airflow.utils.dates import days_ago
from airflow.utils.decorators import apply_defaults
from datetime import timedelta
import logging
import requests


class CustomHttpOperator(BaseOperator):
    @apply_defaults
    def __init__(self, url, method='GET', headers=None, params=None, **kwargs):
        super(CustomHttpOperator, self).__init__(**kwargs)
        self.url = url
        self.method = method
        self.headers = headers
        self.params = params

    def execute(self, context):
        response = requests.request(method=self.method, url=self.url, headers=self.headers, params=self.params)
        if response.status_code != 200:
            raise ValueError(f"Request failed with status {response.status_code}")
        self.log.info(f"Response: {response.text}")



# Параметры DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
} 

# Определение DAG
with DAG(
    dag_id='retrain_ml_models',
    default_args=default_args,
    description='DAG для отправки HTTP GET запроса для переобучения моделей',
    schedule_interval='@hourly',  # Запуск ежедневно
    start_date=days_ago(1),
    catchup=False,
) as dag:

    # Задача для отправки GET запроса
    send_get_request = CustomHttpOperator(
        task_id='retrain_ml_models',
        url='http://backend:8000/api/retrain_ml_models/',
        method='GET',
        headers={"Content-Type": "application/json"},
    )

    send_get_request
