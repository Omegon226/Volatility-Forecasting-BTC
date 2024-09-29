#!/bin/bash

# Включить режим остановки скрипта при ошибке любой команды
set -e

echo "Запуск инициализации Airflow..."
docker-compose up airflow-init

echo "Ожидание 90 секунд перед запуском остальных сервисов..."
sleep 90

echo "Запуск остальных сервисов Docker Compose..."
docker-compose up