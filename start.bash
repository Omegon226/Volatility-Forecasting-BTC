# Включить отображение ошибок
$ErrorActionPreference = "Stop"

Write-Host "Запуск инициализации Airflow..."
docker-compose up airflow-init

Write-Host "Ожидание 90 секунд перед запуском остальных сервисов..."
Start-Sleep -Seconds 90

Write-Host "Запуск остальных сервисов Docker Compose..."
docker-compose up