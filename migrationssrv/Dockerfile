FROM python:3.11
USER root

# Установка JupyterLab и необходимых пакетов
RUN pip install --upgrade pip

# Копирование requirements.txt в контейнер
COPY requirements.txt /tmp/

# Установка зависимостей из requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Скопировать скрипт
COPY init_influxdb.py ./

# Установить точку входа для выполнения скрипта
ENTRYPOINT ["python", "init_influxdb.py"]