FROM python:3.11
USER root

# Установка JupyterLab и необходимых пакетов
RUN pip install --upgrade pip

# Создание рабочей директории
WORKDIR /mlflow

RUN pip install mlflow==2.16.0

# Копирование requirements.txt в контейнер
COPY requirements.txt /tmp/

# Установка зависимостей из requirements.txt
RUN pip install -r /tmp/requirements.txt

# Установка SQlite переменную
ENV BACKEND_URI sqlite:///mlflow.db

# Expose the webserver port
EXPOSE 5000

# Запуск MLFlow
CMD mlflow ui --backend-store-uri $BACKEND_URI --host 0.0.0.0

