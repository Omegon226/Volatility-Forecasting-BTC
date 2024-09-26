FROM python:3.11
USER root

# Установка JupyterLab и необходимых пакетов
RUN pip install --upgrade pip

# Копирование requirements.txt в контейнер
COPY requirements.txt /tmp/

# Установка зависимостей из requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Создание рабочей директории
WORKDIR /home/jovyan/work

# Expose the webserver port
EXPOSE 8501

#HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
#ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501"]