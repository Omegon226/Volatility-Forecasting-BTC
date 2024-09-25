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
EXPOSE 8000

# Запуск JupyterLab
#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]