# Используем официальный легкий образ Python
FROM python:3.11-slim

# Установка рабочей директории
WORKDIR /app

# Установка системных зависимостей для сборки некоторых пакетов (если потребуется)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копирование файлов проекта
COPY requirements.txt .
COPY ragflow_client.py .
COPY app.py .
COPY README.md .

# Установка зависимостей Python
RUN pip install --no-cache-dir -r requirements.txt

# Проброс порта Streamlit
EXPOSE 8501

# Настройка переменных окружения для Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Команда для запуска приложения
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
