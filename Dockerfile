FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

RUN mkdir -p data logs mlruns models

ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

ENV PYTHONPATH=/app

EXPOSE 5000

CMD ["python", "src/startup.py"]