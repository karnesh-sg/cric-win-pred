
FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src

ENV PYTHONPATH=/app/src

CMD ["sh", "-c", "uvicorn cricketwp.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
