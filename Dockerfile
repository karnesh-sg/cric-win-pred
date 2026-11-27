
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

RUN useradd --create-home --shell /usr/sbin/nologin app

COPY --chown=app:app src ./src
COPY --chown=app:app artifacts ./artifacts

USER app

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn cricketwp.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
