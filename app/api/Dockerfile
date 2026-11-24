FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app \
    DB_PATH=/app/appdata/predictions.db
WORKDIR /app

RUN pip install --no-cache-dir \
    "APScheduler>=3.10.4" "fastapi>=0.115" "httpx>=0.28,<1.0" \
    "numpy>=1.24" "pandas>=2.0" "uvicorn[standard]>=0.30"

COPY app/ app/
COPY src/cleanup.py src/cleanup.py

RUN useradd --system --uid 10001 app \
    && mkdir -p /app/appdata \
    && chown app:app /app/appdata
USER app

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
