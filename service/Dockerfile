FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.30"
COPY service/stub.py ./stub.py

RUN useradd --system --uid 10001 app
USER app

CMD ["uvicorn", "stub:app", "--host", "0.0.0.0", "--port", "8001"]
