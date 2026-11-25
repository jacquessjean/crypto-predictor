FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# The same library versions as the training run (out/model/v1/versions.json): a pickled model only loads reliably with them
RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.30" \
    "scikit-learn==1.9.1" "numpy==2.5.3" "pandas==3.1.1" "pyarrow==25.0.1" joblib
COPY src/cleanup.py src/final_model.py service/model_service.py ./

RUN useradd --system --uid 10001 app
USER app

# Models and training data are mounted at run time (docker-compose.yml), never baked into the image
ENV MODELS_ROOT=/models MODEL_DIR=/models/v2
CMD ["uvicorn", "model_service:app", "--host", "0.0.0.0", "--port", "8001"]
