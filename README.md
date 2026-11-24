# Crypto Predictor

Crypto Predictor is an educational cryptocurrency forecasting dashboard. It fetches recent Binance market data, builds features for a selected USDT trading pair, requests a prediction from a separate model service, and records the result for later outcome tracking.

> **Important:** This project is for education and experimentation, not financial advice. Predictions are uncertain and should not be used as the basis for investment decisions. The Docker setup in this repository currently runs a demo model stub rather than a trained model.

## Features

- View recent candlestick data for supported USDT pairs.
- Request a prediction with the `BUY`, `SELL`, or `WAIT` action and up/flat/down probabilities.
- Review prediction history, summary statistics, and resolved outcomes.
- For an unexpired `BUY` prediction, optionally watch the position in the open browser tab. The page can re-check it at subsequent 15-minute bars until the one-hour prediction horizon ends. This watch stops when the tab is closed.

The re-check button is intentionally shown only for an unexpired `BUY` prediction when that pair is not already being watched.

## Run with Docker Compose

### Requirements

- Docker Desktop (or Docker Engine) with the Docker Compose plugin.
- Port `8080` available on your machine.

From the repository root, build and start the services:

```sh
docker compose up --build -d
```

Open [http://localhost:8080](http://localhost:8080) in a browser. To inspect service state and logs:

```sh
docker compose ps
docker compose logs -f
```

Stop the services with:

```sh
docker compose down
```

Prediction data is stored in the Compose-managed `api_data` volume and survives container rebuilds and `docker compose down`. Removing that volume will permanently delete the stored database.

## Services and layout

Compose builds the services from the paths that exist in this repository:

| Service | Source | Purpose |
| --- | --- | --- |
| `web` | `app/web/` | Serves the HTML, CSS, and JavaScript with unprivileged Nginx. The browser sends API requests through Nginx at `/api/`. |
| `api` | `app/api/` and `src/cleanup.py` | FastAPI application. Fetches market data, creates live features, calls the model service, stores predictions in SQLite, and resolves completed predictions. |
| `model` | `service/stub.py` | Lightweight HTTP model-service stub used by the current Compose setup. Provides a health endpoint and accepts prediction requests. |

The API reaches the model at `http://model:8001`; the model is not published to the host. The API also needs outbound access to Binance for market candles and outcome resolution. The `front` network provides the web/API connection and outbound access, while the `back` network is internal and connects only the API and model.

The API database is stored at `/app/appdata/predictions.db` in the container and persisted using the `api_data` named volume.

## API endpoints

The web app uses these API routes:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/pairs` | List supported trading pairs. |
| `GET` | `/api/candles` | Fetch recent candles for a pair and interval. |
| `POST` | `/api/predict` | Request and record a prediction. |
| `GET` | `/api/predictions` | Read prediction history. |
| `GET` | `/api/stats` | Read aggregate prediction and outcome statistics. |

The model service exposes `GET /health` and `POST /v1/predict` on the internal Compose network.

## Model limitation

There are no trained model artifacts or production model entry point in the current checkout. To keep the Compose stack buildable, `service/Dockerfile` runs `service/stub.py`. The stub validates the feature payload and returns demonstration outputs; it does **not** make genuine trained-model predictions. Its output can therefore differ from a model used during a local run. For meaningful model results, replace the stub with a trained model service that implements the same health and prediction endpoints before relying on its outputs.
