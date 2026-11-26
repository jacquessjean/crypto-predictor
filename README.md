# Crypto Predictor

Crypto Predictor is an educational cryptocurrency forecasting app (CSCE 3602 course project). It fetches recent Binance market data, builds features for a selected USDT trading pair, asks a trained model whether to buy now, records the prediction, and checks it against the real price an hour later. Checked predictions are sent back to the model, which retrains itself (online learning).

> **Important:** This project is for education and experimentation, not financial advice. Predictions are uncertain and should not be used as the basis for investment decisions.

## Features

- **Predict:** for the last closed 15-minute bar of a pair, the model gives the probability that the price will be up, flat or down by more than 0.2% (the Binance round-trip fee) in the next hour, the expected return after the fee, and an action. It says `BUY` only when the expected return beats a margin learned on validation data (an expected rise of about 0.29%); otherwise `WAIT`. `SELL` is switched off because it lost money on validation data.
- **History:** every prediction with its outcome after the hour, summary statistics, and a model card showing the live version, how many outcomes were sent back for learning, and the last retrain.
- **Past signals:** every trade model v2 would have made in November 2026 (data it never saw during development), with entry, exit and net return after the fee, next to buying at random.
- **Online learning:** each resolved prediction is sent to the model service. Once 50 new outcomes arrive it retrains in the background (plus a daily retrain as a backup), and the **Retrain now** button on the History tab starts one at once. A new version goes live only if it is not worse than the current one on held-out validation data (the safety gate).
- **Replay:** fills the history with the predictions the app would have made for the past hours, each using only the data available when its bar closed.
- For an unexpired `BUY` prediction, the page can re-check the position at the next 15-minute bars until the one-hour horizon ends, while the tab stays open.

## Run with Docker Compose

### Requirements

- Docker Desktop (or Docker Engine) with the Docker Compose plugin, with about 4 GB of memory for retraining.
- Port `8080` available on your machine.
- The training data the model retrains on, `out/processed/features.parquet`. It is not in the repository; build it once (it downloads the Binance history):

  ```sh
  python src/cleanup.py
  ```

From the repository root, build and start the services:

```sh
docker compose up --build -d
```

Open [http://localhost:8080](http://localhost:8080) in a browser. To inspect service state and logs (the model service logs every retrain):

```sh
docker compose ps
docker compose logs -f model
```

Stop the services with:

```sh
docker compose down
```

Prediction data is stored in the Compose-managed `api_data` volume and survives container rebuilds and `docker compose down`. Removing that volume permanently deletes the stored predictions.

### Replay the past 24 hours

```sh
docker compose exec api python -m app.api.replay --hours 24
```

This takes a couple of minutes. The rows appear on the History tab with a `replay` tag, and their outcomes are sent to the model service like live ones.

### Go back to model v2

Retrains write new versions (`v3`, `v4`, ...) and `current.json` into `out/model/`. To start again from v2:

```sh
rm -rf out/model/v3 out/model/v4 out/model/current.json
docker compose restart model
```

## Services and layout

| Service | Source | Purpose |
| --- | --- | --- |
| `web` | `app/web/` | React frontend built with Vite and served by unprivileged Nginx. The browser sends API requests through Nginx at `/api/`. |
| `api` | `app/api/`, `src/cleanup.py` | FastAPI application. Fetches market data, builds live features, calls the model service, stores predictions in SQLite, resolves them after the hour, and sends the outcomes back as feedback. |
| `model` | `service/model_service.py`, `src/final_model.py` | The trained model: a calibrated 3-class Random Forest. Serves predictions, stores feedback in `out/model/feedback.db`, and retrains behind the safety gate. |

The API reaches the model at `http://model:8001`; the model is not published to the host. The `front` network connects the web app and the API and gives the API outbound access to Binance, while the `back` network is internal and connects only the API and the model.

The shipped models are in `out/model/v1` and `out/model/v2` and are mounted read-only; v2 is live until a retrain is promoted.

## API endpoints

The web app uses these API routes:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/pairs` | List supported trading pairs. |
| `GET` | `/api/candles` | Fetch recent candles for a pair and interval. |
| `POST` | `/api/predict` | Request and record a prediction. |
| `GET` | `/api/predictions` | Read prediction history. |
| `GET` | `/api/stats` | Read aggregate prediction and outcome statistics. |
| `GET` | `/api/backtest` | Read the November 2026 trades for the Past signals tab. |
| `GET` | `/api/model` | Read the live model version and online-learning status. |
| `POST` | `/api/model/retrain` | Start a retrain now (409 if one is running). |

The model service exposes these on the internal network:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Live version, feedback counts and the last retrain. |
| `POST` | `/v1/predict` | Probabilities, action, expected return and the Buy margin for 36 features. |
| `POST` | `/v1/feedback` | Store resolved predictions as new training samples. |
| `POST` | `/v1/retrain` | Retrain now and wait for the result. |
| `POST` | `/v1/retrain/start` | Start a retrain in the background (the Retrain now button). |

## Tests

```sh
PYTHONPATH=src:. .venv/bin/python -m pytest src -q
```
