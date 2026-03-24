# AutoDS Agent

Autonomous end-to-end data science agent powered by scikit-learn + OpenAI.
Drop in a dataset and get a cleaned frame, trained model, evaluation metrics, EDA charts, and AI-generated insights.

## Architecture

```text
autods-agent/
├── agents/
│   ├── data_agent.py
│   ├── ml_agent.py
│   └── insight_agent.py
├── pipelines/
│   ├── eda_pipeline.py
│   └── training_pipeline.py
├── api/
│   └── app.py
├── utils/
│   ├── data_loader.py
│   └── visualization.py
├── models/
├── reports/
└── notebooks/
```

## Quickstart

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Set your API key

```bash
export OPENAI_API_KEY="sk-..."
```

### 3) Run API

```bash
uvicorn api.app:app --reload --port 8080
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| GET | `/reports` | List generated reports |
| GET | `/reports/{filename}` | Retrieve a report file |
| GET | `/models` | List trained model files |
| POST | `/eda` | EDA-only run |
| POST | `/analyse` | Full pipeline (EDA + training + recommendations) |
| POST | `/predict` | Batch predictions from saved model |

## Example Requests

### Analyse

```bash
curl -X POST http://localhost:8080/analyse \
  -F "file=@data/dataset.csv" \
  -F "target_col=target"
```

### EDA

```bash
curl -X POST http://localhost:8080/eda \
  -F "file=@data/dataset.csv" \
  -F "target_col=target"
```

### Predict

```bash
curl -X POST http://localhost:8080/predict \
  -F "file=@data/new_data.csv" \
  -F "model_name=RandomForestClassifier.pkl" \
  -F "feature_cols=age,income,score"
```

## Notes

- `target_col` is required for `/eda` and `/analyse`.
- Models are persisted in `models/` as `.pkl`.
- EDA artifacts are stored in `reports/`.
