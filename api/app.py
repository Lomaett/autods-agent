"""
api/app.py — FastAPI REST interface for AutoDS Agent.

Endpoints
---------
POST /analyse    — Full pipeline (upload a dataset file).
POST /eda        — EDA only (no model training).
POST /predict    — Predict with a saved model.
GET  /reports    — List HTML reports.
GET  /models     — List saved model files.
GET  /health     — Health check.

Run locally:
    uvicorn api.app:app --reload --port 8080
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipelines.eda_pipeline import EDAPipeline
from pipelines.training_pipeline import PipelineResult, TrainingPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("OPENAI_API_KEY", "")

app = FastAPI(
    title="AutoDS Agent API",
    description="Autonomous Data Science Agent powered by OpenAI.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyseResponse(BaseModel):
    status: str
    task_type: str
    target_col: str
    best_model: str
    best_metrics: dict
    report_path: str
    warnings: List[str]
    recommendations: List[str]
    risks: List[str]
    next_steps: List[str]


class EDAResponse(BaseModel):
    status: str
    task_type: str
    target_col: str
    shape: dict
    missing_summary: dict
    warnings: List[str]
    llm_summary: str
    report_path: str


class PredictResponse(BaseModel):
    predictions: list


def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "upload.csv").suffix or ".csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.file.read())
    tmp.flush()
    return Path(tmp.name)


@app.get("/health", tags=["System"])
def health() -> dict:
    return {
        "status": "ok",
        "api_key_configured": bool(API_KEY),
        "reports_dir": str(REPORTS_DIR),
        "models_dir": str(MODELS_DIR),
    }


@app.get("/reports", tags=["Results"])
def list_reports():
    files = sorted(
        [*REPORTS_DIR.glob("*.html"), *REPORTS_DIR.glob("*.json")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return {"reports": [f.name for f in files]}


@app.get("/reports/{filename}", tags=["Results"])
def get_report(filename: str):
    path = REPORTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    media_type = "application/json" if path.suffix.lower() == ".json" else "text/html"
    return FileResponse(path, media_type=media_type)


@app.get("/models", tags=["Results"])
def list_models():
    files = sorted(MODELS_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"models": [f.name for f in files]}


@app.post("/analyse", response_model=AnalyseResponse, tags=["Pipeline"])
async def analyse(
    file: UploadFile = File(..., description="CSV / Excel / Parquet / JSON dataset"),
    target_col: Optional[str] = Form(None),
    task_type: Optional[str] = Form(None),
    n_trials: int = Form(15),
    cv_folds: int = Form(5),
    report_title: str = Form("AutoDS Report"),
    generate_llm_summary: bool = Form(True),
):
    """Full AutoDS pipeline: data → EDA → training → report."""
    del n_trials, cv_folds

    tmp_path = _save_upload(file)
    try:
        pipeline = TrainingPipeline(
            model="gpt-5.4-mini",
            reports_dir=REPORTS_DIR,
            models_dir=MODELS_DIR,
        )
        result: PipelineResult = pipeline.run(
            tmp_path,
            target_hint=target_col or None,
            task_type_hint=task_type or None,
            report_title=report_title,
            generate_llm_summary=generate_llm_summary,
        )
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    return AnalyseResponse(
        status="success",
        task_type=result.brief["task_type"],
        target_col=result.brief["target_col"],
        best_model=result.ml_result["best_model_name"],
        best_metrics=result.ml_result["metrics"],
        report_path=result.report["html_path"],
        warnings=result.brief["warnings"],
        recommendations=result.report["recommendations"],
        risks=result.report["risks"],
        next_steps=result.report["next_steps"],
    )


@app.post("/eda", response_model=EDAResponse, tags=["Pipeline"])
async def eda(
    file: UploadFile = File(...),
    target_col: Optional[str] = Form(None),
    task_type: Optional[str] = Form(None),
    report_title: str = Form("EDA Report"),
    generate_llm_summary: bool = Form(True),
):
    """EDA only — fast data profiling and report, no model training."""
    tmp_path = _save_upload(file)
    try:
        pipeline = EDAPipeline(reports_dir=str(REPORTS_DIR))
        brief, report = pipeline.run(
            str(tmp_path),
            target_hint=target_col or None,
            task_type_hint=task_type or None,
            report_title=report_title,
            generate_llm_summary=generate_llm_summary,
        )
    except Exception as exc:
        logger.exception("EDA error")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    return EDAResponse(
        status="success",
        task_type=brief["task_type"],
        target_col=brief["target_col"],
        shape=brief["clean_shape"],
        missing_summary=brief["missing_summary"],
        warnings=brief["warnings"],
        llm_summary=brief["llm_summary"],
        report_path=report["html_path"],
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="New data CSV"),
    model_name: str = Form(..., description="Filename in /models, e.g. XGBoost_best.pkl"),
    feature_cols: str = Form(..., description="Comma-separated feature column names"),
):
    """Run inference with a saved model."""
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    tmp_path = _save_upload(file)
    try:
        from utils.data_loader import load_dataframe

        df = load_dataframe(tmp_path)
        cols = [c.strip() for c in feature_cols.split(",") if c.strip()]
        tp = TrainingPipeline(model="gpt-5.4-mini")
        preds = tp.load_and_predict(str(model_path), df, cols)
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    return PredictResponse(predictions=preds.tolist())
