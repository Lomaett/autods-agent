"""
training_pipeline.py — Full end-to-end AutoDS pipeline.
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from agents.data_agent import DataAgent
from agents.insight_agent import InsightAgent
from agents.ml_agent import MLAgent

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    brief: dict
    ml_result: dict
    report: dict


class TrainingPipeline:
    """Autonomous end-to-end data science pipeline."""

    def __init__(
        self,
        client: str = "openai",
        model: str = "gpt-5.4-mini",
        target_feature: Optional[str] = None,
        task_type_hint: Optional[str] = None,
        reports_dir: Path = Path("reports"),
        models_dir: Path = Path("models"),
    ) -> None:
        self.client = client
        self.model = model
        self.reports_dir = Path(reports_dir)
        self.models_dir = Path(models_dir)
        self.target_feature = target_feature
        self.task_type_hint = task_type_hint
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        source: Any,
        *,
        target_hint: Optional[str] = None,
        task_type_hint: Optional[str] = None,
        report_title: str = "AutoDS Report",
        generate_llm_summary: bool = True,
    ) -> PipelineResult:
        logger.info("════════════ AutoDS Pipeline — START ════════════")

        resolved_target = target_hint or self.target_feature
        if not resolved_target:
            raise ValueError("A target column is required. Pass target_hint or target_feature.")

        # Step 1 — Data understanding
        logger.info("[1/4] DataAgent …")
        data_agent = DataAgent(data_path=source, target_feature=resolved_target)
        data_result = data_agent.run()
        profile = data_result["profile"]

        brief = {
            "task_type": task_type_hint or self.task_type_hint or "auto",
            "target_col": data_result["target_feature"],
            "shape": {
                "rows": profile["shape"][0],
                "columns": profile["shape"][1],
            },
            "clean_shape": {
                "rows": data_result["clean_df"].shape[0],
                "columns": data_result["clean_df"].shape[1],
            },
            "missing_summary": profile.get("missing_pct", {}),
            "warnings": [],
            "llm_summary": "",
            "profile": profile,
        }

        # Step 2 — Insights before training
        logger.info("[2/4] InsightAgent …")
        pre_insights = {"eda_insights": "", "model_card": {}}
        if generate_llm_summary:
            insight_agent = InsightAgent(model=self.model, reports_dir=self.reports_dir)
            pre_insights = insight_agent.run(profile=profile, mode="pre-training")
            brief["llm_summary"] = pre_insights.get("eda_insights", "")

        # Step 3 — Model training
        logger.info("[3/4] MLAgent …")
        forced_task = task_type_hint or self.task_type_hint or pre_insights.get("model_card", {}).get("task_type")
        ml_agent = MLAgent(
            df=data_result["clean_df"],
            target_column=data_result["target_feature"],
            task_type=forced_task or "",
            models_dir=str(self.models_dir),
        )
        ml_result = ml_agent.run()
        brief["task_type"] = ml_result["task_type"]

        # Step 4 — Recommendations
        logger.info("[4/4] InsightAgent …")
        recommendations = []
        if generate_llm_summary:
            insight_agent = InsightAgent(model=self.model, reports_dir=self.reports_dir)
            post = insight_agent.run(
                profile=profile,
                metrics=ml_result,
                insights=pre_insights.get("eda_insights", ""),
                mode="post-training",
            )
            rec_text = post.get("recommendations", "")
            recommendations = [line.strip() for line in rec_text.splitlines() if line.strip()]

        report_path = str(self.reports_dir / f"{report_title.lower().replace(' ', '_')}.json")
        report = {
            "html_path": report_path,
            "recommendations": recommendations,
            "risks": [],
            "next_steps": recommendations[:3],
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "brief": brief,
                    "ml_result": {
                        "task_type": ml_result["task_type"],
                        "best_model_name": ml_result["best_model_name"],
                        "metrics": ml_result["metrics"],
                        "feature_names": ml_result["feature_names"],
                    },
                    "report": report,
                },
                f,
                indent=2,
                default=str,
            )

        logger.info("Best model   : %s", ml_result["best_model_name"])
        logger.info("Best metrics : %s", ml_result["metrics"])
        logger.info("%-15s: %s", report_title, report["html_path"])
        logger.info("════════════ AutoDS Pipeline — DONE  ════════════")

        return PipelineResult(brief=brief, ml_result=ml_result, report=report)

    def predict(self, result: PipelineResult, df: pd.DataFrame):
        """Run inference with the best fitted model."""
        if result.ml_result["best_model"] is None:
            raise RuntimeError("No fitted model in PipelineResult.")
        X = df[result.ml_result["feature_names"]]
        return result.ml_result["best_model"].predict(X)

    def load_and_predict(self, model_path: str, df: pd.DataFrame, feature_cols: list):
        """Load a persisted model and predict."""
        import pickle

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model.predict(df[feature_cols])
