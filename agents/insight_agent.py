"""
InsightAgent: Uses LLM to autonomously generate human-readable insights
from EDA profiles and ML evaluation results.
"""
import json
import openai
import textwrap
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal, Optional, Type

load_dotenv(override=True)  # Load environment variables from .env file

class InsightAgent:
    """
    Sends structured data context to LLM and returns narrative insights,
    recommendations, and a natural-language model card.
    """

    def __init__(self, model: str, reports_dir: Path = Path("reports")):
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(exist_ok=True)
        self.client = openai.OpenAI()
        self.model = model

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        profile: dict,
        metrics: Optional[dict] = None,
        insights: Optional[str] = None,
        mode: Literal["pre-training", "post-training"] = "pre-training",
    ) -> dict:
        """Generate EDA insights, model card, and recommendations."""

        if mode == "pre-training":
            print(f"[InsightAgent] Generating pre-training insights with {self.client} {self.model} …")
            return {
                "eda_insights": self._eda_insights(profile),
                "model_card": self._model_card(profile),  # Convert Pydantic model to dict
            }

        elif mode == "post-training":
            if insights and metrics:
                print(f"[InsightAgent] Generating post-training recommendations with {self.client} {self.model} …")
                return {
                    "recommendations": self._recommendations(profile, insights, metrics),
                }
            else:
                raise ValueError("Insights and metrics are required for post-training mode to generate recommendations.")

        else:
            raise ValueError(f"Invalid mode: {mode}")

    # ------------------------------------------------------------------
    # Internal prompts
    # ------------------------------------------------------------------

    def _call(self, system: str, user: str,) -> str:
        response = self.client.responses.create(
            model       = self.model,
            input    = [
            {"role": "system", "content": system,},
            {"role": "user", "content": user,},
            ],
        )
        return response.output_text.strip()
    
    def _structured_call(self, system: str, user: str, base_model: Type[BaseModel]) -> dict:
        response = self.client.responses.parse(
            model       = self.model,
            input    = [
            {"role": "system", "content": system,},
            {"role": "user", "content": user,},
            ],
            text_format=base_model,
        )
        return response.output_parsed.model_dump(mode="json")  # Convert Pydantic model to dict

    def _eda_insights(self, profile: dict) -> str:
        system = (
            "You are a senior data scientist. Given a dataset profile, "
            "produce concise, actionable EDA insights in plain English. "
            "Highlight data quality issues, distributions, and anomalies."
        )
        user = f"Dataset profile:\n{json.dumps(profile, default=str, indent=2)}"
        return self._call(system, user)

    def _model_card(
        self,
        profile: dict,
    ) -> dict:
        system = (
            "You are an ML engineer. Write a structured model card that includes: "
            "model name, task type, input features, and target feature."
            "Be concise and precise."
        )
        
        user = f"Model EDA insights :\n{json.dumps(profile, indent=2)}"
        return self._structured_call(system=system, user=user, base_model=ModelCard)

    def _recommendations(self, profile: dict, insights: str, ml_results: dict) -> str:
        system = (
            "You are a data science consultant. Given the dataset profile and model results, "
            "provide 3-5 prioritised, concrete next-step recommendations to improve model "
            "performance or data quality. Use numbered list format."
        )
        user = textwrap.dedent(f"""
            Dataset profile (summary):
            - Shape: {profile.get('shape')}
            - Missing pct: {profile.get('missing_pct')}
            - Numeric cols: {len(profile.get('numeric_cols', []))}
            - Categorical cols: {len(profile.get('categorical_cols', []))}

            EDA insights:
            {insights}

            Model profile:
            - Name: {ml_results.get('best_model_name')}
            - Task type: {ml_results.get('task_type')}


            Model performance metrics:
            {json.dumps({k: v for k, v in ml_results.get('metrics', {}).items() if k != 'classification_report'}, indent=2)}
        """).strip()
        return self._call(system, user)


class ModelCard(BaseModel):
    name: str = Field(..., description="A concise name for the model, e.g., 'Random Forest Classifier'")
    task_type: str = Field(..., description="The type of ML task, e.g., 'classification' or 'regression'")
    input_features: list[str] = Field(..., description="List of input feature names used by the model")
    target_feature: str = Field(..., description="The target variable the model is predicting")