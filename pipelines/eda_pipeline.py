"""
EDA Pipeline: Orchestrates DataAgent + visualization to produce a full
exploratory data analysis and saves artefacts to reports/.
"""

import json
from pathlib import Path
from typing import Optional

from agents.data_agent import DataAgent
from utils.visualization import Visualizer


class EDAPipeline:
    """Runs end-to-end EDA and returns API-friendly brief + report dicts."""

    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        data_path: str,
        *,
        target_hint: Optional[str] = None,
        task_type_hint: Optional[str] = None,
        report_title: str = "EDA Report",
        generate_llm_summary: bool = True,
    ):
        if not target_hint:
            raise ValueError("target_hint is required for EDA pipeline.")

        data_agent = DataAgent(data_path, target_feature=target_hint)
        result = data_agent.run()
        profile = result["profile"]
        clean_df = result["clean_df"]
        target = result["target_feature"]

        viz = Visualizer(output_dir=str(self.reports_dir))
        plots = []
        plots += viz.plot_distributions(clean_df, profile["numeric_cols"])
        plots += viz.plot_correlation_heatmap(clean_df)
        plots += viz.plot_missing_values(data_agent.raw_df)
        if target in clean_df.columns:
            plots += viz.plot_target_distribution(clean_df, target)

        profile_path = self.reports_dir / "eda_profile.json"
        with open(profile_path, "w") as f:
            json.dump(profile, f, default=str, indent=2)

        brief = {
            "task_type": task_type_hint or "auto",
            "target_col": target,
            "clean_shape": {
                "rows": clean_df.shape[0],
                "columns": clean_df.shape[1],
            },
            "missing_summary": profile.get("missing_pct", {}),
            "warnings": [],
            "llm_summary": "EDA profiling completed." if generate_llm_summary else "",
        }
        report = {
            "html_path": str(self.reports_dir / f"{report_title.lower().replace(' ', '_')}.json"),
            "plots": plots,
            "profile_path": str(profile_path),
        }
        with open(report["html_path"], "w", encoding="utf-8") as f:
            json.dump({"brief": brief, "report": report}, f, indent=2, default=str)
        return result, brief, report
