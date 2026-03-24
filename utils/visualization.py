"""
Visualizer: Generates and saves EDA charts automatically.
"""
import warnings
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless rendering
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")


class Visualizer:
    """Auto-generates distribution, correlation, and target plots."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def plot_distributions(self, df: pd.DataFrame, numeric_cols: list[str]) -> list[str]:
        """Histogram grid for numeric columns."""
        if not numeric_cols:
            return []
        cols    = [c for c in numeric_cols if c in df.columns][:16]
        n_cols  = 4
        n_rows  = (len(cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

        for i, col in enumerate(cols):
            axes[i].hist(df[col].dropna(), bins=30, color="#4C8BC4", edgecolor="white", alpha=0.85)
            axes[i].set_title(col, fontsize=9, fontweight="bold")
            axes[i].set_xlabel("")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        path = str(self.output_dir / "distributions.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Saved → {path}")
        return [path]

    def plot_correlation_heatmap(self, df: pd.DataFrame) -> list[str]:
        """Correlation heatmap for numeric features."""
        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] < 2:
            return []
        corr = num_df.corr()
        size = max(8, min(corr.shape[0], 20))
        fig, ax = plt.subplots(figsize=(size, size * 0.8))
        # Upper triangle mask
        import numpy as np
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr, mask=mask, annot=corr.shape[0] <= 15, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        path = str(self.output_dir / "correlation_heatmap.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Saved → {path}")
        return [path]

    def plot_missing_values(self, df: pd.DataFrame) -> list[str]:
        """Bar chart of missing-value percentages."""
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]
        if missing_pct.empty:
            return []
        fig, ax = plt.subplots(figsize=(10, max(4, len(missing_pct) * 0.4)))
        bars = ax.barh(missing_pct.index, missing_pct.values, color="#E07B54")
        ax.set_xlabel("Missing (%)")
        ax.set_title("Missing Values per Column", fontsize=13, fontweight="bold")
        ax.axvline(x=60, color="red", linestyle="--", linewidth=1, label="60 % threshold")
        ax.legend()
        for bar, val in zip(bars, missing_pct.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=8)
        plt.tight_layout()
        path = str(self.output_dir / "missing_values.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Saved → {path}")
        return [path]

    def plot_target_distribution(self, df: pd.DataFrame, target: str) -> list[str]:
        """Distribution of the target variable."""
        if target not in df.columns:
            return []
        fig, ax = plt.subplots(figsize=(8, 4))
        if df[target].dtype == object or df[target].nunique() <= 20:
            df[target].value_counts().plot(kind="bar", ax=ax, color="#6BAE75", edgecolor="white")
            ax.set_xlabel(target)
            ax.set_ylabel("Count")
        else:
            ax.hist(df[target].dropna(), bins=40, color="#6BAE75", edgecolor="white")
            ax.set_xlabel(target)
        ax.set_title(f"Target Distribution: {target}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = str(self.output_dir / "target_distribution.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Saved → {path}")
        return [path]
