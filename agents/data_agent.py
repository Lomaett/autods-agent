"""
DataAgent: Autonomous agent for data ingestion, profiling, and preprocessing.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any
from utils.data_loader import DataLoader


class DataAgent:
    """
    Handles data ingestion, profiling, cleaning, and preprocessing.
    Operates autonomously: detects data types, missing values,
    outliers, and encoding requirements without manual intervention.
    """

    def __init__(self, data_path: str, target_feature: str):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"{self.data_path} does not exist")
        self.target_feature = target_feature
        self.raw_df: pd.DataFrame | None = None
        self.clean_df: pd.DataFrame | None = None
        self.profile: dict = {}
        self.loader = DataLoader()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Full autonomous pipeline: load → profile → clean → encode."""
        print("[DataAgent] Starting autonomous data pipeline …")
        self._load()
        self._profile()
        self._clean()
        self._encode()
        print("[DataAgent] Pipeline complete.")
        return {
            "profile": self.profile,
            "clean_df": self.clean_df,
            "target_feature": self.profile.get("target_feature", self.target_feature),
        }

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _load(self):
        print(f"[DataAgent] Loading data from: {self.data_path}")
        self.raw_df = self.loader.load(self.data_path)
        print(f"[DataAgent] Loaded {len(self.raw_df):,} rows × {self.raw_df.shape[1]} cols.")

    def _profile(self):
        df = self.raw_df
        if self.target_feature not in df.columns:
            raise ValueError(f"Target feature '{self.target_feature}' not found in data columns: {df.columns.tolist()}")
        numeric_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols  = df.select_dtypes(include=["datetime"]).columns.tolist()

        missing     = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        self.profile = {
            "shape":           df.shape,
            "numeric_cols":    numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols":   datetime_cols,
            "missing_values":  missing[missing > 0].to_dict(),
            "missing_pct":     missing_pct[missing_pct > 0].to_dict(),
            "duplicates":      int(df.duplicated().sum()),
            "target_feature":   self.target_feature,
            "dtypes":          df.dtypes.astype(str).to_dict(),
            "describe":        df.describe(include="all").to_dict(),
        }
        print(f"[DataAgent] Profile built. Target feature detected: '{self.target_feature}'")

    def _clean(self):
        df = self.raw_df.copy()

        # Drop columns with > 60 % missing
        high_missing = [c for c, pct in self.profile["missing_pct"].items() if pct > 60]
        if high_missing:
            print(f"[DataAgent] Dropping high-missing columns: {high_missing}")
            df.drop(columns=high_missing, inplace=True)

        # Remove duplicate rows
        before = len(df)
        df.drop_duplicates(inplace=True)
        dropped = before - len(df)
        if dropped:
            print(f"[DataAgent] Removed {dropped} duplicate rows.")

        # Impute: median for numerics, mode for categoricals
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)

        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)

        # IQR-based outlier clipping on numeric columns (excl. target)
        target = self.profile["target_feature"]
        for col in df.select_dtypes(include=[np.number]).columns:
            if col == target:
                continue
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        self.clean_df = df
        print(f"[DataAgent] Cleaning done. Shape: {df.shape}")

    def _encode(self):
        df     = self.clean_df.copy()
        target = self.profile["target_feature"]

        for col in df.select_dtypes(include=["object"]).columns:
            if col == target:
                continue
            if df[col].nunique() <= 10:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            else:
                # Frequency encoding for high-cardinality columns
                freq   = df[col].value_counts(normalize=True)
                df[col] = df[col].map(freq)

        self.clean_df = df
        print("[DataAgent] Encoding complete.")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_profile_summary(self) -> str:
        p = self.profile
        return (
            f"Dataset shape : {p['shape']}\n"
            f"Numeric cols  : {p['numeric_cols']}\n"
            f"Categorical   : {p['categorical_cols']}\n"
            f"Missing vals  : {p['missing_values']}\n"
            f"Duplicates    : {p['duplicates']}\n"
            f"Target feature : {p['target_feature']}\n"
        )
    