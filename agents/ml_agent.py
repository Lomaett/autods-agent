"""
MLAgent: Autonomous agent for model selection, training, tuning, and evaluation.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error,
    classification_report,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
)


CLASSIFIER_POOL = {
    "LogisticRegression": (
        LogisticRegression(max_iter=1000),
        {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
    ),
    "RandomForestClassifier": (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10], "min_samples_split": [2, 5]},
    ),
    "GradientBoostingClassifier": (
        GradientBoostingClassifier(random_state=42),
        {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5]},
    ),
}

REGRESSOR_POOL = {
    "Ridge": (
        Ridge(),
        {"alpha": [0.1, 1.0, 10.0, 100.0]},
    ),
    "RandomForestRegressor": (
        RandomForestRegressor(random_state=42),
        {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]},
    ),
    "GradientBoostingRegressor": (
        GradientBoostingRegressor(random_state=42),
        {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5]},
    ),
}


class MLAgent:
    """
    Autonomously:
      1. Detects task type (classification / regression)
      2. Splits data
      3. Trains & cross-validates a pool of models
      4. Tunes the best model via RandomizedSearchCV
      5. Evaluates and saves the winner
    """

    def __init__(self, df: pd.DataFrame, target_column: str, task_type: str, models_dir: str = "models"):
        self.df           = df
        self.target       = target_column
        self.task_type    = task_type or ""
        self.models_dir   = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.best_model       = None
        self.best_model_name  = None
        self.metrics: dict    = {}
        self.label_encoder    = None
        self.feature_names    = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        print("[MLAgent] Starting autonomous ML pipeline …")
        X, y          = self._prepare()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        if not self.task_type:
            self._detect_task(y)
        candidate     = self._select_best(X_tr, y_tr)
        tuned         = self._tune(candidate, X_tr, y_tr)
        self.best_model = tuned
        self.metrics  = self._evaluate(tuned, X_te, y_te)
        self._save(tuned)
        print(f"[MLAgent] Best model: {self.best_model_name} | Metrics: {self.metrics}")
        return {
            "task_type":        self.task_type,
            "best_model_name":  self.best_model_name,
            "best_model":       self.best_model,
            "metrics":          self.metrics,
            "feature_names":    self.feature_names,
        }

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _prepare(self):
        df = self.df.dropna(subset=[self.target]).copy()
        X = df.drop(columns=[self.target]).copy()
        y = df[self.target]

        # Ensure all features are numeric and fully imputed before modeling.
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            X = pd.get_dummies(X, columns=non_numeric, drop_first=True, dtype=float)

        if X.isnull().any().any():
            X = X.apply(lambda col: col.fillna(col.median()) if pd.api.types.is_numeric_dtype(col) else col)
            X = X.fillna(0)

        self.feature_names = X.columns.tolist()

        # Encode string target
        if y.dtype == object:
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        return X.values, y.values

    def _detect_task(self, y):
        unique = np.unique(y)
        if len(unique) <= 20 and (y.dtype in [int, np.int64] or len(unique) < 10):
            self.task_type = "classification"
        else:
            self.task_type = "regression"
        print(f"[MLAgent] Task detected: {self.task_type}")

    def _select_best(self, X_tr, y_tr):
        pool    = CLASSIFIER_POOL if self.task_type == "classification" else REGRESSOR_POOL
        scoring = "f1_weighted" if self.task_type == "classification" else "r2"
        scores  = {}

        for name, (model, _) in pool.items():
            cv_scores = cross_val_score(model, X_tr, y_tr, cv=3, scoring=scoring, n_jobs=-1)
            scores[name] = cv_scores.mean()
            print(f"[MLAgent] {name:<35} CV {scoring}: {cv_scores.mean():.4f}")

        self.best_model_name = max(scores, key=scores.get)
        print(f"[MLAgent] Selected: {self.best_model_name}")
        return pool[self.best_model_name]

    def _tune(self, candidate, X_tr, y_tr):
        model, param_dist = candidate
        scoring = "f1_weighted" if self.task_type == "classification" else "r2"
        search  = RandomizedSearchCV(
            model, param_dist, n_iter=10, cv=3,
            scoring=scoring, random_state=42, n_jobs=-1
        )
        search.fit(X_tr, y_tr)
        print(f"[MLAgent] Best params: {search.best_params_}")
        return search.best_estimator_

    def _evaluate(self, model, X_te, y_te) -> dict:
        y_pred = model.predict(X_te)
        if self.task_type == "classification":
            metrics = {
                "accuracy":  round(accuracy_score(y_te, y_pred), 4),
                "f1_weighted": round(f1_score(y_te, y_pred, average="weighted"), 4),
            }
            try:
                y_prob = model.predict_proba(X_te)
                if y_prob.shape[1] == 2:
                    metrics["roc_auc"] = round(roc_auc_score(y_te, y_prob[:, 1]), 4)
            except AttributeError:
                pass
            metrics["classification_report"] = classification_report(y_te, y_pred)
        else:
            metrics = {
                "rmse": round(np.sqrt(mean_squared_error(y_te, y_pred)), 4),
                "mae":  round(mean_absolute_error(y_te, y_pred), 4),
                "r2":   round(r2_score(y_te, y_pred), 4),
            }
        return metrics

    def _save(self, model):
        path = self.models_dir / f"{self.best_model_name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"[MLAgent] Model saved → {path}")
