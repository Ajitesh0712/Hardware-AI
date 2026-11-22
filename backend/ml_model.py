"""Load and apply the model trained in Data Science/06_model_development.ipynb.

This module deliberately does not retrain on application startup or during a
request. Training data stays in the offline notebook workflow; this function
normalizes only the current live product into the saved feature schema.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import pandas as pd

from backend.product_features import FEATURE_COLUMNS, product_features

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("ELECTRONICS_MODEL_PATH", BASE_DIR / "model_artifacts" / "product_score.joblib"))
_ARTIFACT = None


def _load_artifact():
    global _ARTIFACT
    if _ARTIFACT is None and MODEL_PATH.exists():
        try:
            artifact = joblib.load(MODEL_PATH)
            if artifact.get("feature_columns") != FEATURE_COLUMNS:
                raise ValueError("Saved model feature order does not match live feature schema")
            _ARTIFACT = artifact
        except Exception as exc:
            logger.exception("Could not load ElectronicsAI model artifact: %s", exc)
            _ARTIFACT = False
    return _ARTIFACT


def predict_score(product, requirements=None):
    """Return an estimated 0–1 quality/popularity/discount proxy score."""
    artifact = _load_artifact()
    if not artifact:
        # The application remains usable when an artifact is not packaged.
        # 0.5 communicates neutral evidence without inventing a ranking signal.
        return 0.5
    row = product_features(product)
    frame = pd.DataFrame([[row[name] for name in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    value = float(artifact["model"].predict(frame)[0])
    return max(0.0, min(1.0, value))


def model_metadata():
    artifact = _load_artifact()
    if not artifact:
        return {"available": False, "path": str(MODEL_PATH)}
    return {
        "available": True,
        "model_name": artifact.get("model_name"),
        "feature_columns": artifact.get("feature_columns", []),
        "target_definition": artifact.get("target_definition"),
        "metrics": artifact.get("metrics", []),
    }
