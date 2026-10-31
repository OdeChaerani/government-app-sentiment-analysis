"""
Model Loader and Inference Utility for Government App Sentiment Intelligence Dashboard
Loads trained model artifacts, evaluation benchmarks, confusion matrices, and LOAO results.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.utils.text_preprocessor import clean_text, analyze_text_features

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
NOTEBOOK_MODELS_DIR = REPO_ROOT / "notebooks" / "models"


def _artifact_path(filename: str) -> Path:
    """Prefer files exported by Member 2's notebook; use the integrated copy otherwise."""
    notebook_export = NOTEBOOK_MODELS_DIR / filename
    if notebook_export.exists():
        return notebook_export
    return MODELS_DIR / filename


@st.cache_data
def load_holdout_comparison() -> pd.DataFrame:
    """Loads the holdout evaluation results for the 3 candidate models."""
    path = _artifact_path("holdout_model_comparison.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_loao_results() -> pd.DataFrame:
    """Loads Leave-One-App-Out (LOAO) cross-app generalization benchmarks."""
    path = _artifact_path("loao_results.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_confusion_matrices() -> Dict[str, Any]:
    """Loads confusion matrix arrays for all evaluated models."""
    path = MODELS_DIR / "confusion_matrices.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


@st.cache_data
def load_classification_reports() -> Dict[str, Any]:
    """Derives exact per-class metrics from notebook confusion-matrix counts."""
    cms = load_confusion_matrices()
    labels = cms.get("labels", [])
    reports: Dict[str, Any] = {}
    for model_name, rows in cms.items():
        if model_name == "labels":
            continue
        matrix = np.asarray(rows, dtype=np.float64)
        if matrix.shape != (len(labels), len(labels)):
            continue
        supports = matrix.sum(axis=1)
        predicted = matrix.sum(axis=0)
        true_positive = np.diag(matrix)
        precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted != 0)
        recall = np.divide(true_positive, supports, out=np.zeros_like(true_positive), where=supports != 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
        total = float(supports.sum())
        accuracy = float(true_positive.sum() / total) if total else 0.0
        reports[model_name] = {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1-score": float(f1[i]),
                "support": int(supports[i]),
            }
            for i, label in enumerate(labels)
        }
        reports[model_name]["accuracy"] = accuracy
        reports[model_name]["macro avg"] = {
            "precision": float(precision.mean()),
            "recall": float(recall.mean()),
            "f1-score": float(f1.mean()),
            "support": int(total),
        }
        reports[model_name]["weighted avg"] = {
            "precision": float(np.average(precision, weights=supports)),
            "recall": float(np.average(recall, weights=supports)),
            "f1-score": float(np.average(f1, weights=supports)),
            "support": int(total),
        }
    return reports


@st.cache_resource(show_spinner="Loading trained sentiment model bundle...")
def load_model_bundle() -> Optional[Dict[str, Any]]:
    """
    Loads serialized inference bundle containing model, vectorizer, and metadata.
    Returns None if artifact is unavailable.
    """
    bundle_path = _artifact_path("rating_classifier_bundle.pkl")
    if bundle_path.exists():
        try:
            bundle = joblib.load(bundle_path)
            return bundle
        except Exception as e:
            st.error(f"Error loading model bundle from {bundle_path}: {e}")
            return None
    return None


def run_sentiment_inference(raw_text: str, bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes end-to-end sentiment inference:
    1. Preprocesses raw text with Member 1's exact clean_text pipeline.
    2. Vectorizes cleaned text with trained TF-IDF vectorizer.
    3. Generates class prediction.
    4. Extracts confidence / probability if supported by the model architecture.
    """
    if not raw_text.strip():
        return {"error": "Please enter review text to analyze."}

    text_stats = analyze_text_features(raw_text)
    cleaned_text = text_stats["cleaned_text"]

    if not cleaned_text.strip():
        return {
            "error": "The input review contained only emojis, punctuation, or non-alphanumeric characters, resulting in empty tokens after standard NLP filtering."
        }

    vectorizer = bundle["vectorizer"]
    model = bundle["model"]
    labels = bundle.get("labels", ["Negative", "Neutral", "Positive"])

    # Transform
    X = vectorizer.transform([cleaned_text])
    pred_label = model.predict(X)[0]

    # Probabilities / Confidences
    has_proba = hasattr(model, "predict_proba")
    probabilities = {}
    confidence_score = None

    if has_proba:
        probs = model.predict_proba(X)[0]
        probabilities = {lbl: float(round(p, 4)) for lbl, p in zip(labels, probs)}
        confidence_score = float(round(probabilities.get(pred_label, 0.0) * 100, 2))
    elif hasattr(model, "decision_function"):
        # Linear SVM outputs decision margins, not calibrated probabilities
        dec_scores = model.decision_function(X)[0]
        if hasattr(dec_scores, "__iter__"):
            probabilities = {lbl: float(round(s, 3)) for lbl, s in zip(labels, dec_scores)}
        else:
            probabilities = {"decision_score": float(round(dec_scores, 3))}

    # Explainability text
    interpretation = ""
    if pred_label == "Positive":
        interpretation = "The review expresses user satisfaction, praise for application functionality, or positive service feedback."
    elif pred_label == "Negative":
        interpretation = "The review conveys complaints, service disruption, crash reports, or dissatisfaction with system usability."
    else:
        interpretation = "The review exhibits neutral sentiment, constructive suggestions, or balanced remarks without strong polarity."

    return {
        "predicted_label": pred_label,
        "has_probabilities": has_proba,
        "probabilities": probabilities,
        "confidence_score": confidence_score,
        "text_stats": text_stats,
        "interpretation": interpretation,
        "model_name": bundle.get("model_name", "Trained Classifier")
    }
