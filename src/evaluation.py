"""Reusable helpers for the frozen-policy final test evaluation."""

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from modeling import FEATURE_SETS, build_preprocessor


def build_estimator(feature_set_name, model_name, parameters, random_seed=42):
    """Build one of the model families selected in the validation notebook."""
    features = FEATURE_SETS[feature_set_name]
    if model_name == "logistic_regression":
        model = LogisticRegression(
            C=parameters["C"],
            max_iter=3000,
            solver="lbfgs",
            random_state=random_seed,
        )
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=250,
            max_depth=parameters["max_depth"],
            min_samples_leaf=parameters["min_samples_leaf"],
            max_features=parameters["max_features"],
            n_jobs=-1,
            random_state=random_seed,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline([
        ("preprocessing", build_preprocessor(features)),
        ("model", model),
    ])


def clipped_logit(probability):
    clipped = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def expected_calibration_error(y_true, probability, bins=10):
    y_true = np.asarray(y_true)
    probability = np.asarray(probability)
    edges = np.linspace(0, 1, bins + 1)
    assignments = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for bin_number in range(bins):
        mask = assignments == bin_number
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - probability[mask].mean())
    return float(ece)


def fit_sigmoid_calibrator(raw_probability, y_true):
    calibrator = LogisticRegression(C=1_000_000, solver="lbfgs", max_iter=2000)
    calibrator.fit(clipped_logit(raw_probability), np.asarray(y_true))
    return calibrator


def apply_sigmoid_calibrator(calibrator, raw_probability):
    return calibrator.predict_proba(clipped_logit(raw_probability))[:, 1]


def safe_probability_metrics(y_true, probability, threshold):
    """Return discrimination, calibration, and threshold metrics for one slice."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predictions = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    has_both_classes = np.unique(y_true).size == 2

    return {
        "eligible_rows": int(len(y_true)),
        "true_fill_rate": float(y_true.mean()),
        "predicted_fill_rate": float(predictions.mean()),
        "mean_predicted_probability": float(probability.mean()),
        "roc_auc": float(roc_auc_score(y_true, probability)) if has_both_classes else np.nan,
        "pr_auc": float(average_precision_score(y_true, probability)) if y_true.sum() else np.nan,
        "brier_score": float(brier_score_loss(y_true, probability)),
        "ece_10": expected_calibration_error(y_true, probability, bins=10),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_metric_intervals(
    y_true,
    probability,
    threshold,
    random_seed,
    resamples=1000,
    confidence=0.95,
):
    """Ordinary row bootstrap intervals for final-test overall metrics."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    rng = np.random.default_rng(random_seed)
    metric_names = (
        "roc_auc",
        "pr_auc",
        "brier_score",
        "ece_10",
        "accuracy",
        "precision",
        "recall",
        "f1",
    )
    samples = {metric: [] for metric in metric_names}

    for _ in range(resamples):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        metrics = safe_probability_metrics(
            y_true[indices],
            probability[indices],
            threshold,
        )
        for metric in metric_names:
            if np.isfinite(metrics[metric]):
                samples[metric].append(metrics[metric])

    tail = (1 - confidence) / 2
    intervals = {}
    for metric in metric_names:
        values = np.asarray(samples[metric], dtype=float)
        intervals[metric] = {
            "ci_low": float(np.quantile(values, tail)),
            "ci_high": float(np.quantile(values, 1 - tail)),
        }
    return intervals
