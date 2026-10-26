"""Reusable, model-agnostic explanation helpers for frozen fill models."""

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

try:  # Package import used by tests and external callers.
    from .history_features import MODEL_HISTORY_FEATURES
    from .modeling import (
        A_CATEGORICAL,
        A_NUMERIC,
        B_ADDITIONAL_NUMERIC,
        C_ADDITIONAL_CATEGORICAL,
        C_ADDITIONAL_NUMERIC,
    )
except ImportError:  # Notebook compatibility when src/ is added to sys.path.
    from history_features import MODEL_HISTORY_FEATURES
    from modeling import (
        A_CATEGORICAL,
        A_NUMERIC,
        B_ADDITIONAL_NUMERIC,
        C_ADDITIONAL_CATEGORICAL,
        C_ADDITIONAL_NUMERIC,
    )


FEATURE_GROUPS = {
    "order_state": A_NUMERIC + A_CATEGORICAL,
    "lob_snapshot": B_ADDITIONAL_NUMERIC,
    "recent_flow_activity": C_ADDITIONAL_NUMERIC + C_ADDITIONAL_CATEGORICAL,
    "history_path": MODEL_HISTORY_FEATURES,
}


def feature_group(feature):
    for group_name, group_features in FEATURE_GROUPS.items():
        if feature in group_features:
            return group_name
    return "other"


def logistic_coefficient_table(estimator):
    """Return interpretable transformed-feature coefficients from a fitted pipeline."""
    preprocessing = estimator.named_steps["preprocessing"]
    model = estimator.named_steps["model"]
    transformed_names = preprocessing.get_feature_names_out()
    coefficients = model.coef_[0]

    rows = []
    for transformed_name, coefficient in zip(transformed_names, coefficients):
        if transformed_name.startswith("numeric__"):
            original_feature = transformed_name.removeprefix("numeric__")
            display_feature = original_feature
            feature_type = "numeric_standardized"
            category_level = None
        elif transformed_name.startswith("categorical__"):
            encoded_name = transformed_name.removeprefix("categorical__")
            original_feature = None
            category_level = None
            for categorical_feature in A_CATEGORICAL + C_ADDITIONAL_CATEGORICAL:
                prefix = f"{categorical_feature}_"
                if encoded_name.startswith(prefix):
                    original_feature = categorical_feature
                    category_level = encoded_name.removeprefix(prefix)
                    break
            if original_feature is None:
                original_feature = encoded_name
                display_feature = encoded_name
            else:
                display_feature = f"{original_feature}={category_level}"
            feature_type = "categorical_one_hot"
        else:
            original_feature = transformed_name
            display_feature = transformed_name
            feature_type = "other"
            category_level = None

        rows.append({
            "transformed_feature": transformed_name,
            "display_feature": display_feature,
            "original_feature": original_feature,
            "category_level": category_level,
            "feature_type": feature_type,
            "feature_group": feature_group(original_feature),
            "coefficient": float(coefficient),
            "abs_coefficient": float(abs(coefficient)),
            "odds_ratio": float(np.exp(coefficient)),
            "direction": "higher_fill_probability" if coefficient > 0 else "lower_fill_probability",
        })

    return pd.DataFrame(rows).sort_values("abs_coefficient", ascending=False)


def logistic_categorical_contrasts(coefficient_table, reference_levels):
    """Convert full one-hot coefficients into interpretable within-feature contrasts."""
    categorical = coefficient_table.loc[
        coefficient_table["feature_type"].eq("categorical_one_hot")
    ].copy()
    rows = []
    for original_feature, reference_level in reference_levels.items():
        feature_rows = categorical.loc[
            categorical["original_feature"].eq(original_feature)
        ]
        reference_match = feature_rows.loc[
            feature_rows["category_level"].eq(reference_level)
        ]
        if reference_match.empty:
            raise ValueError(
                f"Reference level {reference_level!r} not found for {original_feature!r}."
            )
        reference_coefficient = float(reference_match.iloc[0]["coefficient"])
        for row in feature_rows.to_dict("records"):
            if row["category_level"] == reference_level:
                continue
            contrast = float(row["coefficient"] - reference_coefficient)
            rows.append({
                "original_feature": original_feature,
                "comparison_level": row["category_level"],
                "reference_level": reference_level,
                "contrast_label": (
                    f"{original_feature}: {row['category_level']} vs {reference_level}"
                ),
                "log_odds_contrast": contrast,
                "odds_ratio_contrast": float(np.exp(contrast)),
                "direction": (
                    "higher_fill_probability" if contrast > 0 else "lower_fill_probability"
                ),
            })
    return pd.DataFrame(rows).sort_values("log_odds_contrast", ascending=False)


def individual_permutation_importance(
    estimator,
    frame,
    y_true,
    features,
    random_seed=42,
    repeats=20,
):
    """Feature-level validation PR-AUC decrease after independent permutation."""
    result = permutation_importance(
        estimator,
        frame[features],
        np.asarray(y_true),
        scoring="average_precision",
        n_repeats=repeats,
        random_state=random_seed,
        n_jobs=-1,
    )
    rows = []
    for feature, values in zip(features, result.importances):
        rows.append({
            "feature": feature,
            "feature_group": feature_group(feature),
            "pr_auc_decrease_mean": float(values.mean()),
            "pr_auc_decrease_std": float(values.std(ddof=1)),
            "pr_auc_decrease_ci_low": float(np.quantile(values, 0.025)),
            "pr_auc_decrease_ci_high": float(np.quantile(values, 0.975)),
            "repeats": int(repeats),
        })
    return pd.DataFrame(rows).sort_values("pr_auc_decrease_mean", ascending=False)


def grouped_permutation_importance(
    estimator,
    frame,
    y_true,
    features,
    random_seed=42,
    repeats=50,
):
    """Jointly permute related columns and measure validation PR-AUC decrease."""
    feature_frame = frame[features].copy()
    y_true = np.asarray(y_true)
    baseline_probability = estimator.predict_proba(feature_frame)[:, 1]
    baseline_pr_auc = average_precision_score(y_true, baseline_probability)
    rng = np.random.default_rng(random_seed)
    rows = []

    for group_name, group_features in FEATURE_GROUPS.items():
        selected = [feature for feature in group_features if feature in features]
        if not selected:
            continue
        decreases = []
        for _ in range(repeats):
            indices = rng.permutation(len(feature_frame))
            permuted = feature_frame.copy()
            permuted.loc[:, selected] = feature_frame.iloc[indices][selected].to_numpy()
            probability = estimator.predict_proba(permuted)[:, 1]
            decreases.append(
                baseline_pr_auc - average_precision_score(y_true, probability)
            )

        values = np.asarray(decreases)
        rows.append({
            "feature_group": group_name,
            "feature_count": len(selected),
            "baseline_pr_auc": float(baseline_pr_auc),
            "pr_auc_decrease_mean": float(values.mean()),
            "pr_auc_decrease_std": float(values.std(ddof=1)),
            "pr_auc_decrease_ci_low": float(np.quantile(values, 0.025)),
            "pr_auc_decrease_ci_high": float(np.quantile(values, 0.975)),
            "repeats": int(repeats),
        })

    return pd.DataFrame(rows).sort_values("pr_auc_decrease_mean", ascending=False)
