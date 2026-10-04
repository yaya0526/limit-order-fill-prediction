"""Helpers for the project-brief model and explanation requirements."""

from itertools import combinations

import numpy as np
import pandas as pd

from lime.lime_tabular import LimeTabularExplainer

from modeling import A_CATEGORICAL, C_ADDITIONAL_CATEGORICAL


CATEGORICAL_FEATURES = A_CATEGORICAL + C_ADDITIONAL_CATEGORICAL


def transformed_feature_mapping(preprocessor):
    """Map preprocessor output columns back to their original input variables."""
    rows = []
    for index, transformed_name in enumerate(preprocessor.get_feature_names_out()):
        if transformed_name.startswith("numeric__"):
            original_feature = transformed_name.removeprefix("numeric__")
            category_level = None
        elif transformed_name.startswith("categorical__"):
            encoded = transformed_name.removeprefix("categorical__")
            original_feature = encoded
            category_level = None
            for categorical_feature in CATEGORICAL_FEATURES:
                prefix = f"{categorical_feature}_"
                if encoded.startswith(prefix):
                    original_feature = categorical_feature
                    category_level = encoded.removeprefix(prefix)
                    break
        else:
            original_feature = transformed_name
            category_level = None

        rows.append({
            "transformed_index": index,
            "transformed_feature": transformed_name,
            "original_feature": original_feature,
            "category_level": category_level,
        })
    return pd.DataFrame(rows)


def aggregate_transformed_values(mapping, values, value_name, absolute=False):
    """Aggregate one value per transformed column back to original variables."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != len(mapping):
        raise ValueError("Expected one value per transformed feature.")
    result = mapping.copy()
    result[value_name] = np.abs(values) if absolute else values
    return (
        result.groupby("original_feature", as_index=False)[value_name]
        .sum()
        .sort_values(value_name, ascending=False)
    )


def positive_class_shap_values(explanation):
    """Normalize current SHAP binary-classifier output to rows x features."""
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if base_values.ndim == 2:
        base_values = base_values[:, 1]
    return values, base_values


def random_forest_impurity_table(estimator):
    preprocessing = estimator.named_steps["preprocessing"]
    model = estimator.named_steps["model"]
    mapping = transformed_feature_mapping(preprocessing)
    return aggregate_transformed_values(
        mapping,
        model.feature_importances_,
        "impurity_importance",
    )


def tree_path_conditions(estimator, row):
    """Return ordered conditions traversed by one row in a fitted tree pipeline."""
    preprocessing = estimator.named_steps["preprocessing"]
    model = estimator.named_steps["model"]
    transformed_names = preprocessing.get_feature_names_out()
    feature_mapping = transformed_feature_mapping(preprocessing).set_index(
        "transformed_index"
    )
    numeric_pipeline = preprocessing.named_transformers_["numeric"]
    numeric_features = list(preprocessing.transformers_[0][2])
    scaler = numeric_pipeline.named_steps["scaler"]
    numeric_scaling = {
        feature: (float(mean), float(scale))
        for feature, mean, scale in zip(
            numeric_features,
            scaler.mean_,
            scaler.scale_,
        )
    }
    transformed = preprocessing.transform(row)
    node_indicator = model.decision_path(transformed)
    leaf_id = int(model.apply(transformed)[0])
    nodes = node_indicator.indices[
        node_indicator.indptr[0]:node_indicator.indptr[1]
    ]

    conditions = []
    for depth, node_id in enumerate(nodes):
        if node_id == leaf_id:
            continue
        feature_index = model.tree_.feature[node_id]
        threshold = model.tree_.threshold[node_id]
        observed = float(transformed[0, feature_index])
        operator = "<=" if observed <= threshold else ">"
        map_row = feature_mapping.loc[feature_index]
        original_feature = map_row["original_feature"]
        category_level = map_row["category_level"]
        if pd.isna(category_level):
            mean, scale = numeric_scaling[original_feature]
            raw_threshold = float(threshold * scale + mean)
            raw_observed = float(observed * scale + mean)
            market_condition = (
                f"{original_feature} {operator} {raw_threshold:.4g}"
            )
        else:
            raw_threshold = np.nan
            raw_observed = np.nan
            market_condition = (
                f"{original_feature} is {category_level}"
                if operator == ">"
                else f"{original_feature} is not {category_level}"
            )
        conditions.append({
            "depth": depth,
            "node_id": int(node_id),
            "transformed_feature": transformed_names[feature_index],
            "observed_transformed_value": observed,
            "operator": operator,
            "threshold": float(threshold),
            "original_feature": original_feature,
            "category_level": category_level,
            "raw_observed_value": raw_observed,
            "raw_threshold": raw_threshold,
            "condition": (
                f"{transformed_names[feature_index]} {operator} {threshold:.4f}"
            ),
            "market_condition": market_condition,
        })
    leaf_probability = float(model.predict_proba(transformed)[0, 1])
    return conditions, leaf_id, leaf_probability


def lime_explanation_table(
    model,
    training_transformed,
    instance_transformed,
    transformed_feature_names,
    random_seed,
    num_features=12,
    num_samples=3000,
):
    """Explain one transformed observation and aggregate weights to raw variables."""
    explainer = LimeTabularExplainer(
        np.asarray(training_transformed),
        feature_names=list(transformed_feature_names),
        class_names=["not_fill", "fill"],
        mode="classification",
        discretize_continuous=True,
        random_state=random_seed,
    )
    explanation = explainer.explain_instance(
        np.asarray(instance_transformed).reshape(-1),
        model.predict_proba,
        labels=(1,),
        num_features=num_features,
        num_samples=num_samples,
    )
    weight_map = explanation.as_map()[1]
    mapping = []
    transformed_to_original = transformed_feature_mapping_from_names(
        transformed_feature_names
    )
    for rank, (feature_index, weight) in enumerate(weight_map, start=1):
        map_row = transformed_to_original.iloc[int(feature_index)]
        mapping.append({
            "rank": rank,
            "transformed_index": int(feature_index),
            "transformed_feature": transformed_feature_names[int(feature_index)],
            "original_feature": map_row["original_feature"],
            "lime_weight": float(weight),
        })
    return pd.DataFrame(mapping)


def transformed_feature_mapping_from_names(transformed_feature_names):
    rows = []
    for index, transformed_name in enumerate(transformed_feature_names):
        if transformed_name.startswith("numeric__"):
            original_feature = transformed_name.removeprefix("numeric__")
            category_level = None
        elif transformed_name.startswith("categorical__"):
            encoded = transformed_name.removeprefix("categorical__")
            original_feature = encoded
            category_level = None
            for categorical_feature in CATEGORICAL_FEATURES:
                prefix = f"{categorical_feature}_"
                if encoded.startswith(prefix):
                    original_feature = categorical_feature
                    category_level = encoded.removeprefix(prefix)
                    break
        else:
            original_feature = transformed_name
            category_level = None
        rows.append({
            "transformed_index": index,
            "transformed_feature": transformed_name,
            "original_feature": original_feature,
            "category_level": category_level,
        })
    return pd.DataFrame(rows)


def lime_seed_stability(explanations, top_n=10):
    """Summarize top-feature overlap and sign agreement across LIME seeds."""
    seeds = sorted(explanations["random_seed"].unique())
    top_sets = {}
    sign_maps = {}
    for seed in seeds:
        seed_rows = explanations.loc[
            explanations["random_seed"].eq(seed)
        ].copy()
        aggregated = (
            seed_rows.groupby("original_feature", as_index=False)["lime_weight"]
            .sum()
        )
        aggregated["abs_weight"] = aggregated["lime_weight"].abs()
        aggregated = aggregated.sort_values("abs_weight", ascending=False).head(top_n)
        top_sets[seed] = set(aggregated["original_feature"])
        sign_maps[seed] = {
            row.original_feature: int(np.sign(row.lime_weight))
            for row in aggregated.itertuples(index=False)
        }

    jaccards = []
    for seed_a, seed_b in combinations(seeds, 2):
        union = top_sets[seed_a] | top_sets[seed_b]
        overlap = top_sets[seed_a] & top_sets[seed_b]
        jaccards.append(len(overlap) / len(union) if union else 1.0)

    all_features = sorted(set().union(*top_sets.values()))
    sign_agreements = []
    for feature in all_features:
        signs = [sign_maps[seed][feature] for seed in seeds if feature in sign_maps[seed]]
        if len(signs) >= 2:
            majority_share = max(signs.count(-1), signs.count(1)) / len(signs)
            sign_agreements.append(majority_share)

    return {
        "mean_pairwise_top_feature_jaccard": float(np.mean(jaccards)),
        "min_pairwise_top_feature_jaccard": float(np.min(jaccards)),
        "mean_sign_agreement": float(np.mean(sign_agreements)) if sign_agreements else np.nan,
        "seeds": len(seeds),
        "top_n": int(top_n),
    }
