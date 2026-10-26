import numpy as np
import pandas as pd

from src.modeling import (
    add_oriented_static_features,
    build_preprocessor,
    probability_metrics,
)


def test_add_oriented_static_features_respects_order_side():
    frame = pd.DataFrame({
        "side": ["buy", "sell"],
        "top_level_imbalance": [0.25, 0.25],
    })

    result = add_oriented_static_features(frame)

    assert result["oriented_top_level_imbalance"].tolist() == [0.25, -0.25]
    assert "oriented_top_level_imbalance" not in frame.columns


def test_preprocessor_handles_missing_and_unseen_categories():
    train = pd.DataFrame({
        "order_size": [100.0, np.nan, 300.0],
        "side": ["buy", "sell", "buy"],
    })
    test = pd.DataFrame({"order_size": [200.0], "side": ["unknown"]})
    preprocessor = build_preprocessor(["order_size", "side"])

    transformed_train = preprocessor.fit_transform(train)
    transformed_test = preprocessor.transform(test)

    assert transformed_train.shape == (3, 3)
    assert transformed_test.shape == (1, 3)
    assert np.isfinite(transformed_train).all()
    assert np.isfinite(transformed_test).all()


def test_probability_metrics_for_perfect_predictions():
    metrics = probability_metrics(
        y_true=np.array([0, 0, 1, 1]),
        probabilities=np.array([0.01, 0.10, 0.90, 0.99]),
        threshold=0.5,
    )

    for name in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"):
        assert metrics[name] == 1.0
    assert 0 <= metrics["brier_score"] < 0.01
