"""Tests for benchmark classification metrics."""

import numpy as np
import pytest

from benchmark.metrics import MetricError, calculate_metrics


def test_calculate_binary_metrics() -> None:
    """Calculate all supported metrics for binary classification."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_prob = np.array(
        [
            [0.90, 0.10],
            [0.70, 0.30],
            [0.20, 0.80],
            [0.40, 0.60],
            [0.10, 0.90],
        ]
    )
    y_pred = y_prob.argmax(axis=1)

    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        metric_names=[
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "f1_weighted",
            "roc_auc",
            "pr_auc",
        ],
    )

    assert list(metrics) == [
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "roc_auc",
        "pr_auc",
    ]

    for value in metrics.values():
        assert value == pytest.approx(1.0)


def test_calculate_multiclass_metrics() -> None:
    """Calculate all supported metrics for multiclass classification."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_prob = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.05, 0.10, 0.85],
            [0.70, 0.20, 0.10],
            [0.10, 0.75, 0.15],
            [0.10, 0.20, 0.70],
        ]
    )
    y_pred = y_prob.argmax(axis=1)

    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        metric_names=[
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "f1_weighted",
            "roc_auc",
            "pr_auc",
        ],
    )

    for value in metrics.values():
        assert value == pytest.approx(1.0)


def test_calculate_metrics_returns_only_requested_metrics() -> None:
    """Return only the metrics requested by the caller."""
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array(
        [
            [0.80, 0.20],
            [0.30, 0.70],
            [0.20, 0.80],
            [0.75, 0.25],
        ]
    )
    y_pred = y_prob.argmax(axis=1)

    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        metric_names=["accuracy", "f1_macro"],
    )

    assert set(metrics) == {"accuracy", "f1_macro"}


@pytest.mark.parametrize(
    "metric_names",
    [
        [],
        ["unknown_metric"],
        ["accuracy", "accuracy"],
        ["accuracy", 1],
    ],
)
def test_calculate_metrics_rejects_invalid_metric_names(
    metric_names: list[object],
) -> None:
    """Reject empty, unsupported, duplicated, or non-string names."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    y_prob = np.array(
        [
            [0.90, 0.10],
            [0.10, 0.90],
        ]
    )

    with pytest.raises(MetricError):
        calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            metric_names=metric_names,  # type: ignore[arg-type]
        )


def test_calculate_metrics_rejects_length_mismatch() -> None:
    """Reject true and predicted labels with different lengths."""
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 1])
    y_prob = np.array(
        [
            [0.90, 0.10],
            [0.10, 0.90],
            [0.20, 0.80],
        ]
    )

    with pytest.raises(
        MetricError,
        match="same number of samples",
    ):
        calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            metric_names=["accuracy"],
        )


def test_calculate_metrics_rejects_invalid_probability_shape() -> None:
    """Require a two-dimensional probability matrix."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    y_prob = np.array([0.10, 0.90])

    with pytest.raises(
        MetricError,
        match="two-dimensional probability matrix",
    ):
        calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            metric_names=["accuracy"],
        )


def test_calculate_metrics_rejects_probability_length_mismatch() -> None:
    """Require probabilities for every sample."""
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 1, 1])
    y_prob = np.array(
        [
            [0.90, 0.10],
            [0.10, 0.90],
        ]
    )

    with pytest.raises(
        MetricError,
        match="y_prob and y_true must contain the same number",
    ):
        calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            metric_names=["accuracy"],
        )


@pytest.mark.parametrize(
    "y_prob",
    [
        np.array(
            [
                [1.20, -0.20],
                [0.10, 0.90],
            ]
        ),
        np.array(
            [
                [0.60, 0.60],
                [0.10, 0.90],
            ]
        ),
        np.array(
            [
                [np.nan, np.nan],
                [0.10, 0.90],
            ]
        ),
    ],
)
def test_calculate_metrics_rejects_invalid_probabilities(
    y_prob: np.ndarray,
) -> None:
    """Reject invalid probability values or row sums."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])

    with pytest.raises(MetricError):
        calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            metric_names=["accuracy"],
        )


def test_calculate_metrics_rejects_non_integer_labels() -> None:
    """Require integer class labels."""
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0, 1])
    y_prob = np.array(
        [
            [0.90, 0.10],
            [0.10, 0.90],
        ]
    )

    with pytest.raises(
        MetricError,
        match="y_true must contain integer class labels",
    ):
        calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            metric_names=["accuracy"],
        )


def test_calculate_metrics_rejects_out_of_range_labels() -> None:
    """Reject labels outside the probability-column range."""
    y_true = np.array([0, 2])
    y_pred = np.array([0, 1])
    y_prob = np.array(
        [
            [0.90, 0.10],
            [0.10, 0.90],
        ]
    )

    with pytest.raises(
        MetricError,
        match="outside the valid range",
    ):
        calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            metric_names=["accuracy"],
        )


def test_calculate_metrics_handles_zero_division_in_f1() -> None:
    """Return finite F1 values when a class is never predicted."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    y_prob = np.array(
        [
            [0.90, 0.10],
            [0.80, 0.20],
            [0.70, 0.30],
            [0.60, 0.40],
        ]
    )

    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        metric_names=[
            "f1_macro",
            "f1_weighted",
        ],
    )

    assert np.isfinite(metrics["f1_macro"])
    assert np.isfinite(metrics["f1_weighted"])