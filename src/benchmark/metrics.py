"""Calculate classification metrics for benchmark experiments."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


SUPPORTED_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "roc_auc",
    "pr_auc",
}


class MetricError(Exception):
    """Raised when metric calculation fails."""


def _as_one_dimensional_array(
    values: pd.Series | np.ndarray | Sequence[int],
    name: str,
) -> np.ndarray:
    """Convert labels to a validated one-dimensional NumPy array."""
    array = np.asarray(values)

    if array.ndim != 1:
        raise MetricError(
            f"{name} must be one-dimensional, "
            f"but received shape {array.shape}."
        )

    if array.size == 0:
        raise MetricError(f"{name} must not be empty.")

    if pd.isna(array).any():
        raise MetricError(
            f"{name} must not contain missing values."
        )

    return array


def _validate_metric_names(metric_names: list[str]) -> None:
    """Validate the requested metric names."""
    if not isinstance(metric_names, list):
        raise MetricError(
            "metric_names must be provided as a list."
        )

    if not metric_names:
        raise MetricError(
            "metric_names must not be empty."
        )

    if not all(isinstance(name, str) for name in metric_names):
        raise MetricError(
            "Every metric name must be a string."
        )

    unknown_metrics = set(metric_names) - SUPPORTED_METRICS

    if unknown_metrics:
        raise MetricError(
            f"Unsupported metrics: {sorted(unknown_metrics)}."
        )

    if len(metric_names) != len(set(metric_names)):
        raise MetricError(
            "metric_names must not contain duplicates."
        )


def _validate_probabilities(
    y_prob: np.ndarray,
    n_samples: int,
) -> int:
    """Validate class probabilities and return the class count."""
    probabilities = np.asarray(y_prob)

    if probabilities.ndim != 2:
        raise MetricError(
            "y_prob must be a two-dimensional probability matrix, "
            f"but received shape {probabilities.shape}."
        )

    if probabilities.shape[0] != n_samples:
        raise MetricError(
            "y_prob and y_true must contain the same number "
            "of samples."
        )

    n_classes = probabilities.shape[1]

    if n_classes < 2:
        raise MetricError(
            "y_prob must contain probabilities for at least "
            "two classes."
        )

    if not np.isfinite(probabilities).all():
        raise MetricError(
            "y_prob must contain only finite values."
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise MetricError(
            "y_prob values must lie between 0 and 1."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
    ):
        raise MetricError(
            "Each row of y_prob must sum to 1."
        )

    return n_classes


def _validate_labels(
    labels: np.ndarray,
    name: str,
    n_classes: int,
) -> None:
    """Ensure that labels are valid integer class identifiers."""
    if not np.issubdtype(labels.dtype, np.integer):
        raise MetricError(
            f"{name} must contain integer class labels."
        )

    invalid_labels = set(labels.tolist()) - set(range(n_classes))

    if invalid_labels:
        raise MetricError(
            f"{name} contains labels outside the valid range "
            f"0 to {n_classes - 1}: {sorted(invalid_labels)}."
        )


def _calculate_roc_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int,
) -> float:
    """Calculate binary or macro one-vs-rest ROC-AUC."""
    if n_classes == 2:
        return float(
            roc_auc_score(
                y_true,
                y_prob[:, 1],
            )
        )

    return float(
        roc_auc_score(
            y_true,
            y_prob,
            labels=np.arange(n_classes),
            multi_class="ovr",
            average="macro",
        )
    )


def _calculate_pr_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int,
) -> float:
    """Calculate binary or macro one-vs-rest average precision."""
    if n_classes == 2:
        return float(
            average_precision_score(
                y_true,
                y_prob[:, 1],
            )
        )

    y_true_binary = label_binarize(
        y_true,
        classes=np.arange(n_classes),
    )

    return float(
        average_precision_score(
            y_true_binary,
            y_prob,
            average="macro",
        )
    )


def calculate_metrics(
    y_true: pd.Series | np.ndarray | Sequence[int],
    y_pred: pd.Series | np.ndarray | Sequence[int],
    y_prob: np.ndarray,
    metric_names: list[str],
) -> dict[str, float]:
    """Calculate requested binary or multiclass metrics.

    Parameters
    ----------
    y_true
        True class labels.
    y_pred
        Predicted class labels.
    y_prob
        Class probabilities with shape
        ``(n_samples, n_classes)``.
    metric_names
        Metrics requested by the experiment configuration.

    Returns
    -------
    dict[str, float]
        Requested metric names and calculated values.

    Raises
    ------
    MetricError
        If inputs are invalid or a metric cannot be calculated.
    """
    _validate_metric_names(metric_names)

    true_labels = _as_one_dimensional_array(
        y_true,
        name="y_true",
    )
    predicted_labels = _as_one_dimensional_array(
        y_pred,
        name="y_pred",
    )

    if len(true_labels) != len(predicted_labels):
        raise MetricError(
            "y_true and y_pred must contain the same number "
            "of samples."
        )

    probabilities = np.asarray(y_prob, dtype=float)

    n_classes = _validate_probabilities(
        y_prob=probabilities,
        n_samples=len(true_labels),
    )

    _validate_labels(
        labels=true_labels,
        name="y_true",
        n_classes=n_classes,
    )
    _validate_labels(
        labels=predicted_labels,
        name="y_pred",
        n_classes=n_classes,
    )

    results: dict[str, float] = {}

    try:
        for metric_name in metric_names:
            if metric_name == "accuracy":
                value = accuracy_score(
                    true_labels,
                    predicted_labels,
                )

            elif metric_name == "balanced_accuracy":
                value = balanced_accuracy_score(
                    true_labels,
                    predicted_labels,
                )

            elif metric_name == "f1_macro":
                value = f1_score(
                    true_labels,
                    predicted_labels,
                    average="macro",
                    zero_division=0,
                )

            elif metric_name == "f1_weighted":
                value = f1_score(
                    true_labels,
                    predicted_labels,
                    average="weighted",
                    zero_division=0,
                )

            elif metric_name == "roc_auc":
                value = _calculate_roc_auc(
                    y_true=true_labels,
                    y_prob=probabilities,
                    n_classes=n_classes,
                )

            elif metric_name == "pr_auc":
                value = _calculate_pr_auc(
                    y_true=true_labels,
                    y_prob=probabilities,
                    n_classes=n_classes,
                )

            results[metric_name] = float(value)

    except ValueError as error:
        raise MetricError(
            f"Could not calculate metrics: {error}"
        ) from error

    return results