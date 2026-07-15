"""Select feature subsets from importance rankings."""

import math

import numpy as np
import pandas as pd


class SelectionError(Exception):
    """Raised when feature-subset selection is invalid."""


def percentage_to_k(
    percentage: float,
    n_features: int,
) -> int:
    """Convert a feature percentage into a number of selected features.

    The result is rounded upward so that the selected subset never contains
    less than the requested percentage. At least one feature is selected.

    Parameters
    ----------
    percentage
        Requested percentage of the total features.
    n_features
        Total number of available features.

    Returns
    -------
    int
        Number of features to select.

    Raises
    ------
    SelectionError
        If the percentage or number of features is invalid.
    """
    if isinstance(percentage, bool) or not isinstance(
        percentage,
        (int, float),
    ):
        raise SelectionError(
            "percentage must be a number."
        )

    if not 0 < percentage <= 100:
        raise SelectionError(
            "percentage must be greater than 0 and no greater than 100."
        )

    if isinstance(n_features, bool) or not isinstance(n_features, int):
        raise SelectionError(
            "n_features must be an integer."
        )

    if n_features < 1:
        raise SelectionError(
            "n_features must be at least 1."
        )

    return min(
        n_features,
        max(
            1,
            math.ceil(percentage * n_features / 100),
        ),
    )


def get_unique_feature_counts(
    percentages: list[float],
    n_features: int,
) -> list[dict[str, float | int]]:
    """Resolve percentages into unique feature-subset sizes.

    Multiple percentages may produce the same value of k for datasets with
    few features. Only the first occurrence of each k is retained.

    Parameters
    ----------
    percentages
        Requested feature percentages.
    n_features
        Total number of available features.

    Returns
    -------
    list[dict[str, float | int]]
        Unique subset sizes, including the requested percentage, number of
        selected features, and actual percentage after rounding.

    Raises
    ------
    SelectionError
        If the percentage list is empty or invalid.
    """
    if not isinstance(percentages, list):
        raise SelectionError(
            "percentages must be provided as a list."
        )

    if not percentages:
        raise SelectionError(
            "percentages must not be empty."
        )

    subset_sizes: list[dict[str, float | int]] = []
    seen_counts: set[int] = set()

    for percentage in percentages:
        k = percentage_to_k(
            percentage=percentage,
            n_features=n_features,
        )

        if k in seen_counts:
            continue

        seen_counts.add(k)

        subset_sizes.append(
            {
                "requested_percentage": float(percentage),
                "n_selected_features": k,
                "actual_percentage": 100.0 * k / n_features,
            }
        )

    return subset_sizes


def select_top_k(
    ranking: pd.Series,
    k: int,
) -> list[str]:
    """Select the top-k feature names from an importance ranking.

    Parameters
    ----------
    ranking
        Importance scores indexed by feature name and sorted from highest
        to lowest importance.
    k
        Number of features to select.

    Returns
    -------
    list[str]
        Names of the selected features in ranking order.

    Raises
    ------
    SelectionError
        If the ranking or requested subset size is invalid.
    """
    if not isinstance(ranking, pd.Series):
        raise SelectionError(
            "ranking must be a pandas Series."
        )

    if ranking.empty:
        raise SelectionError(
            "ranking must not be empty."
        )

    if not ranking.index.is_unique:
        raise SelectionError(
            "ranking must contain unique feature names."
        )

    if ranking.isna().any():
        raise SelectionError(
            "ranking must not contain missing scores."
        )

    if not ranking.is_monotonic_decreasing:
        raise SelectionError(
            "ranking must be sorted in descending order."
        )

    if isinstance(k, bool) or not isinstance(k, int):
        raise SelectionError(
            "k must be an integer."
        )

    if not 1 <= k <= len(ranking):
        raise SelectionError(
            f"k must be between 1 and {len(ranking)}."
        )

    return ranking.head(k).index.astype(str).tolist()


def select_random_features(
    feature_names: list[str],
    k: int,
    random_seed: int,
) -> list[str]:
    """Select a reproducible random subset of feature names.

    The returned features follow their original column order, which keeps
    model inputs deterministic while the subset itself remains random.

    Parameters
    ----------
    feature_names
        Complete ordered list of feature names.
    k
        Number of features to select.
    random_seed
        Seed controlling random feature selection.

    Returns
    -------
    list[str]
        Randomly selected feature names in their original column order.

    Raises
    ------
    SelectionError
        If the feature names, subset size, or seed are invalid.
    """
    if not isinstance(feature_names, list):
        raise SelectionError(
            "feature_names must be provided as a list."
        )

    if not feature_names:
        raise SelectionError(
            "feature_names must not be empty."
        )

    if len(feature_names) != len(set(feature_names)):
        raise SelectionError(
            "feature_names must be unique."
        )

    if not all(
        isinstance(feature_name, str) and feature_name
        for feature_name in feature_names
    ):
        raise SelectionError(
            "Every feature name must be a non-empty string."
        )

    if isinstance(k, bool) or not isinstance(k, int):
        raise SelectionError(
            "k must be an integer."
        )

    if not 1 <= k <= len(feature_names):
        raise SelectionError(
            f"k must be between 1 and {len(feature_names)}."
        )

    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise SelectionError(
            "random_seed must be an integer."
        )

    generator = np.random.default_rng(random_seed)

    selected_indices = generator.choice(
        len(feature_names),
        size=k,
        replace=False,
    )

    selected_indices.sort()

    return [
        feature_names[index]
        for index in selected_indices
    ]