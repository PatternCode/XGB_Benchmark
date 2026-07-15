"""Tests for feature-subset selection."""

import pandas as pd
import pytest

from benchmark.selection import (
    SelectionError,
    get_unique_feature_counts,
    percentage_to_k,
    select_random_features,
    select_top_k,
)


@pytest.fixture
def ranking() -> pd.Series:
    """Return a valid descending feature-importance ranking."""
    return pd.Series(
        [0.9, 0.7, 0.4, 0.1],
        index=[
            "feature_a",
            "feature_b",
            "feature_c",
            "feature_d",
        ],
        name="gain",
    )


@pytest.mark.parametrize(
    ("percentage", "n_features", "expected_k"),
    [
        (10, 30, 3),
        (2.5, 30, 1),
        (5, 16, 1),
        (10, 16, 2),
        (30, 16, 5),
        (100, 16, 16),
    ],
)
def test_percentage_to_k_returns_expected_count(
    percentage: float,
    n_features: int,
    expected_k: int,
) -> None:
    """Convert percentages to upward-rounded feature counts."""
    assert percentage_to_k(percentage, n_features) == expected_k


@pytest.mark.parametrize(
    "percentage",
    [
        0,
        -1,
        101,
        "10",
        True,
    ],
)
def test_percentage_to_k_rejects_invalid_percentage(
    percentage: object,
) -> None:
    """Reject invalid feature percentages."""
    with pytest.raises(SelectionError):
        percentage_to_k(
            percentage=percentage,  # type: ignore[arg-type]
            n_features=20,
        )


@pytest.mark.parametrize(
    "n_features",
    [
        0,
        -1,
        2.5,
        "20",
        True,
    ],
)
def test_percentage_to_k_rejects_invalid_feature_count(
    n_features: object,
) -> None:
    """Reject invalid total feature counts."""
    with pytest.raises(SelectionError):
        percentage_to_k(
            percentage=10,
            n_features=n_features,  # type: ignore[arg-type]
        )


def test_get_unique_feature_counts_removes_duplicate_k_values() -> None:
    """Retain only the first percentage producing each feature count."""
    subset_sizes = get_unique_feature_counts(
        percentages=[2.5, 5, 10, 30],
        n_features=16,
    )

    assert subset_sizes == [
        {
            "requested_percentage": 2.5,
            "n_selected_features": 1,
            "actual_percentage": 6.25,
        },
        {
            "requested_percentage": 10.0,
            "n_selected_features": 2,
            "actual_percentage": 12.5,
        },
        {
            "requested_percentage": 30.0,
            "n_selected_features": 5,
            "actual_percentage": 31.25,
        },
    ]


def test_get_unique_feature_counts_rejects_empty_list() -> None:
    """Require at least one requested percentage."""
    with pytest.raises(
        SelectionError,
        match="percentages must not be empty",
    ):
        get_unique_feature_counts(
            percentages=[],
            n_features=10,
        )


def test_select_top_k_returns_features_in_ranking_order(
    ranking: pd.Series,
) -> None:
    """Select the highest-ranked features in order."""
    selected_features = select_top_k(
        ranking=ranking,
        k=2,
    )

    assert selected_features == [
        "feature_a",
        "feature_b",
    ]


@pytest.mark.parametrize(
    "k",
    [
        0,
        -1,
        5,
        1.5,
        True,
    ],
)
def test_select_top_k_rejects_invalid_k(
    ranking: pd.Series,
    k: object,
) -> None:
    """Reject invalid requested subset sizes."""
    with pytest.raises(SelectionError):
        select_top_k(
            ranking=ranking,
            k=k,  # type: ignore[arg-type]
        )


def test_select_top_k_rejects_unsorted_ranking() -> None:
    """Require rankings to be sorted in descending order."""
    unsorted_ranking = pd.Series(
        [0.4, 0.9, 0.1],
        index=["feature_a", "feature_b", "feature_c"],
    )

    with pytest.raises(
        SelectionError,
        match="sorted in descending order",
    ):
        select_top_k(
            ranking=unsorted_ranking,
            k=2,
        )


def test_select_top_k_rejects_missing_scores() -> None:
    """Reject rankings containing missing importance scores."""
    invalid_ranking = pd.Series(
        [0.9, None, 0.1],
        index=["feature_a", "feature_b", "feature_c"],
    )

    with pytest.raises(
        SelectionError,
        match="must not contain missing scores",
    ):
        select_top_k(
            ranking=invalid_ranking,
            k=2,
        )


def test_select_top_k_rejects_duplicate_feature_names() -> None:
    """Require feature names to be unique."""
    invalid_ranking = pd.Series(
        [0.9, 0.7, 0.4],
        index=["feature_a", "feature_a", "feature_c"],
    )

    with pytest.raises(
        SelectionError,
        match="unique feature names",
    ):
        select_top_k(
            ranking=invalid_ranking,
            k=2,
        )


def test_select_random_features_is_reproducible() -> None:
    """Produce the same random subset when the seed is unchanged."""
    feature_names = [
        "feature_a",
        "feature_b",
        "feature_c",
        "feature_d",
    ]

    first_selection = select_random_features(
        feature_names=feature_names,
        k=2,
        random_seed=42,
    )

    second_selection = select_random_features(
        feature_names=feature_names,
        k=2,
        random_seed=42,
    )

    assert first_selection == second_selection


def test_select_random_features_preserves_original_order() -> None:
    """Return selected features in their original column order."""
    feature_names = [
        "feature_a",
        "feature_b",
        "feature_c",
        "feature_d",
        "feature_e",
    ]

    selected_features = select_random_features(
        feature_names=feature_names,
        k=3,
        random_seed=42,
    )

    selected_positions = [
        feature_names.index(feature)
        for feature in selected_features
    ]

    assert selected_positions == sorted(selected_positions)


def test_select_random_features_returns_requested_count() -> None:
    """Return exactly k unique feature names."""
    feature_names = [
        "feature_a",
        "feature_b",
        "feature_c",
        "feature_d",
    ]

    selected_features = select_random_features(
        feature_names=feature_names,
        k=3,
        random_seed=42,
    )

    assert len(selected_features) == 3
    assert len(set(selected_features)) == 3
    assert set(selected_features).issubset(feature_names)


@pytest.mark.parametrize(
    "feature_names",
    [
        [],
        ["feature_a", "feature_a"],
        ["feature_a", ""],
        ["feature_a", 2],
    ],
)
def test_select_random_features_rejects_invalid_feature_names(
    feature_names: list[object],
) -> None:
    """Reject empty, duplicated, or non-string feature names."""
    with pytest.raises(SelectionError):
        select_random_features(
            feature_names=feature_names,  # type: ignore[arg-type]
            k=1,
            random_seed=42,
        )


@pytest.mark.parametrize(
    "random_seed",
    [
        1.5,
        "42",
        True,
    ],
)
def test_select_random_features_rejects_invalid_seed(
    random_seed: object,
) -> None:
    """Require the random seed to be an integer."""
    with pytest.raises(
        SelectionError,
        match="random_seed must be an integer",
    ):
        select_random_features(
            feature_names=["feature_a", "feature_b"],
            k=1,
            random_seed=random_seed,  # type: ignore[arg-type]
        )