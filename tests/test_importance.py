"""Tests for XGBoost and SHAP feature-importance computation."""

from typing import Any

import pandas as pd
import pytest
import xgboost as xgb

from benchmark.importance import (
    ImportanceError,
    compute_feature_importance,
)


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return a small binary-classification training dataset."""
    X_train = pd.DataFrame(
        {
            "feature_a": [
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
            ],
            "feature_b": [
                1.5,
                1.4,
                1.3,
                1.2,
                1.1,
                1.0,
                0.5,
                0.4,
                0.3,
                0.2,
                0.1,
                0.0,
            ],
            "feature_c": [
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
            ],
            "feature_d": [
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
            ],
        }
    )

    y_train = pd.Series(
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        name="target",
    )

    return X_train, y_train


@pytest.fixture
def xgboost_config() -> dict[str, Any]:
    """Return a small ranking-model configuration."""
    return {
        "booster": "gbtree",
        "objective": "auto",
        "eval_metric": "auto",
        "num_boost_round": 5,
        "max_depth": 2,
        "eta": 0.2,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "tree_method": "hist",
    }


@pytest.fixture
def shap_config() -> dict[str, Any]:
    """Return a small SHAP configuration."""
    return {
        "explainer": "tree",
        "max_samples": None,
    }


def assert_valid_ranking(
    ranking: pd.Series,
    method: str,
    feature_names: list[str],
) -> None:
    """Check the common properties required from every ranking."""
    assert isinstance(ranking, pd.Series)
    assert ranking.name == method
    assert len(ranking) == len(feature_names)
    assert ranking.index.is_unique
    assert set(ranking.index) == set(feature_names)
    assert ranking.notna().all()
    assert (ranking >= 0).all()
    assert ranking.is_monotonic_decreasing


def test_compute_gain_importance_returns_complete_ranking(
    training_data: tuple[pd.DataFrame, pd.Series],
    xgboost_config: dict[str, Any],
    shap_config: dict[str, Any],
) -> None:
    """Compute one complete gain ranking."""
    X_train, y_train = training_data

    booster, rankings = compute_feature_importance(
        X_train=X_train,
        y_train=y_train,
        xgboost_config=xgboost_config,
        shap_config=shap_config,
        ranking_methods=["gain"],
        random_seed=42,
    )

    assert isinstance(booster, xgb.Booster)
    assert list(rankings) == ["gain"]

    assert_valid_ranking(
        ranking=rankings["gain"],
        method="gain",
        feature_names=X_train.columns.tolist(),
    )


def test_compute_all_importance_methods(
    training_data: tuple[pd.DataFrame, pd.Series],
    xgboost_config: dict[str, Any],
    shap_config: dict[str, Any],
) -> None:
    """Compute gain, weight, cover, and SHAP rankings."""
    X_train, y_train = training_data
    methods = ["gain", "weight", "cover", "shap"]

    _, rankings = compute_feature_importance(
        X_train=X_train,
        y_train=y_train,
        xgboost_config=xgboost_config,
        shap_config=shap_config,
        ranking_methods=methods,
        random_seed=42,
    )

    assert list(rankings) == methods

    for method in methods:
        assert_valid_ranking(
            ranking=rankings[method],
            method=method,
            feature_names=X_train.columns.tolist(),
        )


def test_compute_feature_importance_is_reproducible(
    training_data: tuple[pd.DataFrame, pd.Series],
    xgboost_config: dict[str, Any],
    shap_config: dict[str, Any],
) -> None:
    """Produce identical rankings when the same seed is used."""
    X_train, y_train = training_data

    _, first_rankings = compute_feature_importance(
        X_train=X_train,
        y_train=y_train,
        xgboost_config=xgboost_config,
        shap_config=shap_config,
        ranking_methods=["gain", "shap"],
        random_seed=42,
    )

    _, second_rankings = compute_feature_importance(
        X_train=X_train,
        y_train=y_train,
        xgboost_config=xgboost_config,
        shap_config=shap_config,
        ranking_methods=["gain", "shap"],
        random_seed=42,
    )

    pd.testing.assert_series_equal(
        first_rankings["gain"],
        second_rankings["gain"],
    )

    pd.testing.assert_series_equal(
        first_rankings["shap"],
        second_rankings["shap"],
    )


@pytest.mark.parametrize(
    "ranking_methods",
    [
        [],
        ["unknown"],
        ["gain", "gain"],
    ],
)
def test_compute_feature_importance_rejects_invalid_methods(
    training_data: tuple[pd.DataFrame, pd.Series],
    xgboost_config: dict[str, Any],
    shap_config: dict[str, Any],
    ranking_methods: list[str],
) -> None:
    """Reject empty, unsupported, or duplicated ranking methods."""
    X_train, y_train = training_data

    with pytest.raises(ImportanceError):
        compute_feature_importance(
            X_train=X_train,
            y_train=y_train,
            xgboost_config=xgboost_config,
            shap_config=shap_config,
            ranking_methods=ranking_methods,
            random_seed=42,
        )


def test_compute_feature_importance_rejects_length_mismatch(
    training_data: tuple[pd.DataFrame, pd.Series],
    xgboost_config: dict[str, Any],
    shap_config: dict[str, Any],
) -> None:
    """Reject feature and target objects with different lengths."""
    X_train, y_train = training_data
    invalid_y = y_train.iloc[:-1]

    with pytest.raises(
        ImportanceError,
        match="same number of samples",
    ):
        compute_feature_importance(
            X_train=X_train,
            y_train=invalid_y,
            xgboost_config=xgboost_config,
            shap_config=shap_config,
            ranking_methods=["gain"],
            random_seed=42,
        )


def test_compute_feature_importance_rejects_non_contiguous_labels(
    training_data: tuple[pd.DataFrame, pd.Series],
    xgboost_config: dict[str, Any],
    shap_config: dict[str, Any],
) -> None:
    """Require class labels to be encoded from zero consecutively."""
    X_train, y_train = training_data
    invalid_y = y_train.replace({1: 2})

    with pytest.raises(
        ImportanceError,
        match="integer encoded from 0",
    ):
        compute_feature_importance(
            X_train=X_train,
            y_train=invalid_y,
            xgboost_config=xgboost_config,
            shap_config=shap_config,
            ranking_methods=["gain"],
            random_seed=42,
        )


def test_compute_feature_importance_rejects_invalid_boost_rounds(
    training_data: tuple[pd.DataFrame, pd.Series],
    xgboost_config: dict[str, Any],
    shap_config: dict[str, Any],
) -> None:
    """Require a positive number of XGBoost boosting rounds."""
    X_train, y_train = training_data
    invalid_config = dict(xgboost_config)
    invalid_config["num_boost_round"] = 0

    with pytest.raises(
        ImportanceError,
        match="num_boost_round must be a positive integer",
    ):
        compute_feature_importance(
            X_train=X_train,
            y_train=y_train,
            xgboost_config=invalid_config,
            shap_config=shap_config,
            ranking_methods=["gain"],
            random_seed=42,
        )


def test_compute_feature_importance_rejects_invalid_shap_sample_count(
    training_data: tuple[pd.DataFrame, pd.Series],
    xgboost_config: dict[str, Any],
    shap_config: dict[str, Any],
) -> None:
    """Require a positive SHAP sample limit when one is specified."""
    X_train, y_train = training_data
    invalid_shap_config = dict(shap_config)
    invalid_shap_config["max_samples"] = 0

    with pytest.raises(
        ImportanceError,
        match="shap.max_samples must be null or a positive integer",
    ):
        compute_feature_importance(
            X_train=X_train,
            y_train=y_train,
            xgboost_config=xgboost_config,
            shap_config=invalid_shap_config,
            ranking_methods=["shap"],
            random_seed=42,
        )