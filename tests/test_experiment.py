"""Tests for complete benchmark experiment execution."""

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from benchmark.config import load_config
from benchmark.experiment import ExperimentError, run_experiment


@pytest.fixture
def smoke_config() -> dict[str, Any]:
    """Load the real smoke-test configuration."""
    return load_config("configs/smoke_test.yaml")


@pytest.fixture
def experiment_results(
    smoke_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run the smoke experiment once for use across tests."""
    return run_experiment(smoke_config)


def test_run_experiment_returns_expected_number_of_rows(
    experiment_results: list[dict[str, Any]],
) -> None:
    """Return the expected 20 rows for the smoke configuration."""
    assert len(experiment_results) == 20


def test_run_experiment_returns_flat_result_records(
    experiment_results: list[dict[str, Any]],
) -> None:
    """Return a list of flat dictionaries."""
    assert isinstance(experiment_results, list)
    assert experiment_results

    for result in experiment_results:
        assert isinstance(result, dict)

        for value in result.values():
            assert not isinstance(value, (dict, list, tuple, set))


def test_run_experiment_returns_expected_fields(
    experiment_results: list[dict[str, Any]],
) -> None:
    """Include all required result fields."""
    required_fields = {
        "experiment",
        "dataset",
        "outer_fold",
        "selection_method",
        "requested_percentage",
        "actual_percentage",
        "n_selected_features",
        "random_repetition",
        "model",
        "max_depth",
        "n_neighbors",
        "C",
        "gamma",
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
    }

    for result in experiment_results:
        assert required_fields.issubset(result)


def test_run_experiment_uses_only_enabled_dataset(
    experiment_results: list[dict[str, Any]],
) -> None:
    """Evaluate only the dataset enabled in smoke_test.yaml."""
    dataset_names = {
        result["dataset"]
        for result in experiment_results
    }

    assert dataset_names == {"breast_cancer_wisconsin"}


def test_run_experiment_uses_all_outer_folds(
    experiment_results: list[dict[str, Any]],
) -> None:
    """Return results for both smoke-test outer folds."""
    folds = {
        result["outer_fold"]
        for result in experiment_results
    }

    assert folds == {1, 2}


def test_run_experiment_uses_expected_selection_methods(
    experiment_results: list[dict[str, Any]],
) -> None:
    """Evaluate ranked, random, and all-feature strategies."""
    selection_methods = {
        result["selection_method"]
        for result in experiment_results
    }

    assert selection_methods == {
        "gain",
        "random",
        "all_features",
    }


def test_run_experiment_uses_only_enabled_model(
    experiment_results: list[dict[str, Any]],
) -> None:
    """Evaluate only the enabled downstream model."""
    model_names = {
        result["model"]
        for result in experiment_results
    }

    assert model_names == {"decision_tree"}


def test_run_experiment_uses_expected_tree_depths(
    experiment_results: list[dict[str, Any]],
) -> None:
    """Evaluate both configured decision-tree depths."""
    tree_depths = {
        result["max_depth"]
        for result in experiment_results
    }

    assert tree_depths == {2, 4}


def test_run_experiment_metrics_are_finite(
    experiment_results: list[dict[str, Any]],
) -> None:
    """Return finite metric values for every result row."""
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
    ]

    for result in experiment_results:
        for metric_name in metric_names:
            assert np.isfinite(result[metric_name])
            assert 0.0 <= result[metric_name] <= 1.0


def test_run_experiment_all_feature_rows_are_correct(
    experiment_results: list[dict[str, Any]],
) -> None:
    """Represent the all-feature baseline correctly."""
    all_feature_rows = [
        result
        for result in experiment_results
        if result["selection_method"] == "all_features"
    ]

    assert len(all_feature_rows) == 4

    for result in all_feature_rows:
        assert result["requested_percentage"] is None
        assert result["actual_percentage"] == 100.0
        assert result["n_selected_features"] == 30
        assert result["random_repetition"] is None


def test_run_experiment_random_rows_record_repetition(
    experiment_results: list[dict[str, Any]],
) -> None:
    """Record the random-feature repetition number."""
    random_rows = [
        result
        for result in experiment_results
        if result["selection_method"] == "random"
    ]

    assert len(random_rows) == 8

    for result in random_rows:
        assert result["random_repetition"] == 1


def test_run_experiment_ranked_rows_have_no_random_repetition(
    experiment_results: list[dict[str, Any]],
) -> None:
    """Leave random repetition empty for ranked subsets."""
    ranked_rows = [
        result
        for result in experiment_results
        if result["selection_method"] == "gain"
    ]

    assert len(ranked_rows) == 8

    for result in ranked_rows:
        assert result["random_repetition"] is None


def test_run_experiment_rejects_no_enabled_datasets(
    smoke_config: dict[str, Any],
) -> None:
    """Require at least one enabled dataset."""
    invalid_config = deepcopy(smoke_config)

    for dataset_config in invalid_config["datasets"].values():
        dataset_config["enabled"] = False

    with pytest.raises(
        ExperimentError,
        match="At least one dataset must be enabled",
    ):
        run_experiment(invalid_config)


def test_run_experiment_rejects_no_enabled_models(
    smoke_config: dict[str, Any],
) -> None:
    """Require at least one enabled downstream model."""
    invalid_config = deepcopy(smoke_config)

    for model_config in invalid_config["models"].values():
        model_config["enabled"] = False

    with pytest.raises(
        ExperimentError,
        match="At least one downstream model must be enabled",
    ):
        run_experiment(invalid_config)