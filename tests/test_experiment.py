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
def experiment_output(
    smoke_config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Run the smoke experiment once for use across tests."""
    return run_experiment(smoke_config)


@pytest.fixture
def experiment_results(
    experiment_output: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return performance result rows."""
    return experiment_output["results"]


@pytest.fixture
def experiment_rankings(
    experiment_output: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return complete feature-ranking rows."""
    return experiment_output["rankings"]







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
        "ranking_time_seconds",
        "model_training_time_seconds",
        "prediction_time_seconds",
        "actual_tree_depth",
        "n_tree_leaves",
        "n_tree_nodes",
        "n_tree_features_used",
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

def test_run_experiment_records_non_negative_model_times(
    experiment_results: list[dict[str, Any]],
) -> None:
    """Record non-negative downstream training and prediction times."""
    for result in experiment_results:
        assert result["model_training_time_seconds"] >= 0.0
        assert result["prediction_time_seconds"] >= 0.0


def test_run_experiment_records_ranking_time_for_ranked_rows(
    experiment_results: list[dict[str, Any]],
) -> None:
    """Record ranking-stage time only for ranked feature subsets."""
    for result in experiment_results:
        if result["selection_method"] == "gain":
            assert result["ranking_time_seconds"] is not None
            assert result["ranking_time_seconds"] >= 0.0
        else:
            assert result["ranking_time_seconds"] is None


def test_run_experiment_records_valid_tree_complexity(
    experiment_results: list[dict[str, Any]],
) -> None:
    """Record internally consistent fitted-tree complexity."""
    for result in experiment_results:
        actual_depth = result["actual_tree_depth"]
        n_leaves = result["n_tree_leaves"]
        n_nodes = result["n_tree_nodes"]
        n_features_used = result["n_tree_features_used"]

        assert actual_depth is not None
        assert n_leaves is not None
        assert n_nodes is not None
        assert n_features_used is not None

        assert 0 <= actual_depth <= result["max_depth"]
        assert n_leaves >= 1
        assert n_nodes >= 1
        assert n_features_used >= 0

        assert n_nodes == 2 * n_leaves - 1

        assert (
            n_features_used
            <= result["n_selected_features"]
        )


def test_run_experiment_tree_depth_respects_configured_limit(
    experiment_results: list[dict[str, Any]],
) -> None:
    """Ensure fitted trees do not exceed their configured depth."""
    for result in experiment_results:
        assert (
            result["actual_tree_depth"]
            <= result["max_depth"]
        )


def test_run_experiment_features_used_do_not_exceed_subset_size(
    experiment_results: list[dict[str, Any]],
) -> None:
    """Ensure a fitted tree uses no more than the supplied features."""
    for result in experiment_results:
        assert (
            result["n_tree_features_used"]
            <= result["n_selected_features"]
        )

def test_run_experiment_returns_expected_ranking_count(
    experiment_rankings: list[dict[str, Any]],
) -> None:
    """Return one ranking row per feature, method, and fold."""
    assert len(experiment_rankings) == 60


def test_run_experiment_ranking_rows_have_expected_fields(
    experiment_rankings: list[dict[str, Any]],
) -> None:
    """Include all required ranking fields."""
    required_fields = {
        "experiment",
        "dataset",
        "outer_fold",
        "ranking_method",
        "feature",
        "rank",
        "importance_score",
    }

    for row in experiment_rankings:
        assert set(row) == required_fields


def test_run_experiment_rankings_cover_every_feature(
    experiment_rankings: list[dict[str, Any]],
) -> None:
    """Record all 30 features in each outer fold."""
    counts: dict[tuple[int, str], int] = {}

    for row in experiment_rankings:
        key = (
            row["outer_fold"],
            row["ranking_method"],
        )
        counts[key] = counts.get(key, 0) + 1

    assert counts == {
        (1, "gain"): 30,
        (2, "gain"): 30,
    }


def test_run_experiment_ranks_are_consecutive(
    experiment_rankings: list[dict[str, Any]],
) -> None:
    """Assign consecutive ranks from 1 to the feature count."""
    for outer_fold in (1, 2):
        ranks = [
            row["rank"]
            for row in experiment_rankings
            if row["outer_fold"] == outer_fold
            and row["ranking_method"] == "gain"
        ]

        assert ranks == list(range(1, 31))


def test_run_experiment_ranking_scores_are_descending(
    experiment_rankings: list[dict[str, Any]],
) -> None:
    """Store importance scores in descending ranking order."""
    for outer_fold in (1, 2):
        scores = [
            row["importance_score"]
            for row in experiment_rankings
            if row["outer_fold"] == outer_fold
            and row["ranking_method"] == "gain"
        ]

        assert scores == sorted(
            scores,
            reverse=True,
        )


def test_run_experiment_ranking_features_are_unique(
    experiment_rankings: list[dict[str, Any]],
) -> None:
    """Record every feature exactly once per fold and method."""
    for outer_fold in (1, 2):
        features = [
            row["feature"]
            for row in experiment_rankings
            if row["outer_fold"] == outer_fold
            and row["ranking_method"] == "gain"
        ]

        assert len(features) == 30
        assert len(set(features)) == 30


def test_run_experiment_calls_callback_after_dataset(
    smoke_config: dict[str, Any],
) -> None:
    """Call the completion callback once after a dataset finishes."""
    callback_calls: list[
        tuple[
            str,
            dict[str, list[dict[str, Any]]],
        ]
    ] = []

    def on_dataset_complete(
        dataset_name: str,
        output: dict[str, list[dict[str, Any]]],
    ) -> None:
        callback_calls.append(
            (
                dataset_name,
                deepcopy(output),
            )
        )

    final_output = run_experiment(
        smoke_config,
        on_dataset_complete=on_dataset_complete,
    )

    assert len(callback_calls) == 1

    dataset_name, callback_output = callback_calls[0]

    assert dataset_name == "breast_cancer_wisconsin"
    assert callback_output == final_output


def test_run_experiment_callback_receives_cumulative_output(
    smoke_config: dict[str, Any],
) -> None:
    """Provide cumulative results after each completed dataset."""
    config = deepcopy(smoke_config)

    config["datasets"]["steel_plates_faults"][
        "enabled"
    ] = True

    callback_calls: list[
        tuple[
            str,
            dict[str, list[dict[str, Any]]],
        ]
    ] = []

    def on_dataset_complete(
        dataset_name: str,
        output: dict[str, list[dict[str, Any]]],
    ) -> None:
        callback_calls.append(
            (
                dataset_name,
                deepcopy(output),
            )
        )

    final_output = run_experiment(
        config,
        on_dataset_complete=on_dataset_complete,
    )

    assert [
        dataset_name
        for dataset_name, _ in callback_calls
    ] == [
        "breast_cancer_wisconsin",
        "steel_plates_faults",
    ]

    first_output = callback_calls[0][1]
    second_output = callback_calls[1][1]

    assert {
        row["dataset"]
        for row in first_output["results"]
    } == {"breast_cancer_wisconsin"}

    assert {
        row["dataset"]
        for row in second_output["results"]
    } == {
        "breast_cancer_wisconsin",
        "steel_plates_faults",
    }

    assert len(second_output["results"]) > len(
        first_output["results"]
    )

    assert second_output == final_output