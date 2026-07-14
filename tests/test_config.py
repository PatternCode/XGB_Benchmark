"""Tests for benchmark configuration loading and validation."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from benchmark.config import (
    ConfigurationError,
    load_config,
    validate_config,
)


@pytest.fixture
def valid_config() -> dict[str, Any]:
    """Return a minimal valid benchmark configuration."""
    return {
        "experiment": {
            "name": "test_experiment",
            "random_seed": 42,
        },
        "datasets": {
            "breast_cancer_wisconsin": {
                "enabled": True,
            },
        },
        "cross_validation": {
            "n_splits": 2,
            "shuffle": True,
        },
        "feature_selection": {
            "ranking_methods": ["gain"],
            "feature_percentages": [10, 30],
            "random_repetitions": 1,
        },
        "ranking_xgboost": {},
        "shap": {},
        "models": {},
        "metrics": ["accuracy"],
        "output": {
            "directory": "results/runs",
        },
    }


def write_yaml(path: Path, content: Any) -> None:
    """Write Python content to a YAML file."""
    path.write_text(
        yaml.safe_dump(content, sort_keys=False),
        encoding="utf-8",
    )


def test_load_config_returns_valid_dictionary(
    tmp_path: Path,
    valid_config: dict[str, Any],
) -> None:
    """Load a valid YAML configuration."""
    config_path = tmp_path / "valid.yaml"
    write_yaml(config_path, valid_config)

    loaded_config = load_config(config_path)

    assert loaded_config == valid_config


def test_load_config_raises_for_missing_file(tmp_path: Path) -> None:
    """Reject a configuration path that does not exist."""
    config_path = tmp_path / "missing.yaml"

    with pytest.raises(
        FileNotFoundError,
        match="Configuration file does not exist",
    ):
        load_config(config_path)


def test_load_config_raises_for_empty_file(tmp_path: Path) -> None:
    """Reject an empty configuration file."""
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(
        ConfigurationError,
        match="Configuration file is empty",
    ):
        load_config(config_path)


def test_load_config_raises_for_non_mapping_yaml(tmp_path: Path) -> None:
    """Reject YAML whose top-level value is not a mapping."""
    config_path = tmp_path / "list.yaml"
    write_yaml(config_path, ["experiment", "datasets"])

    with pytest.raises(
        ConfigurationError,
        match="top-level configuration value must be a YAML mapping",
    ):
        load_config(config_path)


def test_validate_config_raises_for_missing_section(
    valid_config: dict[str, Any],
) -> None:
    """Reject a configuration with a missing required section."""
    invalid_config = deepcopy(valid_config)
    del invalid_config["models"]

    with pytest.raises(
        ConfigurationError,
        match="Missing required configuration section: 'models'",
    ):
        validate_config(invalid_config)


@pytest.mark.parametrize("n_splits", [None, "5", 2.5])
def test_validate_config_rejects_non_integer_n_splits(
    valid_config: dict[str, Any],
    n_splits: Any,
) -> None:
    """Require the number of folds to be an integer."""
    invalid_config = deepcopy(valid_config)
    invalid_config["cross_validation"]["n_splits"] = n_splits

    with pytest.raises(
        ConfigurationError,
        match="cross_validation.n_splits must be an integer",
    ):
        validate_config(invalid_config)


@pytest.mark.parametrize("n_splits", [-1, 0, 1])
def test_validate_config_rejects_too_few_splits(
    valid_config: dict[str, Any],
    n_splits: int,
) -> None:
    """Require at least two cross-validation folds."""
    invalid_config = deepcopy(valid_config)
    invalid_config["cross_validation"]["n_splits"] = n_splits

    with pytest.raises(
        ConfigurationError,
        match="cross_validation.n_splits must be at least 2",
    ):
        validate_config(invalid_config)


@pytest.mark.parametrize("percentage", [-10, 0, 101])
def test_validate_config_rejects_out_of_range_percentage(
    valid_config: dict[str, Any],
    percentage: int,
) -> None:
    """Require feature percentages to lie between 0 and 100."""
    invalid_config = deepcopy(valid_config)
    invalid_config["feature_selection"]["feature_percentages"] = [
        percentage
    ]

    with pytest.raises(
        ConfigurationError,
        match="Every feature percentage must be greater than 0",
    ):
        validate_config(invalid_config)


def test_validate_config_rejects_empty_percentage_list(
    valid_config: dict[str, Any],
) -> None:
    """Require at least one feature percentage."""
    invalid_config = deepcopy(valid_config)
    invalid_config["feature_selection"]["feature_percentages"] = []

    with pytest.raises(
        ConfigurationError,
        match="feature_percentages must not be empty",
    ):
        validate_config(invalid_config)


@pytest.mark.parametrize("random_repetitions", [0, -1])
def test_validate_config_rejects_invalid_random_repetitions(
    valid_config: dict[str, Any],
    random_repetitions: int,
) -> None:
    """Require at least one random-feature repetition."""
    invalid_config = deepcopy(valid_config)
    invalid_config["feature_selection"][
        "random_repetitions"
    ] = random_repetitions

    with pytest.raises(
        ConfigurationError,
        match="random_repetitions must be at least 1",
    ):
        validate_config(invalid_config)
