"""Load and validate configuration files for XGB Benchmark experiments."""

from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(Exception):
    """Raised when a benchmark configuration file is invalid."""


def _require_section(
    config: dict[str, Any],
    section: str,
    expected_type: type,
) -> None:
    """Ensure that a required section exists and has the expected type."""
    if section not in config:
        raise ConfigurationError(
            f"Missing required configuration section: '{section}'."
        )

    if not isinstance(config[section], expected_type):
        raise ConfigurationError(
            f"Configuration section '{section}' must be of type "
            f"{expected_type.__name__}, but received "
            f"{type(config[section]).__name__}."
        )


def validate_config(config: dict[str, Any]) -> None:
    """Validate the essential structure and values of a configuration.

    Parameters
    ----------
    config
        Parsed benchmark configuration.

    Raises
    ------
    ConfigurationError
        If a required section is missing or an essential value is invalid.
    """
    required_sections = {
        "experiment": dict,
        "datasets": dict,
        "cross_validation": dict,
        "feature_selection": dict,
        "ranking_xgboost": dict,
        "shap": dict,
        "models": dict,
        "metrics": list,
        "output": dict,
    }

    for section, expected_type in required_sections.items():
        _require_section(config, section, expected_type)

    n_splits = config["cross_validation"].get("n_splits")

    if not isinstance(n_splits, int):
        raise ConfigurationError(
            "cross_validation.n_splits must be an integer."
        )

    if n_splits < 2:
        raise ConfigurationError(
            "cross_validation.n_splits must be at least 2."
        )

    feature_percentages = config["feature_selection"].get(
        "feature_percentages"
    )

    if not isinstance(feature_percentages, list):
        raise ConfigurationError(
            "feature_selection.feature_percentages must be a list."
        )

    if not feature_percentages:
        raise ConfigurationError(
            "feature_selection.feature_percentages must not be empty."
        )

    for percentage in feature_percentages:
        if not isinstance(percentage, (int, float)):
            raise ConfigurationError(
                "Every feature percentage must be a number."
            )

        if not 0 < percentage <= 100:
            raise ConfigurationError(
                "Every feature percentage must be greater than 0 "
                "and no greater than 100."
            )

    random_repetitions = config["feature_selection"].get(
        "random_repetitions"
    )

    if not isinstance(random_repetitions, int):
        raise ConfigurationError(
            "feature_selection.random_repetitions must be an integer."
        )

    if random_repetitions < 1:
        raise ConfigurationError(
            "feature_selection.random_repetitions must be at least 1."
        )


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a benchmark configuration from a YAML file.

    Parameters
    ----------
    config_path
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed and validated benchmark configuration.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ConfigurationError
        If the path is not a file, the YAML is invalid, the file is empty,
        the top-level value is not a mapping, or validation fails.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {path}"
        )

    if not path.is_file():
        raise ConfigurationError(
            f"Configuration path is not a file: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ConfigurationError(
            f"Invalid YAML in configuration file '{path}': {error}"
        ) from error
    except OSError as error:
        raise ConfigurationError(
            f"Could not read configuration file '{path}': {error}"
        ) from error

    if config is None:
        raise ConfigurationError(
            f"Configuration file is empty: {path}"
        )

    if not isinstance(config, dict):
        raise ConfigurationError(
            "The top-level configuration value must be a YAML mapping "
            f"(dictionary), but '{path}' contains "
            f"{type(config).__name__}."
        )

    validate_config(config)

    return config
