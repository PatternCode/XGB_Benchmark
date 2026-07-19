"""Tests for the experiment command-line entry point."""

from pathlib import Path
from typing import Any

import pytest

from experiments import run_experiment as run_experiment_cli


@pytest.fixture
def experiment_output() -> dict[str, list[dict[str, Any]]]:
    """Return a minimal valid experiment output."""
    return {
        "results": [
            {
                "dataset": "breast_cancer_wisconsin",
                "accuracy": 0.95,
            }
        ],
        "rankings": [
            {
                "dataset": "breast_cancer_wisconsin",
                "feature": "mean_radius",
                "rank": 1,
            }
        ],
        "selected_features": [
            {
                "dataset": "breast_cancer_wisconsin",
                "feature": "mean_radius",
            }
        ],
    }


def test_main_writes_dataset_checkpoint_and_final_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    experiment_output: dict[str, list[dict[str, Any]]],
) -> None:
    """Write output and progress during experiment execution."""
    config = {
        "experiment": {
            "name": "smoke_test",
        },
        "output": {
            "directory": str(tmp_path),
        },
    }

    run_directory = tmp_path / "smoke_test_run"

    write_calls: list[
        dict[str, list[dict[str, Any]]]
    ] = []

    progress_calls: list[list[str]] = []

    def fake_load_config(
        config_path: Path,
    ) -> dict[str, Any]:
        assert config_path == Path(
            "configs/smoke_test.yaml"
        )
        return config

    def fake_create_run_directory(
        *,
        output_directory: str | Path,
        experiment_name: str,
    ) -> Path:
        assert output_directory == str(tmp_path)
        assert experiment_name == "smoke_test"
        return run_directory

    def fake_write_experiment_output(
        *,
        experiment_output: dict[
            str,
            list[dict[str, Any]],
        ],
        run_directory: str | Path,
        experiment_name: str,
        config: dict[str, Any],
        started_at_utc: str | None = None,
        completed_at_utc: str | None = None,
        runtime_seconds: float | None = None,
    ) -> dict[str, Path]:
        write_calls.append(experiment_output)

        path = Path(run_directory)

        return {
            "run_directory": path,
            "results": path / "results.csv",
            "rankings": path / "rankings.csv",
            "selected_features": (
                path / "selected_features.csv"
            ),
            "metadata": path / "run_metadata.json",
        }

    def fake_write_progress(
        *,
        run_directory: str | Path,
        experiment_name: str,
        completed_datasets: list[str],
    ) -> Path:
        received_run_directory = Path(run_directory)

        assert received_run_directory == (
            tmp_path / "smoke_test_run"
        )
        assert experiment_name == "smoke_test"

        progress_calls.append(
            completed_datasets.copy()
        )

        return (
            received_run_directory / "progress.json"
        )

    def fake_run_experiment(
        received_config: dict[str, Any],
        on_dataset_complete: Any = None,
    ) -> dict[str, list[dict[str, Any]]]:
        assert received_config is config
        assert on_dataset_complete is not None

        on_dataset_complete(
            "breast_cancer_wisconsin",
            experiment_output,
        )

        return experiment_output

    monkeypatch.setattr(
        run_experiment_cli,
        "load_config",
        fake_load_config,
    )
    monkeypatch.setattr(
        run_experiment_cli,
        "create_run_directory",
        fake_create_run_directory,
    )
    monkeypatch.setattr(
        run_experiment_cli,
        "write_experiment_output",
        fake_write_experiment_output,
    )
    monkeypatch.setattr(
        run_experiment_cli,
        "write_progress",
        fake_write_progress,
    )
    monkeypatch.setattr(
        run_experiment_cli,
        "run_experiment",
        fake_run_experiment,
    )

    returned_directory = run_experiment_cli.main(
        [
            "--config",
            "configs/smoke_test.yaml",
        ]
    )

    assert returned_directory == run_directory

    assert len(write_calls) == 2
    assert write_calls[0] is experiment_output
    assert write_calls[1] is experiment_output

    assert progress_calls == [
        ["breast_cancer_wisconsin"],
    ]


def test_main_creates_run_directory_before_experiment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    experiment_output: dict[str, list[dict[str, Any]]],
) -> None:
    """Create the run directory before experiment execution."""
    config = {
        "experiment": {
            "name": "smoke_test",
        },
        "output": {
            "directory": str(tmp_path),
        },
    }

    call_order: list[str] = []
    run_directory = tmp_path / "smoke_test_run"

    def fake_load_config(
        config_path: Path,
    ) -> dict[str, Any]:
        return config

    def fake_create_run_directory(
        *,
        output_directory: str | Path,
        experiment_name: str,
    ) -> Path:
        call_order.append("create_run_directory")
        return run_directory

    def fake_run_experiment(
        received_config: dict[str, Any],
        on_dataset_complete: Any = None,
    ) -> dict[str, list[dict[str, Any]]]:
        call_order.append("run_experiment")
        return experiment_output

    def fake_write_experiment_output(
        *,
        experiment_output: dict[
            str,
            list[dict[str, Any]],
        ],
        run_directory: str | Path,
        experiment_name: str,
        config: dict[str, Any],
        started_at_utc: str | None = None,
        completed_at_utc: str | None = None,
        runtime_seconds: float | None = None,
    ) -> dict[str, Path]:
        path = Path(run_directory)

        return {
            "run_directory": path,
            "results": path / "results.csv",
            "rankings": path / "rankings.csv",
            "selected_features": (
                path / "selected_features.csv"
            ),
            "metadata": path / "run_metadata.json",
        }

    def fake_write_progress(
        *,
        run_directory: str | Path,
        experiment_name: str,
        completed_datasets: list[str],
    ) -> Path:
        return Path(run_directory) / "progress.json"

    monkeypatch.setattr(
        run_experiment_cli,
        "load_config",
        fake_load_config,
    )
    monkeypatch.setattr(
        run_experiment_cli,
        "create_run_directory",
        fake_create_run_directory,
    )
    monkeypatch.setattr(
        run_experiment_cli,
        "run_experiment",
        fake_run_experiment,
    )
    monkeypatch.setattr(
        run_experiment_cli,
        "write_experiment_output",
        fake_write_experiment_output,
    )
    monkeypatch.setattr(
        run_experiment_cli,
        "write_progress",
        fake_write_progress,
    )

    run_experiment_cli.main(
        [
            "--config",
            "configs/smoke_test.yaml",
        ]
    )

    assert call_order == [
        "create_run_directory",
        "run_experiment",
    ]


def test_main_writes_checkpoint_before_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    experiment_output: dict[str, list[dict[str, Any]]],
) -> None:
    """Mark a dataset complete after writing its output."""
    config = {
        "experiment": {
            "name": "smoke_test",
        },
        "output": {
            "directory": str(tmp_path),
        },
    }

    run_directory = tmp_path / "smoke_test_run"
    call_order: list[str] = []

    def fake_load_config(
        config_path: Path,
    ) -> dict[str, Any]:
        return config

    def fake_create_run_directory(
        *,
        output_directory: str | Path,
        experiment_name: str,
    ) -> Path:
        return run_directory

    def fake_write_experiment_output(
        *,
        experiment_output: dict[
            str,
            list[dict[str, Any]],
        ],
        run_directory: str | Path,
        experiment_name: str,
        config: dict[str, Any],
        started_at_utc: str | None = None,
        completed_at_utc: str | None = None,
        runtime_seconds: float | None = None,
    ) -> dict[str, Path]:
        call_order.append("write_output")

        path = Path(run_directory)

        return {
            "run_directory": path,
            "results": path / "results.csv",
            "rankings": path / "rankings.csv",
            "selected_features": (
                path / "selected_features.csv"
            ),
            "metadata": path / "run_metadata.json",
        }

    def fake_write_progress(
        *,
        run_directory: str | Path,
        experiment_name: str,
        completed_datasets: list[str],
    ) -> Path:
        call_order.append("write_progress")

        assert completed_datasets == [
            "breast_cancer_wisconsin",
        ]

        return Path(run_directory) / "progress.json"

    def fake_run_experiment(
        received_config: dict[str, Any],
        on_dataset_complete: Any = None,
    ) -> dict[str, list[dict[str, Any]]]:
        assert on_dataset_complete is not None

        on_dataset_complete(
            "breast_cancer_wisconsin",
            experiment_output,
        )

        return experiment_output

    monkeypatch.setattr(
        run_experiment_cli,
        "load_config",
        fake_load_config,
    )
    monkeypatch.setattr(
        run_experiment_cli,
        "create_run_directory",
        fake_create_run_directory,
    )
    monkeypatch.setattr(
        run_experiment_cli,
        "write_experiment_output",
        fake_write_experiment_output,
    )
    monkeypatch.setattr(
        run_experiment_cli,
        "write_progress",
        fake_write_progress,
    )
    monkeypatch.setattr(
        run_experiment_cli,
        "run_experiment",
        fake_run_experiment,
    )

    run_experiment_cli.main(
        [
            "--config",
            "configs/smoke_test.yaml",
        ]
    )

    assert call_order == [
        "write_output",
        "write_progress",
        "write_output",
    ]