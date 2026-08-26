from pathlib import Path

import pytest

import formalizer.eval.cli as cli_module
from formalizer.eval import LeanVerified
from formalizer.settings import Settings


class FakeDataset:
    def __init__(self) -> None:
        self.evaluators: list[object] = []

    def add_evaluator(self, evaluator: object) -> None:
        self.evaluators.append(evaluator)


class FakeReport:
    def __init__(
        self,
        *,
        failures: list[object] | None = None,
        evaluator_failures: list[object] | None = None,
        case_evaluator_failures: list[object] | None = None,
    ) -> None:
        self.failures = failures or []
        self.report_evaluator_failures = evaluator_failures or []
        self.cases = [FakeCase(case_evaluator_failures or [])]
        self.printed = False

    def print(self) -> None:
        self.printed = True


class FakeCase:
    def __init__(self, evaluator_failures: list[object]) -> None:
        self.evaluator_failures = evaluator_failures


def test_eval_cli_loads_dataset_runs_experiment_and_configures_logfire(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "baseline.yaml"
    runs_dir = tmp_path / "runs"
    dataset = FakeDataset()
    report = FakeReport()
    loaded_paths: list[Path] = []
    logfire_calls = 0
    evaluation_calls: list[tuple[FakeDataset, Settings, int, dict[str, object]]] = []

    class FakeDatasetType:
        @classmethod
        def from_file(cls, path: Path) -> FakeDataset:
            loaded_paths.append(path)
            return dataset

    def fake_configure_logfire() -> None:
        nonlocal logfire_calls
        logfire_calls += 1

    async def fake_evaluate_dataset(
        received_dataset: FakeDataset,
        settings: Settings,
        *,
        repeat: int,
        metadata: dict[str, object],
    ) -> FakeReport:
        evaluation_calls.append((received_dataset, settings, repeat, metadata))
        return report

    monkeypatch.setattr(cli_module, "FormalizerDataset", FakeDatasetType)
    monkeypatch.setattr(cli_module, "configure_logfire", fake_configure_logfire)
    monkeypatch.setattr(cli_module, "evaluate_dataset", fake_evaluate_dataset)

    exit_code = cli_module.main(
        [
            "--dataset",
            str(dataset_path),
            "--model",
            "test:model",
            "--runs-dir",
            str(runs_dir),
            "--repeat",
            "3",
            "--logfire",
        ]
    )

    assert exit_code == 0
    assert loaded_paths == [dataset_path]
    assert len(dataset.evaluators) == 1
    assert isinstance(dataset.evaluators[0], LeanVerified)
    assert logfire_calls == 1
    assert len(evaluation_calls) == 1
    received_dataset, settings, repeat, metadata = evaluation_calls[0]
    assert received_dataset is dataset
    assert settings.model_name == "test:model"
    assert settings.run.runs_dir == runs_dir
    assert repeat == 3
    assert metadata == {"model_name": "test:model"}
    assert report.printed


@pytest.mark.parametrize(
    "report",
    [
        FakeReport(failures=[object()]),
        FakeReport(evaluator_failures=[object()]),
        FakeReport(case_evaluator_failures=[object()]),
    ],
    ids=["task", "report-evaluator", "case-evaluator"],
)
def test_eval_cli_returns_failure_for_execution_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    report: FakeReport,
) -> None:
    class FakeDatasetType:
        @classmethod
        def from_file(cls, path: Path) -> FakeDataset:
            return FakeDataset()

    async def fake_evaluate_dataset(*args: object, **kwargs: object) -> FakeReport:
        return report

    monkeypatch.setattr(cli_module, "FormalizerDataset", FakeDatasetType)
    monkeypatch.setattr(cli_module, "evaluate_dataset", fake_evaluate_dataset)

    exit_code = cli_module.main(
        [
            "--dataset",
            str(tmp_path / "baseline.yaml"),
            "--model",
            "test:model",
        ]
    )

    assert exit_code == 1
    assert report.printed


def test_eval_cli_handles_interruption(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    class FakeDatasetType:
        @classmethod
        def from_file(cls, path: Path) -> FakeDataset:
            return FakeDataset()

    async def interrupt(*args: object, **kwargs: object) -> FakeReport:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_module, "FormalizerDataset", FakeDatasetType)
    monkeypatch.setattr(cli_module, "evaluate_dataset", interrupt)

    exit_code = cli_module.main(
        [
            "--dataset",
            str(tmp_path / "baseline.yaml"),
            "--model",
            "test:model",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 130
    assert "interrupted" in captured.err.lower()


def test_eval_cli_rejects_nonpositive_repeat() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(
            [
                "--dataset",
                "baseline.yaml",
                "--model",
                "test:model",
                "--repeat",
                "0",
            ]
        )

    assert exc_info.value.code == 2
