from datetime import datetime
from hashlib import sha256
from pathlib import Path

import pytest

import formalizer.eval.cli as cli_module
from formalizer.eval import LeanVerified
from formalizer.settings import Settings


class FakeDataset:
    def __init__(self) -> None:
        self.name = "baseline-v1"
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
    dataset_path.write_text("name: baseline-v1\ncases: []\n", encoding="utf-8")
    output_dir = tmp_path / "experiment"
    dataset = FakeDataset()
    report = FakeReport()
    loaded_paths: list[Path] = []
    logfire_calls = 0
    evaluation_calls: list[tuple[FakeDataset, Settings, str, int, dict[str, object]]] = []
    persisted_reports: list[tuple[Path, FakeReport]] = []

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
        name: str,
        repeat: int,
        metadata: dict[str, object],
    ) -> FakeReport:
        evaluation_calls.append((received_dataset, settings, name, repeat, metadata))
        return report

    def fake_write_evaluation_report(path: Path, received_report: FakeReport) -> None:
        persisted_reports.append((path, received_report))

    monkeypatch.setattr(cli_module, "load_formalizer_dataset", FakeDatasetType.from_file)
    monkeypatch.setattr(cli_module, "configure_logfire", fake_configure_logfire)
    monkeypatch.setattr(cli_module, "evaluate_dataset", fake_evaluate_dataset)
    monkeypatch.setattr(cli_module, "write_evaluation_report", fake_write_evaluation_report)

    exit_code = cli_module.main(
        [
            "--dataset",
            str(dataset_path),
            "--model",
            "test:model",
            "--name",
            "baseline-deepseek",
            "--output-dir",
            str(output_dir),
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
    received_dataset, settings, name, repeat, metadata = evaluation_calls[0]
    assert received_dataset is dataset
    assert settings.model_name == "test:model"
    assert settings.run.runs_dir == output_dir / "runs"
    assert name == "baseline-deepseek"
    assert repeat == 3
    assert metadata["model_name"] == "test:model"
    assert metadata["dataset_name"] == "baseline-v1"
    assert metadata["repeat"] == 3
    assert metadata["dataset_sha256"]
    assert persisted_reports == [(output_dir / "report.json", report)]
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
    dataset_path = tmp_path / "baseline.yaml"
    dataset_path.write_text("name: baseline-v1\ncases: []\n", encoding="utf-8")
    output_dir = tmp_path / "experiment"
    persisted_reports: list[FakeReport] = []

    class FakeDatasetType:
        @classmethod
        def from_file(cls, path: Path) -> FakeDataset:
            return FakeDataset()

    async def fake_evaluate_dataset(*args: object, **kwargs: object) -> FakeReport:
        return report

    monkeypatch.setattr(cli_module, "load_formalizer_dataset", FakeDatasetType.from_file)
    monkeypatch.setattr(cli_module, "evaluate_dataset", fake_evaluate_dataset)
    monkeypatch.setattr(
        cli_module,
        "write_evaluation_report",
        lambda path, received_report: persisted_reports.append(received_report),
    )

    exit_code = cli_module.main(
        [
            "--dataset",
            str(dataset_path),
            "--model",
            "test:model",
            "--name",
            "baseline-deepseek",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 1
    assert persisted_reports == [report]
    assert report.printed


def test_eval_cli_handles_interruption(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "baseline.yaml"
    dataset_path.write_text("name: baseline-v1\ncases: []\n", encoding="utf-8")
    output_dir = tmp_path / "experiment"

    class FakeDatasetType:
        @classmethod
        def from_file(cls, path: Path) -> FakeDataset:
            return FakeDataset()

    async def interrupt(*args: object, **kwargs: object) -> FakeReport:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_module, "load_formalizer_dataset", FakeDatasetType.from_file)
    monkeypatch.setattr(cli_module, "evaluate_dataset", interrupt)

    exit_code = cli_module.main(
        [
            "--dataset",
            str(dataset_path),
            "--model",
            "test:model",
            "--name",
            "baseline-deepseek",
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 130
    assert "interrupted" in captured.err.lower()
    assert not (output_dir / "report.json").exists()


def test_eval_cli_rejects_nonpositive_repeat() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(
            [
                "--dataset",
                "baseline.yaml",
                "--model",
                "test:model",
                "--name",
                "baseline-deepseek",
                "--output-dir",
                "experiment",
                "--repeat",
                "0",
            ]
        )

    assert exc_info.value.code == 2


def test_eval_cli_refuses_to_reuse_an_experiment_directory(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "experiment"
    output_dir.mkdir()

    exit_code = cli_module.main(
        [
            "--dataset",
            str(tmp_path / "baseline.yaml"),
            "--model",
            "test:model",
            "--name",
            "baseline-deepseek",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 1
    assert "already exists" in capsys.readouterr().err


def test_experiment_metadata_records_reproducibility_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "baseline.yaml"
    dataset_bytes = b"name: baseline-v1\ncases: []\n"
    dataset_path.write_bytes(dataset_bytes)
    settings = Settings(model_name="test:model")
    monkeypatch.setattr(cli_module, "_git_provenance", lambda: ("abc123", True))
    monkeypatch.setattr(cli_module, "_docker_image_id", lambda image: "sha256:def456")

    metadata = cli_module._experiment_metadata(
        dataset_path=dataset_path,
        dataset_name="baseline-v1",
        settings=settings,
        repeat=3,
    )

    assert metadata["model_name"] == "test:model"
    assert metadata["dataset_name"] == "baseline-v1"
    assert metadata["dataset_path"] == str(dataset_path)
    assert metadata["dataset_sha256"] == sha256(dataset_bytes).hexdigest()
    assert metadata["repeat"] == 3
    started_at = datetime.fromisoformat(str(metadata["started_at"]))
    assert started_at.tzinfo is not None
    assert metadata["formalizer_version"]
    assert metadata["pydantic_evals_version"]
    assert metadata["logfire_version"]
    assert metadata["formalizer_git_commit"] == "abc123"
    assert metadata["formalizer_git_dirty"] is True
    assert metadata["docker_image"] == settings.sandbox.docker_image
    assert metadata["docker_image_id"] == "sha256:def456"
