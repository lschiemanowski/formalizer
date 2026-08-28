from datetime import datetime
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import formalizer.eval.cli as cli_module
from formalizer.eval import LeanVerified
from formalizer.settings import Settings


class FakeDataset:
    def __init__(self) -> None:
        self.name = "baseline-v1"
        self.cases = [FakeDatasetCase("logic/example")]
        self.evaluators: list[object] = []

    def add_evaluator(self, evaluator: object) -> None:
        self.evaluators.append(evaluator)


class FakeDatasetCase:
    def __init__(self, name: str) -> None:
        self.name = name


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
        self.include_averages: bool | None = None

    def print(self, *, include_averages: bool = True) -> None:
        self.printed = True
        self.include_averages = include_averages


class FakeCase:
    def __init__(self, evaluator_failures: list[object]) -> None:
        self.evaluator_failures = evaluator_failures
        self.assertions = (
            {} if evaluator_failures else {"LeanVerified": SimpleNamespace(value=True)}
        )


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
    evaluation_calls: list[tuple[FakeDataset, Settings, str, int, int, int, dict[str, object]]] = []
    persisted_reports: list[tuple[Path, FakeReport]] = []
    printed_summaries: list[FakeReport] = []

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
        max_concurrency: int,
        infrastructure_retries: int,
        metadata: dict[str, object],
    ) -> FakeReport:
        evaluation_calls.append(
            (
                received_dataset,
                settings,
                name,
                repeat,
                max_concurrency,
                infrastructure_retries,
                metadata,
            )
        )
        return report

    def fake_write_evaluation_report(path: Path, received_report: FakeReport) -> None:
        persisted_reports.append((path, received_report))

    monkeypatch.setattr(cli_module, "load_formalizer_dataset", FakeDatasetType.from_file)
    monkeypatch.setattr(cli_module, "configure_logfire", fake_configure_logfire)
    monkeypatch.setattr(cli_module, "evaluate_dataset", fake_evaluate_dataset)
    monkeypatch.setattr(cli_module, "write_evaluation_report", fake_write_evaluation_report)
    monkeypatch.setattr(
        cli_module,
        "print_outcome_summary",
        lambda received_report: printed_summaries.append(received_report),
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
    (
        received_dataset,
        settings,
        name,
        repeat,
        max_concurrency,
        infrastructure_retries,
        metadata,
    ) = evaluation_calls[0]
    assert received_dataset is dataset
    assert settings.model_name == "test:model"
    assert settings.model_settings == {"max_tokens": 8192}
    assert settings.run.runs_dir == output_dir / "runs"
    assert settings.run.usage_limits.request_limit == 30
    assert settings.run.usage_limits.output_tokens_limit == 50_000
    assert settings.run.usage_limits.total_tokens_limit == 1_000_000
    assert name == "baseline-deepseek"
    assert repeat == 3
    assert max_concurrency == 1
    assert infrastructure_retries == 2
    assert metadata["model_name"] == "test:model"
    assert metadata["dataset_name"] == "baseline-v1"
    assert metadata["repeat"] == 3
    assert metadata["dataset_sha256"]
    assert metadata["selection"] == {
        "splits": [],
        "difficulties": [],
        "case_names": [],
        "selected_case_count": 1,
    }
    assert metadata["budget"] == {
        "max_tokens": 8192,
        "request_limit": 30,
        "output_tokens_limit": 50_000,
        "total_tokens_limit": 1_000_000,
    }
    assert metadata["model_settings"] == {
        "max_tokens": 8192,
        "temperature": None,
        "top_p": None,
        "thinking": None,
    }
    assert metadata["retry_policy"] == {"infrastructure_retries": 2}
    assert metadata["execution"] == {"max_concurrency": 1}
    assert persisted_reports == [(output_dir / "report.json", report)]
    assert report.printed
    assert report.include_averages is False
    assert printed_summaries == [report]


def test_eval_cli_applies_and_records_budget_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "baseline.yaml"
    dataset_path.write_text("name: baseline-v1\ncases: []\n", encoding="utf-8")
    output_dir = tmp_path / "experiment"
    dataset = FakeDataset()
    received: list[tuple[Settings, int, int, dict[str, object]]] = []

    async def fake_evaluate_dataset(
        received_dataset: FakeDataset,
        settings: Settings,
        *,
        name: str,
        repeat: int,
        max_concurrency: int,
        infrastructure_retries: int,
        metadata: dict[str, object],
    ) -> FakeReport:
        received.append((settings, max_concurrency, infrastructure_retries, metadata))
        return FakeReport()

    monkeypatch.setattr(cli_module, "load_formalizer_dataset", lambda path: dataset)
    monkeypatch.setattr(cli_module, "evaluate_dataset", fake_evaluate_dataset)
    monkeypatch.setattr(cli_module, "write_evaluation_report", lambda path, report: None)

    exit_code = cli_module.main(
        [
            "--dataset",
            str(dataset_path),
            "--model",
            "test:model",
            "--name",
            "bounded-run",
            "--output-dir",
            str(output_dir),
            "--max-tokens",
            "4096",
            "--request-limit",
            "12",
            "--output-tokens-limit",
            "20000",
            "--total-tokens-limit",
            "150000",
            "--infrastructure-retries",
            "0",
            "--max-concurrency",
            "3",
        ]
    )

    assert exit_code == 0
    settings, max_concurrency, infrastructure_retries, metadata = received[0]
    assert settings.model_settings == {"max_tokens": 4096}
    assert settings.run.usage_limits.request_limit == 12
    assert settings.run.usage_limits.output_tokens_limit == 20_000
    assert settings.run.usage_limits.total_tokens_limit == 150_000
    assert max_concurrency == 3
    assert infrastructure_retries == 0
    assert metadata["budget"] == {
        "max_tokens": 4096,
        "request_limit": 12,
        "output_tokens_limit": 20_000,
        "total_tokens_limit": 150_000,
    }
    assert metadata["retry_policy"] == {"infrastructure_retries": 0}
    assert metadata["execution"] == {"max_concurrency": 3}


def test_eval_cli_applies_and_records_thinking_and_sampling_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "baseline.yaml"
    dataset_path.write_text("name: baseline-v1\ncases: []\n", encoding="utf-8")
    output_dir = tmp_path / "experiment"
    dataset = FakeDataset()
    received: list[tuple[Settings, dict[str, object]]] = []

    async def fake_evaluate_dataset(
        received_dataset: FakeDataset,
        settings: Settings,
        **kwargs: object,
    ) -> FakeReport:
        assert received_dataset is dataset
        received.append((settings, cast(dict[str, object], kwargs["metadata"])))
        return FakeReport()

    monkeypatch.setattr(cli_module, "load_formalizer_dataset", lambda path: dataset)
    monkeypatch.setattr(cli_module, "evaluate_dataset", fake_evaluate_dataset)
    monkeypatch.setattr(cli_module, "write_evaluation_report", lambda path, report: None)

    exit_code = cli_module.main(
        [
            "--dataset",
            str(dataset_path),
            "--model",
            "openrouter:z-ai/glm-5.3-flash",
            "--name",
            "glm-low-thinking",
            "--output-dir",
            str(output_dir),
            "--thinking",
            "low",
            "--temperature",
            "0.6",
        ]
    )

    assert exit_code == 0
    settings, metadata = received[0]
    assert settings.model_settings == {
        "max_tokens": 8192,
        "temperature": 0.6,
        "thinking": "low",
    }
    assert metadata["model_settings"] == {
        "max_tokens": 8192,
        "temperature": 0.6,
        "top_p": None,
        "thinking": "low",
    }


def test_eval_cli_selects_cases_and_records_filters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "basic-problems.yaml"
    dataset_path.write_text("name: basic-problems-v1\ncases: []\n", encoding="utf-8")
    output_dir = tmp_path / "experiment"
    dataset = FakeDataset()
    report = FakeReport()
    selection_calls: list[tuple[set[str], set[str], set[str]]] = []
    recorded_metadata: list[dict[str, object]] = []

    def fake_select_formalizer_dataset(
        received_dataset: FakeDataset,
        *,
        splits: set[str],
        difficulties: set[str],
        case_names: set[str],
    ) -> FakeDataset:
        assert received_dataset is dataset
        selection_calls.append((splits, difficulties, case_names))
        received_dataset.cases = [FakeDatasetCase("basic-problems/problem-8")]
        return received_dataset

    async def fake_evaluate_dataset(
        received_dataset: FakeDataset,
        settings: Settings,
        *,
        name: str,
        repeat: int,
        max_concurrency: int,
        infrastructure_retries: int,
        metadata: dict[str, object],
    ) -> FakeReport:
        assert received_dataset is dataset
        assert max_concurrency == 1
        assert infrastructure_retries == 2
        recorded_metadata.append(metadata)
        return report

    monkeypatch.setattr(cli_module, "load_formalizer_dataset", lambda path: dataset)
    monkeypatch.setattr(
        cli_module,
        "select_formalizer_dataset",
        fake_select_formalizer_dataset,
    )
    monkeypatch.setattr(cli_module, "evaluate_dataset", fake_evaluate_dataset)
    monkeypatch.setattr(cli_module, "write_evaluation_report", lambda path, value: None)

    exit_code = cli_module.main(
        [
            "--dataset",
            str(dataset_path),
            "--model",
            "test:model",
            "--name",
            "easy-validation",
            "--output-dir",
            str(output_dir),
            "--split",
            "validation",
            "--difficulty",
            "easy",
            "--case",
            "basic-problems/problem-8",
        ]
    )

    assert exit_code == 0
    assert selection_calls == [
        (
            {"validation"},
            {"easy"},
            {"basic-problems/problem-8"},
        )
    ]
    assert recorded_metadata[0]["selection"] == {
        "splits": ["validation"],
        "difficulties": ["easy"],
        "case_names": ["basic-problems/problem-8"],
        "selected_case_count": 1,
    }


def test_eval_cli_rejects_an_empty_selection_before_creating_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.yaml"
    dataset_path.write_text(
        """\
name: easy-only-v1
cases:
  - name: logic/true
    inputs:
      source: |
        namespace FormalizerProblem
        def Target : Prop := True
        end FormalizerProblem
    metadata:
      difficulty: easy
      split: validation
      domain: logic
      provenance:
        origin: original
        source: Test fixture
        source_id: logic/true
        license: Apache-2.0
""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "experiment"
    evaluation_called = False

    async def unexpected_evaluation(*args: object, **kwargs: object) -> FakeReport:
        nonlocal evaluation_called
        evaluation_called = True
        return FakeReport()

    monkeypatch.setattr(cli_module, "evaluate_dataset", unexpected_evaluation)

    exit_code = cli_module.main(
        [
            "--dataset",
            str(dataset_path),
            "--model",
            "test:model",
            "--name",
            "hard-validation",
            "--output-dir",
            str(output_dir),
            "--difficulty",
            "hard",
        ]
    )

    assert exit_code == 1
    assert "No cases match" in capsys.readouterr().err
    assert not evaluation_called
    assert not output_dir.exists()


@pytest.mark.parametrize(
    "report",
    [
        FakeReport(failures=[SimpleNamespace(error_message="RuntimeError: failed")]),
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


@pytest.mark.parametrize(
    "option",
    [
        "--max-tokens",
        "--request-limit",
        "--output-tokens-limit",
        "--total-tokens-limit",
    ],
)
def test_eval_cli_rejects_nonpositive_budget(option: str) -> None:
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
                option,
                "0",
            ]
        )

    assert exc_info.value.code == 2


def test_eval_cli_rejects_negative_infrastructure_retries() -> None:
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
                "--infrastructure-retries",
                "-1",
            ]
        )

    assert exc_info.value.code == 2


def test_eval_cli_rejects_nonpositive_max_concurrency() -> None:
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
                "--max-concurrency",
                "0",
            ]
        )

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--temperature", "-0.1"),
        ("--temperature", "inf"),
        ("--temperature", "nan"),
        ("--top-p", "0"),
        ("--top-p", "1.1"),
        ("--top-p", "nan"),
    ],
)
def test_eval_cli_rejects_invalid_sampling_settings(option: str, value: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(
            [
                "--dataset",
                "baseline.yaml",
                "--model",
                "test:model",
                "--name",
                "sampling-test",
                "--output-dir",
                "experiment",
                option,
                value,
            ]
        )

    assert exc_info.value.code == 2


def test_eval_cli_rejects_temperature_and_top_p_together() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(
            [
                "--dataset",
                "baseline.yaml",
                "--model",
                "test:model",
                "--name",
                "sampling-test",
                "--output-dir",
                "experiment",
                "--temperature",
                "0.6",
                "--top-p",
                "0.95",
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
        selection={
            "splits": ["validation"],
            "difficulties": ["easy"],
            "case_names": [],
            "selected_case_count": 3,
        },
        infrastructure_retries=2,
        max_concurrency=3,
    )

    assert metadata["model_name"] == "test:model"
    assert metadata["dataset_name"] == "baseline-v1"
    assert metadata["dataset_path"] == str(dataset_path)
    assert metadata["dataset_sha256"] == sha256(dataset_bytes).hexdigest()
    assert metadata["repeat"] == 3
    assert metadata["selection"] == {
        "splits": ["validation"],
        "difficulties": ["easy"],
        "case_names": [],
        "selected_case_count": 3,
    }
    started_at = datetime.fromisoformat(str(metadata["started_at"]))
    assert started_at.tzinfo is not None
    assert metadata["formalizer_version"]
    assert metadata["pydantic_evals_version"]
    assert metadata["logfire_version"]
    assert metadata["formalizer_git_commit"] == "abc123"
    assert metadata["formalizer_git_dirty"] is True
    assert metadata["docker_image"] == settings.sandbox.docker_image
    assert metadata["docker_image_id"] == "sha256:def456"
    assert metadata["budget"] == {
        "max_tokens": None,
        "request_limit": 50,
        "output_tokens_limit": None,
        "total_tokens_limit": None,
    }
    assert metadata["model_settings"] == {
        "max_tokens": None,
        "temperature": None,
        "top_p": None,
        "thinking": None,
    }
    assert metadata["retry_policy"] == {"infrastructure_retries": 2}
    assert metadata["execution"] == {"max_concurrency": 3}
