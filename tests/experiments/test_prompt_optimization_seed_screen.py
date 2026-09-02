import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from formalizer.agent import _INSTRUCTIONS
from formalizer.eval import load_formalizer_dataset

CONFIG_PATH = Path("experiments/prompt_optimization/screening_panel_v1.json").resolve()
SCREEN_CONFIG_PATH = Path(
    "experiments/prompt_optimization/screening_panel_screen_v1.json"
).resolve()
SCREEN_V2_CONFIG_PATH = Path(
    "experiments/prompt_optimization/screening_panel_screen_v2.json"
).resolve()
PANEL_SPLIT_PATH = Path("experiments/prompt_optimization/panel_split_v1.json").resolve()
MODULE_PATH = Path("experiments/prompt_optimization/seed_screen.py").resolve()
MODULE_SPEC = importlib.util.spec_from_file_location("formalizer_seed_screen", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
seed_screen = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = seed_screen
MODULE_SPEC.loader.exec_module(seed_screen)


def load_validated_config() -> tuple[seed_screen.ScreeningConfig, Path, Path]:
    config = seed_screen._load_config(CONFIG_PATH)
    dataset_path, prompt_path = seed_screen._validate_inputs(config)
    return config, dataset_path, prompt_path


def test_screening_panel_is_frozen_to_unique_training_cases() -> None:
    config, dataset_path, _ = load_validated_config()
    dataset = load_formalizer_dataset(dataset_path)
    dataset_cases = {case.name: case for case in dataset.cases}

    names = [case.case_name for case in config.panel]

    assert len(names) == len(set(names)) == 16
    for name in names:
        metadata = dataset_cases[name].metadata
        assert metadata is not None
        assert metadata.split == "train"
    assert sum(case.provisional_role == "smoke-canary" for case in config.panel) == 1


def test_seed_snapshot_is_the_exact_production_default() -> None:
    config, _, prompt_path = load_validated_config()

    prompt = prompt_path.read_text(encoding="utf-8")

    assert prompt == _INSTRUCTIONS
    assert seed_screen._sha256_bytes(prompt.encode()) == config.seed_prompt.sha256


def test_canary_and_screen_phases_are_disjoint_and_complete() -> None:
    config, _, _ = load_validated_config()

    canary = seed_screen._cases_for_phase(config, "canary")
    screen = seed_screen._cases_for_phase(config, "screen")
    all_cases = seed_screen._cases_for_phase(config, "all")

    assert [case.case_name for case in canary] == ["basic-problems/problem-13"]
    assert len(screen) == 15
    assert {case.case_name for case in canary}.isdisjoint(case.case_name for case in screen)
    assert set(all_cases) == set(canary) | set(screen)


def test_budget_ceiling_is_explicit_for_each_phase() -> None:
    config, _, _ = load_validated_config()

    assert seed_screen._budget_ceiling(config, 1) == {
        "selected_attempts": 1,
        "model_requests": 50,
        "output_tokens": 50_000,
        "total_tokens": 1_500_000,
        "wall_time_s": 1_800,
    }
    assert seed_screen._budget_ceiling(config, 15) == {
        "selected_attempts": 15,
        "model_requests": 750,
        "output_tokens": 750_000,
        "total_tokens": 22_500_000,
        "wall_time_s": 27_000,
    }


def test_request_limited_pilot_config_has_tighter_budgets_and_completed_parent() -> None:
    config = seed_screen._load_config(SCREEN_CONFIG_PATH)
    seed_screen._validate_inputs(config)

    assert config.parent_run is not None
    assert config.parent_run.run_id == "37011fc3-6ac0-4cba-8781-9ba7214b8deb"
    assert config.execution.request_limit == 30
    assert config.execution.output_tokens_limit == 50_000
    assert config.execution.total_tokens_limit == 1_000_000
    assert seed_screen._budget_ceiling(config, 15) == {
        "selected_attempts": 15,
        "model_requests": 450,
        "output_tokens": 750_000,
        "total_tokens": 15_000_000,
        "wall_time_s": 27_000,
    }


def test_approved_screen_v2_uses_tokens_as_the_controlling_budget() -> None:
    config = seed_screen._load_config(SCREEN_V2_CONFIG_PATH)
    seed_screen._validate_inputs(config)

    assert config.parent_run is not None
    assert config.parent_run.run_id == "37011fc3-6ac0-4cba-8781-9ba7214b8deb"
    assert config.execution.request_limit == 200
    assert config.execution.output_tokens_limit == 50_000
    assert config.execution.total_tokens_limit == 1_000_000
    assert seed_screen._budget_ceiling(config, 15) == {
        "selected_attempts": 15,
        "model_requests": 3_000,
        "output_tokens": 750_000,
        "total_tokens": 15_000_000,
        "wall_time_s": 27_000,
    }


def test_gepa_panel_split_is_disjoint_complete_and_train_only() -> None:
    split = json.loads(PANEL_SPLIT_PATH.read_text(encoding="utf-8"))
    dataset = load_formalizer_dataset(Path(split["dataset"]["path"]))
    dataset_cases = {case.name: case for case in dataset.cases}
    discovery = split["discovery_cases"]
    selection = split["selection_cases"]
    excluded = [case["case_name"] for case in split["excluded_cases"]]

    assert len(discovery) == len(set(discovery)) == 9
    assert len(selection) == len(set(selection)) == 5
    assert set(discovery).isdisjoint(selection)
    assert set(discovery).isdisjoint(excluded)
    assert set(selection).isdisjoint(excluded)
    assert len(set(discovery) | set(selection) | set(excluded)) == 15
    for name in discovery + selection:
        metadata = dataset_cases[name].metadata
        assert metadata is not None
        assert metadata.split == "train"


def test_gepa_panel_split_is_pinned_to_completed_screen() -> None:
    split = json.loads(PANEL_SPLIT_PATH.read_text(encoding="utf-8"))
    source = split["source_screen"]
    source_root = Path(source["path"])

    assert source["run_id"] == "ec78d996-109d-45d7-b6fa-94b0b8f42e05"
    assert seed_screen._sha256_file(source_root / "manifest.json") == source["manifest_sha256"]
    assert seed_screen._sha256_file(source_root / "status.json") == source["status_sha256"]
    assert seed_screen._sha256_file(source_root / "result.json") == source["result_sha256"]
    assert (
        seed_screen._sha256_file(source_root / "evaluations.jsonl") == source["evaluations_sha256"]
    )
    assert json.loads((source_root / "status.json").read_text())["status"] == (
        "completed_with_invalid"
    )


def test_eval_command_uses_chat_completions_and_exact_case() -> None:
    config, dataset_path, _ = load_validated_config()
    panel_case = seed_screen._cases_for_phase(config, "canary")[0]

    command = seed_screen._eval_command(
        config,
        dataset_path=dataset_path,
        panel_case=panel_case,
        output_dir=Path("unused-output"),
    )

    assert "openai-chat:unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL" in command
    assert command[command.index("--case") + 1] == "basic-problems/problem-13"
    assert command[command.index("--max-concurrency") + 1] == "1"
    assert command[command.index("--repeat") + 1] == "1"
    assert "--split" not in command


def test_default_invocation_is_a_non_mutating_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = seed_screen.main(["--config", str(CONFIG_PATH), "--phase", "canary"])

    assert exit_code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert plan["case_names"] == ["basic-problems/problem-13"]
    assert plan["execute_requires"] == [
        "--execute",
        "--approve-model-calls",
        "--output-dir",
    ]


def test_execute_requires_explicit_model_call_approval(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = seed_screen.RUNS_ROOT / tmp_path.name

    exit_code = seed_screen.main(
        [
            "--config",
            str(CONFIG_PATH),
            "--phase",
            "canary",
            "--execute",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 1
    assert not output_dir.exists()
    assert "--approve-model-calls" in capsys.readouterr().err


def test_output_directory_must_stay_under_prompt_optimization_runs(tmp_path: Path) -> None:
    config, dataset_path, prompt_path = load_validated_config()

    with pytest.raises(ValueError, match="output directory must be under"):
        seed_screen._execute(
            config,
            config_path=CONFIG_PATH,
            dataset_path=dataset_path,
            prompt_path=prompt_path,
            phase="canary",
            output_dir=tmp_path / "outside",
        )


def test_execute_writes_complete_audit_artifacts_without_real_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, dataset_path, prompt_path = load_validated_config()
    output_dir = tmp_path / "canary-run"
    monkeypatch.setattr(seed_screen, "RUNS_ROOT", tmp_path)
    monkeypatch.setattr(seed_screen, "_server_metadata", lambda _config: {"data": []})
    monkeypatch.setattr(
        seed_screen,
        "_git_provenance",
        lambda _paths: {
            "commit": "test-commit",
            "is_dirty": False,
            "status_sha256": "status-hash",
            "diff_sha256": "diff-hash",
            "file_sha256s": {},
        },
    )

    real_run = seed_screen.subprocess.run

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        if "--output-dir" not in command:
            return real_run(command, **kwargs)
        child_output = Path(command[command.index("--output-dir") + 1])
        child_output.mkdir(parents=True)
        (child_output / "report.json").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(seed_screen.subprocess, "run", fake_run)
    monkeypatch.setattr(
        seed_screen,
        "_classify_report",
        lambda _path: {
            "terminal_status": "verified",
            "score": 1.0,
            "verified": True,
            "infrastructure_invalid": False,
            "failure_type": None,
            "error": None,
            "task_run_id": "test-run-id",
            "task_run_path": "task-runs/test-run-id",
            "all_run_manifest_paths": ["task-runs/test-run-id/run.json"],
            "usage": {"model_responses": 1},
            "duration_s": 1.25,
        },
    )

    exit_code = seed_screen._execute(
        config,
        config_path=CONFIG_PATH,
        dataset_path=dataset_path,
        prompt_path=prompt_path,
        phase="canary",
        output_dir=output_dir,
    )

    assert exit_code == 0
    assert {
        "candidates",
        "environment.json",
        "evaluations.jsonl",
        "events.jsonl",
        "manifest.json",
        "result.json",
        "status.json",
        "task-runs",
    } <= {path.name for path in output_dir.iterdir()}
    evaluation_events = [
        json.loads(line) for line in (output_dir / "evaluations.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in evaluation_events] == ["started", "finished"]
    assert evaluation_events[-1]["terminal_status"] == "verified"
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["dataset"]["selected_cases"] == ["basic-problems/problem-13"]
    assert manifest["task_model"]["endpoint"] == "http://logos:8080/v1"
    assert manifest["gepa"] == {"enabled": False}
