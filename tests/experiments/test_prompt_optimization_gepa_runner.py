from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.exceptions import ModelHTTPError

PROMPT_OPTIMIZATION_DIR = Path("experiments/prompt_optimization").resolve()
MODULE_PATH = PROMPT_OPTIMIZATION_DIR / "gepa_runner.py"
sys.path.insert(0, str(PROMPT_OPTIMIZATION_DIR))
MODULE_SPEC = importlib.util.spec_from_file_location("formalizer_gepa_runner", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
runner = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = runner
MODULE_SPEC.loader.exec_module(runner)


def _config() -> Any:
    return runner._load_config(
        Path("experiments/prompt_optimization/gepa_overnight_v1.json").resolve()
    )


def _log_root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    root.mkdir()
    (root / "candidates").mkdir()
    (root / "task-runs").mkdir()
    return root


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_frozen_config_validates_train_only_disjoint_panel_and_large_context() -> None:
    config = _config()
    _, _, _, discovery, selection = runner._validate_inputs(config)

    assert len(discovery) == 9
    assert len(selection) == 5
    assert {example["case_name"] for example in discovery}.isdisjoint(
        example["case_name"] for example in selection
    )
    assert {example["feedback_mode"] for example in discovery} == {"discovery"}
    assert {example["feedback_mode"] for example in selection} == {"selection"}
    assert config.task_model.endpoint == "http://127.0.0.1:8000/v1"
    assert config.task_model.minimum_context_tokens == 262_144
    assert config.execution.request_limit == 200


def test_execute_requires_separate_model_call_approval_before_creating_output(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "must-not-exist"

    exit_code = runner.main(
        [
            "--phase",
            "canary",
            "--output-dir",
            str(output_dir),
            "--execute",
        ]
    )

    assert exit_code == 1
    assert not output_dir.exists()


def test_dry_run_checks_server_but_makes_no_output_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "dry-run"
    metadata = {"data": [{"id": "qwen-3.6-35b-a3b", "meta": {"n_ctx": 262_144}}]}
    monkeypatch.setattr(runner, "_server_metadata", lambda _config: metadata)

    exit_code = runner.main(["--phase", "optimize", "--output-dir", str(output_dir)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["model_calls"] == 0
    assert payload["server_metadata"] == metadata
    assert payload["execute_requires"] == ["--execute", "--approve-model-calls"]
    assert not output_dir.exists()


def test_discovery_evaluator_injects_candidate_and_returns_transcript_feedback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _log_root(tmp_path)
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    seen: dict[str, object] = {}

    async def fake_formalize(
        problem: object,
        settings: Any,
        *,
        run_id: object,
        instructions: str,
    ) -> object:
        seen.update(problem=problem, instructions=instructions)
        run_dir = settings.run.runs_dir / str(run_id)
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "started_at": "2026-09-01T10:00:00+00:00",
                    "finished_at": "2026-09-01T10:01:00+00:00",
                    "error_type": "UsageLimitExceeded",
                    "error": "Exceeded the total_tokens_limit",
                    "usage": {"input_tokens": 123, "output_tokens": 45},
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "messages.json").write_text("[]", encoding="utf-8")
        return SimpleNamespace(output=SimpleNamespace(check=SimpleNamespace(accepted=False)))

    monkeypatch.setattr(runner, "formalize", fake_formalize)
    log = runner.ExperimentLog(root, "test-run", "optimize")
    candidate = "candidate experiment instructions"
    source = "import Mathlib\n theorem panel_problem : True := by trivial"

    score, side_info = runner.FormalizerEvaluator(_config(), log)(
        candidate,
        {
            "case_name": "basic-problems/problem-test",
            "source": source,
            "feedback_mode": "discovery",
        },
    )

    assert score == 0.0
    assert seen["instructions"] == candidate
    assert side_info["problem"] == {"source_excerpt": source}
    assert "trajectory" in side_info
    assert "feedback_withheld" not in side_info
    digest = runner._sha256_bytes(candidate.encode())
    assert (root / "candidates" / f"{digest}.txt").read_text(encoding="utf-8") == candidate
    records = _jsonl(root / "evaluations.jsonl")
    assert [record["event"] for record in records] == ["started", "finished"]
    assert records[-1]["formalizer_artifacts_complete"] is True
    assert records[-1]["feedback_sha256"]
    assert records[-1]["optimizer_side_info"] == side_info


def test_selection_evaluator_never_parses_transcript(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _log_root(tmp_path)
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)

    async def fake_formalize(
        _problem: object,
        settings: Any,
        *,
        run_id: object,
        instructions: str,
    ) -> object:
        assert instructions == "candidate"
        run_dir = settings.run.runs_dir / str(run_id)
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps({"status": "verified", "usage": {}}), encoding="utf-8"
        )
        (run_dir / "messages.json").write_text("not valid JSON", encoding="utf-8")
        return SimpleNamespace(output=SimpleNamespace(check=SimpleNamespace(accepted=True)))

    monkeypatch.setattr(runner, "formalize", fake_formalize)
    score, side_info = runner.FormalizerEvaluator(
        _config(), runner.ExperimentLog(root, "test-run", "optimize")
    )(
        "candidate",
        {
            "case_name": "basic-problems/problem-selection",
            "source": "theorem hidden : True := by trivial",
            "feedback_mode": "selection",
        },
    )

    assert score == 1.0
    assert side_info == {
        "schema_version": 1,
        "case_id": "basic-problems/problem-selection",
        "outcome": "verified",
        "scores": {"verified": 1.0},
        "feedback_withheld": True,
    }


def test_retryable_infrastructure_attempts_are_all_logged_even_without_transcripts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _log_root(tmp_path)
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    calls = 0

    async def unavailable(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise ModelHTTPError(503, "test-model", {"error": "server unavailable"})

    monkeypatch.setattr(runner, "formalize", unavailable)
    evaluator = runner.FormalizerEvaluator(
        _config(), runner.ExperimentLog(root, "test-run", "optimize")
    )

    with pytest.raises(runner.InfrastructureInvalidEvaluation, match=r"after 3 attempt\(s\)"):
        evaluator(
            "candidate",
            {
                "case_name": "basic-problems/problem-test",
                "source": "theorem test : True := by trivial",
                "feedback_mode": "discovery",
            },
        )

    assert calls == 3
    records = _jsonl(root / "evaluations.jsonl")
    finished = [record for record in records if record["event"] == "finished"]
    assert len(finished) == 3
    assert all(record["terminal_status"] == "infrastructure_invalid" for record in finished)
    assert all(record["formalizer_artifacts_complete"] is False for record in finished)
    assert all(record["missing_artifacts"] == ["run.json", "messages.json"] for record in finished)


def test_http_failure_classification_distinguishes_context_from_invalid_endpoint() -> None:
    context = ModelHTTPError(
        400,
        "test-model",
        {"error": "requested tokens exceed the available context window"},
    )
    unauthorized = ModelHTTPError(401, "test-model", {"error": "unauthorized"})

    assert runner._classify_exception(context) == ("model_failed", False)
    assert runner._classify_exception(unauthorized) == ("infrastructure_invalid", False)


def test_reflection_logger_pairs_exact_prompt_output_and_usage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _log_root(tmp_path)
    log = runner.ExperimentLog(root, "test-run", "optimize")
    reflection = runner.LoggedReflectionLM(_config().reflection_model, log)

    class FakeLM:
        total_tokens_in = 10
        total_tokens_out = 4
        total_cost = 0.0

        def __call__(self, prompt: object) -> str:
            assert prompt == [{"role": "user", "content": "inspect exact transcript feedback"}]
            self.total_tokens_in += 25
            self.total_tokens_out += 7
            return "revised candidate"

    monkeypatch.setattr(reflection, "inner", FakeLM())
    prompt = [{"role": "user", "content": "inspect exact transcript feedback"}]

    assert reflection(prompt) == "revised candidate"

    records = _jsonl(root / "reflections.jsonl")
    assert [record["event"] for record in records] == ["started", "finished"]
    assert records[0]["prompt"] == prompt
    assert records[1]["output"] == "revised candidate"
    assert records[1]["usage_delta"] == {
        "input_tokens": 25,
        "output_tokens": 7,
        "cost_usd": 0.0,
    }
