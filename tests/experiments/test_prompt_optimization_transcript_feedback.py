from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

MODULE_PATH = Path("experiments/prompt_optimization/transcript_feedback.py").resolve()
MODULE_SPEC = importlib.util.spec_from_file_location("formalizer_transcript_feedback", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
feedback = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = feedback
MODULE_SPEC.loader.exec_module(feedback)


def _tool_call(name: str, call_id: str, args: Mapping[str, object]) -> dict[str, object]:
    return {
        "part_kind": "tool-call",
        "tool_name": name,
        "tool_call_id": call_id,
        "args": dict(args),
    }


def _tool_return(name: str, call_id: str, content: Mapping[str, object]) -> dict[str, object]:
    return {
        "part_kind": "tool-return",
        "tool_name": name,
        "tool_call_id": call_id,
        "content": dict(content),
    }


def _messages(*parts: Mapping[str, object]) -> list[dict[str, object]]:
    return [{"parts": [dict(part) for part in parts]}]


def _failed_run(error: str = "Exceeded the total_tokens_limit") -> dict[str, object]:
    return {
        "status": "failed",
        "error_type": "UsageLimitExceeded",
        "error": error,
        "started_at": "2026-09-01T10:00:00+00:00",
        "finished_at": "2026-09-01T10:01:00+00:00",
        "usage": {"model_responses": 12, "input_tokens": 34_000, "output_tokens": 5_000},
    }


def test_discovery_feedback_is_deterministic_bounded_and_redacted() -> None:
    secret = "sk-this-secret-must-not-survive"
    diagnostic = (
        "/workspace/Main.lean:42:7: error: Unknown constant `Invented.theorem`\n"
        f"authorization: Bearer abc.def.ghi api_key={secret}\n" + "large diagnostic context " * 300
    )
    messages = _messages(
        {"part_kind": "thinking", "content": "Try a different approach. PRIVATE-REASONING"},
        _tool_call("mathlib_search", "search-1", {"query": "Invented.theorem"}),
        _tool_return(
            "mathlib_search",
            "search-1",
            {
                "error": f"unknown identifier Invented.theorem Bearer token-value {secret}",
                "total_count": 0,
                "hits": [],
            },
        ),
        _tool_call(
            "lean_execute",
            "lean-1",
            {"code": "import FormalizerProblem\nexample : True := by trivial"},
        ),
        _tool_return(
            "lean_execute",
            "lean-1",
            {"exit_code": 1, "timed_out": False, "stdout": diagnostic, "stderr": ""},
        ),
    )

    first_score, first = feedback.build_side_info(
        case_id="basic-problems/problem-test",
        outcome="model_failed",
        run_record=_failed_run(),
        messages=messages,
        mode="discovery",
        artifact_path="runs/prompt-optimization/example/run",
        artifact_hashes={"run_sha256": "a" * 64, "messages_sha256": "b" * 64},
        char_limit=3_000,
    )
    second_score, second = feedback.build_side_info(
        case_id="basic-problems/problem-test",
        outcome="model_failed",
        run_record=_failed_run(),
        messages=messages,
        mode="discovery",
        artifact_path="runs/prompt-optimization/example/run",
        artifact_hashes={"run_sha256": "a" * 64, "messages_sha256": "b" * 64},
        char_limit=3_000,
    )

    serialized = json.dumps(first, sort_keys=True)
    assert first_score == second_score == 0.0
    assert first == second
    assert feedback.reflection_feedback_character_count(first) <= 3_000
    assert secret not in serialized
    assert "abc.def.ghi" not in serialized
    assert "PRIVATE-REASONING" not in serialized
    assert "different approach 1 time" in first["feedback"]
    assert "Main.lean:<line>:<col>" in serialized


def test_placeholder_compilation_is_not_counted_as_a_valid_lean_success() -> None:
    messages = _messages(
        _tool_call(
            "lean_execute",
            "lean-1",
            {"code": "import FormalizerProblem\nexample : True := by sorry"},
        ),
        _tool_return(
            "lean_execute",
            "lean-1",
            {
                "exit_code": 0,
                "timed_out": False,
                "stdout": "warning: declaration uses sorry",
                "verification_error": "Solution depends on disallowed axioms: sorryAx",
            },
        ),
    )

    _, side_info = feedback.build_side_info(
        case_id="basic-problems/problem-test",
        outcome="model_failed",
        run_record=_failed_run(),
        messages=messages,
        mode="discovery",
    )

    lean = side_info["trajectory"]["lean"]
    assert lean["placeholder_compilations"] == 1
    assert lean["placeholder_free_successes"] == 0
    assert lean["failed_checks"] == 1
    assert "prohibited_placeholder" in side_info["failure"]["secondary"]


class _ExplodingMessages(Sequence[Mapping[str, Any]]):
    def __getitem__(self, index: int) -> Mapping[str, Any]:
        raise AssertionError(f"selection transcript was accessed at {index}")

    def __len__(self) -> int:
        raise AssertionError("selection transcript length was accessed")

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        raise AssertionError("selection transcript was iterated")


def test_selection_feedback_is_score_only_and_does_not_access_messages() -> None:
    score, side_info = feedback.build_side_info(
        case_id="basic-problems/problem-selection",
        outcome="model_failed",
        run_record=_failed_run(),
        messages=_ExplodingMessages(),
        mode="selection",
        artifact_path="must-not-leak",
        artifact_hashes={"messages_sha256": "must-not-leak"},
    )

    assert score == 0.0
    assert side_info == {
        "schema_version": 1,
        "case_id": "basic-problems/problem-selection",
        "outcome": "model_failed",
        "scores": {"verified": 0.0},
        "feedback_withheld": True,
    }


def test_infrastructure_invalid_evaluation_raises_instead_of_scoring_zero() -> None:
    with pytest.raises(feedback.InfrastructureInvalidEvaluation, match="not score zero"):
        feedback.build_side_info(
            case_id="basic-problems/problem-invalid",
            outcome="infrastructure_invalid",
            run_record={"error": "connection reset"},
            messages=[],
            mode="discovery",
        )


def test_recurring_rewrite_failures_are_classified_as_nonconvergence() -> None:
    parts: list[dict[str, object]] = []
    for index in range(4):
        call_id = f"lean-{index}"
        parts.extend(
            [
                _tool_call(
                    "lean_execute",
                    call_id,
                    {"code": "import FormalizerProblem\nexample : True := by trivial"},
                ),
                _tool_return(
                    "lean_execute",
                    call_id,
                    {
                        "exit_code": 1,
                        "timed_out": False,
                        "stdout": (
                            f"/workspace/Main.lean:{index + 1}:2: error: "
                            "Tactic `rewrite` failed: pattern not found\n"
                            f"/workspace/Main.lean:{index + 2}:2: error: unsolved goals\n"
                            "⊢ x * y = y * x"
                        ),
                    },
                ),
            ]
        )

    _, side_info = feedback.build_side_info(
        case_id="basic-problems/problem-rewrite",
        outcome="model_failed",
        run_record=_failed_run(),
        messages=_messages(*parts),
        mode="discovery",
    )

    assert side_info["failure"]["primary"] == "rewrite_nonconvergence"
    repeated = side_info["trajectory"]["lean"]["repeated_error_fingerprints"]
    assert repeated[0]["count"] == 4
    assert side_info["failure"]["terminal_reason"] == "token_exhaustion"


def test_verified_score_is_not_reduced_by_search_inefficiency() -> None:
    messages = _messages(
        _tool_call("mathlib_search", "search-1", {"query": "bad_name"}),
        _tool_return(
            "mathlib_search",
            "search-1",
            {"error": "unknown identifier bad_name", "total_count": 0},
        ),
        _tool_call("final_submission", "final-1", {"code": "verified code"}),
        _tool_return("final_submission", "final-1", {"message": "Final result processed."}),
    )

    score, side_info = feedback.build_side_info(
        case_id="basic-problems/problem-verified",
        outcome="verified",
        run_record={"status": "verified", "usage": {}},
        messages=messages,
        mode="discovery",
    )

    assert score == 1.0
    assert side_info["scores"] == {"verified": 1.0}
    assert side_info["failure"]["primary"] == "verified"
    assert "search_protocol_error" in side_info["failure"]["secondary"]


def test_feedback_limit_has_a_safe_minimum() -> None:
    with pytest.raises(ValueError, match="at least"):
        feedback.build_side_info(
            case_id="basic-problems/problem-test",
            outcome="model_failed",
            run_record=_failed_run(),
            messages=[],
            mode="discovery",
            char_limit=500,
        )


def test_failed_evaluation_can_resolve_its_single_recorded_run_manifest(tmp_path: Path) -> None:
    source_root = tmp_path / "source-screen"
    task_root = source_root / "task-runs/problem/evaluation/runs/run-id"
    task_root.mkdir(parents=True)
    run_path = task_root / "run.json"
    messages_path = task_root / "messages.json"
    run_path.write_text(json.dumps(_failed_run()), encoding="utf-8")
    messages_path.write_text("[]", encoding="utf-8")

    run_record, messages, resolved_root, hashes = feedback._load_discovery_task_run(
        source_root=source_root,
        case_name="basic-problems/problem-failed",
        evaluation={"task_run_path": None, "all_run_manifest_paths": [str(run_path)]},
    )

    assert run_record["status"] == "failed"
    assert messages == []
    assert resolved_root == task_root.resolve()
    assert set(hashes) == {"run_sha256", "messages_sha256"}
