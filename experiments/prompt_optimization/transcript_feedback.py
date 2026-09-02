"""Distill allowed Formalizer transcripts into bounded GEPA side information.

This module is deliberately deterministic. Full messages remain in the task-run artifact;
GEPA receives only counts, normalized failure evidence, and a few bounded excerpts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from formalizer.eval import load_formalizer_dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPOSITORY_ROOT / "runs/prompt-optimization"
DEFAULT_PANEL_SPLIT = REPOSITORY_ROOT / "experiments/prompt_optimization/panel_split_v1.json"
DEFAULT_FEEDBACK_CHAR_LIMIT = 6_000
MINIMUM_FEEDBACK_CHAR_LIMIT = 1_200

Outcome = Literal["verified", "model_failed", "infrastructure_invalid"]
FeedbackMode = Literal["discovery", "selection"]

_PLACEHOLDER_RE = re.compile(r"(?m)^\s*(?:sorry|admit)\b|\bby\s+sorry\b")
_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;}]+"),
)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_LOCATION_RE = re.compile(r"(?:/[^\s:]+/)?(?P<file>[A-Za-z][^\s:]*\.lean):\d+:\d+")
_ERROR_HEADER_RE = re.compile(
    r"(?m)^(?:(?:/[^\n:]+/)?[^\n:]+\.lean:\d+:\d+:\s*)?"
    r"error(?:\([^)]*\))?:\s*(?P<head>[^\n]+)"
)
_SPACE_RE = re.compile(r"[ \t]+")
_METAVAR_RE = re.compile(r"\?m\.\d+")


class FeedbackIntegrityError(RuntimeError):
    """Raised when transcript feedback cannot be produced without breaking the contract."""


class InfrastructureInvalidEvaluation(FeedbackIntegrityError):
    """Raised so infrastructure-invalid attempts cannot silently become score zero."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def feedback_character_count(side_info: Mapping[str, object]) -> int:
    """Return the canonical compact-JSON size used for hashing and artifact comparison."""

    return len(_canonical_json(side_info))


def reflection_feedback_character_count(side_info: Mapping[str, object]) -> int:
    """Return the GEPA 0.1.4 Markdown size for one reflective-dataset record."""

    def render_value(value: object, level: int = 3) -> str:
        if isinstance(value, dict):
            rendered = ""
            for key, item in value.items():
                rendered += f"{'#' * level} {key}\n"
                rendered += render_value(item, min(level + 1, 6))
            if not value:
                rendered += "\n"
            return rendered
        if isinstance(value, list | tuple):
            rendered = ""
            for index, item in enumerate(value):
                rendered += f"{'#' * level} Item {index + 1}\n"
                rendered += render_value(item, min(level + 1, 6))
            if not value:
                rendered += "\n"
            return rendered
        return f"{str(value).strip()}\n\n"

    adapted = {
        "Scores (Higher is Better)" if key == "scores" else key: value
        for key, value in side_info.items()
    }
    rendered = "# Example 1\n"
    for key, value in adapted.items():
        rendered += f"## {key}\n"
        rendered += render_value(value)
    return len(rendered)


def _redact(text: str) -> str:
    redacted = _ANSI_RE.sub("", text)
    for pattern in _SECRET_PATTERNS:
        if pattern.pattern.lower().startswith("(?i)(api"):
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _trim(text: str, limit: int) -> str:
    text = _redact(text).strip()
    if len(text) <= limit:
        return text
    if limit <= 1:
        return "…"[:limit]
    return text[: limit - 1].rstrip() + "…"


def _as_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _message_parts(messages: Sequence[Mapping[str, object]]) -> Iterable[dict[str, Any]]:
    for message in messages:
        parts = message.get("parts", [])
        if not isinstance(parts, list):
            continue
        for part in parts:
            if isinstance(part, dict):
                yield part


def _duration_seconds(run_record: Mapping[str, object]) -> float | None:
    started = run_record.get("started_at")
    finished = run_record.get("finished_at")
    if not isinstance(started, str) or not isinstance(finished, str):
        return None
    try:
        start_dt = datetime.fromisoformat(started)
        finish_dt = datetime.fromisoformat(finished)
    except ValueError:
        return None
    return round((finish_dt - start_dt).total_seconds(), 6)


def _normalize_location(text: str) -> str:
    return _LOCATION_RE.sub(lambda match: f"{Path(match.group('file')).name}:<line>:<col>", text)


def _error_blocks(text: str) -> list[tuple[str, str]]:
    """Extract normalized Lean error headlines and short blocks, omitting warning preambles."""

    text = _redact(text)
    matches = list(_ERROR_HEADER_RE.finditer(text))
    blocks: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = _normalize_location(text[match.start() : stop].strip())
        block = re.sub(r"\n{3,}", "\n\n", block)
        headline = _SPACE_RE.sub(" ", match.group("head").strip())
        fingerprint = _METAVAR_RE.sub("?m.<id>", headline).lower()
        blocks.append((fingerprint, _trim(block, 700)))
    return blocks


def _tool_sequence(tool_names: Sequence[str], *, maximum_runs: int = 40) -> str:
    if not tool_names:
        return ""
    runs: list[tuple[str, int]] = []
    for name in tool_names:
        if runs and runs[-1][0] == name:
            previous_name, count = runs[-1]
            runs[-1] = (previous_name, count + 1)
        else:
            runs.append((name, 1))
    rendered = [f"{name}×{count}" if count > 1 else name for name, count in runs[:maximum_runs]]
    if len(runs) > maximum_runs:
        rendered.append(f"…{len(runs) - maximum_runs} more runs")
    return " → ".join(rendered)


def score_outcome(outcome: Outcome) -> float:
    """Implement the optimizer score without folding invalid infrastructure into failure."""

    if outcome == "verified":
        return 1.0
    if outcome == "model_failed":
        return 0.0
    if outcome == "infrastructure_invalid":
        raise InfrastructureInvalidEvaluation(
            "infrastructure-invalid evaluations must be retried or abort GEPA; they are not score zero"
        )
    raise FeedbackIntegrityError(f"unsupported outcome: {outcome}")


def _collect_transcript_evidence(
    messages: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    calls_by_id: dict[str, dict[str, Any]] = {}
    tool_names: list[str] = []
    search_calls: list[dict[str, object]] = []
    lean_calls: list[dict[str, object]] = []
    final_submission_calls = 0
    thinking_texts: list[str] = []

    for part in _message_parts(messages):
        part_kind = part.get("part_kind")
        if part_kind == "thinking" and isinstance(part.get("content"), str):
            thinking_texts.append(str(part["content"]))
            continue
        if part_kind == "tool-call":
            name = part.get("tool_name")
            call_id = part.get("tool_call_id")
            if not isinstance(name, str):
                continue
            args = _as_object(part.get("args"))
            tool_names.append(name)
            call = {"tool_name": name, "args": args}
            if isinstance(call_id, str):
                calls_by_id[call_id] = call
            if name == "final_submission":
                final_submission_calls += 1
            continue
        if part_kind != "tool-return":
            continue

        name = part.get("tool_name")
        call_id = part.get("tool_call_id")
        if not isinstance(name, str):
            continue
        call = calls_by_id.get(call_id, {}) if isinstance(call_id, str) else {}
        args = _as_object(call.get("args"))
        result = _as_object(part.get("content"))
        if name == "mathlib_search":
            search_calls.append({"args": args, "result": result})
        elif name == "lean_execute":
            lean_calls.append({"args": args, "result": result})

    search_errors: list[dict[str, str]] = []
    successful_search: dict[str, object] | None = None
    unique_queries: set[str] = set()
    zero_hit_searches = 0
    for call in search_calls:
        args = call["args"]
        result = call["result"]
        assert isinstance(args, dict) and isinstance(result, dict)
        query = str(args.get("query", "")).strip()
        if query:
            unique_queries.add(query)
        error = result.get("error")
        total_count = result.get("total_count")
        if isinstance(total_count, int) and total_count == 0:
            zero_hit_searches += 1
        if error:
            search_errors.append({"query": _trim(query, 180), "error": _trim(str(error), 300)})
        elif successful_search is None and isinstance(result.get("hits"), list):
            hit_names = [
                str(hit.get("name"))
                for hit in result["hits"][:3]
                if isinstance(hit, dict) and hit.get("name")
            ]
            if hit_names:
                successful_search = {"query": _trim(query, 180), "hit_names": hit_names}

    all_blocks: list[tuple[str, str]] = []
    placeholder_compilations = 0
    placeholder_free_successes = 0
    failed_checks = 0
    candidate_sizes: list[int] = []
    for call in lean_calls:
        args = call["args"]
        result = call["result"]
        assert isinstance(args, dict) and isinstance(result, dict)
        code = str(args.get("code", ""))
        if code:
            candidate_sizes.append(len(code))
        compiled = result.get("exit_code") == 0 and not result.get("timed_out", False)
        successful = compiled and not result.get("verification_error")
        has_placeholder = bool(_PLACEHOLDER_RE.search(code))
        if compiled and has_placeholder:
            placeholder_compilations += 1
        if successful and not has_placeholder:
            placeholder_free_successes += 1
        if not successful:
            failed_checks += 1
        diagnostic_text = "\n".join(
            str(result.get(key, "")) for key in ("stdout", "stderr", "verification_error")
        )
        all_blocks.extend(_error_blocks(diagnostic_text))

    fingerprint_counts = Counter(fingerprint for fingerprint, _excerpt in all_blocks)
    first_excerpt_by_fingerprint: dict[str, str] = {}
    for fingerprint, excerpt in all_blocks:
        first_excerpt_by_fingerprint.setdefault(fingerprint, excerpt)
    repeated = [
        {"count": count, "fingerprint": _trim(fingerprint, 260)}
        for fingerprint, count in fingerprint_counts.most_common(6)
        if count > 1
    ]

    representative_diagnostics: list[str] = []
    if all_blocks:
        candidates = [all_blocks[0]]
        most_common_fingerprint = fingerprint_counts.most_common(1)[0][0]
        candidates.append(
            (most_common_fingerprint, first_excerpt_by_fingerprint[most_common_fingerprint])
        )
        candidates.append(all_blocks[-1])
        seen: set[str] = set()
        for fingerprint, excerpt in candidates:
            if fingerprint not in seen:
                representative_diagnostics.append(excerpt)
                seen.add(fingerprint)

    searches_before_first_lean = 0
    for name in tool_names:
        if name == "lean_execute":
            break
        if name == "mathlib_search":
            searches_before_first_lean += 1

    diagnostic_text_lower = "\n".join(excerpt for _fp, excerpt in all_blocks).lower()
    return {
        "tool_counts": dict(sorted(Counter(tool_names).items())),
        "tool_sequence": _tool_sequence(tool_names),
        "search": {
            "calls": len(search_calls),
            "errors": len(search_errors),
            "zero_hits": zero_hit_searches,
            "unique_queries": len(unique_queries),
            "searches_before_first_lean": searches_before_first_lean,
            "representative_failures": search_errors[:2],
            "representative_success": successful_search,
        },
        "lean": {
            "checks": len(lean_calls),
            "failed_checks": failed_checks,
            "placeholder_free_successes": placeholder_free_successes,
            "placeholder_compilations": placeholder_compilations,
            "candidate_chars": {
                "minimum": min(candidate_sizes) if candidate_sizes else 0,
                "maximum": max(candidate_sizes) if candidate_sizes else 0,
            },
            "repeated_error_fingerprints": repeated,
            "representative_diagnostics": representative_diagnostics,
            "diagnostic_counts": {
                "rewrite_failed": diagnostic_text_lower.count("rewrite` failed")
                + diagnostic_text_lower.count("rewrite failed"),
                "unsolved_goals": diagnostic_text_lower.count("unsolved goals"),
                "unknown_declaration": diagnostic_text_lower.count("unknown identifier")
                + diagnostic_text_lower.count("unknown constant")
                + diagnostic_text_lower.count("unknown field")
                + diagnostic_text_lower.count("invalid projection")
                + diagnostic_text_lower.count("object file"),
                "type_mismatch": diagnostic_text_lower.count("type mismatch")
                + diagnostic_text_lower.count("type expected"),
                "typeclass_or_coercion": diagnostic_text_lower.count("failed to synthesize")
                + diagnostic_text_lower.count("application type mismatch")
                + diagnostic_text_lower.count("failed to coerce"),
            },
        },
        "final_submission": {
            "calls": final_submission_calls,
        },
        "decision_signal_counts": {
            "different_approach": sum(
                text.lower().count("different approach") for text in thinking_texts
            )
        },
    }


def _failure_labels(
    *,
    outcome: Outcome,
    run_record: Mapping[str, object],
    evidence: Mapping[str, object],
) -> tuple[str, list[str], str | None]:
    search = evidence["search"]
    lean = evidence["lean"]
    final_submission = evidence["final_submission"]
    decisions = evidence["decision_signal_counts"]
    assert isinstance(search, dict)
    assert isinstance(lean, dict)
    assert isinstance(final_submission, dict)
    assert isinstance(decisions, dict)
    diagnostic_counts = lean["diagnostic_counts"]
    candidate_chars = lean["candidate_chars"]
    assert isinstance(diagnostic_counts, dict)
    assert isinstance(candidate_chars, dict)

    error_text = " ".join(str(run_record.get(key, "")) for key in ("error_type", "error")).lower()
    context_exhaustion = any(
        fragment in error_text
        for fragment in (
            "context window",
            "context length",
            "exceeds the available context",
            "n_ctx",
        )
    )
    token_exhaustion = "token" in error_text and any(
        fragment in error_text for fragment in ("limit", "exceeded", "maximum")
    )

    secondary: list[str] = []
    search_calls = int(search["calls"])
    search_errors = int(search["errors"])
    rewrite_failed = int(diagnostic_counts["rewrite_failed"])
    unsolved_goals = int(diagnostic_counts["unsolved_goals"])
    unknown_declaration = int(diagnostic_counts["unknown_declaration"])
    type_mismatch = int(diagnostic_counts["type_mismatch"])
    typeclass_or_coercion = int(diagnostic_counts["typeclass_or_coercion"])

    if outcome == "verified":
        primary = "verified"
    elif context_exhaustion:
        primary = "trajectory_context_exhaustion"
    elif rewrite_failed >= 3 and unsolved_goals >= 3:
        primary = "rewrite_nonconvergence"
    elif search_calls >= 10 and search_errors / search_calls >= 0.45:
        primary = "declaration_discovery"
    elif unknown_declaration:
        primary = "namespace_or_api_mismatch"
    elif typeclass_or_coercion:
        primary = "typeclass_or_coercion"
    elif type_mismatch:
        primary = "binder_or_goal_shape_mismatch"
    elif token_exhaustion:
        primary = "token_exhaustion"
    else:
        primary = "proof_repair_nonconvergence"

    if search_errors:
        secondary.append("search_protocol_error")
    if unknown_declaration:
        secondary.append("namespace_or_api_mismatch")
    if type_mismatch:
        secondary.append("binder_or_goal_shape_mismatch")
    if typeclass_or_coercion:
        secondary.append("typeclass_or_coercion")
    if rewrite_failed >= 3:
        secondary.append("rewrite_nonconvergence")
    if int(candidate_chars["maximum"]) >= 8_000:
        secondary.append("oversized_candidate")
    if int(lean["placeholder_compilations"]):
        secondary.append("prohibited_placeholder")
    if int(search["searches_before_first_lean"]) >= 20:
        secondary.append("late_execution")
    if int(decisions["different_approach"]) >= 5:
        secondary.append("strategy_reset_loop")
    if int(final_submission["calls"]) and outcome != "verified":
        secondary.append("premature_submission")
    if context_exhaustion:
        secondary.append("trajectory_context_exhaustion")
    elif token_exhaustion:
        secondary.append("token_exhaustion")

    secondary = list(dict.fromkeys(label for label in secondary if label != primary))
    terminal_reason = (
        "trajectory_context_exhaustion"
        if context_exhaustion
        else "token_exhaustion"
        if token_exhaustion
        else None
    )
    return primary, secondary, terminal_reason


def _render_feedback(
    *,
    outcome: Outcome,
    primary: str,
    terminal_reason: str | None,
    evidence: Mapping[str, object],
) -> str:
    search = evidence["search"]
    lean = evidence["lean"]
    decisions = evidence["decision_signal_counts"]
    assert isinstance(search, dict) and isinstance(lean, dict) and isinstance(decisions, dict)
    if outcome == "verified":
        opening = (
            f"Verified after {search['calls']} Mathlib searches and {lean['checks']} Lean checks."
        )
    else:
        opening = (
            f"Failed to verify; the primary observed failure was {primary.replace('_', ' ')}. "
            f"The trajectory used {search['calls']} Mathlib searches and {lean['checks']} Lean checks."
        )
    sentences = [opening]
    if search["errors"]:
        sentences.append(
            f"{search['errors']} searches returned parser or declaration errors and "
            f"{search['zero_hits']} returned no hits."
        )
    if lean["placeholder_compilations"]:
        sentences.append(
            f"{lean['placeholder_compilations']} placeholder-containing candidates compiled; these "
            "are scaffolds, not valid successes."
        )
    if lean["failed_checks"]:
        sentences.append(f"{lean['failed_checks']} Lean checks failed.")
    if decisions["different_approach"]:
        count = int(decisions["different_approach"])
        noun = "time" if count == 1 else "times"
        sentences.append(
            f"The hidden reasoning announced a different approach {count} {noun}; only this count, "
            "not the reasoning text, is exposed."
        )
    if terminal_reason == "token_exhaustion":
        sentences.append("The run ended at the configured token limit.")
    elif terminal_reason == "trajectory_context_exhaustion":
        sentences.append("The accumulated trajectory exceeded the model context window.")
    return _trim(" ".join(sentences), 900)


def _fit_side_info(side_info: dict[str, object], char_limit: int) -> dict[str, object]:
    """Shrink optional evidence in a deterministic order until it fits the hard limit."""

    if char_limit < MINIMUM_FEEDBACK_CHAR_LIMIT:
        raise ValueError(f"feedback character limit must be at least {MINIMUM_FEEDBACK_CHAR_LIMIT}")
    fitted = copy.deepcopy(side_info)

    def current_size() -> int:
        return reflection_feedback_character_count(fitted)

    trajectory = fitted.get("trajectory")
    if not isinstance(trajectory, dict):
        raise FeedbackIntegrityError("discovery feedback is missing trajectory evidence")
    lean = trajectory.get("lean")
    search = trajectory.get("search")
    if not isinstance(lean, dict) or not isinstance(search, dict):
        raise FeedbackIntegrityError("discovery feedback has malformed trajectory evidence")

    diagnostics = lean.get("representative_diagnostics")
    failures = search.get("representative_failures")
    fingerprints = lean.get("repeated_error_fingerprints")
    failure = fitted.get("failure")
    assert isinstance(diagnostics, list)
    assert isinstance(failures, list)
    assert isinstance(fingerprints, list)
    assert isinstance(failure, dict)

    while current_size() > char_limit and len(diagnostics) > 1:
        diagnostics.pop()
    while current_size() > char_limit and len(failures) > 1:
        failures.pop()
    while current_size() > char_limit and len(fingerprints) > 3:
        fingerprints.pop()
    if current_size() > char_limit:
        lean["representative_diagnostics"] = [_trim(str(value), 350) for value in diagnostics]
    if current_size() > char_limit and failure.get("final_diagnostic"):
        failure["final_diagnostic"] = _trim(str(failure["final_diagnostic"]), 350)
    if current_size() > char_limit:
        trajectory["tool_sequence"] = _trim(str(trajectory.get("tool_sequence", "")), 350)
    if current_size() > char_limit:
        fitted["feedback"] = _trim(str(fitted.get("feedback", "")), 450)
    if current_size() > char_limit:
        artifact = fitted.get("artifact")
        if isinstance(artifact, dict):
            artifact.pop("task_run_path", None)
    if current_size() > char_limit:
        lean["representative_diagnostics"] = []
        search["representative_failures"] = []
        failure["final_diagnostic"] = None
    if current_size() > char_limit:
        search["representative_success"] = None
        lean["repeated_error_fingerprints"] = []
        trajectory["tool_sequence"] = ""
    if current_size() > char_limit:
        fitted.pop("artifact", None)
        fitted.pop("duration_s", None)
    if current_size() > char_limit:
        raise FeedbackIntegrityError(
            f"minimal side_info exceeds configured {char_limit}-character limit"
        )
    return fitted


def build_side_info(
    *,
    case_id: str,
    outcome: Outcome,
    run_record: Mapping[str, object],
    messages: Sequence[Mapping[str, object]] | None,
    mode: FeedbackMode,
    artifact_path: str | None = None,
    artifact_hashes: Mapping[str, str] | None = None,
    problem_source: str | None = None,
    char_limit: int = DEFAULT_FEEDBACK_CHAR_LIMIT,
) -> tuple[float, dict[str, object]]:
    """Build the GEPA evaluator result for one case.

    Selection mode intentionally ignores ``messages``. This makes transcript withholding an
    implementation property rather than a convention at the call site.
    """

    score = score_outcome(outcome)
    if mode == "selection":
        return score, {
            "schema_version": 1,
            "case_id": case_id,
            "outcome": outcome,
            "scores": {"verified": score},
            "feedback_withheld": True,
        }
    if mode != "discovery":
        raise FeedbackIntegrityError(f"unsupported feedback mode: {mode}")
    if messages is None:
        raise FeedbackIntegrityError("discovery feedback requires messages")

    evidence = _collect_transcript_evidence(messages)
    primary, secondary, terminal_reason = _failure_labels(
        outcome=outcome,
        run_record=run_record,
        evidence=evidence,
    )
    lean = evidence["lean"]
    assert isinstance(lean, dict)
    representative_diagnostics = lean["representative_diagnostics"]
    assert isinstance(representative_diagnostics, list)

    side_info: dict[str, object] = {
        "schema_version": 1,
        "case_id": case_id,
        "problem": {"source_excerpt": _trim(problem_source, 1_200)}
        if problem_source is not None
        else None,
        "outcome": outcome,
        "scores": {"verified": score},
        "feedback": _render_feedback(
            outcome=outcome,
            primary=primary,
            terminal_reason=terminal_reason,
            evidence=evidence,
        ),
        "failure": {
            "primary": primary,
            "secondary": secondary,
            "terminal_reason": terminal_reason,
            "final_diagnostic": representative_diagnostics[-1]
            if representative_diagnostics
            else None,
        },
        "trajectory": evidence,
        "usage": dict(run_record.get("usage", {}))
        if isinstance(run_record.get("usage"), dict)
        else {},
        "duration_s": _duration_seconds(run_record),
        "artifact": {
            "task_run_path": artifact_path,
            "run_sha256": (artifact_hashes or {}).get("run_sha256"),
            "messages_sha256": (artifact_hashes or {}).get("messages_sha256"),
        },
    }
    return score, _fit_side_info(side_info, char_limit)


def _resolve_repository_path(path: str) -> Path:
    resolved = (REPOSITORY_ROOT / path).resolve()
    if not resolved.is_relative_to(REPOSITORY_ROOT):
        raise FeedbackIntegrityError(f"path escapes repository: {path}")
    return resolved


def _validate_panel_split(panel_path: Path) -> tuple[dict[str, object], Path]:
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    discovery = panel.get("discovery_cases")
    selection = panel.get("selection_cases")
    if not isinstance(discovery, list) or not all(isinstance(item, str) for item in discovery):
        raise FeedbackIntegrityError("panel discovery_cases must be a string list")
    if not isinstance(selection, list) or not all(isinstance(item, str) for item in selection):
        raise FeedbackIntegrityError("panel selection_cases must be a string list")
    if not discovery or len(discovery) != len(set(discovery)):
        raise FeedbackIntegrityError("panel discovery_cases must be nonempty and unique")
    if len(selection) != len(set(selection)) or set(discovery) & set(selection):
        raise FeedbackIntegrityError("discovery and selection cases must be unique and disjoint")

    dataset_config = panel.get("dataset")
    if not isinstance(dataset_config, dict):
        raise FeedbackIntegrityError("panel is missing dataset configuration")
    dataset_path = _resolve_repository_path(str(dataset_config.get("path", "")))
    if dataset_path != REPOSITORY_ROOT / "datasets/basic-problems-v1.yaml":
        raise FeedbackIntegrityError("feedback replay only permits datasets/basic-problems-v1.yaml")
    if _sha256_file(dataset_path) != dataset_config.get("sha256"):
        raise FeedbackIntegrityError("dataset SHA-256 does not match the frozen panel")
    dataset = load_formalizer_dataset(dataset_path)
    cases = {case.name: case for case in dataset.cases}
    for case_name in [*discovery, *selection]:
        case = cases.get(case_name)
        split = case.metadata.split if case and case.metadata else None
        if split != "train":
            raise FeedbackIntegrityError(f"optimization case is not train-only: {case_name}")

    source = panel.get("source_screen")
    if not isinstance(source, dict):
        raise FeedbackIntegrityError("panel is missing source_screen")
    source_root = _resolve_repository_path(str(source.get("path", "")))
    if not source_root.is_relative_to(RUNS_ROOT.resolve()):
        raise FeedbackIntegrityError("source screen must be under runs/prompt-optimization")
    for filename, field in (
        ("manifest.json", "manifest_sha256"),
        ("status.json", "status_sha256"),
        ("result.json", "result_sha256"),
        ("evaluations.jsonl", "evaluations_sha256"),
    ):
        if _sha256_file(source_root / filename) != source.get(field):
            raise FeedbackIntegrityError(f"source {filename} SHA-256 does not match panel")
    return panel, source_root


def _finished_evaluations(source_root: Path) -> dict[str, dict[str, object]]:
    finished: dict[str, dict[str, object]] = {}
    for raw_line in (source_root / "evaluations.jsonl").read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        if record.get("event") != "finished":
            continue
        case_name = record.get("case_name")
        if not isinstance(case_name, str):
            raise FeedbackIntegrityError("finished evaluation is missing case_name")
        if case_name in finished:
            raise FeedbackIntegrityError(f"multiple finished evaluations for {case_name}")
        finished[case_name] = record
    return finished


def _load_discovery_task_run(
    *,
    source_root: Path,
    case_name: str,
    evaluation: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]], Path, dict[str, str]]:
    task_run_path = evaluation.get("task_run_path")
    if isinstance(task_run_path, str):
        task_root = Path(task_run_path).resolve()
    else:
        manifest_paths = evaluation.get("all_run_manifest_paths")
        if (
            not isinstance(manifest_paths, list)
            or len(manifest_paths) != 1
            or not isinstance(manifest_paths[0], str)
        ):
            raise FeedbackIntegrityError(
                f"discovery evaluation has no unambiguous task-run manifest: {case_name}"
            )
        manifest_path = Path(manifest_paths[0])
        if not manifest_path.is_absolute():
            manifest_path = REPOSITORY_ROOT / manifest_path
        task_root = manifest_path.resolve().parent
    if not task_root.is_relative_to(source_root.resolve()):
        raise FeedbackIntegrityError(f"task run escapes source screen: {case_name}")
    run_path = task_root / "run.json"
    messages_path = task_root / "messages.json"
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    messages = json.loads(messages_path.read_text(encoding="utf-8"))
    if not isinstance(run_record, dict) or not isinstance(messages, list):
        raise FeedbackIntegrityError(f"malformed task-run artifacts: {case_name}")
    if not all(isinstance(message, dict) for message in messages):
        raise FeedbackIntegrityError(f"malformed messages list: {case_name}")
    hashes = {
        "run_sha256": _sha256_file(run_path),
        "messages_sha256": _sha256_file(messages_path),
    }
    return run_record, messages, task_root, hashes


def _git_provenance(paths: Sequence[Path]) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "commit": commit,
        "is_dirty": bool(status.strip()),
        "status_sha256": _sha256_bytes(status.encode()),
        "diff_sha256": _sha256_bytes(diff),
        "file_sha256s": {
            str(path.resolve().relative_to(REPOSITORY_ROOT)): _sha256_file(path) for path in paths
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(_canonical_json(value) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _normalize_outcome(evaluation: Mapping[str, object]) -> Outcome:
    terminal_status = evaluation.get("terminal_status")
    if terminal_status == "verified":
        return "verified"
    if terminal_status == "model_failed":
        return "model_failed"
    if terminal_status == "infrastructure_invalid":
        return "infrastructure_invalid"
    raise FeedbackIntegrityError(f"unsupported terminal_status: {terminal_status}")


def replay_discovery_feedback(
    *,
    panel_path: Path,
    output_dir: Path,
    char_limit: int = DEFAULT_FEEDBACK_CHAR_LIMIT,
) -> dict[str, object]:
    """Replay frozen discovery transcripts into a fresh, auditable preview artifact."""

    panel_path = panel_path.resolve()
    panel, source_root = _validate_panel_split(panel_path)
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(RUNS_ROOT.resolve()):
        raise FeedbackIntegrityError("output directory must be under runs/prompt-optimization")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")

    discovery = panel["discovery_cases"]
    selection = panel["selection_cases"]
    assert isinstance(discovery, list) and isinstance(selection, list)
    finished = _finished_evaluations(source_root)
    missing = [case_name for case_name in discovery if case_name not in finished]
    if missing:
        raise FeedbackIntegrityError(f"missing discovery evaluations: {missing}")

    run_id = str(uuid4())
    created_at = _utc_now()
    output_dir.mkdir(parents=True)
    events_path = output_dir / "events.jsonl"
    preview_path = output_dir / "feedback_preview.jsonl"
    status_path = output_dir / "status.json"
    code_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "phase": "discovery_feedback_preview",
        "run_id": run_id,
        "created_at": created_at,
        "status_at_launch": "running",
        "parent_run": {
            "path": str(source_root.relative_to(REPOSITORY_ROOT)),
            "run_id": panel["source_screen"]["run_id"],
            "manifest_sha256": _sha256_file(source_root / "manifest.json"),
            "status_sha256": _sha256_file(source_root / "status.json"),
            "evaluations_sha256": _sha256_file(source_root / "evaluations.jsonl"),
        },
        "panel_split": {
            "path": str(panel_path.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256_file(panel_path),
            "discovery_cases": discovery,
            "selection_cases": selection,
        },
        "feedback_contract": {
            "schema_version": 1,
            "character_limit": char_limit,
            "primary_score": "verified:1.0, model_failed:0.0",
            "infrastructure_invalid": "raise; never score as model failure",
            "selection_feedback": "score-only with feedback_withheld=true",
            "full_transcripts": "retained only in parent task-runs",
        },
        "analysis": {
            "model_calls": 0,
            "network_calls": 0,
            "selection_transcripts_inspected": False,
            "reference_proofs_inspected": False,
            "method": "Deterministic rule-based extraction from frozen discovery artifacts.",
        },
        "git": _git_provenance([code_path, panel_path]),
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(
        output_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "gepa": importlib.metadata.version("gepa"),
            "credentials_recorded": False,
        },
    )
    _append_jsonl(events_path, {"event": "run_started", "run_id": run_id, "at": created_at})

    counts: Counter[str] = Counter()
    character_counts: list[int] = []
    try:
        for case_name in discovery:
            evaluation = finished[case_name]
            outcome = _normalize_outcome(evaluation)
            run_record, messages, task_root, artifact_hashes = _load_discovery_task_run(
                source_root=source_root,
                case_name=case_name,
                evaluation=evaluation,
            )
            score, side_info = build_side_info(
                case_id=case_name,
                outcome=outcome,
                run_record=run_record,
                messages=messages,
                mode="discovery",
                artifact_path=str(task_root.relative_to(REPOSITORY_ROOT)),
                artifact_hashes=artifact_hashes,
                problem_source=(run_record.get("problem") or {}).get("source")
                if isinstance(run_record.get("problem"), dict)
                else None,
                char_limit=char_limit,
            )
            side_info_json = _canonical_json(side_info)
            reflection_chars = reflection_feedback_character_count(side_info)
            record = {
                "case_name": case_name,
                "score": score,
                "side_info": side_info,
                "side_info_character_count": reflection_chars,
                "side_info_canonical_json_character_count": len(side_info_json),
                "side_info_sha256": _sha256_bytes(side_info_json.encode()),
                "source_artifact_hashes": artifact_hashes,
            }
            _append_jsonl(preview_path, record)
            counts[outcome] += 1
            character_counts.append(reflection_chars)
            _append_jsonl(
                events_path,
                {
                    "event": "case_distilled",
                    "run_id": run_id,
                    "case_name": case_name,
                    "outcome": outcome,
                    "side_info_sha256": record["side_info_sha256"],
                    "at": _utc_now(),
                },
            )

        finished_at = _utc_now()
        result = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "completed",
            "case_count": len(discovery),
            "outcome_counts": dict(sorted(counts.items())),
            "feedback_character_limit": char_limit,
            "maximum_feedback_characters": max(character_counts, default=0),
            "mean_feedback_characters": round(sum(character_counts) / len(character_counts), 3)
            if character_counts
            else 0,
            "feedback_preview_sha256": _sha256_file(preview_path),
            "selection_transcripts_inspected": False,
            "model_calls": 0,
        }
        _write_json(output_dir / "result.json", result)
        _write_json(
            status_path,
            {
                "schema_version": 1,
                "run_id": run_id,
                "phase": "discovery_feedback_preview",
                "status": "completed",
                "finished_at": finished_at,
                "case_count": len(discovery),
                "selection_transcripts_inspected": False,
                "model_calls": 0,
            },
        )
        _append_jsonl(
            events_path,
            {"event": "run_completed", "run_id": run_id, "at": finished_at},
        )
        return result
    except BaseException as exc:
        failed_at = _utc_now()
        _write_json(
            status_path,
            {
                "schema_version": 1,
                "run_id": run_id,
                "phase": "discovery_feedback_preview",
                "status": "failed",
                "finished_at": failed_at,
                "error_type": type(exc).__name__,
                "error": _trim(str(exc), 1_000),
                "selection_transcripts_inspected": False,
                "model_calls": 0,
            },
        )
        _append_jsonl(
            events_path,
            {
                "event": "run_failed",
                "run_id": run_id,
                "error_type": type(exc).__name__,
                "error": _trim(str(exc), 1_000),
                "at": failed_at,
            },
        )
        raise


def _dry_run_plan(panel_path: Path, output_dir: Path, char_limit: int) -> dict[str, object]:
    panel, source_root = _validate_panel_split(panel_path.resolve())
    return {
        "dry_run": True,
        "panel_split": str(panel_path.resolve()),
        "source_run": str(source_root),
        "output_dir": str(output_dir.resolve()),
        "discovery_cases": panel["discovery_cases"],
        "selection_case_count": len(panel["selection_cases"]),
        "selection_transcript_policy": "not read; GEPA receives score-only feedback",
        "feedback_character_limit": char_limit,
        "model_calls": 0,
        "execute_requires": ["--execute"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-split", type=Path, default=DEFAULT_PANEL_SPLIT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feedback-char-limit", type=int, default=DEFAULT_FEEDBACK_CHAR_LIMIT)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if not args.execute:
            print(
                json.dumps(
                    _dry_run_plan(args.panel_split, args.output_dir, args.feedback_char_limit),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        result = replay_discovery_feedback(
            panel_path=args.panel_split,
            output_dir=args.output_dir,
            char_limit=args.feedback_char_limit,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (FeedbackIntegrityError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
