"""Run a fully logged Formalizer prompt optimization with pinned GEPA 0.1.4."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from gepa.lm import LM
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    TrackingConfig,
    optimize_anything,
)
from gepa.utils import SignalStopper, TimeoutStopCondition
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import ModelSettings, UsageLimits
from pydantic_ai.exceptions import ModelHTTPError
from transcript_feedback import (
    FeedbackIntegrityError,
    InfrastructureInvalidEvaluation,
    build_side_info,
    reflection_feedback_character_count,
)

from formalizer.agent import _INSTRUCTIONS
from formalizer.eval import load_formalizer_dataset
from formalizer.eval.runner import is_retryable_infrastructure_failure
from formalizer.problem import FormalizationProblem
from formalizer.run import formalize
from formalizer.settings import RunSettings, Settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPOSITORY_ROOT / "runs/prompt-optimization"
DEFAULT_CONFIG = REPOSITORY_ROOT / "experiments/prompt_optimization/gepa_overnight_v1.json"

OPTIMIZATION_OBJECTIVE = (
    "Improve the Formalizer Lean 4 agent instructions so the task model produces more "
    "Lean-verified proofs on held-out selection cases from the original training split."
)
OPTIMIZATION_BACKGROUND = """\
The candidate is a complete drop-in replacement for the Formalizer agent instructions. Preserve
the immutable FormalizerProblem.Target contract, the required FormalizerSubmission.solution
declaration, the prohibition on sorry/admit/axioms, the complete-source semantics of lean_execute,
and final_submission as the only terminal answer. Encourage efficient Mathlib declaration search,
early small Lean checks, diagnostic-driven repair, and convergence instead of repeated strategy
resets. Do not specialize the instructions to one theorem or include solutions from the examples.
Only discovery examples may provide transcript feedback; selection examples are score-only.
"""

Outcome = Literal["verified", "model_failed", "infrastructure_invalid"]
Phase = Literal["canary", "optimize"]


class FileReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    sha256: str


class TaskModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["openai-compatible"]
    endpoint: str
    model: str
    server_model_id: str
    minimum_context_tokens: int = Field(ge=1)
    max_tokens: int = Field(ge=1)
    tool_choice: Literal["auto", "required"]
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)

    @model_validator(mode="after")
    def one_sampling_control(self) -> TaskModelConfig:
        if self.temperature is not None and self.top_p is not None:
            raise ValueError("temperature and top_p are mutually exclusive")
        return self


class ReflectionModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["openai-compatible-litellm"]
    endpoint: str
    model: str
    server_model_id: str
    max_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0)
    timeout_s: int = Field(ge=1)
    retries: int = Field(ge=0)


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    repetitions: Literal[1]
    concurrency: Literal[1]
    run_timeout_s: int = Field(ge=1)
    request_limit: int = Field(ge=1)
    output_tokens_limit: int = Field(ge=1)
    total_tokens_limit: int = Field(ge=1)
    infrastructure_retries: int = Field(ge=0)
    feedback_char_limit: int = Field(ge=1)
    max_candidate_chars: int = Field(ge=1)


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    max_metric_calls: int = Field(ge=1)
    max_candidate_proposals: int = Field(ge=1)
    max_wall_time_s: int = Field(ge=1)
    reflection_minibatch_size: int = Field(ge=1)
    cache_evaluation: Literal[True]
    acceptance_criterion: Literal["strict_improvement"]


class OvernightConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    name: str
    dataset: FileReference
    panel_split: FileReference
    seed_prompt: FileReference
    task_model: TaskModelConfig
    reflection_model: ReflectionModelConfig
    execution: ExecutionConfig
    optimizer: OptimizerConfig
    canary_case: str

    @model_validator(mode="after")
    def endpoints_match(self) -> OvernightConfig:
        if self.task_model.endpoint.rstrip("/") != self.reflection_model.endpoint.rstrip("/"):
            raise ValueError("task and reflection endpoints must match for overnight v1")
        if self.task_model.server_model_id != self.reflection_model.server_model_id:
            raise ValueError("task and reflection server model ids must match")
        return self


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _resolve_repository_path(path: str) -> Path:
    resolved = (REPOSITORY_ROOT / path).resolve()
    if not resolved.is_relative_to(REPOSITORY_ROOT):
        raise FeedbackIntegrityError(f"path escapes repository: {path}")
    return resolved


def _load_config(path: Path) -> OvernightConfig:
    return OvernightConfig.model_validate_json(path.read_bytes())


def _validate_file_reference(reference: FileReference, expected: Path | None = None) -> Path:
    path = _resolve_repository_path(reference.path)
    if expected is not None and path != expected.resolve():
        raise FeedbackIntegrityError(f"unexpected path: {reference.path}")
    if _sha256_file(path) != reference.sha256:
        raise FeedbackIntegrityError(f"SHA-256 mismatch: {reference.path}")
    return path


def _server_metadata(config: OvernightConfig) -> dict[str, object]:
    request = urllib.request.Request(
        f"{config.task_model.endpoint.rstrip('/')}/models",
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise FeedbackIntegrityError("model endpoint returned a non-object payload")
    data = payload.get("data", [])
    entries = [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
    matching = [item for item in entries if item.get("id") == config.task_model.server_model_id]
    if len(matching) != 1:
        raise FeedbackIntegrityError(
            f"server metadata does not contain exactly one {config.task_model.server_model_id!r}"
        )
    meta = matching[0].get("meta")
    context = meta.get("n_ctx") if isinstance(meta, dict) else None
    if not isinstance(context, int) or context < config.task_model.minimum_context_tokens:
        raise FeedbackIntegrityError(
            f"server context {context!r} is below required {config.task_model.minimum_context_tokens}"
        )
    return payload


def _validate_inputs(
    config: OvernightConfig,
) -> tuple[Path, Path, Path, list[dict[str, str]], list[dict[str, str]]]:
    dataset_path = _validate_file_reference(
        config.dataset,
        REPOSITORY_ROOT / "datasets/basic-problems-v1.yaml",
    )
    panel_path = _validate_file_reference(
        config.panel_split,
        REPOSITORY_ROOT / "experiments/prompt_optimization/panel_split_v1.json",
    )
    prompt_path = _validate_file_reference(config.seed_prompt)
    if prompt_path.read_text(encoding="utf-8") != _INSTRUCTIONS:
        raise FeedbackIntegrityError("seed prompt no longer matches the production default")

    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    discovery_names = panel.get("discovery_cases")
    selection_names = panel.get("selection_cases")
    if not isinstance(discovery_names, list) or not all(
        isinstance(item, str) for item in discovery_names
    ):
        raise FeedbackIntegrityError("panel discovery_cases must be a string list")
    if not isinstance(selection_names, list) or not all(
        isinstance(item, str) for item in selection_names
    ):
        raise FeedbackIntegrityError("panel selection_cases must be a string list")
    if set(discovery_names) & set(selection_names):
        raise FeedbackIntegrityError("discovery and selection cases overlap")
    if config.canary_case not in discovery_names:
        raise FeedbackIntegrityError("canary_case must be in discovery_cases")

    dataset = load_formalizer_dataset(dataset_path)
    cases = {case.name: case for case in dataset.cases}

    def examples(names: Sequence[str], mode: Literal["discovery", "selection"]):
        built: list[dict[str, str]] = []
        for name in names:
            case = cases.get(name)
            split = case.metadata.split if case and case.metadata else None
            if case is None or split != "train":
                raise FeedbackIntegrityError(f"optimization case is not train-only: {name}")
            built.append(
                {
                    "case_name": name,
                    "source": case.inputs.source,
                    "feedback_mode": mode,
                }
            )
        return built

    return (
        dataset_path,
        panel_path,
        prompt_path,
        examples(discovery_names, "discovery"),
        examples(selection_names, "selection"),
    )


def _git_provenance(paths: Sequence[Path]) -> dict[str, object]:
    def git(*args: str, binary: bool = False) -> str | bytes:
        result = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *args],
            check=True,
            capture_output=True,
            text=not binary,
        )
        return result.stdout

    commit = str(git("rev-parse", "HEAD")).strip()
    status = str(git("status", "--porcelain=v1"))
    diff = git("diff", "--binary", "HEAD", binary=True)
    assert isinstance(diff, bytes)
    return {
        "commit": commit,
        "is_dirty": bool(status),
        "status_sha256": _sha256_bytes(status.encode()),
        "diff_sha256": _sha256_bytes(diff),
        "file_sha256s": {
            str(path.resolve().relative_to(REPOSITORY_ROOT)): _sha256_file(path) for path in paths
        },
    }


def _write_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, value: object, lock: threading.Lock | None = None) -> None:
    if lock is not None:
        with lock:
            _append_jsonl(path, value)
        return
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _model_settings(config: TaskModelConfig) -> ModelSettings:
    settings = ModelSettings(max_tokens=config.max_tokens, tool_choice=config.tool_choice)
    if config.temperature is not None:
        settings["temperature"] = config.temperature
    if config.top_p is not None:
        settings["top_p"] = config.top_p
    return settings


def _exception_is_context_exhaustion(error: BaseException) -> bool:
    text = str(error).lower()
    return any(
        phrase in text
        for phrase in (
            "context window",
            "context length",
            "exceeds the available context",
            "n_ctx",
            "too many tokens",
        )
    )


def _is_retryable_infrastructure(error: BaseException) -> bool:
    return is_retryable_infrastructure_failure(error)


def _classify_exception(error: BaseException) -> tuple[Outcome, bool]:
    """Separate invalid infrastructure from genuine task-model failures.

    Context exhaustion is a real failure under the frozen candidate and token budget. Other
    non-retryable HTTP errors indicate an endpoint/authentication/request-contract problem and
    invalidate the experiment rather than lowering the candidate's score.
    """

    if _is_retryable_infrastructure(error):
        return "infrastructure_invalid", True
    if isinstance(error, ModelHTTPError):
        if error.status_code == 400 and _exception_is_context_exhaustion(error):
            return "model_failed", False
        return "infrastructure_invalid", False
    return "model_failed", False


class ExperimentLog:
    def __init__(self, root: Path, run_id: str, phase: Phase):
        self.root = root
        self.run_id = run_id
        self.phase = phase
        self.lock = threading.Lock()
        self.counts: Counter[str] = Counter()
        self.evaluation_index = 0

    def event(self, event: str, **fields: object) -> None:
        _append_jsonl(
            self.root / "events.jsonl",
            {"event": event, "run_id": self.run_id, "at": _utc_now(), **fields},
            self.lock,
        )

    def evaluation(self, value: Mapping[str, object]) -> None:
        _append_jsonl(self.root / "evaluations.jsonl", dict(value), self.lock)

    def reflection(self, value: Mapping[str, object]) -> None:
        _append_jsonl(self.root / "reflections.jsonl", dict(value), self.lock)

    def next_evaluation_id(self) -> str:
        with self.lock:
            index = self.evaluation_index
            self.evaluation_index += 1
        return f"{self.run_id}:m{index:05d}"

    def materialize_candidate(self, candidate: str) -> str:
        digest = _sha256_bytes(candidate.encode())
        path = self.root / "candidates" / f"{digest}.txt"
        with self.lock:
            if path.exists():
                if path.read_text(encoding="utf-8") != candidate:
                    raise FeedbackIntegrityError(f"candidate hash collision: {digest}")
            else:
                with path.open("x", encoding="utf-8") as stream:
                    stream.write(candidate)
                    stream.flush()
                    os.fsync(stream.fileno())
        return digest

    def record_terminal(self, status: str, **fields: object) -> None:
        with self.lock:
            self.counts[status] += 1
            status_payload = {
                "schema_version": 1,
                "run_id": self.run_id,
                "phase": self.phase,
                "status": "running",
                "updated_at": _utc_now(),
                "evaluation_counts": dict(sorted(self.counts.items())),
                "last_evaluation": fields,
            }
            _write_json(self.root / "status.json", status_payload)


class LoggedReflectionLM:
    """Log each exact GEPA reflection prompt and response around a pinned local LM."""

    def __init__(self, config: ReflectionModelConfig, log: ExperimentLog):
        self.config = config
        self.log = log
        self.inner = LM(
            config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            num_retries=config.retries,
            api_base=config.endpoint,
            api_key=os.environ.get("OPENAI_API_KEY", "local-llama-cpp"),
            timeout=config.timeout_s,
        )
        self.call_index = 0

    @property
    def total_cost(self) -> float:
        return self.inner.total_cost

    @property
    def total_tokens_in(self) -> int:
        return self.inner.total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        return self.inner.total_tokens_out

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        index = self.call_index
        self.call_index += 1
        serialized = prompt if isinstance(prompt, str) else _canonical_json(prompt)
        reflection_id = f"{self.log.run_id}:r{index:05d}"
        started_at = _utc_now()
        self.log.reflection(
            {
                "event": "started",
                "reflection_id": reflection_id,
                "started_at": started_at,
                "prompt": prompt,
                "prompt_sha256": _sha256_bytes(serialized.encode()),
                "model": self.config.model,
            }
        )
        before_in = self.total_tokens_in
        before_out = self.total_tokens_out
        before_cost = self.total_cost
        try:
            output = self.inner(prompt)
        except BaseException as error:
            self.log.reflection(
                {
                    "event": "failed",
                    "reflection_id": reflection_id,
                    "started_at": started_at,
                    "finished_at": _utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            raise
        self.log.reflection(
            {
                "event": "finished",
                "reflection_id": reflection_id,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "output": output,
                "output_sha256": _sha256_bytes(output.encode()),
                "usage_delta": {
                    "input_tokens": self.total_tokens_in - before_in,
                    "output_tokens": self.total_tokens_out - before_out,
                    "cost_usd": self.total_cost - before_cost,
                },
            }
        )
        return output

    def __repr__(self) -> str:
        return f"LoggedReflectionLM(model={self.config.model!r}, endpoint={self.config.endpoint!r})"


class FormalizerEvaluator:
    def __init__(self, config: OvernightConfig, log: ExperimentLog):
        self.config = config
        self.log = log

    def _settings(self, task_runs_dir: Path) -> Settings:
        execution = self.config.execution
        return Settings(
            model_name=self.config.task_model.model,
            model_settings=_model_settings(self.config.task_model),
            run=RunSettings(
                runs_dir=task_runs_dir,
                run_timeout_s=execution.run_timeout_s,
                usage_limits=UsageLimits(
                    request_limit=execution.request_limit,
                    output_tokens_limit=execution.output_tokens_limit,
                    total_tokens_limit=execution.total_tokens_limit,
                ),
            ),
        )

    def __call__(
        self,
        candidate: str,
        example: Mapping[str, str],
    ) -> tuple[float, dict[str, object]]:
        if not candidate.strip():
            raise FeedbackIntegrityError("GEPA proposed a blank prompt candidate")
        if len(candidate) > self.config.execution.max_candidate_chars:
            raise FeedbackIntegrityError(
                f"candidate has {len(candidate)} characters; limit is "
                f"{self.config.execution.max_candidate_chars}"
            )
        case_name = example.get("case_name")
        source = example.get("source")
        feedback_mode = example.get("feedback_mode")
        if not case_name or not source or feedback_mode not in {"discovery", "selection"}:
            raise FeedbackIntegrityError("malformed GEPA example")

        candidate_sha256 = self.log.materialize_candidate(candidate)
        evaluation_id = self.log.next_evaluation_id()
        evaluation_root = self.log.root / "task-runs" / evaluation_id.replace(":", "_")
        evaluation_root.mkdir(parents=True)
        problem = FormalizationProblem(source=source)

        for infrastructure_attempt in range(self.config.execution.infrastructure_retries + 1):
            attempt_id = f"{evaluation_id}:i{infrastructure_attempt}"
            attempt_root = evaluation_root / f"attempt-{infrastructure_attempt:02d}"
            task_runs_dir = attempt_root / "runs"
            task_runs_dir.mkdir(parents=True)
            task_run_id = uuid4()
            task_run_path = task_runs_dir / str(task_run_id)
            started_at = _utc_now()
            base = {
                "schema_version": 1,
                "evaluation_id": attempt_id,
                "metric_evaluation_id": evaluation_id,
                "candidate_sha256": candidate_sha256,
                "case_name": case_name,
                "feedback_mode": feedback_mode,
                "repetition": 0,
                "seed": self.config.optimizer.seed,
                "infrastructure_attempt": infrastructure_attempt,
            }
            self.log.evaluation({**base, "event": "started", "started_at": started_at})
            self.log.event(
                "evaluation_started",
                evaluation_id=attempt_id,
                candidate_sha256=candidate_sha256,
                case_name=case_name,
                feedback_mode=feedback_mode,
            )

            error: BaseException | None = None
            retryable_infrastructure = False
            try:
                result = asyncio.run(
                    formalize(
                        problem,
                        self._settings(task_runs_dir),
                        run_id=task_run_id,
                        instructions=candidate,
                    )
                )
                outcome: Outcome = "verified" if result.output.check.accepted else "model_failed"
            except (KeyboardInterrupt, asyncio.CancelledError):
                raise
            except Exception as caught:  # noqa: BLE001 - classification is part of the evaluator
                error = caught
                outcome, retryable_infrastructure = _classify_exception(caught)

            run_path = task_run_path / "run.json"
            messages_path = task_run_path / "messages.json"
            artifacts_complete = run_path.is_file() and messages_path.is_file()
            if outcome != "infrastructure_invalid" and not artifacts_complete:
                raise FeedbackIntegrityError(
                    f"Formalizer did not persist complete task artifacts for {attempt_id}"
                ) from error
            if artifacts_complete:
                run_record = json.loads(run_path.read_text(encoding="utf-8"))
                if not isinstance(run_record, dict):
                    raise FeedbackIntegrityError(f"malformed run.json for {attempt_id}")
                artifact_hashes = {
                    "run_sha256": _sha256_file(run_path),
                    "messages_sha256": _sha256_file(messages_path),
                }
            else:
                run_record = {}
                artifact_hashes = {}
            finished_at = _utc_now()

            if outcome == "infrastructure_invalid":
                finished = {
                    **base,
                    "event": "finished",
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "terminal_status": outcome,
                    "score": None,
                    "verified": False,
                    "infrastructure_invalid": True,
                    "retryable_infrastructure": retryable_infrastructure,
                    "failure_type": type(error).__name__ if error else None,
                    "error": str(error) if error else None,
                    "task_run_id": str(task_run_id),
                    "task_run_path": str(task_run_path.relative_to(REPOSITORY_ROOT)),
                    "formalizer_artifacts_complete": artifacts_complete,
                    "missing_artifacts": []
                    if artifacts_complete
                    else [
                        name
                        for name, path in (("run.json", run_path), ("messages.json", messages_path))
                        if not path.is_file()
                    ],
                    "usage": run_record.get("usage", {}),
                    "duration_s": None,
                    "feedback_sha256": None,
                    "artifact_hashes": artifact_hashes,
                }
                self.log.evaluation(finished)
                self.log.record_terminal(outcome, **finished)
                self.log.event(
                    "evaluation_finished",
                    evaluation_id=attempt_id,
                    terminal_status=outcome,
                    retrying=(
                        retryable_infrastructure
                        and infrastructure_attempt < self.config.execution.infrastructure_retries
                    ),
                )
                if (
                    retryable_infrastructure
                    and infrastructure_attempt < self.config.execution.infrastructure_retries
                ):
                    continue
                raise InfrastructureInvalidEvaluation(
                    f"infrastructure invalid after {infrastructure_attempt + 1} attempt(s): {error}"
                ) from error

            messages: list[dict[str, object]] | None = None
            if feedback_mode == "discovery":
                raw_messages = json.loads(messages_path.read_text(encoding="utf-8"))
                if not isinstance(raw_messages, list) or not all(
                    isinstance(message, dict) for message in raw_messages
                ):
                    raise FeedbackIntegrityError(f"malformed messages.json for {attempt_id}")
                messages = raw_messages

            score, side_info = build_side_info(
                case_id=case_name,
                outcome=outcome,
                run_record=run_record,
                messages=messages,
                mode=feedback_mode,
                artifact_path=str(task_run_path.relative_to(REPOSITORY_ROOT)),
                artifact_hashes=artifact_hashes,
                problem_source=source,
                char_limit=self.config.execution.feedback_char_limit,
            )
            feedback_json = _canonical_json(side_info)
            finished = {
                **base,
                "event": "finished",
                "started_at": started_at,
                "finished_at": finished_at,
                "terminal_status": outcome,
                "score": score,
                "verified": outcome == "verified",
                "infrastructure_invalid": False,
                "retryable_infrastructure": False,
                "failure_type": type(error).__name__ if error else None,
                "error": str(error) if error else None,
                "task_run_id": str(task_run_id),
                "task_run_path": str(task_run_path.relative_to(REPOSITORY_ROOT)),
                "formalizer_artifacts_complete": True,
                "usage": run_record.get("usage", {}),
                "duration_s": side_info.get("duration_s"),
                "feedback_sha256": _sha256_bytes(feedback_json.encode()),
                "feedback_rendered_characters": reflection_feedback_character_count(side_info),
                "optimizer_side_info": side_info,
                "artifact_hashes": artifact_hashes,
            }
            self.log.evaluation(finished)
            self.log.record_terminal(outcome, **finished)
            self.log.event(
                "evaluation_finished",
                evaluation_id=attempt_id,
                terminal_status=outcome,
                score=score,
            )
            return score, side_info

        raise AssertionError("unreachable infrastructure retry loop")


def _prepare_run(
    *,
    config: OvernightConfig,
    config_path: Path,
    output_dir: Path,
    phase: Phase,
    dataset_path: Path,
    panel_path: Path,
    prompt_path: Path,
    discovery: Sequence[Mapping[str, str]],
    selection: Sequence[Mapping[str, str]],
    server_metadata: Mapping[str, object],
) -> tuple[ExperimentLog, dict[str, object]]:
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(RUNS_ROOT.resolve()):
        raise FeedbackIntegrityError("output directory must be under runs/prompt-optimization")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    run_id = str(uuid4())
    relevant_paths = (
        config_path,
        dataset_path,
        panel_path,
        prompt_path,
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "experiments/prompt_optimization/transcript_feedback.py",
        REPOSITORY_ROOT / "src/formalizer/agent.py",
        REPOSITORY_ROOT / "src/formalizer/run.py",
        REPOSITORY_ROOT / "pyproject.toml",
        REPOSITORY_ROOT / "uv.lock",
    )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "phase": phase,
        "created_at": _utc_now(),
        "status": "running",
        "status_at_launch": "running",
        "config": {
            "path": str(config_path.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256_file(config_path),
            "content": config.model_dump(mode="json"),
        },
        "git": _git_provenance(relevant_paths),
        "dependencies": {
            "formalizer": importlib.metadata.version("formalizer"),
            "gepa": importlib.metadata.version("gepa"),
            "lock_sha256": _sha256_file(REPOSITORY_ROOT / "uv.lock"),
        },
        "dataset": {
            "path": str(dataset_path.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256_file(dataset_path),
            "required_split": "train",
            "discovery_cases": [example["case_name"] for example in discovery],
            "selection_cases": [example["case_name"] for example in selection],
        },
        "seed_prompt": {
            "path": str(prompt_path.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256_file(prompt_path),
        },
        "evaluator": {
            "path": "experiments/prompt_optimization/gepa_runner.py",
            "sha256": _sha256_file(Path(__file__).resolve()),
            "transcript_feedback_path": ("experiments/prompt_optimization/transcript_feedback.py"),
            "transcript_feedback_sha256": _sha256_file(
                REPOSITORY_ROOT / "experiments/prompt_optimization/transcript_feedback.py"
            ),
            "feedback_char_limit": config.execution.feedback_char_limit,
            "selection_feedback": "score-only; messages.json is not parsed",
        },
        "task_model": config.task_model.model_dump(mode="json"),
        "reflection_model": config.reflection_model.model_dump(mode="json"),
        "execution": config.execution.model_dump(mode="json"),
        "gepa": {
            "engine": {
                "run_dir": "gepa/",
                "seed": config.optimizer.seed,
                "max_metric_calls": config.optimizer.max_metric_calls,
                "max_candidate_proposals": config.optimizer.max_candidate_proposals,
                "parallel": False,
                "max_workers": 1,
                "raise_on_exception": True,
                "cache_evaluation": config.optimizer.cache_evaluation,
                "cache_evaluation_storage": "disk",
                "acceptance_criterion": config.optimizer.acceptance_criterion,
            },
            "reflection": {
                "reflection_minibatch_size": config.optimizer.reflection_minibatch_size,
                "perfect_score": 1.0,
                "skip_perfect_score": False,
                "model": config.reflection_model.model,
            },
            "tracking": {"wandb": False, "mlflow": False},
            "stop_conditions": {
                "timeout_seconds": config.optimizer.max_wall_time_s,
                "signal_stopper": True,
                "stop_file": "gepa/gepa.stop",
            },
        },
        "objective": OPTIMIZATION_OBJECTIVE,
        "background": OPTIMIZATION_BACKGROUND,
        "logging_contract": {
            "candidate_texts": "candidates/<sha256>.txt",
            "task_attempts": "evaluations.jsonl plus task-runs/",
            "reflection_prompts_and_outputs": "reflections.jsonl",
            "optimizer_state": "gepa/",
            "append_only_logs_fsync": True,
            "selection_transcripts_exposed_to_reflection": False,
        },
    }
    environment = {
        "created_at": _utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "server_metadata": server_metadata,
        "openai_api_key": "present" if os.environ.get("OPENAI_API_KEY") else "local-placeholder",
        "secrets_recorded": False,
    }

    output_dir.mkdir(parents=True)
    for dirname in ("candidates", "task-runs", "gepa"):
        (output_dir / dirname).mkdir()
    _write_json(output_dir / "manifest.json", manifest, exclusive=True)
    _write_json(output_dir / "environment.json", environment, exclusive=True)
    _write_json(
        output_dir / "status.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "phase": phase,
            "status": "running",
            "updated_at": _utc_now(),
            "evaluation_counts": {},
        },
        exclusive=True,
    )
    log = ExperimentLog(output_dir, run_id, phase)
    log.event("run_started", phase=phase)
    return log, manifest


def _run_canary(
    config: OvernightConfig,
    prompt: str,
    discovery: Sequence[Mapping[str, str]],
    log: ExperimentLog,
) -> dict[str, object]:
    example = next(item for item in discovery if item["case_name"] == config.canary_case)
    score, side_info = FormalizerEvaluator(config, log)(prompt, example)
    candidate_sha256 = log.materialize_candidate(prompt)
    (log.root / "best_prompt.txt").write_text(prompt, encoding="utf-8")
    return {
        "schema_version": 1,
        "run_id": log.run_id,
        "phase": "canary",
        "status": "completed",
        "case_name": config.canary_case,
        "candidate_sha256": candidate_sha256,
        "score": score,
        "feedback_sha256": _sha256_bytes(_canonical_json(side_info).encode()),
        "feedback_rendered_characters": reflection_feedback_character_count(side_info),
        "model_calls": "task-model only; no reflection call",
    }


def _run_optimization(
    config: OvernightConfig,
    prompt: str,
    discovery: list[dict[str, str]],
    selection: list[dict[str, str]],
    log: ExperimentLog,
) -> dict[str, object]:
    evaluator = FormalizerEvaluator(config, log)
    reflection_lm = LoggedReflectionLM(config.reflection_model, log)
    signal_stopper = SignalStopper()
    try:
        gepa_config = GEPAConfig(
            engine=EngineConfig(
                run_dir=str(log.root / "gepa"),
                seed=config.optimizer.seed,
                max_metric_calls=config.optimizer.max_metric_calls,
                max_candidate_proposals=config.optimizer.max_candidate_proposals,
                parallel=False,
                max_workers=1,
                raise_on_exception=True,
                cache_evaluation=config.optimizer.cache_evaluation,
                cache_evaluation_storage="disk",
                acceptance_criterion=config.optimizer.acceptance_criterion,
            ),
            reflection=ReflectionConfig(
                reflection_lm=reflection_lm,
                reflection_minibatch_size=config.optimizer.reflection_minibatch_size,
                perfect_score=1.0,
                skip_perfect_score=False,
            ),
            tracking=TrackingConfig(),
            stop_callbacks=(
                TimeoutStopCondition(config.optimizer.max_wall_time_s),
                signal_stopper,
            ),
        )
        result = optimize_anything(
            seed_candidate=prompt,
            evaluator=evaluator,
            dataset=discovery,
            valset=selection,
            objective=OPTIMIZATION_OBJECTIVE,
            background=OPTIMIZATION_BACKGROUND,
            config=gepa_config,
        )
    finally:
        signal_stopper.cleanup()

    for candidate in result.candidates:
        if isinstance(candidate, str):
            log.materialize_candidate(candidate)
        elif isinstance(candidate, dict):
            for value in candidate.values():
                if isinstance(value, str):
                    log.materialize_candidate(value)
    best_prompt = result.best_candidate
    if not isinstance(best_prompt, str):
        raise FeedbackIntegrityError("GEPA returned a non-string best candidate")
    best_sha256 = log.materialize_candidate(best_prompt)
    (log.root / "best_prompt.txt").write_text(best_prompt, encoding="utf-8")
    best_score = result.val_aggregate_scores[result.best_idx]
    return {
        "schema_version": 1,
        "run_id": log.run_id,
        "phase": "optimize",
        "status": "completed",
        "gepa_result": result.to_dict(),
        "best_idx": result.best_idx,
        "best_score": best_score,
        "best_candidate_sha256": best_sha256,
        "total_metric_calls": result.total_metric_calls,
        "candidate_count": len(result.candidates),
        "reflection_usage": {
            "input_tokens": reflection_lm.total_tokens_in,
            "output_tokens": reflection_lm.total_tokens_out,
            "cost_usd": reflection_lm.total_cost,
            "calls": reflection_lm.call_index,
        },
    }


def _execute(
    *,
    config: OvernightConfig,
    config_path: Path,
    output_dir: Path,
    phase: Phase,
    validated: tuple[Path, Path, Path, list[dict[str, str]], list[dict[str, str]]],
) -> int:
    dataset_path, panel_path, prompt_path, discovery, selection = validated
    server_metadata = _server_metadata(config)
    log, _manifest = _prepare_run(
        config=config,
        config_path=config_path,
        output_dir=output_dir,
        phase=phase,
        dataset_path=dataset_path,
        panel_path=panel_path,
        prompt_path=prompt_path,
        discovery=discovery,
        selection=selection,
        server_metadata=server_metadata,
    )
    prompt = prompt_path.read_text(encoding="utf-8")
    os.environ["OPENAI_BASE_URL"] = config.task_model.endpoint
    os.environ.setdefault("OPENAI_API_KEY", "local-llama-cpp")

    try:
        result = (
            _run_canary(config, prompt, discovery, log)
            if phase == "canary"
            else _run_optimization(config, prompt, discovery, selection, log)
        )
        result["finished_at"] = _utc_now()
        result["evaluation_counts"] = dict(sorted(log.counts.items()))
        _write_json(log.root / "result.json", result, exclusive=True)
        _write_json(
            log.root / "status.json",
            {
                "schema_version": 1,
                "run_id": log.run_id,
                "phase": phase,
                "status": "completed",
                "finished_at": result["finished_at"],
                "evaluation_counts": dict(sorted(log.counts.items())),
            },
        )
        log.event("run_completed", phase=phase)
        return 0
    except KeyboardInterrupt as error:
        terminal_status = "cancelled"
        exit_code = 130
        caught: BaseException = error
    except Exception as error:  # noqa: BLE001 - terminal status must survive every run failure
        terminal_status = "failed"
        exit_code = 1
        caught = error
    _write_json(
        log.root / "status.json",
        {
            "schema_version": 1,
            "run_id": log.run_id,
            "phase": phase,
            "status": terminal_status,
            "finished_at": _utc_now(),
            "evaluation_counts": dict(sorted(log.counts.items())),
            "error_type": type(caught).__name__,
            "error": str(caught),
        },
    )
    log.event(
        "run_stopped",
        phase=phase,
        status=terminal_status,
        error_type=type(caught).__name__,
        error=str(caught),
    )
    print(f"gepa-runner: {type(caught).__name__}: {caught}", file=sys.stderr)
    return exit_code


def _dry_run(
    config: OvernightConfig,
    config_path: Path,
    output_dir: Path,
    phase: Phase,
    validated: tuple[Path, Path, Path, list[dict[str, str]], list[dict[str, str]]],
) -> dict[str, object]:
    dataset_path, panel_path, prompt_path, discovery, selection = validated
    server_metadata = _server_metadata(config)
    return {
        "dry_run": True,
        "phase": phase,
        "config": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "output_dir": str(output_dir.resolve()),
        "dataset": str(dataset_path),
        "panel_split": str(panel_path),
        "seed_prompt": str(prompt_path),
        "discovery_cases": [example["case_name"] for example in discovery],
        "selection_cases": [example["case_name"] for example in selection],
        "canary_case": config.canary_case,
        "task_model": config.task_model.model_dump(mode="json"),
        "reflection_model": config.reflection_model.model_dump(mode="json"),
        "execution": config.execution.model_dump(mode="json"),
        "optimizer": config.optimizer.model_dump(mode="json"),
        "server_metadata": server_metadata,
        "model_calls": 0,
        "execute_requires": ["--execute", "--approve-model-calls"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("canary", "optimize"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approve-model-calls", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    phase: Phase = args.phase
    try:
        config = _load_config(config_path)
        validated = _validate_inputs(config)
        if not args.execute:
            print(
                json.dumps(
                    _dry_run(config, config_path, output_dir, phase, validated),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if not args.approve_model_calls:
            raise ValueError("--execute also requires --approve-model-calls")
        return _execute(
            config=config,
            config_path=config_path,
            output_dir=output_dir,
            phase=phase,
            validated=validated,
        )
    except (FeedbackIntegrityError, FileExistsError, ValueError) as error:
        print(f"gepa-runner: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
