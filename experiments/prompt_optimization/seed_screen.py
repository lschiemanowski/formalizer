#!/usr/bin/env python3
"""Run a provenance-rich, serial seed-prompt screen over approved training cases."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from formalizer.agent import _INSTRUCTIONS
from formalizer.eval import load_formalizer_dataset
from formalizer.eval.artifacts import read_evaluation_report
from formalizer.settings import RunSettings

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "experiments/prompt_optimization/screening_panel_v1.json"
RUNS_ROOT = REPOSITORY_ROOT / "runs/prompt-optimization"
INTEGER_USAGE_FIELDS = (
    "model_responses",
    "input_tokens",
    "cache_write_tokens",
    "cache_read_tokens",
    "output_tokens",
    "reasoning_tokens",
    "pydantic_costed_responses",
    "provider_costed_responses",
)


class DatasetConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    sha256: str


class SeedPromptConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    sha256: str


class TaskModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    endpoint: str
    model: str
    server_model_id: str
    tool_choice: Literal["auto", "required"]
    max_tokens: int = Field(ge=1)
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)

    @model_validator(mode="after")
    def one_sampling_control(self) -> TaskModelConfig:
        if self.temperature is not None and self.top_p is not None:
            raise ValueError("temperature and top_p are mutually exclusive")
        return self


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    repeat: int = Field(ge=1)
    max_concurrency: int = Field(ge=1)
    run_timeout_s: int = Field(ge=1)
    request_limit: int = Field(ge=1)
    output_tokens_limit: int = Field(ge=1)
    total_tokens_limit: int | None = Field(default=None, ge=1)
    infrastructure_retries: int = Field(ge=0)


class PanelCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_name: str
    provisional_role: Literal[
        "smoke-canary",
        "regression-anchor",
        "library-navigation",
        "proof-repair",
        "stretch",
    ]
    behavior_tags: tuple[str, ...]


class ParentRunConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    run_id: str
    manifest_sha256: str
    status_sha256: str


class ScreeningConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    name: str
    dataset: DatasetConfig
    seed_prompt: SeedPromptConfig
    task_model: TaskModelConfig
    execution: ExecutionConfig
    panel: tuple[PanelCase, ...]
    parent_run: ParentRunConfig | None = None

    @model_validator(mode="after")
    def validate_panel_shape(self) -> ScreeningConfig:
        names = [case.case_name for case in self.panel]
        if len(names) != len(set(names)):
            raise ValueError("panel case names must be unique")
        canaries = [case for case in self.panel if case.provisional_role == "smoke-canary"]
        if len(canaries) != 1:
            raise ValueError("panel must contain exactly one smoke-canary")
        if self.execution.repeat != 1:
            raise ValueError("seed-screen v1 requires repeat=1; repeat in a new recorded run")
        if self.execution.max_concurrency != 1:
            raise ValueError("seed-screen v1 requires max_concurrency=1")
        return self


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_config(path: Path) -> ScreeningConfig:
    return ScreeningConfig.model_validate_json(path.read_bytes())


def _resolve_repository_path(path: str) -> Path:
    resolved = (REPOSITORY_ROOT / path).resolve()
    if not resolved.is_relative_to(REPOSITORY_ROOT):
        raise ValueError(f"path escapes repository: {path}")
    return resolved


def _validate_inputs(config: ScreeningConfig) -> tuple[Path, Path]:
    dataset_path = _resolve_repository_path(config.dataset.path)
    prompt_path = _resolve_repository_path(config.seed_prompt.path)
    if dataset_path != REPOSITORY_ROOT / "datasets/basic-problems-v1.yaml":
        raise ValueError("seed-screen v1 only permits datasets/basic-problems-v1.yaml")
    if _sha256_file(dataset_path) != config.dataset.sha256:
        raise ValueError("dataset SHA-256 does not match the frozen screening config")
    prompt = prompt_path.read_text(encoding="utf-8")
    if _sha256_bytes(prompt.encode()) != config.seed_prompt.sha256:
        raise ValueError("seed prompt SHA-256 does not match the frozen screening config")
    if prompt != _INSTRUCTIONS:
        raise ValueError("seed prompt snapshot no longer matches Formalizer's default instructions")
    if RunSettings().run_timeout_s != config.execution.run_timeout_s:
        raise ValueError("configured run_timeout_s does not match Formalizer's effective default")

    dataset = load_formalizer_dataset(dataset_path)
    cases = {case.name: case for case in dataset.cases}
    for panel_case in config.panel:
        case = cases.get(panel_case.case_name)
        if case is None:
            raise ValueError(f"unknown panel case: {panel_case.case_name}")
        split = case.metadata.split if case.metadata is not None else None
        if split != "train":
            raise ValueError(f"optimization case is not in the train split: {panel_case.case_name}")
    if config.parent_run is not None:
        parent_path = _resolve_repository_path(config.parent_run.path)
        if not parent_path.is_relative_to(RUNS_ROOT.resolve()):
            raise ValueError("parent run must be under runs/prompt-optimization")
        parent_manifest_path = parent_path / "manifest.json"
        parent_status_path = parent_path / "status.json"
        if _sha256_file(parent_manifest_path) != config.parent_run.manifest_sha256:
            raise ValueError("parent manifest SHA-256 does not match the frozen screening config")
        if _sha256_file(parent_status_path) != config.parent_run.status_sha256:
            raise ValueError("parent status SHA-256 does not match the frozen screening config")
        parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
        parent_status = json.loads(parent_status_path.read_text(encoding="utf-8"))
        if parent_manifest.get("run_id") != config.parent_run.run_id:
            raise ValueError("parent manifest run id does not match the frozen screening config")
        if parent_status.get("run_id") != config.parent_run.run_id:
            raise ValueError("parent status run id does not match the frozen screening config")
        if parent_status.get("status") != "completed":
            raise ValueError("parent run did not complete cleanly")
    return dataset_path, prompt_path


def _cases_for_phase(config: ScreeningConfig, phase: str) -> tuple[PanelCase, ...]:
    if phase == "canary":
        return tuple(case for case in config.panel if case.provisional_role == "smoke-canary")
    if phase == "screen":
        return tuple(case for case in config.panel if case.provisional_role != "smoke-canary")
    return config.panel


def _budget_ceiling(config: ScreeningConfig, case_count: int) -> dict[str, int | None]:
    execution = config.execution
    attempts = case_count * execution.repeat
    return {
        "selected_attempts": attempts,
        "model_requests": attempts * execution.request_limit,
        "output_tokens": attempts * execution.output_tokens_limit,
        "total_tokens": (
            attempts * execution.total_tokens_limit
            if execution.total_tokens_limit is not None
            else None
        ),
        "wall_time_s": attempts * execution.run_timeout_s,
    }


def _eval_command(
    config: ScreeningConfig,
    *,
    dataset_path: Path,
    panel_case: PanelCase,
    output_dir: Path,
) -> list[str]:
    execution = config.execution
    model = config.task_model
    command = [
        sys.executable,
        "-m",
        "formalizer.eval.cli",
        "--dataset",
        str(dataset_path),
        "--model",
        model.model,
        "--tool-choice",
        model.tool_choice,
        "--name",
        f"{config.name}-{panel_case.case_name.rsplit('/', 1)[-1]}",
        "--output-dir",
        str(output_dir),
        "--repeat",
        str(execution.repeat),
        "--max-concurrency",
        str(execution.max_concurrency),
        "--max-tokens",
        str(model.max_tokens),
        "--request-limit",
        str(execution.request_limit),
        "--output-tokens-limit",
        str(execution.output_tokens_limit),
        "--infrastructure-retries",
        str(execution.infrastructure_retries),
        "--case",
        panel_case.case_name,
    ]
    if execution.total_tokens_limit is None:
        command.append("--no-total-tokens-limit")
    else:
        command.extend(("--total-tokens-limit", str(execution.total_tokens_limit)))
    if model.temperature is not None:
        command.extend(("--temperature", str(model.temperature)))
    if model.top_p is not None:
        command.extend(("--top-p", str(model.top_p)))
    return command


def _dry_run_plan(
    config: ScreeningConfig,
    *,
    config_path: Path,
    dataset_path: Path,
    phase: str,
) -> dict[str, object]:
    cases = _cases_for_phase(config, phase)
    placeholder = Path("<output-dir>")
    return {
        "dry_run": True,
        "config": str(config_path.relative_to(REPOSITORY_ROOT)),
        "config_sha256": _sha256_file(config_path),
        "phase": phase,
        "case_names": [case.case_name for case in cases],
        "case_count": len(cases),
        "task_model": config.task_model.model_dump(mode="json"),
        "execution": config.execution.model_dump(mode="json"),
        "aggregate_ceiling": _budget_ceiling(config, len(cases)),
        "first_command": _eval_command(
            config,
            dataset_path=dataset_path,
            panel_case=cases[0],
            output_dir=placeholder,
        ),
        "execute_requires": ["--execute", "--approve-model-calls", "--output-dir"],
    }


def _server_metadata(config: ScreeningConfig) -> dict[str, object]:
    request = urllib.request.Request(
        f"{config.task_model.endpoint.rstrip('/')}/models",
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.load(response)
    ids = {
        item.get("id")
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    }
    ids.update(
        item.get("model") or item.get("name")
        for item in payload.get("models", [])
        if isinstance(item, dict) and (item.get("model") or item.get("name"))
    )
    if config.task_model.server_model_id not in ids:
        raise ValueError(
            f"configured model id is absent from server metadata: {config.task_model.server_model_id}"
        )
    return payload


def _git_provenance(relevant_paths: Sequence[Path]) -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    commit = git("rev-parse", "HEAD").strip()
    status = git("status", "--porcelain=v1")
    diff = git("diff", "--binary", "HEAD")
    return {
        "commit": commit,
        "is_dirty": bool(status),
        "status_sha256": _sha256_bytes(status.encode()),
        "diff_sha256": _sha256_bytes(diff.encode()),
        "file_sha256s": {
            str(path.relative_to(REPOSITORY_ROOT)): _sha256_file(path) for path in relevant_paths
        },
    }


def _write_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    payload = f"{json.dumps(value, indent=2, sort_keys=True)}\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _usage_summary(runs_dir: Path) -> tuple[dict[str, object], list[str]]:
    totals = Counter[str]()
    pydantic_cost = Decimal(0)
    provider_cost = Decimal(0)
    pydantic_cost_seen = False
    provider_cost_seen = False
    paths: list[str] = []
    for manifest_path in sorted(runs_dir.glob("*/run.json")):
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths.append(str(manifest_path))
        usage = data.get("usage", {})
        for field in INTEGER_USAGE_FIELDS:
            totals[field] += int(usage.get(field, 0))
        if usage.get("pydantic_estimated_cost_usd") is not None:
            pydantic_cost += Decimal(str(usage["pydantic_estimated_cost_usd"]))
            pydantic_cost_seen = True
        if usage.get("provider_reported_cost_usd") is not None:
            provider_cost += Decimal(str(usage["provider_reported_cost_usd"]))
            provider_cost_seen = True
    summary: dict[str, object] = {field: totals[field] for field in INTEGER_USAGE_FIELDS}
    summary["agent_runs_including_retries"] = len(paths)
    summary["pydantic_estimated_cost_recorded_usd"] = (
        str(pydantic_cost) if pydantic_cost_seen else None
    )
    summary["provider_reported_cost_recorded_usd"] = (
        str(provider_cost) if provider_cost_seen else None
    )
    return summary, paths


def _failure_type(error: str) -> str:
    return error.split(":", 1)[0].strip() or "UnknownFailure"


def _classify_report(report_path: Path) -> dict[str, object]:
    report = read_evaluation_report(report_path)
    usage, manifest_paths = _usage_summary(report_path.parent / "runs")
    if report.cases:
        case = report.cases[0]
        assertion = case.assertions.get("LeanVerified")
        verified = bool(assertion and assertion.value)
        output = case.output
        run_id = output.run_id if output is not None else None
        task_run_path = report_path.parent / "runs" / run_id if run_id is not None else None
        return {
            "terminal_status": "verified" if verified else "model_failed",
            "score": 1.0 if verified else 0.0,
            "verified": verified,
            "infrastructure_invalid": False,
            "failure_type": None if verified else "RejectedSubmission",
            "error": None,
            "task_run_id": run_id,
            "task_run_path": str(task_run_path) if task_run_path is not None else None,
            "all_run_manifest_paths": manifest_paths,
            "usage": usage,
            "duration_s": case.task_duration,
        }
    if not report.failures:
        raise ValueError("report contains neither completed cases nor failures")
    error = report.failures[0].error_message
    failure_type = _failure_type(error)
    infrastructure_invalid = failure_type in {
        "LeanInfrastructureError",
        "ModelAPIError",
        "ModelHTTPError",
        "SearchInfrastructureError",
    }
    return {
        "terminal_status": "infrastructure_invalid" if infrastructure_invalid else "model_failed",
        "score": None if infrastructure_invalid else 0.0,
        "verified": False,
        "infrastructure_invalid": infrastructure_invalid,
        "failure_type": failure_type,
        "error": error,
        "task_run_id": None,
        "task_run_path": None,
        "all_run_manifest_paths": manifest_paths,
        "usage": usage,
        "duration_s": None,
    }


def _execute(
    config: ScreeningConfig,
    *,
    config_path: Path,
    dataset_path: Path,
    prompt_path: Path,
    phase: str,
    output_dir: Path,
) -> int:
    resolved_output = output_dir.resolve()
    if not resolved_output.is_relative_to(RUNS_ROOT.resolve()):
        raise ValueError(f"output directory must be under {RUNS_ROOT}")
    if resolved_output.exists():
        raise FileExistsError(f"output directory already exists: {resolved_output}")

    server_metadata = _server_metadata(config)
    cases = _cases_for_phase(config, phase)
    run_id = str(uuid4())
    candidate_hash = config.seed_prompt.sha256
    relevant_paths = (
        config_path,
        dataset_path,
        prompt_path,
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "pyproject.toml",
        REPOSITORY_ROOT / "uv.lock",
        REPOSITORY_ROOT / "src/formalizer/agent.py",
        REPOSITORY_ROOT / "src/formalizer/eval/cli.py",
        REPOSITORY_ROOT / "src/formalizer/eval/runner.py",
    )
    git = _git_provenance(relevant_paths)
    manifest = {
        "schema_version": 1,
        "phase": "seed_screen",
        "screen_phase": phase,
        "run_id": run_id,
        "created_at": _utc_now(),
        "status": "launched",
        "parent_run": (
            config.parent_run.model_dump(mode="json") if config.parent_run is not None else None
        ),
        "config_path": str(config_path.relative_to(REPOSITORY_ROOT)),
        "config_sha256": _sha256_file(config_path),
        "git": git,
        "dependencies": {
            "formalizer": importlib.metadata.version("formalizer"),
            "gepa": importlib.metadata.version("gepa"),
            "lock_sha256": _sha256_file(REPOSITORY_ROOT / "uv.lock"),
        },
        "dataset": {
            **config.dataset.model_dump(mode="json"),
            "selected_cases": [case.case_name for case in cases],
            "split_assertion": "train",
        },
        "seed_prompt": config.seed_prompt.model_dump(mode="json"),
        "task_model": config.task_model.model_dump(mode="json"),
        "reflection_model": None,
        "gepa": {"enabled": False},
        "execution": config.execution.model_dump(mode="json"),
        "aggregate_ceiling": _budget_ceiling(config, len(cases)),
    }
    environment = {
        "created_at": _utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "server_metadata": server_metadata,
        "openai_api_key_source": (
            "environment" if os.environ.get("OPENAI_API_KEY") else "local-placeholder"
        ),
    }

    resolved_output.mkdir(parents=True)
    (resolved_output / "candidates").mkdir()
    (resolved_output / "task-runs").mkdir()
    shutil.copyfile(prompt_path, resolved_output / "candidates" / f"{candidate_hash}.txt")
    _write_json(resolved_output / "manifest.json", manifest, exclusive=True)
    _write_json(resolved_output / "environment.json", environment, exclusive=True)
    _write_json(
        resolved_output / "status.json",
        {"run_id": run_id, "status": "running", "updated_at": _utc_now()},
    )
    _append_jsonl(
        resolved_output / "events.jsonl",
        {"event": "screen_started", "run_id": run_id, "at": _utc_now(), "phase": phase},
    )

    counts: Counter[str] = Counter()
    child_env = os.environ.copy()
    child_env["OPENAI_BASE_URL"] = config.task_model.endpoint
    child_env.setdefault("OPENAI_API_KEY", "local-llama-cpp")

    def record_stopped(error: BaseException, *, status: str, exit_code: int) -> int:
        _append_jsonl(
            resolved_output / "events.jsonl",
            {"event": "screen_stopped", "at": _utc_now(), "status": status, "error": str(error)},
        )
        _write_json(
            resolved_output / "status.json",
            {
                "run_id": run_id,
                "status": status,
                "updated_at": _utc_now(),
                "counts": dict(counts),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        return exit_code

    try:
        for index, panel_case in enumerate(cases):
            evaluation_id = f"{run_id}:{index:03d}:r0"
            case_root = resolved_output / "task-runs" / panel_case.case_name.replace("/", "_")
            case_root.mkdir()
            eval_output = case_root / "evaluation"
            command = _eval_command(
                config,
                dataset_path=dataset_path,
                panel_case=panel_case,
                output_dir=eval_output,
            )
            started_at = _utc_now()
            base_record = {
                "evaluation_id": evaluation_id,
                "candidate_sha256": candidate_hash,
                "case_name": panel_case.case_name,
                "provisional_role": panel_case.provisional_role,
                "behavior_tags": list(panel_case.behavior_tags),
                "repetition": 0,
                "seed": None,
            }
            _append_jsonl(
                resolved_output / "evaluations.jsonl",
                {**base_record, "event": "started", "started_at": started_at},
            )
            _append_jsonl(
                resolved_output / "events.jsonl",
                {
                    "event": "case_started",
                    "evaluation_id": evaluation_id,
                    "case_name": panel_case.case_name,
                    "at": started_at,
                    "command": command,
                },
            )
            print(f"[{index + 1}/{len(cases)}] {panel_case.case_name}", flush=True)
            with (
                (case_root / "stdout.log").open("w", encoding="utf-8") as stdout,
                (case_root / "stderr.log").open("w", encoding="utf-8") as stderr,
            ):
                completed = subprocess.run(
                    command,
                    cwd=REPOSITORY_ROOT,
                    env=child_env,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                    text=True,
                )
            report_path = eval_output / "report.json"
            if not report_path.is_file():
                result = {
                    "terminal_status": "integrity_error",
                    "score": None,
                    "verified": False,
                    "infrastructure_invalid": False,
                    "failure_type": "MissingReport",
                    "error": f"formalizer-eval exited {completed.returncode} without report.json",
                    "task_run_id": None,
                    "task_run_path": None,
                    "all_run_manifest_paths": [],
                    "usage": {},
                    "duration_s": None,
                }
            else:
                result = _classify_report(report_path)
            counts[str(result["terminal_status"])] += 1
            finished_at = _utc_now()
            _append_jsonl(
                resolved_output / "evaluations.jsonl",
                {
                    **base_record,
                    **result,
                    "event": "finished",
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "child_exit_code": completed.returncode,
                    "feedback_sha256": None,
                },
            )
            _append_jsonl(
                resolved_output / "events.jsonl",
                {
                    "event": "case_finished",
                    "evaluation_id": evaluation_id,
                    "case_name": panel_case.case_name,
                    "at": finished_at,
                    "terminal_status": result["terminal_status"],
                },
            )
            if result["terminal_status"] == "integrity_error":
                raise RuntimeError(str(result["error"]))
    except KeyboardInterrupt as error:
        return record_stopped(error, status="cancelled", exit_code=130)
    except Exception as error:  # noqa: BLE001 - terminal evidence must survive every failure
        return record_stopped(error, status="failed", exit_code=1)

    final_status = "completed_with_invalid" if counts["infrastructure_invalid"] else "completed"
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "phase": phase,
        "status": final_status,
        "finished_at": _utc_now(),
        "candidate_sha256": candidate_hash,
        "counts": dict(counts),
    }
    _write_json(resolved_output / "result.json", result, exclusive=True)
    _write_json(resolved_output / "status.json", result)
    _append_jsonl(
        resolved_output / "events.jsonl",
        {"event": "screen_finished", "at": result["finished_at"], "status": final_status},
    )
    return 2 if final_status == "completed_with_invalid" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("canary", "screen", "all"), default="canary")
    parser.add_argument(
        "--execute", action="store_true", help="Perform model calls; default is dry-run"
    )
    parser.add_argument(
        "--approve-model-calls",
        action="store_true",
        help="Required with --execute after the launch contract is approved",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config.resolve()
    try:
        config = _load_config(config_path)
        dataset_path, prompt_path = _validate_inputs(config)
        if not args.execute:
            plan = _dry_run_plan(
                config,
                config_path=config_path,
                dataset_path=dataset_path,
                phase=args.phase,
            )
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        if not args.approve_model_calls:
            raise ValueError("--execute also requires --approve-model-calls")
        if args.output_dir is None:
            raise ValueError("--execute requires --output-dir")
        if args.phase == "screen" and config.parent_run is None:
            raise ValueError("screen execution requires a frozen parent_run")
        return _execute(
            config,
            config_path=config_path,
            dataset_path=dataset_path,
            prompt_path=prompt_path,
            phase=args.phase,
            output_dir=args.output_dir,
        )
    except Exception as error:  # noqa: BLE001
        print(f"seed-screen: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
