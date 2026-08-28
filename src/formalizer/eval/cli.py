import argparse
import asyncio
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from subprocess import CalledProcessError, run

from pydantic_ai import ModelSettings, UsageLimits
from pydantic_evals.reporting import EvaluationReport

from formalizer.eval.artifacts import write_evaluation_report
from formalizer.eval.datasets import load_formalizer_dataset, select_formalizer_dataset
from formalizer.eval.evaluators import LeanVerified
from formalizer.eval.models import EvalOutput, ProblemMetadata
from formalizer.eval.observability import configure_logfire
from formalizer.eval.runner import evaluate_dataset
from formalizer.problem import FormalizationProblem
from formalizer.settings import RunSettings, Settings

DEFAULT_MAX_TOKENS = 8_192
DEFAULT_REQUEST_LIMIT = 30
DEFAULT_OUTPUT_TOKENS_LIMIT = 50_000
DEFAULT_TOTAL_TOKENS_LIMIT = 1_000_000
DEFAULT_INFRASTRUCTURE_RETRIES = 2


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def _non_blank(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("must not be blank")
    return stripped


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="formalizer-eval",
        description="Evaluate a Formalizer model on a Pydantic Evals dataset.",
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True, type=_non_blank)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeat", type=_positive_int, default=1)
    parser.add_argument(
        "--max-concurrency",
        type=_positive_int,
        default=1,
        help="Maximum number of case attempts evaluated concurrently.",
    )
    parser.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum output tokens for each model request.",
    )
    parser.add_argument(
        "--request-limit",
        type=_positive_int,
        default=DEFAULT_REQUEST_LIMIT,
        help="Maximum number of model requests per case.",
    )
    parser.add_argument(
        "--output-tokens-limit",
        type=_positive_int,
        default=DEFAULT_OUTPUT_TOKENS_LIMIT,
        help="Maximum cumulative output tokens per case.",
    )
    parser.add_argument(
        "--total-tokens-limit",
        type=_positive_int,
        default=DEFAULT_TOTAL_TOKENS_LIMIT,
        help="Maximum cumulative input and output tokens per case.",
    )
    parser.add_argument(
        "--infrastructure-retries",
        type=_nonnegative_int,
        default=DEFAULT_INFRASTRUCTURE_RETRIES,
        help="Retries after transient provider, Lean, or Loogle infrastructure failures.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "validation", "test"),
        help="Select a dataset split; repeat to select more than one.",
    )
    parser.add_argument(
        "--difficulty",
        action="append",
        choices=("easy", "medium", "hard"),
        help="Select a difficulty; repeat to select more than one.",
    )
    parser.add_argument(
        "--case",
        action="append",
        type=_non_blank,
        help="Select an exact case name; repeat to select more than one.",
    )
    parser.add_argument("--logfire", action="store_true")
    return parser


def _has_execution_failures(
    report: EvaluationReport[FormalizationProblem, EvalOutput, ProblemMetadata],
) -> bool:
    return bool(
        report.failures
        or report.report_evaluator_failures
        or any(case.evaluator_failures for case in report.cases)
    )


def _git_provenance() -> tuple[str | None, bool | None]:
    repository_root = next(
        (parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()),
        None,
    )
    if repository_root is None:
        return None, None

    try:
        commit = run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = run(
            ["git", "-C", str(repository_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, CalledProcessError):
        return None, None

    return commit, bool(status)


def _docker_image_id(image: str) -> str | None:
    try:
        return run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, CalledProcessError):
        return None


def _experiment_metadata(
    *,
    dataset_path: Path,
    dataset_name: str,
    settings: Settings,
    repeat: int,
    selection: dict[str, object],
    infrastructure_retries: int,
    max_concurrency: int,
) -> dict[str, object]:
    git_commit, git_dirty = _git_provenance()
    return {
        "model_name": settings.model_name,
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256(dataset_path.read_bytes()).hexdigest(),
        "repeat": repeat,
        "selection": selection,
        "budget": {
            "max_tokens": settings.model_settings.get("max_tokens"),
            "request_limit": settings.run.usage_limits.request_limit,
            "output_tokens_limit": settings.run.usage_limits.output_tokens_limit,
            "total_tokens_limit": settings.run.usage_limits.total_tokens_limit,
        },
        "retry_policy": {
            "infrastructure_retries": infrastructure_retries,
        },
        "execution": {
            "max_concurrency": max_concurrency,
        },
        "started_at": datetime.now(UTC).isoformat(),
        "formalizer_version": version("formalizer"),
        "pydantic_evals_version": version("pydantic-evals"),
        "logfire_version": version("logfire"),
        "formalizer_git_commit": git_commit,
        "formalizer_git_dirty": git_dirty,
        "docker_image": settings.sandbox.docker_image,
        "docker_image_id": _docker_image_id(settings.sandbox.docker_image),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    try:
        if args.output_dir.exists():
            raise FileExistsError(f"output directory already exists: {args.output_dir}")

        dataset = load_formalizer_dataset(args.dataset)
        splits = set(args.split or ())
        difficulties = set(args.difficulty or ())
        case_names = set(args.case or ())
        dataset = select_formalizer_dataset(
            dataset,
            splits=splits,
            difficulties=difficulties,
            case_names=case_names,
        )
        selection: dict[str, object] = {
            "splits": sorted(splits),
            "difficulties": sorted(difficulties),
            "case_names": sorted(case_names),
            "selected_case_count": len(dataset.cases),
        }
        settings = Settings(
            model_name=args.model,
            model_settings=ModelSettings(max_tokens=args.max_tokens),
            run=RunSettings(
                runs_dir=args.output_dir / "runs",
                usage_limits=UsageLimits(
                    request_limit=args.request_limit,
                    output_tokens_limit=args.output_tokens_limit,
                    total_tokens_limit=args.total_tokens_limit,
                ),
            ),
        )
        metadata = _experiment_metadata(
            dataset_path=args.dataset,
            dataset_name=dataset.name,
            settings=settings,
            repeat=args.repeat,
            selection=selection,
            infrastructure_retries=args.infrastructure_retries,
            max_concurrency=args.max_concurrency,
        )
        args.output_dir.mkdir(parents=True)

        if args.logfire:
            configure_logfire()

        dataset.add_evaluator(LeanVerified())
        report = asyncio.run(
            evaluate_dataset(
                dataset,
                settings,
                name=args.name,
                repeat=args.repeat,
                max_concurrency=args.max_concurrency,
                infrastructure_retries=args.infrastructure_retries,
                metadata=metadata,
            )
        )
        write_evaluation_report(args.output_dir / "report.json", report)
        report.print()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("formalizer-eval: interrupted", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001
        print(f"formalizer-eval: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    return 1 if _has_execution_failures(report) else 0


if __name__ == "__main__":
    raise SystemExit(main())
