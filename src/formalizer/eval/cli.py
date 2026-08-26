import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic_evals.reporting import EvaluationReport

from formalizer.eval.evaluators import LeanVerified
from formalizer.eval.models import EvalOutput, ProblemMetadata
from formalizer.eval.observability import configure_logfire
from formalizer.eval.runner import FormalizerDataset, evaluate_dataset
from formalizer.problem import FormalizationProblem
from formalizer.settings import RunSettings, Settings


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="formalizer-eval",
        description="Evaluate a Formalizer model on a Pydantic Evals dataset.",
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--repeat", type=_positive_int, default=1)
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    try:
        if args.logfire:
            configure_logfire()

        dataset = FormalizerDataset.from_file(args.dataset)
        dataset.add_evaluator(LeanVerified())
        settings = Settings(
            model_name=args.model,
            run=RunSettings(runs_dir=args.runs_dir),
        )
        report = asyncio.run(
            evaluate_dataset(
                dataset,
                settings,
                repeat=args.repeat,
                metadata={"model_name": settings.model_name},
            )
        )
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
