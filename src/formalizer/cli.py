"""Command-line interface for Formalizer."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from formalizer.lean import LeanResult
from formalizer.problem import FormalizationProblem
from formalizer.run import formalize
from formalizer.settings import RunSettings, Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="formalizer",
        description="A simple Lean formalization agent",
    )
    parser.add_argument(
        "problem",
        help="Trusted FormalizerProblem.lean source",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Pydantic AI model name",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory in which to store run artifacts (default: runs)",
    )
    return parser


def _rejection_diagnostic(result: LeanResult) -> str:
    diagnostic = "\n".join(
        output.strip()
        for output in (result.verification_error, result.stdout, result.stderr)
        if output and output.strip()
    )
    if diagnostic:
        return diagnostic
    if result.timed_out:
        return "Lean checking timed out"
    return f"Lean exited with status {result.exit_code} without diagnostics"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = Settings(
            model_name=args.model,
            run=RunSettings(runs_dir=args.runs_dir),
        )
        result = asyncio.run(
            formalize(
                FormalizationProblem(source=args.problem),
                settings,
            )
        )
    except KeyboardInterrupt:
        print("formalizer: interrupted", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001 - The CLI converts runtime failures to exit code 1.
        print(f"formalizer: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    submission = result.output
    if not submission.check.accepted:
        print(
            f"formalizer: submission rejected\n{_rejection_diagnostic(submission.check)}",
            file=sys.stderr,
        )
        return 1

    code = submission.code
    sys.stdout.write(code)
    if not code.endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
