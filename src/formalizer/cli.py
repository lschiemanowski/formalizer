"""Command-line interface for Formalizer."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from formalizer.run import formalize
from formalizer.settings import RunSettings, Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="formalizer",
        description="A simple Lean formalization agent",
    )
    parser.add_argument(
        "problem",
        help="Mathematical problem to formalize",
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = Settings(
            model_name=args.model,
            run=RunSettings(runs_dir=args.runs_dir),
        )
        result = asyncio.run(formalize(args.problem, settings))
    except KeyboardInterrupt:
        print("formalizer: interrupted", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001 - The CLI converts runtime failures to exit code 1.
        print(f"formalizer: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    code = result.output.code
    sys.stdout.write(code)
    if not code.endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
