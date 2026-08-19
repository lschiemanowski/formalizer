"""Command-line interface for Formalizer."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="formalizer",
        description="A simple Lean formalization agent",
    )


def main() -> None:
    parser = build_parser()
    parser.parse_args()


if __name__ == "__main__":
    main()
