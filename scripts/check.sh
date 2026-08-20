#!/usr/bin/env bash
set -Eeuo pipefail

run_check() {
    printf '\n\033[1m>>>'
    printf ' %q' "$@"
    printf '\033[0m\n\n'
    "$@"
}

run_check uv lock --check
run_check uv run ruff format --check .
run_check uv run ruff check .
run_check uv run ty check
run_check uv run pytest
