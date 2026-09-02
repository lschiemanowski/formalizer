#!/usr/bin/env python3
"""Validate the local GEPA dependency and Formalizer optimization dataset boundary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import sys
import tomllib
from pathlib import Path

EXPECTED_GEPA_VERSION = "0.1.4"
EXPECTED_DEPENDENCY = "gepa[full]==0.1.4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/basic-problems-v1.yaml"),
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        dest="case_names",
        help="Allowed optimization case name; repeat for every selected case.",
    )
    args = parser.parse_args()

    root = _repository_root()
    problems: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        marker = "OK" if condition else "FAIL"
        print(f"[{marker}] {label}{f': {detail}' if detail else ''}")
        if not condition:
            problems.append(label)

    try:
        installed = importlib.metadata.version("gepa")
    except importlib.metadata.PackageNotFoundError:
        installed = "missing"
    check("gepa version is pinned", installed == EXPECTED_GEPA_VERSION, installed)

    try:
        from gepa.optimize_anything import (  # noqa: F401
            EngineConfig,
            GEPAConfig,
            ReflectionConfig,
            TrackingConfig,
            optimize_anything,
        )
    except ImportError as error:
        check("GEPA 0.1.4 API imports", False, str(error))
    else:
        check("GEPA 0.1.4 API imports", True)

    pyproject = root / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    dependencies = project.get("dependency-groups", {}).get("prompt-optimization", [])
    check("prompt-optimization dependency group", EXPECTED_DEPENDENCY in dependencies)

    dataset_path = args.dataset if args.dataset.is_absolute() else root / args.dataset
    check("dataset exists", dataset_path.is_file(), str(dataset_path))
    if dataset_path.is_file():
        print(f"[INFO] dataset sha256: {_sha256(dataset_path)}")
        try:
            from formalizer.eval import load_formalizer_dataset

            dataset = load_formalizer_dataset(dataset_path)
        except Exception as error:  # noqa: BLE001 - preflight must report every load failure
            check("dataset loads through Formalizer", False, str(error))
        else:
            check("dataset loads through Formalizer", True, dataset.name)
            cases = {case.name: case for case in dataset.cases}
            train_count = sum(
                case.metadata is not None and case.metadata.split == "train"
                for case in dataset.cases
            )
            check("dataset contains training cases", train_count > 0, str(train_count))
            for case_name in args.case_names:
                case = cases.get(case_name)
                check(f"case exists: {case_name}", case is not None)
                if case is not None:
                    split = case.metadata.split if case.metadata is not None else "missing"
                    check(f"case is optimization-safe: {case_name}", split == "train", split)
            if not args.case_names:
                print("[INFO] no --case values supplied; install-only preflight")

    skill_link = root / ".agents/skills/formalizer-gepa-prompt-optimization"
    check("project-local skill link exists", skill_link.is_symlink(), str(skill_link))
    if skill_link.is_symlink():
        check(
            "project-local skill link resolves",
            skill_link.resolve().is_dir(),
            str(skill_link.resolve()),
        )

    if problems:
        print(f"\n{len(problems)} preflight check(s) failed.")
        return 1
    print("\nPreflight passed. Supply explicit --case values before a real run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
