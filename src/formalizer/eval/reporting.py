import sys
from collections import Counter
from dataclasses import dataclass
from typing import TextIO

from formalizer.eval.artifacts import FormalizerEvaluationReport

_LEAN_VERIFIED_ASSERTION = "LeanVerified"


@dataclass(frozen=True, slots=True)
class FormalizerOutcomeSummary:
    selected_attempts: int
    lean_verified: int
    rejected_submissions: int
    unscored_cases: int
    task_failures: int
    failure_types: tuple[tuple[str, int], ...]

    @property
    def lean_verified_rate(self) -> float:
        if self.selected_attempts == 0:
            return 0.0
        return self.lean_verified / self.selected_attempts


def _failure_type(error_message: str) -> str:
    error_type, separator, _ = error_message.partition(":")
    if separator and error_type.strip():
        return error_type.strip()
    return "Unknown"


def summarize_outcomes(report: FormalizerEvaluationReport) -> FormalizerOutcomeSummary:
    lean_verified = 0
    rejected_submissions = 0
    unscored_cases = 0

    for case in report.cases:
        assertion = case.assertions.get(_LEAN_VERIFIED_ASSERTION)
        if assertion is None:
            unscored_cases += 1
        elif assertion.value:
            lean_verified += 1
        else:
            rejected_submissions += 1

    failure_types = Counter(_failure_type(failure.error_message) for failure in report.failures)
    return FormalizerOutcomeSummary(
        selected_attempts=len(report.cases) + len(report.failures),
        lean_verified=lean_verified,
        rejected_submissions=rejected_submissions,
        unscored_cases=unscored_cases,
        task_failures=len(report.failures),
        failure_types=tuple(sorted(failure_types.items())),
    )


def print_outcome_summary(
    report: FormalizerEvaluationReport,
    *,
    file: TextIO | None = None,
) -> None:
    summary = summarize_outcomes(report)
    destination = file if file is not None else sys.stdout

    print("Formalizer outcomes", file=destination)
    print(f"Selected attempts: {summary.selected_attempts}", file=destination)
    print(
        f"Lean verified: {summary.lean_verified}/{summary.selected_attempts} "
        f"({summary.lean_verified_rate:.1%})",
        file=destination,
    )
    print(f"Rejected submissions: {summary.rejected_submissions}", file=destination)
    print(f"Unscored completed cases: {summary.unscored_cases}", file=destination)
    print(f"Task failures: {summary.task_failures}", file=destination)
    if summary.failure_types:
        print("Failure types:", file=destination)
        for error_type, count in summary.failure_types:
            print(f"  {error_type}: {count}", file=destination)
