from io import StringIO

from pydantic_evals.evaluators import EvaluatorSpec
from pydantic_evals.reporting import (
    EvaluationReport,
    EvaluationResult,
    ReportCase,
    ReportCaseFailure,
)

from formalizer.agent import Submission
from formalizer.eval import EvalOutput
from formalizer.eval.reporting import print_outcome_summary, summarize_outcomes
from formalizer.lean import LeanResult
from formalizer.problem import FormalizationProblem

PROBLEM = FormalizationProblem(source="def FormalizerProblem.Target : Prop := True\n")
OUTPUT = EvalOutput(
    run_id="7e42281f-548d-4a0e-bb50-df63dd3a5b62",
    submission=Submission(
        code="proof",
        check=LeanResult(stdout="", stderr="", exit_code=0, duration_s=0.1),
    ),
)


def report_case(name: str, verified: bool | None) -> ReportCase:
    assertions = {}
    if verified is not None:
        assertions["LeanVerified"] = EvaluationResult(
            name="LeanVerified",
            value=verified,
            reason=None,
            source=EvaluatorSpec(name="LeanVerified", arguments=None),
        )
    return ReportCase(
        name=name,
        inputs=PROBLEM,
        metadata=None,
        expected_output=None,
        output=OUTPUT,
        metrics={},
        attributes={},
        scores={},
        labels={},
        assertions=assertions,
        task_duration=0.1,
        total_duration=0.1,
    )


def report_failure(name: str, error_message: str) -> ReportCaseFailure:
    return ReportCaseFailure(
        name=name,
        inputs=PROBLEM,
        metadata=None,
        expected_output=None,
        error_message=error_message,
        error_stacktrace="traceback",
    )


def test_outcome_summary_counts_task_failures_in_the_success_rate_denominator() -> None:
    report = EvaluationReport(
        name="mixed-outcomes",
        cases=[
            report_case("verified-1", True),
            report_case("verified-2", True),
            report_case("rejected", False),
            report_case("unscored", None),
        ],
        failures=[
            report_failure("usage-1", "UsageLimitExceeded: request limit reached"),
            report_failure("usage-2", "UsageLimitExceeded: token limit reached"),
            report_failure("model", "UnexpectedModelBehavior: invalid response"),
        ],
    )

    summary = summarize_outcomes(report)

    assert summary.selected_attempts == 7
    assert summary.lean_verified == 2
    assert summary.lean_verified_rate == 2 / 7
    assert summary.rejected_submissions == 1
    assert summary.unscored_cases == 1
    assert summary.task_failures == 3
    assert summary.failure_types == (
        ("UnexpectedModelBehavior", 1),
        ("UsageLimitExceeded", 2),
    )


def test_outcome_summary_prints_the_overall_rate_and_failure_breakdown() -> None:
    report = EvaluationReport(
        name="mixed-outcomes",
        cases=[report_case("verified", True), report_case("rejected", False)],
        failures=[report_failure("usage", "UsageLimitExceeded: token limit reached")],
    )
    output = StringIO()

    print_outcome_summary(report, file=output)

    assert (
        output.getvalue()
        == """\
Formalizer outcomes
Selected attempts: 3
Lean verified: 1/3 (33.3%)
Rejected submissions: 1
Unscored completed cases: 0
Task failures: 1
Failure types:
  UsageLimitExceeded: 1
"""
    )


def test_empty_outcome_summary_has_a_zero_success_rate() -> None:
    summary = summarize_outcomes(EvaluationReport(name="empty", cases=[]))

    assert summary.selected_attempts == 0
    assert summary.lean_verified_rate == 0.0
