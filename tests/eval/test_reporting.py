from datetime import UTC, datetime
from decimal import Decimal
from io import StringIO
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic_evals.evaluators import EvaluatorSpec, ReportEvaluatorContext
from pydantic_evals.reporting import (
    EvaluationReport,
    EvaluationResult,
    ReportCase,
    ReportCaseFailure,
    TableResult,
)

from formalizer.agent import Submission
from formalizer.eval import EvalOutput
from formalizer.eval.reporting import (
    ModelUsageAnalysis,
    print_outcome_summary,
    summarize_model_usage,
    summarize_outcomes,
)
from formalizer.lean import LeanResult
from formalizer.problem import FormalizationProblem
from formalizer.run import ModelUsage, RunManifest

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


def write_run_manifest(
    runs_dir: Path,
    *,
    run_id: UUID,
    status: Literal["verified", "failed"],
    usage: ModelUsage,
) -> None:
    run_dir = runs_dir / str(run_id)
    run_dir.mkdir(parents=True)
    timestamp = datetime.now(UTC)
    manifest = RunManifest(
        run_id=run_id,
        problem=PROBLEM,
        status=status,
        started_at=timestamp,
        finished_at=timestamp,
        usage=usage,
    )
    (run_dir / "run.json").write_text(
        f"{manifest.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )


def test_model_usage_summary_includes_failed_retry_runs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    write_run_manifest(
        runs_dir,
        run_id=UUID("9b0a1ddf-2ee5-47d7-a957-e748d29a5399"),
        status="failed",
        usage=ModelUsage(
            model_responses=1,
            input_tokens=100,
            output_tokens=10,
            reasoning_tokens=3,
            pydantic_estimated_cost_usd=Decimal("0.60"),
            pydantic_costed_responses=1,
            provider_reported_cost_usd=Decimal("0.50"),
            provider_costed_responses=1,
        ),
    )
    write_run_manifest(
        runs_dir,
        run_id=UUID("42e5d0e6-a53d-4a5f-914b-d70a24487062"),
        status="verified",
        usage=ModelUsage(
            model_responses=2,
            input_tokens=200,
            cache_read_tokens=120,
            output_tokens=20,
            reasoning_tokens=5,
            pydantic_estimated_cost_usd=Decimal("1.20"),
            pydantic_costed_responses=2,
            provider_reported_cost_usd=Decimal("1.00"),
            provider_costed_responses=2,
        ),
    )

    summary = summarize_model_usage(runs_dir)

    assert summary.agent_runs == 2
    assert summary.model_responses == 3
    assert summary.input_tokens == 300
    assert summary.cache_read_tokens == 120
    assert summary.output_tokens == 30
    assert summary.reasoning_tokens == 8
    assert summary.pydantic_estimated_cost_usd == Decimal("1.80")
    assert summary.pydantic_costed_responses == 3
    assert summary.provider_reported_cost_usd == Decimal("1.50")
    assert summary.provider_costed_responses == 3


def test_model_usage_analysis_reports_total_and_effective_cost(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    write_run_manifest(
        runs_dir,
        run_id=UUID("71713466-c699-44ff-9081-61208c3b2148"),
        status="failed",
        usage=ModelUsage(
            model_responses=1,
            pydantic_estimated_cost_usd=Decimal("0.60"),
            pydantic_costed_responses=1,
            provider_reported_cost_usd=Decimal("0.50"),
            provider_costed_responses=1,
        ),
    )
    write_run_manifest(
        runs_dir,
        run_id=UUID("6b79f77f-251d-4557-be25-a96de9f8036e"),
        status="verified",
        usage=ModelUsage(
            model_responses=2,
            pydantic_estimated_cost_usd=Decimal("1.20"),
            pydantic_costed_responses=2,
            provider_reported_cost_usd=Decimal("1.00"),
            provider_costed_responses=2,
        ),
    )
    report = EvaluationReport(name="retried", cases=[report_case("verified", True)])
    context = ReportEvaluatorContext(
        name=report.name,
        report=report,
        experiment_metadata=None,
    )

    analysis = ModelUsageAnalysis(runs_dir=runs_dir).evaluate(context)

    assert isinstance(analysis, TableResult)
    assert analysis.title == "Model usage and cost"
    values = {str(row[0]): row[1] for row in analysis.rows}
    assert values["Agent runs, including retries"] == 2
    assert values["Model responses"] == 3
    assert values["Cost basis"] == "provider-reported"
    assert values["Benchmark model cost"] == 1.5
    assert values["Mean cost per selected attempt"] == 1.5
    assert values["Effective cost per Lean-verified attempt"] == 1.5
    assert values["Provider cost coverage"] == "3/3 responses"


def test_model_usage_analysis_falls_back_to_complete_pydantic_estimate(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    write_run_manifest(
        runs_dir,
        run_id=UUID("bf07d5ba-a891-47ec-b489-b8f1e24d3212"),
        status="verified",
        usage=ModelUsage(
            model_responses=2,
            pydantic_estimated_cost_usd=Decimal("1.20"),
            pydantic_costed_responses=2,
            provider_reported_cost_usd=Decimal("0.50"),
            provider_costed_responses=1,
        ),
    )
    report = EvaluationReport(name="partial-provider-cost", cases=[])
    context = ReportEvaluatorContext(
        name=report.name,
        report=report,
        experiment_metadata=None,
    )

    analysis = ModelUsageAnalysis(runs_dir=runs_dir).evaluate(context)

    assert isinstance(analysis, TableResult)
    values = {str(row[0]): row[1] for row in analysis.rows}
    assert values["Cost basis"] == "pydantic-estimated"
    assert values["Benchmark model cost"] == 1.2
    assert values["Provider-reported cost (recorded)"] == 0.5
    assert values["Provider cost coverage"] == "1/2 responses"
    assert values["Mean cost per selected attempt"] is None
    assert values["Effective cost per Lean-verified attempt"] is None


def test_model_usage_analysis_does_not_treat_missing_cost_as_zero(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    write_run_manifest(
        runs_dir,
        run_id=UUID("2ff27cb8-112e-4ceb-ac2e-468b9cd1a7fe"),
        status="verified",
        usage=ModelUsage(model_responses=1, input_tokens=100, output_tokens=10),
    )
    report = EvaluationReport(name="unknown-cost", cases=[report_case("verified", True)])
    context = ReportEvaluatorContext(
        name=report.name,
        report=report,
        experiment_metadata=None,
    )

    analysis = ModelUsageAnalysis(runs_dir=runs_dir).evaluate(context)

    assert isinstance(analysis, TableResult)
    values = {str(row[0]): row[1] for row in analysis.rows}
    assert values["Cost basis"] == "unavailable"
    assert values["Benchmark model cost"] is None
    assert values["Provider cost coverage"] == "0/1 responses"
    assert values["Pydantic cost coverage"] == "0/1 responses"
    assert values["Mean cost per selected attempt"] is None
    assert values["Effective cost per Lean-verified attempt"] is None
