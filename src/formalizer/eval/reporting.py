import sys
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TextIO

from pydantic_evals.evaluators import ReportEvaluator, ReportEvaluatorContext
from pydantic_evals.reporting import TableResult

from formalizer.eval.artifacts import FormalizerEvaluationReport
from formalizer.eval.models import EvalOutput, ProblemMetadata
from formalizer.problem import FormalizationProblem
from formalizer.run import RunManifest

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


@dataclass(frozen=True, slots=True)
class ModelUsageSummary:
    agent_runs: int
    model_responses: int
    input_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    output_tokens: int
    reasoning_tokens: int
    pydantic_estimated_cost_usd: Decimal | None
    pydantic_costed_responses: int
    provider_reported_cost_usd: Decimal | None
    provider_costed_responses: int

    @property
    def pydantic_cost_is_complete(self) -> bool:
        return (
            self.pydantic_estimated_cost_usd is not None
            and self.pydantic_costed_responses == self.model_responses
        )

    @property
    def provider_cost_is_complete(self) -> bool:
        return (
            self.provider_reported_cost_usd is not None
            and self.provider_costed_responses == self.model_responses
        )

    @property
    def benchmark_cost(self) -> tuple[str, Decimal] | None:
        if self.provider_cost_is_complete:
            assert self.provider_reported_cost_usd is not None
            return "provider-reported", self.provider_reported_cost_usd
        if self.pydantic_cost_is_complete:
            assert self.pydantic_estimated_cost_usd is not None
            return "pydantic-estimated", self.pydantic_estimated_cost_usd
        return None


def summarize_model_usage(runs_dir: Path) -> ModelUsageSummary:
    manifests = [
        RunManifest.model_validate_json(path.read_bytes())
        for path in sorted(runs_dir.glob("*/run.json"))
    ]
    pydantic_costs = [
        manifest.usage.pydantic_estimated_cost_usd
        for manifest in manifests
        if manifest.usage.pydantic_estimated_cost_usd is not None
    ]
    provider_costs = [
        manifest.usage.provider_reported_cost_usd
        for manifest in manifests
        if manifest.usage.provider_reported_cost_usd is not None
    ]
    return ModelUsageSummary(
        agent_runs=len(manifests),
        model_responses=sum(manifest.usage.model_responses for manifest in manifests),
        input_tokens=sum(manifest.usage.input_tokens for manifest in manifests),
        cache_write_tokens=sum(manifest.usage.cache_write_tokens for manifest in manifests),
        cache_read_tokens=sum(manifest.usage.cache_read_tokens for manifest in manifests),
        output_tokens=sum(manifest.usage.output_tokens for manifest in manifests),
        reasoning_tokens=sum(manifest.usage.reasoning_tokens for manifest in manifests),
        pydantic_estimated_cost_usd=sum(pydantic_costs, start=Decimal(0))
        if pydantic_costs
        else None,
        pydantic_costed_responses=sum(
            manifest.usage.pydantic_costed_responses for manifest in manifests
        ),
        provider_reported_cost_usd=sum(provider_costs, start=Decimal(0))
        if provider_costs
        else None,
        provider_costed_responses=sum(
            manifest.usage.provider_costed_responses for manifest in manifests
        ),
    )


@dataclass(repr=False)
class ModelUsageAnalysis(ReportEvaluator[FormalizationProblem, EvalOutput, ProblemMetadata]):
    runs_dir: Path

    def evaluate(
        self,
        ctx: ReportEvaluatorContext[
            FormalizationProblem,
            EvalOutput,
            ProblemMetadata,
        ],
    ) -> TableResult:
        usage = summarize_model_usage(self.runs_dir)
        outcomes = summarize_outcomes(ctx.report)
        benchmark_cost = usage.benchmark_cost
        cost_basis = benchmark_cost[0] if benchmark_cost is not None else "unavailable"
        total_cost = benchmark_cost[1] if benchmark_cost is not None else None
        mean_cost = (
            total_cost / outcomes.selected_attempts
            if total_cost is not None and outcomes.selected_attempts > 0
            else None
        )
        effective_cost = (
            total_cost / outcomes.lean_verified
            if total_cost is not None and outcomes.lean_verified > 0
            else None
        )

        return TableResult(
            title="Model usage and cost",
            description=(
                "Totals include every persisted agent run, including failed infrastructure "
                "retries. Provider-reported cost is preferred when it covers every model "
                "response; otherwise a complete Pydantic estimate is used."
            ),
            columns=["Metric", "Value", "Unit"],
            rows=[
                ["Agent runs, including retries", usage.agent_runs, "runs"],
                ["Model responses", usage.model_responses, "responses"],
                ["Input tokens", usage.input_tokens, "tokens"],
                ["Cache-write tokens", usage.cache_write_tokens, "tokens"],
                ["Cache-read tokens", usage.cache_read_tokens, "tokens"],
                ["Output tokens", usage.output_tokens, "tokens"],
                ["Reasoning tokens", usage.reasoning_tokens, "tokens"],
                [
                    "Provider-reported cost (recorded)",
                    _float_or_none(usage.provider_reported_cost_usd),
                    "USD",
                ],
                [
                    "Provider cost coverage",
                    f"{usage.provider_costed_responses}/{usage.model_responses} responses",
                    None,
                ],
                [
                    "Pydantic estimated cost (recorded)",
                    _float_or_none(usage.pydantic_estimated_cost_usd),
                    "USD",
                ],
                [
                    "Pydantic cost coverage",
                    f"{usage.pydantic_costed_responses}/{usage.model_responses} responses",
                    None,
                ],
                ["Cost basis", cost_basis, None],
                ["Benchmark model cost", _float_or_none(total_cost), "USD"],
                ["Mean cost per selected attempt", _float_or_none(mean_cost), "USD"],
                [
                    "Effective cost per Lean-verified attempt",
                    _float_or_none(effective_cost),
                    "USD",
                ],
            ],
        )


def _float_or_none(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None


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
