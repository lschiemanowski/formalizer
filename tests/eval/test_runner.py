import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from uuid import UUID

import pytest
from pydantic_ai import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_evals import Case, Dataset

import formalizer.eval.runner as runner_module
from formalizer.agent import Submission
from formalizer.eval import (
    EvalOutput,
    LeanVerified,
    ProblemMetadata,
    ProblemProvenance,
    evaluate_dataset,
)
from formalizer.lean import LeanInfrastructureError, LeanResult
from formalizer.problem import FormalizationProblem
from formalizer.search import SearchInfrastructureError
from formalizer.settings import Settings

ACCEPTED_PROBLEM = FormalizationProblem(
    source="""\
namespace FormalizerProblem
def Target : Prop := True
end FormalizerProblem
"""
)
REJECTED_PROBLEM = FormalizationProblem(
    source="""\
namespace FormalizerProblem
def Target : Prop := False
end FormalizerProblem
"""
)
FAILING_PROBLEM = FormalizationProblem(
    source="""\
namespace FormalizerProblem
def Target : Prop := 1 + 1 = 2
end FormalizerProblem
"""
)


def problem_metadata(difficulty: Literal["easy", "medium", "hard"]) -> ProblemMetadata:
    return ProblemMetadata(
        difficulty=difficulty,
        split="test",
        domain="test fixture",
        provenance=ProblemProvenance(
            origin="original",
            source="Test fixture",
            source_id=f"{difficulty}-case",
            license="Apache-2.0",
        ),
    )


async def test_dataset_reports_verified_rejected_and_failed_formalizer_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(model_name="test:model")
    accepted_check = LeanResult(stdout="", stderr="", exit_code=0, duration_s=0.1)
    rejected_check = LeanResult(
        stdout="",
        stderr="Main.lean:4:2: error: type mismatch",
        exit_code=1,
        duration_s=0.1,
    )
    accepted_run_id = UUID("1c4d7195-8bd4-4680-9c84-ce5d1cc6151d")
    rejected_run_id = UUID("ed9af771-6716-4a77-a36d-7886521b9a3c")
    received_problems: list[FormalizationProblem] = []

    async def fake_formalizer(
        problem: FormalizationProblem,
        received_settings: Settings,
    ) -> object:
        assert received_settings is settings
        received_problems.append(problem)

        if problem == ACCEPTED_PROBLEM:
            return SimpleNamespace(
                run_id=str(accepted_run_id),
                output=Submission(code="accepted", check=accepted_check),
            )
        if problem == REJECTED_PROBLEM:
            return SimpleNamespace(
                run_id=str(rejected_run_id),
                output=Submission(code="rejected", check=rejected_check),
            )
        raise RuntimeError("provider request failed")

    monkeypatch.setattr(runner_module, "formalize", fake_formalizer)
    dataset = Dataset[FormalizationProblem, EvalOutput, ProblemMetadata](
        name="formalizer-smoke",
        cases=[
            Case(
                name="accepted",
                inputs=ACCEPTED_PROBLEM,
                metadata=problem_metadata("easy"),
            ),
            Case(
                name="rejected",
                inputs=REJECTED_PROBLEM,
                metadata=problem_metadata("medium"),
            ),
            Case(
                name="failed",
                inputs=FAILING_PROBLEM,
                metadata=problem_metadata("hard"),
            ),
        ],
        evaluators=[LeanVerified()],
    )

    report = await evaluate_dataset(
        dataset,
        settings,
        progress=False,
    )

    cases = {case.name: case for case in report.cases}
    assert received_problems == [ACCEPTED_PROBLEM, REJECTED_PROBLEM, FAILING_PROBLEM]
    assert set(cases) == {"accepted", "rejected"}
    assert cases["accepted"].output == EvalOutput(
        run_id=str(accepted_run_id),
        submission=Submission(code="accepted", check=accepted_check),
    )
    assert cases["accepted"].assertions["LeanVerified"].value is True
    assert cases["rejected"].output == EvalOutput(
        run_id=str(rejected_run_id),
        submission=Submission(code="rejected", check=rejected_check),
    )
    assert cases["rejected"].assertions["LeanVerified"].value is False
    assert cases["rejected"].metadata == problem_metadata("medium")
    assert len(report.failures) == 1
    assert report.failures[0].name == "failed"
    assert report.failures[0].error_message == "RuntimeError: provider request failed"


def test_dataset_yaml_loads_typed_lean_problem_and_metadata(tmp_path: Path) -> None:
    dataset_path = tmp_path / "baseline.yaml"
    dataset_path.write_text(
        """\
name: baseline-v1
cases:
  - name: arithmetic/one-plus-one
    inputs:
      source: |
        namespace FormalizerProblem
        def Target : Prop := 1 + 1 = 2
        end FormalizerProblem
    metadata:
      difficulty: easy
      split: test
      domain: arithmetic
      provenance:
        origin: original
        source: Test fixture
        source_id: arithmetic/one-plus-one
        license: Apache-2.0
""",
        encoding="utf-8",
    )

    dataset = Dataset[FormalizationProblem, EvalOutput, ProblemMetadata].from_file(dataset_path)

    assert dataset.name == "baseline-v1"
    assert len(dataset.cases) == 1
    assert dataset.cases[0].name == "arithmetic/one-plus-one"
    assert dataset.cases[0].inputs.source.endswith("end FormalizerProblem\n")
    assert dataset.cases[0].metadata == ProblemMetadata(
        difficulty="easy",
        split="test",
        domain="arithmetic",
        provenance=ProblemProvenance(
            origin="original",
            source="Test fixture",
            source_id="arithmetic/one-plus-one",
            license="Apache-2.0",
        ),
    )


async def test_dataset_evaluation_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(model_name="test:model")

    async def cancel_formalizer(
        problem: FormalizationProblem,
        received_settings: Settings,
    ) -> object:
        raise asyncio.CancelledError

    monkeypatch.setattr(runner_module, "formalize", cancel_formalizer)
    dataset = Dataset[FormalizationProblem, EvalOutput, ProblemMetadata](
        name="cancelled",
        cases=[Case(name="cancelled", inputs=ACCEPTED_PROBLEM)],
        evaluators=[LeanVerified()],
    )

    with pytest.raises(asyncio.CancelledError):
        await evaluate_dataset(
            dataset,
            settings,
            progress=False,
        )


async def test_dataset_evaluation_respects_max_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(model_name="test:model")
    active = 0
    peak_active = 0
    two_started = asyncio.Event()
    accepted_check = LeanResult(stdout="", stderr="", exit_code=0, duration_s=0.1)

    async def overlapping_formalizer(
        problem: FormalizationProblem,
        received_settings: Settings,
    ) -> object:
        nonlocal active, peak_active
        assert received_settings is settings
        active += 1
        peak_active = max(peak_active, active)
        if active == 2:
            two_started.set()
        try:
            await asyncio.wait_for(two_started.wait(), timeout=0.5)
            return SimpleNamespace(
                run_id=f"run-{id(problem)}",
                output=Submission(code="accepted", check=accepted_check),
            )
        finally:
            active -= 1

    monkeypatch.setattr(runner_module, "formalize", overlapping_formalizer)
    dataset = Dataset[FormalizationProblem, EvalOutput, ProblemMetadata](
        name="concurrent",
        cases=[
            Case(name="first", inputs=ACCEPTED_PROBLEM),
            Case(name="second", inputs=REJECTED_PROBLEM),
            Case(name="third", inputs=FAILING_PROBLEM),
        ],
        evaluators=[LeanVerified()],
    )

    report = await evaluate_dataset(
        dataset,
        settings,
        progress=False,
        max_concurrency=2,
    )

    assert peak_active == 2
    assert not report.failures
    assert {case.name for case in report.cases} == {"first", "second", "third"}


async def test_dataset_evaluation_rejects_nonpositive_max_concurrency() -> None:
    dataset = Dataset[FormalizationProblem, EvalOutput, ProblemMetadata](
        name="invalid-concurrency",
        cases=[Case(name="accepted", inputs=ACCEPTED_PROBLEM)],
    )

    with pytest.raises(ValueError, match="max_concurrency"):
        await evaluate_dataset(
            dataset,
            Settings(model_name="test:model"),
            progress=False,
            max_concurrency=0,
        )


async def test_concurrent_dataset_evaluation_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def cancel_formalizer(
        problem: FormalizationProblem,
        received_settings: Settings,
    ) -> object:
        nonlocal calls
        calls += 1
        raise asyncio.CancelledError

    monkeypatch.setattr(runner_module, "formalize", cancel_formalizer)
    dataset = Dataset[FormalizationProblem, EvalOutput, ProblemMetadata](
        name="concurrently-cancelled",
        cases=[
            Case(name="first", inputs=ACCEPTED_PROBLEM),
            Case(name="second", inputs=REJECTED_PROBLEM),
            Case(name="queued", inputs=FAILING_PROBLEM),
        ],
    )

    with pytest.raises(asyncio.CancelledError):
        await evaluate_dataset(
            dataset,
            Settings(model_name="test:model"),
            progress=False,
            max_concurrency=2,
        )

    assert calls <= 2


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ModelAPIError("test:model", "connection timed out"), True),
        (ModelHTTPError(408, "test:model"), True),
        (ModelHTTPError(429, "test:model"), True),
        (ModelHTTPError(500, "test:model"), True),
        (ModelHTTPError(400, "test:model"), False),
        (ModelHTTPError(401, "test:model"), False),
        (LeanInfrastructureError("Docker unavailable"), True),
        (SearchInfrastructureError("Loogle unavailable"), True),
        (UnexpectedModelBehavior("invalid model response"), False),
        (UsageLimitExceeded("request limit reached"), False),
        (TimeoutError("run timed out"), False),
    ],
)
def test_retryable_infrastructure_failure_classification(
    error: BaseException,
    expected: bool,
) -> None:
    assert runner_module.is_retryable_infrastructure_failure(error) is expected


async def test_dataset_retries_an_infrastructure_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(model_name="test:model")
    attempts = 0
    accepted_check = LeanResult(stdout="", stderr="", exit_code=0, duration_s=0.1)

    async def flaky_formalizer(
        problem: FormalizationProblem,
        received_settings: Settings,
    ) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ModelAPIError("test:model", "connection timed out")
        return SimpleNamespace(
            run_id="2dc4f495-1513-484d-9082-4f7a229edcd7",
            output=Submission(code="accepted", check=accepted_check),
        )

    monkeypatch.setattr(runner_module, "formalize", flaky_formalizer)
    dataset = Dataset[FormalizationProblem, EvalOutput, ProblemMetadata](
        name="infrastructure-retry",
        cases=[Case(name="accepted", inputs=ACCEPTED_PROBLEM)],
        evaluators=[LeanVerified()],
    )

    report = await evaluate_dataset(
        dataset,
        settings,
        progress=False,
        infrastructure_retries=1,
    )

    assert attempts == 2
    assert not report.failures
    assert report.cases[0].assertions["LeanVerified"].value is True


async def test_dataset_does_not_retry_a_model_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(model_name="test:model")
    attempts = 0

    async def budget_exhausted(
        problem: FormalizationProblem,
        received_settings: Settings,
    ) -> object:
        nonlocal attempts
        attempts += 1
        raise UsageLimitExceeded("request limit reached")

    monkeypatch.setattr(runner_module, "formalize", budget_exhausted)
    dataset = Dataset[FormalizationProblem, EvalOutput, ProblemMetadata](
        name="model-failure",
        cases=[Case(name="failed", inputs=FAILING_PROBLEM)],
    )

    report = await evaluate_dataset(
        dataset,
        settings,
        progress=False,
        infrastructure_retries=2,
    )

    assert attempts == 1
    assert len(report.failures) == 1
    assert report.failures[0].error_message.startswith("UsageLimitExceeded:")


async def test_dataset_stops_after_the_configured_infrastructure_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(model_name="test:model")
    attempts = 0

    async def unavailable_provider(
        problem: FormalizationProblem,
        received_settings: Settings,
    ) -> object:
        nonlocal attempts
        attempts += 1
        raise ModelAPIError("test:model", "connection timed out")

    monkeypatch.setattr(runner_module, "formalize", unavailable_provider)
    dataset = Dataset[FormalizationProblem, EvalOutput, ProblemMetadata](
        name="infrastructure-retry-limit",
        cases=[Case(name="failed", inputs=FAILING_PROBLEM)],
    )

    report = await evaluate_dataset(
        dataset,
        settings,
        progress=False,
        infrastructure_retries=2,
    )

    assert attempts == 3
    assert len(report.failures) == 1
    assert report.failures[0].error_message == "ModelAPIError: connection timed out"
