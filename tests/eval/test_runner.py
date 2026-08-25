import asyncio
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic_evals import Case, Dataset

import formalizer.eval.runner as runner_module
from formalizer.agent import Submission
from formalizer.eval import EvalOutput, LeanVerified, ProblemMetadata, evaluate_dataset
from formalizer.lean import LeanResult
from formalizer.problem import FormalizationProblem
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
                metadata=ProblemMetadata(difficulty="easy", split="test"),
            ),
            Case(
                name="rejected",
                inputs=REJECTED_PROBLEM,
                metadata=ProblemMetadata(difficulty="medium", split="test"),
            ),
            Case(
                name="failed",
                inputs=FAILING_PROBLEM,
                metadata=ProblemMetadata(difficulty="hard", split="test"),
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
    assert cases["rejected"].metadata == ProblemMetadata(
        difficulty="medium",
        split="test",
    )
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
