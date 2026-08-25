import asyncio
from types import SimpleNamespace
from uuid import UUID

import pytest

import formalizer.eval.runner as runner_module
from formalizer.eval import Problem, TrialResult, evaluate_problem
from formalizer.lean import LeanResult
from formalizer.problem import FormalizationProblem
from formalizer.settings import Settings


async def test_evaluate_problem_links_problem_to_formalizer_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = FormalizationProblem(
        source="""\
namespace FormalizerProblem
def Target : Prop := 1 + 1 = 2
end FormalizerProblem
"""
    )
    problem = Problem(
        id="arithmetic/one-plus-one",
        task=task,
    )
    settings = Settings(model_name="test:model")
    received_calls: list[tuple[FormalizationProblem, Settings, UUID]] = []
    accepted_check = LeanResult(stdout="", stderr="", exit_code=0, duration_s=0.1)

    async def fake_formalizer(
        received_task: FormalizationProblem,
        received_settings: Settings,
        *,
        run_id: UUID,
    ) -> object:
        received_calls.append((received_task, received_settings, run_id))
        return SimpleNamespace(output=SimpleNamespace(check=accepted_check))

    monkeypatch.setattr(runner_module, "formalize", fake_formalizer)

    result = await evaluate_problem(problem, settings)

    assert received_calls == [(problem.task, settings, result.run_id)]
    assert result == TrialResult(
        problem_id=problem.id,
        run_id=result.run_id,
        status="verified",
    )


async def test_evaluate_problem_records_rejected_submission_as_failed_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = Problem(
        id="arithmetic/rejected",
        task=FormalizationProblem(
            source="""\
namespace FormalizerProblem
def Target : Prop := False
end FormalizerProblem
"""
        ),
    )
    settings = Settings(model_name="test:model")
    rejected_check = LeanResult(
        stdout="",
        stderr="Main.lean:4:2: error: type mismatch",
        exit_code=1,
        duration_s=0.1,
    )

    async def return_rejected_submission(
        received_task: FormalizationProblem,
        received_settings: Settings,
        *,
        run_id: UUID,
    ) -> object:
        assert received_task == problem.task
        assert received_settings is settings
        return SimpleNamespace(output=SimpleNamespace(check=rejected_check))

    monkeypatch.setattr(runner_module, "formalize", return_rejected_submission)

    result = await evaluate_problem(problem, settings)

    assert result == TrialResult(
        problem_id=problem.id,
        run_id=result.run_id,
        status="failed",
    )


async def test_evaluate_problem_records_failed_formalizer_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = Problem(
        id="arithmetic/invalid-proof",
        task=FormalizationProblem(
            source="""\
namespace FormalizerProblem
def Target : Prop := False
end FormalizerProblem
"""
        ),
    )
    settings = Settings(model_name="test:model")
    expected_error = RuntimeError("model exhausted its retries")
    received_run_ids: list[UUID] = []

    async def fail_formalizer(
        received_task: FormalizationProblem,
        received_settings: Settings,
        *,
        run_id: UUID,
    ) -> object:
        assert received_task == problem.task
        assert received_settings is settings
        received_run_ids.append(run_id)
        raise expected_error

    monkeypatch.setattr(runner_module, "formalize", fail_formalizer)

    result = await evaluate_problem(problem, settings)

    assert received_run_ids == [result.run_id]
    assert result == TrialResult(
        problem_id=problem.id,
        run_id=result.run_id,
        status="failed",
        error_type="RuntimeError",
        error="model exhausted its retries",
    )


async def test_evaluate_problem_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = Problem(
        id="arithmetic/cancelled",
        task=FormalizationProblem(
            source="""\
namespace FormalizerProblem
def Target : Prop := True
end FormalizerProblem
"""
        ),
    )
    settings = Settings(model_name="test:model")

    async def cancel_formalizer(
        received_task: FormalizationProblem,
        received_settings: Settings,
        *,
        run_id: UUID,
    ) -> object:
        raise asyncio.CancelledError

    monkeypatch.setattr(runner_module, "formalize", cancel_formalizer)

    with pytest.raises(asyncio.CancelledError):
        await evaluate_problem(problem, settings)
