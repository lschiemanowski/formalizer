from uuid import UUID

import pytest
from formalizer.problem import FormalizationProblem

import formalizer.eval.runner as runner_module
from formalizer.eval import Problem, TrialResult, evaluate_problem
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

    async def fake_formalizer(
        received_task: FormalizationProblem,
        received_settings: Settings,
        *,
        run_id: UUID,
    ) -> object:
        received_calls.append((received_task, received_settings, run_id))
        return object()

    monkeypatch.setattr(runner_module, "formalize", fake_formalizer)

    result = await evaluate_problem(problem, settings)

    assert received_calls == [(problem.task, settings, result.run_id)]
    assert result == TrialResult(
        problem_id=problem.id,
        run_id=result.run_id,
        status="verified",
    )
