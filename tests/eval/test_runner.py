from uuid import UUID

import pytest

import formalizer.eval.runner as runner_module
from formalizer.eval import Problem, TrialResult, evaluate_problem
from formalizer.settings import Settings


async def test_evaluate_problem_links_problem_to_formalizer_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = Problem(
        id="arithmetic/one-plus-one",
        prompt="Prove that 1 + 1 = 2.",
    )
    settings = Settings(model_name="test:model")
    received_calls: list[tuple[str, Settings, UUID]] = []

    async def fake_formalizer(
        prompt: str,
        received_settings: Settings,
        *,
        run_id: UUID,
    ) -> object:
        received_calls.append((prompt, received_settings, run_id))
        return object()

    monkeypatch.setattr(runner_module, "formalize", fake_formalizer)

    result = await evaluate_problem(problem, settings)

    assert received_calls == [(problem.prompt, settings, result.run_id)]
    assert result == TrialResult(
        problem_id=problem.id,
        run_id=result.run_id,
        status="verified",
    )
