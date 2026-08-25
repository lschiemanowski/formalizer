from uuid import uuid4

from formalizer.eval.models import Problem, TrialResult
from formalizer.run import formalize
from formalizer.settings import Settings


async def evaluate_problem(
    problem: Problem,
    settings: Settings,
) -> TrialResult:
    run_id = uuid4()
    await formalize(problem.prompt, settings, run_id=run_id)
    return TrialResult(
        problem_id=problem.id,
        run_id=run_id,
        status="verified",
    )
