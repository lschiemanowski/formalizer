from uuid import uuid4

from formalizer.eval.models import Problem, TrialResult
from formalizer.run import formalize
from formalizer.settings import Settings


async def evaluate_problem(
    problem: Problem,
    settings: Settings,
) -> TrialResult:
    run_id = uuid4()

    try:
        run_result = await formalize(problem.task, settings, run_id=run_id)
    except Exception as error:  # noqa: BLE001 - An evaluator records individual trial failures.
        return TrialResult(
            problem_id=problem.id,
            run_id=run_id,
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
        )

    return TrialResult(
        problem_id=problem.id,
        run_id=run_id,
        status="verified" if run_result.output.check.accepted else "failed",
    )
