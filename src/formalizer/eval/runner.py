import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic_evals import Dataset
from pydantic_evals.reporting import EvaluationReport

from formalizer.eval.models import EvalOutput, ProblemMetadata
from formalizer.problem import FormalizationProblem
from formalizer.run import formalize
from formalizer.settings import Settings

FormalizerTask = Callable[[FormalizationProblem], Awaitable[EvalOutput]]
FormalizerDataset = Dataset[FormalizationProblem, EvalOutput, ProblemMetadata]


class _EvaluationCancelled(Exception):
    pass


def create_formalizer_task(settings: Settings) -> FormalizerTask:
    async def formalizer_task(problem: FormalizationProblem) -> EvalOutput:
        result = await formalize(problem, settings)
        return EvalOutput(
            run_id=result.run_id,
            submission=result.output,
        )

    return formalizer_task


async def evaluate_dataset(
    dataset: FormalizerDataset,
    settings: Settings,
    *,
    name: str | None = None,
    progress: bool = True,
    repeat: int = 1,
    metadata: dict[str, Any] | None = None,
) -> EvaluationReport[FormalizationProblem, EvalOutput, ProblemMetadata]:
    formalizer_task = create_formalizer_task(settings)
    cancelled = False

    async def tracked_task(problem: FormalizationProblem) -> EvalOutput:
        nonlocal cancelled
        if cancelled:
            raise _EvaluationCancelled
        try:
            return await formalizer_task(problem)
        except asyncio.CancelledError:
            cancelled = True
            raise _EvaluationCancelled from None

    report = await dataset.evaluate(
        tracked_task,
        name=name,
        max_concurrency=1,
        progress=progress,
        repeat=repeat,
        metadata=metadata,
        task_name="formalize",
    )
    if cancelled:
        raise asyncio.CancelledError
    return report
