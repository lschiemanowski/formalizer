import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic_ai import ModelAPIError, ModelHTTPError
from pydantic_ai.retries import RetryConfig
from pydantic_evals import Dataset
from pydantic_evals.reporting import EvaluationReport
from tenacity import retry_if_exception, stop_after_attempt, wait_random_exponential

from formalizer.eval.models import EvalOutput, ProblemMetadata
from formalizer.lean import LeanInfrastructureError
from formalizer.problem import FormalizationProblem
from formalizer.run import formalize
from formalizer.search import SearchInfrastructureError
from formalizer.settings import Settings

FormalizerTask = Callable[[FormalizationProblem], Awaitable[EvalOutput]]
FormalizerDataset = Dataset[FormalizationProblem, EvalOutput, ProblemMetadata]


class _EvaluationCancelled(Exception):
    pass


_RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 409, 425, 429})


def is_retryable_infrastructure_failure(error: BaseException) -> bool:
    if isinstance(error, ModelHTTPError):
        return error.status_code in _RETRYABLE_HTTP_STATUS_CODES or error.status_code >= 500
    return isinstance(
        error,
        (ModelAPIError, LeanInfrastructureError, SearchInfrastructureError),
    )


def _infrastructure_retry_config(retries: int) -> RetryConfig | None:
    if retries < 0:
        raise ValueError("infrastructure_retries must not be negative")
    if retries == 0:
        return None
    return RetryConfig(
        retry=retry_if_exception(is_retryable_infrastructure_failure),
        stop=stop_after_attempt(retries + 1),
        wait=wait_random_exponential(multiplier=1, max=10),
        reraise=True,
    )


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
    max_concurrency: int = 1,
    infrastructure_retries: int = 0,
    metadata: dict[str, Any] | None = None,
) -> EvaluationReport[FormalizationProblem, EvalOutput, ProblemMetadata]:
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")

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
        max_concurrency=max_concurrency,
        progress=progress,
        repeat=repeat,
        retry_task=_infrastructure_retry_config(infrastructure_retries),
        metadata=metadata,
        task_name="formalize",
    )
    if cancelled:
        raise asyncio.CancelledError
    return report
