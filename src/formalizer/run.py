import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, AgentRunResult, capture_run_messages
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelResponse

from formalizer.agent import AgentDependencies, Submission, create_agent, problem_prompt
from formalizer.lean import DockerLeanChecker, LeanResult, LeanWorkspace
from formalizer.problem import FormalizationProblem
from formalizer.search import DockerLoogleBackend
from formalizer.settings import RunSettings, Settings


class ModelUsage(BaseModel):
    """Stable model usage recorded for one agent run."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    model_responses: int = 0
    input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    pydantic_estimated_cost_usd: Decimal | None = None
    pydantic_costed_responses: int = 0
    provider_reported_cost_usd: Decimal | None = None
    provider_costed_responses: int = 0


class RunManifest(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    run_id: UUID
    problem: FormalizationProblem
    status: Literal["verified", "failed"]
    started_at: datetime
    finished_at: datetime
    error_type: str | None = None
    error: str | None = None
    verification: LeanResult | None = None
    usage: ModelUsage = Field(default_factory=ModelUsage)


def _cost_decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _reasoning_tokens(response: ModelResponse) -> int:
    output_reasoning_tokens = getattr(response.usage, "output_reasoning_tokens", None)
    if isinstance(output_reasoning_tokens, int):
        return output_reasoning_tokens
    return response.usage.details.get("reasoning_tokens", 0)


def summarize_model_messages(messages: Sequence[ModelMessage]) -> ModelUsage:
    responses = [message for message in messages if isinstance(message, ModelResponse)]
    pydantic_costs = [
        cost for response in responses if (cost := _cost_decimal(response.usage.cost)) is not None
    ]
    provider_costs = [
        cost
        for response in responses
        if (cost := _cost_decimal((response.provider_details or {}).get("cost"))) is not None
    ]
    return ModelUsage(
        model_responses=len(responses),
        input_tokens=sum(response.usage.input_tokens for response in responses),
        cache_write_tokens=sum(response.usage.cache_write_tokens for response in responses),
        cache_read_tokens=sum(response.usage.cache_read_tokens for response in responses),
        output_tokens=sum(response.usage.output_tokens for response in responses),
        reasoning_tokens=sum(_reasoning_tokens(response) for response in responses),
        pydantic_estimated_cost_usd=sum(pydantic_costs, start=Decimal(0))
        if pydantic_costs
        else None,
        pydantic_costed_responses=len(pydantic_costs),
        provider_reported_cost_usd=sum(provider_costs, start=Decimal(0))
        if provider_costs
        else None,
        provider_costed_responses=len(provider_costs),
    )


def _write_manifest(run_dir: Path, manifest: RunManifest) -> None:
    (run_dir / "run.json").write_text(
        f"{manifest.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )


def _write_workspace(run_dir: Path, workspace: LeanWorkspace) -> None:
    workspace_dir = run_dir / "workspace"
    for archive_path, source in workspace.sources.items():
        destination = workspace_dir / archive_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source, encoding="utf-8")


async def run_formalizer(
    problem: FormalizationProblem,
    *,
    agent: Agent[AgentDependencies, Submission],
    deps: AgentDependencies,
    settings: RunSettings,
    run_id: UUID | None = None,
) -> AgentRunResult[Submission]:
    resolved_run_id = run_id or uuid4()
    run_id_text = str(resolved_run_id)
    run_dir = settings.runs_dir / run_id_text
    started_at = datetime.now(UTC)
    run_dir.mkdir(parents=True)
    (run_dir / "problem.lean").write_text(problem.source, encoding="utf-8")

    with capture_run_messages() as messages:
        try:
            async with asyncio.timeout(settings.run_timeout_s):
                result = await agent.run(
                    problem_prompt(problem),
                    deps=deps,
                    run_id=run_id_text,
                    usage_limits=settings.usage_limits,
                )
        except (asyncio.CancelledError, Exception) as error:
            failed_manifest = RunManifest(
                run_id=resolved_run_id,
                problem=problem,
                status="failed",
                started_at=started_at,
                finished_at=datetime.now(UTC),
                error_type=type(error).__name__,
                error=str(error),
                usage=summarize_model_messages(messages),
            )
            (run_dir / "messages.json").write_bytes(ModelMessagesTypeAdapter.dump_json(messages))
            _write_workspace(run_dir, deps.lean_workspace)
            _write_manifest(run_dir, failed_manifest)
            raise

    finished_at = datetime.now(UTC)
    status: Literal["verified", "failed"] = "verified" if result.output.check.accepted else "failed"
    manifest = RunManifest(
        run_id=resolved_run_id,
        problem=problem,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        verification=result.output.check,
        usage=summarize_model_messages(result.all_messages()),
    )

    (run_dir / "messages.json").write_bytes(result.all_messages_json())
    _write_workspace(run_dir, deps.lean_workspace)
    _write_manifest(run_dir, manifest)
    (run_dir / "final.lean").write_text(result.output.code, encoding="utf-8")

    return result


async def formalize(
    problem: FormalizationProblem,
    settings: Settings,
    *,
    run_id: UUID | None = None,
) -> AgentRunResult[Submission]:
    lean_checker = DockerLeanChecker(
        settings.sandbox,
        problem_code=problem.source,
    )
    await lean_checker.validate_problem()

    agent = create_agent(
        settings.model_name,
        model_settings=settings.model_settings,
    )

    async with DockerLoogleBackend(settings.sandbox) as search_backend:
        return await run_formalizer(
            problem,
            agent=agent,
            deps=AgentDependencies(
                lean_checker=lean_checker,
                search_backend=search_backend,
            ),
            settings=settings.run,
            run_id=run_id,
        )
