import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, AgentRunResult, capture_run_messages
from pydantic_ai.messages import ModelMessagesTypeAdapter

from formalizer.agent import AgentDependencies, Submission, create_agent, problem_prompt
from formalizer.lean import DockerLeanChecker, LeanResult
from formalizer.problem import FormalizationProblem
from formalizer.search import DockerLoogleBackend
from formalizer.settings import RunSettings, Settings


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


def _write_manifest(run_dir: Path, manifest: RunManifest) -> None:
    (run_dir / "run.json").write_text(
        f"{manifest.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )


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
            )
            (run_dir / "messages.json").write_bytes(ModelMessagesTypeAdapter.dump_json(messages))
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
    )

    (run_dir / "messages.json").write_bytes(result.all_messages_json())
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
