from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, AgentRunResult

from formalizer.agent import AgentDependencies, VerifiedSubmission


class RunManifest(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    run_id: UUID
    problem: str
    status: Literal["verified"]
    started_at: datetime
    finished_at: datetime


async def run_formalizer(
    problem: str,
    *,
    agent: Agent[AgentDependencies, VerifiedSubmission],
    deps: AgentDependencies,
    runs_dir: Path,
    run_id: UUID | None = None,
) -> AgentRunResult[VerifiedSubmission]:
    resolved_run_id = run_id or uuid4()
    run_id_text = str(resolved_run_id)
    run_dir = runs_dir / run_id_text
    started_at = datetime.now(UTC)

    result = await agent.run(
        problem,
        deps=deps,
        run_id=run_id_text,
    )

    finished_at = datetime.now(UTC)
    manifest = RunManifest(
        run_id=resolved_run_id,
        problem=problem,
        status="verified",
        started_at=started_at,
        finished_at=finished_at,
    )

    run_dir.mkdir(parents=True)
    (run_dir / "messages.json").write_bytes(result.all_messages_json())
    (run_dir / "run.json").write_text(
        f"{manifest.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )
    (run_dir / "final.lean").write_text(result.output.code, encoding="utf-8")

    return result
