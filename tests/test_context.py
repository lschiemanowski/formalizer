import json
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pytest
from pydantic_ai import UsageLimits
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from formalizer.agent import AgentDependencies, create_agent
from formalizer.context import ContextPolicyInput, ContextProjection
from formalizer.lean import LeanResult
from formalizer.problem import FormalizationProblem
from formalizer.run import run_formalizer
from formalizer.search import SearchResult
from formalizer.settings import RunSettings

PROBLEM = FormalizationProblem(
    source="""import Mathlib.Data.Nat.Basic

namespace FormalizerProblem

def Target : Prop := 1 + 1 = 2

end FormalizerProblem
"""
)
VALID_CODE = """import FormalizerProblem
import Mathlib.Tactic

namespace FormalizerSubmission

theorem solution : FormalizerProblem.Target := by
  norm_num [FormalizerProblem.Target]

end FormalizerSubmission
"""


class AcceptingLeanChecker:
    async def check(
        self,
        code: str,
        *,
        auxiliary_sources: Mapping[str, str] | None = None,
    ) -> LeanResult:
        return LeanResult(stdout="", stderr="", exit_code=0, duration_s=0.1)


class UnexpectedSearchBackend:
    async def search(self, query: str, *, max_results: int = 10) -> SearchResult:
        raise AssertionError(f"Search was not expected: {query}, max_results={max_results}")


class DropInitialRequestPolicy:
    name: ClassVar[str] = "drop-initial-request-v1"

    async def project(self, context: ContextPolicyInput) -> ContextProjection:
        if context.request_index == 1:
            return ContextProjection(
                messages=context.messages[1:],
                action="drop_initial_request",
                metadata={"removed_messages": 1},
            )
        return ContextProjection(messages=context.messages, action="passthrough")


class FailingPolicy:
    name: ClassVar[str] = "failing-policy-v1"

    async def project(self, context: ContextPolicyInput) -> ContextProjection:
        raise RuntimeError("context projection failed")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _user_prompt_text(messages: list[ModelMessage]) -> str:
    return "\n".join(
        part.content
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    )


def _all_serialized_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_all_serialized_text(item) for item in value)
    if isinstance(value, dict):
        return "\n".join(_all_serialized_text(item) for item in value.values())
    return ""


async def test_default_run_does_not_create_context_artifacts(tmp_path: Path) -> None:
    run_id = UUID("91061519-e6f5-48b0-acf6-0115bbf8236e")

    async def submit(
        messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        assert any(
            isinstance(part, UserPromptPart) for message in messages for part in message.parts
        )
        return ModelResponse(
            parts=[ToolCallPart(tool_name="final_submission", args={"code": VALID_CODE})]
        )

    await run_formalizer(
        PROBLEM,
        agent=create_agent(FunctionModel(submit)),
        deps=AgentDependencies(
            lean_checker=AcceptingLeanChecker(),
            search_backend=UnexpectedSearchBackend(),
        ),
        settings=RunSettings(runs_dir=tmp_path),
        run_id=run_id,
    )

    assert not (tmp_path / str(run_id) / "context-events.jsonl").exists()


async def test_context_policy_projects_history_and_preserves_audit_input(
    tmp_path: Path,
) -> None:
    run_id = UUID("83af9fb6-f1c6-42d7-81e3-61e2f9b666ac")
    model_calls = 0

    async def check_then_submit(
        messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            assert PROBLEM.source in _user_prompt_text(messages)
            return ModelResponse(
                parts=[ToolCallPart(tool_name="lean_execute", args={"code": VALID_CODE})]
            )

        assert PROBLEM.source not in _user_prompt_text(messages)
        assert any(
            isinstance(part, ToolReturnPart) and part.tool_name == "lean_execute"
            for message in messages
            for part in message.parts
        )
        return ModelResponse(
            parts=[ToolCallPart(tool_name="final_submission", args={"code": VALID_CODE})]
        )

    await run_formalizer(
        PROBLEM,
        agent=create_agent(FunctionModel(check_then_submit)),
        deps=AgentDependencies(
            lean_checker=AcceptingLeanChecker(),
            search_backend=UnexpectedSearchBackend(),
        ),
        settings=RunSettings(runs_dir=tmp_path),
        run_id=run_id,
        context_policy=DropInitialRequestPolicy(),
    )

    records = _read_jsonl(tmp_path / str(run_id) / "context-events.jsonl")
    manifest = json.loads((tmp_path / str(run_id) / "run.json").read_text())
    assert [(record["event"], record["request_index"]) for record in records] == [
        ("context_projection_started", 0),
        ("context_projection_completed", 0),
        ("context_projection_started", 1),
        ("context_projection_completed", 1),
    ]
    assert records[0]["before_message_count"] == 1
    assert records[1]["after_message_count"] == 1
    assert records[2]["before_message_count"] == 3
    assert records[3]["after_message_count"] == 2
    assert records[3]["action"] == "drop_initial_request"
    assert records[3]["metadata"] == {"removed_messages": 1}
    assert PROBLEM.source in _all_serialized_text(records[2]["before_messages"])
    assert PROBLEM.source not in _all_serialized_text(records[3]["after_messages"])
    assert records[2]["before_sha256"] != records[3]["after_sha256"]
    assert manifest["context_policy"] == DropInitialRequestPolicy.name


async def test_context_policy_failure_is_appended_and_run_failure_is_persisted(
    tmp_path: Path,
) -> None:
    run_id = UUID("e4807c32-4754-4ae7-b944-fc28df6a0507")

    async def unexpected_model_call(
        messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        raise AssertionError("The model should not be called after context projection fails")

    with pytest.raises(RuntimeError, match="context projection failed"):
        await run_formalizer(
            PROBLEM,
            agent=create_agent(FunctionModel(unexpected_model_call)),
            deps=AgentDependencies(
                lean_checker=AcceptingLeanChecker(),
                search_backend=UnexpectedSearchBackend(),
            ),
            settings=RunSettings(
                runs_dir=tmp_path,
                usage_limits=UsageLimits(request_limit=1),
            ),
            run_id=run_id,
            context_policy=FailingPolicy(),
        )

    run_dir = tmp_path / str(run_id)
    records = _read_jsonl(run_dir / "context-events.jsonl")
    assert [record["event"] for record in records] == [
        "context_projection_started",
        "context_projection_failed",
    ]
    assert records[1]["error_type"] == "RuntimeError"
    assert records[1]["error"] == "context projection failed"
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["error_type"] == "RuntimeError"
    assert manifest["context_policy"] == FailingPolicy.name
    assert (run_dir / "messages.json").is_file()
