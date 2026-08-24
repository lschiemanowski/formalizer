import asyncio
import json
from pathlib import Path
from uuid import UUID

import pytest
from pydantic_ai import UnexpectedModelBehavior, UsageLimitExceeded, UsageLimits
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from formalizer.agent import AgentDependencies, VerifiedSubmission, create_agent
from formalizer.lean import LeanResult
from formalizer.run import RunManifest, run_formalizer
from formalizer.search import SearchResult
from formalizer.settings import RunSettings

PROBLEM = "Prove that 1 + 1 = 2."
VALID_CODE = """import Mathlib

example : 1 + 1 = 2 := by norm_num
"""
INVALID_CODE = """import Mathlib

example : 1 + 1 = 2 := by
  exact 0
"""


class AcceptingLeanChecker:
    def __init__(self) -> None:
        self.checked_code: list[str] = []

    async def check(self, code: str) -> LeanResult:
        self.checked_code.append(code)
        return LeanResult(stdout="", stderr="", exit_code=0, duration_s=0.1)


class RejectingLeanChecker:
    def __init__(self) -> None:
        self.checked_code: list[str] = []

    async def check(self, code: str) -> LeanResult:
        self.checked_code.append(code)
        return LeanResult(
            stdout="",
            stderr="Main.lean:4:2: error: type mismatch",
            exit_code=1,
            duration_s=0.1,
        )


class UnexpectedSearchBackend:
    async def search(self, query: str) -> SearchResult:
        raise AssertionError(f"Search was not expected: {query}")


async def test_successful_run_writes_run_artifacts(tmp_path: Path) -> None:
    run_id = UUID("0a8552d3-cf94-4640-b67c-9938387bdf7a")
    checker = AcceptingLeanChecker()

    async def submit_valid_code(
        messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        assert messages
        assert [tool.name for tool in agent_info.output_tools] == ["final_submission"]
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_submission",
                    args={"code": VALID_CODE},
                )
            ]
        )

    result = await run_formalizer(
        PROBLEM,
        agent=create_agent(FunctionModel(submit_valid_code)),
        deps=AgentDependencies(
            lean_checker=checker,
            search_backend=UnexpectedSearchBackend(),
        ),
        settings=RunSettings(runs_dir=tmp_path),
        run_id=run_id,
    )

    run_dir = tmp_path / str(run_id)
    manifest = RunManifest.model_validate_json((run_dir / "run.json").read_bytes())
    stored_messages = ModelMessagesTypeAdapter.validate_json(
        (run_dir / "messages.json").read_bytes()
    )

    assert result.run_id == str(run_id)
    assert result.output == VerifiedSubmission(code=VALID_CODE)
    assert checker.checked_code == [VALID_CODE]
    assert stored_messages == result.all_messages()
    assert manifest.run_id == run_id
    assert manifest.problem == PROBLEM
    assert manifest.status == "verified"
    assert manifest.started_at <= manifest.finished_at
    assert (run_dir / "final.lean").read_text() == VALID_CODE


async def test_failed_run_writes_diagnostics_without_final_lean(
    tmp_path: Path,
) -> None:
    run_id = UUID("e3b0a33a-4d88-48ce-a41b-35e018499bc8")
    checker = RejectingLeanChecker()

    async def submit_invalid_code(
        messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        assert messages
        assert [tool.name for tool in agent_info.output_tools] == ["final_submission"]
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_submission",
                    args={"code": INVALID_CODE},
                )
            ]
        )

    with pytest.raises(UnexpectedModelBehavior):
        await run_formalizer(
            PROBLEM,
            agent=create_agent(FunctionModel(submit_invalid_code)),
            deps=AgentDependencies(
                lean_checker=checker,
                search_backend=UnexpectedSearchBackend(),
            ),
            settings=RunSettings(runs_dir=tmp_path),
            run_id=run_id,
        )

    run_dir = tmp_path / str(run_id)
    manifest = json.loads((run_dir / "run.json").read_bytes())
    stored_messages = ModelMessagesTypeAdapter.validate_json(
        (run_dir / "messages.json").read_bytes()
    )
    submission_calls = [
        part
        for message in stored_messages
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_name == "final_submission"
    ]
    retry_prompts = [
        part
        for message in stored_messages
        for part in message.parts
        if isinstance(part, RetryPromptPart)
    ]

    assert checker.checked_code
    assert set(checker.checked_code) == {INVALID_CODE}
    assert submission_calls
    assert any(
        isinstance(prompt.content, str) and "type mismatch" in prompt.content
        for prompt in retry_prompts
    )
    assert manifest["run_id"] == str(run_id)
    assert manifest["problem"] == PROBLEM
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "UnexpectedModelBehavior"
    assert manifest["error"]
    assert manifest["started_at"] <= manifest["finished_at"]
    assert not (run_dir / "final.lean").exists()


async def test_cancelled_run_persists_its_error_type(tmp_path: Path) -> None:
    run_id = UUID("c54a8562-8a7a-4d58-8bb9-995ce536c92e")
    model_started = asyncio.Event()
    never_finishes = asyncio.Event()

    async def block_model_request(
        messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        assert messages
        assert [tool.name for tool in agent_info.output_tools] == ["final_submission"]
        model_started.set()
        await never_finishes.wait()
        raise AssertionError("The blocked model request unexpectedly resumed")

    task = asyncio.create_task(
        run_formalizer(
            PROBLEM,
            agent=create_agent(FunctionModel(block_model_request)),
            deps=AgentDependencies(
                lean_checker=AcceptingLeanChecker(),
                search_backend=UnexpectedSearchBackend(),
            ),
            settings=RunSettings(runs_dir=tmp_path),
            run_id=run_id,
        )
    )

    try:
        await asyncio.wait_for(model_started.wait(), timeout=5)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    run_dir = tmp_path / str(run_id)
    manifest = json.loads((run_dir / "run.json").read_bytes())
    stored_messages = ModelMessagesTypeAdapter.validate_json(
        (run_dir / "messages.json").read_bytes()
    )

    assert stored_messages
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "CancelledError"
    assert not (run_dir / "final.lean").exists()


async def test_run_timeout_is_enforced_and_persisted(tmp_path: Path) -> None:
    run_id = UUID("f9703931-7885-47f8-b6cd-6668ef507bc0")
    never_finishes = asyncio.Event()

    async def block_model_request(
        messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        assert messages
        assert [tool.name for tool in agent_info.output_tools] == ["final_submission"]
        await never_finishes.wait()
        raise AssertionError("The blocked model request unexpectedly resumed")

    with pytest.raises(TimeoutError):
        await run_formalizer(
            PROBLEM,
            agent=create_agent(FunctionModel(block_model_request)),
            deps=AgentDependencies(
                lean_checker=AcceptingLeanChecker(),
                search_backend=UnexpectedSearchBackend(),
            ),
            settings=RunSettings(
                runs_dir=tmp_path,
                run_timeout_s=0.01,
            ),
            run_id=run_id,
        )

    run_dir = tmp_path / str(run_id)
    manifest = json.loads((run_dir / "run.json").read_bytes())
    stored_messages = ModelMessagesTypeAdapter.validate_json(
        (run_dir / "messages.json").read_bytes()
    )

    assert stored_messages
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "TimeoutError"
    assert not (run_dir / "final.lean").exists()


async def test_model_request_limit_is_enforced_and_persisted(
    tmp_path: Path,
) -> None:
    run_id = UUID("4413ac2e-c0a7-4e12-a6a7-11d31bb4b3e1")
    checker = AcceptingLeanChecker()
    model_calls = 0

    async def check_candidate(
        messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        assert "lean_execute" in [tool.name for tool in agent_info.function_tools]
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="lean_execute",
                    args={"code": VALID_CODE},
                )
            ]
        )

    with pytest.raises(UsageLimitExceeded):
        await run_formalizer(
            PROBLEM,
            agent=create_agent(FunctionModel(check_candidate)),
            deps=AgentDependencies(
                lean_checker=checker,
                search_backend=UnexpectedSearchBackend(),
            ),
            settings=RunSettings(
                runs_dir=tmp_path,
                usage_limits=UsageLimits(request_limit=1),
            ),
            run_id=run_id,
        )

    run_dir = tmp_path / str(run_id)
    manifest = json.loads((run_dir / "run.json").read_bytes())
    stored_messages = ModelMessagesTypeAdapter.validate_json(
        (run_dir / "messages.json").read_bytes()
    )
    lean_results = [
        part.content
        for message in stored_messages
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "lean_execute"
    ]

    assert model_calls == 1
    assert checker.checked_code == [VALID_CODE]
    assert lean_results == [
        {
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "duration_s": 0.1,
            "timed_out": False,
        }
    ]
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "UsageLimitExceeded"
    assert not (run_dir / "final.lean").exists()
