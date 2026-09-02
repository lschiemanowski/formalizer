import asyncio
import json
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Self
from uuid import UUID

import pytest
from pydantic_ai import ModelSettings, UsageLimitExceeded, UsageLimits
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

import formalizer.run as run_module
from formalizer.agent import AgentDependencies, create_agent
from formalizer.lean import InvalidLeanProblem, LeanResult, LeanWorkspace
from formalizer.problem import FormalizationProblem
from formalizer.run import RunManifest, run_formalizer
from formalizer.search import SearchResult
from formalizer.settings import RunSettings, SandboxSettings, Settings

PROBLEM_SOURCE = """import Mathlib.Data.Nat.Basic

namespace FormalizerProblem

def Target : Prop := 1 + 1 = 2

end FormalizerProblem
"""
PROBLEM = FormalizationProblem(source=PROBLEM_SOURCE)
VALID_CODE = """import FormalizerProblem
import Mathlib.Tactic

namespace FormalizerSubmission

theorem solution : FormalizerProblem.Target := by
  norm_num [FormalizerProblem.Target]

end FormalizerSubmission
"""
INVALID_CODE = """import FormalizerProblem

namespace FormalizerSubmission

theorem solution : FormalizerProblem.Target := by
  exact 0

end FormalizerSubmission
"""


class AcceptingLeanChecker:
    def __init__(self) -> None:
        self.checked_code: list[str] = []
        self.checked_auxiliary_sources: list[dict[str, str]] = []
        self.result = LeanResult(stdout="", stderr="", exit_code=0, duration_s=0.1)

    async def check(
        self,
        code: str,
        *,
        auxiliary_sources: Mapping[str, str] | None = None,
    ) -> LeanResult:
        self.checked_code.append(code)
        self.checked_auxiliary_sources.append(dict(auxiliary_sources or {}))
        return self.result


class RejectingLeanChecker:
    def __init__(self) -> None:
        self.checked_code: list[str] = []
        self.checked_auxiliary_sources: list[dict[str, str]] = []
        self.result = LeanResult(
            stdout="",
            stderr="Main.lean:4:2: error: type mismatch",
            exit_code=1,
            duration_s=0.1,
        )

    async def check(
        self,
        code: str,
        *,
        auxiliary_sources: Mapping[str, str] | None = None,
    ) -> LeanResult:
        self.checked_code.append(code)
        self.checked_auxiliary_sources.append(dict(auxiliary_sources or {}))
        return self.result


class ManagedLeanChecker(AcceptingLeanChecker):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    async def validate_problem(self) -> None:
        self.events.append("validate problem")


class UnexpectedSearchBackend:
    async def search(self, query: str, *, max_results: int = 10) -> SearchResult:
        raise AssertionError(f"Search was not expected: {query}, max_results={max_results}")


class ManagedSearchBackend(UnexpectedSearchBackend):
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.exit_error: BaseException | None = None

    async def __aenter__(self) -> Self:
        self.events.append("enter search")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.exit_error = exc
        self.events.append("exit search")


async def test_successful_run_writes_run_artifacts(tmp_path: Path) -> None:
    run_id = UUID("0a8552d3-cf94-4640-b67c-9938387bdf7a")
    checker = AcceptingLeanChecker()
    workspace = LeanWorkspace()
    workspace.save("theorem helper : True := by trivial\n", "Helper.lean")

    async def submit_valid_code(
        messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        user_prompts = [
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, UserPromptPart) and isinstance(part.content, str)
        ]
        assert len(user_prompts) == 1
        assert PROBLEM.source in user_prompts[0]
        assert "FormalizerSubmission.solution : FormalizerProblem.Target" in user_prompts[0]
        assert [tool.name for tool in agent_info.output_tools] == ["final_submission"]
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_submission",
                    args={"code": VALID_CODE},
                )
            ],
            usage=RequestUsage(
                input_tokens=120,
                cache_read_tokens=80,
                output_tokens=15,
                details={"reasoning_tokens": 4},
                cost=Decimal("0.012"),
            ),
            provider_details={"cost": 0.01},
        )

    result = await run_formalizer(
        PROBLEM,
        agent=create_agent(FunctionModel(submit_valid_code)),
        deps=AgentDependencies(
            lean_checker=checker,
            lean_workspace=workspace,
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
    assert result.output.code == VALID_CODE
    assert result.output.check == checker.result
    assert checker.checked_code == [VALID_CODE]
    assert stored_messages == result.all_messages()
    assert manifest.run_id == run_id
    assert manifest.problem == PROBLEM
    assert manifest.status == "verified"
    assert manifest.usage.model_responses == 1
    assert manifest.usage.input_tokens == 120
    assert manifest.usage.cache_read_tokens == 80
    assert manifest.usage.output_tokens == 15
    assert manifest.usage.reasoning_tokens == 4
    assert manifest.usage.pydantic_estimated_cost_usd == Decimal("0.012")
    assert manifest.usage.pydantic_costed_responses == 1
    assert manifest.usage.provider_reported_cost_usd == Decimal("0.01")
    assert manifest.usage.provider_costed_responses == 1
    assert manifest.verification == checker.result
    assert manifest.started_at <= manifest.finished_at
    assert (run_dir / "final.lean").read_text() == VALID_CODE
    assert (run_dir / "problem.lean").read_text() == PROBLEM.source
    assert (
        run_dir / "workspace" / "FormalizerWorkspace" / "Helper.lean"
    ).read_text() == "theorem helper : True := by trivial\n"


async def test_rejected_submission_writes_failed_run_artifacts(
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
            ],
            usage=RequestUsage(
                input_tokens=50,
                output_tokens=7,
                cost=Decimal("0.004"),
            ),
            provider_details={"cost": 0.0035},
        )

    result = await run_formalizer(
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

    assert result.output.code == INVALID_CODE
    assert result.output.check == checker.result
    assert checker.checked_code == [INVALID_CODE]
    assert len(submission_calls) == 1
    assert retry_prompts == []
    assert manifest["run_id"] == str(run_id)
    assert manifest["problem"] == PROBLEM.model_dump()
    assert manifest["status"] == "failed"
    assert manifest["error_type"] is None
    assert manifest["error"] is None
    assert manifest["verification"] == {
        "stdout": "",
        "stderr": "Main.lean:4:2: error: type mismatch",
        "exit_code": 1,
        "duration_s": 0.1,
        "timed_out": False,
        "verification_error": None,
    }
    assert manifest["started_at"] <= manifest["finished_at"]
    assert (run_dir / "final.lean").read_text() == INVALID_CODE
    assert (run_dir / "problem.lean").read_text() == PROBLEM.source


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
            ],
            usage=RequestUsage(
                input_tokens=50,
                output_tokens=7,
                cost=Decimal("0.004"),
            ),
            provider_details={"cost": 0.0035},
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
            "verification_error": None,
        }
    ]
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "UsageLimitExceeded"
    assert manifest["usage"] == {
        "model_responses": 1,
        "input_tokens": 50,
        "cache_write_tokens": 0,
        "cache_read_tokens": 0,
        "output_tokens": 7,
        "reasoning_tokens": 0,
        "pydantic_estimated_cost_usd": "0.004",
        "pydantic_costed_responses": 1,
        "provider_reported_cost_usd": "0.0035",
        "provider_costed_responses": 1,
    }
    assert not (run_dir / "final.lean").exists()


async def test_failed_run_persists_auxiliary_workspace(tmp_path: Path) -> None:
    run_id = UUID("a275d296-f49b-48b9-a479-6463994800c7")

    async def save_helper(
        messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        assert messages
        assert "save" in [tool.name for tool in agent_info.function_tools]
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="save",
                    args={
                        "code": "theorem helper : True := by trivial\n",
                        "filename": "Helper.lean",
                    },
                )
            ]
        )

    with pytest.raises(UsageLimitExceeded):
        await run_formalizer(
            PROBLEM,
            agent=create_agent(FunctionModel(save_helper)),
            deps=AgentDependencies(
                lean_checker=AcceptingLeanChecker(),
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

    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "UsageLimitExceeded"
    assert (
        run_dir / "workspace" / "FormalizerWorkspace" / "Helper.lean"
    ).read_text() == "theorem helper : True := by trivial\n"


async def test_formalize_wires_settings_and_manages_search_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = UUID("6628e2cf-3404-4f30-a146-c9c00e6e05ef")
    settings = Settings(
        model_name="test:model",
        model_settings=ModelSettings(temperature=0.2, max_tokens=4096),
        sandbox=SandboxSettings(docker_image="formalizer:test"),
        run=RunSettings(runs_dir=tmp_path),
    )
    events: list[str] = []
    checker = ManagedLeanChecker(events)
    search_backend = ManagedSearchBackend(events)
    expected_agent = object()
    expected_result = object()
    lean_settings: list[SandboxSettings] = []
    search_settings: list[SandboxSettings] = []
    model_names: list[object] = []
    seen_model_settings: list[ModelSettings | None] = []
    seen_instructions: list[str] = []
    run_problems: list[FormalizationProblem] = []
    run_agents: list[object] = []
    run_dependencies: list[AgentDependencies] = []
    run_settings: list[RunSettings] = []
    run_ids: list[UUID | None] = []

    problem_codes: list[str | None] = []

    def fake_lean_checker(
        sandbox_settings: SandboxSettings,
        *,
        problem_code: str | None = None,
    ) -> ManagedLeanChecker:
        lean_settings.append(sandbox_settings)
        problem_codes.append(problem_code)
        return checker

    def fake_search_backend(sandbox_settings: SandboxSettings) -> ManagedSearchBackend:
        search_settings.append(sandbox_settings)
        return search_backend

    def fake_create_agent(
        model: object,
        *,
        model_settings: ModelSettings | None = None,
        instructions: str,
    ) -> object:
        events.append("create agent")
        model_names.append(model)
        seen_model_settings.append(model_settings)
        seen_instructions.append(instructions)
        return expected_agent

    async def fake_run_formalizer(
        problem: FormalizationProblem,
        *,
        agent: object,
        deps: AgentDependencies,
        settings: RunSettings,
        run_id: UUID | None = None,
    ) -> object:
        events.append("run")
        run_problems.append(problem)
        run_agents.append(agent)
        run_dependencies.append(deps)
        run_settings.append(settings)
        run_ids.append(run_id)
        return expected_result

    monkeypatch.setattr(run_module, "DockerLeanChecker", fake_lean_checker)
    monkeypatch.setattr(run_module, "DockerLoogleBackend", fake_search_backend)
    monkeypatch.setattr(run_module, "create_agent", fake_create_agent)
    monkeypatch.setattr(run_module, "run_formalizer", fake_run_formalizer)

    experiment_instructions = "experiment instructions"
    result = await run_module.formalize(
        PROBLEM,
        settings,
        run_id=run_id,
        instructions=experiment_instructions,
    )

    assert result is expected_result
    assert lean_settings == [settings.sandbox]
    assert problem_codes == [PROBLEM.source]
    assert search_settings == [settings.sandbox]
    assert model_names == [settings.model_name]
    assert seen_model_settings == [settings.model_settings]
    assert seen_instructions == [experiment_instructions]
    assert run_problems == [PROBLEM]
    assert run_agents == [expected_agent]
    assert len(run_dependencies) == 1
    assert run_dependencies[0].lean_checker is checker
    assert run_dependencies[0].search_backend is search_backend
    assert run_settings == [settings.run]
    assert run_ids == [run_id]
    assert events == [
        "validate problem",
        "create agent",
        "enter search",
        "run",
        "exit search",
    ]


async def test_formalize_closes_search_backend_when_run_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(model_name="test:model")
    events: list[str] = []
    checker = ManagedLeanChecker(events)
    search_backend = ManagedSearchBackend(events)
    expected_error = RuntimeError("run failed")

    def fake_lean_checker(
        settings: SandboxSettings,
        *,
        problem_code: str | None = None,
    ) -> ManagedLeanChecker:
        assert problem_code == PROBLEM.source
        return checker

    def fake_search_backend(settings: SandboxSettings) -> ManagedSearchBackend:
        return search_backend

    def fake_create_agent(
        model: object,
        *,
        model_settings: ModelSettings | None = None,
    ) -> object:
        events.append("create agent")
        return object()

    async def fail_run(
        problem: FormalizationProblem,
        *,
        agent: object,
        deps: AgentDependencies,
        settings: RunSettings,
        run_id: UUID | None = None,
    ) -> object:
        assert run_id is None
        events.append("run")
        raise expected_error

    monkeypatch.setattr(run_module, "DockerLeanChecker", fake_lean_checker)
    monkeypatch.setattr(run_module, "DockerLoogleBackend", fake_search_backend)
    monkeypatch.setattr(run_module, "create_agent", fake_create_agent)
    monkeypatch.setattr(run_module, "run_formalizer", fail_run)

    with pytest.raises(RuntimeError) as exc_info:
        await run_module.formalize(PROBLEM, settings)

    assert exc_info.value is expected_error
    assert search_backend.exit_error is expected_error
    assert events == [
        "validate problem",
        "create agent",
        "enter search",
        "run",
        "exit search",
    ]


async def test_formalize_stops_before_agent_and_search_when_problem_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = Settings(
        model_name="test:model",
        run=RunSettings(runs_dir=tmp_path),
    )
    events: list[str] = []
    expected_error = InvalidLeanProblem("trusted problem is invalid")

    class InvalidProblemChecker:
        async def validate_problem(self) -> None:
            events.append("validate problem")
            raise expected_error

    def fake_lean_checker(
        settings: SandboxSettings,
        *,
        problem_code: str | None = None,
    ) -> InvalidProblemChecker:
        assert problem_code == PROBLEM.source
        return InvalidProblemChecker()

    def unexpected_create_agent(
        model: object,
        *,
        model_settings: ModelSettings | None = None,
    ) -> object:
        raise AssertionError("Agent creation was not expected")

    def unexpected_search_backend(settings: SandboxSettings) -> object:
        raise AssertionError("Search startup was not expected")

    monkeypatch.setattr(run_module, "DockerLeanChecker", fake_lean_checker)
    monkeypatch.setattr(run_module, "create_agent", unexpected_create_agent)
    monkeypatch.setattr(run_module, "DockerLoogleBackend", unexpected_search_backend)

    with pytest.raises(InvalidLeanProblem) as exc_info:
        await run_module.formalize(PROBLEM, settings)

    assert exc_info.value is expected_error
    assert events == ["validate problem"]
    assert list(tmp_path.iterdir()) == []
