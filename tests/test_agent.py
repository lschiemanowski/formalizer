from collections.abc import Iterator, Mapping

import pytest
from pydantic_ai import ModelSettings
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from formalizer.agent import AgentDependencies, create_agent
from formalizer.lean import DeletedLeanFile, LeanResult, LeanWorkspace, SavedLeanFile
from formalizer.search import SearchHit, SearchResult

VALID_CODE = """import Mathlib

example : 1 + 1 = 2 := by norm_num
"""

INVALID_CODE = """import Mathlib

example : 1 + 1 = 2 := by
  exact 0
"""


class FakeLeanChecker:
    def __init__(self, results: list[LeanResult]) -> None:
        self._results: Iterator[LeanResult] = iter(results)
        self.checked_code: list[str] = []
        self.checked_auxiliary_sources: list[dict[str, str]] = []

    async def check(
        self,
        code: str,
        *,
        auxiliary_sources: Mapping[str, str] | None = None,
    ) -> LeanResult:
        self.checked_code.append(code)
        self.checked_auxiliary_sources.append(dict(auxiliary_sources or {}))
        return next(self._results)


class UnexpectedSearchBackend:
    async def search(self, query: str, *, max_results: int = 10) -> SearchResult:
        raise AssertionError(f"Search was not expected: {query}, max_results={max_results}")


class FakeSearchBackend:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results: Iterator[SearchResult] = iter(results)
        self.queries: list[tuple[str, int]] = []

    async def search(self, query: str, *, max_results: int = 10) -> SearchResult:
        self.queries.append((query, max_results))
        return next(self._results)


def final_submission_response(code: str) -> ModelResponse:
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name="final_submission",
                args={"code": code},
            )
        ]
    )


async def test_verified_final_submission_ends_the_run() -> None:
    accepted_result = LeanResult(stdout="", stderr="", exit_code=0, duration_s=0.1)
    checker = FakeLeanChecker([accepted_result])

    async def submit_valid_code(
        messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        assert messages
        assert [tool.name for tool in agent_info.output_tools] == ["final_submission"]
        return final_submission_response(VALID_CODE)

    agent = create_agent(FunctionModel(submit_valid_code))

    result = await agent.run(
        "Prove that 1 + 1 = 2.",
        deps=AgentDependencies(
            lean_checker=checker,
            search_backend=UnexpectedSearchBackend(),
        ),
    )

    assert result.output.code == VALID_CODE
    assert result.output.check == accepted_result
    assert checker.checked_code == [VALID_CODE]


async def test_rejected_final_submission_ends_the_run_without_retry() -> None:
    rejected_result = LeanResult(
        stdout="",
        stderr="Main.lean:3:28: error: tactic 'rfl' failed",
        exit_code=1,
        duration_s=0.1,
    )
    checker = FakeLeanChecker([rejected_result])
    model_calls = 0

    async def submit_invalid_code(
        messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        assert model_calls == 1, "final_submission must not trigger another model request"
        assert messages
        assert [tool.name for tool in agent_info.output_tools] == ["final_submission"]
        return final_submission_response(INVALID_CODE)

    agent = create_agent(FunctionModel(submit_invalid_code))

    result = await agent.run(
        "Prove that 1 + 1 = 2.",
        deps=AgentDependencies(
            lean_checker=checker,
            search_backend=UnexpectedSearchBackend(),
        ),
    )

    assert model_calls == 1
    assert result.output.code == INVALID_CODE
    assert result.output.check == rejected_result
    assert not result.output.check.accepted
    assert checker.checked_code == [INVALID_CODE]


async def test_lean_execute_returns_diagnostics_without_ending_the_run() -> None:
    rejected_result = LeanResult(
        stdout="",
        stderr="Main.lean:4:2: error: type mismatch",
        exit_code=1,
        duration_s=0.1,
    )
    checker = FakeLeanChecker(
        [
            rejected_result,
            LeanResult(stdout="", stderr="", exit_code=0, duration_s=0.1),
        ]
    )
    model_calls = 0

    async def check_then_submit(
        messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1

        assert "lean_execute" in [tool.name for tool in agent_info.function_tools]

        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="lean_execute",
                        args={"code": INVALID_CODE},
                    )
                ]
            )

        lean_results = [
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "lean_execute"
        ]
        assert lean_results == [rejected_result]
        return final_submission_response(VALID_CODE)

    agent = create_agent(FunctionModel(check_then_submit))

    result = await agent.run(
        "Prove that 1 + 1 = 2.",
        deps=AgentDependencies(
            lean_checker=checker,
            search_backend=UnexpectedSearchBackend(),
        ),
    )

    assert result.output.code == VALID_CODE
    assert result.output.check.accepted
    assert checker.checked_code == [INVALID_CODE, VALID_CODE]


async def test_agent_can_save_import_and_delete_auxiliary_lean_modules() -> None:
    accepted_result = LeanResult(stdout="", stderr="", exit_code=0, duration_s=0.1)
    checker = FakeLeanChecker([accepted_result, accepted_result])
    workspace = LeanWorkspace()
    model_calls = 0
    main_with_helper = """\
import FormalizerWorkspace.Helper

namespace FormalizerSubmission

theorem solution : True := FormalizerWorkspace.helper

end FormalizerSubmission
"""

    async def manage_workspace_then_submit(
        messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        function_tools = [tool.name for tool in agent_info.function_tools]
        assert "save" in function_tools
        assert "delete" in function_tools
        assert "lean_execute" in function_tools

        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="save",
                        args={
                            "code": (
                                "namespace FormalizerWorkspace\n"
                                "theorem helper : True := by trivial\n"
                                "end FormalizerWorkspace\n"
                            ),
                            "filename": "Helper.lean",
                        },
                    )
                ]
            )

        if model_calls == 2:
            saved = [
                part.content
                for message in messages
                for part in message.parts
                if isinstance(part, ToolReturnPart) and part.tool_name == "save"
            ]
            assert saved == [
                SavedLeanFile(
                    filename="Helper.lean",
                    module="FormalizerWorkspace.Helper",
                    revision=1,
                )
            ]
            return ModelResponse(
                parts=[ToolCallPart(tool_name="lean_execute", args={"code": main_with_helper})]
            )

        if model_calls == 3:
            assert checker.checked_auxiliary_sources == [
                {
                    "FormalizerWorkspace/Helper.lean": (
                        "namespace FormalizerWorkspace\n"
                        "theorem helper : True := by trivial\n"
                        "end FormalizerWorkspace\n"
                    )
                }
            ]
            return ModelResponse(
                parts=[ToolCallPart(tool_name="delete", args={"filename": "Helper.lean"})]
            )

        deleted = [
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "delete"
        ]
        assert deleted == [DeletedLeanFile(filename="Helper.lean", deleted=True, revision=2)]
        return final_submission_response(VALID_CODE)

    agent = create_agent(FunctionModel(manage_workspace_then_submit))

    result = await agent.run(
        "Prove that 1 + 1 = 2.",
        deps=AgentDependencies(
            lean_checker=checker,
            lean_workspace=workspace,
            search_backend=UnexpectedSearchBackend(),
        ),
    )

    assert model_calls == 4
    assert result.output.check.accepted
    assert checker.checked_code == [main_with_helper, VALID_CODE]
    assert checker.checked_auxiliary_sources[-1] == {}
    assert workspace.sources == {}


@pytest.mark.parametrize(
    ("tool_args", "expected_max_results"),
    [
        ({"query": "Nat.add_comm"}, 10),
        ({"query": "Nat.add_comm", "max_results": 100}, 100),
    ],
)
async def test_mathlib_search_returns_results_without_ending_the_run(
    tool_args: dict[str, object],
    expected_max_results: int,
) -> None:
    expected_search_result = SearchResult(
        query="Nat.add_comm",
        hits=(
            SearchHit(
                name="Nat.add_comm",
                type="Nat.add_comm (n m : Nat) : n + m = m + n",
                module="Mathlib.Data.Nat.Basic",
                doc="Addition is commutative.",
            ),
        ),
        total_count=1,
        header="Found 1 declaration",
        duration_s=0.1,
    )
    search_backend = FakeSearchBackend([expected_search_result])
    checker = FakeLeanChecker([LeanResult(stdout="", stderr="", exit_code=0, duration_s=0.1)])
    model_calls = 0

    async def search_then_submit(
        messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1

        search_tool = next(
            tool for tool in agent_info.function_tools if tool.name == "mathlib_search"
        )
        max_results_schema = search_tool.parameters_json_schema["properties"]["max_results"]
        assert max_results_schema["default"] == 10
        assert max_results_schema["minimum"] == 1
        assert max_results_schema["maximum"] == 100

        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="mathlib_search",
                        args=tool_args,
                    )
                ]
            )

        search_results = [
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "mathlib_search"
        ]
        assert search_results == [expected_search_result]
        return final_submission_response(VALID_CODE)

    agent = create_agent(FunctionModel(search_then_submit))

    result = await agent.run(
        "Prove that addition of natural numbers is commutative.",
        deps=AgentDependencies(
            lean_checker=checker,
            search_backend=search_backend,
        ),
    )

    assert search_backend.queries == [("Nat.add_comm", expected_max_results)]
    assert checker.checked_code == [VALID_CODE]
    assert result.output.code == VALID_CODE
    assert result.output.check.accepted


@pytest.mark.parametrize("max_results", [0, 101])
async def test_mathlib_search_rejects_result_limits_outside_agent_bounds(
    max_results: int,
) -> None:
    checker = FakeLeanChecker([LeanResult(stdout="", stderr="", exit_code=0, duration_s=0.1)])
    model_calls = 0

    async def invalid_search_then_submit(
        messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1

        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="mathlib_search",
                        args={"query": "Nat.add_comm", "max_results": max_results},
                    )
                ]
            )

        retry_prompts = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, RetryPromptPart) and part.tool_name == "mathlib_search"
        ]
        assert retry_prompts
        assert agent_info.function_tools
        return final_submission_response(VALID_CODE)

    agent = create_agent(FunctionModel(invalid_search_then_submit))

    result = await agent.run(
        "Prove that addition of natural numbers is commutative.",
        deps=AgentDependencies(
            lean_checker=checker,
            search_backend=UnexpectedSearchBackend(),
        ),
    )

    assert model_calls == 2
    assert result.output.code == VALID_CODE
    assert result.output.check.accepted


async def test_agent_applies_configured_model_settings() -> None:
    checker = FakeLeanChecker([LeanResult(stdout="", stderr="", exit_code=0, duration_s=0.1)])
    model_settings = ModelSettings(temperature=0.2, max_tokens=4096)

    async def assert_model_settings(
        messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        assert messages
        assert agent_info.model_settings == model_settings
        return final_submission_response(VALID_CODE)

    agent = create_agent(
        FunctionModel(assert_model_settings),
        model_settings=model_settings,
    )

    result = await agent.run(
        "Prove that 1 + 1 = 2.",
        deps=AgentDependencies(
            lean_checker=checker,
            search_backend=UnexpectedSearchBackend(),
        ),
    )

    assert result.output.code == VALID_CODE
    assert result.output.check.accepted
    assert checker.checked_code == [VALID_CODE]
