from collections.abc import Iterator

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from formalizer.agent import AgentDependencies, VerifiedSubmission, create_agent
from formalizer.lean import LeanResult
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

    async def check(self, code: str) -> LeanResult:
        self.checked_code.append(code)
        return next(self._results)


class UnexpectedSearchBackend:
    async def search(self, query: str) -> SearchResult:
        raise AssertionError(f"Search was not expected: {query}")


class FakeSearchBackend:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results: Iterator[SearchResult] = iter(results)
        self.queries: list[str] = []

    async def search(self, query: str) -> SearchResult:
        self.queries.append(query)
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
    checker = FakeLeanChecker([LeanResult(stdout="", stderr="", exit_code=0, duration_s=0.1)])

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

    assert result.output == VerifiedSubmission(code=VALID_CODE)
    assert checker.checked_code == [VALID_CODE]


async def test_rejected_final_submission_is_retried() -> None:
    checker = FakeLeanChecker(
        [
            LeanResult(
                stdout="",
                stderr="Main.lean:3:28: error: tactic 'rfl' failed",
                exit_code=1,
                duration_s=0.1,
            ),
            LeanResult(stdout="", stderr="", exit_code=0, duration_s=0.1),
        ]
    )
    responses = iter(
        [
            final_submission_response(INVALID_CODE),
            final_submission_response(VALID_CODE),
        ]
    )

    async def submit_then_correct(
        messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        assert messages
        assert [tool.name for tool in agent_info.output_tools] == ["final_submission"]
        return next(responses)

    agent = create_agent(FunctionModel(submit_then_correct))

    result = await agent.run(
        "Prove that 1 + 1 = 2.",
        deps=AgentDependencies(
            lean_checker=checker,
            search_backend=UnexpectedSearchBackend(),
        ),
    )

    assert result.output == VerifiedSubmission(code=VALID_CODE)
    assert checker.checked_code == [INVALID_CODE, VALID_CODE]


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

    assert result.output == VerifiedSubmission(code=VALID_CODE)
    assert checker.checked_code == [INVALID_CODE, VALID_CODE]


async def test_mathlib_search_returns_results_without_ending_the_run() -> None:
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

        assert "mathlib_search" in [tool.name for tool in agent_info.function_tools]

        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="mathlib_search",
                        args={"query": "Nat.add_comm"},
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

    assert search_backend.queries == ["Nat.add_comm"]
    assert checker.checked_code == [VALID_CODE]
    assert result.output == VerifiedSubmission(code=VALID_CODE)
