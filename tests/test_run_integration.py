from pathlib import Path
from uuid import UUID

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from formalizer.agent import AgentDependencies, Submission, create_agent
from formalizer.lean import DockerLeanChecker, LeanResult
from formalizer.problem import FormalizationProblem
from formalizer.run import RunManifest, run_formalizer
from formalizer.search import DockerLoogleBackend, SearchResult
from formalizer.settings import RunSettings, SandboxSettings

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


@pytest.mark.integration
async def test_docker_backed_run_uses_tools_and_persists_verified_artifacts(
    tmp_path: Path,
) -> None:
    run_id = UUID("51fd78df-8990-451a-8625-6c5d4fa88ac4")
    model_calls = 0

    async def search_check_and_submit(
        messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1

        assert {tool.name for tool in agent_info.function_tools} == {
            "lean_execute",
            "mathlib_search",
        }
        assert [tool.name for tool in agent_info.output_tools] == ["final_submission"]

        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="mathlib_search",
                        args={"query": "Nat.add_comm"},
                    )
                ]
            )

        if model_calls == 2:
            search_results = [
                part.content
                for message in messages
                for part in message.parts
                if isinstance(part, ToolReturnPart) and part.tool_name == "mathlib_search"
            ]
            assert len(search_results) == 1
            search_result = search_results[0]
            assert isinstance(search_result, SearchResult)
            assert any(hit.name == "Nat.add_comm" for hit in search_result.hits)

            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="lean_execute",
                        args={"code": VALID_CODE},
                    )
                ]
            )

        if model_calls == 3:
            lean_results = [
                part.content
                for message in messages
                for part in message.parts
                if isinstance(part, ToolReturnPart) and part.tool_name == "lean_execute"
            ]
            assert len(lean_results) == 1
            lean_result = lean_results[0]
            assert isinstance(lean_result, LeanResult)
            assert lean_result.accepted

            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="final_submission",
                        args={"code": VALID_CODE},
                    )
                ]
            )

        raise AssertionError("The agent made an unexpected additional model request")

    sandbox_settings = SandboxSettings()
    checker = DockerLeanChecker(
        sandbox_settings,
        problem_code=PROBLEM.source,
    )

    async with DockerLoogleBackend(sandbox_settings) as search_backend:
        result = await run_formalizer(
            PROBLEM,
            agent=create_agent(FunctionModel(search_check_and_submit)),
            deps=AgentDependencies(
                lean_checker=checker,
                search_backend=search_backend,
            ),
            settings=RunSettings(runs_dir=tmp_path),
            run_id=run_id,
        )

    run_dir = tmp_path / str(run_id)
    manifest = RunManifest.model_validate_json((run_dir / "run.json").read_bytes())
    stored_messages = ModelMessagesTypeAdapter.validate_json(
        (run_dir / "messages.json").read_bytes()
    )
    tool_calls = [
        part.tool_name
        for message in stored_messages
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]

    assert model_calls == 3
    assert tool_calls == ["mathlib_search", "lean_execute", "final_submission"]
    assert stored_messages == ModelMessagesTypeAdapter.validate_json(result.all_messages_json())
    assert isinstance(result.output, Submission)
    assert result.output.code == VALID_CODE
    assert result.output.check.accepted
    assert manifest.run_id == run_id
    assert manifest.problem == PROBLEM
    assert manifest.status == "verified"
    assert manifest.verification == result.output.check
    assert (run_dir / "problem.lean").read_text() == PROBLEM.source
    assert (run_dir / "final.lean").read_text() == VALID_CODE
