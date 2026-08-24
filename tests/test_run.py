from pathlib import Path
from uuid import UUID

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from formalizer.agent import AgentDependencies, VerifiedSubmission, create_agent
from formalizer.lean import LeanResult
from formalizer.run import RunManifest, run_formalizer
from formalizer.search import SearchResult

PROBLEM = "Prove that 1 + 1 = 2."
VALID_CODE = """import Mathlib

example : 1 + 1 = 2 := by norm_num
"""


class AcceptingLeanChecker:
    def __init__(self) -> None:
        self.checked_code: list[str] = []

    async def check(self, code: str) -> LeanResult:
        self.checked_code.append(code)
        return LeanResult(stdout="", stderr="", exit_code=0, duration_s=0.1)


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
        runs_dir=tmp_path,
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
