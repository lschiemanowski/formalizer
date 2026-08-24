from dataclasses import dataclass

from pydantic_ai import Agent, ModelRetry, ModelSettings, RunContext, ToolOutput
from pydantic_ai.models import Model

from formalizer.lean import LeanChecker, LeanResult
from formalizer.search import SearchBackend, SearchResult

_INSTRUCTIONS = """\
Formalize the user's mathematical problem as a complete Lean file using Mathlib.
Use mathlib_search to find relevant declarations in Mathlib.
Use lean_execute to check candidate files and inspect Lean diagnostics without ending the run.
Finish only by calling final_submission with the complete file. The submitted file is checked again
by Lean, and a rejected submission must be corrected before the run can succeed.
"""


@dataclass(frozen=True, slots=True)
class AgentDependencies:
    lean_checker: LeanChecker
    search_backend: SearchBackend


@dataclass(frozen=True, slots=True)
class VerifiedSubmission:
    code: str


def _rejection_diagnostic(result: LeanResult) -> str:
    diagnostic = "\n".join(
        output.strip() for output in (result.stdout, result.stderr) if output.strip()
    )
    if diagnostic:
        return diagnostic
    if result.timed_out:
        return "Lean checking timed out"
    return f"Lean exited with status {result.exit_code} without diagnostics"


async def mathlib_search(
    ctx: RunContext[AgentDependencies],
    query: str,
) -> SearchResult:
    """Search Mathlib declarations without ending the run."""
    return await ctx.deps.search_backend.search(query)


async def lean_execute(
    ctx: RunContext[AgentDependencies],
    code: str,
) -> LeanResult:
    """Check a complete Lean file and return its result without ending the run."""
    return await ctx.deps.lean_checker.check(code)


async def final_submission(
    ctx: RunContext[AgentDependencies],
    code: str,
) -> VerifiedSubmission:
    """Submit a complete Lean file as the final answer."""
    result = await ctx.deps.lean_checker.check(code)
    if not result.accepted:
        raise ModelRetry(
            "Lean rejected the submitted file. Correct it and submit the complete file again.\n"
            f"{_rejection_diagnostic(result)}"
        )

    return VerifiedSubmission(code=code)


def create_agent(
    model: Model | str,
    *,
    model_settings: ModelSettings | None = None,
) -> Agent[AgentDependencies, VerifiedSubmission]:
    return Agent[AgentDependencies, VerifiedSubmission](
        model,
        deps_type=AgentDependencies,
        instructions=_INSTRUCTIONS,
        model_settings=model_settings,
        tools=[mathlib_search, lean_execute],
        output_type=ToolOutput[VerifiedSubmission](
            final_submission,
            name="final_submission",
        ),
    )
