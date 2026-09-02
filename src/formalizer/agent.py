from dataclasses import dataclass, field
from typing import Annotated

from pydantic import Field
from pydantic_ai import Agent, ModelRetry, ModelSettings, RunContext, TextOutput, ToolOutput
from pydantic_ai.models import Model

from formalizer.lean import (
    InvalidLeanWorkspacePath,
    LeanChecker,
    LeanResult,
    LeanWorkspace,
    SavedLeanFile,
)
from formalizer.problem import FormalizationProblem
from formalizer.search import InvalidSearchQuery, SearchBackend, SearchResult

_INSTRUCTIONS = """\
You are a Lean 4 theorem-proving agent working with Mathlib.

Each task provides an immutable module named FormalizerProblem. This module contains the problem's
imports, definitions, and a proposition named FormalizerProblem.Target.

Your job is to write a separate, complete module named Main.lean that proves this exact target.
Main.lean must define:

    FormalizerSubmission.solution : FormalizerProblem.Target

You may add imports, open namespaces or scopes, and introduce helper definitions or lemmas in
Main.lean. Do not redefine or replace the supplied problem definitions or target. Do not use
`sorry`, `admit`, or introduce axioms.

Formalizer verifies your Main.lean by compiling it and then compiling this trusted module:

    import FormalizerProblem
    import Main

    example : FormalizerProblem.Target :=
      FormalizerSubmission.solution

    #print axioms FormalizerSubmission.solution

Therefore, Main.lean must expose FormalizerSubmission.solution with a type definitionally equal to
FormalizerProblem.Target. Merely compiling some other theorem or example does not solve the task.

You have four tools:

- mathlib_search searches Mathlib for relevant declarations and returns matching names, types,
  modules, and documentation. Its query must be one nonblank line. It returns at most 10 results by
  default; you may request between 1 and 100 results when a broader or narrower result set would
  help.

- save stores an auxiliary Lean file for the rest of the run. Give it a relative filename such as
  `Sylvester/Blocks.lean`; the resulting module is `FormalizerWorkspace.Sylvester.Blocks`. Saved
  files may import FormalizerProblem and one another. Saving the same filename overwrites it. Saved
  files that are not imported directly or transitively by Main.lean do not affect a check.
  Main.lean and the trusted verification modules cannot be saved this way.

- lean_execute checks a candidate solution without ending the run. Pass the complete contents of
  Main.lean, not a fragment or patch. It compiles FormalizerProblem.lean, saved auxiliary files
  imported directly or transitively by Main.lean, your Main.lean, and the trusted
  FormalizerCheck.lean shown above. It returns Lean's output, diagnostics, exit status, duration,
  and timeout status. Each check uses a fresh isolated environment; only files explicitly stored
  with save persist across checks.

- final_submission is the only way to submit your final answer and finish the run. Pass the
  complete contents of the final Main.lean. It performs the same complete check again in a
  fresh environment. The run then ends, whether the submission is accepted or rejected. A rejected
  final submission is recorded as a
  failed trial, and you will not receive another turn to correct it.

Use mathlib_search and lean_execute as often as needed, including to correct rejected candidates.
Call final_submission only when you are ready to end the run with the submitted code.
"""


@dataclass(frozen=True, slots=True)
class AgentDependencies:
    lean_checker: LeanChecker
    search_backend: SearchBackend
    lean_workspace: LeanWorkspace = field(default_factory=LeanWorkspace)


@dataclass(frozen=True, slots=True)
class Submission:
    code: str
    check: LeanResult


def problem_prompt(problem: FormalizationProblem) -> str:
    return f"""\
Prove the target defined by the following immutable problem module.

Your submission should be a complete Main.lean. A minimal submission has this shape:

    import FormalizerProblem

    namespace FormalizerSubmission

    theorem solution : FormalizerProblem.Target := by
      -- proof

    end FormalizerSubmission

This defines the required declaration:

    FormalizerSubmission.solution : FormalizerProblem.Target

You may add any further Mathlib imports, opens, or helper declarations needed for the proof.

Here is FormalizerProblem.lean:

```lean
{problem.source}
```

"""


async def mathlib_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    max_results: Annotated[int, Field(ge=1, le=100)] = 10,
) -> SearchResult:
    """Search Mathlib declarations using one nonblank query line without ending the run."""
    try:
        return await ctx.deps.search_backend.search(query, max_results=max_results)
    except InvalidSearchQuery as error:
        raise ModelRetry(str(error)) from error


async def save(
    ctx: RunContext[AgentDependencies],
    code: str,
    filename: str,
) -> SavedLeanFile:
    """Save or overwrite an auxiliary module under FormalizerWorkspace."""
    try:
        return ctx.deps.lean_workspace.save(code, filename)
    except InvalidLeanWorkspacePath as error:
        raise ModelRetry(str(error)) from error


async def lean_execute(
    ctx: RunContext[AgentDependencies],
    code: str,
) -> LeanResult:
    """Check a candidate Main.lean against the immutable problem without ending the run."""
    return await ctx.deps.lean_checker.check(
        code,
        auxiliary_sources=ctx.deps.lean_workspace.sources,
    )


async def final_submission(
    ctx: RunContext[AgentDependencies],
    code: str,
) -> Submission:
    """End the run with this Main.lean and its verification result."""
    check = await ctx.deps.lean_checker.check(
        code,
        auxiliary_sources=ctx.deps.lean_workspace.sources,
    )
    return Submission(code=code, check=check)


def _reject_plain_text(_text: str) -> Submission:
    raise ModelRetry("Plain text cannot complete the run; call one of the available tools.")


def create_agent(
    model: Model | str,
    *,
    model_settings: ModelSettings | None = None,
    instructions: str = _INSTRUCTIONS,
) -> Agent[AgentDependencies, Submission]:
    final_output = ToolOutput[Submission](
        final_submission,
        name="final_submission",
    )
    output_type = (
        [final_output, TextOutput[Submission](_reject_plain_text)]
        if model_settings is not None and model_settings.get("tool_choice") == "auto"
        else final_output
    )
    return Agent[AgentDependencies, Submission](
        model,
        deps_type=AgentDependencies,
        instructions=instructions,
        model_settings=model_settings,
        tools=[mathlib_search, save, lean_execute],
        output_type=output_type,
    )
