from dataclasses import dataclass

from pydantic_ai import Agent, ModelRetry, ModelSettings, RunContext, ToolOutput
from pydantic_ai.models import Model

from formalizer.lean import LeanChecker, LeanResult
from formalizer.problem import FormalizationProblem
from formalizer.search import SearchBackend, SearchResult

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

Therefore, Main.lean must expose FormalizerSubmission.solution with a type definitionally equal to
FormalizerProblem.Target. Merely compiling some other theorem or example does not solve the task.

You have three tools:

- mathlib_search searches Mathlib for relevant declarations and returns matching names, types,
  modules, and documentation.

- lean_execute checks a candidate solution without ending the run. Pass the complete contents of
  Main.lean, not a fragment or patch. It compiles FormalizerProblem.lean, your Main.lean, and the
  trusted FormalizerCheck.lean shown above. It returns Lean's output, diagnostics, exit status,
  duration, and timeout status. Each call uses a fresh isolated environment, so files and state do
  not persist between calls.

- final_submission is the only way to finish successfully. Pass the complete contents of the final
  Main.lean. It performs the same three-module check again in a fresh environment. If the submission
  is accepted, the run ends with that verified source. If it is rejected, the run continues and you
  receive diagnostics so you can correct and resubmit it.

Use mathlib_search and lean_execute as often as needed. Call final_submission only when you believe
the complete solution is ready.
"""


@dataclass(frozen=True, slots=True)
class AgentDependencies:
    lean_checker: LeanChecker
    search_backend: SearchBackend


@dataclass(frozen=True, slots=True)
class VerifiedSubmission:
    code: str


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
    """Check a candidate Main.lean against the immutable problem without ending the run."""
    return await ctx.deps.lean_checker.check(code)


async def final_submission(
    ctx: RunContext[AgentDependencies],
    code: str,
) -> VerifiedSubmission:
    """Submit the final Main.lean proving FormalizerProblem.Target."""
    result = await ctx.deps.lean_checker.check(code)
    if not result.accepted:
        raise ModelRetry(
            "Lean rejected the submitted Main.lean. Correct it and submit the complete file again.\n"
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
