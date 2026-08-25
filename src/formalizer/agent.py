from dataclasses import dataclass

from pydantic_ai import Agent, ModelSettings, RunContext, ToolOutput
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

    #print axioms FormalizerSubmission.solution

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

- final_submission is the only way to submit your final answer and finish the run. Pass the
  complete contents of the final Main.lean. It performs the same three-module check again in a
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
) -> Submission:
    """End the run with this Main.lean and its verification result."""
    check = await ctx.deps.lean_checker.check(code)
    return Submission(code=code, check=check)


def create_agent(
    model: Model | str,
    *,
    model_settings: ModelSettings | None = None,
) -> Agent[AgentDependencies, Submission]:
    return Agent[AgentDependencies, Submission](
        model,
        deps_type=AgentDependencies,
        instructions=_INSTRUCTIONS,
        model_settings=model_settings,
        tools=[mathlib_search, lean_execute],
        output_type=ToolOutput[Submission](
            final_submission,
            name="final_submission",
        ),
    )
