# formalizer

`formalizer` is a simple agent that uses a language model to prove propositions in
Lean. Given a Lean statement, it searches for useful lemmas, writes a proof, and
revises it using Lean's feedback. The final proof is checked against the original
statement.

The agent uses Pydantic AI, Docker for isolated execution, Lean and Mathlib for
proof checking, and Loogle for lemma search. Evaluations optionally use Logfire
for tracing.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Docker](https://docs.docker.com/get-started/get-docker/). Docker must be running
and accessible to your user. Python 3.12 or newer is required; uv can install it.

```sh
git clone https://github.com/lschiemanowski/formalizer.git
cd formalizer
uv sync --locked --no-dev
./docker/build.sh
```

The Docker image includes Lean, Mathlib, and Loogle; the first build can take a while.

## Run an example

Set your OpenRouter API key, then ask the agent to prove that `1 + 1 = 2`:

```sh
export OPENROUTER_API_KEY="your-api-key"

uv run --no-dev formalizer \
  --model openrouter:deepseek/deepseek-v4-flash-0731 \
  'import Mathlib.Data.Nat.Basic

namespace FormalizerProblem
def Target : Prop := 1 + 1 = 2
end FormalizerProblem'
```

The argument is Lean source text defining `FormalizerProblem.Target`, the
proposition the agent will attempt to prove.
On success, the command prints the verified Lean proof. Run artifacts are saved
under `runs/`.

## More

The repository also contains benchmarking and evaluation code in
`src/formalizer/eval/` and experiment code in `experiments/`. Ask a coding agent
to explore the repository and explain how to use these.

Related [blog posts](https://lschiemanowski.github.io/blog/):

- [Building and evaluating a simple agent for proving mathematical results in Lean](https://lschiemanowski.github.io/blog/prover_1.html)
- [Improving agent performance via prompt optimization](https://lschiemanowski.github.io/blog/prompt-optimization.html) (work in progress)
