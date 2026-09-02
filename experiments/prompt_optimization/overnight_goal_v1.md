# Draft goal: Qwen 3.6 GEPA overnight prompt optimization

Status: **inactive draft for user review**. Activating this goal authorizes the approved model calls
described below; creating or running it before approval is prohibited.

## Goal text

Run the frozen Formalizer GEPA 0.1.4 prompt-optimization experiment against the local
`qwen-3.6-35b-a3b` server at `http://127.0.0.1:8000/v1`, using
`experiments/prompt_optimization/gepa_overnight_v1.json` exactly as approved. First run and audit
the one-case canary. Continue to the overnight optimization only if the canary proves that prompt
injection, task transcripts, bounded discovery feedback, selection withholding, hashes, and
terminal status are all correct. Preserve precise append-only logs of every task evaluation and
every reflection prompt/output. Stop cleanly at the first configured budget or stop signal, audit
the complete artifact tree, and report the seed and selected candidate without modifying or
promoting the production prompt and without running any repository evaluation set.

## Required execution sequence

1. Verify the current checkout and rerun the project-local GEPA preflight with all nine discovery
   and five selection case names. Verify `/v1/models` contains `qwen-3.6-35b-a3b` with
   `n_ctx >= 262144`. Do not continue if the frozen configuration or any pinned input hash differs.
2. Run the runner's dry plans for both `canary` and `optimize`. Confirm that the endpoint, exact
   allowlists, seed prompt, budgets, concurrency one, and output directories match the approved
   contract.
3. Launch a fresh canary directory with `--execute --approve-model-calls`. Audit its immutable
   manifest, environment snapshot, candidate file, started/finished evaluation records, complete
   Formalizer `run.json` and `messages.json`, feedback hash and rendered size, result, and terminal
   status. Confirm no selection transcript and no reflection call was made.
4. If and only if the canary audit passes, launch the full optimization in a different fresh output
   directory with `--execute --approve-model-calls`. Never reuse or overwrite an existing run.
5. Monitor the run from `status.json`, `events.jsonl`, `evaluations.jsonl`, `reflections.jsonl`, and
   GEPA's own run log/state. Preserve every infrastructure-invalid attempt separately and retry it
   only according to the frozen policy. Do not score infrastructure-invalid attempts as model
   failures. Do not manually edit candidates or feed selection transcripts into reflection.
6. Stop at the earliest of 60 metric calls, 10 candidate proposals, 12 hours, `gepa/gepa.stop`, a
   termination signal, or an experiment-integrity failure. A phase-boundary stop may finish the
   already-started evaluation; record the observed totals.
7. On completion or failure, verify source and output hashes, candidate and evaluation counts,
   append-only event pairing, feedback bounds, selection withholding, reflection prompt/output
   pairing, and terminal status. Summarize accepted and rejected proposals, seed versus selected
   selection scores, failure-distribution changes, task/reflection token usage, unavailable cost
   coverage, and all invalid attempts. Do not run `formalizer-eval-v1` or promote `best_prompt.txt`
   to `src/formalizer/agent.py`; both require separate user approval.

## Frozen proposed budget

- Task and reflection server: `qwen-3.6-35b-a3b`, local OpenAI-compatible endpoint,
  `n_ctx >= 262144`.
- Discovery cases: 9; selection cases: 5; both are disjoint original `train` cases.
- Task concurrency: 1; stochastic repetitions: 1.
- Per task attempt: 200 requests, 100,000 output tokens, 2,000,000 total tokens, 30-minute wall
  timeout, and two identical infrastructure retries.
- Reflection: minibatch 3, temperature 0.7, 8,192 output tokens, two transport retries.
- GEPA: seed 0, strict improvement, disk evaluation cache, at most 60 metric calls, 10 proposals,
  and 12 hours.
- Feedback: at most 6,000 characters under GEPA 0.1.4's actual Markdown rendering.

## Logging stipulation

The run is invalid unless it preserves exact candidate texts, exact reflection prompts and outputs,
per-attempt task transcripts, usage, timestamps, terminal classifications, source and feedback
hashes, immutable launch metadata, incrementally flushed JSONL events/evaluations, GEPA state, the
selected prompt, and a terminal result/status. Secrets and authorization headers must never be
recorded.
