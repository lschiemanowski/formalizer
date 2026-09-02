---
name: formalizer-gepa-prompt-optimization
description: Design, implement, run, or audit Formalizer prompt-optimization experiments with GEPA 0.1.4. Use for selecting non-eval Lean problems, turning Formalizer run transcripts into optimizer feedback, comparing prompt candidates, or producing reproducible prompt-optimization artifacts. Do not use for ordinary Formalizer evaluations or unrelated dependency work.
---

# Formalizer GEPA prompt optimization

Optimize the agent instructions in `src/formalizer/agent.py` as an experimental artifact. Do not
edit the production prompt while searching. Materialize each candidate separately and promote a
winner only after the experiment has finished and the user approves that change.

## Start safely

1. Run `uv run --group prompt-optimization python experiments/prompt_optimization/skill/formalizer-gepa-prompt-optimization/scripts/preflight.py`.
2. Read [experiment-contract.md](references/experiment-contract.md) before selecting problems or
   creating a run.
3. Read [gepa-0.1.4-api.md](references/gepa-0.1.4-api.md) before writing or reviewing GEPA code.
4. Inspect the current prompt, evaluator, run artifacts, and relevant tests. Artifact truth wins
   over assumptions in this skill.
5. Do not make model calls until the user has approved the case allowlist, model endpoints,
   repetition count, concurrency, and budgets.

## Protect the evaluation boundary

- Initially source optimization cases only from `datasets/basic-problems-v1.yaml` entries whose
  metadata says `split: train`.
- Use exact case-name allowlists. Assert their split at runtime before creating an output directory.
- Never expose `validation` or `test` cases, their transcripts, or their outcomes to GEPA, its
  reflection model, or manual prompt revisions during the search.
- Treat `datasets/baseline-eval-v1.yaml`, `datasets/formalizer-eval-v1.yaml`, and challenge-problem
  cases as evaluation-only unless the user explicitly establishes a different dataset policy.
- Derive both GEPA's discovery set and selection `valset` from disjoint subsets of the original
  `train` split. In this context, `valset` means optimizer selection data, not the repository's
  evaluation split.
- GEPA 0.1.4 has no `test_set` argument. Run the frozen project evaluation separately, once, only
  after candidate selection. Do not feed that result back into optimization.

Stop and surface the conflict if a requested case appears in an evaluation-only source or does not
have `split: train`.

## Build an auditable experiment

Keep experiment code and versioned configuration under `experiments/prompt_optimization/`. Put
generated runs under the already-ignored `runs/prompt-optimization/<run-id>/` tree.

Before launching, persist a manifest containing at least:

- schema version, run id, creation time, status, and parent run if any;
- Git commit plus dirty-state evidence; hash every relevant dirty source/config file;
- exact dependency version and optimizer/reflection configuration;
- task-model and reflection-model provider identities, endpoints, model ids, and decoding settings;
- source dataset path and SHA-256, exact discovery and selection case names, and split assertions;
- seed prompt path/content hash, evaluator implementation hash, repetitions, timeouts, concurrency,
  and all request/token/evaluation budgets.

Create a fresh run directory; never resume into or overwrite an unrelated run. Record terminal
status for success, failure, cancellation, and infrastructure-invalid termination.

The run directory must retain:

```text
manifest.json                 immutable launch contract
environment.json              dependency/platform snapshot, with secrets redacted
events.jsonl                  append-only optimizer lifecycle
candidates/<sha256>.txt       every unique prompt candidate
evaluations.jsonl             one append-only record per candidate/example/repetition
task-runs/                    Formalizer run.json, messages.json, final.lean, workspace/
gepa/                         GEPA's run_dir, including run_log.txt and state
result.json                   serialized GEPAResult plus derived best index/score
best_prompt.txt               exact selected candidate
status.json                   terminal state and aggregate counts
```

Write records incrementally and flush them. Use candidate hashes and stable evaluation ids so a
partial run can be audited without guessing what completed. Keep infrastructure-invalid failures
separate from genuine model failures and exclude them from the optimization score unless the
experiment contract explicitly says otherwise.

## Make transcripts useful without making them dangerous

Each Formalizer task run already stores `run.json` and `messages.json`. The evaluator should inspect
these artifacts and return a bounded, structured feedback dictionary to GEPA:

- verified/failed/infrastructure-invalid outcome;
- final Lean diagnostic or exception class;
- tool-call sequence and counts;
- concise excerpts around failed `lean_execute` calls and the final submission;
- usage, duration, and the task-run artifact path;
- recurring failure labels such as declaration discovery, type mismatch, namespace/import,
  timeout, premature submission, or token exhaustion.

Store full transcripts on disk but limit optimizer feedback by a documented character budget.
Redact secrets and provider headers. Do not include reference proofs, expected solutions, held-out
case content, or feedback derived from evaluation-only runs. A human or coding agent may inspect
the full allowed training transcripts to diagnose the evaluator, but any manual intervention must
be logged as a new experiment or explicit parent/child phase.

## Design the evaluator before running GEPA

- Primary score: mean verified rate across repeated task-model runs.
- Return rich side information; a bare numeric score makes reflection nearly blind.
- Average stochastic repetitions inside one candidate/example evaluation and log each repetition.
- Gate any efficiency component on verification. Never reward short or cheap failed trajectories.
- Catch expected task failures and return feedback. Let corrupted artifacts, dataset-boundary
  violations, and other experiment-integrity failures abort the run.
- Establish a seed baseline and confirm the chosen discovery cases contain informative failures.
  If the seed is saturated, choose harder allowed training cases rather than touching eval data.

Start with one candidate and one or a few allowed cases. Confirm the full artifact tree and score
calculation before spending the real budget. Keep Formalizer task-model concurrency at 1 unless a
later measured VRAM/service constraint permits more.

## Use GEPA 0.1.4 deliberately

Construct `GEPAConfig` with an explicit `EngineConfig(max_metric_calls=..., run_dir=..., seed=...)`
and `ReflectionConfig(reflection_lm=...)`. A stopping condition is mandatory, but
`max_metric_calls` is checked between optimizer phases and a required full-selection evaluation can
take the observed count beyond the configured threshold. Record the actual count and leave budget
headroom. Treat the optimizer's selection score as optimistic; report the separately frozen
evaluation result as the final result.

After the run, inspect accepted and rejected proposals, not just the winner. Compare the seed and
winner on identical allowed repetitions, summarize failure-distribution changes, verify all hashes
and artifact counts, and state any missing cost coverage as unavailable rather than zero.

See [upstream.md](references/upstream.md) for the pinned upstream source and adaptation boundary.
