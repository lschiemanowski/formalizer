# Formalizer prompt-optimization experiment contract

## Dataset boundary

The default optimization source is `datasets/basic-problems-v1.yaml`. Every selected case must have
`metadata.split == "train"`. Maintain two explicit, disjoint allowlists:

- `discovery_cases`: evaluated during candidate mutation;
- `selection_cases`: passed to GEPA as `valset` and used to select a candidate.

Both lists come from the dataset's original `train` split. Never infer membership from position or
copy a case into a new file without retaining its identity and provenance. Persist the source YAML
SHA-256 and the ordered names in `manifest.json`.

These are evaluation-only by default and must not be read by the optimizer or used for manual
candidate edits:

- all `validation` and `test` entries in `basic-problems-v1.yaml`;
- `datasets/baseline-eval-v1.yaml`;
- `datasets/formalizer-eval-v1.yaml`;
- `datasets/challenge-problems-v1.yaml`.

The project evaluation is a separate post-selection invocation. Its output directory must not sit
inside the GEPA run directory, and its result must not be added to reflection feedback.

## Required manifest fields

Use a versioned JSON schema. At minimum record:

```text
schema_version, run_id, created_at, status
git.commit, git.is_dirty, git.diff_sha256, git.file_sha256s
dependencies.gepa, dependencies.lock_sha256
dataset.path, dataset.sha256, discovery_cases, selection_cases
seed_prompt.path, seed_prompt.sha256
evaluator.path, evaluator.sha256, evaluator.feedback_char_limit
task_model.provider, task_model.model, task_model.endpoint, task_model.settings
reflection_model.provider, reflection_model.model, reflection_model.endpoint, reflection_model.settings
gepa.engine, gepa.reflection, gepa.tracking
repetitions, concurrency, timeouts, usage_limits
```

Never persist API keys or authorization headers. Record whether credentials were present, not their
values.

## Per-evaluation record

Append a record before and after each candidate/example/repetition so interrupted work is visible.
Include:

```text
evaluation_id, candidate_sha256, case_name, repetition, seed
started_at, finished_at, terminal_status
score, verified, infrastructure_invalid, failure_type
task_run_path, task_run_id, usage, duration_s
feedback_sha256, error
```

Use `terminal_status` values such as `running`, `verified`, `model_failed`,
`infrastructure_invalid`, `cancelled`, and `integrity_error`. Only `verified` and `model_failed`
belong in the model-behavior denominator by default.

## Result interpretation

GEPA's selection set is reused across candidates, so the maximum selection score is subject to
winner's curse. Preserve the entire candidate pool and per-instance scores. Report:

- seed and selected candidate hashes;
- selection score and sample size;
- proposal and metric-call counts;
- accepted/rejected proposal counts when available;
- verified/model-failed/infrastructure-invalid counts;
- usage and cost with explicit coverage;
- the separately run frozen-evaluation score, clearly labeled post-selection.

Never claim generalization from discovery or selection scores alone.
