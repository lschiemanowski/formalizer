# Formalizer prompt optimization

This directory contains the versioned code and configuration for optimizing the Formalizer agent
instructions without using the repository's evaluation cases. Generated artifacts belong under
`runs/prompt-optimization/`.

## Transcript feedback

`transcript_feedback.py` deterministically distills an allowed discovery transcript into GEPA 0.1.4
`side_info`. It retains full messages in the parent task run and exposes only:

- verified or model-failed outcome and the binary verified score;
- bounded search, Lean-check, final-submission, and usage evidence;
- normalized recurring-error fingerprints and a few representative excerpts;
- rule-based failure labels and an artifact path plus input hashes.

The default 6,000-character limit is measured using GEPA 0.1.4's actual Markdown rendering for one
reflection record, not compact JSON size. Selection mode is score-only and does not access its
messages argument. Infrastructure-invalid evaluations raise and must be retried or invalidate the
optimizer run; they never become model score zero.

Preview the frozen discovery replay without writing artifacts:

```console
uv run --group prompt-optimization python experiments/prompt_optimization/transcript_feedback.py \
  --output-dir runs/prompt-optimization/example-feedback-preview
```

Add `--execute` to create the fresh child artifact. This replay makes no model or network calls.
The output contains an immutable launch manifest, environment snapshot, append-only events and
feedback records, derived result, and terminal status. Existing output directories are never
overwritten.

Before a real optimization run, follow the project-local
`formalizer-gepa-prompt-optimization` skill and run its preflight with the exact approved cases.

## GEPA runner

`gepa_runner.py` connects the frozen panel, candidate prompt injection, Formalizer task runs,
transcript feedback, score-only selection, and GEPA 0.1.4. The proposed overnight configuration is
`gepa_overnight_v1.json`; `overnight_goal_v1.md` is an inactive goal draft and does not authorize or
start an experiment.

Inspect either phase without making model calls or creating an output directory:

```console
uv run --group prompt-optimization python experiments/prompt_optimization/gepa_runner.py \
  --phase canary \
  --output-dir runs/prompt-optimization/qwen3.6-gepa-canary-v1

uv run --group prompt-optimization python experiments/prompt_optimization/gepa_runner.py \
  --phase optimize \
  --output-dir runs/prompt-optimization/qwen3.6-gepa-overnight-v1
```

Execution is deliberately gated by both `--execute` and `--approve-model-calls`. Use those flags
only after the configuration and inactive goal have been approved. The canary must be run and
audited before the optimization phase.

Each fresh run directory records an immutable manifest and environment snapshot; exact candidates;
append-only, fsynced task-attempt and event records; full Formalizer task artifacts; exact
reflection prompts and responses; GEPA state; hashes, usage, timestamps, terminal classifications,
and final result/status. Infrastructure-invalid attempts are logged separately and never scored as
model failures. Discovery may receive bounded transcript evidence; selection never parses its
transcripts and returns only the verified score.
