# formalizer

A very simple Lean formalizing agent

## Context-policy seam

`formalizer.run.formalize` and `run_formalizer` accept an optional `context_policy`. The policy is
called before each model request and returns a `ContextProjection`: the structured message history
that the model should see, an action label, and JSON metadata.

Supplying no policy preserves the ordinary full-history run and creates no context-specific
artifact. Supplying a policy creates `context-events.jsonl` in the task-run directory. Each
projection is recorded as append-only started/completed records containing hashes and exact
pre-policy and post-policy messages; policy failures are recorded before the run fails normally.
The selected policy name is also recorded in `run.json`.

The core seam deliberately does not define compression triggers, summarizer prompts, token
estimation, or experimental treatments. Those belong in a versioned experiment that implements
the `ContextPolicy` protocol.
