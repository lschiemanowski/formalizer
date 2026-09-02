# GEPA 0.1.4 API used by this experiment

This reference is intentionally pinned to installed package version `0.1.4`. Do not copy examples
from GEPA `main` without checking them against the installed objects.

## Entry point

```python
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    TrackingConfig,
    optimize_anything,
)

config = GEPAConfig(
    engine=EngineConfig(
        run_dir="runs/prompt-optimization/example/gepa",
        seed=0,
        max_metric_calls=200,
        parallel=False,
        raise_on_exception=True,
    ),
    reflection=ReflectionConfig(
        reflection_lm="openai/my-reflection-model",
        reflection_minibatch_size=3,
    ),
    tracking=TrackingConfig(),
)

result = optimize_anything(
    seed_candidate=seed_prompt,
    evaluator=evaluate,
    dataset=discovery_examples,
    valset=selection_examples,
    objective="Improve the Formalizer Lean agent instructions.",
    background="Preserve the tool and verification contract.",
    config=config,
)
```

The evaluator is called as `evaluate(candidate, example)` and may return either a score or
`(score, side_info)`. Always return structured side information for this experiment. A
`batch_evaluator` is also supported, but it must return one result for each `(candidate, example)`
pair in the same order.

`dataset=None, valset=None` is single-task mode. `dataset=<list>, valset=None` reuses the discovery
set for selection. Prompt generalization experiments should supply disjoint explicit lists.

## Configuration facts

- `GEPAConfig` contains `engine`, `reflection`, `tracking`, and optional merge/refiner/callback
  settings.
- `EngineConfig` includes `run_dir`, `seed`, `max_metric_calls`, `max_candidate_proposals`,
  `max_reflection_cost`, `parallel`, `max_workers`, `raise_on_exception`, `cache_evaluation`, and
  `capture_stdio`.
- `ReflectionConfig` includes `reflection_lm`, `reflection_lm_kwargs`,
  `reflection_minibatch_size`, `perfect_score`, and `skip_perfect_score`.
- `TrackingConfig` supports W&B and MLflow, but local experiment artifacts remain mandatory even
  when an external tracker is enabled.
- At least one stopping condition is mandatory. Normally set `max_metric_calls` explicitly. It is a
  phase-boundary stopper rather than a strict request quota: a required full `valset` evaluation may
  make `result.total_metric_calls` exceed it. Enforce any hard provider budget separately and
  record the observed count.
- If `reflection_lm` is a string, GEPA resolves it through LiteLLM. It may instead be a callable
  implementing `__call__(prompt) -> str`, which is useful for a self-hosted endpoint wrapper.
- A custom `reflection_prompt_template` cannot be combined with `objective` or `background`.

## Result handling

`GEPAResult` provides `candidates`, `parents`, `val_aggregate_scores`, `val_subscores`,
`discovery_eval_counts`, `total_metric_calls`, `run_dir`, `seed`, `best_idx`, `best_candidate`, and
`to_dict()`.

Version 0.1.4 does **not** expose `result.best_score`. Compute it as:

```python
best_score = result.val_aggregate_scores[result.best_idx]
```

It also has no `test_set` argument and no `OptimizeAnythingConfig`. Those belong to newer upstream
development. Serialize `result.to_dict()` and separately add derived values such as `best_score`.

The `run_dir` enables GEPA's file logger and state artifacts, but it does not replace the
experiment-level manifest, candidate, evaluation, and terminal-status logs required by this skill.
