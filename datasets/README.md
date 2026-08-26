# Formalizer datasets

Formalizer datasets are version-controlled Pydantic Evals YAML files. Each case supplies one
immutable `FormalizerProblem.lean` module and the metadata needed to curate, audit, and stratify
the benchmark. A model submission is successful only when `LeanVerified` accepts its proof of
`FormalizerProblem.Target`.

`smoke.yaml` tests the evaluation pipeline. It is not a model benchmark.

## Case contract

```yaml
name: example-dataset-v1
cases:
  - name: number-theory/example-problem
    inputs:
      source: |
        import Mathlib

        namespace FormalizerProblem

        def Target : Prop := ...

        end FormalizerProblem
    metadata:
      difficulty: medium
      split: validation
      domain: number-theory
      provenance:
        origin: adapted
        source: Source dataset or publication
        source_id: problem-42
        license: Apache-2.0
        reference: https://example.com/problem-42
```

Case names must be unique and stable within a dataset. The source must be a complete Lean module
that defines exactly the proposition the model is expected to prove as
`FormalizerProblem.Target`.

### Difficulty

`difficulty` is a curator's provisional description of the proof task:

- `easy` indicates a routine proof with a short, direct argument.
- `medium` indicates a multi-step proof or meaningful Mathlib discovery.
- `hard` indicates substantial proof synthesis, abstraction, or library navigation.

Difficulty is not a score assigned by one model. Pilot results may motivate a documented
recalibration before a dataset version is frozen. Report empirical solve rates separately.

### Split

- `train` cases may be inspected and used directly during prompt or policy optimization.
- `validation` cases may be used to select prompts, context strategies, and other configurations.
- `test` cases are reserved for final comparisons after the experimental configuration is fixed.

The test split is held out procedurally even when its source is public. Do not move a case between
splits to improve reported results. Avoid exact and near-duplicate theorems across splits.

### Domain

Use a concise lower-case label and reuse an existing label whenever possible. Initial labels
include `arithmetic`, `algebra`, `logic`, `number-theory`, `combinatorics`, `data-structures`,
`analysis`, and `topology`. Add a new label only when none of the existing labels describes the
problem well.

### Provenance

Every case records:

- `origin`: `original`, `adapted`, or `verbatim`.
- `source`: the dataset, publication, or project from which the problem originates.
- `source_id`: the stable identifier used by that source.
- `license`: the source license, preferably an SPDX identifier.
- `reference`: an optional URL, DOI, or bibliographic reference.

Use `original` only when the problem was authored for this project. Use `adapted` when a sourced
statement was translated, reformulated, or materially changed. Use `verbatim` only when the Lean
problem was copied without a semantic change. Original Formalizer cases use `Apache-2.0`.

## Curation requirements

Before inclusion, each case should satisfy all of the following:

1. The trusted problem module compiles in the pinned Docker environment.
2. `FormalizerProblem.Target` exists and has type `Prop`.
3. The problem module contains no `sorry`, `admit`, local axioms, or answer leakage.
4. The statement and supporting definitions preserve the intended mathematical problem.
5. Provenance and license information have been reviewed.
6. The case is not an exact or near duplicate of a case in another split.

Automated validation complements review. It does not establish mathematical fidelity,
provenance, licensing, or the absence of subtle leakage by itself.

## Versioning and experiments

Dataset names end in a version such as `-v1`. Once a dataset has been used for a reported
experiment, do not change that file in place. Create a new version and document the changes.

Formalizer evaluation reports record the dataset name and SHA-256 hash. Run published experiments
from a clean Git commit and use a new output directory for every invocation.

Core proof benchmarks should fit comfortably within the model context window. Long-context and
context-compression experiments belong in a separate context-stress dataset so that context
limitations do not confound baseline proof competence.

Validate the current smoke dataset without making model requests:

```bash
uv run pytest tests/eval/test_dataset.py
uv run pytest -m integration tests/eval/test_dataset.py
```
