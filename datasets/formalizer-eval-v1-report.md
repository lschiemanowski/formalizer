# Formalizer eval v1 construction report

Status: **frozen**. Candidate 04 has 50 valid selected outcomes: 29 Lean-verified successes and 21
genuine task failures. Thirty infrastructure-invalid attempts are retained for audit but replaced
only by identical retries and do not contribute to the selected aggregate.

## Purpose and interpretation

This is an adaptively calibrated discrimination instrument for models near DeepSeek V4 Flash. It
is not an untouched or model-independent estimate of formalization capability. Candidate
membership was deliberately informed by DeepSeek V4 Flash pilot outcomes, so comparisons to that
model must disclose the adaptive selection.

The deliverable contains 50 substantively different Lean formalization tasks in 50 distinct
domains. All are project-original, clean-room authored modules under Apache-2.0 provenance. No
statement was copied, translated, paraphrased, selected from, or inspired by PutnamBench. An
earlier PutnamBench-derived construction is quarantined as discarded audit history and contributes
neither tasks nor calibration evidence to this deliverable.

## Frozen dataset

- Dataset: `datasets/formalizer-eval-v1.yaml`
- Dataset identity: `formalizer-eval-v1`; case namespace: `formalizer-eval/`
- Current SHA-256: `e1cbd9b6e5fea689321805393e93a9782116696fef128f0da6e1ba1c6460531a`
- Calibration-artifact SHA-256 before identity-only rename:
  `ed9eb09f889295789da6ec5ea77c99023935c7505678ea4e7209d6a54cc4d653`
- Identity-independent case-payload SHA-256:
  `6765d39105f6659286fdaf650fc45c27865f67a7bae802c4ec65d4227743b40c`
- Cases: 50
- Distinct metadata domains: 50
- Trusted-target validation: 50 accepted, 0 rejected, in the pinned Lean/Docker environment
- Source-contract tests: frozen membership and cleanroom/duplicate checks both pass
- Target canonical outcome: at least 20 genuine failures; verified successes comprise the remainder
- Accepted canonical outcome: 29 verified successes and 21 genuine failures

Each task is a complete module defining `FormalizerProblem.Target`. The frozen contract rejects
`sorry`, `admit`, local axioms, theorem or lemma solution leakage, exact target duplication, and a
target similarity ratio of 0.80 or greater against selected and existing dataset cases. Failure
candidates are retained only when their target has independent accepted Lean proof evidence; false
curator targets are quarantined and never counted as model failures.

## Fixed evaluation protocol

All DeepSeek trials use `/home/lothar/.local/bin/formalizer-deepseek-eval` with:

```text
model: openrouter:deepseek/deepseek-v4-flash-0731
downstream provider: deepseek only
provider fallbacks: disabled
required parameters: enforced
sampling: provider defaults
tool choice: auto
max tokens per response: 16384
request limit per case: 1000
output-token limit per case: 100000
total-token limit per case: 2000000
infrastructure retries: 2
whole-run timeout: 30 minutes
max concurrency: 8
repeat: 1
```

Valid stochastic trials are never rerun to obtain a preferred label. Authentication, payment,
quota, rate-limit, overload, connection, provider-response, Docker, Lean, Loogle, harness, and
artifact failures are infrastructure-invalid. Only those invalid attempts may be retried, under
identical settings.

## Canonical calibration history

| Run | Dataset SHA | Verified | Genuine failures | Infrastructure-invalid | Disposition |
| --- | --- | ---: | ---: | ---: | --- |
| cleanroom canonical 01 | `db51297b...` | 35 | 14 | 1 | Rejected: invalid case and wrong boundary |
| cleanroom canonical 02 | `515a6ad0...` | 31 | 16 | 3 | Rejected: invalid cases and wrong boundary |
| cleanroom canonical 03 | `4c7b9799...` | 34 | 16 | 0 | Rejected: clean but wrong boundary |
| cleanroom canonical 04 base | `ed9eb09f...` | 10 | 12 | 28 | Retain valid outcomes; retry invalid cases only |
| canonical 04 retry 01 | `ed9eb09f...` | 19 | 7 | 2 | Retain valid outcomes; retry two provider-invalid cases |
| canonical 04 retry 02 | `ed9eb09f...` | 0 | 2 | 0 | Accepted; completes the clean selected aggregate |

Canonical 03 used 428 model responses, 11,091,452 input tokens, 1,375,449 output tokens,
1,117,465 reasoning tokens, and $2.139362824 provider-reported cost with 428/428 response
coverage. Its successes ranged from short proofs to 835-second trajectories; genuine failures
included immediate response-limit failures, multi-step failures, and two 100,000-output-token
exhaustions. Thus the observed outcome classes overlap in effort rather than forming a trivial
easy/hard split.

Canonical 04's first-valid selection rule uses the base-run outcome when valid, otherwise retry 01,
then retry 02. This yields 29 verified and 21 genuine failures with no unresolved invalid cases.
Failure types are 18 per-response model token/protocol exhaustions, two rejected Lean submissions,
and one cumulative total-token exhaustion. Verified trajectories span 61.88–780.888 seconds and
3–24 model responses; failed trajectories span 109.96–880.165 seconds and 1–40 responses. These
ranges materially overlap and rule out a completely bimodal easy-versus-impossible construction.

The selected outcomes used 440 model responses, 10,398,990 input tokens, 1,419,491 output tokens,
and 1,215,493 reasoning tokens. Provider-reported cost is $2.185248648 with complete 440/440
selected-response coverage. Including all retained invalid attempts, the canonical chain used 515
responses and records $2.445390424, with 513/515 provider-cost coverage; its all-attempt benchmark
cost basis is therefore conservatively unavailable.

## Cost accounting

The append-only ledger records $14.856266768 in provider-reported cleanroom construction
and evaluation cost across screens, infrastructure retries, and canonical runs. Some earlier runs
have incomplete per-response cost coverage and therefore retain an unavailable benchmark-cost
basis even though their recorded provider cost is preserved. The accepted selected-outcome cost
and the larger all-attempt canonical-chain cost are reported separately above.

## Failure taxonomy

Admissible genuine failures are:

- a rejected final Lean submission;
- per-response model token exhaustion;
- cumulative output- or total-token exhaustion under the fixed budgets;
- unchanged whole-run timeout; or
- a clearly evidenced coherent model/tool-protocol failure.

Infrastructure-invalid outcomes remain separate. Canonical 04's 28 HTTP 403 quota errors and two
DeepSeek HTTP 400 assistant-message errors are retained but are not failures for benchmark
calibration or scoring.

## Dataset rename

The frozen task set was renamed from the model-specific working identity
`deepseek-v4-flash-discrimination-v1` to the neutral public identity `formalizer-eval-v1`.
Only the dataset name, case-name namespace, and public artifact names changed. Case inputs,
metadata, order, and selected calibration outcomes are identical. Historical candidate and run
paths retain their original names so the immutable evidence referenced by the ledger remains
addressable.

## Artifacts and audit trail

- Dataset: `datasets/formalizer-eval-v1.yaml`
- Frozen contract: `tests/eval/test_formalizer_eval_dataset.py`
- Append-only ledger: `datasets/formalizer-eval-v1-calibration.jsonl`
- Candidate/reference workspace: `evaluations/deepseek-v4-flash-discrimination-v1/`
- Exact invalid-case retry: `evaluations/deepseek-v4-flash-discrimination-v1/retry_canonical_04_invalid.sh`
- Canonical-04 base report: `runs/evals/deepseek-v4-flash-discrimination-v1-cleanroom-canonical-04/report.json`
- Canonical-04 retry-01 report: `runs/evals/deepseek-v4-flash-discrimination-v1-cleanroom-canonical-04-infra-retry-01/report.json`
- Canonical-04 retry-02 report: `runs/evals/deepseek-v4-flash-discrimination-v1-cleanroom-canonical-04-infra-retry-02/report.json`

## Limitations

This benchmark is intentionally model-targeted and adaptively selected, so its DeepSeek V4 Flash
failure rate is not an unbiased capability estimate. A single stochastic selected outcome per case
does not measure pass probability or outcome variance. The cleanroom policy reduces recognizable
benchmark contamination but cannot establish absence from all model training corpora. Provider
errors and incomplete cost metadata also mean recorded construction spend is a lower bound whenever
coverage is incomplete.
