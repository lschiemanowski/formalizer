import asyncio
import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import pytest
import yaml
from pydantic_evals import Case

from formalizer.eval import EvalOutput, ProblemMetadata, load_formalizer_dataset
from formalizer.lean import DockerLeanChecker, InvalidLeanProblem
from formalizer.problem import FormalizationProblem
from formalizer.settings import SandboxSettings

DATASETS_PATH = Path(__file__).parents[2] / "datasets"
DATASET_PATH = DATASETS_PATH / "formalizer-eval-v1.yaml"
CALIBRATION_PATH = DATASETS_PATH / "formalizer-eval-v1-calibration.jsonl"
EXPECTED_CASE_NAMES = [
    "relational-equijoin-symmetry",
    "trie-compaction-correctness",
    "bidirectional-typechecker-correctness",
    "typed-language-substitution",
    "binary-search-tree-insertion",
    "canonical-run-compression",
    "group-word-reduction-correctness",
    "expression-normalization-fixedpoint",
    "stable-topk-selection-contract",
    "prefix-parser-encoder-adequacy",
    "interval-abstract-interpretation",
    "interval-union-insertion",
    "piece-table-split-preservation",
    "banker-queue-normalization",
    "regular-derivative-correctness",
    "lsm-compaction-correctness",
    "lru-touch-invariants",
    "quadtree-compression-correctness",
    "segment-tree-point-update",
    "mesh-boundary-toggle-laws",
    "state-transformer-laws",
    "ripple-carry-adder-correctness",
    "streaming-summary-merge-laws",
    "weighted-distribution-bind-laws",
    "symmetric-route-reversal",
    "dead-store-elimination-soundness",
    "spreadsheet-copy-reference-laws",
    "round-robin-schedule-balance",
    "arc-consistency-pruning",
    "retry-state-machine-accounting",
    "xor-parity-erasure-recovery",
    "call-auction-bookkeeping",
    "graph-coloring-vertex-renaming",
    "histogram-canonical-expansion",
    "filesystem-tree-size-mapping",
    "polygon-translation-area",
    "allocator-adjacent-coalescing",
    "posting-list-intersection-contract",
    "mapreduce-partition-conservation",
    "json-escape-decoder-roundtrip",
    "sparse-matrix-transpose-laws",
    "fir-convolution-polynomial-evaluation",
    "bounded-ring-buffer-rotation",
    "css-cascade-winner-laws",
    "feistel-network-roundtrip",
    "warehouse-batch-pick-accounting",
    "merkle-membership-proof-adequacy",
    "vector-clock-merge-algebra",
    "quarter-turn-pose-composition",
    "balanced-delimiter-depth-parser",
]


def _target_text(source: str) -> str:
    without_scaffolding = re.sub(
        r"import[^\n]*|namespace FormalizerProblem|end FormalizerProblem|"
        r"def Target\s*:\s*Prop\s*:=",
        " ",
        source,
    )
    return re.sub(r"\s+", " ", without_scaffolding).strip()


def test_formalizer_eval_v1_has_frozen_diverse_membership() -> None:
    dataset = load_formalizer_dataset(DATASET_PATH)

    assert dataset.name == "formalizer-eval-v1"
    assert all(case.name.startswith("formalizer-eval/") for case in dataset.cases)
    assert [case.name.rsplit("/", 1)[-1] for case in dataset.cases] == EXPECTED_CASE_NAMES
    assert len(dataset.cases) == 50

    metadata = [case.metadata for case in dataset.cases]
    assert all(item is not None for item in metadata)
    assert {item.split for item in metadata if item is not None} == {"validation"}
    # Domains are deliberately one-to-one with cases. This prevents a future frozen-dataset edit
    # from introducing a renamed or constant-changed variant of an existing proof family.
    domains = [item.domain for item in metadata if item is not None]
    assert len(domains) == len(set(domains)) == 50

    assert all(case.metadata.provenance.origin == "original" for case in dataset.cases)
    assert all(
        case.metadata.provenance.source
        == "Formalizer DeepSeek V4 Flash clean-room benchmark"
        for case in dataset.cases
    )
    assert all(case.metadata.provenance.license == "Apache-2.0" for case in dataset.cases)
    assert all(case.metadata.provenance.reference is None for case in dataset.cases)
    assert "putnam" not in DATASET_PATH.read_text(encoding="utf-8").lower()


def test_formalizer_eval_v1_has_no_solution_leakage_or_duplicate_targets() -> None:
    dataset = load_formalizer_dataset(DATASET_PATH)
    other_cases = [
        case
        for path in DATASETS_PATH.glob("*.yaml")
        if path != DATASET_PATH
        for case in load_formalizer_dataset(path).cases
    ]

    targets: dict[str, str] = {}
    for case in dataset.cases:
        source = case.inputs.source
        assert source.count("def Target : Prop :=") == 1
        assert "theorem " not in source
        assert "lemma " not in source
        assert "sorry" not in source
        assert "admit" not in source
        assert "axiom " not in source
        targets[case.name] = _target_text(source)

    assert len(set(targets.values())) == 50
    for name, target in targets.items():
        comparisons = ((other.name, _target_text(other.inputs.source)) for other in other_cases)
        for other_name, other_target in comparisons:
            assert target != other_target, f"{name} duplicates {other_name}"
            assert SequenceMatcher(None, target, other_target).ratio() < 0.80, (
                f"{name} is a near duplicate of {other_name}"
            )

    pairs = [
        (left_name, right_name, SequenceMatcher(None, left, right).ratio())
        for index, (left_name, left) in enumerate(targets.items())
        for right_name, right in list(targets.items())[index + 1 :]
    ]
    assert all(ratio < 0.80 for _, _, ratio in pairs), max(pairs, key=lambda item: item[2])


def test_formalizer_eval_v1_calibration_ledger_matches_lock_and_overlaps_effort() -> None:
    events = [json.loads(line) for line in CALIBRATION_PATH.read_text().splitlines()]
    assert [event["event"] for event in events[:3]] == ["calibration_run"] * 3
    lock = [event for event in events if event["event"] == "dataset_lock"][-1]

    document = yaml.safe_load(DATASET_PATH.read_text(encoding="utf-8"))
    payload = [
        {"inputs": case["inputs"], "metadata": case["metadata"]}
        for case in document["cases"]
    ]
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    assert lock["dataset"] == "formalizer-eval-v1"
    assert lock["dataset_sha256"] == hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
    assert lock["case_payload_sha256"] == hashlib.sha256(payload_bytes).hexdigest()
    assert lock["case_count"] == 50
    membership = lock["calibration_membership"]
    assert membership["task_failed"] >= 20
    assert membership["lean_verified"] + membership["task_failed"] == 50
    assert lock["composition"] == {
        "clean_room_original_cases": 50,
        "unique_domains": 50,
    }

    verified = lock["calibration_trajectory_mix"]["verified_duration_s"]
    failed = lock["calibration_trajectory_mix"]["failed_duration_s"]
    # Both outcomes occupy the middle of the observed effort range. This guards against curating
    # a trivially solved cluster plus an uniformly intractable cluster.
    assert verified["min"] < failed["min"] < verified["max"]
    assert failed["min"] < verified["max"] < failed["max"] * 1.10
    assert lock["calibration_trajectory_mix"]["verified_model_responses"]["max"] > 1
    assert lock["calibration_trajectory_mix"]["failed_model_responses"]["max"] > 1


@pytest.mark.integration
async def test_formalizer_eval_v1_contains_valid_lean_problems() -> None:
    dataset = load_formalizer_dataset(DATASET_PATH)
    # This benchmark is deliberately large; isolated Docker workspaces make eight-way validation
    # safe and keep the frozen-dataset check practical on the calibration host.
    semaphore = asyncio.Semaphore(8)

    async def validate_case(
        case: Case[FormalizationProblem, EvalOutput, ProblemMetadata],
    ) -> str | None:
        async with semaphore:
            checker = DockerLeanChecker(SandboxSettings(), problem_code=case.inputs.source)
            try:
                await checker.validate_problem()
            except InvalidLeanProblem as error:
                return f"Invalid trusted problem {case.name}: {error}"
            return None

    results = await asyncio.gather(*(validate_case(case) for case in dataset.cases))
    failures = [result for result in results if result is not None]
    assert not failures, "\n\n".join(failures)
