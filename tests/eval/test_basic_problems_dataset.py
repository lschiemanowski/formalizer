import asyncio
from collections import Counter
from pathlib import Path

import pytest
from pydantic_evals import Case

from formalizer.eval import EvalOutput, ProblemMetadata, load_formalizer_dataset
from formalizer.lean import DockerLeanChecker, InvalidLeanProblem
from formalizer.problem import FormalizationProblem
from formalizer.settings import SandboxSettings

BASIC_PROBLEMS_DATASET_PATH = Path(__file__).parents[2] / "datasets" / "basic-problems-v1.yaml"

EXPECTED_SOURCE_IDS = {
    "train": {
        *(f"problem_{number}" for number in range(9, 78)),
    }
    - {
        "problem_12",
        "problem_15",
        "problem_17",
        "problem_34",
        "problem_35",
        "problem_36",
        "problem_38",
        "problem_39",
        "problem_40",
        "problem_42",
        "problem_43",
        "problem_47",
        "problem_48",
        "problem_50",
        "problem_54",
        "problem_64",
        "problem_67",
    },
    "validation": {
        "problem_1",
        "problem_5",
        "problem_6",
        "problem_7",
        "problem_8",
        "problem_15",
        "problem_36",
        "problem_47",
        "problem_50",
        "problem_54",
        "problem_64",
        "problem_67",
    },
    "test": {
        "problem_2",
        "problem_3",
        "problem_4",
        "problem_12",
        "problem_17",
        "problem_34",
        "problem_35",
        "problem_38",
        "problem_39",
        "problem_40",
        "problem_43",
        "problem_48",
    },
}


def test_basic_problems_dataset_has_curated_partition_and_provenance() -> None:
    dataset = load_formalizer_dataset(BASIC_PROBLEMS_DATASET_PATH)

    assert dataset.name == "formurmel-basic-problems-v1"
    assert len(dataset.cases) == 76
    assert len({case.name for case in dataset.cases}) == len(dataset.cases)
    assert len({case.inputs.source for case in dataset.cases}) == len(dataset.cases)

    source_ids_by_split: dict[str, set[str]] = {split: set() for split in EXPECTED_SOURCE_IDS}
    difficulty_counts: Counter[str] = Counter()

    for case in dataset.cases:
        assert case.name is not None
        assert case.metadata is not None
        metadata = case.metadata
        provenance = metadata.provenance

        source_ids_by_split[metadata.split].add(provenance.source_id)
        difficulty_counts[metadata.difficulty] += 1

        assert case.name == f"formurmel/{provenance.source_id.replace('_', '-')}"
        assert provenance.origin == "adapted"
        assert provenance.source == "Formurmel basic problems dataset"
        assert provenance.license == "Apache-2.0"
        assert provenance.reference is not None
        assert "602d13f97a98d4e30956fc4ee3eb6dadd99dedbf" in provenance.reference

    assert source_ids_by_split == EXPECTED_SOURCE_IDS
    assert difficulty_counts == {"easy": 17, "medium": 59}


def test_basic_problems_dataset_contains_targets_without_source_proofs() -> None:
    dataset = load_formalizer_dataset(BASIC_PROBLEMS_DATASET_PATH)

    for case in dataset.cases:
        source = case.inputs.source

        assert source.startswith("import Mathlib\n\nnamespace FormalizerProblem\n\n")
        assert source.count("def Target : Prop :=") == 1
        assert "theorem " not in source
        assert "sorry" not in source
        assert "admit" not in source
        assert "axiom " not in source
        assert source.endswith("\n\nend FormalizerProblem\n")


@pytest.mark.integration
async def test_basic_problems_dataset_contains_valid_lean_problems() -> None:
    dataset = load_formalizer_dataset(BASIC_PROBLEMS_DATASET_PATH)
    semaphore = asyncio.Semaphore(4)

    async def validate_case(
        case: Case[FormalizationProblem, EvalOutput, ProblemMetadata],
    ) -> str | None:
        async with semaphore:
            checker = DockerLeanChecker(
                SandboxSettings(),
                problem_code=case.inputs.source,
            )
            try:
                await checker.validate_problem()
            except InvalidLeanProblem as error:
                return f"Invalid trusted problem {case.name}: {error}"
            return None

    results = await asyncio.gather(*(validate_case(case) for case in dataset.cases))
    failures = [result for result in results if result is not None]

    assert not failures, "\n\n".join(failures)
