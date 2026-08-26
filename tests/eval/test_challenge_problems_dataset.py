from pathlib import Path

import pytest

from formalizer.eval import load_formalizer_dataset
from formalizer.lean import DockerLeanChecker
from formalizer.settings import SandboxSettings

CHALLENGE_PROBLEMS_DATASET_PATH = (
    Path(__file__).parents[2] / "datasets" / "challenge-problems-v1.yaml"
)


def test_challenge_problems_dataset_contains_curated_cases() -> None:
    dataset = load_formalizer_dataset(CHALLENGE_PROBLEMS_DATASET_PATH)

    assert dataset.name == "challenge-problems-v1"
    assert [case.name for case in dataset.cases] == [
        "challenge-problems/polya-enumeration",
        "challenge-problems/majorization",
    ]

    expected_metadata = [
        ("medium", "validation", "combinatorics", "problem_10"),
        ("medium", "validation", "linear-algebra", "problem_8"),
    ]
    for case, expected in zip(dataset.cases, expected_metadata, strict=True):
        assert case.metadata is not None
        metadata = case.metadata
        assert (
            metadata.difficulty,
            metadata.split,
            metadata.domain,
            metadata.provenance.source_id,
        ) == expected
        assert metadata.provenance.origin == "adapted"
        assert metadata.provenance.source == "Challenge problems collection"
        assert metadata.provenance.license == "Apache-2.0"
        assert metadata.provenance.reference is None


def test_challenge_problem_contains_only_trusted_context_and_target() -> None:
    dataset = load_formalizer_dataset(CHALLENGE_PROBLEMS_DATASET_PATH)

    for case in dataset.cases:
        source = case.inputs.source
        assert source.count("def Target : Prop :=") == 1
        assert "theorem " not in source
        assert "sorry" not in source
        assert "admit" not in source
        assert "axiom " not in source

    polya_source = dataset.cases[0].inputs.source
    assert "fixedColoringCount" in polya_source
    assert "coloringOrbitCount" in polya_source
    assert "sum_card_fixedBy_eq_card_orbits_mul_card_group" not in polya_source

    majorization_source = dataset.cases[1].inputs.source
    assert "InPermutationConvexHull" in majorization_source
    assert "InDoublyStochasticImage" in majorization_source
    assert "mulVecRightLinear" not in majorization_source
    assert "image_convexHull" not in majorization_source


@pytest.mark.integration
async def test_challenge_problems_dataset_contains_valid_lean_problems() -> None:
    dataset = load_formalizer_dataset(CHALLENGE_PROBLEMS_DATASET_PATH)

    for case in dataset.cases:
        checker = DockerLeanChecker(
            SandboxSettings(),
            problem_code=case.inputs.source,
        )
        await checker.validate_problem()
