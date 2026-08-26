from pathlib import Path

import pytest

from formalizer.eval import FormalizerDataset, ProblemMetadata
from formalizer.lean import DockerLeanChecker
from formalizer.problem import FormalizationProblem
from formalizer.settings import SandboxSettings

SMOKE_DATASET_PATH = Path(__file__).parents[2] / "datasets" / "smoke.yaml"


def test_smoke_dataset_loads_typed_cases() -> None:
    dataset = FormalizerDataset.from_file(SMOKE_DATASET_PATH)

    assert dataset.name == "formalizer-smoke-v1"
    assert [case.name for case in dataset.cases] == [
        "arithmetic/one-plus-one",
        "arithmetic/add-zero",
        "logic/and-commutes",
    ]
    assert all(isinstance(case.inputs, FormalizationProblem) for case in dataset.cases)
    assert all(
        case.metadata == ProblemMetadata(difficulty="easy", split="test") for case in dataset.cases
    )


@pytest.mark.integration
async def test_smoke_dataset_contains_valid_lean_problems() -> None:
    dataset = FormalizerDataset.from_file(SMOKE_DATASET_PATH)

    for case in dataset.cases:
        checker = DockerLeanChecker(
            SandboxSettings(),
            problem_code=case.inputs.source,
        )
        await checker.validate_problem()
