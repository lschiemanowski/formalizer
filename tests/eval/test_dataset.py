from pathlib import Path

import pytest

from formalizer.eval import InvalidFormalizerDataset, load_formalizer_dataset
from formalizer.lean import DockerLeanChecker
from formalizer.problem import FormalizationProblem
from formalizer.settings import SandboxSettings

SMOKE_DATASET_PATH = Path(__file__).parents[2] / "datasets" / "smoke.yaml"


def test_smoke_dataset_loads_typed_cases() -> None:
    dataset = load_formalizer_dataset(SMOKE_DATASET_PATH)

    assert dataset.name == "formalizer-smoke-v1"
    assert [case.name for case in dataset.cases] == [
        "arithmetic/one-plus-one",
        "arithmetic/add-zero",
        "logic/and-commutes",
    ]
    assert all(isinstance(case.inputs, FormalizationProblem) for case in dataset.cases)
    metadata = [case.metadata for case in dataset.cases if case.metadata is not None]
    assert len(metadata) == len(dataset.cases)
    assert [item.domain for item in metadata] == [
        "arithmetic",
        "arithmetic",
        "logic",
    ]
    assert all(item.difficulty == "easy" for item in metadata)
    assert all(item.split == "test" for item in metadata)
    assert [item.provenance.source_id for item in metadata] == [case.name for case in dataset.cases]
    assert all(item.provenance.origin == "original" for item in metadata)
    assert all(item.provenance.license == "Apache-2.0" for item in metadata)


def test_formalizer_dataset_rejects_a_case_without_metadata(tmp_path: Path) -> None:
    dataset_path = tmp_path / "missing-metadata.yaml"
    dataset_path.write_text(
        """\
name: missing-metadata-v1
cases:
  - name: logic/true
    inputs:
      source: |
        namespace FormalizerProblem
        def Target : Prop := True
        end FormalizerProblem
""",
        encoding="utf-8",
    )

    with pytest.raises(InvalidFormalizerDataset, match="logic/true"):
        load_formalizer_dataset(dataset_path)


def test_formalizer_dataset_rejects_an_unnamed_case(tmp_path: Path) -> None:
    dataset_path = tmp_path / "unnamed.yaml"
    dataset_path.write_text(
        """\
name: unnamed-v1
cases:
  - inputs:
      source: |
        namespace FormalizerProblem
        def Target : Prop := True
        end FormalizerProblem
    metadata:
      difficulty: easy
      split: train
      domain: logic
      provenance:
        origin: original
        source: Test fixture
        source_id: logic/true
        license: Apache-2.0
""",
        encoding="utf-8",
    )

    with pytest.raises(InvalidFormalizerDataset, match="name"):
        load_formalizer_dataset(dataset_path)


@pytest.mark.integration
async def test_smoke_dataset_contains_valid_lean_problems() -> None:
    dataset = load_formalizer_dataset(SMOKE_DATASET_PATH)

    for case in dataset.cases:
        checker = DockerLeanChecker(
            SandboxSettings(),
            problem_code=case.inputs.source,
        )
        await checker.validate_problem()
