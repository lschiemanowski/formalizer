import asyncio
from collections import Counter
from pathlib import Path

import pytest
from pydantic_evals import Case

from formalizer.eval import EvalOutput, ProblemMetadata, load_formalizer_dataset
from formalizer.lean import DockerLeanChecker, InvalidLeanProblem
from formalizer.problem import FormalizationProblem
from formalizer.settings import SandboxSettings

DATASETS_PATH = Path(__file__).parents[2] / "datasets"
BASELINE_EVAL_DATASET_PATH = DATASETS_PATH / "baseline-eval-v1.yaml"
SOURCE_DATASET_PATHS = [
    DATASETS_PATH / "basic-problems-v1.yaml",
    DATASETS_PATH / "challenge-problems-v1.yaml",
]

EXPECTED_CASE_NAMES = [
    "basic-problems/problem-1",
    "basic-problems/problem-5",
    "basic-problems/problem-6",
    "basic-problems/problem-7",
    "basic-problems/problem-8",
    "basic-problems/problem-15",
    "basic-problems/problem-36",
    "basic-problems/problem-47",
    "basic-problems/problem-50",
    "basic-problems/problem-54",
    "basic-problems/problem-64",
    "basic-problems/problem-67",
    "challenge-problems/polya-enumeration",
    "challenge-problems/majorization",
]


def test_baseline_eval_v1_has_frozen_case_membership() -> None:
    dataset = load_formalizer_dataset(BASELINE_EVAL_DATASET_PATH)

    assert dataset.name == "baseline-eval-v1"
    assert [case.name for case in dataset.cases] == EXPECTED_CASE_NAMES
    difficulty_counts: Counter[str] = Counter()
    splits: set[str] = set()
    for case in dataset.cases:
        assert case.metadata is not None
        difficulty_counts[case.metadata.difficulty] += 1
        splits.add(case.metadata.split)

    assert difficulty_counts == {
        "easy": 3,
        "medium": 11,
    }
    assert splits == {"validation"}


def test_baseline_eval_v1_copies_cases_exactly_from_source_datasets() -> None:
    dataset = load_formalizer_dataset(BASELINE_EVAL_DATASET_PATH)
    source_cases = {
        case.name: case
        for path in SOURCE_DATASET_PATHS
        for case in load_formalizer_dataset(path).cases
    }

    for case in dataset.cases:
        source_case = source_cases[case.name]
        assert case.inputs == source_case.inputs
        assert case.metadata == source_case.metadata


@pytest.mark.integration
async def test_baseline_eval_v1_contains_valid_lean_problems() -> None:
    dataset = load_formalizer_dataset(BASELINE_EVAL_DATASET_PATH)
    semaphore = asyncio.Semaphore(2)

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
