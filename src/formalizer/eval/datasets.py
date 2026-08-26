from collections.abc import Collection
from pathlib import Path

from formalizer.eval.runner import FormalizerDataset


class InvalidFormalizerDataset(ValueError):
    """A dataset does not satisfy the Formalizer case contract."""


def load_formalizer_dataset(path: Path) -> FormalizerDataset:
    dataset = FormalizerDataset.from_file(path)
    unnamed_positions = [
        str(index) for index, case in enumerate(dataset.cases) if case.name is None
    ]
    if unnamed_positions:
        positions = ", ".join(unnamed_positions)
        raise InvalidFormalizerDataset(f"Cases missing required names at positions: {positions}")

    missing_metadata = [
        case.name for case in dataset.cases if case.name is not None and case.metadata is None
    ]
    if missing_metadata:
        names = ", ".join(missing_metadata)
        raise InvalidFormalizerDataset(f"Cases missing required metadata: {names}")
    return dataset


def select_formalizer_dataset(
    dataset: FormalizerDataset,
    *,
    splits: Collection[str] = (),
    difficulties: Collection[str] = (),
    case_names: Collection[str] = (),
) -> FormalizerDataset:
    if not dataset.cases:
        raise InvalidFormalizerDataset("No cases match the requested selection")
    if not splits and not difficulties and not case_names:
        return dataset

    available_names = {case.name for case in dataset.cases if case.name is not None}
    unknown_names = set(case_names) - available_names
    if unknown_names:
        names = ", ".join(sorted(unknown_names))
        raise InvalidFormalizerDataset(f"Unknown case names: {names}")

    selected_cases = []
    for case in dataset.cases:
        metadata = case.metadata
        if metadata is None:
            raise InvalidFormalizerDataset(f"Case missing required metadata: {case.name}")
        if splits and metadata.split not in splits:
            continue
        if difficulties and metadata.difficulty not in difficulties:
            continue
        if case_names and case.name not in case_names:
            continue
        selected_cases.append(case)

    if not selected_cases:
        raise InvalidFormalizerDataset("No cases match the requested selection")
    if len(selected_cases) == len(dataset.cases):
        return dataset

    return FormalizerDataset(
        name=dataset.name,
        cases=selected_cases,
        evaluators=dataset.evaluators,
        report_evaluators=dataset.report_evaluators,
    )
