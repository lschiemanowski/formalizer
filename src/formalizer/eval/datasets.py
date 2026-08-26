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
