import pytest
from pydantic import ValidationError

from formalizer.eval import ProblemMetadata, ProblemProvenance


def test_problem_metadata_records_domain_and_provenance() -> None:
    metadata = ProblemMetadata(
        difficulty="medium",
        split="validation",
        domain="number theory",
        provenance=ProblemProvenance(
            origin="adapted",
            source="Example benchmark",
            source_id="problem-42",
            license="Apache-2.0",
            reference="https://example.com/problems/42",
        ),
    )

    assert metadata.domain == "number theory"
    assert metadata.provenance.origin == "adapted"
    assert metadata.provenance.source_id == "problem-42"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("domain", "   "),
        ("source", ""),
        ("source_id", "   "),
        ("license", ""),
        ("reference", "   "),
    ],
)
def test_problem_metadata_rejects_blank_contract_fields(field: str, value: str) -> None:
    provenance: dict[str, object] = {
        "origin": "original",
        "source": "Formalizer",
        "source_id": "logic/true",
        "license": "Apache-2.0",
    }
    metadata: dict[str, object] = {
        "difficulty": "easy",
        "split": "train",
        "domain": "logic",
        "provenance": provenance,
    }
    if field == "domain":
        metadata[field] = value
    else:
        provenance[field] = value

    with pytest.raises(ValidationError):
        ProblemMetadata.model_validate(metadata)
