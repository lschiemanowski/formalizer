from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

from formalizer.agent import Submission

type NonBlankString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class ProblemProvenance(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    origin: Literal["original", "adapted", "verbatim"]
    source: NonBlankString
    source_id: NonBlankString
    license: NonBlankString
    reference: NonBlankString | None = None


class ProblemMetadata(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    difficulty: Literal["easy", "medium", "hard"]
    split: Literal["train", "validation", "test"]
    domain: NonBlankString
    provenance: ProblemProvenance


@dataclass(frozen=True, slots=True)
class EvalOutput:
    run_id: str
    submission: Submission
