from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from formalizer.agent import Submission


class ProblemMetadata(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    difficulty: Literal["easy", "medium", "hard"]
    split: Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class EvalOutput:
    run_id: str
    submission: Submission
