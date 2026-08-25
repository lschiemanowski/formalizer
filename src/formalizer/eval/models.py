from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class Problem(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    id: str
    prompt: str


class TrialResult(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    problem_id: str
    run_id: UUID
    status: Literal["verified", "failed"]
