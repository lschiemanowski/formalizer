from pydantic import BaseModel, ConfigDict


class FormalizationProblem(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    source: str
