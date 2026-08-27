from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from pydantic_ai import ModelSettings, UsageLimits

type NonBlankString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class SandboxSettings(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    docker_image: str = "formalizer:latest"
    cpus: float = Field(default=2, gt=0)
    memory_mib: int = Field(default=10_240, ge=256)
    pids_limit: int = Field(default=64, ge=1)
    workspace_mib: int = Field(default=256, ge=16)
    tmp_mib: int = Field(default=256, ge=16)
    lean_timeout_s: float = Field(default=30, gt=0)
    search_timeout_s: float = Field(default=30, gt=0)
    search_max_results: int = Field(default=100, ge=1, le=100)


class RunSettings(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    runs_dir: Path = Path("runs")
    run_timeout_s: float = Field(default=1_800, gt=0)
    usage_limits: UsageLimits = Field(default_factory=UsageLimits)


class Settings(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
    )

    model_name: NonBlankString
    model_settings: ModelSettings = Field(default_factory=ModelSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    run: RunSettings = Field(default_factory=RunSettings)
