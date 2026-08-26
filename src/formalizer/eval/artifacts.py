from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from pydantic_evals.reporting import EvaluationReport

from formalizer.eval.models import EvalOutput, ProblemMetadata
from formalizer.problem import FormalizationProblem

type FormalizerEvaluationReport = EvaluationReport[
    FormalizationProblem,
    EvalOutput,
    ProblemMetadata,
]


class LegacyProblemMetadata(BaseModel):
    """Problem metadata written before the dataset provenance contract."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    difficulty: Literal["easy", "medium", "hard"]
    split: Literal["train", "validation", "test"]


type LegacyFormalizerEvaluationReport = EvaluationReport[
    FormalizationProblem,
    EvalOutput,
    LegacyProblemMetadata,
]

_REPORT_ADAPTER = TypeAdapter(FormalizerEvaluationReport)
_LEGACY_REPORT_ADAPTER = TypeAdapter(LegacyFormalizerEvaluationReport)


def write_evaluation_report(path: Path, report: FormalizerEvaluationReport) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_bytes(_REPORT_ADAPTER.dump_json(report, indent=2) + b"\n")
    temporary_path.replace(path)


def read_evaluation_report(
    path: Path,
) -> FormalizerEvaluationReport | LegacyFormalizerEvaluationReport:
    data = path.read_bytes()
    try:
        return _REPORT_ADAPTER.validate_json(data)
    except ValidationError as current_error:
        try:
            return _LEGACY_REPORT_ADAPTER.validate_json(data)
        except ValidationError:
            raise current_error from None
