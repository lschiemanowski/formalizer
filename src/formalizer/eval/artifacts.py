from pathlib import Path

from pydantic import TypeAdapter
from pydantic_evals.reporting import EvaluationReport

from formalizer.eval.models import EvalOutput, ProblemMetadata
from formalizer.problem import FormalizationProblem

type FormalizerEvaluationReport = EvaluationReport[
    FormalizationProblem,
    EvalOutput,
    ProblemMetadata,
]

_REPORT_ADAPTER = TypeAdapter(FormalizerEvaluationReport)


def write_evaluation_report(path: Path, report: FormalizerEvaluationReport) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_bytes(_REPORT_ADAPTER.dump_json(report, indent=2) + b"\n")
    temporary_path.replace(path)


def read_evaluation_report(path: Path) -> FormalizerEvaluationReport:
    return _REPORT_ADAPTER.validate_json(path.read_bytes())
