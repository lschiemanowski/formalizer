from pathlib import Path

from pydantic_evals import Case

from formalizer.agent import Submission
from formalizer.eval import EvalOutput, FormalizerDataset, LeanVerified, ProblemMetadata
from formalizer.eval.artifacts import read_evaluation_report, write_evaluation_report
from formalizer.lean import LeanResult
from formalizer.problem import FormalizationProblem


async def test_evaluation_report_round_trips_as_json(tmp_path: Path) -> None:
    problem = FormalizationProblem(
        source="""\
namespace FormalizerProblem
def Target : Prop := True
end FormalizerProblem
"""
    )
    output = EvalOutput(
        run_id="7e42281f-548d-4a0e-bb50-df63dd3a5b62",
        submission=Submission(
            code="proof",
            check=LeanResult(stdout="", stderr="", exit_code=0, duration_s=0.1),
        ),
    )
    dataset = FormalizerDataset(
        name="formalizer-smoke-v1",
        cases=[
            Case(
                name="logic/true",
                inputs=problem,
                metadata=ProblemMetadata(difficulty="easy", split="test"),
            )
        ],
        evaluators=[LeanVerified()],
    )

    async def task(received_problem: FormalizationProblem) -> EvalOutput:
        assert received_problem == problem
        return output

    report = await dataset.evaluate(
        task,
        name="smoke-deepseek",
        progress=False,
        metadata={"model_name": "test:model", "dataset_sha256": "abc123"},
    )
    report.trace_id = "0123456789abcdef0123456789abcdef"
    report.span_id = "0123456789abcdef"
    report_path = tmp_path / "report.json"

    write_evaluation_report(report_path, report)
    loaded = read_evaluation_report(report_path)

    assert loaded == report
    assert report_path.read_bytes().endswith(b"\n")


async def test_evaluation_report_with_task_failure_round_trips(tmp_path: Path) -> None:
    problem = FormalizationProblem(source="def FormalizerProblem.Target : Prop := True\n")
    dataset = FormalizerDataset(
        name="failure-test",
        cases=[Case(name="failed", inputs=problem)],
    )

    async def task(received_problem: FormalizationProblem) -> EvalOutput:
        raise RuntimeError("provider unavailable")

    report = await dataset.evaluate(task, progress=False)
    report_path = tmp_path / "report.json"

    write_evaluation_report(report_path, report)
    loaded = read_evaluation_report(report_path)

    assert loaded == report
    assert len(loaded.failures) == 1
    assert loaded.failures[0].error_message == "RuntimeError: provider unavailable"
