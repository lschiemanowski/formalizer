from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from formalizer.eval.models import EvalOutput, ProblemMetadata
from formalizer.problem import FormalizationProblem


@dataclass
class LeanVerified(Evaluator[FormalizationProblem, EvalOutput, ProblemMetadata]):
    def evaluate(
        self,
        ctx: EvaluatorContext[FormalizationProblem, EvalOutput, ProblemMetadata],
    ) -> bool:
        return ctx.output.submission.check.accepted
