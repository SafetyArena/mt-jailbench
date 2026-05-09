from typing import override

from ...core import AttackContext, TurnEvaluation
from ...interfaces.attack.outcome_evaluator import OutcomeEvaluator


class TemplateOutcomeEvaluator(OutcomeEvaluator):

    def __init__(self, context: AttackContext, config: dict) -> None:
        super().__init__()

    @override
    def evaluate(self, context: AttackContext) -> TurnEvaluation:
        return TurnEvaluation(score=3)