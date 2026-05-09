"""Interactive outcome evaluator: pauses and prompts user for evaluation via console."""

from typing import override

from ...core import AttackContext, TurnEvaluation
from ...interfaces.attack.outcome_evaluator import OutcomeEvaluator


def _prompt_evaluation(context: AttackContext) -> TurnEvaluation:
    """Prompt user for turn evaluation with predefined score options."""

    print("\n--- Interactive Outcome Evaluator : evaluate ---")
    while True:
        choice = input("Score [1-5]: ").strip() or "3"
        try:
            idx = int(choice)
            if 1 <= idx <= 5:
                score = float(idx)
                break
        except ValueError:
            pass
        print("Invalid. Use 1-5.")

    return TurnEvaluation(score=score)


class InteractiveOutcomeEvaluator(OutcomeEvaluator):

    @override
    def evaluate(self, context: AttackContext) -> TurnEvaluation:
        return _prompt_evaluation(context)
