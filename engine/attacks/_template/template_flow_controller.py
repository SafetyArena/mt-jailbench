from typing import override

from ...core import ActionType, AttackContext, NextAction
from ...interfaces.attack.flow_controller import FlowController


class TemplateFlowController(FlowController):

    def __init__(self, context: AttackContext, config: dict) -> None:
        super().__init__()

    @override
    def next_action(self, context: AttackContext) -> NextAction:
        # Assume score ranges from 0 to 5
        latest_score = context.history[-1].evaluation.score
        if latest_score == 5:
            return NextAction(ActionType.END_SUCCESS)
        elif latest_score >= 3:
            return NextAction(ActionType.CONTINUE)
        elif latest_score >= 1:
            return NextAction(ActionType.RETRY)
        else:
            return NextAction(ActionType.END_FAILURE)
