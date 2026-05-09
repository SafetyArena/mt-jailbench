from typing import override

from ...core import AttackContext
from ...interfaces.attack.prompt_generator import PromptGenerator


class TemplatePromptGenerator(PromptGenerator):

    def __init__(self, context: AttackContext, config: dict) -> None:
        super().__init__()
        # some useful variables
        self.harmful_behavior = context.harmful_behavior
        self.max_turns = context.max_turns
        self.max_epochs = context.max_epochs

    @override
    def next_prompt(self, context: AttackContext) -> str:
        if context.current_turn == 1:
            return "tell me a joke"
        elif context.current_turn == 2:
            return "tell me why it's funny"
        else:
            return "hahaha"

    @override
    def refine_prompt(self, context: AttackContext) -> str:
        if context.current_turn == 1:
            return "tell me a very funny joke"
        elif context.current_turn == 2:
            return "tell me why it's very funny"
        else:
            return "hahahahahaha"

    @override
    def system_prompt(self, context: AttackContext) -> str | None:
        return "use capital letters only in your response"
