from typing import override

from ...core import AttackContext
from ...interfaces.defense.text_processor import TextProcessor
from .guard import GuardDefense


class GuardTextProcessor(TextProcessor):
    def __init__(self, context: AttackContext, config: dict) -> None:
        super().__init__()
        self._guard = GuardDefense(config)

    @override
    def process_prompt(self, context: AttackContext) -> str | None:
        return None

    @override
    def process_response(self, context: AttackContext) -> str | None:
        attempt = context.get_current_attempt()
        response = attempt.response
        if not response:
            return None

        if context.conversation:
            messages = list(context.conversation.serialize(include_system=False))
        else:
            messages = [{"role": "user", "content": attempt.prompt or ""}]

        if not any(m.get("role") == "assistant" for m in messages):
            messages.append({"role": "assistant", "content": response})

        final, info = self._guard.apply(messages=messages)
        attempt.defense_info["guard"] = info

        if final == response:
            return None
        return final
