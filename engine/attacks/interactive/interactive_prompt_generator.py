"""Interactive prompt generator: pauses and prompts user for prompts via console."""

from typing import override

from ...core import AttackContext
from ...interfaces.attack.prompt_generator import PromptGenerator


def _prompt_actual(prompt_label: str, harmful_behavior: str) -> str:
    """Actual prompt: [1] harmful_behavior, [2] Custom."""
    print(f"\n--- Interactive Prompt Generator : {prompt_label} ---")
    preview = harmful_behavior[:60] + "..." if len(harmful_behavior) > 60 else harmful_behavior
    print(f"  [1] {preview}")
    print("  [2] Custom")

    while True:
        choice = input("Choice [1/2]: ").strip()
        if choice == "1":
            return harmful_behavior
        if choice == "2":
            text = input("Custom: ").strip()
            return text or harmful_behavior
        print("Invalid. Use 1 or 2.")


def _prompt_system() -> str | None:
    """System prompt: [1] None, [2] You're a helpful assistant, [3] Custom."""
    print("\n--- Interactive Prompt Generator : system_prompt ---")
    print("  [1] None")
    print("  [2] You're a helpful assistant")
    print("  [3] Custom")

    while True:
        choice = input("Choice [1/2/3]: ").strip()
        if choice == "1":
            return None
        if choice == "2":
            return "You're a helpful assistant."
        if choice == "3":
            text = input("Custom: ").strip()
            return text if text else None
        print("Invalid. Use 1, 2, or 3.")


_UNSET = object()


class InteractivePromptGenerator(PromptGenerator):

    def __init__(self) -> None:
        self._cached_system_prompt: str | None | object = _UNSET

    @override
    def next_prompt(self, context: AttackContext) -> str:
        harmful = context.harmful_behavior or "Please respond to my request."
        return _prompt_actual("next_prompt", harmful)

    @override
    def refine_prompt(self, context: AttackContext) -> str:
        harmful = context.harmful_behavior or "Please respond to my request."
        return _prompt_actual("refine_prompt", harmful)

    @override
    def system_prompt(self, context: AttackContext) -> str | None:
        if self._cached_system_prompt is not _UNSET:
            return self._cached_system_prompt
        self._cached_system_prompt = _prompt_system()
        return self._cached_system_prompt
