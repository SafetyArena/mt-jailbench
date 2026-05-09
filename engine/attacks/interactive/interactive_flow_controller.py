from typing import override

from ...core import ActionType, AttackContext, NextAction
from ...interfaces.attack.flow_controller import FlowController


def _prompt_action(context: AttackContext) -> NextAction:
    """Prompt user to choose next action with predefined options."""
    options = [
        ("c", "Continue", ActionType.CONTINUE, None),
        ("r", "Retry", ActionType.RETRY, None),
        ("s", "End success", ActionType.END_SUCCESS, None),
        ("f", "End failure", ActionType.END_FAILURE, None),
        ("j", "Jump to turn", ActionType.JUMP_TO, "turn"),
    ]

    print("\n--- Interactive Flow Controller : next_action ---")
    for key, label, _, _ in options:
        if key == "j":
            payload_hint = " (then enter turn number)"
        elif key == "c":
            payload_hint = " (optionally: 'c <number>')"
        else:
            payload_hint = ""
        print(f"  [{key}] {label}{payload_hint}")
    print("  ---")

    while True:
        raw = input("Choice [c/r/s/f/j]: ").strip().lower() or "c"
        parts = raw.split(None, 1)
        choice = parts[0]
        if choice == "c":
            payload = int(parts[1]) if len(parts) > 1 else None
            return NextAction(ActionType.CONTINUE, payload=payload)
        if choice == "j":
            try:
                turn_str = input("  Jump to turn: ").strip()
                turn = int(turn_str)
                return NextAction(ActionType.JUMP_TO, payload=turn)
            except ValueError:
                print("  Invalid: enter a number")
            continue
        for key, _, action_type, _ in options:
            if key == choice and action_type != ActionType.JUMP_TO:
                return NextAction(action_type)
        print("Invalid choice. Use c, r, s, f, or j.")


class InteractiveFlowController(FlowController):

    @override
    def next_action(self, context: AttackContext) -> NextAction:
        return _prompt_action(context)
