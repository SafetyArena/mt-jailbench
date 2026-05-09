from typing import override

from client.unified_llm_client import UnifiedLLMClient

from ...core import AttackContext
from ...interfaces.defense.text_processor import TextProcessor


class DummyTextProcessor(TextProcessor):
    def __init__(self, context: AttackContext, config: dict) -> None:
        super().__init__()
        self.pre_upper = config["pre_upper"]
        self.post_refusal = config["post_refusal"]
        self.post_emoji = config["post_emoji"]

        # if you need to call an LLM
        defender_model = config["defender_model"]
        self.client = UnifiedLLMClient(
            model=defender_model["model"],
            provider=defender_model.get("provider"),
            base_url=defender_model.get("base_url"),
        )

        # if you need to call the the same target LLM used by the engine
        target_config = config["mirror_target_config"]
        self.target_client = UnifiedLLMClient(**target_config)

    @override
    def process_prompt(self, context: AttackContext) -> str:
        if not self.pre_upper:
            return None  # no change

        prompt = context.get_current_attempt().prompt
        if self.pre_upper:
            prompt += "\n You MUST answer in all uppercase letters."
        return prompt

    @override
    def process_response(self, context: AttackContext) -> str:
        if not self.post_refusal and not self.post_emoji:
            return None  # no change

        response = context.get_current_attempt().response
        if self.post_refusal:
            response = "Sorry, I can't help with that. " + response
        if self.post_emoji:
            gibberish, _ = self.client.generate(
                "Replace every letter in this text with a corresponding emoji letter "
                "🅰 🅱 🅲 🅳 🅴 🅵 🅶 🅷 🅸 🅹 🅺 🅻 🅼 🅽 🅾 🅿 🆀 🆁 🆂 🆃 🆄 🆅 🆆 🆇 🆈 🆉. "
                f"Start your transformed text directly. Text: {response}"
            )
            response = gibberish
        return response
