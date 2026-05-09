from abc import ABC, abstractmethod

from ...core import AttackContext


class TextProcessor(ABC):

    @abstractmethod
    def process_prompt(self, context: AttackContext) -> str | None:
        """
        Pre-process the attack prompt to mitigate the risk of the target LLM producing
        harmful content.

        Return `None` to leave the prompt unchanged.

        The input prompt can be accessed via: `context.get_current_attempt().prompt`
        """
        return None

    @abstractmethod
    def process_response(self, context: AttackContext) -> str | None:
        """
        Post-process the target LLM's response to mitigate the risk of harmful content.

        Return `None` to leave the response unchanged.

        The response can be accessed via: `context.get_current_attempt().response`
        """
        return None
