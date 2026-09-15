from __future__ import annotations

from typing import Any

from client.unified_llm_client import Conversation, UnifiedLLMClient


class VictimAdapter:
    """Expose the victim interface used by the defenses without changing the LLM client API."""

    def __init__(self, client: UnifiedLLMClient) -> None:
        self._client = client

    def generate(
        self,
        messages: list[dict[str, str]] | str,
        *,
        max_tokens: int | None = None,
    ) -> str:
        if isinstance(messages, str):
            response, _ = self._client.generate(
                user_input=messages,
                max_output_tokens=max_tokens,
            )
            return str(response or "")

        if not messages or messages[-1].get("role") != "user":
            raise ValueError("messages must end with role 'user'")

        prior = Conversation.from_list(messages[:-1])
        response, _ = self._client.generate(
            user_input=str(messages[-1].get("content") or ""),
            conversation=prior,
            system_prompt=prior.system_prompt,
            max_output_tokens=max_tokens,
        )
        return str(response or "")


class DefendedModel:
    """Select exactly one D-Judge defense and wrap a victim model with it."""

    SUPPORTED_METHODS = (
        "aligner",
        "backtranslation",
        "guard",
        "nbf",
        "paraphrase",
        "smoothllm",
        "tca",
    )

    def __init__(self, *, config: dict[str, Any] | None, victim: Any) -> None:
        self.config = dict(config or {})
        self.victim = victim
        self.defense_method = str(self.config.get("defense_method") or "").strip().lower()
        if self.defense_method in {"none", "null"}:
            self.defense_method = ""
        if self.defense_method in {"djudge", "proact"}:
            raise ValueError(
                f"Defense {self.defense_method!r} is intentionally not supported by MT-JailBench"
            )
        if self.defense_method and self.defense_method not in self.SUPPORTED_METHODS:
            supported = ", ".join(self.SUPPORTED_METHODS)
            raise ValueError(
                f"Unknown defense_method={self.defense_method!r}; expected one of: {supported}"
            )

        self._defense = self._build_defense() if self.defense_method else None

    @property
    def enabled(self) -> bool:
        return self._defense is not None

    def _build_defense(self) -> Any:
        method = self.defense_method
        if method == "aligner":
            from .aligner import AlignerDefense

            defense_class = AlignerDefense
        elif method == "backtranslation":
            from .backtranslation import BackTranslationDefense

            defense_class = BackTranslationDefense
        elif method == "guard":
            from .guard import GuardDefense

            defense_class = GuardDefense
        elif method == "nbf":
            from .nbf import NBFDefense

            defense_class = NBFDefense
        elif method == "paraphrase":
            from .paraphrase import ParaphraseDefense

            defense_class = ParaphraseDefense
        elif method == "smoothllm":
            from .smoothllm import SmoothLLMDefense

            defense_class = SmoothLLMDefense
        elif method == "tca":
            from .tca import TCADefense

            defense_class = TCADefense
        else:  # validated in __init__
            raise AssertionError(f"Unhandled defense method: {method}")

        method_config = self.config.get(method) or {}
        if not isinstance(method_config, dict):
            raise TypeError(f"defense.{method} must be a mapping")
        return defense_class(method_config, victim=self.victim)

    def process(
        self,
        messages: list[dict[str, str]],
        victim: Any | None = None,
        prev_defense_info: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        if self._defense is None:
            return self.process_none(messages, victim=victim)
        if victim is not None and victim is not self.victim:
            return self._defense.process(
                messages,
                victim=victim,
                prev_defense_info=prev_defense_info,
            )
        return self._defense.process(messages, prev_defense_info=prev_defense_info)

    def process_none(
        self,
        messages: list[dict[str, str]],
        victim: Any | None = None,
        prev_defense_info: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        del prev_defense_info
        active_victim = victim if victim is not None else self.victim
        return str(active_victim.generate(messages)), None
