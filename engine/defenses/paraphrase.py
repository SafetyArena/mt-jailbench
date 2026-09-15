"""Input paraphrasing defense from arXiv:2309.00614."""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Any

from openai import OpenAI

DEFAULT_USER_TEMPLATE = 'paraphrase the following paragraph: \n"{prompt}"\n\n'
DEFAULT_MODEL = "gpt-3.5-turbo-0125"


class ParaphraseDefense:
    def __init__(self, config: dict[str, Any], victim: Any | None = None) -> None:
        self._victim = victim
        cfg = dict(config or {})
        nested = dict(cfg.get("paraphrase") or {})
        nested.update({key: value for key, value in cfg.items() if key != "paraphrase"})
        self._model = str(os.getenv("PARAPHRASE_MODEL") or nested.get("model") or DEFAULT_MODEL)
        self._base_url = os.getenv("PARAPHRASE_BASE_URL") or nested.get("base_url")
        self._max_tokens = int(
            os.getenv("PARAPHRASE_MAX_TOKENS")
            or nested.get("max_tokens")
            or nested.get("max_output_tokens")
            or 512
        )
        self._template = str(
            os.getenv("PARAPHRASE_USER_TEMPLATE")
            or nested.get("user_template")
            or DEFAULT_USER_TEMPLATE
        )
        if self._base_url:
            base_url = str(self._base_url).rstrip("/")
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
            self._client = OpenAI(base_url=base_url)
        else:
            self._client = OpenAI()

    def paraphrase(self, prompt: str) -> str:
        original = str(prompt or "")
        if not original.strip():
            return original
        try:
            result = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": self._template.format(prompt=original)}],
                max_tokens=self._max_tokens,
                temperature=0.0,
                seed=0,
                top_p=1.0,
            )
            cleaned = str(result.choices[0].message.content or "").strip().strip("]").strip("[").strip()
            if "\n" in cleaned:
                warnings.warn(
                    "A newline was found in paraphrase output; the first line is removed.",
                    stacklevel=2,
                )
                cleaned = "\n".join(cleaned.split("\n")[1:]).strip().strip("]").strip("[").strip()
            return cleaned or original
        except Exception:
            return original

    def _paraphrase(self, *, prompt: str) -> str:
        return self.paraphrase(prompt)

    def process(
        self,
        messages: list[dict[str, str]],
        victim: Any | None = None,
        prev_defense_info: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        del prev_defense_info
        active_victim = victim if victim is not None else self._victim
        if active_victim is None:
            raise ValueError("ParaphraseDefense requires a victim model")
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("messages must end with role 'user'")
        before = str(messages[-1].get("content") or "")
        after = self._paraphrase(prompt=before)
        victim_messages = list(messages)
        victim_messages[-1] = {"role": "user", "content": after}
        response = str(active_victim.generate(victim_messages))
        return response, {"paraphrase": {"before": before, "after": after}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an input paraphrase smoke test")
    parser.add_argument("prompt")
    args = parser.parse_args()
    print(ParaphraseDefense({}).paraphrase(args.prompt))


if __name__ == "__main__":
    main()
