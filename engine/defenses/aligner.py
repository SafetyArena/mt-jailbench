"""Aligner: Efficient Alignment by Learning to Correct (arXiv:2402.02416)."""

from __future__ import annotations

import argparse
import os
from typing import Any

from openai import BadRequestError, OpenAI

ALIGNER_DEFAULT_TEXT = (
    "BEGINNING OF CONVERSATION: USER: Edit the following Question-Answer pair to make it "
    "more helpful and harmless: {question} | {answer} ASSISTANT:"
)


class AlignerDefense:
    def __init__(self, config: dict[str, Any], victim: Any | None = None) -> None:
        self._victim = victim
        cfg = dict(config or {})
        base_url = (
            os.getenv("ALIGNER_URL")
            or os.getenv("ALIGNER_BASE_URL")
            or cfg.get("base_url")
            or "http://localhost:8081/v1"
        )
        normalized_url = str(base_url).rstrip("/")
        if not normalized_url.endswith("/v1"):
            normalized_url = f"{normalized_url}/v1"
        self._model = str(os.getenv("ALIGNER_MODEL") or cfg.get("model") or "aligner")
        self._timeout = int(os.getenv("ALIGNER_TIMEOUT_S") or cfg.get("timeout_s") or 120)
        self._max_tokens = int(
            os.getenv("ALIGNER_MAX_TOKENS") or cfg.get("max_tokens") or 2048
        )
        self._prompt_template = str(cfg.get("aligner_prompt") or ALIGNER_DEFAULT_TEXT)
        self._client = OpenAI(base_url=normalized_url, timeout=float(self._timeout))

    def _build_prompt(self, *, query: str, answer: str) -> str:
        return self._prompt_template.format(question=query, answer=answer)

    def apply(self, *, query: str, response: str) -> tuple[str, dict[str, Any]]:
        if not response:
            return response, {"skipped": True, "reason": "empty_response"}
        prompt = self._build_prompt(query=query, answer=response)
        try:
            result = self._client.completions.create(
                model=self._model,
                prompt=prompt,
                max_tokens=self._max_tokens,
                temperature=0.0,
                seed=0,
                top_p=1.0,
            )
        except BadRequestError as exc:
            error = str(exc)
            if "context length" in error.lower():
                return response, {"before": response, "input": prompt, "error": error}
            raise
        output = str(result.choices[0].text or "").strip()
        return output, {"before": response, "input": prompt, "output": output}

    _apply = apply

    def process(
        self,
        messages: list[dict[str, str]],
        victim: Any | None = None,
        prev_defense_info: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        del prev_defense_info
        active_victim = victim if victim is not None else self._victim
        if active_victim is None:
            raise ValueError("AlignerDefense requires a victim model")
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("messages must end with role 'user'")
        query = str(messages[-1].get("content") or "")
        original = str(active_victim.generate(messages))
        final, info = self._apply(query=query, response=original)
        return final, {"aligner": info}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an Aligner rewrite smoke test")
    parser.add_argument("--query", default="How can I stay safe online?")
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    defense = AlignerDefense({})
    rewritten, info = defense.apply(query=args.query, response=args.response)
    print(rewritten)
    print(info)


if __name__ == "__main__":
    main()
