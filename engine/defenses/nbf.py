"""Neural Barrier Function defense from arXiv:2503.00187."""

from __future__ import annotations

import argparse
import os
import threading
from typing import Any

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

SORRY_RESPONSE = "I'm sorry, but I cannot provide assistance or information on that topic."
_LOAD_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()
_CACHE: tuple[str, Any, Any, Any, torch.device] | None = None


class NeuralStateSpaceModel(nn.Module):
    def __init__(self, state_dim: int, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.state_transition = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.observation_model = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, previous_state: torch.Tensor, input_value: torch.Tensor) -> tuple:
        state_input = torch.cat([previous_state, input_value], dim=-1)
        state = self.state_transition(state_input)
        observation = self.observation_model(torch.cat([state, input_value], dim=-1))
        return state, observation


class NeuralBarrierFunction(nn.Module):
    def __init__(self, state_dim: int, input_dim: int, hidden_dim: int, class_num: int = 5) -> None:
        super().__init__()
        self.nbf = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, class_num),
        )

    def forward(self, state: torch.Tensor, input_value: torch.Tensor) -> torch.Tensor:
        return self.nbf(torch.cat([state, input_value], dim=-1))


def load_models(model_path: str) -> tuple[Any, Any, Any, torch.device]:
    global _CACHE
    if _CACHE is not None and _CACHE[0] == model_path:
        return _CACHE[1:]
    with _LOAD_LOCK:
        if _CACHE is not None and _CACHE[0] == model_path:
            return _CACHE[1:]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = SentenceTransformer("all-mpnet-base-v2")
        state_model = NeuralStateSpaceModel(768, 768, 768, 512)
        barrier = NeuralBarrierFunction(768, 768, 32)
        checkpoint = torch.load(model_path, map_location=device)
        state_model.load_state_dict(checkpoint["ssm"])
        barrier.load_state_dict(checkpoint["nbf"])
        state_model.to(device).eval()
        barrier.to(device).eval()
        _CACHE = (model_path, encoder, state_model, barrier, device)
        return encoder, state_model, barrier, device


def calculate_score(
    prior_user_turns: list[str],
    current_query: str,
    encoder: Any,
    state_model: NeuralStateSpaceModel,
    barrier: NeuralBarrierFunction,
    device: torch.device,
) -> float:
    state_dim = state_model.state_transition[-1].out_features
    state = torch.zeros(1, state_dim, device=device)
    with torch.inference_mode():
        for text in prior_user_turns:
            encoded = encoder.encode(
                text,
                convert_to_tensor=True,
                show_progress_bar=False,
            ).unsqueeze(0).to(device)
            state, _ = state_model(state, encoded)
        encoded = encoder.encode(
            current_query,
            convert_to_tensor=True,
            show_progress_bar=False,
        ).unsqueeze(0).to(device)
        logits = barrier(state, encoded)
        probabilities = torch.softmax(logits, dim=-1)
        score = probabilities[:, -1] - torch.max(probabilities[:, :-1], dim=1).values
    return float(score[0].item())


def default_model_path() -> str:
    candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_best_nbf_released.pth")
    return candidate if os.path.isfile(candidate) else ""


class NBFDefense:
    def __init__(self, config: dict[str, Any], victim: Any | None = None) -> None:
        self._victim = victim
        cfg = dict(config or {})
        self._model_path = str(
            os.getenv("NBF_MODEL_PATH") or cfg.get("model_path") or default_model_path()
        ).strip()
        self._threshold = float(os.getenv("NBF_THRESHOLD") or cfg.get("threshold") or 0)
        self._sorry_response = str(
            cfg.get("sorry_response")
            or os.getenv("NBF_SORRY_RESPONSE")
            or SORRY_RESPONSE
        )

    def filter(self, messages: list[dict[str, Any]]) -> tuple[str | None, dict[str, Any] | None]:
        if not self._model_path:
            return None, None
        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(
                f"NBF model path not found: {self._model_path}. Set NBF_MODEL_PATH or nbf.model_path."
            )
        if not messages or messages[-1].get("role") != "user":
            return None, None
        current_query = str(messages[-1].get("content") or "")
        prior_user_turns = [
            str(message.get("content") or "")
            for message in messages[:-1]
            if message.get("role") == "user"
        ]
        encoder, state_model, barrier, device = load_models(self._model_path)
        with _INFER_LOCK:
            score = calculate_score(
                prior_user_turns,
                current_query,
                encoder,
                state_model,
                barrier,
                device,
            )
        info = {
            "prior_user_turns": prior_user_turns,
            "input_summary_query": current_query,
            "safety_index": score,
            "filtered": score >= -self._threshold,
        }
        if info["filtered"]:
            info["sorry_response"] = self._sorry_response
            return self._sorry_response, info
        return None, info

    _filter = filter

    def process(
        self,
        messages: list[dict[str, str]],
        victim: Any | None = None,
        prev_defense_info: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        del prev_defense_info
        active_victim = victim if victim is not None else self._victim
        if active_victim is None:
            raise ValueError("NBFDefense requires a victim model")
        refusal, info = self._filter(messages=messages)
        if refusal is not None:
            return refusal, {"nbf": info}
        return str(active_victim.generate(messages)), {"nbf": info}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or run the NBF defense")
    parser.add_argument("prompt", nargs="?", default="How do I bake a cake?")
    parser.add_argument("--model-path", default=default_model_path())
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(f"model_path={args.model_path or '(missing)'} threshold={args.threshold}")
    if not args.dry_run:
        defense = NBFDefense({"model_path": args.model_path, "threshold": args.threshold})
        _, info = defense.filter([{"role": "user", "content": args.prompt}])
        print(info)


if __name__ == "__main__":
    main()
