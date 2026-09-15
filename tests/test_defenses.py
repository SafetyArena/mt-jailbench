from __future__ import annotations

from pathlib import Path

import pytest
import torch

from client.unified_llm_client import Conversation, UnifiedLLMClient
from engine.core import AttackAttempt, AttackContext, AttackTurn
from engine.defenses.aligner import AlignerDefense
from engine.defenses.backtranslation import REJECT_RESPONSE, BackTranslationDefense
from engine.defenses.defended_model import VictimAdapter
from engine.defenses.guard import GuardDefense
from engine.defenses.nbf import NBFDefense
from engine.defenses.paraphrase import ParaphraseDefense
from engine.defenses.smoothllm import SmoothLLMDefense
from engine.defenses.tca import RiskCalculator, RiskConfig, TCADefense
from engine.jailbreak_engine import JailbreakEngine


class FakeVictim:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or ["victim response"])
        self.calls = []

    def generate(self, messages, *, max_tokens=None):
        self.calls.append((messages, max_tokens))
        return self.responses.pop(0)


class FakeUnifiedClient:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        conversation = Conversation.from_list(
            kwargs["conversation"].serialize()
            if isinstance(kwargs.get("conversation"), Conversation)
            else []
        )
        conversation.add_user_message(kwargs["user_input"])
        conversation.add_assistant_message("adapted")
        return "adapted", conversation


def build_without_init(cls, victim: FakeVictim):
    defense = object.__new__(cls)
    defense._victim = victim
    return defense


def test_victim_adapter_supports_messages_and_one_shot_text():
    client = FakeUnifiedClient()
    adapter = VictimAdapter(client)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
    ]
    assert adapter.generate(messages, max_tokens=42) == "adapted"
    assert client.calls[0]["user_input"] == "second"
    assert client.calls[0]["conversation"].history[-1]["content"] == "answer"
    assert client.calls[0]["system_prompt"] == "system"
    assert client.calls[0]["max_output_tokens"] == 42
    assert adapter.generate("one shot") == "adapted"
    assert client.calls[1]["user_input"] == "one shot"


def test_local_client_does_not_pass_api_key(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("client.unified_llm_client.OpenAI", FakeOpenAI)
    UnifiedLLMClient(
        model="local-model",
        provider="local",
        base_url="http://localhost:30000/v1",
    )
    assert captured == {"base_url": "http://localhost:30000/v1"}


def test_paraphrase_changes_only_victim_input():
    victim = FakeVictim()
    defense = build_without_init(ParaphraseDefense, victim)
    defense._paraphrase = lambda *, prompt: f"rewritten: {prompt}"
    output, info = defense.process([{"role": "user", "content": "raw"}])
    assert output == "victim response"
    assert victim.calls[0][0][-1]["content"] == "rewritten: raw"
    assert info == {"paraphrase": {"before": "raw", "after": "rewritten: raw"}}


def test_nbf_block_skips_victim():
    victim = FakeVictim()
    defense = build_without_init(NBFDefense, victim)
    defense._filter = lambda *, messages: ("blocked", {"filtered": True})
    output, info = defense.process([{"role": "user", "content": "raw"}])
    assert output == "blocked"
    assert info["nbf"]["filtered"] is True
    assert victim.calls == []


def test_smoothllm_calls_victim_for_every_perturbation():
    victim = FakeVictim(["safe", "safe", "I'm sorry"])
    defense = SmoothLLMDefense({"num_perturbations": 3}, victim=victim)
    defense._random_perturb = lambda prompt: f"{prompt}-perturbed"
    output, info = defense.process([{"role": "user", "content": "raw"}])
    assert output == "safe"
    assert len(victim.calls) == 3
    assert len(info["smoothllm"]["runs"]) == 3


def test_aligner_process_wraps_postprocessor():
    victim = FakeVictim()
    defense = build_without_init(AlignerDefense, victim)
    defense._apply = lambda **kwargs: ("aligned", kwargs)
    output, info = defense.process([{"role": "user", "content": "raw"}])
    assert output == "aligned"
    assert info["aligner"] == {"query": "raw", "response": "victim response"}


def test_guard_process_wraps_postprocessor():
    victim = FakeVictim()
    defense = build_without_init(GuardDefense, victim)
    defense.apply = lambda **kwargs: ("guarded", kwargs)
    output, info = defense.process([{"role": "user", "content": "raw"}])
    assert output == "guarded"
    assert info["guard"]["response_text"] == "victim response"


def test_backtranslation_requeries_victim_and_blocks_refusal():
    victim = FakeVictim(["original answer", "I'm sorry, I cannot help"])
    defense = build_without_init(BackTranslationDefense, victim)
    defense._max_tokens = 64
    defense._return_new_response_anyway = False
    defense._infer_question = lambda response: "inferred harmful request"
    output, info = defense.process([{"role": "user", "content": "raw"}])
    assert output == REJECT_RESPONSE
    assert len(victim.calls) == 2
    assert victim.calls[1] == ("inferred harmful request", 64)
    assert info["backtranslation"]["replaced"] is True


def test_tca_uses_previous_effective_risk():
    victim = FakeVictim()
    defense = build_without_init(TCADefense, victim)
    defense._risk = RiskConfig(warn_threshold=99, block_threshold=100)
    defense._calculator = RiskCalculator(defense._risk)
    defense.analyze = lambda previous, current: {
        "overall_progression_summary": {"final_risk_level": 2},
        "patterns": {},
    }
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second"},
    ]
    output, info = defense.process(
        messages,
        prev_defense_info={"tca": {"progressive_risk": 1.5}},
    )
    assert output == "victim response"
    assert info["tca"]["historical_risk"] == 1.5
    assert info["tca"]["progressive_risk"] == pytest.approx(1.45)


def test_engine_builds_canonical_defended_history_and_previous_state():
    previous_attempt = AttackAttempt(
        epoch=1,
        prompt="first",
        response="defended first",
        defense_info={"tca": {"progressive_risk": 1.25}},
    )
    previous_turn = AttackTurn(1, [previous_attempt], previous_attempt)
    conversation = Conversation(system_prompt="system")
    conversation.add_user_message("first")
    conversation.add_assistant_message("defended first")
    engine = object.__new__(JailbreakEngine)
    engine.context = AttackContext(
        harmful_behavior_id="id",
        harmful_behavior="behavior",
        max_turns=3,
        max_epochs=3,
        history=[previous_turn, AttackTurn(2)],
        full_history=[previous_attempt],
        current_turn=2,
        current_epoch=1,
        conversation=conversation,
        attack_outcome=None,
    )
    attempt = AttackAttempt(epoch=2, prompt="raw second", response="defended second")
    messages = engine._build_defense_messages(attempt)
    canonical = engine._canonical_conversation(attempt)
    assert messages[-1] == {"role": "user", "content": "raw second"}
    assert canonical.history[-2:] == [
        {"role": "user", "content": "raw second"},
        {"role": "assistant", "content": "defended second"},
    ]
    assert engine._previous_defense_info() == {"tca": {"progressive_risk": 1.25}}


def test_bundled_nbf_checkpoint_has_expected_state_dicts():
    checkpoint_path = (
        Path(__file__).parents[1] / "engine" / "defenses" / "models_best_nbf_released.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert {"ssm", "nbf"}.issubset(checkpoint)
