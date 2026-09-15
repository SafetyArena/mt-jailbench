from __future__ import annotations

import pytest

from engine.defenses import DefendedModel
from run_benchmark import _parse_defense


@pytest.mark.parametrize("raw", [None, {}, {"defense_method": ""}, {"defense_method": "none"}])
def test_parse_no_defense(raw):
    config = {"defense": raw}
    _parse_defense(config)
    if config["defense"] is not None:
        assert config["defense"]["defense_method"] == ""


@pytest.mark.parametrize("method", DefendedModel.SUPPORTED_METHODS)
def test_parse_supported_defense_normalizes_method(method):
    config = {"defense": {"defense_method": method.upper(), method: {}}}
    _parse_defense(config)
    assert config["defense"]["defense_method"] == method


@pytest.mark.parametrize("method", ["djudge", "proact", "unknown"])
def test_parse_rejects_unsupported_defense(method):
    with pytest.raises(ValueError):
        _parse_defense({"defense": {"defense_method": method}})


@pytest.mark.parametrize("legacy_key", ["defense_types", "defense_type", "defense_config"])
def test_parse_rejects_legacy_keys(legacy_key):
    with pytest.raises(ValueError, match="Legacy defense configuration"):
        _parse_defense({legacy_key: None})


def test_parse_rejects_non_mapping_method_config():
    with pytest.raises(TypeError, match="defense.guard"):
        _parse_defense({"defense": {"defense_method": "guard", "guard": "bad"}})


def test_defended_model_validates_without_importing_method(monkeypatch):
    monkeypatch.setattr(DefendedModel, "_build_defense", lambda self: object())
    for method in DefendedModel.SUPPORTED_METHODS:
        defended = DefendedModel(
            config={"defense_method": method, method: {}},
            victim=object(),
        )
        assert defended.enabled
        assert defended.defense_method == method
