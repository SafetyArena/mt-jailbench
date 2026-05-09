"""
UnifiedLLMClient will determine parameters in the following order:
1. load default params for the model (get_model_default_params)
2. use params provided in .generate() to override any default params
3. remove model-unsafe params (remove_model_unsafe_params)
"""

def get_model_default_params(model_name: str, deterministic_mode: bool) -> dict:
    """
    Define default params based on model names.

    Use deterministic model by default (best-effort determinism, not guaranteed depending
    on specific model and provider)
    """
    model_name = model_name.lower()
    params = {}

    # For best-effort determinism:
    if deterministic_mode:
        params["seed"] = 123
        params["temperature"] = 0
        params["top_p"] = 1
    else:
        params["temperature"] = 0.7
        params["top_p"] = 0.9

    # reduce reasoning for expensive models
    if model_name in ("claude-sonnet-4.5", "gemini-3-pro", "gpt-5"):
        params["reasoning_effort"] = "low"

    return params


def remove_model_unsafe_params(model_name: str, params: dict) -> None:
    """
    Remove params that are unsafe for the model, modifying the dict in place.
    """
    # GPT-5 family doesn't support temperature or top_p
    if model_name.startswith("gpt-5") and not model_name.startswith("gpt-5."):
        params.pop("temperature", None)
        params.pop("top_p", None)
