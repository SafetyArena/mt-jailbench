# Unified LLM Client User Guide

`UnifiedLLMClient` provides a unified interface for invoking LLMs across multiple providers.

## Initialization

`UnifiedLLMClient` takes a model name (required), a provider (recommended), and a base URL (optional) as parameters. Models and providers must first be declared in [`supported_models.yaml`](../client/supported_models.yaml) before you can use them. The declaration file maps each model name to its available providers, along with the provider-specific model IDs used by those providers. Locally deployed models (i.e., `provider="local"`) do not need to be declared but usually require the base URL to be set.

By default, the client uses best-effort deterministic mode. To disable it, initialize the client with `UnifiedLLMClient(deterministic_mode=False)`.

## Invocation Params

[`model_params.py`](../client/model_params.py) defines the default parameters used when invoking LLMs. You can configure defaults for individual models or model families. At runtime, parameters are resolved in three steps:

1. Load default parameters using `get_model_default_params()` from [`model_params.py`](../client/model_params.py).
2. Add or override parameters with the values passed to `client.generate()`.
3. Remove unsafe parameters using `remove_model_unsafe_params()` from [`model_params.py`](../client/model_params.py).

## Conversation Tracking

`Conversation` is a data class for tracking multi-turn conversations. `UnifiedLLMClient` always returns a `Conversation` object along with the model response, and it can also accept an existing `Conversation` object as input. See the sample usage below for examples. 

## Supported Providers

The client currently supports the following API providers:

- OpenAI
- Azure OpenAI
- AWS Bedrock
- Google Gemini
- Anthropic
- OpenRouter
- Local deployments, such as vLLM and SGLang

## API Pricing Reference

- OpenAI: https://platform.openai.com/docs/pricing
- Azure: https://azure.microsoft.com/en-us/pricing/details/ai-foundry-models/model-router/
- AWS Bedrock: https://aws.amazon.com/bedrock/pricing/
- Google Gemini: https://ai.google.dev/gemini-api/docs/pricing
- Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
- OpenRouter: https://openrouter.ai/pricing

## API Key Setup

Configure the corresponding API keys as environment variables:

- OpenAI: `OPENAI_API_KEY`
- Azure OpenAI: `OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`
- AWS Bedrock: `AWS_DEFAULT_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- Google Gemini: `GEMINI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- OpenRouter: `OPENROUTER_API_KEY`

## Sample Usage

```python
# Online provider
client = UnifiedLLMClient(model="gpt-5-mini", provider="openai")

# Local deployment
client = UnifiedLLMClient(
    model="model-name",
    provider="local",
    base_url="http://localhost:30000/v1/",
)

# Single-turn generation without conversation tracking
output, _ = client.generate("Tell me a joke")

# Multi-turn generation with conversation tracking
output_1, conv = client.generate("Tell me a joke")
output_2, conv = client.generate("Now explain why it's funny", conversation=conv)