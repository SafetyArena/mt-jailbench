from __future__ import annotations

import copy
import json
import logging
import os
import re
from pathlib import Path

import boto3
import yaml
from anthropic import Anthropic
from botocore.config import Config
from google import genai
from openai import OpenAI

from .model_params import get_model_default_params, remove_model_unsafe_params

logger = logging.getLogger(__name__)


class Conversation:
    """
    Tracks the conversation between user and assistant. System prompt (if any)
    is stored separately from the conversation history. Conversation can be
    serialized in below format:
    ```
    [
        {"role": "system", "content": "..."}
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]
    ```
    """

    def __init__(self, system_prompt: str | None = None) -> None:
        self.history = []
        self.system_prompt = system_prompt

    def add_user_message(self, message: str) -> None:
        self.history.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str) -> None:
        self.history.append({"role": "assistant", "content": message})

    def serialize(self, include_system=True) -> list[dict]:
        copy = self.history.copy()
        if include_system and self.system_prompt:
            copy.insert(0, {"role": "system", "content": self.system_prompt})
        return copy

    def __str__(self, include_system=True) -> str:
        if include_system and self.system_prompt:
            all_history = [
                {"role": "system", "content": self.system_prompt}
            ] + self.history
            return json.dumps(all_history, indent=2)
        return json.dumps(self.history, indent=2)

    @classmethod
    def from_list(cls, messages: list) -> Conversation:
        system_prompt = None
        history = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system" and system_prompt is None:
                system_prompt = content
            else:
                history.append({"role": role, "content": content})
        conv = cls(system_prompt=system_prompt)
        conv.history = history
        return conv


class UnifiedLLMClient:
    """
    A unified client for invoking LLM from different providers.

    Detailed documentation is available at: `docs/unified_llm_client.md`
    """

    # load supported models from supported_models.yaml
    SUPPORTED_MODELS = yaml.safe_load(
        (Path(__file__).parent / "supported_models.yaml").read_text(encoding="utf-8")
    )

    def __init__(
        self, model: str, 
        provider: str | None = None,
        base_url: str | None = None,
        deterministic_mode: bool = True,
    ) -> None:
        """
        Initialize an LLM client. Can optionally specify which provider to use.
        Supported (model, provider) combinations are listed in `supported_models.yaml`

        Args:
        - model: Model name to use.
        - provider: Optional provider name. Defaults to the first provider for the model.
        - base_url: Optional base URL for local providers.
        - deterministic_mode: Whether to use best-effort deterministic settings.
        """
        if provider == "local":
            self.model = model
            self.provider = provider
            self.model_id_at_provider = model
        else:
            if model not in self.SUPPORTED_MODELS:
                raise ValueError(f"UnifiedLLMClient doesn't support {model} yet")

            if provider is None:
                provider = next(iter(self.SUPPORTED_MODELS[model]))
            elif provider not in self.SUPPORTED_MODELS[model]:
                raise ValueError(f"UnifiedLLMClient doesn't support {model} from {provider} yet")

            self.model = model
            self.model_id_at_provider = self.SUPPORTED_MODELS[model][provider]
            self.provider = provider
        
        # deterministic mode is best-effort (not guaranteed)
        self.deterministic_mode = deterministic_mode

        def check_env_var(key: str):
            if key not in os.environ:
                raise OSError(f"Please configure {key} in environment variable")

        match self.provider:
            case "openai":
                check_env_var("OPENAI_API_KEY")
                self.client = OpenAI()

            case "azure-openai":
                check_env_var("OPENAI_API_KEY")
                check_env_var("AZURE_OPENAI_ENDPOINT")
                self.client = OpenAI(base_url=os.environ.get("AZURE_OPENAI_ENDPOINT"))

            case "openrouter":
                check_env_var("OPENROUTER_API_KEY")
                self.client = OpenAI(
                    api_key=os.environ.get("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1"
                )

            case "gemini":
                check_env_var("GEMINI_API_KEY")
                self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

            case "aws":
                check_env_var("AWS_ACCESS_KEY_ID")
                check_env_var("AWS_SECRET_ACCESS_KEY")
                check_env_var("AWS_DEFAULT_REGION")
                self.client = boto3.client(
                    "bedrock-runtime", 
                    config=Config(retries={"max_attempts": 1, "mode": "standard"}),
                )

            case "anthropic":
                check_env_var("ANTHROPIC_API_KEY")
                self.client = Anthropic()

            case "local":
                if base_url is None:
                    raise ValueError("Please specify base_url to use local model")
                if not base_url.rstrip("/").endswith("/v1"):
                    raise ValueError("base_url should end with '/v1' (e.g. http://localhost:30000/v1)")
                self.client = OpenAI(base_url=base_url, api_key="")

            case _:
                raise ValueError(f"Invalid provider {self.provider}")

    def generate(
        self,
        user_input: str,
        conversation: Conversation | list | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        response_format: dict | None = None,
        top_p: float | None = None,
        n: int | None = None,
        seed: int | None = None,
        allow_overwrite_sysprompt: bool = True,
    ) -> (str, Conversation):
        """
        Generate a response based on the user input and optional parameters.

        Params:
        - user_input: can be string or textgrad variable-like (has .`value`)

        Return: (response_text, conversation) as a tuple
        """
        # Normalize user_input to string (support textgrad Variable with .value)
        user_text = user_input

        # Append user input to conversation history (if no conversation is provided, create one)
        if isinstance(conversation, list):
            new_conversation = Conversation.from_list(conversation)
        elif isinstance(conversation, Conversation):
            new_conversation = copy.deepcopy(conversation)
        elif conversation is None:
            new_conversation = Conversation(system_prompt=system_prompt)
        else:
            raise TypeError("conversation must have type Conversation, list, or None")
        new_conversation.add_user_message(user_text)

        if allow_overwrite_sysprompt and system_prompt is not None:
            new_conversation.system_prompt = system_prompt

        # check if the provided system prompt matches with conversation system promp
        if system_prompt is not None and new_conversation.system_prompt != system_prompt:
            raise ValueError(
                "System prompt does not match the one in the conversation.\n"
                f"Existing: {new_conversation.system_prompt}\n"
                f"Provided: {system_prompt}"
            )

        # Load model default params (configured in model_default_params.py)
        # Then override with any provided params
        # Finally remove model unsafe params (e.g. temperature for GPT-5)
        params = get_model_default_params(self.model, self.deterministic_mode)

        if temperature is not None:
            params["temperature"] = temperature
        if max_output_tokens is not None:
            params["max_output_tokens"] = max_output_tokens
        if reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort
        if response_format is not None:
            params["response_format"] = response_format
        if top_p is not None:
            params["top_p"] = top_p
        if n is not None:
            params["n"] = n
        if seed is not None:
            params["seed"] = seed

        # Remove params not supported by certain models
        remove_model_unsafe_params(self.model, params)

        # Some API provider may not support all params above. If that's the case, the unsupported
        # params will simply be ignored rather than causing an error.
        input = new_conversation.history
        system_prompt = new_conversation.system_prompt
        match self.provider:
            case "openai" | "azure-openai" | "openrouter" | "local":
                # TODO: using legacy Chat Completion API for now. consider switching to newer Responses API
                output_text = self._generate_openai_chat(input=input, system_prompt=system_prompt, **params)
            case "gemini":
                output_text = self._generate_gemini(input=input, system_prompt=system_prompt, **params)
            case "aws":
                output_text = self._generate_aws_converse(input=input, system_prompt=system_prompt, **params)
            case "anthropic":
                output_text = self._generate_anthropic(input=input, system_prompt=system_prompt, **params)
            case _:
                raise RuntimeError(f"Internal error: Invalid provider {self.provider}")

        # If output_text starts with <think>, remove all content between <think> and </think> (including the tags)
        if isinstance(output_text, str) and output_text.startswith("<think>"):
            output_text = re.sub(r"<think>.*?</think>", "", output_text, count=1, flags=re.DOTALL).lstrip()
 
        # Append response to conversation and return
        new_conversation.add_assistant_message(output_text)
        return output_text, new_conversation

    def _generate_openai_chat(
        self,
        input: list[dict],
        system_prompt: str | None = None,
        **params
    ) -> str:
        """
        Generate a response using OpenAI's legacy Chat Completion API:
        https://platform.openai.com/docs/api-reference/chat/create
        """
        # handle input and system prompt
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages += input
        request_params = {
            "model": self.model_id_at_provider,
            "messages": messages,
        }

        # handle additional inference parameters below
        if (value := params.pop("temperature", None)) is not None:
            request_params["temperature"] = value
        if (value := params.pop("max_output_tokens", None)) is not None:
            request_params["max_completion_tokens"] = value
        if (value := params.pop("reasoning_effort", None)) is not None:
            if self.provider == "openrouter": # OpenRouter uses special format
                request_params["extra_body"] = {
                    "reasoning": {
                        "enabled": True,
                        "effort": value
                    }
                }
            else:
                request_params["reasoning_effort"] = value
        if (value := params.pop("response_format", None)) is not None:
            request_params["response_format"] = value
        if (value := params.pop("top_p", None)) is not None:
            request_params["top_p"] = value
        if (value := params.pop("n", None)) is not None:
            request_params["n"] = value
        if (value := params.pop("seed", None)) is not None:
            request_params["seed"] = value

        if params:
            logger.warning(
                "OpenAI Chat Completion API doesn't support the following args: %s",
                ", ".join(params.keys()),
            )

        response = self.client.chat.completions.create(**request_params)
        if not response.choices[0].message.content:
            logger.warning(f"Received None response: {response}")

        return response.choices[0].message.content

    def _generate_gemini(
        self,
        input: list[dict],
        system_prompt: str | None = None,
        **params
    ) -> str:
        """
        Generate a response using Google Gemini via `google-genai`.
        """
        # handle input and system prompt
        contents = []
        for entry in input:
            contents.append(
                {
                    "role": "model" if entry.get("role") != "user" else "user",
                    "parts": [{"text": entry.get("content")}],
                }
            )

        # handle additional inference parameters below
        cfg = {}
        cfg["automatic_function_calling"] = {"disable": True}
        if system_prompt is not None:
            cfg["system_instruction"] = system_prompt
        if (value := params.pop("temperature", None)) is not None:
            cfg["temperature"] = value
        if (value := params.pop("max_output_tokens", None)) is not None:
            cfg["max_output_tokens"] = value
        if (value := params.pop("reasoning_effort", None)) is not None:
            cfg["thinking_config"] = {
                "include_thoughts": False,
                "thinking_level": value
            }
        if (value := params.pop("response_format", None)) is not None:
            cfg["response_mime_type"] = value
        if (value := params.pop("top_p", None)) is not None:
            cfg["top_p"] = float(value)
        if (value := params.pop("seed", None)) is not None:
            cfg["seed"] = int(value)

        if params:
            logger.warning(
                "Gemini API doesn't support the following args: %s",
                ", ".join(params.keys()),
            )

        response = self.client.models.generate_content(
            model=self.model_id_at_provider,
            contents=contents,
            config=cfg,
        )
        return response.text

    def _generate_openai_responses(
        self,
        input: list[dict],
        system_prompt: str | None = None,
        **params,
    ) -> str:
        """
        Generate a response using OpenAI's new Responses API (released March 2025):
        https://platform.openai.com/docs/api-reference/responses

        Note: Responses API doesn't support legacy params such as seed.
        """
        # handle input and system prompt
        request_params = {
            "model": self.model_id_at_provider,
            "input": input,
            "store": False,
        }
        if system_prompt is not None:
            request_params["instructions"] = system_prompt

        # handle additional inference parameters below
        if (value := params.pop("temperature", None)) is not None:
            request_params["temperature"] = value
        if (value := params.pop("max_output_tokens", None)) is not None:
            request_params["max_output_tokens"] = value
        if (value := params.pop("reasoning_effort", None)) is not None:
            request_params["reasoning"] = {"effort": value}
        if (value := params.pop("response_format", None)) is not None:
            request_params["text"] = {"format": value}
        if (value := params.pop("top_p", None)) is not None:
            request_params["top_p"] = value

        if params:
            logger.warning(
                "OpenAI Responses API doesn't support the following args: %s",
                ", ".join(params.keys()),
            )

        response = self.client.responses.create(**request_params)
        return response.output_text

    def _generate_aws_converse(
        self,
        input: list[dict],
        system_prompt: str | None = None,
        **params
    ) -> str:
        """
        Generate a response using AWS's Converse API:
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html
        """
        # By default, input has following structure:
        #   [ {"role": "user", "content": "Hi"} ]
        # Bedrock requires the following structure:
        #   [ {"role": "user", "content": [{"text": "Hi"}]} ]
        messages = []
        for entry in input:
            messages.append(
                {
                    "role": entry.get("role"),
                    "content": [{"text": entry.get("content")}],
                }
            )
        request_params = {
            "modelId": self.model_id_at_provider,
            "messages": messages,
        }
        if system_prompt is not None:
            request_params["system"] = [{"text": system_prompt}]

        # handle additional inference parameters below
        inference_config = {}       
        if (value := params.pop("temperature", None)) is not None:
            inference_config["temperature"] = value
        if (value := params.pop("max_output_tokens", None)) is not None:
            inference_config["maxTokens"] = value
        if inference_config:
            request_params["inferenceConfig"] = inference_config

        if params:
            logger.warning(
                "Bedrock Converse API doesn't support the following args: %s",
                ", ".join(params.keys()),
            )
        
        response = self.client.converse(**request_params)
        return response["output"]["message"]["content"][0]["text"]

    def _generate_anthropic(
        self,
        input: list[dict],
        system_prompt: str | None = None,
        **params
    ) -> str:
        """
        Generate a response using Anthropic's Messages API:
        https://platform.claude.com/docs/en/api/messages/create
        """
        request_params = {
            "model": self.model_id_at_provider,
            "messages": input,
        }
        if system_prompt is not None:
            request_params["system"] = system_prompt

        # handle additional inference parameters below
        if (value := params.pop("temperature", None)) is not None:
            request_params["temperature"] = value
        if (value := params.pop("max_output_tokens", None)) is not None:
            request_params["max_tokens"] = value
        if (value := params.pop("response_format", None)) is not None:
            request_params["output_config"] = {"format": value}
        if (value := params.pop("top_p", None)) is not None:
            request_params["top_p"] = value

        # Anthropic requires max_tokens to be set
        if "max_tokens" not in request_params:
            request_params["max_tokens"] = 10000

        if params:
            logger.warning(
                "Anthropic Messaages API doesn't support the following args: %s",
                ", ".join(params.keys()),
            )

        response = self.client.messages.create(**request_params)
        return response.content[0].text
