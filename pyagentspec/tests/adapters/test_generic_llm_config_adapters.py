# Copyright © 2026 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""Tests that each adapter correctly converts GenericLlmConfig."""

import os

import pytest

from pyagentspec.llms import AuthConfig, GenericLlmConfig, ProviderConfig


def _make_generic_config(
    *,
    provider_type: str = "openai",
    endpoint: str | None = None,
    credential_ref: str | None = None,
) -> GenericLlmConfig:
    auth = AuthConfig(type="api_key", credential_ref=credential_ref) if credential_ref else None
    return GenericLlmConfig(
        id="test-llm",
        name="test-llm",
        model_id="my-model",
        provider=ProviderConfig(type=provider_type, endpoint=endpoint),
        auth=auth,
    )


# ---------------------------------------------------------------------------
# LangGraph adapter
# ---------------------------------------------------------------------------


class TestLangGraphGenericLlmConfig:
    @pytest.fixture(autouse=True)
    def _import_deps(self) -> None:
        pytest.importorskip("langgraph")
        pytest.importorskip("langchain_openai")

    def test_with_endpoint(self) -> None:
        from langchain_openai import ChatOpenAI

        from pyagentspec.adapters.langgraph._langgraphconverter import (
            AgentSpecToLangGraphConverter,
        )

        config = _make_generic_config(
            endpoint="http://localhost:8000", credential_ref="test-key"
        )
        converter = AgentSpecToLangGraphConverter()
        result = converter._llm_convert_to_langgraph(config, {})

        assert isinstance(result, ChatOpenAI)
        assert result.model_name == "my-model"
        assert "/v1" in str(result.openai_api_base)
        assert result.openai_api_key.get_secret_value() == "test-key"

    def test_without_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from langchain_openai import ChatOpenAI

        from pyagentspec.adapters.langgraph._langgraphconverter import (
            AgentSpecToLangGraphConverter,
        )

        monkeypatch.setenv("OPENAI_API_KEY", "dummy")
        config = _make_generic_config()
        converter = AgentSpecToLangGraphConverter()
        result = converter._llm_convert_to_langgraph(config, {})

        assert isinstance(result, ChatOpenAI)
        assert result.model_name == "my-model"

    def test_vllm_provider(self) -> None:
        from langchain_openai import ChatOpenAI

        from pyagentspec.adapters.langgraph._langgraphconverter import (
            AgentSpecToLangGraphConverter,
        )

        config = _make_generic_config(
            provider_type="vllm",
            endpoint="http://localhost:8000",
        )
        converter = AgentSpecToLangGraphConverter()
        result = converter._llm_convert_to_langgraph(config, {})

        assert isinstance(result, ChatOpenAI)
        assert result.model_name == "my-model"
        assert "/v1" in str(result.openai_api_base)
        assert result.openai_api_key.get_secret_value() == "EMPTY"

    def test_ollama_provider(self) -> None:
        pytest.importorskip("langchain_ollama")
        from langchain_ollama import ChatOllama

        from pyagentspec.adapters.langgraph._langgraphconverter import (
            AgentSpecToLangGraphConverter,
        )

        config = _make_generic_config(
            provider_type="ollama",
            endpoint="http://localhost:11434",
        )
        converter = AgentSpecToLangGraphConverter()
        result = converter._llm_convert_to_langgraph(config, {})

        assert isinstance(result, ChatOllama)
        assert result.model == "my-model"

    def test_openai_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from langchain_openai import ChatOpenAI

        from pyagentspec.adapters.langgraph._langgraphconverter import (
            AgentSpecToLangGraphConverter,
        )

        monkeypatch.setenv("OPENAI_API_KEY", "dummy")
        config = _make_generic_config(provider_type="openai")
        converter = AgentSpecToLangGraphConverter()
        result = converter._llm_convert_to_langgraph(config, {})

        assert isinstance(result, ChatOpenAI)
        assert result.model_name == "my-model"


# ---------------------------------------------------------------------------
# OpenAI Agents adapter
# ---------------------------------------------------------------------------


class TestOpenAIAgentsGenericLlmConfig:
    @pytest.fixture(autouse=True)
    def _import_deps(self) -> None:
        pytest.importorskip("agents")
        pytest.importorskip("openai")

    def test_with_endpoint(self) -> None:
        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

        from pyagentspec.adapters.openaiagents._openaiagentsconverter import (
            AgentSpecToOpenAIConverter,
        )

        config = _make_generic_config(
            endpoint="http://localhost:8000", credential_ref="test-key"
        )
        converter = AgentSpecToOpenAIConverter()
        result = converter._llm_convert_to_openai(config)

        assert isinstance(result, OpenAIChatCompletionsModel)
        client = getattr(result, "_client", None)
        base_url = getattr(client, "base_url", None) if client else None
        assert base_url is not None
        assert "v1" in str(base_url)

    def test_without_endpoint(self) -> None:
        from pyagentspec.adapters.openaiagents._openaiagentsconverter import (
            AgentSpecToOpenAIConverter,
        )

        config = _make_generic_config()
        converter = AgentSpecToOpenAIConverter()
        result = converter._llm_convert_to_openai(config)

        assert result == "my-model"

    def test_vllm_provider(self) -> None:
        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

        from pyagentspec.adapters.openaiagents._openaiagentsconverter import (
            AgentSpecToOpenAIConverter,
        )

        config = _make_generic_config(
            provider_type="vllm",
            endpoint="http://localhost:8000",
        )
        converter = AgentSpecToOpenAIConverter()
        result = converter._llm_convert_to_openai(config)

        assert isinstance(result, OpenAIChatCompletionsModel)

    def test_ollama_provider(self) -> None:
        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

        from pyagentspec.adapters.openaiagents._openaiagentsconverter import (
            AgentSpecToOpenAIConverter,
        )

        config = _make_generic_config(
            provider_type="ollama",
            endpoint="http://localhost:11434",
        )
        converter = AgentSpecToOpenAIConverter()
        result = converter._llm_convert_to_openai(config)

        assert isinstance(result, OpenAIChatCompletionsModel)

    def test_openai_provider(self) -> None:
        from pyagentspec.adapters.openaiagents._openaiagentsconverter import (
            AgentSpecToOpenAIConverter,
        )

        config = _make_generic_config(provider_type="openai")
        converter = AgentSpecToOpenAIConverter()
        result = converter._llm_convert_to_openai(config)

        assert result == "my-model"


# ---------------------------------------------------------------------------
# AutoGen adapter
# ---------------------------------------------------------------------------


class TestAutoGenGenericLlmConfig:
    @pytest.fixture(autouse=True)
    def _import_deps(self) -> None:
        pytest.importorskip("autogen_ext")
        pytest.importorskip("autogen_core")

    def test_with_endpoint(self) -> None:
        from pyagentspec.adapters.autogen._autogenconverter import AgentSpecToAutogenConverter
        from pyagentspec.adapters.autogen._types import AutogenOpenAIChatCompletionClient

        config = _make_generic_config(
            endpoint="localhost:8000", credential_ref="test-key"
        )
        converter = AgentSpecToAutogenConverter()
        result = converter._llm_convert_to_autogen(config, tool_registry={})

        assert isinstance(result, AutogenOpenAIChatCompletionClient)

    def test_with_endpoint_passes_auth(self) -> None:
        from pyagentspec.adapters.autogen._autogenconverter import AgentSpecToAutogenConverter

        config = _make_generic_config(
            endpoint="localhost:8000", credential_ref="my-secret-key"
        )
        converter = AgentSpecToAutogenConverter()
        result = converter._llm_convert_to_autogen(config, tool_registry={})

        assert result is not None

    def test_vllm_provider(self) -> None:
        from pyagentspec.adapters.autogen._autogenconverter import AgentSpecToAutogenConverter
        from pyagentspec.adapters.autogen._types import AutogenOpenAIChatCompletionClient

        config = _make_generic_config(
            provider_type="vllm",
            endpoint="localhost:8000",
        )
        converter = AgentSpecToAutogenConverter()
        result = converter._llm_convert_to_autogen(config, tool_registry={})

        assert isinstance(result, AutogenOpenAIChatCompletionClient)

    def test_ollama_provider(self) -> None:
        from pyagentspec.adapters.autogen._autogenconverter import AgentSpecToAutogenConverter
        from pyagentspec.adapters.autogen._types import AutogenOllamaChatCompletionClient

        config = _make_generic_config(
            provider_type="ollama",
            endpoint="localhost:11434",
        )
        converter = AgentSpecToAutogenConverter()
        result = converter._llm_convert_to_autogen(config, tool_registry={})

        assert isinstance(result, AutogenOllamaChatCompletionClient)

    def test_openai_provider(self) -> None:
        from pyagentspec.adapters.autogen._autogenconverter import AgentSpecToAutogenConverter
        from pyagentspec.adapters.autogen._types import AutogenOpenAIChatCompletionClient

        config = _make_generic_config(provider_type="openai")
        converter = AgentSpecToAutogenConverter()
        result = converter._llm_convert_to_autogen(config, tool_registry={})

        assert isinstance(result, AutogenOpenAIChatCompletionClient)


# ---------------------------------------------------------------------------
# Agent Framework adapter
# ---------------------------------------------------------------------------


class TestAgentFrameworkGenericLlmConfig:
    @pytest.fixture(autouse=True)
    def _import_deps(self) -> None:
        pytest.importorskip("microsoft.agents")

    def test_with_endpoint(self) -> None:
        from pyagentspec.adapters.agent_framework._agentframeworkconverter import (
            AgentSpecToAgentFrameworkConverter,
        )

        config = _make_generic_config(
            endpoint="localhost:8000", credential_ref="test-key"
        )
        converter = AgentSpecToAgentFrameworkConverter()
        result = converter._llm_convert_to_agent_framework(
            config, tool_registry={}, converted_components={}
        )

        from microsoft.agents.builder.chat_completion_client_base import (
            BaseChatClient,
        )

        assert isinstance(result, BaseChatClient)

    def test_without_endpoint(self) -> None:
        from pyagentspec.adapters.agent_framework._agentframeworkconverter import (
            AgentSpecToAgentFrameworkConverter,
        )

        config = _make_generic_config(credential_ref="test-key")
        converter = AgentSpecToAgentFrameworkConverter()
        result = converter._llm_convert_to_agent_framework(
            config, tool_registry={}, converted_components={}
        )

        from microsoft.agents.builder.chat_completion_client_base import (
            BaseChatClient,
        )

        assert isinstance(result, BaseChatClient)

    def test_vllm_provider(self) -> None:
        from pyagentspec.adapters.agent_framework._agentframeworkconverter import (
            AgentSpecToAgentFrameworkConverter,
        )

        config = _make_generic_config(
            provider_type="vllm",
            endpoint="localhost:8000",
        )
        converter = AgentSpecToAgentFrameworkConverter()
        result = converter._llm_convert_to_agent_framework(
            config, tool_registry={}, converted_components={}
        )

        from microsoft.agents.builder.chat_completion_client_base import (
            BaseChatClient,
        )

        assert isinstance(result, BaseChatClient)

    def test_ollama_provider(self) -> None:
        from pyagentspec.adapters.agent_framework._agentframeworkconverter import (
            AgentSpecToAgentFrameworkConverter,
        )

        config = _make_generic_config(
            provider_type="ollama",
            endpoint="http://localhost:11434",
        )
        converter = AgentSpecToAgentFrameworkConverter()
        result = converter._llm_convert_to_agent_framework(
            config, tool_registry={}, converted_components={}
        )

        from microsoft.agents.builder.chat_completion_client_base import (
            BaseChatClient,
        )

        assert isinstance(result, BaseChatClient)

    def test_openai_provider(self) -> None:
        from pyagentspec.adapters.agent_framework._agentframeworkconverter import (
            AgentSpecToAgentFrameworkConverter,
        )

        config = _make_generic_config(provider_type="openai", credential_ref="test-key")
        converter = AgentSpecToAgentFrameworkConverter()
        result = converter._llm_convert_to_agent_framework(
            config, tool_registry={}, converted_components={}
        )

        from microsoft.agents.builder.chat_completion_client_base import (
            BaseChatClient,
        )

        assert isinstance(result, BaseChatClient)
