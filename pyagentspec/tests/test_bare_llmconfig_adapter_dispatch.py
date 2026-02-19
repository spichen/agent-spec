# Copyright © 2026 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""Tests for bare LlmConfig adapter dispatch.

Each adapter test is skipped if the corresponding adapter package is not installed.
"""

import pytest

from pyagentspec.llms import LlmConfig


@pytest.fixture
def bare_llmconfig_openai() -> LlmConfig:
    return LlmConfig(name="test", model_id="gpt-4o", api_provider="openai")


@pytest.fixture
def bare_llmconfig_unsupported() -> LlmConfig:
    return LlmConfig(name="test", model_id="some-model", api_provider="unsupported_provider")


class TestOpenAiAgentsDispatch:
    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        pytest.importorskip("openai")
        pytest.importorskip("agents")

    def test_openai_provider_returns_model_id(self, bare_llmconfig_openai: LlmConfig) -> None:
        from pyagentspec.adapters.openaiagents._openaiagentsconverter import OpenAiAgentsConverter

        converter = OpenAiAgentsConverter()
        result = converter._llm_convert_to_openai(bare_llmconfig_openai)
        assert result == "gpt-4o"

    def test_unsupported_provider_raises(self, bare_llmconfig_unsupported: LlmConfig) -> None:
        from pyagentspec.adapters.openaiagents._openaiagentsconverter import OpenAiAgentsConverter

        converter = OpenAiAgentsConverter()
        with pytest.raises(NotImplementedError, match="unsupported_provider"):
            converter._llm_convert_to_openai(bare_llmconfig_unsupported)


class TestAutogenDispatch:
    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        pytest.importorskip("autogen_agentchat")
        pytest.importorskip("autogen_ext")

    def test_unsupported_provider_raises(self, bare_llmconfig_unsupported: LlmConfig) -> None:
        from pyagentspec.adapters.autogen._autogenconverter import AutogenConverter

        converter = AutogenConverter()
        with pytest.raises(NotImplementedError, match="unsupported_provider"):
            converter._llm_convert_to_autogen(bare_llmconfig_unsupported)


class TestCrewAiDispatch:
    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        pytest.importorskip("crewai")

    def test_unsupported_provider_raises(self, bare_llmconfig_unsupported: LlmConfig) -> None:
        from pyagentspec.adapters.crewai._crewaiconverter import CrewAiConverter

        converter = CrewAiConverter()
        with pytest.raises(NotImplementedError, match="unsupported_provider"):
            converter._llm_convert_to_crewai(bare_llmconfig_unsupported)
