# Copyright © 2026 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""Tests for the OpenRouter empty-turn mitigation in the LangGraph adapter.

OpenRouter fans a single model id out to several upstream backends; some drop
tool-calls or return reasoning-only output, yielding an empty assistant turn
that trips the empty-turn guard in ``adapters/langgraph/tracing.py``. The
adapter constrains OpenRouter routing (``provider.require_parameters``) and
re-issues empty turns. Both are scoped to OpenRouter, leaving other
OpenAI-compatible endpoints untouched.
"""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from pyagentspec.adapters.langgraph import _langgraphconverter as conv
from pyagentspec.adapters.langgraph._langgraphconverter import AgentSpecToLangGraphConverter
from pyagentspec.adapters.langgraph._openrouter import RetryOnEmptyChatOpenAI
from pyagentspec.llms import OpenAiCompatibleConfig


def _openrouter_config() -> OpenAiCompatibleConfig:
    return OpenAiCompatibleConfig(
        name="minimax",
        model_id="minimax/minimax-m2.7",
        url="https://openrouter.ai/api/v1",
        api_key="sk-test-key",
    )


def _vllm_config() -> OpenAiCompatibleConfig:
    return OpenAiCompatibleConfig(
        name="local",
        model_id="local-model",
        url="http://vllm.internal:8000/v1",
        api_key="sk-test-key",
    )


# --- URL detection -----------------------------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://openrouter.ai/api/v1", True),
        ("https://OpenRouter.ai/api/v1", True),
        ("https://api.openai.com/v1", False),
        ("http://vllm.internal:8000/v1", False),
        (None, False),
    ],
)
def test_is_openrouter_url(url, expected) -> None:
    assert conv._is_openrouter_url(url) is expected


# --- model construction (gating) ---------------------------------------------


def test_openrouter_model_gets_provider_routing_and_retry_subclass() -> None:
    converter = AgentSpecToLangGraphConverter()
    model = converter._llm_convert_to_langgraph(_openrouter_config(), RunnableConfig())
    assert isinstance(model, RetryOnEmptyChatOpenAI)
    assert model.extra_body == {"provider": {"require_parameters": True}}


def test_non_openrouter_model_is_untouched() -> None:
    converter = AgentSpecToLangGraphConverter()
    model = converter._llm_convert_to_langgraph(_vllm_config(), RunnableConfig())
    assert type(model) is ChatOpenAI
    assert model.extra_body is None


# --- retry behaviour ---------------------------------------------------------


def _chunk(content: str = "", tool_call_chunks=None) -> ChatGenerationChunk:
    return ChatGenerationChunk(
        message=AIMessageChunk(content=content, tool_call_chunks=tool_call_chunks or [])
    )


def _result(content: str = "", tool_calls=None) -> ChatResult:
    return ChatResult(
        generations=[ChatGeneration(message=AIMessage(content=content, tool_calls=tool_calls or []))]
    )


def _model():
    return RetryOnEmptyChatOpenAI(model="x", api_key="k", base_url="https://openrouter.ai/api/v1")


@pytest.mark.asyncio
async def test_agenerate_retries_until_non_empty(monkeypatch) -> None:
    calls = {"n": 0}

    async def fake(self, messages, stop=None, run_manager=None, **kw):
        calls["n"] += 1
        return _result("") if calls["n"] < 3 else _result("hello")

    monkeypatch.setattr(ChatOpenAI, "_agenerate", fake)
    result = await _model()._agenerate([])
    assert result.generations[0].message.content == "hello"
    assert calls["n"] == 3  # 1 initial + 2 retries


@pytest.mark.asyncio
async def test_agenerate_gives_up_after_max_retries(monkeypatch) -> None:
    calls = {"n": 0}

    async def fake(self, messages, stop=None, run_manager=None, **kw):
        calls["n"] += 1
        return _result("")

    monkeypatch.setattr(ChatOpenAI, "_agenerate", fake)
    result = await _model()._agenerate([])
    # Returns the empty turn so the downstream guard fires, does not loop forever.
    assert result.generations[0].message.content == ""
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_astream_suppresses_empty_attempt_then_streams_retry(monkeypatch) -> None:
    attempt = {"n": 0}

    async def fake(self, messages, stop=None, run_manager=None, **kw):
        attempt["n"] += 1
        if attempt["n"] == 1:
            for _ in range(3):
                yield _chunk("")  # reasoning-only / empty
        else:
            for tok in ("He", "llo"):
                yield _chunk(tok)

    monkeypatch.setattr(ChatOpenAI, "_astream", fake)
    out = [c.message.content async for c in _model()._astream([])]
    assert "".join(out) == "Hello"  # empty attempt not yielded downstream
    assert attempt["n"] == 2


@pytest.mark.asyncio
async def test_astream_emits_empty_when_all_attempts_empty(monkeypatch) -> None:
    attempt = {"n": 0}

    async def fake(self, messages, stop=None, run_manager=None, **kw):
        attempt["n"] += 1
        yield _chunk("")

    monkeypatch.setattr(ChatOpenAI, "_astream", fake)
    out = [c async for c in _model()._astream([])]
    assert "".join(c.message.content for c in out) == ""
    assert attempt["n"] == 3  # 1 initial + 2 retries, then emit empty for the guard


@pytest.mark.asyncio
async def test_astream_treats_tool_call_as_real(monkeypatch) -> None:
    attempt = {"n": 0}

    async def fake(self, messages, stop=None, run_manager=None, **kw):
        attempt["n"] += 1
        yield _chunk("", tool_call_chunks=[{"name": "f", "args": "{}", "id": "1", "index": 0}])

    monkeypatch.setattr(ChatOpenAI, "_astream", fake)
    out = [c async for c in _model()._astream([])]
    assert len(out) == 1
    assert attempt["n"] == 1  # a tool call is real output, no retry
