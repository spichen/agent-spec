# Copyright © 2026 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""OpenRouter-specific handling for the LangGraph OpenAI-compatible adapter.

OpenRouter is a broker that fans a single model id out to several upstream
backends and picks one per request. Those backends differ in tool-calling
support, so the same prompt can intermittently come back as an empty assistant
turn (no content, no tool_calls) — either because the chosen backend silently
drops tool-calls or because a reasoning model returned reasoning-only output.
That trips the empty-turn guard in ``adapters/langgraph/tracing.py``.

This module is imported lazily (only when an OpenRouter endpoint is built) so
the adapter keeps its optional dependency on ``langchain_openai``.
"""

from typing import Any

from langchain_openai import ChatOpenAI

# Constrain OpenRouter routing to backends that support every request parameter
# (including ``tools``), which removes the backends that silently drop
# tool-calling. Passed through ChatOpenAI as top-level request body.
PROVIDER_ROUTING_EXTRA_BODY = {"provider": {"require_parameters": True}}

# Extra attempts to re-issue a request that returned an empty turn.
_MAX_RETRIES = 2


def _is_empty_turn(message: Any) -> bool:
    """True when an assistant message has no content and no tool calls.

    Fires on exactly the turns the guard in ``tracing.py`` rejects
    (``content == "" and not tool_calls``); ``not content`` also covers the
    ``None``/empty-list content shapes that fail downstream the same way.
    """
    content = getattr(message, "content", None)
    tool_calls = getattr(message, "tool_calls", None) or getattr(
        message, "additional_kwargs", {}
    ).get("tool_calls")
    return not content and not tool_calls


def _chunk_has_real_output(chunk: Any) -> bool:
    """True once a streamed chunk carries content or (part of) a tool call."""
    message = getattr(chunk, "message", None)
    if message is None:
        return False
    # Streamed tool calls arrive incrementally as tool_call_chunks before the
    # assembled tool_calls appear, so check them in addition to the final shape.
    return bool(getattr(message, "tool_call_chunks", None)) or not _is_empty_turn(message)


class RetryOnEmptyChatOpenAI(ChatOpenAI):
    """ChatOpenAI that re-issues a request when the turn comes back empty.

    A fresh request usually re-routes OpenRouter to a healthy backend. If every
    attempt is empty we emit the empty turn so the downstream guard fails
    exactly as it does today. Both the streaming (``_astream``) and
    non-streaming (``_agenerate``) paths are exercised by the runtime, via
    ``astream_events`` and ``AgentNodeExecutor``'s ``ainvoke`` respectively.
    """

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[no-untyped-def]
        for _ in range(_MAX_RETRIES + 1):
            result = await super()._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            generations = getattr(result, "generations", None)
            if not generations or not _is_empty_turn(generations[0].message):
                break
        return result

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[no-untyped-def]
        for attempt in range(_MAX_RETRIES + 1):
            buffer: list[Any] = []
            flushed = False
            async for chunk in super()._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                if flushed:
                    yield chunk
                    continue
                # Hold chunks back until we see real output, so an empty attempt
                # yields nothing (no stray message events) and can be retried
                # transparently. TTFT is preserved: the buffer flushes on the
                # first content/tool-call chunk.
                buffer.append(chunk)
                if _chunk_has_real_output(chunk):
                    for buffered in buffer:
                        yield buffered
                    flushed = True
            if flushed:
                return
            # Empty attempt. Retry unless exhausted; on the final attempt emit
            # the buffered (empty) chunks so the empty-turn guard fires.
            if attempt == _MAX_RETRIES:
                for buffered in buffer:
                    yield buffered
