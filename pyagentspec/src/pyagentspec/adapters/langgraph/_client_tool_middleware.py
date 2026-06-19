"""Middleware that stamps the originating ``tool_call_id`` onto a client tool's
interrupt.

A converted :class:`~pyagentspec.tools.ClientTool` hands control back to the
caller by raising ``interrupt({"type": "client_tool_request", ...})``. That
payload does not carry the id of the tool call that triggered it, but a caller
resuming the run needs that id to correlate the tool result it supplies back to
the specific parked interrupt (multiple client tool calls can be pending at
once). This middleware runs inside ``create_agent``'s tool-call path — where the
``tool_call_id`` is available on the request — and stamps it onto the
``client_tool_request`` interrupt as it propagates, so resume can match it.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langgraph.errors import GraphInterrupt

_ToolResult = Any


def _stamp_tool_call_id(exc: GraphInterrupt, tool_call_id: str | None) -> None:
    """Add ``tool_call_id`` to any pending ``client_tool_request`` interrupt that
    is missing one. Mutates the interrupt value in place so the id is persisted
    when LangGraph parks the interrupt."""
    if not tool_call_id:
        return
    args = getattr(exc, "args", ())
    pending = args[0] if args and isinstance(args[0], (list, tuple)) else ()
    for interrupt in list(pending) + list(getattr(exc, "interrupts", ()) or ()):
        value = getattr(interrupt, "value", None)
        if (
            isinstance(value, dict)
            and value.get("type") == "client_tool_request"
            and not value.get("tool_call_id")
        ):
            value["tool_call_id"] = tool_call_id


class ClientToolCallIdMiddleware(AgentMiddleware):
    """Attach the originating ``tool_call_id`` to ``client_tool_request``
    interrupts so a resuming caller can correlate the tool result it returns."""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], _ToolResult],
    ) -> _ToolResult:
        try:
            return handler(request)
        except GraphInterrupt as exc:
            _stamp_tool_call_id(exc, request.tool_call.get("id"))
            raise

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[_ToolResult]],
    ) -> _ToolResult:
        try:
            return await handler(request)
        except GraphInterrupt as exc:
            _stamp_tool_call_id(exc, request.tool_call.get("id"))
            raise
