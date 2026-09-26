# Copyright © 2025, 2026 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""How an agent's declared ``outputFormat`` decides between free text and structured output.

The model is a fake injected through the converter, so these run without an LLM endpoint.
"""

from typing import Any, Dict, List, Optional

import pytest

from pyagentspec.adapters.langgraph._agent_output_guard import StructuredOutputNotProducedError
from pyagentspec.adapters.langgraph._node_execution import extract_outputs_from_invoke_result
from pyagentspec.agent import Agent
from pyagentspec.llms import OpenAiCompatibleConfig
from pyagentspec.property import StringProperty


def _run(
    monkeypatch: pytest.MonkeyPatch,
    replies: List[Any],
    output_titles: List[str],
    output_format: Optional[str] = None,
    structured_output_profile: bool = False,
) -> Dict[str, Any]:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import HumanMessage

    from pyagentspec.adapters.langgraph import AgentSpecLoader
    from pyagentspec.adapters.langgraph._langgraphconverter import (
        AgentSpecToLangGraphConverter,
    )

    bound_tools: List[Any] = []

    class _FakeModel(GenericFakeChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            bound_tools.extend(tools)
            return self

    model = _FakeModel(messages=iter(replies))
    if structured_output_profile:
        # Makes LangChain pick ProviderStrategy if the adapter lets it choose.
        model.profile = {"structured_output": True}
    monkeypatch.setattr(
        AgentSpecToLangGraphConverter,
        "_llm_convert_to_langgraph",
        lambda self, *args, **kwargs: model,
    )

    outputs = [StringProperty(title=title) for title in output_titles]
    agent = Agent(
        name="classifier",
        llm_config=OpenAiCompatibleConfig(name="llm", model_id="some-model", url="http://x"),
        system_prompt="Classify the sentiment.",
        outputs=outputs,
        metadata={"outputFormat": output_format} if output_format else {},
    )
    graph = AgentSpecLoader().load_component(agent)
    result = graph.invoke(
        {"messages": [HumanMessage(content="I love it")]},
        {"configurable": {"thread_id": "t"}},
    )
    return {
        "outputs": extract_outputs_from_invoke_result(result, outputs),
        "bound_tools": bound_tools,
    }


def _prose(n: int = 10) -> List[Any]:
    from langchain_core.messages import AIMessage

    return [AIMessage(content="positive, I think")] * n


def _structured(**args: str) -> List[Any]:
    from langchain_core.messages import AIMessage

    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "AgentOutputModel", "args": args, "id": "call_1"}],
        )
    ]


@pytest.mark.parametrize("output_format", [None, "TEXT", "text"])
def test_text_single_string_output_is_the_final_message(
    monkeypatch: pytest.MonkeyPatch, output_format: Optional[str]
) -> None:
    run = _run(monkeypatch, _prose(), ["sentiment"], output_format)

    assert run["outputs"] == {"sentiment": "positive, I think"}
    assert run["bound_tools"] == []


def test_json_single_string_output_uses_structured_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run(monkeypatch, _structured(sentiment="positive"), ["sentiment"], "JSON")

    assert run["outputs"] == {"sentiment": "positive"}


def test_json_single_string_output_error_points_at_text_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generic advice, declaring a single string output, is what this caller did."""
    with pytest.raises(StructuredOutputNotProducedError) as excinfo:
        _run(monkeypatch, _prose(), ["sentiment"], "JSON")

    message = str(excinfo.value)
    assert "outputFormat to TEXT" in message
    assert "Declare a single string output" not in message


def test_multi_field_output_error_keeps_the_generic_advice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(StructuredOutputNotProducedError) as excinfo:
        _run(monkeypatch, _prose(), ["sentiment", "reason"])

    assert "Declare a single string output" in str(excinfo.value)


def test_structured_output_is_a_described_tool_even_on_provider_capable_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ToolStrategy is forced, so the structured response is a tool call the output guard
    can recognise, and the schema carries the description some providers require."""
    run = _run(
        monkeypatch,
        _structured(sentiment="positive"),
        ["sentiment"],
        "JSON",
        structured_output_profile=True,
    )

    assert run["outputs"] == {"sentiment": "positive"}
    [output_tool] = [t for t in run["bound_tools"] if t.name == "AgentOutputModel"]
    assert output_tool.description == "Structured output for the agent."
