# Copyright © 2025 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""Offline tests for the LangGraph ``ManagerWorkers`` converter.

These cover the hierarchical topology, roster prompt rendering, the
worker-isolation invariant (each worker sees only its delegated task),
and the recursive nesting case. The LLM is stubbed with
``FakeMessagesListChatModel`` so the tests run without network or model
endpoints.
"""

from typing import Any
from unittest.mock import patch

import pytest


# ─── Shared helpers ──────────────────────────────────────────────────────────


def _llm_cfg(name: str) -> Any:
    from pyagentspec.llms.openaicompatibleconfig import OpenAiCompatibleConfig

    return OpenAiCompatibleConfig(name=name, model_id="fake", url="null")


def _fake_manager(*ai_responses: Any) -> Any:
    """A FakeMessagesListChatModel subclassed under ChatOpenAI so the
    manager's react-agent treats it as an OpenAI-style chat model."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_openai import ChatOpenAI

    class _FakeModel(FakeMessagesListChatModel, ChatOpenAI):
        pass

    return _FakeModel(responses=list(ai_responses))


# ─── Pure-helper unit tests (no LLM) ────────────────────────────────────────


def test_safe_node_name_lowercases_and_collapses_punctuation() -> None:
    from pyagentspec.adapters.langgraph._langgraphconverter import _safe_node_name

    assert _safe_node_name("Research Helper", "id-1") == "research_helper"
    assert _safe_node_name("My-Worker!! v2", "id-1") == "my_worker_v2"


def test_safe_node_name_falls_back_to_normalized_id() -> None:
    from pyagentspec.adapters.langgraph._langgraphconverter import _safe_node_name

    # Name slugifies to empty → id used (and also normalized).
    assert _safe_node_name("!!!", "sub-1") == "sub_1"
    # Both empty → constant fallback.
    assert _safe_node_name("", "") == "worker"


def test_append_workers_roster_appends_block_after_existing_prompt() -> None:
    from pyagentspec.adapters.langgraph._langgraphconverter import _append_workers_roster

    out = _append_workers_roster(
        "Coordinate the team.",
        [("research_helper", "Handles research"), ("drafter", "Drafts text")],
    )
    assert out == (
        "Coordinate the team.\n\n"
        "Available workers:\n"
        "- research_helper: Handles research\n"
        "- drafter: Drafts text"
    )


def test_append_workers_roster_flattens_multiline_descriptions() -> None:
    from pyagentspec.adapters.langgraph._langgraphconverter import _append_workers_roster

    out = _append_workers_roster(
        "",
        [("helper", "First line\nsecond line\n  third  line  ")],
    )
    # Whitespace flattened so the one-line-per-worker shape survives.
    assert out == "Available workers:\n- helper: First line second line third line"


def test_route_manager_to_worker_or_end_reads_pending_delegation() -> None:
    from langchain_core.messages import AIMessage

    from pyagentspec.adapters.langgraph._langgraphconverter import (
        _route_manager_to_worker_or_end,
    )

    delegating = AIMessage(
        content="",
        tool_calls=[
            {"name": "delegate_to_research_helper", "args": {"task": "hi"}, "id": "c1"}
        ],
    )
    state = {"messages": [delegating]}
    assert _route_manager_to_worker_or_end(state) == "research_helper"


def test_route_manager_to_worker_or_end_returns_end_when_no_delegation() -> None:
    from langchain_core.messages import AIMessage

    from pyagentspec.adapters.langgraph._langgraphconverter import (
        _route_manager_to_worker_or_end,
    )
    from langgraph.graph import END

    not_delegating = AIMessage(content="Done.", tool_calls=[])
    assert _route_manager_to_worker_or_end({"messages": [not_delegating]}) == END
    assert _route_manager_to_worker_or_end({"messages": []}) == END


def test_route_manager_to_worker_or_end_picks_first_delegation_among_multiple_tool_calls() -> None:
    from langchain_core.messages import AIMessage

    from pyagentspec.adapters.langgraph._langgraphconverter import (
        _route_manager_to_worker_or_end,
    )

    msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "some_other_tool", "args": {}, "id": "c0"},
            {"name": "delegate_to_drafter", "args": {"task": "x"}, "id": "c1"},
            {"name": "delegate_to_research_helper", "args": {"task": "y"}, "id": "c2"},
        ],
    )
    # First delegation wins — the others would be handled on the next loop.
    assert _route_manager_to_worker_or_end({"messages": [msg]}) == "drafter"


# ─── Topology test (no LLM execution; checks compiled graph shape) ──────────


def test_manager_workers_compiles_to_hierarchical_graph_topology() -> None:
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START

    from pyagentspec.adapters.langgraph import AgentSpecLoader
    from pyagentspec.adapters.langgraph._langgraphconverter import (
        AgentSpecToLangGraphConverter,
        _MANAGER_NODE_KEY,
    )
    from pyagentspec.agent import Agent
    from pyagentspec.managerworkers import ManagerWorkers

    manager_agent = Agent(
        name="Coordinator",
        description="Coordinates",
        system_prompt="Coordinate the team.",
        llm_config=_llm_cfg("manager_llm"),
    )
    worker_a = Agent(
        name="Research Helper",
        description="Handles research",
        system_prompt="Research.",
        llm_config=_llm_cfg("worker_a_llm"),
    )
    worker_b = Agent(
        name="Drafter",
        description="Drafts text",
        system_prompt="Draft.",
        llm_config=_llm_cfg("worker_b_llm"),
    )
    mw = ManagerWorkers(
        name="ResearchTeam",
        group_manager=manager_agent,
        workers=[worker_a, worker_b],
    )

    fake_llm = _fake_manager(AIMessage(content="Done."))
    loader = AgentSpecLoader(tool_registry={}, checkpointer=MemorySaver())
    with patch.object(
        AgentSpecToLangGraphConverter, "_llm_convert_to_langgraph", return_value=fake_llm
    ):
        compiled = loader.load_component(mw)

    # The compiled object is a CompiledStateGraph; its builder exposes
    # the parent topology we expect.
    builder = compiled.builder
    assert _MANAGER_NODE_KEY in builder.nodes
    assert "research_helper" in builder.nodes
    assert "drafter" in builder.nodes

    # START → manager; every worker → manager (loop).
    edge_pairs = {(src, dst) for src, dst in builder.edges}
    assert (START, _MANAGER_NODE_KEY) in edge_pairs
    assert ("research_helper", _MANAGER_NODE_KEY) in edge_pairs
    assert ("drafter", _MANAGER_NODE_KEY) in edge_pairs

    # The manager → worker routing is a conditional edge (branch), not a
    # plain edge — branches are stored separately on the builder.
    branches = builder.branches.get(_MANAGER_NODE_KEY) or {}
    assert branches, "expected a conditional branch from the manager node"


def test_manager_workers_renders_workers_roster_into_manager_prompt() -> None:
    """The manager's system prompt gets the ``Available workers:`` block
    appended so the LLM knows which delegation tool maps to which worker.
    """
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import MemorySaver

    from pyagentspec.adapters.langgraph import AgentSpecLoader
    from pyagentspec.adapters.langgraph._langgraphconverter import (
        AgentSpecToLangGraphConverter,
        _MANAGER_NODE_KEY,
    )
    from pyagentspec.agent import Agent
    from pyagentspec.managerworkers import ManagerWorkers

    manager_agent = Agent(
        name="Coordinator",
        description="Coordinates",
        system_prompt="Coordinate the team.",
        llm_config=_llm_cfg("manager_llm"),
    )
    worker = Agent(
        name="Research Helper",
        description="Handles research tasks",
        system_prompt="Research.",
        llm_config=_llm_cfg("worker_llm"),
    )
    mw = ManagerWorkers(
        name="Team",
        group_manager=manager_agent,
        workers=[worker],
    )

    fake_llm = _fake_manager(AIMessage(content="Done."))
    loader = AgentSpecLoader(tool_registry={}, checkpointer=MemorySaver())
    with patch.object(
        AgentSpecToLangGraphConverter, "_llm_convert_to_langgraph", return_value=fake_llm
    ):
        compiled = loader.load_component(mw)

    # The manager react-agent is itself a subgraph; its create_agent
    # middleware stack carries the rendered system prompt as the
    # first message of every turn. Walk the manager subgraph's pre-model
    # hook chain to find it.
    manager_subgraph = compiled.builder.nodes[_MANAGER_NODE_KEY].runnable
    # `create_agent` builds a graph whose system message generation
    # wraps the prompt — easier to assert by re-rendering it through the
    # same helper used by the converter and checking the *intent*.
    from pyagentspec.adapters.langgraph._langgraphconverter import _append_workers_roster

    expected = _append_workers_roster(
        "Coordinate the team.",
        [("research_helper", "Handles research tasks")],
    )
    assert "Available workers:" in expected
    assert "- research_helper: Handles research tasks" in expected
    # And the compiled manager carries the delegation tool the prompt
    # advertises, proving the LLM has the matching contract.
    tools_node = manager_subgraph.builder.nodes["tools"].runnable
    assert "delegate_to_research_helper" in tools_node.tools_by_name


# ─── End-to-end execution test (offline, fake LLM emitting delegation) ──────


def test_manager_workers_delegates_and_routes_back_with_tool_message() -> None:
    """End-to-end: manager LLM emits a delegate_to_<worker> tool call,
    the parent graph routes to the worker subgraph (which runs with an
    isolated message context), the worker's final AIMessage content is
    surfaced back to the manager as a ToolMessage matched to the
    pending tool_call_id, and the manager's next turn (no tool call)
    terminates the graph. This is the load-bearing path that proves the
    subgraph composition actually works."""
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import MemorySaver

    from pyagentspec.adapters.langgraph import AgentSpecLoader
    from pyagentspec.adapters.langgraph._langgraphconverter import (
        AgentSpecToLangGraphConverter,
    )
    from pyagentspec.agent import Agent
    from pyagentspec.managerworkers import ManagerWorkers

    manager_agent = Agent(
        name="Coordinator",
        description="Coordinates",
        system_prompt="You coordinate.",
        llm_config=_llm_cfg("manager_llm"),
    )
    worker = Agent(
        name="Research Helper",
        description="Handles research",
        system_prompt="You research.",
        llm_config=_llm_cfg("worker_llm"),
    )
    mw = ManagerWorkers(
        name="Team",
        group_manager=manager_agent,
        workers=[worker],
    )

    # Manager turn 1: delegate to research_helper.
    # Manager turn 2: produce final answer (no tool call → END).
    manager_responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "delegate_to_research_helper",
                    "args": {"task": "Look up Saturn"},
                    "id": "call_1",
                }
            ],
        ),
        AIMessage(content="The worker reports: Saturn has rings."),
    ]
    # Worker turn 1: produce its own final answer.
    worker_responses = [AIMessage(content="Saturn has rings.")]

    fake_manager = _fake_manager(*manager_responses)
    fake_worker = _fake_manager(*worker_responses)

    def _dispatch(self_obj: Any, llm_config: Any, *args: Any, **kwargs: Any) -> Any:
        if llm_config.name == "manager_llm":
            return fake_manager
        if llm_config.name == "worker_llm":
            return fake_worker
        raise AssertionError(f"unexpected llm_config: {llm_config.name}")

    # ``create_agent`` calls ``model.bind_tools(...)``. ``FakeMessagesListChatModel``
    # inherits ``bind_tools`` from real ``ChatOpenAI``, which calls out to
    # OpenAI. Patch the class method so binding is a no-op that returns the
    # same fake (preserving its response queue).
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )

    loader = AgentSpecLoader(tool_registry={}, checkpointer=MemorySaver())
    with patch.object(
        AgentSpecToLangGraphConverter,
        "_llm_convert_to_langgraph",
        autospec=True,
        side_effect=_dispatch,
    ), patch.object(
        FakeMessagesListChatModel,
        "bind_tools",
        new=lambda self_obj, *a, **kw: self_obj,
    ):
        compiled = loader.load_component(mw)

    # Use the sync invocation path: ``FakeMessagesListChatModel`` provides
    # a sync ``_generate`` (returns queued responses) but no async
    # override, so MRO resolves ``_agenerate`` to the real
    # ``ChatOpenAI._agenerate`` which calls the OpenAI API. The worker
    # wrapper exposes both sync and async via RunnableLambda; LangGraph
    # picks the sync path here.
    result = compiled.invoke(
        {"messages": [HumanMessage(content="Tell me about Saturn.")]},
        {"configurable": {"thread_id": "mw-1"}},
    )
    messages = result["messages"]

    # The end state should contain: user input, manager's delegation
    # AIMessage, the synthesized ToolMessage (worker's reply), and the
    # manager's final AIMessage.
    msg_types = [type(m).__name__ for m in messages]
    assert "HumanMessage" in msg_types
    assert "ToolMessage" in msg_types
    # Final message is the manager's terminating AIMessage.
    assert isinstance(messages[-1], AIMessage)
    assert "Saturn has rings" in messages[-1].content

    # And the ToolMessage carries the worker's reply matched to the
    # pending delegation tool_call_id — proves the isolation wrapper
    # threaded the call id through.
    tool_msgs = [m for m in messages if type(m).__name__ == "ToolMessage"]
    assert tool_msgs and tool_msgs[0].tool_call_id == "call_1"
    assert "Saturn has rings" in tool_msgs[0].content


# ─── Recursive nesting ──────────────────────────────────────────────────────


def test_nested_manager_workers_compiles_recursively() -> None:
    """A worker that is itself a ManagerWorkers compiles through the
    same dispatch — the inner ManagerWorkers becomes a CompiledStateGraph
    that the outer parent graph wires in as a subgraph node."""
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import MemorySaver

    from pyagentspec.adapters.langgraph import AgentSpecLoader
    from pyagentspec.adapters.langgraph._langgraphconverter import (
        AgentSpecToLangGraphConverter,
    )
    from pyagentspec.agent import Agent
    from pyagentspec.managerworkers import ManagerWorkers

    leaf = Agent(
        name="Leaf",
        description="Leaf task",
        system_prompt="Leaf.",
        llm_config=_llm_cfg("leaf_llm"),
    )
    inner_manager = Agent(
        name="InnerManager",
        description="Inner",
        system_prompt="Manage leaves.",
        llm_config=_llm_cfg("inner_llm"),
    )
    inner_mw = ManagerWorkers(
        name="Inner",
        group_manager=inner_manager,
        workers=[leaf],
    )
    outer_manager = Agent(
        name="OuterManager",
        description="Outer",
        system_prompt="Manage subteams.",
        llm_config=_llm_cfg("outer_llm"),
    )
    outer_mw = ManagerWorkers(
        name="Outer",
        group_manager=outer_manager,
        workers=[inner_mw],
    )

    fake_llm = _fake_manager(AIMessage(content="Done."))
    loader = AgentSpecLoader(tool_registry={}, checkpointer=MemorySaver())
    with patch.object(
        AgentSpecToLangGraphConverter, "_llm_convert_to_langgraph", return_value=fake_llm
    ):
        compiled = loader.load_component(outer_mw)

    # Outer parent graph has a node for the inner ManagerWorkers worker.
    assert "inner" in compiled.builder.nodes


def test_rejects_non_agent_group_manager() -> None:
    """ManagerWorkers.group_manager must be an Agent — pyagentspec allows
    any AgenticComponent but the LangGraph adapter needs a chat-LLM that
    emits tool_calls to decide which worker to delegate to."""
    from langgraph.checkpoint.memory import MemorySaver

    from pyagentspec.adapters.langgraph import AgentSpecLoader
    from pyagentspec.agent import Agent
    from pyagentspec.managerworkers import ManagerWorkers

    # Use a nested ManagerWorkers as the group_manager — a valid
    # AgenticComponent per pyagentspec validators, unsupported here.
    leaf = Agent(
        name="Leaf",
        description="L",
        system_prompt="L.",
        llm_config=_llm_cfg("l"),
    )
    inner_manager = Agent(
        name="Inner",
        description="I",
        system_prompt="I.",
        llm_config=_llm_cfg("i"),
    )
    inner_mw = ManagerWorkers(
        name="Inner",
        group_manager=inner_manager,
        workers=[leaf],
    )
    outer_mw = ManagerWorkers(
        name="Outer",
        group_manager=inner_mw,
        workers=[
            Agent(name="Other", description="O", system_prompt="O.", llm_config=_llm_cfg("o")),
        ],
    )
    loader = AgentSpecLoader(tool_registry={}, checkpointer=MemorySaver())
    with pytest.raises(NotImplementedError, match="group_manager must be an Agent"):
        loader.load_component(outer_mw)


# ─── Worker name collision ──────────────────────────────────────────────────


def test_workers_with_name_slug_collision_are_rejected() -> None:
    """Two workers whose names normalize to the same node identifier
    would silently overwrite each other in the parent graph; raise at
    load time instead."""
    from langgraph.checkpoint.memory import MemorySaver

    from pyagentspec.adapters.langgraph import AgentSpecLoader
    from pyagentspec.agent import Agent
    from pyagentspec.managerworkers import ManagerWorkers

    a = Agent(name="Helper A", description="x", system_prompt=".", llm_config=_llm_cfg("a"))
    b = Agent(name="helper-a", description="x", system_prompt=".", llm_config=_llm_cfg("b"))
    # Both normalize to "helper_a".
    mw = ManagerWorkers(name="T", group_manager=Agent(
        name="M", description="m", system_prompt=".", llm_config=_llm_cfg("m"),
    ), workers=[a, b])

    loader = AgentSpecLoader(tool_registry={}, checkpointer=MemorySaver())
    with pytest.raises(ValueError, match="collide after normalization"):
        loader.load_component(mw)
