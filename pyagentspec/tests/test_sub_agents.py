# Copyright © 2025 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""Tests for the Agent.sub_agents field introduced in the sub-agents RFC."""

import pytest

from pyagentspec.agent import Agent
from pyagentspec.llms import OpenAiConfig
from pyagentspec.serialization import AgentSpecSerializer
from pyagentspec.serialization.deserializer import AgentSpecDeserializer
from pyagentspec.versioning import AgentSpecVersionEnum


@pytest.fixture
def llm_config():
    return OpenAiConfig(name="default", model_id="gpt-4o")


@pytest.fixture
def simple_agent(llm_config):
    return Agent(
        name="parent",
        system_prompt="You are a helpful parent agent.",
        llm_config=llm_config,
    )


@pytest.fixture
def child_agent(llm_config):
    return Agent(
        name="child",
        description="A specialist child agent.",
        system_prompt="You are a specialist.",
        llm_config=llm_config,
    )


# ---------------------------------------------------------------------------
# Basic field tests
# ---------------------------------------------------------------------------


def test_agent_with_no_sub_agents_has_empty_list(simple_agent):
    assert simple_agent.sub_agents == []


def test_agent_with_sub_agents_can_be_created(llm_config, child_agent):
    parent = Agent(
        name="parent",
        system_prompt="Delegate to child.",
        llm_config=llm_config,
        sub_agents=[child_agent],
    )
    assert len(parent.sub_agents) == 1
    assert parent.sub_agents[0].name == "child"


def test_agent_with_multiple_sub_agents(llm_config):
    sub1 = Agent(name="sub1", system_prompt="Sub 1", llm_config=llm_config)
    sub2 = Agent(name="sub2", system_prompt="Sub 2", llm_config=llm_config)
    parent = Agent(
        name="parent",
        system_prompt="Parent",
        llm_config=llm_config,
        sub_agents=[sub1, sub2],
    )
    assert len(parent.sub_agents) == 2


# ---------------------------------------------------------------------------
# Versioning tests
# ---------------------------------------------------------------------------


def test_agent_without_sub_agents_does_not_require_v26_2_0(simple_agent):
    # An agent with no sub_agents should be serializable at v25_4_1
    serializer = AgentSpecSerializer()
    yaml_str = serializer.to_yaml(simple_agent, agentspec_version=AgentSpecVersionEnum.v25_4_1)
    assert "agentspec_version: 25.4.1" in yaml_str


def test_agent_with_sub_agents_requires_v26_2_0(llm_config, child_agent):
    parent = Agent(
        name="parent",
        system_prompt="Delegate.",
        llm_config=llm_config,
        sub_agents=[child_agent],
    )
    min_ver = parent._infer_min_agentspec_version_from_configuration()
    assert min_ver >= AgentSpecVersionEnum.v26_2_0


def test_agent_with_sub_agents_cannot_serialize_below_v26_2_0(llm_config, child_agent):
    parent = Agent(
        name="parent",
        system_prompt="Delegate.",
        llm_config=llm_config,
        sub_agents=[child_agent],
    )
    serializer = AgentSpecSerializer()
    with pytest.raises(ValueError, match="Invalid agentspec_version"):
        serializer.to_yaml(parent, agentspec_version=AgentSpecVersionEnum.v26_1_0)


# ---------------------------------------------------------------------------
# Serialization round-trip tests
# ---------------------------------------------------------------------------


def test_agent_with_sub_agents_serializes_and_deserializes(llm_config, child_agent):
    parent = Agent(
        name="parent",
        system_prompt="Delegate to child.",
        llm_config=llm_config,
        sub_agents=[child_agent],
    )
    serializer = AgentSpecSerializer()
    yaml_str = serializer.to_yaml(parent)
    assert "sub_agents" in yaml_str
    assert "child" in yaml_str

    deserialized = AgentSpecDeserializer().from_yaml(yaml_str)
    assert isinstance(deserialized, Agent)
    assert len(deserialized.sub_agents) == 1
    assert deserialized.sub_agents[0].name == "child"


def test_agent_without_sub_agents_round_trips_cleanly(simple_agent):
    serializer = AgentSpecSerializer()
    yaml_str = serializer.to_yaml(simple_agent)
    # sub_agents should not appear for an agent with an empty list
    assert "sub_agents" not in yaml_str

    deserialized = AgentSpecDeserializer().from_yaml(yaml_str)
    assert isinstance(deserialized, Agent)
    assert deserialized.sub_agents == []


def test_nested_sub_agents_round_trip(llm_config):
    grandchild = Agent(name="grandchild", system_prompt="Grandchild", llm_config=llm_config)
    child = Agent(
        name="child",
        system_prompt="Child",
        llm_config=llm_config,
        sub_agents=[grandchild],
    )
    parent = Agent(
        name="parent",
        system_prompt="Parent",
        llm_config=llm_config,
        sub_agents=[child],
    )

    serializer = AgentSpecSerializer()
    yaml_str = serializer.to_yaml(parent)

    deserialized = AgentSpecDeserializer().from_yaml(yaml_str)
    assert isinstance(deserialized, Agent)
    assert len(deserialized.sub_agents) == 1
    child_deserialized = deserialized.sub_agents[0]
    assert child_deserialized.name == "child"
    assert isinstance(child_deserialized, Agent)
    assert len(child_deserialized.sub_agents) == 1
    assert child_deserialized.sub_agents[0].name == "grandchild"


# ---------------------------------------------------------------------------
# Validation: unique names
# ---------------------------------------------------------------------------


def test_duplicate_sub_agent_names_raise_validation_error(llm_config):
    sub1 = Agent(name="dup", system_prompt="First", llm_config=llm_config)
    sub2 = Agent(name="dup", system_prompt="Second", llm_config=llm_config)
    with pytest.raises(Exception, match="Duplicate name"):
        Agent(
            name="parent",
            system_prompt="Parent",
            llm_config=llm_config,
            sub_agents=[sub1, sub2],
        )


# ---------------------------------------------------------------------------
# Validation: cycle detection
# ---------------------------------------------------------------------------


def test_direct_cycle_raises_validation_error(llm_config):
    """An agent whose sub_agent list contains itself should be rejected."""
    # We cannot construct the cycle directly at init time because it would
    # require the parent to exist before creation. We instead test by patching
    # sub_agents after construction and then triggering model validation.
    parent = Agent(name="self_ref", system_prompt="Hello", llm_config=llm_config)

    # Manually inject the cycle to simulate what would happen if construction
    # were attempted. We use model_validate to trigger validators.
    with pytest.raises(Exception, match="[Cc]ycle"):
        Agent.model_validate(
            {
                "name": "self_ref",
                "system_prompt": "Hello",
                "llm_config": parent.llm_config,
                "sub_agents": [parent],
            }
        )


def test_transitive_cycle_raises_validation_error(llm_config):
    """A transitive cycle (A -> B -> A) should be detected."""
    # Build A and B independently first
    agent_a = Agent(name="agent_a", system_prompt="A", llm_config=llm_config)
    agent_b = Agent(name="agent_b", system_prompt="B", llm_config=llm_config, sub_agents=[agent_a])

    # Now agent_a has agent_b in its sub_agents, creating a cycle: a -> b -> a
    with pytest.raises(Exception, match="[Cc]ycle"):
        Agent.model_validate(
            {
                "name": "agent_a",
                "system_prompt": "A",
                "llm_config": agent_a.llm_config,
                "sub_agents": [agent_b],
            }
        )


def test_shared_sub_agent_no_cycle(llm_config):
    """A sub-agent shared across multiple parents should NOT trigger a cycle error."""
    shared = Agent(name="shared", system_prompt="Shared", llm_config=llm_config)
    parent1 = Agent(
        name="parent1",
        system_prompt="Parent 1",
        llm_config=llm_config,
        sub_agents=[shared],
    )
    parent2 = Agent(
        name="parent2",
        system_prompt="Parent 2",
        llm_config=llm_config,
        sub_agents=[shared],
    )
    # No exception should be raised
    assert parent1.sub_agents[0].name == "shared"
    assert parent2.sub_agents[0].name == "shared"
