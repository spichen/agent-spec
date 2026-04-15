/**
 * Tests for Agent.subAgents - first-class sub-agent delegation.
 */
import { describe, it, expect } from "vitest";
import {
  createAgent,
  createOpenAiCompatibleConfig,
  AgentSpecSerializer,
  AgentSpecVersion,
} from "../../src/index.js";

function makeLlmConfig() {
  return createOpenAiCompatibleConfig({
    name: "test-llm",
    url: "http://localhost:8000",
    modelId: "gpt-4",
  });
}

function makeAgent(name: string) {
  return createAgent({
    name,
    llmConfig: makeLlmConfig(),
    systemPrompt: `You are ${name}.`,
  });
}

// ---------------------------------------------------------------------------
// Basic field tests
// ---------------------------------------------------------------------------

describe("Agent.subAgents — basic", () => {
  it("defaults to an empty array", () => {
    const agent = makeAgent("parent");
    expect(agent.subAgents).toEqual([]);
  });

  it("accepts a list of sub-agents", () => {
    const child = makeAgent("child");
    const parent = createAgent({
      name: "parent",
      llmConfig: makeLlmConfig(),
      systemPrompt: "Delegate.",
      subAgents: [child],
    });
    expect(parent.subAgents).toHaveLength(1);
    expect((parent.subAgents[0] as { name: string }).name).toBe("child");
  });

  it("accepts multiple sub-agents", () => {
    const child1 = makeAgent("child1");
    const child2 = makeAgent("child2");
    const parent = createAgent({
      name: "parent",
      llmConfig: makeLlmConfig(),
      systemPrompt: "Delegate.",
      subAgents: [child1, child2],
    });
    expect(parent.subAgents).toHaveLength(2);
  });

  it("is frozen", () => {
    const parent = createAgent({
      name: "parent",
      llmConfig: makeLlmConfig(),
      systemPrompt: "Hello",
      subAgents: [makeAgent("child")],
    });
    expect(Object.isFrozen(parent)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Validation: unique names
// ---------------------------------------------------------------------------

describe("Agent.subAgents — unique name validation", () => {
  it("throws when two sub-agents share the same name", () => {
    const dup1 = makeAgent("dup");
    const dup2 = makeAgent("dup");
    expect(() =>
      createAgent({
        name: "parent",
        llmConfig: makeLlmConfig(),
        systemPrompt: "Delegate.",
        subAgents: [dup1, dup2],
      }),
    ).toThrow(/[Dd]uplicate/);
  });

  it("does not throw when sub-agents have distinct names", () => {
    const a = makeAgent("alpha");
    const b = makeAgent("beta");
    expect(() =>
      createAgent({
        name: "parent",
        llmConfig: makeLlmConfig(),
        systemPrompt: "Delegate.",
        subAgents: [a, b],
      }),
    ).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// Validation: cycle detection
// ---------------------------------------------------------------------------

describe("Agent.subAgents — cycle detection", () => {
  it("throws when agent contains itself as a sub-agent (direct cycle)", () => {
    const parent = makeAgent("self-ref");
    expect(() =>
      createAgent({
        name: "self-ref",
        llmConfig: makeLlmConfig(),
        systemPrompt: "Hello",
        subAgents: [parent],
      }),
    ).toThrow(/[Cc]ycle/);
  });

  it("throws on a transitive cycle (A -> B -> A)", () => {
    const agentA = makeAgent("agent-a");
    const agentB = createAgent({
      name: "agent-b",
      llmConfig: makeLlmConfig(),
      systemPrompt: "B",
      subAgents: [agentA],
    });
    // Creating agent-a with agent-b as sub-agent would form a cycle
    expect(() =>
      createAgent({
        name: "agent-a",
        llmConfig: makeLlmConfig(),
        systemPrompt: "A",
        subAgents: [agentB],
      }),
    ).toThrow(/[Cc]ycle/);
  });

  it("does not throw when a sub-agent is shared across multiple parents (no cycle)", () => {
    const shared = makeAgent("shared");
    expect(() => {
      createAgent({
        name: "parent1",
        llmConfig: makeLlmConfig(),
        systemPrompt: "P1",
        subAgents: [shared],
      });
      createAgent({
        name: "parent2",
        llmConfig: makeLlmConfig(),
        systemPrompt: "P2",
        subAgents: [shared],
      });
    }).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// Serialization / version-gate tests
// ---------------------------------------------------------------------------

describe("Agent.subAgents — serialization", () => {
  const serializer = new AgentSpecSerializer();

  it("serializes subAgents at current version", () => {
    const child = makeAgent("child");
    const parent = createAgent({
      name: "parent",
      llmConfig: makeLlmConfig(),
      systemPrompt: "Delegate.",
      subAgents: [child],
    });
    const json = serializer.toJson(parent) as string;
    const dict = JSON.parse(json) as Record<string, unknown>;
    expect("sub_agents" in dict).toBe(true);
    const subAgents = dict["sub_agents"] as Record<string, unknown>[];
    expect(subAgents).toHaveLength(1);
    expect(subAgents[0]!["name"]).toBe("child");
  });

  it("excludes sub_agents when serializing to v26.1.0", () => {
    const child = makeAgent("child");
    const parent = createAgent({
      name: "parent",
      llmConfig: makeLlmConfig(),
      systemPrompt: "Delegate.",
      subAgents: [child],
    });
    const json = serializer.toJson(parent, {
      agentspecVersion: AgentSpecVersion.V26_1_0,
    }) as string;
    const dict = JSON.parse(json) as Record<string, unknown>;
    expect("sub_agents" in dict).toBe(false);
  });

  it("includes sub_agents when serializing to v26.2.0", () => {
    const child = makeAgent("child");
    const parent = createAgent({
      name: "parent",
      llmConfig: makeLlmConfig(),
      systemPrompt: "Delegate.",
      subAgents: [child],
    });
    const json = serializer.toJson(parent, {
      agentspecVersion: AgentSpecVersion.V26_2_0,
    }) as string;
    const dict = JSON.parse(json) as Record<string, unknown>;
    expect("sub_agents" in dict).toBe(true);
  });

  it("serializes empty subAgents as an empty array", () => {
    const parent = makeAgent("parent");
    const json = serializer.toJson(parent) as string;
    const dict = JSON.parse(json) as Record<string, unknown>;
    expect(dict["sub_agents"]).toEqual([]);
  });

  it("serializes nested sub-agents", () => {
    const grandchild = makeAgent("grandchild");
    const child = createAgent({
      name: "child",
      llmConfig: makeLlmConfig(),
      systemPrompt: "Child",
      subAgents: [grandchild],
    });
    const parent = createAgent({
      name: "parent",
      llmConfig: makeLlmConfig(),
      systemPrompt: "Parent",
      subAgents: [child],
    });

    const json = serializer.toJson(parent) as string;
    const dict = JSON.parse(json) as Record<string, unknown>;
    const subAgents = dict["sub_agents"] as Record<string, unknown>[];
    expect(subAgents).toHaveLength(1);
    const childDict = subAgents[0]!;
    expect(childDict["name"]).toBe("child");
    const grandchildren = childDict["sub_agents"] as Record<string, unknown>[];
    expect(grandchildren).toHaveLength(1);
    expect(grandchildren[0]!["name"]).toBe("grandchild");
  });
});
