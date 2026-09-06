/**
 * Swarm tests for the LangGraph adapter.
 *
 * Mirrors the offline-able Swarm behaviors of the Python suite: loading an
 * Agent Spec Swarm compiles a `@langchain/langgraph-swarm` graph with one node
 * per participating agent, `transfer_to_<agent>` handoff tools are injected
 * per relationship direction, a handoff round trip runs with fake LLMs, and
 * the two conversion errors (HandoffMode NEVER, non-Agent participant) carry
 * the Python message text. All tests run offline.
 */
import { describe, expect, it } from "vitest";
import { AIMessage, HumanMessage, type BaseMessage } from "@langchain/core/messages";
import { MemorySaver, START } from "@langchain/langgraph";
import {
  createAgent as createAgentSpecAgent,
  createManagerWorkers,
  createSwarm,
  HandoffMode,
} from "../../../src/index.js";
import type { Agent, Swarm } from "../../../src/index.js";
import { AgentSpecLoader } from "../../../src/adapters/langgraph/agentspec-loader.js";
import {
  FakeLlmAgentSpecLoader,
  makeLlmConfig,
  threadConfig,
  toolCallMessage,
  type FakeLlmResponses,
} from "./test-helpers.js";

/** Structural surface of a compiled swarm graph used by these tests. */
interface CompiledSwarmLike {
  lg_is_pregel?: boolean;
  name?: string;
  builder: {
    nodes: Record<string, { runnable?: unknown } | undefined>;
    branches: Record<string, unknown>;
    channels: Record<string, unknown>;
  };
  invoke(input: unknown, config?: unknown): Promise<Record<string, unknown>>;
}

/** A minimal swarm participant with a named LLM config. */
function swarmAgent(opts: {
  name: string;
  llmName: string;
  systemPrompt?: string;
}): Agent {
  return createAgentSpecAgent({
    name: opts.name,
    llmConfig: makeLlmConfig({ name: opts.llmName }),
    systemPrompt: opts.systemPrompt ?? ".",
  });
}

/** Compile a Swarm offline, answering each LLM config with a queued fake. */
async function loadWithFakeLlms(
  swarm: Swarm,
  responses: FakeLlmResponses,
): Promise<{ graph: CompiledSwarmLike; loader: FakeLlmAgentSpecLoader }> {
  const loader = new FakeLlmAgentSpecLoader(responses, {
    checkpointer: new MemorySaver(),
  });
  const graph = (await loader.loadComponent(swarm)) as CompiledSwarmLike;
  return { graph, loader };
}

/** Tool names registered on the `tools` node of one swarm agent's subgraph. */
function agentToolNames(graph: CompiledSwarmLike, agentName: string): string[] {
  const agentGraph = graph.builder.nodes[agentName]?.runnable as
    | CompiledSwarmLike
    | undefined;
  expect(agentGraph).toBeDefined();
  const toolsNode = agentGraph!.builder.nodes["tools"]?.runnable as
    | { tools?: Array<{ name?: unknown }> }
    | undefined;
  return (toolsNode?.tools ?? []).map((registered) => String(registered.name));
}

function twoAgentSwarm(overrides?: {
  relationships?: [Record<string, unknown>, Record<string, unknown>][];
  handoff?: HandoffMode;
}): { swarm: Swarm; alice: Agent; bob: Agent } {
  const alice = swarmAgent({
    name: "alice",
    llmName: "alice_llm",
    systemPrompt: "You are Alice. Hand off to bob when needed.",
  });
  const bob = swarmAgent({
    name: "bob",
    llmName: "bob_llm",
    systemPrompt: "You are Bob.",
  });
  const swarm = createSwarm({
    name: "SwarmTeam",
    firstAgent: alice,
    relationships: overrides?.relationships ?? [
      [alice, bob],
      [bob, alice],
    ],
    ...(overrides?.handoff !== undefined ? { handoff: overrides.handoff } : {}),
  });
  return { swarm, alice, bob };
}

describe("swarm loading", () => {
  it("compiles to a swarm graph with one node per participating agent", async () => {
    const { swarm } = twoAgentSwarm();
    const { graph } = await loadWithFakeLlms(swarm, {
      alice_llm: [],
      bob_llm: [],
    });

    expect(graph.lg_is_pregel).toBe(true);
    expect(graph.name).toBe("SwarmTeam");
    const nodeNames = Object.keys(graph.builder.nodes);
    expect(nodeNames).toContain("alice");
    expect(nodeNames).toContain("bob");
    // The swarm state tracks the active agent and routes off START.
    expect(Object.keys(graph.builder.channels)).toContain("messages");
    expect(Object.keys(graph.builder.channels)).toContain("activeAgent");
    expect(graph.builder.branches[START]).toBeTruthy();
  });

  it("injects a transfer_to_<agent> handoff tool per relationship", async () => {
    const { swarm } = twoAgentSwarm();
    const { graph } = await loadWithFakeLlms(swarm, {
      alice_llm: [],
      bob_llm: [],
    });

    expect(agentToolNames(graph, "alice")).toContain("transfer_to_bob");
    expect(agentToolNames(graph, "bob")).toContain("transfer_to_alice");
    // Handoff tools follow the relationship direction only.
    expect(agentToolNames(graph, "alice")).not.toContain("transfer_to_alice");
    expect(agentToolNames(graph, "bob")).not.toContain("transfer_to_bob");
  });

  it("injects no handoff tool against the relationship direction", async () => {
    const alice = swarmAgent({ name: "alice", llmName: "alice_llm" });
    const bob = swarmAgent({ name: "bob", llmName: "bob_llm" });
    const swarm = createSwarm({
      name: "OneWaySwarm",
      firstAgent: alice,
      relationships: [[alice, bob]],
    });
    const { graph } = await loadWithFakeLlms(swarm, {
      alice_llm: [],
      bob_llm: [],
    });

    expect(agentToolNames(graph, "alice")).toContain("transfer_to_bob");
    expect(agentToolNames(graph, "bob")).toEqual([]);
  });
});

describe("swarm execution", () => {
  it("hands off between agents through the injected tools", async () => {
    const { swarm } = twoAgentSwarm();
    const { graph, loader } = await loadWithFakeLlms(swarm, {
      alice_llm: [toolCallMessage("transfer_to_bob", {}, "handoff_1")],
      bob_llm: [new AIMessage("Hi, this is Bob.")],
    });

    const result = await graph.invoke(
      { messages: [new HumanMessage("hello")] },
      threadConfig("swarm-1"),
    );

    const messages = result["messages"] as BaseMessage[];
    const finalMessage = messages[messages.length - 1]!;
    expect(finalMessage.getType()).toBe("ai");
    expect(finalMessage.content).toBe("Hi, this is Bob.");
    expect(result["activeAgent"]).toBe("bob");
    // Bob's model actually ran; Alice's ran exactly once (the handoff turn).
    expect(loader.getFakeModel("bob_llm").calls.length).toBe(1);
    expect(loader.getFakeModel("alice_llm").calls.length).toBe(1);
  });
});

describe("swarm conversion errors", () => {
  it("rejects HandoffMode NEVER with the Python message", async () => {
    const { swarm } = twoAgentSwarm({ handoff: HandoffMode.NEVER });
    const loader = new AgentSpecLoader();
    await expect(loader.loadComponent(swarm)).rejects.toThrow(
      "Handoff mode NEVER is not supported for conversion in LangGraph adapter",
    );
  });

  it("rejects non-Agent participants", async () => {
    const alice = swarmAgent({ name: "alice", llmName: "alice_llm" });
    const managerWorkers = createManagerWorkers({
      name: "SubTeam",
      groupManager: swarmAgent({ name: "m", llmName: "m_llm" }),
      workers: [swarmAgent({ name: "w", llmName: "w_llm" })],
    });
    const swarm = createSwarm({
      name: "BadSwarm",
      firstAgent: alice,
      relationships: [[managerWorkers, alice]],
    });

    const loader = new AgentSpecLoader();
    await expect(loader.loadComponent(swarm)).rejects.toThrow(
      /Only Agents are supported as part of a Swarm/,
    );
  });
});
