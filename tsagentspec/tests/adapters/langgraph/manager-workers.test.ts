/**
 * ManagerWorkers tests for the LangGraph adapter.
 *
 * Mirrors `pyagentspec/tests/adapters/langgraph/test_managerworkers.py`:
 * worker-name slug normalization + collision rejection, the workers roster
 * appended to the manager prompt (exact format), the hierarchical graph
 * topology, per-worker `__delegate_to__` tools on the manager, the manager
 * router (END / one Send per delegation / non-worker suffixes ignored), the
 * full delegation round trip with per-agent fake LLMs (single and multiple
 * delegations per turn), nested ManagerWorkers, non-Agent group manager
 * rejection, ManagerWorkers as a flow AgentNode step, and the
 * `DELEGATE_TOOL_PREFIX` / `isDelegationToolName` contract. All tests run
 * offline. Since the Python suite reaches its private helpers directly
 * (`_safe_node_name`, `_append_workers_roster`, `_make_manager_router`), the
 * equivalents here are asserted through the compiled graph: node names, the
 * system message received by the fake manager model, and the conditional-edge
 * branch function registered on `__manager__`.
 */
import { describe, expect, it } from "vitest";
import {
  AIMessage,
  HumanMessage,
  ToolMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import { END, MemorySaver, START, Send } from "@langchain/langgraph";
import {
  createAgent as createAgentSpecAgent,
  createAgentNode,
  createControlFlowEdge,
  createDataFlowEdge,
  createEndNode,
  createFlow,
  createManagerWorkers,
  createStartNode,
  stringProperty,
} from "../../../src/index.js";
import type { Agent, Flow, ManagerWorkers, Property } from "../../../src/index.js";
import { AgentSpecLoader } from "../../../src/adapters/langgraph/agentspec-loader.js";
import {
  DELEGATE_TOOL_PREFIX,
  isDelegationToolName,
} from "../../../src/adapters/langgraph/manager-workers.js";
import {
  FakeLlmAgentSpecLoader,
  makeLlmConfig,
  threadConfig,
  type FakeLlmResponses,
} from "./test-helpers.js";

const MANAGER_NODE_KEY = "__manager__";
const DELEGATE_TASK_KEY = "__delegate_task__";
const DELEGATE_CALL_ID_KEY = "__delegate_tool_call_id__";

/** Conditional-edge branch shape on a StateGraph builder. */
interface BranchLike {
  path: { func: (state: Record<string, unknown>) => unknown };
  ends?: Record<string, string>;
}

/** Structural surface of a compiled LangGraph used by these tests. */
interface CompiledGraphLike {
  lg_is_pregel?: boolean;
  builder: {
    nodes: Record<string, { runnable?: unknown } | undefined>;
    edges: Set<[string, string]>;
    branches: Record<string, Record<string, BranchLike> | undefined>;
  };
  invoke(input: unknown, config?: unknown): Promise<Record<string, unknown>>;
}

/** Python's `_agent` fixture: a minimal Agent with a named LLM config. */
function mwAgent(opts: {
  name: string;
  llmName: string;
  description?: string;
  systemPrompt?: string;
  outputs?: Property[];
  id?: string;
}): Agent {
  return createAgentSpecAgent({
    name: opts.name,
    llmConfig: makeLlmConfig({ name: opts.llmName }),
    systemPrompt: opts.systemPrompt ?? ".",
    ...(opts.description !== undefined ? { description: opts.description } : {}),
    ...(opts.outputs !== undefined ? { outputs: opts.outputs } : {}),
    ...(opts.id !== undefined ? { id: opts.id } : {}),
  });
}

/**
 * Python's `_load_with_fake_llms`: compile a ManagerWorkers offline, answering
 * each LLM config (keyed by `llmConfig.name`) with a queued fake.
 */
async function loadWithFakeLlms(
  managerWorkers: ManagerWorkers | Flow,
  responses: FakeLlmResponses,
): Promise<{ graph: CompiledGraphLike; loader: FakeLlmAgentSpecLoader }> {
  const loader = new FakeLlmAgentSpecLoader(responses, {
    checkpointer: new MemorySaver(),
  });
  const graph = (await loader.loadComponent(managerWorkers)) as CompiledGraphLike;
  return { graph, loader };
}

/** Tool names registered on the `tools` node of a compiled react agent. */
function toolNamesOf(agentGraph: unknown): string[] {
  const nodes = (agentGraph as CompiledGraphLike).builder.nodes;
  const toolsNode = nodes["tools"]?.runnable as
    | { tools?: Array<{ name?: unknown }> }
    | undefined;
  return (toolsNode?.tools ?? []).map((registered) => String(registered.name));
}

/** The [from, to] pairs of a compiled graph's plain edges. */
function edgePairs(graph: CompiledGraphLike): string[] {
  return [...graph.builder.edges].map(([from, to]) => `${from}->${to}`);
}

/** The router function registered as the manager's conditional edge. */
function managerRouter(
  graph: CompiledGraphLike,
): (state: Record<string, unknown>) => unknown {
  const branchMap = graph.builder.branches[MANAGER_NODE_KEY];
  expect(branchMap).toBeDefined();
  const branch = branchMap!["condition"];
  expect(branch).toBeDefined();
  return branch!.path.func;
}

/** The messages of an invoke result. */
function messagesOf(result: Record<string, unknown>): BaseMessage[] {
  return result["messages"] as BaseMessage[];
}

/**
 * The plain text of a message: langchain JS may deliver the system prompt as
 * a `[{type: "text", text}]` content-blocks array instead of a plain string.
 */
function textOf(message: BaseMessage): string {
  const content = message.content as unknown;
  if (typeof content === "string") {
    return content;
  }
  return (content as Array<{ type?: string; text?: string }>)
    .map((block) => block.text ?? "")
    .join("");
}

/** The ToolMessages of an invoke result, in order. */
function toolMessagesOf(result: Record<string, unknown>): ToolMessage[] {
  return messagesOf(result).filter(
    (message): message is ToolMessage => message.getType() === "tool",
  );
}

/** The ResearchTeam fixture of the Python topology test. */
function researchTeamSpec(): ManagerWorkers {
  return createManagerWorkers({
    name: "ResearchTeam",
    groupManager: mwAgent({
      name: "Coordinator",
      llmName: "manager_llm",
      systemPrompt: "Coordinate the team.",
    }),
    workers: [
      mwAgent({
        name: "Research Helper",
        llmName: "worker_a_llm",
        description: "Handles research",
      }),
      mwAgent({
        name: "Drafter",
        llmName: "worker_b_llm",
        description: "Drafts text",
      }),
    ],
  });
}

describe("delegation tool name contract", () => {
  it("exposes the __delegate_to__ prefix constant", () => {
    expect(DELEGATE_TOOL_PREFIX).toBe("__delegate_to__");
  });

  it("matches only names starting with the synthetic prefix", () => {
    expect(isDelegationToolName("__delegate_to__research_helper")).toBe(true);
    expect(isDelegationToolName("get_weather")).toBe(false);
    // A real tool plausibly named delegate_to_<something> is not a delegation.
    expect(isDelegationToolName("delegate_to_someone")).toBe(false);
    // Nor is one merely containing the prefix mid-name.
    expect(isDelegationToolName("please__delegate_to__someone")).toBe(false);
    expect(isDelegationToolName(undefined)).toBe(false);
    expect(isDelegationToolName(null)).toBe(false);
    expect(isDelegationToolName(123)).toBe(false);
  });
});

describe("worker node name normalization", () => {
  it("slugifies worker names, falling back to the id and then a constant", async () => {
    const spec = createManagerWorkers({
      name: "T",
      groupManager: mwAgent({ name: "M", llmName: "manager_llm" }),
      workers: [
        mwAgent({ name: "Research Helper", llmName: "w1_llm" }),
        mwAgent({ name: "My-Worker!! v2", llmName: "w2_llm" }),
        // Name slugifies to empty -> normalized id.
        mwAgent({ name: "!!!", llmName: "w3_llm", id: "sub-1" }),
        // Name and id both slugify to empty -> constant fallback.
        mwAgent({ name: "!!!", llmName: "w4_llm", id: "???" }),
      ],
    });
    const { graph } = await loadWithFakeLlms(spec, []);

    const nodeNames = Object.keys(graph.builder.nodes);
    expect(nodeNames).toContain("research_helper");
    expect(nodeNames).toContain("my_worker_v2");
    expect(nodeNames).toContain("sub_1");
    expect(nodeNames).toContain("worker");
  });

  it("rejects workers whose names collide after normalization", async () => {
    // Both worker names normalize to "helper_a"; they would silently
    // overwrite each other in the parent graph.
    const spec = createManagerWorkers({
      name: "T",
      groupManager: mwAgent({ name: "M", llmName: "m_llm" }),
      workers: [
        mwAgent({ name: "Helper A", llmName: "a_llm" }),
        mwAgent({ name: "helper-a", llmName: "b_llm" }),
      ],
    });
    const loader = new AgentSpecLoader();
    await expect(loader.loadComponent(spec)).rejects.toThrow(
      /collide after normalization/,
    );
  });
});

describe("workers roster", () => {
  it("appends one roster line per worker to the manager system prompt", async () => {
    const { graph, loader } = await loadWithFakeLlms(researchTeamSpec(), {
      manager_llm: [new AIMessage("Done.")],
      worker_a_llm: [],
      worker_b_llm: [],
    });
    await graph.invoke(
      { messages: [new HumanMessage("hi")] },
      threadConfig("mw-roster"),
    );

    const managerModel = loader.getFakeModel("manager_llm");
    const firstCall = managerModel.calls[0]!;
    expect(firstCall[0]!.getType()).toBe("system");
    expect(textOf(firstCall[0]!)).toBe(
      "Coordinate the team.\n\n" +
        "Available workers:\n" +
        "- research_helper: Handles research\n" +
        "- drafter: Drafts text",
    );
  });

  it("flattens multiline descriptions so the one-line-per-worker shape survives", async () => {
    const spec = createManagerWorkers({
      name: "T",
      groupManager: mwAgent({
        name: "M",
        llmName: "manager_llm",
        systemPrompt: "Coordinate.",
      }),
      workers: [
        mwAgent({
          name: "helper",
          llmName: "worker_llm",
          description: "First line\nsecond line\n  third  line  ",
        }),
      ],
    });
    const { graph, loader } = await loadWithFakeLlms(spec, {
      manager_llm: [new AIMessage("Done.")],
      worker_llm: [],
    });
    await graph.invoke(
      { messages: [new HumanMessage("hi")] },
      threadConfig("mw-roster-flat"),
    );

    const firstCall = loader.getFakeModel("manager_llm").calls[0]!;
    expect(textOf(firstCall[0]!)).toBe(
      "Coordinate.\n\nAvailable workers:\n- helper: First line second line third line",
    );
  });

  it("renders the roster alone when the manager prompt is empty", async () => {
    const spec = createManagerWorkers({
      name: "T",
      groupManager: mwAgent({
        name: "M",
        llmName: "manager_llm",
        systemPrompt: "",
      }),
      workers: [
        mwAgent({ name: "helper", llmName: "worker_llm", description: "Helps" }),
      ],
    });
    const { graph, loader } = await loadWithFakeLlms(spec, {
      manager_llm: [new AIMessage("Done.")],
      worker_llm: [],
    });
    await graph.invoke(
      { messages: [new HumanMessage("hi")] },
      threadConfig("mw-roster-empty"),
    );

    const firstCall = loader.getFakeModel("manager_llm").calls[0]!;
    expect(textOf(firstCall[0]!)).toBe("Available workers:\n- helper: Helps");
  });
});

describe("graph topology", () => {
  it("compiles to a hierarchical graph: START -> manager, workers loop back", async () => {
    const { graph } = await loadWithFakeLlms(researchTeamSpec(), [
      new AIMessage("Done."),
    ]);

    const nodeNames = Object.keys(graph.builder.nodes);
    expect(nodeNames).toContain(MANAGER_NODE_KEY);
    expect(nodeNames).toContain("research_helper");
    expect(nodeNames).toContain("drafter");

    // START -> manager; every worker -> manager (loop).
    const edges = edgePairs(graph);
    expect(edges).toContain(`${START}->${MANAGER_NODE_KEY}`);
    expect(edges).toContain(`research_helper->${MANAGER_NODE_KEY}`);
    expect(edges).toContain(`drafter->${MANAGER_NODE_KEY}`);

    // Manager -> worker is a conditional edge whose path map covers every
    // worker plus END.
    const branchMap = graph.builder.branches[MANAGER_NODE_KEY];
    expect(branchMap).toBeTruthy();
    expect(branchMap!["condition"]!.ends).toEqual({
      research_helper: "research_helper",
      drafter: "drafter",
      [END]: END,
    });
  });

  it("registers a __delegate_to__ tool per worker on the manager react agent", async () => {
    const spec = createManagerWorkers({
      name: "Team",
      groupManager: mwAgent({ name: "Coordinator", llmName: "manager_llm" }),
      workers: [
        mwAgent({
          name: "Research Helper",
          llmName: "worker_llm",
          description: "Handles research tasks",
        }),
      ],
    });
    const { graph } = await loadWithFakeLlms(spec, [new AIMessage("Done.")]);

    // The delegation tool the roster advertises is registered on the manager
    // react-agent's tools node, so the LLM has the matching contract.
    const managerSubgraph = graph.builder.nodes[MANAGER_NODE_KEY]!.runnable;
    expect(toolNamesOf(managerSubgraph)).toContain(
      "__delegate_to__research_helper",
    );
  });
});

describe("manager router", () => {
  async function compileRouter(): Promise<
    (state: Record<string, unknown>) => unknown
  > {
    const { graph } = await loadWithFakeLlms(researchTeamSpec(), []);
    return managerRouter(graph);
  }

  it("returns END when the manager did not delegate", async () => {
    const route = await compileRouter();
    const notDelegating = new AIMessage({ content: "Done.", tool_calls: [] });
    expect(route({ messages: [notDelegating] })).toBe(END);
    expect(route({ messages: [] })).toBe(END);
  });

  it("fans out one Send per delegation, carrying the task and tool_call_id", async () => {
    const route = await compileRouter();
    const message = new AIMessage({
      content: "",
      tool_calls: [
        { name: "some_other_tool", args: {}, id: "c0", type: "tool_call" },
        {
          name: "__delegate_to__drafter",
          args: { task: "x" },
          id: "c1",
          type: "tool_call",
        },
        {
          name: "__delegate_to__research_helper",
          args: { task: "y" },
          id: "c2",
          type: "tool_call",
        },
      ],
    });

    const sends = route({ messages: [message] }) as Send[];
    // Every delegation gets its own Send carrying the task and the
    // tool_call_id its reply must answer. The non-delegation tool call
    // already ran inside the manager's react loop and is ignored by routing.
    expect(Array.isArray(sends)).toBe(true);
    expect(sends.every((send) => send instanceof Send)).toBe(true);
    expect(sends.map((send) => send.node)).toEqual([
      "drafter",
      "research_helper",
    ]);
    expect(
      sends.map((send) => (send.args as Record<string, unknown>)[DELEGATE_TASK_KEY]),
    ).toEqual(["x", "y"]);
    expect(
      sends.map(
        (send) => (send.args as Record<string, unknown>)[DELEGATE_CALL_ID_KEY],
      ),
    ).toEqual(["c1", "c2"]);
  });

  it("ignores a prefixed tool call whose suffix is not a worker", async () => {
    // A tool call that merely looks like a delegation must not be routed: its
    // suffix is not a worker node, so a Send would target a non-existing
    // node. It already ran as a plain tool inside the react loop.
    const route = await compileRouter();
    const message = new AIMessage({
      content: "",
      tool_calls: [
        {
          name: "__delegate_to__nobody",
          args: { task: "x" },
          id: "c1",
          type: "tool_call",
        },
      ],
    });
    expect(route({ messages: [message] })).toBe(END);
  });
});

describe("delegation round trip", () => {
  it("delegates, routes the worker answer back as a ToolMessage, and terminates", async () => {
    const spec = createManagerWorkers({
      name: "Team",
      groupManager: mwAgent({
        name: "Coordinator",
        llmName: "manager_llm",
        systemPrompt: "You coordinate.",
      }),
      workers: [
        mwAgent({
          name: "Research Helper",
          llmName: "worker_llm",
          description: "Handles research",
        }),
      ],
    });

    // Manager turn 1: delegate. Manager turn 2: final answer (no tool call
    // -> END).
    const { graph } = await loadWithFakeLlms(spec, {
      manager_llm: [
        new AIMessage({
          content: "",
          tool_calls: [
            {
              name: "__delegate_to__research_helper",
              args: { task: "Look up Saturn" },
              id: "call_1",
              type: "tool_call",
            },
          ],
        }),
        new AIMessage("The worker reports: Saturn has rings."),
      ],
      worker_llm: [new AIMessage("Saturn has rings.")],
    });

    const result = await graph.invoke(
      { messages: [new HumanMessage("Tell me about Saturn.")] },
      threadConfig("mw-1"),
    );

    const messages = messagesOf(result);
    const finalMessage = messages[messages.length - 1]!;
    expect(finalMessage.getType()).toBe("ai");
    expect(String(finalMessage.content)).toContain("Saturn has rings");

    // The worker ran in an isolated message context and its answer came back
    // as a ToolMessage matched to the pending tool_call_id.
    const toolMessages = toolMessagesOf(result);
    expect(toolMessages.length).toBeGreaterThan(0);
    expect(toolMessages[0]!.tool_call_id).toBe("call_1");
    expect(String(toolMessages[0]!.content)).toContain("Saturn has rings");
  });

  it("answers every delegation of a single manager turn with its own ToolMessage", async () => {
    const spec = createManagerWorkers({
      name: "Team",
      groupManager: mwAgent({
        name: "Coordinator",
        llmName: "manager_llm",
        systemPrompt: "You coordinate.",
      }),
      workers: [
        mwAgent({
          name: "Sub Agent",
          llmName: "worker_llm",
          description: "Writes poems",
        }),
      ],
    });

    // Turn 1: three delegations to the same worker in one AIMessage.
    // Turn 2: terminate. An unanswered delegation would be an invalid
    // tool-call/result sequence the manager would hallucinate around.
    const { graph } = await loadWithFakeLlms(spec, {
      manager_llm: [
        new AIMessage({
          content: "",
          tool_calls: [
            {
              name: "__delegate_to__sub_agent",
              args: { task: "Spanish poem" },
              id: "call_1",
              type: "tool_call",
            },
            {
              name: "__delegate_to__sub_agent",
              args: { task: "French poem" },
              id: "call_2",
              type: "tool_call",
            },
            {
              name: "__delegate_to__sub_agent",
              args: { task: "German poem" },
              id: "call_3",
              type: "tool_call",
            },
          ],
        }),
        new AIMessage("Here are your three poems."),
      ],
      worker_llm: [1, 2, 3, 4, 5].map(
        (index) => new AIMessage(`poem #${index}`),
      ),
    });

    const result = await graph.invoke(
      { messages: [new HumanMessage("Write 3 poems via sub-agents.")] },
      threadConfig("mw-multi"),
    );

    const toolMessages = toolMessagesOf(result);
    const answeredCallIds = toolMessages
      .map((message) => message.tool_call_id)
      .sort();
    expect(answeredCallIds).toEqual(["call_1", "call_2", "call_3"]);
    expect(
      toolMessages.every((message) =>
        String(message.content).startsWith("poem #"),
      ),
    ).toBe(true);
  });
});

describe("nested ManagerWorkers", () => {
  it("compiles a ManagerWorkers worker recursively as a subgraph node", async () => {
    const innerSpec = createManagerWorkers({
      name: "Inner",
      groupManager: mwAgent({
        name: "InnerManager",
        llmName: "inner_llm",
        systemPrompt: "Manage leaves.",
      }),
      workers: [
        mwAgent({ name: "Leaf", llmName: "leaf_llm", description: "Leaf task" }),
      ],
    });
    const outerSpec = createManagerWorkers({
      name: "Outer",
      groupManager: mwAgent({
        name: "OuterManager",
        llmName: "outer_llm",
        systemPrompt: "Manage subteams.",
      }),
      workers: [innerSpec],
    });

    const { graph } = await loadWithFakeLlms(outerSpec, [
      new AIMessage("Done."),
    ]);
    expect(Object.keys(graph.builder.nodes)).toContain("inner");
  });

  it("rejects a non-Agent group manager", async () => {
    // A nested ManagerWorkers as groupManager is valid per the Agent Spec
    // validators, but the adapter needs a chat-LLM emitting tool_calls to
    // route on.
    const innerSpec = createManagerWorkers({
      name: "Inner",
      groupManager: mwAgent({ name: "Inner", llmName: "i_llm" }),
      workers: [mwAgent({ name: "Leaf", llmName: "l_llm" })],
    });
    const outerSpec = createManagerWorkers({
      name: "Outer",
      groupManager: innerSpec,
      workers: [mwAgent({ name: "Other", llmName: "o_llm" })],
    });

    const loader = new AgentSpecLoader();
    await expect(loader.loadComponent(outerSpec)).rejects.toThrow(
      /group_manager must be an Agent/,
    );
  });
});

describe("ManagerWorkers as a flow step", () => {
  /**
   * Python's `_flow_with_manager_workers_step`: start -> AgentNode(MW) -> end,
   * with data edges resolving the manager's `joke` input and every output.
   */
  function flowWithManagerWorkersStep(outputs: Property[]): Flow {
    const joke = stringProperty({ title: "joke" });
    const manager = mwAgent({
      name: "manager",
      llmName: "manager_llm",
      systemPrompt: "Translate the following to Arabic:\n\n{{joke}}",
      outputs,
    });
    const worker = mwAgent({
      name: "worker",
      llmName: "worker_llm",
      systemPrompt: "You translate.",
    });
    const managerWorkers = createManagerWorkers({
      name: "translator",
      groupManager: manager,
      workers: [worker],
    });
    // The ManagerWorkers exposes the group manager's prompt placeholders as
    // inputs, so the AgentNode declares an input port the DataFlowEdge below
    // can resolve.
    expect((managerWorkers.inputs ?? []).map((input) => input.title)).toEqual([
      "joke",
    ]);

    const managerNode = createAgentNode({
      name: "manager_node",
      agent: managerWorkers,
    });
    const startNode = createStartNode({
      name: "start",
      inputs: [joke],
      outputs: [joke],
    });
    const endNode = createEndNode({ name: "end", inputs: outputs, outputs });
    return createFlow({
      name: "flow",
      startNode,
      nodes: [startNode, managerNode, endNode],
      controlFlowConnections: [
        createControlFlowEdge({
          name: "start_to_node",
          fromNode: startNode,
          toNode: managerNode,
        }),
        createControlFlowEdge({
          name: "node_to_end",
          fromNode: managerNode,
          toNode: endNode,
        }),
      ],
      dataFlowConnections: [
        createDataFlowEdge({
          name: "joke_edge",
          sourceNode: startNode,
          sourceOutput: joke.title,
          destinationNode: managerNode,
          destinationInput: joke.title,
        }),
        ...outputs.map((output) =>
          createDataFlowEdge({
            name: `${output.title}_edge`,
            sourceNode: managerNode,
            sourceOutput: output.title,
            destinationNode: endNode,
            destinationInput: output.title,
          }),
        ),
      ],
      outputs,
    });
  }

  it("runs as a flow step with data-edge inputs and a single string output", async () => {
    const flow = flowWithManagerWorkersStep([
      stringProperty({ title: "translated" }),
    ]);
    // The final message has no tool_calls -> the manager routes to END
    // without delegating; its answer is the node's single string output.
    const { graph } = await loadWithFakeLlms(flow, [new AIMessage("لماذا...")]);

    const result = await graph.invoke(
      {
        inputs: { joke: "Why did the car..." },
        messages: [{ role: "user", content: "" }],
      },
      threadConfig("managerworkers-node"),
    );

    expect((result["outputs"] as Record<string, unknown>)["translated"]).toBe(
      "لماذا...",
    );
  });

  it("rejects unsupported output shapes at conversion time, not mid-run", async () => {
    const flow = flowWithManagerWorkersStep([
      stringProperty({ title: "translated" }),
      stringProperty({ title: "notes" }),
    ]);
    const loader = new FakeLlmAgentSpecLoader([], {
      checkpointer: new MemorySaver(),
    });
    await expect(loader.loadComponent(flow)).rejects.toThrow(
      /single string output/,
    );
  });
});
