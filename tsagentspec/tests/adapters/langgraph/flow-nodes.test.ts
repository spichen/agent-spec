/**
 * Per-node flow execution tests for the LangGraph adapter.
 *
 * Mirrors `pyagentspec/tests/adapters/langgraph/flows/` (test_toolnode,
 * test_branchingnode, test_llmnode, test_agentnode, test_flownode,
 * test_catchexceptionode, test_inputmessagenode, test_outputmessagenode,
 * test_mapnode, test_apinode) with fake chat models and a mocked fetch so
 * every test runs offline.
 *
 * Documented divergences exercised here:
 * - Tuples do not exist in JS: arrays map positionally onto multiple declared
 *   tool-node outputs (Python restricts positional mapping to tuples).
 * - The TS SDK ApiNode has no `urlAllowList` field yet, so the Python
 *   allow-list rejection test has no TS equivalent (the adapter always calls
 *   the validation helper with `undefined`).
 *
 * Note on node construction: the Python SDK infers the missing IO side of
 * Start/End nodes, so Python specs always carry both sides on the wire; the
 * TS factories default the missing side to `[]`, so these tests pass both
 * sides explicitly, matching the serialized wire format.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { AIMessage, type BaseMessage } from "@langchain/core/messages";
import { Command, MemorySaver } from "@langchain/langgraph";
import {
  createAgentNode,
  createBranchingNode,
  createCatchExceptionNode,
  createClientTool,
  createControlFlowEdge,
  createDataFlowEdge,
  createEndNode,
  createFlow,
  createFlowNode,
  createInputMessageNode,
  createLlmNode,
  createMapNode,
  createOutputMessageNode,
  createServerTool,
  createStartNode,
  createToolNode,
  createApiNode,
  integerProperty,
  listProperty,
  nullProperty,
  numberProperty,
  objectProperty,
  stringProperty,
  unionProperty,
  type ComponentWithIO,
  type EndNode,
  type Flow,
  type LlmConfig,
  type Property,
  type ServerTool,
  type StartNode,
} from "../../../src/index.js";
import { DEFAULT_HTTP_REQUEST_TIMEOUT_MS } from "../../../src/adapters/common/tools-common.js";
import { AgentSpecLoader } from "../../../src/adapters/langgraph/agentspec-loader.js";
import { AgentSpecToLangGraphConverter } from "../../../src/adapters/langgraph/langgraph-converter.js";
import {
  FakeToolCallingChatModel,
  getInterrupts,
  installMockFetch,
  loadWithFakeLlm,
  makeAgent,
  makeLlmConfig,
  threadConfig,
  toolCallMessage,
  type MockFetchController,
} from "./test-helpers.js";

/** The invocable surface of a compiled flow graph. */
interface CompiledFlow {
  invoke(
    input: unknown,
    config?: unknown,
  ): Promise<Record<string, unknown>>;
}

/** A StartNode declaring the same properties as inputs and outputs. */
function ioStartNode(name: string, props: Property[] = []): StartNode {
  return createStartNode({ name, inputs: props, outputs: props });
}

/** An EndNode declaring the same properties as inputs and outputs. */
function ioEndNode(
  name: string,
  props: Property[] = [],
  branchName?: string,
): EndNode {
  return createEndNode({
    name,
    inputs: props,
    outputs: props,
    ...(branchName !== undefined ? { branchName } : {}),
  });
}

function ctrl(
  fromNode: Record<string, unknown>,
  toNode: Record<string, unknown>,
  fromBranch?: string,
) {
  return createControlFlowEdge({
    name: `${String(fromNode["name"])}_to_${String(toNode["name"])}${
      fromBranch !== undefined ? `_${fromBranch}` : ""
    }`,
    fromNode,
    toNode,
    ...(fromBranch !== undefined ? { fromBranch } : {}),
  });
}

function dataEdge(
  sourceNode: ComponentWithIO,
  destinationNode: ComponentWithIO,
  sourceOutput: string,
  destinationInput: string = sourceOutput,
) {
  return createDataFlowEdge({
    name: `${sourceNode.name}.${sourceOutput}_to_${destinationNode.name}.${destinationInput}`,
    sourceNode,
    sourceOutput,
    destinationNode,
    destinationInput,
  });
}

function outputsOf(result: Record<string, unknown>): Record<string, unknown> {
  return result["outputs"] as Record<string, unknown>;
}

function messagesOf(result: Record<string, unknown>): BaseMessage[] {
  return result["messages"] as BaseMessage[];
}

function detailsOf(result: Record<string, unknown>): Record<string, unknown> {
  return result["node_execution_details"] as Record<string, unknown>;
}

async function loadFlow(
  flow: Flow,
  options?: {
    toolRegistry?: Record<string, unknown>;
    checkpointer?: MemorySaver;
  },
): Promise<CompiledFlow> {
  const loader = new AgentSpecLoader({
    ...(options?.toolRegistry !== undefined
      ? { toolRegistry: options.toolRegistry }
      : {}),
    ...(options?.checkpointer !== undefined
      ? { checkpointer: options.checkpointer }
      : {}),
  });
  return (await loader.loadComponent(flow)) as CompiledFlow;
}

describe("ToolNode output-mapping matrix", () => {
  /** Python's `_build_flow_with_client_tool`: start -> ClientTool -> end. */
  function buildClientToolFlow(
    inputProp: Property,
    outputProps: Property[],
  ): Flow {
    const start = ioStartNode("start", [inputProp]);
    const clientTool = createClientTool({
      name: "echo_tool",
      description: "Client-side tool used for testing",
      inputs: [inputProp],
      outputs: outputProps,
    });
    const toolNode = createToolNode({ name: "tool", tool: clientTool });
    const end = ioEndNode("end", outputProps);
    return createFlow({
      name: "tool_output_flow",
      startNode: start,
      nodes: [start, toolNode, end],
      controlFlowConnections: [ctrl(start, toolNode), ctrl(toolNode, end)],
      dataFlowConnections: [
        dataEdge(start, toolNode, inputProp.title),
        ...outputProps.map((prop) => dataEdge(toolNode, end, prop.title)),
      ],
    });
  }

  /** Interrupt at the client tool, then resume with the given payload. */
  async function runFlowAndResume(
    flow: Flow,
    resumePayload: unknown,
  ): Promise<Record<string, unknown>> {
    const graph = await loadFlow(flow, { checkpointer: new MemorySaver() });
    const config = threadConfig("t");
    const first = await graph.invoke(
      { inputs: { [flow.inputs![0]!.title]: 123 } },
      config,
    );
    expect(getInterrupts(first)).toHaveLength(1);
    const resumed = await graph.invoke(
      new Command({ resume: resumePayload }),
      config,
    );
    return outputsOf(resumed);
  }

  it("interrupts with the client_tool_request payload and resumes with the value", async () => {
    const inputProp = numberProperty({ title: "input" });
    const outputProp = numberProperty({ title: "input_square" });
    const squareTool = createClientTool({
      name: "square_tool",
      description: "Computes the square of a number",
      inputs: [inputProp],
      outputs: [outputProp],
    });
    const start = ioStartNode("subflow_start", [inputProp]);
    const toolNode = createToolNode({ name: "square_tool_node", tool: squareTool });
    const end = ioEndNode("subflow_end", [outputProp]);
    const flow = createFlow({
      name: "Square number flow",
      startNode: start,
      nodes: [start, toolNode, end],
      controlFlowConnections: [ctrl(start, toolNode), ctrl(toolNode, end)],
      dataFlowConnections: [
        dataEdge(start, toolNode, "input"),
        dataEdge(toolNode, end, "input_square"),
      ],
    });

    const graph = await loadFlow(flow, { checkpointer: new MemorySaver() });
    const config = threadConfig("1");
    const first = await graph.invoke({ inputs: { input: 4 } }, config);
    const interrupts = getInterrupts(first);
    expect(interrupts).toHaveLength(1);
    expect(interrupts[0]!.value).toEqual({
      type: "client_tool_request",
      name: "square_tool",
      description: "Computes the square of a number",
      inputs: { args: [], kwargs: { input: 4 } },
    });

    const resumed = await graph.invoke(new Command({ resume: 16 }), config);
    expect(outputsOf(resumed)["input_square"]).toBe(16);
  });

  it("single ObjectProperty output wraps a multi-key dict under the declared key", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      objectProperty({ title: "out_dict", properties: {} }),
    ]);
    const outputs = await runFlowAndResume(flow, { a: 1, b: 2 });
    expect(outputs).toEqual({ out_dict: { a: 1, b: 2 } });
  });

  it("single ObjectProperty output wraps a single-key dict under the declared key", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      objectProperty({ title: "out_dict", properties: {} }),
    ]);
    const outputs = await runFlowAndResume(flow, { a: 1 });
    expect(outputs).toEqual({ out_dict: { a: 1 } });
  });

  it("single output uses a dict keyed by the declared title as-is", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      objectProperty({ title: "out_dict", properties: {} }),
    ]);
    const outputs = await runFlowAndResume(flow, { out_dict: 1 });
    expect(outputs).toEqual({ out_dict: 1 });
  });

  it("scalar output passes through under the declared key", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      stringProperty({ title: "out_string" }),
    ]);
    const outputs = await runFlowAndResume(flow, "value");
    expect(outputs).toEqual({ out_string: "value" });
  });

  it("multiple outputs filter the dict and defaults fill missing keys", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      numberProperty({ title: "a" }),
      numberProperty({ title: "b", default: 0 }),
    ]);
    const outputs = await runFlowAndResume(flow, { a: 5 });
    expect(outputs).toEqual({ a: 5, b: 0 });
  });

  it("list output maps to a single declared list output", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      listProperty({ title: "out", itemType: numberProperty({ title: "item" }) }),
    ]);
    const outputs = await runFlowAndResume(flow, [1, 2, 3]);
    expect(outputs).toEqual({ out: [1, 2, 3] });
  });

  it("scalar output maps to a single declared number output", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      numberProperty({ title: "out_number" }),
    ]);
    const outputs = await runFlowAndResume(flow, 42);
    expect(outputs).toEqual({ out_number: 42 });
  });

  it("array output onto a single declared string output is stringified", async () => {
    // Python (tuple payload) stringifies via json.dumps -> "[1, 2]"; the TS
    // cast mirrors json.dumps formatting (", " separator, not "[1,2]").
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      stringProperty({ title: "out" }),
    ]);
    const outputs = await runFlowAndResume(flow, [1, 2]);
    expect(outputs).toEqual({ out: "[1, 2]" });
  });

  it("array output shorter than the declared outputs raises like Python", async () => {
    // Python raises IndexError instead of silently mapping undefined.
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      numberProperty({ title: "a" }),
      stringProperty({ title: "b" }),
    ]);
    await expect(runFlowAndResume(flow, [7])).rejects.toThrow(
      "Tool node `tool` returned 1 value(s) but declares 2 outputs; " +
        "no value for output `b`.",
    );
  });

  it("content-block list shorter than the declared outputs raises like Python", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      stringProperty({ title: "text_out" }),
      stringProperty({ title: "image_out" }),
    ]);
    await expect(
      runFlowAndResume(flow, [{ type: "text", text: "hello" }]),
    ).rejects.toThrow(
      "Tool node `tool` returned 1 content block(s) but declares 2 outputs; " +
        "no value for output `image_out`.",
    );
  });

  it("array output maps positionally onto multiple outputs", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      numberProperty({ title: "a" }),
      stringProperty({ title: "b" }),
    ]);
    const outputs = await runFlowAndResume(flow, [7, "ok"]);
    expect(outputs).toEqual({ a: 7, b: "ok" });
  });

  it("mixed array output maps positionally onto number/object/array outputs", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      numberProperty({ title: "num" }),
      objectProperty({ title: "obj", properties: {} }),
      listProperty({ title: "array", itemType: numberProperty({ title: "elem" }) }),
    ]);
    const outputs = await runFlowAndResume(flow, [7, { key: "val" }, [1]]);
    expect(outputs).toEqual({ num: 7, obj: { key: "val" }, array: [1] });
  });

  it("MCP content-block lists extract payloads positionally", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      stringProperty({ title: "text_out" }),
      stringProperty({ title: "image_out" }),
    ]);
    const outputs = await runFlowAndResume(flow, [
      { type: "text", text: "hello" },
      { type: "image", base64: "imgdata" },
    ]);
    expect(outputs).toEqual({ text_out: "hello", image_out: "imgdata" });
  });
});

describe("BranchingNode", () => {
  it("routes on the mapping, falls back to the default branch, and keeps defaults on untaken paths", async () => {
    const customInput = stringProperty({ title: "custom_input" });
    const outputA = stringProperty({ title: "output_a", default: "no_value" });
    const outputB = stringProperty({ title: "output_b", default: "no_value" });
    const branchingNode = createBranchingNode({
      name: "branching",
      mapping: { a: "branch_a", b: "branch_b" },
      inputs: [customInput],
    });
    const start = ioStartNode("start", [customInput]);
    const endA = ioEndNode("end_a", [outputA]);
    const endB = ioEndNode("end_b", [outputB]);
    const endDefault = ioEndNode("end_default");

    const flow = createFlow({
      name: "flow",
      startNode: start,
      nodes: [start, branchingNode, endA, endB, endDefault],
      controlFlowConnections: [
        ctrl(start, branchingNode),
        ctrl(branchingNode, endA, "branch_a"),
        ctrl(branchingNode, endB, "branch_b"),
        ctrl(branchingNode, endDefault, "default"),
      ],
      dataFlowConnections: [
        dataEdge(start, branchingNode, "custom_input"),
        dataEdge(start, endB, "custom_input", "output_b"),
        dataEdge(start, endA, "custom_input", "output_a"),
      ],
      outputs: [outputA, outputB],
    });

    const graph = await loadFlow(flow);

    let result = await graph.invoke({ inputs: { custom_input: "a" } });
    expect(outputsOf(result)).toEqual({ output_a: "a", output_b: "no_value" });
    expect(result).toHaveProperty("messages");

    result = await graph.invoke({ inputs: { custom_input: "b" } });
    expect(outputsOf(result)).toEqual({ output_a: "no_value", output_b: "b" });

    result = await graph.invoke({ inputs: { custom_input: "no_match" } });
    expect(outputsOf(result)).toEqual({
      output_a: "no_value",
      output_b: "no_value",
    });
  });

  it("raises the missing-input error when nothing feeds the branching input", async () => {
    const customInput = stringProperty({ title: "custom_input" });
    const branchingNode = createBranchingNode({
      name: "branching",
      mapping: { a: "branch_a" },
      inputs: [customInput],
    });
    const start = ioStartNode("start");
    const endA = ioEndNode("end_a");
    const endDefault = ioEndNode("end_default");
    const flow = createFlow({
      name: "flow",
      startNode: start,
      nodes: [start, branchingNode, endA, endDefault],
      controlFlowConnections: [
        ctrl(start, branchingNode),
        ctrl(branchingNode, endA, "branch_a"),
        ctrl(branchingNode, endDefault, "default"),
      ],
      dataFlowConnections: [],
    });

    const graph = await loadFlow(flow);
    await expect(graph.invoke({ inputs: {} })).rejects.toThrow(
      "Expected node `branching` to have a value for property `custom_input`, but none was found.",
    );
  });
});

/** Duck-typed chat-model fake for LlmNode tests. */
function makeChatModelFake(opts: {
  reply?: string;
  structured?: Record<string, unknown>;
}) {
  const captured = {
    prompts: [] as unknown[],
    structuredSchemas: [] as Record<string, unknown>[],
  };
  const model = {
    invoke: async (input: unknown) => {
      captured.prompts.push(input);
      return new AIMessage(opts.reply ?? "");
    },
    withStructuredOutput: (schema: Record<string, unknown>) => {
      captured.structuredSchemas.push(schema);
      return {
        invoke: async (input: unknown) => {
          captured.prompts.push(input);
          if (opts.structured === undefined) {
            throw new Error("No structured response configured.");
          }
          return opts.structured;
        },
      };
    },
  };
  return { model, captured };
}

describe("LlmNode", () => {
  const nationality = stringProperty({ title: "nationality" });
  const car = stringProperty({ title: "car" });

  function buildLlmFlow(outputs: Property[]): Flow {
    const llmNode = createLlmNode({
      name: "llm_node",
      llmConfig: makeLlmConfig(),
      promptTemplate:
        "Answer in one short sentence. What is the fastest {{nationality}} car?",
      inputs: [nationality],
      outputs,
    });
    const start = ioStartNode("start", [nationality]);
    const end = ioEndNode("end", outputs);
    return createFlow({
      name: "flow",
      startNode: start,
      nodes: [start, llmNode, end],
      controlFlowConnections: [ctrl(start, llmNode), ctrl(llmNode, end)],
      dataFlowConnections: [
        dataEdge(start, llmNode, "nationality"),
        ...outputs.map((prop) => dataEdge(llmNode, end, prop.title)),
      ],
      outputs,
    });
  }

  it("unstructured: a single string output takes the message content of the rendered prompt call", async () => {
    const { model, captured } = makeChatModelFake({ reply: "The Ferrari." });
    const { agent } = await loadWithFakeLlm(buildLlmFlow([car]), () => model);

    const result = await agent.invoke({ inputs: { nationality: "italian" } });
    expect(outputsOf(result)).toEqual({ car: "The Ferrari." });

    // The prompt template was rendered against the node inputs.
    expect(captured.structuredSchemas).toHaveLength(0);
    expect(captured.prompts).toHaveLength(1);
    const promptMessages = captured.prompts[0] as Array<{
      role: string;
      content: string;
    }>;
    expect(promptMessages).toEqual([
      {
        role: "user",
        content:
          "Answer in one short sentence. What is the fastest italian car?",
      },
    ]);
  });

  it("structured: multiple outputs use withStructuredOutput with the built JSON schema", async () => {
    const rating = integerProperty({ title: "rating" });
    const { model, captured } = makeChatModelFake({
      structured: { car: "Ferrari", rating: 9 },
    });
    const { agent } = await loadWithFakeLlm(
      buildLlmFlow([car, rating]),
      () => model,
    );

    const result = await agent.invoke({ inputs: { nationality: "italian" } });
    expect(outputsOf(result)).toEqual({ car: "Ferrari", rating: 9 });

    expect(captured.structuredSchemas).toHaveLength(1);
    expect(captured.structuredSchemas[0]).toEqual({
      title: "structured_output",
      type: "object",
      properties: {
        car: car.jsonSchema,
        rating: rating.jsonSchema,
      },
    });
  });

  it("structured: a flattened single-property result is rewrapped under the declared title", async () => {
    const wrapped = objectProperty({ title: "wrapped", properties: {} });
    const { model } = makeChatModelFake({ structured: { inner: 1 } });
    const { agent } = await loadWithFakeLlm(
      buildLlmFlow([wrapped]),
      () => model,
    );

    const result = await agent.invoke({ inputs: { nationality: "italian" } });
    expect(outputsOf(result)).toEqual({ wrapped: { inner: 1 } });
  });
});

describe("AgentNode in a flow", () => {
  const nationality = stringProperty({ title: "nationality" });
  const car = stringProperty({ title: "car" });

  function buildAgentFlow(): Flow {
    const agentSpec = makeAgent({
      name: "agent",
      systemPrompt: "What is the fastest {{nationality}} car?",
      inputs: [nationality],
      outputs: [car],
    });
    const agentNode = createAgentNode({ name: "agent_node", agent: agentSpec });
    const start = ioStartNode("start", [nationality]);
    const end = ioEndNode("end", [car]);
    return createFlow({
      name: "flow",
      startNode: start,
      nodes: [start, agentNode, end],
      controlFlowConnections: [ctrl(start, agentNode), ctrl(agentNode, end)],
      dataFlowConnections: [
        dataEdge(start, agentNode, "nationality"),
        dataEdge(agentNode, end, "car"),
      ],
      outputs: [car],
    });
  }

  it("renders the system prompt from node inputs and extracts declared outputs", async () => {
    const { agent, loader } = await loadWithFakeLlm(buildAgentFlow(), [
      toolCallMessage("AgentOutputModel", { car: "Ferrari 296" }),
    ]);

    const result = await agent.invoke({ inputs: { nationality: "italian" } });
    expect(outputsOf(result)).toEqual({ car: "Ferrari 296" });

    // The compiled react agent received the RENDERED system prompt (langchain
    // v1 normalizes the prompt into a content-blocks array).
    const fakeModel = loader.getFakeModel();
    const systemMessage = fakeModel.calls[0]![0]!;
    expect(systemMessage.getType()).toBe("system");
    const systemText = JSON.stringify(systemMessage.content);
    expect(systemText).toContain("What is the fastest italian car?");
    // No placeholder survives rendering.
    expect(systemText).not.toContain("{{");
  });

  it("emits the agent's answer as an assistant message when the node declares no outputs", async () => {
    const chatAgent = makeAgent({ name: "chat_agent", systemPrompt: "Say hi." });
    const agentNode = createAgentNode({ name: "agent_node", agent: chatAgent });
    const start = ioStartNode("start");
    const end = ioEndNode("end");
    const flow = createFlow({
      name: "flow",
      startNode: start,
      nodes: [start, agentNode, end],
      controlFlowConnections: [ctrl(start, agentNode), ctrl(agentNode, end)],
    });

    const { agent } = await loadWithFakeLlm(flow, [new AIMessage("Ciao!")]);
    const result = await agent.invoke({ inputs: {} });

    const messages = messagesOf(result);
    expect(messages).toHaveLength(1);
    expect(messages[0]!.getType()).toBe("ai");
    expect(messages[0]!.content).toBe("Ciao!");
    expect(outputsOf(result)).toEqual({});
  });

  it("caches the compiled agent per rendered system prompt across invokes", async () => {
    /** Converter that counts react-agent compilations and injects a fake LLM. */
    class CountingFakeConverter extends AgentSpecToLangGraphConverter {
      compileCount = 0;

      constructor(private readonly model: unknown) {
        super();
      }

      protected override async convertLlmConfig(
        _llmConfig: LlmConfig,
      ): Promise<unknown> {
        return this.model;
      }

      protected override async createReactAgentWithGivenInfo(
        info: unknown,
        context: unknown,
      ): Promise<unknown> {
        this.compileCount += 1;
        return super.createReactAgentWithGivenInfo(
          info as never,
          context as never,
        );
      }
    }

    const fakeModel = new FakeToolCallingChatModel({
      responses: [toolCallMessage("AgentOutputModel", { car: "Ferrari" })],
    });
    const converter = new CountingFakeConverter(fakeModel);
    const graph = (await converter.convert(buildAgentFlow(), {})) as CompiledFlow;

    // Compilation is lazy: nothing is compiled at load time.
    expect(converter.compileCount).toBe(0);

    let result = await graph.invoke({ inputs: { nationality: "italian" } });
    expect(outputsOf(result)["car"]).toBe("Ferrari");
    expect(converter.compileCount).toBe(1);

    // Same rendered prompt: the cached agent is reused.
    result = await graph.invoke({ inputs: { nationality: "italian" } });
    expect(converter.compileCount).toBe(1);

    // A different rendered prompt compiles a new agent.
    result = await graph.invoke({ inputs: { nationality: "french" } });
    expect(outputsOf(result)["car"]).toBe("Ferrari");
    expect(converter.compileCount).toBe(2);
  });
});

describe("FlowNode", () => {
  it("executes the subflow and passes its outputs through", async () => {
    const customProp = stringProperty({ title: "custom_prop" });
    const subStart = ioStartNode("start", [customProp]);
    const subEnd = ioEndNode("end", [customProp]);
    const subflow = createFlow({
      name: "subflow",
      startNode: subStart,
      nodes: [subStart, subEnd],
      controlFlowConnections: [ctrl(subStart, subEnd)],
      dataFlowConnections: [dataEdge(subStart, subEnd, "custom_prop")],
      inputs: [customProp],
      outputs: [customProp],
    });

    const flowNode = createFlowNode({ name: "flow_node", subflow });
    const start = ioStartNode("start", [customProp]);
    const end = ioEndNode("end", [customProp]);
    const flow = createFlow({
      name: "outer",
      startNode: start,
      nodes: [start, flowNode, end],
      controlFlowConnections: [ctrl(start, flowNode), ctrl(flowNode, end)],
      dataFlowConnections: [
        dataEdge(start, flowNode, "custom_prop"),
        dataEdge(flowNode, end, "custom_prop"),
      ],
      inputs: [customProp],
      outputs: [customProp],
    });

    const graph = await loadFlow(flow);
    const result = await graph.invoke({ inputs: { custom_prop: "custom" } });
    expect(result).toHaveProperty("messages");
    expect(outputsOf(result)).toEqual({ custom_prop: "custom" });
  });
});

describe("CatchExceptionNode", () => {
  const inp = integerProperty({ title: "x" });
  const outp = stringProperty({ title: "y", default: "" });

  function makeErrorInfoProperty(): Property {
    return unionProperty({
      title: "error_info",
      anyOf: [
        stringProperty({ title: "error_info" }),
        nullProperty({ title: "error_info" }),
      ],
      default: null,
    });
  }

  function buildToolSubflow(tool: ServerTool, endBranch?: string): Flow {
    const subStart = ioStartNode("sub_start", [inp]);
    const toolNode = createToolNode({ name: `${tool.name}_node`, tool });
    const subEnd = ioEndNode("sub_end", [outp], endBranch);
    return createFlow({
      name: `${tool.name}_subflow`,
      startNode: subStart,
      nodes: [subStart, toolNode, subEnd],
      controlFlowConnections: [ctrl(subStart, toolNode), ctrl(toolNode, subEnd)],
      dataFlowConnections: [
        dataEdge(subStart, toolNode, "x"),
        dataEdge(toolNode, subEnd, "y"),
      ],
      inputs: [inp],
      outputs: [outp],
    });
  }

  it("routes exceptions to the caught_exception_branch with default outputs and caught_exception_info", async () => {
    const flakyTool = createServerTool({
      name: "flaky_tool",
      description: "Raises for negative inputs",
      inputs: [inp],
      outputs: [outp],
    });
    const subflow = buildToolSubflow(flakyTool);
    const catchNode = createCatchExceptionNode({ name: "catch", subflow });
    const errorInfo = makeErrorInfoProperty();
    const start = ioStartNode("start", [inp]);
    const end = ioEndNode("end", [outp]);
    const errorEnd = ioEndNode("error_end", [errorInfo], "ERROR");
    const flow = createFlow({
      name: "outer",
      startNode: start,
      nodes: [start, catchNode, end, errorEnd],
      controlFlowConnections: [
        ctrl(start, catchNode),
        ctrl(catchNode, end),
        ctrl(catchNode, errorEnd, "caught_exception_branch"),
      ],
      dataFlowConnections: [
        dataEdge(start, catchNode, "x"),
        dataEdge(catchNode, end, "y"),
        dataEdge(catchNode, errorEnd, "caught_exception_info", "error_info"),
      ],
      inputs: [inp],
      outputs: [outp, errorInfo],
    });

    const graph = await loadFlow(flow, {
      toolRegistry: {
        flaky_tool: (input: unknown) => {
          const { x } = input as { x: number };
          if (x < 0) {
            throw new Error("x must be non-negative");
          }
          return "ok";
        },
      },
    });

    // Case 1: no exception -> subflow output passes through.
    let result = await graph.invoke({ inputs: { x: 1 } });
    expect(outputsOf(result)["y"]).toBe("ok");
    expect(outputsOf(result)["error_info"]).toBeNull();

    // Case 2: exception -> default output value, ERROR end branch, and the
    // exception message routed through caught_exception_info.
    result = await graph.invoke({ inputs: { x: -1 } });
    expect(outputsOf(result)["y"]).toBe("");
    expect(detailsOf(result)["branch"]).toBe("ERROR");
    const caught = outputsOf(result)["error_info"];
    expect(typeof caught).toBe("string");
    expect(String(caught)).toContain("x must be non-negative");
  });

  it("propagates a custom subflow end branch on success with null exception info", async () => {
    const okTool = createServerTool({
      name: "ok_tool",
      description: "Always returns ok",
      inputs: [inp],
      outputs: [outp],
    });
    const subflow = buildToolSubflow(okTool, "OK");
    const catchNode = createCatchExceptionNode({ name: "catch", subflow });
    const errorInfo = makeErrorInfoProperty();
    const start = ioStartNode("start", [inp]);
    const okEnd = ioEndNode("ok_end", [outp, errorInfo]);
    const otherEnd = ioEndNode("other_end");
    const flow = createFlow({
      name: "outer",
      startNode: start,
      nodes: [start, catchNode, okEnd, otherEnd],
      controlFlowConnections: [
        ctrl(start, catchNode),
        ctrl(catchNode, okEnd, "OK"),
        ctrl(catchNode, otherEnd),
      ],
      dataFlowConnections: [
        dataEdge(start, catchNode, "x"),
        dataEdge(catchNode, okEnd, "y"),
        dataEdge(catchNode, okEnd, "caught_exception_info", "error_info"),
      ],
      inputs: [inp],
      outputs: [outp, errorInfo],
    });

    const graph = await loadFlow(flow, {
      toolRegistry: { ok_tool: () => "ok" },
    });
    const result = await graph.invoke({ inputs: { x: 7 } });
    expect(detailsOf(result)["branch"]).toBe("next");
    expect(outputsOf(result)["y"]).toBe("ok");
    expect(outputsOf(result)["error_info"]).toBeNull();
  });

  it("uses the default next branch on success with null exception info", async () => {
    const okTool = createServerTool({
      name: "ok_tool_default",
      description: "Always returns ok",
      inputs: [inp],
      outputs: [outp],
    });
    const subflow = buildToolSubflow(okTool);
    const catchNode = createCatchExceptionNode({ name: "catch", subflow });
    const errorInfo = makeErrorInfoProperty();
    const start = ioStartNode("start", [inp]);
    const nextEnd = ioEndNode("next_end", [outp, errorInfo]);
    const flow = createFlow({
      name: "outer",
      startNode: start,
      nodes: [start, catchNode, nextEnd],
      controlFlowConnections: [ctrl(start, catchNode), ctrl(catchNode, nextEnd)],
      dataFlowConnections: [
        dataEdge(start, catchNode, "x"),
        dataEdge(catchNode, nextEnd, "y"),
        dataEdge(catchNode, nextEnd, "caught_exception_info", "error_info"),
      ],
      inputs: [inp],
      outputs: [outp, errorInfo],
    });

    const graph = await loadFlow(flow, {
      toolRegistry: { ok_tool_default: () => "ok" },
    });
    const result = await graph.invoke({ inputs: { x: 5 } });
    expect(detailsOf(result)["branch"]).toBe("next");
    expect(outputsOf(result)["y"]).toBe("ok");
    expect(outputsOf(result)["error_info"]).toBeNull();
  });
});

describe("InputMessageNode", () => {
  it("interrupts with an empty payload; the resume value becomes the output and a user message", async () => {
    const customInput = stringProperty({ title: "custom_input" });
    const inputMessageNode = createInputMessageNode({
      name: "input_message",
      outputs: [customInput],
    });
    const start = ioStartNode("start");
    const end = ioEndNode("end", [customInput]);
    const flow = createFlow({
      name: "flow",
      startNode: start,
      nodes: [start, inputMessageNode, end],
      controlFlowConnections: [
        ctrl(start, inputMessageNode),
        ctrl(inputMessageNode, end),
      ],
      dataFlowConnections: [dataEdge(inputMessageNode, end, "custom_input")],
      outputs: [customInput],
    });

    const graph = await loadFlow(flow, { checkpointer: new MemorySaver() });
    const config = threadConfig("1");

    const first = await graph.invoke({}, config);
    const interrupts = getInterrupts(first);
    expect(interrupts).toHaveLength(1);
    expect(interrupts[0]!.value).toBe("");

    const result = await graph.invoke(new Command({ resume: "3" }), config);
    expect(outputsOf(result)).toEqual({ custom_input: "3" });

    const messages = messagesOf(result);
    expect(messages).toHaveLength(1);
    expect(messages[0]!.getType()).toBe("human");
    expect(messages[0]!.content).toBe("3");
  });
});

describe("OutputMessageNode", () => {
  it("emits the rendered template as an assistant message", async () => {
    const customInput = stringProperty({ title: "custom_input" });
    const outputMessageNode = createOutputMessageNode({
      name: "output_message",
      message: "Hey {{custom_input}}",
      inputs: [customInput],
    });
    const start = ioStartNode("start", [customInput]);
    const end = ioEndNode("end");
    const flow = createFlow({
      name: "flow",
      startNode: start,
      nodes: [start, outputMessageNode, end],
      controlFlowConnections: [
        ctrl(start, outputMessageNode),
        ctrl(outputMessageNode, end),
      ],
      dataFlowConnections: [dataEdge(start, outputMessageNode, "custom_input")],
      inputs: [customInput],
    });

    const graph = await loadFlow(flow);
    const result = await graph.invoke({ inputs: { custom_input: "custom" } });

    expect(result).toHaveProperty("outputs");
    const messages = messagesOf(result);
    expect(messages).toHaveLength(1);
    expect(messages[0]!.getType()).toBe("ai");
    expect(messages[0]!.content).toBe("Hey custom");
  });
});

describe("MapNode", () => {
  function buildSquareSubflow(): Flow {
    const xProp = numberProperty({ title: "input" });
    const xSquareProp = numberProperty({ title: "input_square" });
    const squareTool = createServerTool({
      name: "square_tool",
      description: "Computes the square of a number",
      inputs: [xProp],
      outputs: [xSquareProp],
    });
    const subStart = ioStartNode("subflow_start", [xProp]);
    const toolNode = createToolNode({ name: "square_tool_node", tool: squareTool });
    const subEnd = ioEndNode("subflow_end", [xSquareProp]);
    return createFlow({
      name: "Square number flow",
      startNode: subStart,
      nodes: [subStart, toolNode, subEnd],
      controlFlowConnections: [ctrl(subStart, toolNode), ctrl(toolNode, subEnd)],
      dataFlowConnections: [
        dataEdge(subStart, toolNode, "input"),
        dataEdge(toolNode, subEnd, "input_square"),
      ],
    });
  }

  const iteratedInput = unionProperty({
    title: "iterated_input",
    anyOf: [
      numberProperty({ title: "input" }),
      listProperty({ title: "input", itemType: numberProperty({ title: "item" }) }),
    ],
  });
  const collectedSquare = listProperty({
    title: "collected_input_square",
    itemType: numberProperty({ title: "item" }),
  });
  const squareRegistry = {
    square_tool: (input: unknown) => {
      const { input: value } = input as { input: number };
      return value * value;
    },
  };

  it("iterates the subflow over the list input and collects the outputs", async () => {
    const mapNode = createMapNode({
      name: "square_number_map_node",
      subflow: buildSquareSubflow(),
      inputs: [iteratedInput],
      outputs: [collectedSquare],
    });
    const inputList = listProperty({
      title: "input_list",
      itemType: numberProperty({ title: "item" }),
    });
    const start = ioStartNode("outer_start", [inputList]);
    const end = ioEndNode("outer_end", [collectedSquare]);
    const flow = createFlow({
      name: "flow to square all elements of a list",
      startNode: start,
      nodes: [start, mapNode, end],
      controlFlowConnections: [ctrl(start, mapNode), ctrl(mapNode, end)],
      dataFlowConnections: [
        dataEdge(start, mapNode, "input_list", "iterated_input"),
        dataEdge(mapNode, end, "collected_input_square"),
      ],
    });

    const graph = await loadFlow(flow, { toolRegistry: squareRegistry });
    const result = await graph.invoke({ inputs: { input_list: [1, 2, 3, 4] } });
    expect(outputsOf(result)).toEqual({
      collected_input_square: [1, 4, 9, 16],
    });
  });

  it("raises when iterated inputs have different lengths", async () => {
    const aProp = numberProperty({ title: "a" });
    const bProp = numberProperty({ title: "b" });
    const totalProp = numberProperty({ title: "total" });
    const sumTool = createServerTool({
      name: "sum_tool",
      description: "Adds two numbers",
      inputs: [aProp, bProp],
      outputs: [totalProp],
    });
    const subStart = ioStartNode("sum_start", [aProp, bProp]);
    const toolNode = createToolNode({ name: "sum_tool_node", tool: sumTool });
    const subEnd = ioEndNode("sum_end", [totalProp]);
    const sumSubflow = createFlow({
      name: "sum_subflow",
      startNode: subStart,
      nodes: [subStart, toolNode, subEnd],
      controlFlowConnections: [ctrl(subStart, toolNode), ctrl(toolNode, subEnd)],
      dataFlowConnections: [
        dataEdge(subStart, toolNode, "a"),
        dataEdge(subStart, toolNode, "b"),
        dataEdge(toolNode, subEnd, "total"),
      ],
      inputs: [aProp, bProp],
      outputs: [totalProp],
    });

    const iteratedA = unionProperty({
      title: "iterated_a",
      anyOf: [
        numberProperty({ title: "a" }),
        listProperty({ title: "a", itemType: numberProperty({ title: "item" }) }),
      ],
    });
    const iteratedB = unionProperty({
      title: "iterated_b",
      anyOf: [
        numberProperty({ title: "b" }),
        listProperty({ title: "b", itemType: numberProperty({ title: "item" }) }),
      ],
    });
    const collectedTotal = listProperty({
      title: "collected_total",
      itemType: numberProperty({ title: "item" }),
    });
    const mapNode = createMapNode({
      name: "sum_map_node",
      subflow: sumSubflow,
      inputs: [iteratedA, iteratedB],
      outputs: [collectedTotal],
    });

    const listA = listProperty({
      title: "list_a",
      itemType: numberProperty({ title: "item" }),
    });
    const listB = listProperty({
      title: "list_b",
      itemType: numberProperty({ title: "item" }),
    });
    const start = ioStartNode("outer_start", [listA, listB]);
    const end = ioEndNode("outer_end", [collectedTotal]);
    const flow = createFlow({
      name: "sum_map_flow",
      startNode: start,
      nodes: [start, mapNode, end],
      controlFlowConnections: [ctrl(start, mapNode), ctrl(mapNode, end)],
      dataFlowConnections: [
        dataEdge(start, mapNode, "list_a", "iterated_a"),
        dataEdge(start, mapNode, "list_b", "iterated_b"),
        dataEdge(mapNode, end, "collected_total"),
      ],
    });

    const graph = await loadFlow(flow, {
      toolRegistry: {
        sum_tool: (input: unknown) => {
          const { a, b } = input as { a: number; b: number };
          return a + b;
        },
      },
    });
    await expect(
      graph.invoke({ inputs: { list_a: [1, 2], list_b: [10, 20, 30] } }),
    ).rejects.toThrow("Found inputs to iterate with different sizes");
  });

  it("raises naming the input when an iterated input has no length at runtime", async () => {
    // The converter selects iterated_input statically (list-typed schema),
    // but the runtime value is a scalar: the error names the node and the
    // offending input instead of reusing the size-mismatch text.
    const mapNode = createMapNode({
      name: "square_number_map_node",
      subflow: buildSquareSubflow(),
      inputs: [iteratedInput],
      outputs: [collectedSquare],
    });
    const inputList = listProperty({
      title: "input_list",
      itemType: numberProperty({ title: "item" }),
    });
    const start = ioStartNode("outer_start", [inputList]);
    const end = ioEndNode("outer_end", [collectedSquare]);
    const flow = createFlow({
      name: "flow to square all elements of a list",
      startNode: start,
      nodes: [start, mapNode, end],
      controlFlowConnections: [ctrl(start, mapNode), ctrl(mapNode, end)],
      dataFlowConnections: [
        dataEdge(start, mapNode, "input_list", "iterated_input"),
        dataEdge(mapNode, end, "collected_input_square"),
      ],
    });

    const graph = await loadFlow(flow, { toolRegistry: squareRegistry });
    await expect(graph.invoke({ inputs: { input_list: 7 } })).rejects.toThrow(
      "MapNode `square_number_map_node` cannot iterate over input " +
        "`iterated_input`: 7 has no length",
    );
  });

  it("raises when no data-flow edge selects an input to iterate", async () => {
    const mapNode = createMapNode({
      name: "square_map_scalar",
      subflow: buildSquareSubflow(),
      inputs: [iteratedInput],
      outputs: [collectedSquare],
    });
    // The edge feeds a SCALAR into iterated_input, so the converter finds no
    // list-typed source matching the subflow input and selects nothing.
    const singleX = numberProperty({ title: "single_x" });
    const start = ioStartNode("outer_start", [singleX]);
    const end = ioEndNode("outer_end", [collectedSquare]);
    const flow = createFlow({
      name: "scalar_map_flow",
      startNode: start,
      nodes: [start, mapNode, end],
      controlFlowConnections: [ctrl(start, mapNode), ctrl(mapNode, end)],
      dataFlowConnections: [
        dataEdge(start, mapNode, "single_x", "iterated_input"),
        dataEdge(mapNode, end, "collected_input_square"),
      ],
    });

    const graph = await loadFlow(flow, { toolRegistry: squareRegistry });
    await expect(graph.invoke({ inputs: { single_x: 3 } })).rejects.toThrow(
      "MapNode has no inputs to iterate",
    );
  });
});

describe("ApiNode", () => {
  let mockFetch: MockFetchController | undefined;

  afterEach(() => {
    mockFetch?.restore();
    mockFetch = undefined;
    vi.restoreAllMocks();
  });

  function buildApiFlow(
    apiNode: Record<string, unknown>,
    inputProps: Property[],
    outputProps: Property[],
  ): Flow {
    const start = ioStartNode("start", inputProps);
    const end = ioEndNode("end", outputProps);
    return createFlow({
      name: "api_flow",
      startNode: start,
      nodes: [start, apiNode, end],
      controlFlowConnections: [ctrl(start, apiNode), ctrl(apiNode, end)],
      dataFlowConnections: [
        ...inputProps.map((prop) =>
          dataEdge(start, apiNode as unknown as ComponentWithIO, prop.title),
        ),
        ...outputProps.map((prop) =>
          dataEdge(apiNode as unknown as ComponentWithIO, end, prop.title),
        ),
      ],
      inputs: inputProps,
      outputs: outputProps,
    });
  }

  it("GET: templates the URL, query params and headers, and maps the JSON response", async () => {
    // Templated URL destination without an allow list warns per Python rules.
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const inputProps = [
      stringProperty({ title: "host" }),
      stringProperty({ title: "order_id" }),
      stringProperty({ title: "flag" }),
      stringProperty({ title: "token" }),
    ];
    const status = stringProperty({ title: "status" });
    const apiNode = createApiNode({
      name: "api",
      url: "https://{{host}}/orders/{{order_id}}",
      httpMethod: "GET",
      queryParams: { verbose: "{{flag}}" },
      headers: { "X-Auth": "Bearer {{token}}" },
      inputs: inputProps,
      outputs: [status],
    });
    const flow = buildApiFlow(apiNode, inputProps, [status]);
    const graph = await loadFlow(flow);
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining("ApiNode `api` uses placeholders in the URL destination"),
    );

    mockFetch = installMockFetch(() => ({ status: "ok" }));
    const result = await graph.invoke({
      inputs: {
        host: "allowed.example.com",
        order_id: "123",
        flag: "yes",
        token: "tok-1",
      },
    });

    expect(outputsOf(result)).toEqual({ status: "ok" });
    expect(mockFetch.calls).toHaveLength(1);
    expect(mockFetch.calls[0]!.url).toBe(
      "https://allowed.example.com/orders/123?verbose=yes",
    );
    const init = mockFetch.calls[0]!.init!;
    expect(init.method).toBe("GET");
    expect((init.headers as Record<string, string>)["X-Auth"]).toBe(
      "Bearer tok-1",
    );
    expect(init.body).toBeUndefined();
  });

  it("GET: warns when declared request data is dropped (fetch forbids GET bodies)", async () => {
    // Python's httpx sends the body on GET; fetch cannot, so the adapter
    // must at least warn instead of silently discarding the declared data.
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const inputProps = [stringProperty({ title: "term" })];
    const echo = stringProperty({ title: "echo" });
    const apiNode = createApiNode({
      name: "api",
      url: "https://api.example.com/search",
      httpMethod: "GET",
      data: { q: "{{term}}" },
      inputs: inputProps,
      outputs: [echo],
    });
    const flow = buildApiFlow(apiNode, inputProps, [echo]);
    const graph = await loadFlow(flow);

    mockFetch = installMockFetch(() => ({ echo: "done" }));
    await graph.invoke({ inputs: { term: "boots" } });

    expect(mockFetch.calls[0]!.init!.body).toBeUndefined();
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining(
        "ApiNode `api` declares request data for HTTP method GET",
      ),
    );
  });

  it("GET: does not warn about a dropped body for the default empty data", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const echo = stringProperty({ title: "echo" });
    const apiNode = createApiNode({
      name: "api",
      url: "https://api.example.com/plain",
      httpMethod: "GET",
      outputs: [echo],
    });
    const flow = buildApiFlow(apiNode, [], [echo]);
    const graph = await loadFlow(flow);

    mockFetch = installMockFetch(() => ({ echo: "done" }));
    await graph.invoke({ inputs: {} });

    expect(warnSpy).not.toHaveBeenCalledWith(
      expect.stringContaining("declares request data"),
    );
  });

  it("POST: templated dict data is sent as a JSON body with a JSON content type", async () => {
    const inputProps = [
      stringProperty({ title: "order_id" }),
      stringProperty({ title: "tag" }),
    ];
    const echo = stringProperty({ title: "echo" });
    const apiNode = createApiNode({
      name: "api",
      url: "https://api.example.com/orders",
      httpMethod: "POST",
      data: { order: { id: "{{order_id}}" }, tags: ["{{tag}}", "static"] },
      inputs: inputProps,
      outputs: [echo],
    });
    const flow = buildApiFlow(apiNode, inputProps, [echo]);
    const graph = await loadFlow(flow);

    mockFetch = installMockFetch(() => ({ echo: "done" }));
    const result = await graph.invoke({
      inputs: { order_id: "777", tag: "blue" },
    });

    expect(outputsOf(result)).toEqual({ echo: "done" });
    const init = mockFetch.calls[0]!.init!;
    expect(init.method).toBe("POST");
    expect(
      (init.headers as Record<string, string>)["Content-Type"],
    ).toBe("application/json");
    expect(JSON.parse(String(init.body))).toEqual({
      order: { id: "777" },
      tags: ["blue", "static"],
    });
  });

  it("POST: an urlencoded content type sends dict data as a form body and templates header keys", async () => {
    const inputProps = [
      stringProperty({ title: "a" }),
      stringProperty({ title: "key_name" }),
      stringProperty({ title: "key_val" }),
    ];
    const echo = stringProperty({ title: "echo" });
    const apiNode = createApiNode({
      name: "api",
      url: "https://api.example.com/form",
      httpMethod: "POST",
      data: { a: "{{a}}", b: "static" },
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        "X-{{key_name}}": "{{key_val}}",
      },
      inputs: inputProps,
      outputs: [echo],
    });
    const flow = buildApiFlow(apiNode, inputProps, [echo]);
    const graph = await loadFlow(flow);

    mockFetch = installMockFetch(() => ({ echo: "done" }));
    const result = await graph.invoke({
      inputs: { a: "1", key_name: "Trace", key_val: "on" },
    });

    expect(outputsOf(result)).toEqual({ echo: "done" });
    const init = mockFetch.calls[0]!.init!;
    const headers = init.headers as Record<string, string>;
    expect(headers["Content-Type"]).toBe("application/x-www-form-urlencoded");
    expect(headers["X-Trace"]).toBe("on");
    expect(init.body).toBeInstanceOf(URLSearchParams);
    expect(String(init.body)).toBe("a=1&b=static");
  });

  it("POST: an empty-string Content-Type falls through to the lowercase header (Python `or` parity)", async () => {
    // Python looks the content type up with `get("Content-Type") or
    // get("content-type")`: an empty-string uppercase header is falsy, so
    // the lowercase urlencoded header wins and dict data goes out as a form
    // body (a `??` lookup would stop at the empty string and send JSON).
    const echo = stringProperty({ title: "echo" });
    const apiNode = createApiNode({
      name: "api",
      url: "https://api.example.com/form",
      httpMethod: "POST",
      data: { a: "1" },
      headers: {
        "Content-Type": "",
        "content-type": "application/x-www-form-urlencoded",
      },
      outputs: [echo],
    });
    const flow = buildApiFlow(apiNode, [], [echo]);
    const graph = await loadFlow(flow);

    mockFetch = installMockFetch(() => ({ echo: "done" }));
    const result = await graph.invoke({ inputs: {} });

    expect(outputsOf(result)).toEqual({ echo: "done" });
    const init = mockFetch.calls[0]!.init!;
    expect(init.body).toBeInstanceOf(URLSearchParams);
    expect(String(init.body)).toBe("a=1");
  });

  it("does not follow redirects: a 3xx response body maps to the node outputs like any status", async () => {
    // Python's httpx does not follow redirects (follow_redirects defaults to
    // False) and parses the returned 3xx body like any other status; the
    // adapter uses redirect: "manual" so undici returns the 3xx response
    // itself instead of requesting the Location target.
    const echo = stringProperty({ title: "echo" });
    const apiNode = createApiNode({
      name: "api",
      url: "https://api.example.com/redirecting",
      httpMethod: "GET",
      outputs: [echo],
    });
    const flow = buildApiFlow(apiNode, [], [echo]);
    const graph = await loadFlow(flow);

    mockFetch = installMockFetch(
      () =>
        new Response('{"echo": "from-redirect-response"}', {
          status: 302,
          headers: {
            "Content-Type": "application/json",
            Location: "https://attacker.example/exfil",
          },
        }),
    );
    const result = await graph.invoke({ inputs: {} });

    expect(outputsOf(result)).toEqual({ echo: "from-redirect-response" });
    expect(mockFetch.calls).toHaveLength(1);
    expect(mockFetch.calls[0]!.init!.redirect).toBe("manual");
  });

  it("attaches the default httpx-parity timeout and names the node on a timeout abort", async () => {
    const echo = stringProperty({ title: "echo" });
    const apiNode = createApiNode({
      name: "api",
      url: "https://api.example.com/slow",
      httpMethod: "GET",
      outputs: [echo],
    });
    const flow = buildApiFlow(apiNode, [], [echo]);
    const graph = await loadFlow(flow);

    mockFetch = installMockFetch(() => {
      throw new DOMException(
        "The operation was aborted due to timeout",
        "TimeoutError",
      );
    });

    await expect(graph.invoke({ inputs: {} })).rejects.toThrow(
      `ApiNode \`api\` HTTP request timed out after ${DEFAULT_HTTP_REQUEST_TIMEOUT_MS}ms.`,
    );
    expect(mockFetch.calls).toHaveLength(1);
    expect(mockFetch.calls[0]!.init!.signal).toBeInstanceOf(AbortSignal);
  });

  it("POST: string data is sent as a raw body without forcing a content type", async () => {
    const inputProps = [stringProperty({ title: "val" })];
    const echo = stringProperty({ title: "echo" });
    const apiNode = createApiNode({
      name: "api",
      url: "https://api.example.com/raw",
      httpMethod: "POST",
      data: "payload={{val}}",
      inputs: inputProps,
      outputs: [echo],
    });
    const flow = buildApiFlow(apiNode, inputProps, [echo]);
    const graph = await loadFlow(flow);

    mockFetch = installMockFetch(() => ({ echo: "done" }));
    const result = await graph.invoke({ inputs: { val: "hello" } });

    expect(outputsOf(result)).toEqual({ echo: "done" });
    const init = mockFetch.calls[0]!.init!;
    expect(init.body).toBe("payload=hello");
    const headerKeys = Object.keys(init.headers as Record<string, string>);
    expect(
      headerKeys.some((key) => key.toLowerCase() === "content-type"),
    ).toBe(false);
  });
});
