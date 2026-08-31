/**
 * Exporter tests for the LangGraph adapter (LangGraph -> Agent Spec).
 *
 * Mirrors the offline-able behaviors of the Python suite
 * (`pyagentspec/tests/adapters/langgraph/test_langgraph_to_agentspec.py` and
 * `test_disaggregated_config.py`): structured tools to ServerTools, chat
 * models to LLM configs, langchain react agents to Agents, generic state
 * graphs to Flows (plain edges, conditional edges, subgraphs), shared-object
 * memoization, disaggregated exports and the documented TS-only rejections
 * (swarm graphs and bare compiled agent graphs). All tests run offline: chat
 * models are only constructed, never invoked.
 */
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { SystemMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import {
  Annotation,
  END,
  MemorySaver,
  START,
  StateGraph,
} from "@langchain/langgraph";
import { createSwarm } from "@langchain/langgraph-swarm";
import { ChatOllama } from "@langchain/ollama";
import { ChatOpenAI } from "@langchain/openai";
import { createAgent } from "langchain";
import {
  DEFAULT_BRANCH,
  DEFAULT_INPUT,
  OpenAIAPIType,
  createOpenAiCompatibleConfig,
  createServerTool,
  stringProperty,
} from "../../../src/index.js";
import type {
  Agent,
  Flow,
  OllamaConfig,
  OpenAiCompatibleConfig,
  OpenAiConfig,
  Property,
  ServerTool,
} from "../../../src/index.js";
import { LangGraphToAgentSpecConverter } from "../../../src/adapters/langgraph/agentspec-converter.js";
import { AgentSpecExporter } from "../../../src/adapters/langgraph/agentspec-exporter.js";
import { AgentSpecLoader } from "../../../src/adapters/langgraph/agentspec-loader.js";
import {
  FakeToolCallingChatModel,
  makeAgent,
  type LoadedReactAgent,
} from "./test-helpers.js";

const MODEL_ID = "Llama-3.1-70B-Instruct";
const LLAMA_URL = "https://url.to.my.llama.model/v1";

/** ChatOpenAI pointing at a fake OpenAI-compatible server (never invoked). */
function makeChatOpenAI(overrides?: {
  baseURL?: string;
  useResponsesApi?: boolean;
}): ChatOpenAI {
  return new ChatOpenAI({
    model: MODEL_ID,
    apiKey: "EMPTY",
    ...(overrides?.useResponsesApi !== undefined
      ? { useResponsesApi: overrides.useResponsesApi }
      : {}),
    configuration: { baseURL: overrides?.baseURL ?? LLAMA_URL },
  });
}

/** The langchain structured weather tool used across the agent tests. */
function makeWeatherLangChainTool() {
  return tool(() => "The weather is sunny.", {
    name: "get_weather",
    description: "Returns the weather in a certain city",
    schema: z.object({ city: z.string().describe("The city to check") }),
  });
}

/** Structural view of an exported flow node used by the assertions. */
interface ExportedNodeView {
  id: string;
  componentType: string;
  name: string;
  inputs?: Property[];
  outputs?: Property[];
  tool?: ServerTool;
  mapping?: Record<string, string>;
  branches?: string[];
  subflow?: Flow;
}

function nodesOf(flow: Flow): ExportedNodeView[] {
  return flow.nodes as unknown as ExportedNodeView[];
}

function nodeNamed(flow: Flow, name: string): ExportedNodeView {
  const node = nodesOf(flow).find((candidate) => candidate.name === name);
  if (node === undefined) {
    throw new Error(`Flow has no node named '${name}'.`);
  }
  return node;
}

function controlFlowNames(flow: Flow): string[] {
  return flow.controlFlowConnections.map((edge) => edge.name ?? "");
}

function dataFlowNames(flow: Flow): string[] {
  return (flow.dataFlowConnections ?? []).map((edge) => edge.name ?? "");
}

/** Keys listed by a synthetic `state` property. */
function statePropertyKeys(property: Property): string[] {
  return Object.keys(
    (property.jsonSchema["properties"] as Record<string, unknown>) ?? {},
  );
}

describe("AgentSpecExporter: structured tools", () => {
  it("converts a zod structured tool into a ServerTool with typed inputs", () => {
    const exporter = new AgentSpecExporter();
    const weatherTool = tool(() => "sunny", {
      name: "get_weather",
      description: "Returns the weather in a certain city",
      schema: z.object({
        city: z.string().describe("The city to check"),
        days: z.number().int().default(3),
      }),
    });

    const serverTool = exporter.toComponent(weatherTool) as ServerTool;

    expect(serverTool.componentType).toBe("ServerTool");
    expect(serverTool.name).toBe("get_weather");
    expect(serverTool.description).toBe("Returns the weather in a certain city");
    expect(serverTool.inputs).toHaveLength(2);
    const [city, days] = serverTool.inputs as [Property, Property];
    expect(city.title).toBe("city");
    expect(city.type).toBe("string");
    expect(city.description).toBe("The city to check");
    expect(city.jsonSchema["title"]).toBe("city");
    expect(days.title).toBe("days");
    expect(days.type).toBe("integer");
    expect(days.default).toBe(3);
  });

  it("converts a raw JSON-schema structured tool into a ServerTool", () => {
    const exporter = new AgentSpecExporter();
    const rawTool = tool(() => "ok", {
      name: "raw_weather",
      description: "Raw-schema weather tool",
      schema: {
        type: "object",
        properties: {
          city: { type: "string", description: "City name" },
          unit: { type: "string", default: "celsius" },
        },
        required: ["city"],
      } as const,
    });

    const serverTool = exporter.toComponent(rawTool) as ServerTool;

    expect(serverTool.componentType).toBe("ServerTool");
    expect(serverTool.name).toBe("raw_weather");
    expect(serverTool.description).toBe("Raw-schema weather tool");
    expect(serverTool.inputs).toHaveLength(2);
    const [city, unit] = serverTool.inputs as [Property, Property];
    expect(city.title).toBe("city");
    expect(city.type).toBe("string");
    expect(city.description).toBe("City name");
    expect(unit.title).toBe("unit");
    expect(unit.default).toBe("celsius");
    // The synthesized title lands in the json schema as well.
    expect(unit.jsonSchema["title"]).toBe("unit");
  });

  it("exports a tool argument without a JSON-schema type as a permissive property", () => {
    // Python exports an untyped Property here; the TS SDK Property model
    // requires `type` or `anyOf`, so the exporter synthesizes the closest
    // representable "any" union instead of failing the export.
    const exporter = new AgentSpecExporter();
    const anyTool = tool(() => "ok", {
      name: "any_tool",
      schema: {
        type: "object",
        properties: { mystery: { description: "no type here" } },
      } as const,
    });

    const serverTool = exporter.toComponent(anyTool) as ServerTool;
    expect(serverTool.inputs).toHaveLength(1);
    const mystery = serverTool.inputs![0]!;
    expect(mystery.title).toBe("mystery");
    expect(mystery.description).toBe("no type here");
    expect(mystery.type).toBeUndefined();
    expect(mystery.jsonSchema["anyOf"]).toEqual([
      { type: "object" },
      { type: "array" },
      { type: "string" },
      { type: "number" },
      { type: "integer" },
      { type: "boolean" },
      { type: "null" },
    ]);
  });

  it("exports a tool with a z.any() argument instead of failing", () => {
    const exporter = new AgentSpecExporter();
    const anyTool = tool(() => "ok", {
      name: "zany_tool",
      description: "any arg",
      schema: z.object({ payload: z.any() }),
    });

    expect(() => exporter.toComponent(anyTool)).not.toThrow();
  });
});

describe("AgentSpecExporter: chat models", () => {
  it("converts ChatOpenAI with the api.openai.com base URL to OpenAiConfig", () => {
    const exporter = new AgentSpecExporter();
    const model = new ChatOpenAI({
      model: "gpt-4o-mini",
      apiKey: "sk-test",
      configuration: { baseURL: "https://api.openai.com/v1" },
    });

    const config = exporter.toComponent(model) as OpenAiConfig;

    expect(config.componentType).toBe("OpenAiConfig");
    expect(config.modelId).toBe("gpt-4o-mini");
    expect(config.name).toBe("gpt-4o-mini");
    expect(config.apiType).toBe(OpenAIAPIType.CHAT_COMPLETIONS);
    expect("url" in config).toBe(false);
  });

  it("converts ChatOpenAI with a custom base URL to OpenAiCompatibleConfig", () => {
    const exporter = new AgentSpecExporter();
    const model = makeChatOpenAI();

    const config = exporter.toComponent(model) as OpenAiCompatibleConfig;

    expect(config.componentType).toBe("OpenAiCompatibleConfig");
    expect(config.modelId).toBe(MODEL_ID);
    expect(config.url).toBe(LLAMA_URL);
    expect(config.apiType).toBe(OpenAIAPIType.CHAT_COMPLETIONS);
  });

  it("converts ChatOpenAI without a base URL to OpenAiCompatibleConfig with an empty url", () => {
    // Python parity: `(model.openai_api_base or "").startswith(...)` routes an
    // unset base URL to OpenAiCompatibleConfig(url="").
    const exporter = new AgentSpecExporter();
    const model = new ChatOpenAI({ model: "gpt-4o-mini", apiKey: "sk-test" });

    const config = exporter.toComponent(model) as OpenAiCompatibleConfig;

    expect(config.componentType).toBe("OpenAiCompatibleConfig");
    expect(config.modelId).toBe("gpt-4o-mini");
    expect(config.url).toBe("");
  });

  it("detects the responses API on ChatOpenAI", () => {
    const exporter = new AgentSpecExporter();
    const model = makeChatOpenAI({ useResponsesApi: true });

    const config = exporter.toComponent(model) as OpenAiCompatibleConfig;

    expect(config.apiType).toBe(OpenAIAPIType.RESPONSES);
  });

  it("converts ChatOllama to OllamaConfig with base url and model id", () => {
    const exporter = new AgentSpecExporter();
    const model = new ChatOllama({
      model: "llama3.1",
      baseUrl: "http://ollama.example.com:11434",
    });

    const config = exporter.toComponent(model) as OllamaConfig;

    expect(config.componentType).toBe("OllamaConfig");
    expect(config.modelId).toBe("llama3.1");
    expect(config.name).toBe("llama3.1");
    expect(config.url).toBe("http://ollama.example.com:11434");
  });

  it("keeps the ChatOllama default base url", () => {
    const exporter = new AgentSpecExporter();
    const config = exporter.toComponent(
      new ChatOllama({ model: "llama3.1" }),
    ) as OllamaConfig;

    expect(config.url).toBe("http://127.0.0.1:11434");
  });

  it("rejects unsupported chat model types with the Python error text", () => {
    const exporter = new AgentSpecExporter();
    const model = new FakeToolCallingChatModel({ responses: [] });

    expect(() => exporter.toComponent(model)).toThrow(
      "The LLM instance provided is of an unsupported type `FakeToolCallingChatModel`.",
    );
  });
});

describe("AgentSpecExporter: react agents", () => {
  it("converts a langchain react agent with tools into an Agent Spec Agent", () => {
    const exporter = new AgentSpecExporter();
    const agent = createAgent({
      model: makeChatOpenAI(),
      tools: [makeWeatherLangChainTool()],
      systemPrompt: "You are a helpful assistant.",
      name: "weather_agent",
    });

    const agentSpecAgent = exporter.toComponent(agent) as Agent;

    expect(agentSpecAgent.componentType).toBe("Agent");
    expect(agentSpecAgent.name).toBe("weather_agent");
    expect(agentSpecAgent.systemPrompt).toBe("You are a helpful assistant.");
    const config = agentSpecAgent.llmConfig as OpenAiCompatibleConfig;
    expect(config.componentType).toBe("OpenAiCompatibleConfig");
    expect(config.modelId).toBe(MODEL_ID);
    expect(config.url).toBe(LLAMA_URL);
    expect(agentSpecAgent.tools).toHaveLength(1);
    const exportedTool = agentSpecAgent.tools[0] as ServerTool;
    expect(exportedTool.componentType).toBe("ServerTool");
    expect(exportedTool.name).toBe("get_weather");
    expect(exportedTool.description).toBe(
      "Returns the weather in a certain city",
    );
    expect(exportedTool.inputs.map((input) => input.title)).toEqual(["city"]);
  });

  it("converts a react agent without tools", () => {
    const exporter = new AgentSpecExporter();
    const agent = createAgent({ model: makeChatOpenAI(), tools: [] });

    const agentSpecAgent = exporter.toComponent(agent) as Agent;

    expect(agentSpecAgent.componentType).toBe("Agent");
    expect(agentSpecAgent.tools).toHaveLength(0);
    const config = agentSpecAgent.llmConfig as OpenAiCompatibleConfig;
    expect(config.modelId).toBe(MODEL_ID);
    expect(config.url).toBe(LLAMA_URL);
  });

  it("falls back to the default agent name when none is set", () => {
    const exporter = new AgentSpecExporter();
    const agent = createAgent({ model: makeChatOpenAI(), tools: [] });

    const agentSpecAgent = exporter.toComponent(agent) as Agent;

    expect(agentSpecAgent.name).toBe("LangGraph Agent");
  });

  it("extracts the system prompt from a SystemMessage", () => {
    const exporter = new AgentSpecExporter();
    const agent = createAgent({
      model: makeChatOpenAI(),
      tools: [],
      systemPrompt: new SystemMessage("Prompt from a message."),
    });

    const agentSpecAgent = exporter.toComponent(agent) as Agent;

    expect(agentSpecAgent.systemPrompt).toBe("Prompt from a message.");
  });

  it("does not export the responseFormat of a structured-output agent", () => {
    // The TS adapter ignores responseFormat on export (structured outputs are
    // not reconstructed into Agent outputs).
    const exporter = new AgentSpecExporter();
    const agent = createAgent({
      model: makeChatOpenAI(),
      tools: [makeWeatherLangChainTool()],
      systemPrompt: "Report the weather.",
      name: "structured_agent",
      responseFormat: z.object({ answer: z.string() }),
    });

    const agentSpecAgent = exporter.toComponent(agent) as Agent;

    expect(agentSpecAgent.componentType).toBe("Agent");
    expect(agentSpecAgent.name).toBe("structured_agent");
    expect(agentSpecAgent.outputs ?? []).toHaveLength(0);
    expect(agentSpecAgent.tools).toHaveLength(1);
  });

  it("rejects an agent created from a model identifier string", () => {
    const exporter = new AgentSpecExporter();
    const agent = createAgent({ model: "openai:gpt-4o", tools: [] });

    expect(() => exporter.toComponent(agent)).toThrow(
      "Exporting an agent created from a model identifier string is not " +
        "supported; pass a chat model instance to createAgent instead.",
    );
  });

  it("rejects a bare compiled agent graph", () => {
    const exporter = new AgentSpecExporter();
    const agent = createAgent({
      model: makeChatOpenAI(),
      tools: [makeWeatherLangChainTool()],
      name: "weather_agent",
    });

    expect(() => exporter.toComponent(agent.graph)).toThrow(
      "Exporting a compiled agent graph is not supported by the TypeScript " +
        "adapter: the compiled graph does not retain its chat model or " +
        "system prompt. Export the langchain ReactAgent instance (the " +
        "createAgent result) instead.",
    );
  });
});

describe("AgentSpecExporter: swarm graphs", () => {
  function makeSwarmBuilder() {
    const model = makeChatOpenAI();
    const alice = createAgent({ model, tools: [], name: "alice" });
    const bob = createAgent({ model, tools: [], name: "bob" });
    return createSwarm({
      agents: [alice.graph, bob.graph],
      defaultActiveAgent: "alice",
    });
  }

  it("rejects a compiled swarm graph", () => {
    const exporter = new AgentSpecExporter();
    const compiledSwarm = makeSwarmBuilder().compile({
      checkpointer: new MemorySaver(),
      name: "swarm",
    });

    expect(() => exporter.toComponent(compiledSwarm)).toThrow(
      "Exporting a LangGraph swarm is not supported by the TypeScript " +
        "adapter: the compiled per-agent graphs do not retain their chat " +
        "model or system prompt.",
    );
  });

  it("rejects an uncompiled swarm builder", () => {
    const exporter = new AgentSpecExporter();

    expect(() => exporter.toComponent(makeSwarmBuilder())).toThrow(
      "Exporting a LangGraph swarm is not supported",
    );
  });
});

describe("AgentSpecExporter: state graph flows", () => {
  const CodeGenState = Annotation.Root({
    language: Annotation<string>,
    request: Annotation<string>,
    output: Annotation<string>,
  });

  it("converts a linear compiled graph into a Flow", () => {
    const exporter = new AgentSpecExporter();
    const graph = new StateGraph(CodeGenState)
      .addNode("llm_code_gen", () => ({ output: "generated" }))
      .addEdge(START, "llm_code_gen")
      .addEdge("llm_code_gen", END);
    const compiled = graph.compile({ name: "CodeGen Assistant" });

    const flow = exporter.toComponent(compiled) as Flow;

    expect(flow.componentType).toBe("Flow");
    expect(flow.name).toBe("CodeGen Assistant");
    // llm_code_gen + synthesized __start__ + __end__
    expect(flow.nodes).toHaveLength(3);
    expect(
      nodesOf(flow).map((node) => [node.componentType, node.name]),
    ).toEqual([
      ["ToolNode", "llm_code_gen"],
      ["StartNode", "__start__"],
      ["EndNode", "__end__"],
    ]);
    // One ctrl+data pair per LangGraph edge, with the Python edge names.
    expect(controlFlowNames(flow)).toEqual([
      "__start___to_llm_code_gen",
      "llm_code_gen_to___end__",
    ]);
    expect(dataFlowNames(flow)).toEqual([
      "__start___to_llm_code_gen_data_edge",
      "llm_code_gen_to___end___data_edge",
    ]);

    // The synthetic tool mirrors the node, over a single `state` property
    // listing the channel keys.
    const toolNode = nodeNamed(flow, "llm_code_gen");
    expect(toolNode.tool?.componentType).toBe("ServerTool");
    expect(toolNode.tool?.name).toBe("llm_code_gen_tool");
    const toolInput = toolNode.tool?.inputs[0] as Property;
    expect(toolInput.title).toBe("state");
    expect(toolInput.type).toBe("object");
    expect(statePropertyKeys(toolInput)).toEqual([
      "language",
      "request",
      "output",
    ]);

    // Flow inputs/outputs are inferred from the synthesized start/end nodes.
    expect(flow.inputs?.map((input) => input.title)).toEqual(["state"]);
    expect(flow.outputs?.map((output) => output.title)).toEqual(["state"]);
    expect(statePropertyKeys(flow.inputs?.[0] as Property)).toEqual([
      "language",
      "request",
      "output",
    ]);
  });

  it("converts an uncompiled builder into a Flow with the default name", () => {
    const exporter = new AgentSpecExporter();
    const graph = new StateGraph(CodeGenState)
      .addNode("llm_code_gen", () => ({ output: "generated" }))
      .addEdge(START, "llm_code_gen")
      .addEdge("llm_code_gen", END);

    const flow = exporter.toComponent(graph) as Flow;

    expect(flow.componentType).toBe("Flow");
    expect(flow.name).toBe("LangGraph Flow");
  });

  it("synthesizes END edges for sink nodes without outgoing edges", () => {
    const exporter = new AgentSpecExporter();
    const graph = new StateGraph(CodeGenState)
      .addNode("sink", () => ({}))
      .addEdge(START, "sink");

    const flow = exporter.toComponent(graph.compile()) as Flow;

    expect(flow.nodes).toHaveLength(3);
    expect(controlFlowNames(flow)).toEqual([
      "__start___to_sink",
      "sink_to___end__",
    ]);
    expect(dataFlowNames(flow)).toEqual([
      "__start___to_sink_data_edge",
      "sink_to___end___data_edge",
    ]);
  });

  it("converts a graph with distinct input/output/node schemas", () => {
    // Per-node `input` options are the JS equivalent of the Python function
    // annotations the Python adapter introspects.
    const exporter = new AgentSpecExporter();
    const InputSchema = Annotation.Root({ city: Annotation<string> });
    const OutputSchema = Annotation.Root({ response: Annotation<string> });
    const WeatherSchema = Annotation.Root({
      weather_data: Annotation<string>,
    });
    const InternalState = Annotation.Root({
      city: Annotation<string>,
      weather_data: Annotation<string>,
      response: Annotation<string>,
    });
    const graph = new StateGraph({
      state: InternalState,
      input: InputSchema,
      output: OutputSchema,
    })
      .addNode("get_weather", () => ({ weather_data: "sunny" }), {
        input: InputSchema,
      })
      .addNode("llm_node", () => ({ response: "reformulated" }), {
        input: WeatherSchema,
      })
      .addEdge(START, "get_weather")
      .addEdge("get_weather", "llm_node")
      .addEdge("llm_node", END);

    const flow = exporter.toComponent(graph.compile({ name: "Weather Flow" })) as Flow;

    expect(flow.name).toBe("Weather Flow");
    // get_weather + llm_node + __start__ + __end__
    expect(flow.nodes).toHaveLength(4);
    expect(flow.controlFlowConnections).toHaveLength(3);
    expect(flow.dataFlowConnections).toHaveLength(3);
    const startNode = nodeNamed(flow, "__start__");
    const endNode = nodeNamed(flow, "__end__");
    expect(statePropertyKeys(startNode.outputs?.[0] as Property)).toEqual([
      "city",
    ]);
    expect(statePropertyKeys(endNode.outputs?.[0] as Property)).toEqual([
      "response",
    ]);
    const getWeatherNode = nodeNamed(flow, "get_weather");
    expect(statePropertyKeys(getWeatherNode.inputs?.[0] as Property)).toEqual([
      "city",
    ]);
    expect(statePropertyKeys(getWeatherNode.outputs?.[0] as Property)).toEqual([
      "weather_data",
    ]);
  });

  it("expands a conditional edge into a conditional ToolNode plus a BranchingNode", () => {
    const exporter = new AgentSpecExporter();
    const CaseState = Annotation.Root({ sentence: Annotation<string> });
    const graph = new StateGraph(CaseState)
      .addNode("lowercase", () => ({}))
      .addNode("uppercase", () => ({}))
      .addNode("messycase", () => ({}))
      .addConditionalEdges(START, () => "lowercase", {
        lowercase: "lowercase",
        uppercase: "uppercase",
        messycase: "messycase",
      });

    const flow = exporter.toComponent(
      graph.compile({ name: "Casecheck Flow" }),
    ) as Flow;

    expect(flow.name).toBe("Casecheck Flow");
    // 3 case nodes + __start__ + __end__ + conditional node + branching node
    expect(flow.nodes).toHaveLength(7);

    // The conditional ToolNode computes the branch name (LangGraph JS names
    // every conditional branch "condition").
    const conditionalNode = nodeNamed(flow, "condition");
    expect(conditionalNode.componentType).toBe("ToolNode");
    expect(conditionalNode.tool?.name).toBe("condition_tool");
    expect(conditionalNode.tool?.outputs.map((output) => output.title)).toEqual(
      [DEFAULT_INPUT],
    );

    const branchingNode = nodeNamed(flow, "condition_branching_node");
    expect(branchingNode.componentType).toBe("BranchingNode");
    expect(branchingNode.mapping).toEqual({
      lowercase: "lowercase",
      uppercase: "uppercase",
      messycase: "messycase",
    });
    expect(new Set(branchingNode.branches)).toEqual(
      new Set([DEFAULT_BRANCH, "lowercase", "uppercase", "messycase"]),
    );

    // Control edges: source -> conditional -> branching -> per-branch targets
    // plus the default fall-through to END and auto-END edges for the sinks.
    const edgesWithBranch = flow.controlFlowConnections.map((edge) => [
      edge.name,
      edge.fromBranch,
    ]);
    expect(edgesWithBranch).toEqual([
      ["__start___to_condition", undefined],
      ["condition_to_condition_branching_node", undefined],
      ["condition_branching_node_to_lowercase", "lowercase"],
      ["condition_branching_node_to_uppercase", "uppercase"],
      ["condition_branching_node_to_messycase", "messycase"],
      ["condition_branching_node_to___end__", DEFAULT_BRANCH],
      ["lowercase_to___end__", undefined],
      ["uppercase_to___end__", undefined],
      ["messycase_to___end__", undefined],
    ]);

    const dataNames = dataFlowNames(flow);
    expect(dataNames).toContain("__start___to_condition_data_edge");
    expect(dataNames).toContain(
      "condition_to_condition_branching_node_data_edge",
    );
    expect(dataNames).toContain("data___start___to_lowercase");
    const branchingDataEdge = (flow.dataFlowConnections ?? []).find(
      (edge) => edge.name === "condition_to_condition_branching_node_data_edge",
    );
    expect(branchingDataEdge?.sourceOutput).toBe(DEFAULT_INPUT);
    expect(branchingDataEdge?.destinationInput).toBe(DEFAULT_INPUT);
  });

  it("keeps a real node named 'condition' distinct from the synthetic conditional node", () => {
    // LangGraph JS stores every conditional edge's branch under the fixed key
    // "condition"; a user node with that literal name must not be overwritten
    // by the synthetic conditional ToolNode. The synthetic names are suffixed
    // instead (only in the colliding case).
    const exporter = new AgentSpecExporter();
    const CaseState = Annotation.Root({ sentence: Annotation<string> });
    const graph = new StateGraph(CaseState)
      .addNode("condition", () => ({}))
      .addNode("other", () => ({}))
      .addConditionalEdges(START, () => "condition", {
        condition: "condition",
        other: "other",
      });

    const flow = exporter.toComponent(
      graph.compile({ name: "Collision Flow" }),
    ) as Flow;

    // 2 real nodes + __start__ + __end__ + conditional node + branching node.
    expect(flow.nodes).toHaveLength(6);

    const realNode = nodeNamed(flow, "condition");
    expect(realNode.componentType).toBe("ToolNode");
    expect(realNode.tool?.name).toBe("condition_tool");

    const conditionalNode = nodeNamed(flow, "condition_1");
    expect(conditionalNode.componentType).toBe("ToolNode");
    expect(conditionalNode.tool?.name).toBe("condition_1_tool");

    const branchingNode = nodeNamed(flow, "condition_1_branching_node");
    expect(branchingNode.componentType).toBe("BranchingNode");
    expect(branchingNode.mapping).toEqual({
      condition: "condition",
      other: "other",
    });

    // The branch-target edge is wired to the REAL node, not the synthetic one.
    const branchTargetEdge = flow.controlFlowConnections.find(
      (edge) => edge.name === "condition_1_branching_node_to_condition",
    );
    expect(branchTargetEdge?.fromBranch).toBe("condition");
    expect((branchTargetEdge?.toNode as unknown as ExportedNodeView).id).toBe(
      realNode.id,
    );

    // And the real node stays connected downstream (auto edge to END).
    expect(controlFlowNames(flow)).toContain("condition_to___end__");
  });

  it("rejects a conditional edge without a path map", () => {
    const exporter = new AgentSpecExporter();
    const CaseState = Annotation.Root({ sentence: Annotation<string> });
    const graph = new StateGraph(CaseState)
      .addNode("lowercase", () => ({}))
      .addConditionalEdges(START, () => "lowercase");

    expect(() => exporter.toComponent(graph.compile())).toThrow(
      "Mapping for condition not found.\n" +
        "            Make sure to add proper return type hints to the branching function.",
    );
  });

  it("rejects multiple conditional edges with the same source node", () => {
    const exporter = new AgentSpecExporter();
    const CaseState = Annotation.Root({ sentence: Annotation<string> });
    const graph = new StateGraph(CaseState)
      .addNode("node_a", () => ({}))
      .addNode("node_b", () => ({}))
      .addConditionalEdges(START, () => "node_a", { go: "node_a" });
    // LangGraph JS names every conditional branch "condition" and refuses a
    // second one on the same source, so the runtime shape the exporter guards
    // against is reproduced on the builder directly.
    const branches = (
      graph as unknown as {
        branches: Record<string, Record<string, unknown>>;
      }
    ).branches;
    branches[START]!["condition2"] = branches[START]!["condition"]!;

    expect(() => exporter.toComponent(graph)).toThrow(
      "Conversion of multiple conditional edges with the same source node is not yet supported",
    );
  });

  it("converts subgraph nodes into FlowNodes recursively", () => {
    const exporter = new AgentSpecExporter();
    const SubState = Annotation.Root({ foo: Annotation<string> });
    const subgraph = new StateGraph(SubState)
      .addNode("subgraph_node_1", (state) => ({ foo: `hi! ${state.foo}` }))
      .addEdge(START, "subgraph_node_1")
      .compile();
    const parent = new StateGraph(SubState)
      .addNode("node_1", subgraph)
      .addEdge(START, "node_1");
    const compiled = parent.compile({ name: "GraphWithSubgraph" });

    const flow = exporter.toComponent(compiled) as Flow;

    expect(flow.componentType).toBe("Flow");
    expect(flow.name).toBe("GraphWithSubgraph");
    const flowNodes = nodesOf(flow).filter(
      (node) => node.componentType === "FlowNode",
    );
    expect(flowNodes).toHaveLength(1);
    expect(flowNodes[0]!.name).toBe("node_1");

    const subflow = flowNodes[0]!.subflow as Flow;
    expect(subflow.componentType).toBe("Flow");
    // Both levels synthesize __start__/__end__ around their single node.
    expect(flow.nodes).toHaveLength(3);
    expect(subflow.nodes).toHaveLength(3);
    expect(
      nodesOf(subflow).map((node) => [node.componentType, node.name]),
    ).toEqual([
      ["ToolNode", "subgraph_node_1"],
      ["StartNode", "__start__"],
      ["EndNode", "__end__"],
    ]);
    // Explicit edge + implicit edge to END at each level.
    expect(controlFlowNames(flow)).toEqual([
      "__start___to_node_1",
      "node_1_to___end__",
    ]);
    expect(controlFlowNames(subflow as Flow)).toEqual([
      "__start___to_subgraph_node_1",
      "subgraph_node_1_to___end__",
    ]);
  });
});

describe("AgentSpecExporter: shared components and disaggregation", () => {
  it("memoizes a chat model and tool shared by two agents", () => {
    const converter = new LangGraphToAgentSpecConverter();
    const referencedObjects = new Map<string, never>();
    const model = makeChatOpenAI();
    const sharedTool = makeWeatherLangChainTool();
    const first = converter.convert(
      createAgent({ model, tools: [sharedTool], name: "first" }),
      referencedObjects,
    ) as Agent;
    const second = converter.convert(
      createAgent({ model, tools: [sharedTool], name: "second" }),
      referencedObjects,
    ) as Agent;

    // The runtime objects converted once: both agents reference components
    // with the same ids (fresh conversions would generate fresh ids).
    expect(first.llmConfig.id).toBe(second.llmConfig.id);
    expect(first.tools[0]!.id).toBe(second.tools[0]!.id);
  });

  it("replaces disaggregated components with $component_ref in the export", () => {
    const exporter = new AgentSpecExporter();
    const model = makeChatOpenAI();
    const agent = createAgent({
      model,
      tools: [makeWeatherLangChainTool()],
      systemPrompt: "You are a helpful assistant.",
      name: "weather_agent",
    });

    const [mainDict, disagDict] = exporter.toDict(agent, {
      disaggregatedComponents: [[model, "llm_config_id"]],
      exportDisaggregatedComponents: true,
    }) as [Record<string, unknown>, Record<string, unknown>];

    expect(mainDict["component_type"]).toBe("Agent");
    expect(mainDict["llm_config"]).toEqual({ $component_ref: "llm_config_id" });
    const referenced = disagDict["$referenced_components"] as Record<
      string,
      Record<string, unknown>
    >;
    expect(Object.keys(disagDict)).toEqual(["$referenced_components"]);
    expect(Object.keys(referenced)).toContain("llm_config_id");
    expect(referenced["llm_config_id"]!["component_type"]).toBe(
      "OpenAiCompatibleConfig",
    );
  });

  it("keeps the component's own id in the disaggregated dump; a custom id is only the mapping key", () => {
    // Python parity: a custom disaggregation id is applied only as the
    // serialization-time mapping key ($referenced_components key and
    // $component_ref target). The converted component keeps its own id, so
    // the same component object/id is used in the root tree and the
    // disaggregated registry (no re-keyed copy).
    const exporter = new AgentSpecExporter();
    const model = makeChatOpenAI();
    const sharedTool = makeWeatherLangChainTool();
    const agent = createAgent({
      model,
      tools: [sharedTool],
      systemPrompt: "You are a helpful assistant.",
      name: "weather_agent",
    });

    const [mainDict, disagDict] = exporter.toDict(agent, {
      disaggregatedComponents: [[model, "llm_config_id"], sharedTool],
      exportDisaggregatedComponents: true,
    }) as [Record<string, unknown>, Record<string, unknown>];

    const referenced = disagDict["$referenced_components"] as Record<
      string,
      Record<string, unknown>
    >;

    // Custom-id entry: registry keyed by the custom id, component id kept.
    const llmEntry = referenced["llm_config_id"]!;
    expect(typeof llmEntry["id"]).toBe("string");
    expect(llmEntry["id"]).not.toBe("llm_config_id");
    expect(mainDict["llm_config"]).toEqual({ $component_ref: "llm_config_id" });

    // Bare entry: registry keyed by the component's own id, referenced from
    // the root under that same id.
    const toolRefs = mainDict["tools"] as Array<Record<string, unknown>>;
    expect(toolRefs).toHaveLength(1);
    const toolRefId = toolRefs[0]!["$component_ref"] as string;
    expect(toolRefId).not.toBe("llm_config_id");
    expect(referenced[toolRefId]!["id"]).toBe(toolRefId);
    expect(referenced[toolRefId]!["component_type"]).toBe("ServerTool");
  });
});

describe("AgentSpecExporter: export output shapes", () => {
  function makeExportableAgent() {
    return createAgent({
      model: makeChatOpenAI(),
      tools: [makeWeatherLangChainTool()],
      systemPrompt: "You are a helpful assistant.",
      name: "weather_agent",
    });
  }

  it("toJson returns a JSON string of the serialized component", () => {
    const exporter = new AgentSpecExporter();
    const json = exporter.toJson(makeExportableAgent());

    expect(typeof json).toBe("string");
    const parsed = JSON.parse(json as string) as Record<string, unknown>;
    expect(parsed["component_type"]).toBe("Agent");
    expect(parsed["name"]).toBe("weather_agent");
    expect(parsed["system_prompt"]).toBe("You are a helpful assistant.");
  });

  it("toYaml returns a YAML string of the serialized component", () => {
    const exporter = new AgentSpecExporter();
    const yaml = exporter.toYaml(makeExportableAgent());

    expect(typeof yaml).toBe("string");
    expect(yaml as string).toContain("component_type");
    expect(yaml as string).toContain("Agent");
  });

  it("toDict returns the serialized dictionary", () => {
    const exporter = new AgentSpecExporter();
    const dict = exporter.toDict(makeExportableAgent()) as Record<
      string,
      unknown
    >;

    expect(Array.isArray(dict)).toBe(false);
    expect(dict["component_type"]).toBe("Agent");
    expect(dict["name"]).toBe("weather_agent");
  });

  it("returns [main, referenced] pairs for disaggregated json and yaml exports", () => {
    const exporter = new AgentSpecExporter();
    const model = makeChatOpenAI();
    const agent = createAgent({
      model,
      tools: [],
      systemPrompt: "You are a helpful assistant.",
      name: "weather_agent",
    });
    const options = {
      disaggregatedComponents: [[model, "llm_config_id"] as const],
      exportDisaggregatedComponents: true,
    };

    const [mainJson, disagJson] = exporter.toJson(agent, options) as [
      string,
      string,
    ];
    expect(mainJson).toContain('"component_type": "Agent"');
    expect(mainJson).toContain("llm_config_id");
    expect(disagJson).toContain("$referenced_components");
    expect(disagJson).toContain("llm_config_id");

    const [mainYaml, disagYaml] = exporter.toYaml(agent, options) as [
      string,
      string,
    ];
    expect(mainYaml).toContain("component_type");
    expect(mainYaml).toContain("Agent");
    expect(mainYaml).toContain("llm_config_id");
    expect(disagYaml).toContain("$referenced_components");
    expect(disagYaml).toContain("llm_config_id");
  });
});

describe("AgentSpecExporter: loader round trip", () => {
  it("re-exports a loaded Agent spec with equivalent llm, prompt and tools", async () => {
    const llmConfig = createOpenAiCompatibleConfig({
      name: "llama",
      url: LLAMA_URL,
      modelId: MODEL_ID,
    });
    const weatherTool = createServerTool({
      name: "get_weather",
      description: "Returns the weather in a certain city",
      inputs: [stringProperty({ title: "city" })],
      outputs: [stringProperty({ title: "weather" })],
    });
    const spec = makeAgent({
      name: "weather_agent",
      systemPrompt: "You are a helpful assistant.",
      llmConfig,
      tools: [weatherTool],
    });

    const loader = new AgentSpecLoader({
      toolRegistry: { get_weather: () => "sunny" },
    });
    const loaded = (await loader.loadComponent(spec)) as LoadedReactAgent;
    expect(
      (loaded.options["model"] as { constructor: { name: string } }).constructor
        .name,
    ).toBe("ChatOpenAI");

    const exporter = new AgentSpecExporter();
    const exported = exporter.toComponent(loaded) as Agent;

    expect(exported.componentType).toBe("Agent");
    expect(exported.name).toBe("weather_agent");
    expect(exported.systemPrompt).toBe("You are a helpful assistant.");
    const exportedConfig = exported.llmConfig as OpenAiCompatibleConfig;
    expect(exportedConfig.componentType).toBe("OpenAiCompatibleConfig");
    expect(exportedConfig.modelId).toBe(MODEL_ID);
    expect(exportedConfig.url).toBe(LLAMA_URL);
    expect(exported.tools).toHaveLength(1);
    const exportedTool = exported.tools[0] as ServerTool;
    expect(exportedTool.name).toBe("get_weather");
    expect(exportedTool.description).toBe(
      "Returns the weather in a certain city",
    );
    expect(exportedTool.inputs.map((input) => input.title)).toEqual(["city"]);
    expect(exportedTool.inputs[0]!.type).toBe("string");
  });
});

describe("AgentSpecExporter: unsupported runtime components", () => {
  it("rejects values that match no supported runtime shape", () => {
    const exporter = new AgentSpecExporter();

    expect(() => exporter.toComponent(42)).toThrow(
      "Conversion for 42 not implemented yet",
    );
  });
});
