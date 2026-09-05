/**
 * Exporter tests for the LangGraph adapter (LangGraph -> Agent Spec).
 *
 * Mirrors the offline-able behaviors of the Python suite
 * (`pyagentspec/tests/adapters/langgraph/test_langgraph_to_agentspec.py` and
 * `test_disaggregated_config.py`): structured tools to ServerTools, chat
 * models to LLM configs, langchain react agents to Agents, shared-object
 * memoization, disaggregated exports and the documented TS-only rejections
 * (swarm graphs and bare compiled agent graphs). State-graph-to-Flow
 * conversion lives in `exporter-flow.test.ts`. All tests run offline: chat
 * models are only constructed, never invoked.
 */
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { SystemMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { MemorySaver } from "@langchain/langgraph";
import { createSwarm } from "@langchain/langgraph-swarm";
import { ChatOllama } from "@langchain/ollama";
import { ChatOpenAI } from "@langchain/openai";
import { createAgent } from "langchain";
import {
  OpenAIAPIType,
  createOpenAiCompatibleConfig,
  createServerTool,
  stringProperty,
} from "../../../src/index.js";
import type {
  Agent,
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

  it("exports explicit ChatOpenAI maxRetries and (ms) timeout as a retryPolicy", () => {
    const exporter = new AgentSpecExporter();
    const model = new ChatOpenAI({
      model: MODEL_ID,
      apiKey: "EMPTY",
      maxRetries: 5,
      timeout: 45_000,
      configuration: { baseURL: LLAMA_URL },
    });

    const config = exporter.toComponent(model) as OpenAiCompatibleConfig;

    expect(config.retryPolicy?.maxAttempts).toBe(5);
    // The JS ChatOpenAI timeout is milliseconds; the spec's requestTimeout
    // is seconds.
    expect(config.retryPolicy?.requestTimeout).toBe(45);
  });

  it("exports a timeout-only ChatOpenAI with the default retry count", () => {
    const exporter = new AgentSpecExporter();
    const model = new ChatOpenAI({
      model: "gpt-4o-mini",
      apiKey: "sk-test",
      timeout: 30_000,
      configuration: { baseURL: "https://api.openai.com/v1" },
    });

    const config = exporter.toComponent(model) as OpenAiConfig;

    expect(config.componentType).toBe("OpenAiConfig");
    expect(config.retryPolicy?.maxAttempts).toBe(2);
    expect(config.retryPolicy?.requestTimeout).toBe(30);
  });

  it("omits the retryPolicy when retries and timeout sit at their defaults", () => {
    const exporter = new AgentSpecExporter();

    const defaultConfig = exporter.toComponent(
      makeChatOpenAI(),
    ) as OpenAiCompatibleConfig;
    expect("retryPolicy" in defaultConfig).toBe(false);

    // An explicit maxRetries equal to the RetryPolicy default (2) is not a
    // customization, like Python's default-comparison.
    const explicitDefault = exporter.toComponent(
      new ChatOpenAI({
        model: MODEL_ID,
        apiKey: "EMPTY",
        maxRetries: 2,
        configuration: { baseURL: LLAMA_URL },
      }),
    ) as OpenAiCompatibleConfig;
    expect("retryPolicy" in explicitDefault).toBe(false);
  });

  it("rejects a non-scalar ChatOpenAI timeout with the Python text", () => {
    const exporter = new AgentSpecExporter();
    const model = makeChatOpenAI();
    // The JS field is typed number, so a non-scalar can only arrive through
    // an unsound cast — mirror Python's httpx.Timeout rejection anyway.
    (model as unknown as { timeout: unknown }).timeout = { read: 10 };

    expect(() => exporter.toComponent(model)).toThrow(
      "LangGraph ChatOpenAI timeout conversion supports only a single timeout value " +
        "because Agent Spec `RetryPolicy.request_timeout` exposes one per-request timeout.",
    );
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
