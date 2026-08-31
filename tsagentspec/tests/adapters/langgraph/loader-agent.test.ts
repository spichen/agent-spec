/**
 * Loader + agent tests for the LangGraph adapter.
 *
 * Mirrors the offline-able behaviors of the Python suite
 * (`pyagentspec/tests/adapters/langgraph/`): load entry points, agent +
 * ServerTool round trips through a fake LLM, tool registry semantics, the
 * ClientTool interrupt protocol, `requiresConfirmation` human-in-the-loop,
 * structured outputs, middleware plumbing, disaggregated configurations and
 * the component load policy. All tests run offline.
 */
import { describe, expect, it, vi } from "vitest";
import { AIMessage, type BaseMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import type { StructuredToolInterface } from "@langchain/core/tools";
import { Command, MemorySaver } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { createMiddleware } from "langchain";
import {
  AgentSpecSerializer,
  createClientTool,
  createMCPTool,
  createServerTool,
  createStdioTransport,
  createVllmConfig,
  integerProperty,
  stringProperty,
} from "../../../src/index.js";
import { AgentSpecLoader } from "../../../src/adapters/langgraph/agentspec-loader.js";
import {
  FakeLlmAgentSpecLoader,
  FakeToolCallingChatModel,
  approveCommand,
  getInterrupts,
  loadWithFakeLlm,
  makeAgent,
  makeLlmConfig,
  rejectCommand,
  threadConfig,
  toolCallMessage,
  type LoadedReactAgent,
} from "./test-helpers.js";

const STRUCTURED_OUTPUT_PROMPT_SUFFIX =
  "\n\n" +
  "After using the available tools, provide the final result by calling the " +
  "structured output tool. Do not respond with a plain-text final answer.";

function makeWeatherServerTool(overrides?: { requiresConfirmation?: boolean }) {
  return createServerTool({
    name: "get_weather",
    description: "Returns the weather for a city",
    inputs: [stringProperty({ title: "city" })],
    outputs: [stringProperty({ title: "weather" })],
    ...(overrides?.requiresConfirmation !== undefined
      ? { requiresConfirmation: overrides.requiresConfirmation }
      : {}),
  });
}

function getWeather(input: unknown): string {
  const { city } = input as { city: string };
  return `The weather in ${city} is sunny.`;
}

function messagesOf(result: Record<string, unknown>): BaseMessage[] {
  return result["messages"] as BaseMessage[];
}

describe("AgentSpecLoader load entry points", () => {
  const agentSpec = makeAgent({ name: "weather_agent" });
  const serializer = new AgentSpecSerializer();

  function assertLoadedAgent(loaded: unknown): void {
    const agent = loaded as LoadedReactAgent;
    expect(typeof agent.invoke).toBe("function");
    expect(agent.graph.lg_is_pregel).toBe(true);
    expect(agent.graph.getName()).toBe("weather_agent");
  }

  it("loadYaml returns a compiled react agent preserving the agent name", async () => {
    const yaml = serializer.toYaml(agentSpec) as string;
    const loader = new FakeLlmAgentSpecLoader([new AIMessage("hello")]);
    assertLoadedAgent(await loader.loadYaml(yaml));
  });

  it("loadJson returns a compiled react agent preserving the agent name", async () => {
    const json = serializer.toJson(agentSpec) as string;
    const loader = new FakeLlmAgentSpecLoader([new AIMessage("hello")]);
    assertLoadedAgent(await loader.loadJson(json));
  });

  it("loadDict returns a compiled react agent preserving the agent name", async () => {
    const dict = JSON.parse(serializer.toJson(agentSpec) as string) as Record<
      string,
      unknown
    >;
    const loader = new FakeLlmAgentSpecLoader([new AIMessage("hello")]);
    assertLoadedAgent(await loader.loadDict(dict));
  });
});

describe("agent with server tool", () => {
  it("runs the tool loop: tool_call -> ToolMessage -> final answer", async () => {
    const agentSpec = makeAgent({
      name: "weather_agent",
      tools: [makeWeatherServerTool()],
    });
    const { agent, loader } = await loadWithFakeLlm(
      agentSpec,
      [
        toolCallMessage("get_weather", { city: "Agadir" }),
        new AIMessage("It is sunny in Agadir."),
      ],
      { toolRegistry: { get_weather: getWeather } },
    );

    const result = await agent.invoke({
      messages: [{ role: "user", content: "What is the weather in Agadir?" }],
    });

    const messages = messagesOf(result);
    expect(messages.length).toBeGreaterThan(2);
    const finalMessage = messages[messages.length - 1]!;
    expect(finalMessage.getType()).toBe("ai");
    expect(finalMessage.content).toBe("It is sunny in Agadir.");
    const toolMessage = messages[messages.length - 2]!;
    expect(toolMessage.getType()).toBe("tool");
    expect(String(toolMessage.content)).toContain(
      "The weather in Agadir is sunny.",
    );

    // The converted tool was bound onto the model by createAgent.
    const fakeModel = loader.getFakeModel();
    const boundNames = fakeModel.bound.map(
      (boundTool) => (boundTool as { name?: string }).name,
    );
    expect(boundNames).toContain("get_weather");
  });
});

describe("tool registry semantics", () => {
  const doubleToolSpec = createServerTool({
    name: "double_tool",
    description: "Doubles input",
    inputs: [integerProperty({ title: "x" })],
    outputs: [integerProperty({ title: "result" })],
  });

  it("converts a plain sync function using the spec name/description/schema", async () => {
    const loader = new AgentSpecLoader({
      toolRegistry: { double_tool: (input: unknown) => (input as { x: number }).x * 2 },
    });
    const converted = (await loader.loadComponent(
      doubleToolSpec,
    )) as StructuredToolInterface;
    expect(converted.name).toBe("double_tool");
    expect(converted.description).toBe("Doubles input");
    expect(await converted.invoke({ x: 5 })).toBe(10);
  });

  it("converts an async function", async () => {
    const loader = new AgentSpecLoader({
      toolRegistry: {
        double_tool: async (input: unknown) => (input as { x: number }).x * 2,
      },
    });
    const converted = (await loader.loadComponent(
      doubleToolSpec,
    )) as StructuredToolInterface;
    expect(await converted.invoke({ x: 7 })).toBe(14);
  });

  it("reuses name, description and schema of a registered structured tool", async () => {
    const argsSchema = {
      title: "RegisteredArgs",
      type: "object",
      properties: { x: { title: "x", type: "integer" } },
      required: ["x"],
    };
    const registered = tool(
      (input: unknown) => (input as { x: number }).x * 2,
      {
        name: "registered_double",
        description: "Registered description",
        schema: argsSchema,
      },
    );
    const loader = new AgentSpecLoader({
      toolRegistry: { double_tool: registered },
    });
    const converted = (await loader.loadComponent(
      doubleToolSpec,
    )) as StructuredToolInterface;
    expect(converted.name).toBe("registered_double");
    expect(converted.description).toBe("Registered description");
    expect(converted.schema).toBe(argsSchema);
    expect(await converted.invoke({ x: 6 })).toBe(12);
  });

  it("raises the Python error text for a tool missing from the registry", async () => {
    const loader = new AgentSpecLoader();
    await expect(loader.loadComponent(doubleToolSpec)).rejects.toThrow(
      "The Agent Spec representation includes a tool 'double_tool' " +
        "but this tool does not appear in the tool registry",
    );
  });

  it("raises the missing-registry error for tool names on Object.prototype", async () => {
    // Registry membership must be own-keys only (Python dict semantics): a
    // spec-controlled name like "constructor" must not resolve to the
    // inherited Object function.
    const adversarialSpec = createServerTool({
      name: "constructor",
      description: "Adversarially named tool",
    });
    const loader = new AgentSpecLoader({ toolRegistry: {} });
    await expect(loader.loadComponent(adversarialSpec)).rejects.toThrow(
      "The Agent Spec representation includes a tool 'constructor' " +
        "but this tool does not appear in the tool registry",
    );
  });

  it("injects declared input defaults into the function input like Python", async () => {
    // Python's pydantic args models fill Property defaults before the tool
    // body runs; langchain JS does not apply JSON-schema defaults, so the
    // adapter injects them itself.
    const received: unknown[] = [];
    const searchSpec = createServerTool({
      name: "search_tool",
      description: "Searches",
      inputs: [
        stringProperty({ title: "query" }),
        integerProperty({ title: "limit", default: 10 }),
      ],
      outputs: [integerProperty({ title: "result" })],
    });
    const loader = new AgentSpecLoader({
      toolRegistry: {
        search_tool: (input: unknown) => {
          received.push(input);
          return 1;
        },
      },
    });
    const converted = (await loader.loadComponent(
      searchSpec,
    )) as StructuredToolInterface;
    await converted.invoke({ query: "abc" });
    expect(received).toEqual([{ query: "abc", limit: 10 }]);
  });

  it("raises for an unsupported registry entry type", async () => {
    const loader = new AgentSpecLoader({ toolRegistry: { double_tool: 42 } });
    await expect(loader.loadComponent(doubleToolSpec)).rejects.toThrow(
      "Unsupported tool type for 'double_tool': number. " +
        "Expected callable, StructuredTool, or supported BaseTool.",
    );
  });
});

describe("client tool interrupt protocol", () => {
  const clientToolSpec = createClientTool({
    name: "get_weather",
    description: "Ask the client for the weather",
    inputs: [stringProperty({ title: "city" })],
  });

  it("interrupts with the client_tool_request payload and resumes with the value", async () => {
    const agentSpec = makeAgent({
      name: "weather_agent",
      tools: [clientToolSpec],
    });
    const { agent } = await loadWithFakeLlm(
      agentSpec,
      [
        toolCallMessage("get_weather", { city: "Agadir" }),
        new AIMessage("It is sunny in Agadir."),
      ],
      { checkpointer: new MemorySaver() },
    );
    const config = threadConfig("client-tool-1");

    const first = await agent.invoke(
      { messages: [{ role: "user", content: "Weather in Agadir?" }] },
      config,
    );
    const interrupts = getInterrupts(first);
    expect(interrupts).toHaveLength(1);
    expect(interrupts[0]!.value).toEqual({
      type: "client_tool_request",
      name: "get_weather",
      description: "Ask the client for the weather",
      inputs: { args: [], kwargs: { city: "Agadir" } },
    });

    const second = await agent.invoke(new Command({ resume: "sunny" }), config);
    const messages = messagesOf(second);
    const toolMessage = messages[messages.length - 2]!;
    expect(toolMessage.getType()).toBe("tool");
    expect(String(toolMessage.content)).toContain("sunny");
    expect(messages[messages.length - 1]!.content).toBe(
      "It is sunny in Agadir.",
    );
  });

  it("requires a checkpointer at load time", async () => {
    const agentSpec = makeAgent({ tools: [clientToolSpec] });
    const loader = new FakeLlmAgentSpecLoader([new AIMessage("hello")]);
    await expect(loader.loadComponent(agentSpec)).rejects.toThrow(
      "A Checkpointer is required when using ClientTool 'get_weather'.",
    );
  });
});

describe("requiresConfirmation human-in-the-loop", () => {
  function makeConfirmationFixture() {
    const called = { count: 0 };
    const doubleToolSpec = createServerTool({
      name: "double_tool",
      description: "Doubles input",
      inputs: [integerProperty({ title: "x" })],
      outputs: [integerProperty({ title: "result" })],
      requiresConfirmation: true,
    });
    const agentSpec = makeAgent({ tools: [doubleToolSpec] });
    const registry = {
      double_tool: (input: unknown) => {
        called.count += 1;
        return (input as { x: number }).x * 2;
      },
    };
    return { agentSpec, registry, called };
  }

  async function invokeUntilInterrupt(
    agent: LoadedReactAgent,
    config: { configurable: { thread_id: string } },
  ): Promise<Record<string, unknown>> {
    const result = await agent.invoke(
      { messages: [{ role: "user", content: "Double 5" }] },
      config,
    );
    const interrupts = getInterrupts(result);
    expect(interrupts).toHaveLength(1);
    return interrupts[0]!.value as Record<string, unknown>;
  }

  it("interrupts with action_requests and executes the tool on approve", async () => {
    const { agentSpec, registry, called } = makeConfirmationFixture();
    const { agent } = await loadWithFakeLlm(
      agentSpec,
      [toolCallMessage("double_tool", { x: 5 }), new AIMessage("Done")],
      { toolRegistry: registry, checkpointer: new MemorySaver() },
    );
    const config = threadConfig("confirm-approve");

    const payload = await invokeUntilInterrupt(agent, config);
    const actionRequests = payload["action_requests"] as Array<
      Record<string, unknown>
    >;
    expect(actionRequests[0]!["name"]).toBe("double_tool");
    expect(actionRequests[0]!["arguments"]).toEqual({ x: 5 });
    expect(String(actionRequests[0]!["description"])).toContain(
      "Tool execution pending approval",
    );
    const reviewConfigs = payload["review_configs"] as Array<
      Record<string, unknown>
    >;
    expect(reviewConfigs[0]!["action_name"]).toBe("double_tool");
    expect(reviewConfigs[0]!["allowed_decisions"]).toEqual([
      "approve",
      "reject",
    ]);

    const result = await agent.invoke(approveCommand(), config);
    expect(called.count).toBe(1);
    const messages = messagesOf(result);
    const toolMessage = messages[messages.length - 2]!;
    expect(toolMessage.getType()).toBe("tool");
    expect(String(toolMessage.content)).toContain("10");
    expect(messages[messages.length - 1]!.content).toBe("Done");
  });

  it("raises the denial error on reject and does not execute the tool", async () => {
    const { agentSpec, registry, called } = makeConfirmationFixture();
    const { agent } = await loadWithFakeLlm(
      agentSpec,
      [toolCallMessage("double_tool", { x: 5 }), new AIMessage("Done")],
      { toolRegistry: registry, checkpointer: new MemorySaver() },
    );
    const config = threadConfig("confirm-reject");

    await invokeUntilInterrupt(agent, config);
    await expect(agent.invoke(rejectCommand("no"), config)).rejects.toThrow(
      "Tool 'double_tool' was denied by the user (reason: no).",
    );
    expect(called.count).toBe(0);
  });

  it("raises the Python validation error on a malformed resume payload", async () => {
    const { agentSpec, registry, called } = makeConfirmationFixture();
    const { agent } = await loadWithFakeLlm(
      agentSpec,
      [toolCallMessage("double_tool", { x: 5 }), new AIMessage("Done")],
      { toolRegistry: registry, checkpointer: new MemorySaver() },
    );
    const config = threadConfig("confirm-malformed");

    await invokeUntilInterrupt(agent, config);
    await expect(
      agent.invoke(new Command({ resume: { not_decisions: [] } }), config),
    ).rejects.toThrow(
      "Tool confirmation result for tool double_tool is not valid, " +
        "should be a dict with a 'decisions' key",
    );
    expect(called.count).toBe(0);
  });

  it("requires a checkpointer at load time", async () => {
    const { agentSpec, registry } = makeConfirmationFixture();
    const loader = new FakeLlmAgentSpecLoader([new AIMessage("hello")], {
      toolRegistry: registry,
    });
    await expect(loader.loadComponent(agentSpec)).rejects.toThrow(
      "A Checkpointer is required for tool 'double_tool' because requires_confirmation=True",
    );
  });
});

describe("structured outputs", () => {
  const outputs = [
    integerProperty({ title: "temperature_rating" }),
    stringProperty({ title: "weather" }),
  ];

  it("configures a response format and appends the structured-output sentence", async () => {
    const systemPrompt = "You are a helpful agent.";
    const agentSpec = makeAgent({ systemPrompt, outputs });
    const { agent } = await loadWithFakeLlm(agentSpec, [
      toolCallMessage("AgentOutputModel", {
        temperature_rating: 8,
        weather: "sunny",
      }),
    ]);

    expect(agent.options.responseFormat).toBeDefined();
    expect(agent.options.systemPrompt).toBe(
      systemPrompt + STRUCTURED_OUTPUT_PROMPT_SUFFIX,
    );
    expect(Object.keys(agent.graph.builder.channels)).toContain(
      "structuredResponse",
    );

    const result = await agent.invoke({
      messages: [{ role: "user", content: "Rate the weather" }],
    });
    expect(result["structuredResponse"]).toEqual({
      temperature_rating: 8,
      weather: "sunny",
    });
  });

  it("does not alter the prompt nor add a response format without outputs", async () => {
    const systemPrompt = "You are a helpful agent.";
    const agentSpec = makeAgent({ systemPrompt });
    const { agent } = await loadWithFakeLlm(agentSpec, [new AIMessage("hi")]);

    expect(agent.options.responseFormat).toBeUndefined();
    expect(agent.options.systemPrompt).toBe(systemPrompt);
    expect(Object.keys(agent.graph.builder.channels)).not.toContain(
      "structuredResponse",
    );
  });
});

describe("middleware plumbing", () => {
  it("does not pass the middleware option when omitted", async () => {
    const { agent } = await loadWithFakeLlm(makeAgent(), [new AIMessage("hi")]);
    expect("middleware" in agent.options).toBe(false);
  });

  it("does not pass the middleware option when empty", async () => {
    const { agent } = await loadWithFakeLlm(makeAgent(), [new AIMessage("hi")], {
      middleware: [],
    });
    expect("middleware" in agent.options).toBe(false);
  });

  it("forwards middleware in order", async () => {
    const middlewareA = createMiddleware({ name: "MwA" });
    const middlewareB = createMiddleware({ name: "MwB" });
    const { agent } = await loadWithFakeLlm(makeAgent(), [new AIMessage("hi")], {
      middleware: [middlewareA, middlewareB],
    });
    expect(agent.options.middleware).toHaveLength(2);
    expect(agent.options.middleware![0]).toBe(middlewareA);
    expect(agent.options.middleware![1]).toBe(middlewareB);
  });

  it("copies the middleware list at construction time", async () => {
    const middlewareA = createMiddleware({ name: "MwA" });
    const middlewareList: unknown[] = [middlewareA];
    const loader = new FakeLlmAgentSpecLoader([new AIMessage("hi")], {
      middleware: middlewareList,
    });
    middlewareList.push(createMiddleware({ name: "MwLate" }));
    const agent = (await loader.loadComponent(makeAgent())) as LoadedReactAgent;
    expect(agent.options.middleware).toHaveLength(1);
    expect(agent.options.middleware![0]).toBe(middlewareA);
  });

  it("middleware hooks execute during agent runs", async () => {
    const hookCalls: string[] = [];
    const recordingMiddleware = createMiddleware({
      name: "RecordingMw",
      beforeModel: () => {
        hookCalls.push("beforeModel");
        return undefined;
      },
    });
    const { agent } = await loadWithFakeLlm(makeAgent(), [new AIMessage("hi")], {
      middleware: [recordingMiddleware],
    });
    expect(Object.keys(agent.graph.builder.nodes)).toContain(
      "RecordingMw.before_model",
    );
    await agent.invoke({ messages: [{ role: "user", content: "Hello" }] });
    expect(hookCalls).toEqual(["beforeModel"]);
  });
});

describe("disaggregated configurations", () => {
  const llmConfig = makeLlmConfig({ name: "llm_config" });
  const agentSpec = makeAgent({ name: "disagg_agent", llmConfig });
  const [mainYaml, disagYaml] = new AgentSpecSerializer().toYaml(agentSpec, {
    disaggregatedComponents: [llmConfig],
    exportDisaggregatedComponents: true,
  }) as [string, string];

  it("importOnlyReferencedComponents loads the referenced components alone", async () => {
    const loader = new FakeLlmAgentSpecLoader([new AIMessage("hi")]);
    const runtimeComponents = (await loader.loadYaml(disagYaml, {
      importOnlyReferencedComponents: true,
    })) as Record<string, unknown>;
    expect(Object.keys(runtimeComponents)).toEqual([llmConfig.id]);
    expect(runtimeComponents[llmConfig.id]).toBeInstanceOf(
      FakeToolCallingChatModel,
    );
  });

  it("componentsRegistry can swap in a fresh Agent Spec LLM config", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const replacement = createVllmConfig({
        name: "llm_config",
        url: "http://localhost:9000",
        modelId: "swapped-model",
      });
      const loader = new FakeLlmAgentSpecLoader([new AIMessage("hi")]);
      const agent = (await loader.loadYaml(mainYaml, {
        componentsRegistry: { [llmConfig.id]: replacement },
      })) as LoadedReactAgent;
      expect(agent.graph.lg_is_pregel).toBe(true);
      expect(agent.graph.getName()).toBe("disagg_agent");
      expect(
        loader.convertedLlmConfigs.map((config) => config.modelId),
      ).toContain("swapped-model");
    } finally {
      warnSpy.mockRestore();
    }
  });

  it("componentsRegistry accepts a runtime chat model", async () => {
    const chatModel = new ChatOpenAI({
      model: "exported-model",
      apiKey: "test-key",
      configuration: { baseURL: "http://localhost:8000/v1" },
    });
    const loader = new FakeLlmAgentSpecLoader([new AIMessage("hi")]);
    const agent = (await loader.loadYaml(mainYaml, {
      componentsRegistry: { [llmConfig.id]: chatModel },
    })) as LoadedReactAgent;
    expect(agent.graph.lg_is_pregel).toBe(true);
    expect(
      loader.convertedLlmConfigs.map((config) => config.modelId),
    ).toContain("exported-model");
  });
});

describe("component load policy", () => {
  const stdioTransport = createStdioTransport({
    name: "stdio_transport",
    command: "echo",
  });
  const mcpToolSpec = createMCPTool({
    name: "fooza_tool",
    clientTransport: stdioTransport,
  });

  it("blocks StdioTransport by default", async () => {
    const loader = new AgentSpecLoader();
    await expect(loader.loadComponent(mcpToolSpec)).rejects.toThrow(
      "Loading Agent Spec component type `StdioTransport` is in the block list.",
    );
  });

  it("blockedComponents: [] unblocks StdioTransport", async () => {
    const cachedTool = tool(
      (input: unknown) => {
        const { a, b } = input as { a: number; b: number };
        return a * 2 + b * 3 - 1;
      },
      {
        name: "fooza_tool",
        description: "fooza",
        schema: {
          title: "FoozaArgs",
          type: "object",
          properties: {
            a: { title: "a", type: "number" },
            b: { title: "b", type: "number" },
          },
          required: ["a", "b"],
        },
      },
    );
    // Pre-seeding the `${transportId}::${toolName}` registry cache keeps the
    // test offline: the adapter reuses cached MCP tools without connecting.
    const loader = new AgentSpecLoader({
      blockedComponents: [],
      toolRegistry: { [`${stdioTransport.id}::fooza_tool`]: cachedTool },
    });
    const converted = (await loader.loadComponent(
      mcpToolSpec,
    )) as StructuredToolInterface;
    expect(converted).toBe(cachedTool);
    expect(await converted.invoke({ a: 2, b: 5 })).toBe(18);
  });
});
