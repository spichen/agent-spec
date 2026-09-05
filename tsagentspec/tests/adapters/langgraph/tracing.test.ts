/**
 * Tracing tests for the LangGraph adapter.
 *
 * Port of `pyagentspec/tests/adapters/langgraph/test_tracing_async.py` (the
 * authoritative suite for the async-only TS API) plus the sync suite's
 * extras: the `tcid__` correlation test and the exact tool-payload flow test.
 * Python's sync-vs-async processor segregation and the NotImplementedError
 * fallback tests are N/A with a single async API. All tests run offline: LLM
 * calls go through fakes injected at the converter seam (carrying the
 * adapter's LLM callback handler, as `convertLlmConfig` does for real
 * models), and MCP loading is mocked like in `mcp.test.ts`.
 */
import { describe, expect, it, vi } from "vitest";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import type { BaseChatModelParams } from "@langchain/core/language_models/chat_models";
import { AIMessage, AIMessageChunk, HumanMessage } from "@langchain/core/messages";
import type { BaseMessage, ToolCallChunk } from "@langchain/core/messages";
import { ChatGenerationChunk } from "@langchain/core/outputs";
import {
  AgentExecutionEnd,
  AgentExecutionSpan,
  AgentExecutionStart,
  Event,
  ExceptionRaised,
  FlowExecutionEnd,
  FlowExecutionSpan,
  FlowExecutionStart,
  LlmGenerationChunkReceived,
  LlmGenerationRequest,
  LlmGenerationResponse,
  LlmGenerationSpan,
  ManagerWorkersExecutionEnd,
  ManagerWorkersExecutionSpan,
  ManagerWorkersExecutionStart,
  NodeExecutionEnd,
  NodeExecutionSpan,
  NodeExecutionStart,
  Span,
  SpanProcessor,
  ToolExecutionRequest,
  ToolExecutionResponse,
  ToolExecutionSpan,
  Trace,
  createAgent as createAgentSpecAgent,
  createCatchExceptionNode,
  createClientTool,
  createFlow,
  createManagerWorkers,
  createSSETransport,
  createServerTool,
  createToolNode,
  getCurrentSpan,
  integerProperty,
  objectProperty,
  stringProperty,
} from "../../../src/index.js";
import type { LlmConfig, MCPTool } from "../../../src/index.js";
import {
  convertClientTransport,
  getOrCreateMcpTools,
} from "../../../src/adapters/langgraph/mcp.js";
import { convertLlmConfig } from "../../../src/adapters/langgraph/llm.js";
import {
  convertClientTool,
  convertServerTool,
} from "../../../src/adapters/langgraph/tools.js";
import {
  AgentSpecLlmCallbackHandler,
  AgentSpecToolCallbackHandler,
  patchWithExecutionSpan,
} from "../../../src/adapters/langgraph/tracing.js";
import {
  FakeLlmAgentSpecLoader,
  FakeToolCallingChatModel,
  ctrl,
  dataEdge,
  detailsOf,
  ioEndNode,
  ioStartNode,
  loadFlow,
  loadWithFakeLlm,
  makeAgent,
  makeLlmConfig,
  outputsOf,
  threadConfig,
  toolCallMessage,
} from "./test-helpers.js";
import { MemorySaver } from "@langchain/langgraph";

const mcpMocks: { tools: unknown[] } = { tools: [] };

vi.mock("@langchain/mcp-adapters", () => ({
  MultiServerMCPClient: class {
    constructor(_config: unknown) {}
    async getTools(..._servers: string[]): Promise<unknown[]> {
      return mcpMocks.tools;
    }
  },
}));

/** Recording processor mirroring the Python tests' DummySpanProcessor. */
class RecordingSpanProcessor extends SpanProcessor {
  startedUp = false;
  shutDown = false;
  starts: Span[] = [];
  ends: Span[] = [];
  events: Array<[Event, Span]> = [];

  onStart(span: Span): void {
    this.starts.push(span);
  }

  onEnd(span: Span): void {
    this.ends.push(span);
  }

  onEvent(event: Event, span: Span): void {
    this.events.push([event, span]);
  }

  startup(): void {
    this.startedUp = true;
  }

  shutdown(): void {
    this.shutDown = true;
  }
}

type SpanCtor = new (...args: never[]) => Span;
type EventCtor<T extends Event> = new (...args: never[]) => T;

function startedSpans(proc: RecordingSpanProcessor, ctor: SpanCtor): Span[] {
  return proc.starts.filter((span) => span instanceof ctor);
}

function endedSpans(proc: RecordingSpanProcessor, ctor: SpanCtor): Span[] {
  return proc.ends.filter((span) => span instanceof ctor);
}

function eventsOf<T extends Event>(
  proc: RecordingSpanProcessor,
  ctor: EventCtor<T>,
): T[] {
  return proc.events
    .map(([event]) => event)
    .filter((event): event is T => event instanceof ctor);
}

/** The invocable + streamable surface of a loaded graph. */
interface RunnableGraph {
  invoke(input: unknown, config?: unknown): Promise<Record<string, unknown>>;
  stream(input: unknown, config?: unknown): Promise<AsyncIterable<unknown>>;
}

/** Port of `_assert_agent_llm_tool_async` (single-API: no sync/async split). */
function assertAgentLlmTool(proc: RecordingSpanProcessor): void {
  expect(proc.startedUp).toBe(true);
  expect(proc.shutDown).toBe(true);

  expect(startedSpans(proc, AgentExecutionSpan).length).toBeGreaterThan(0);
  expect(endedSpans(proc, AgentExecutionSpan).length).toBeGreaterThan(0);
  expect(startedSpans(proc, LlmGenerationSpan).length).toBeGreaterThan(0);
  expect(endedSpans(proc, LlmGenerationSpan).length).toBeGreaterThan(0);
  expect(startedSpans(proc, ToolExecutionSpan).length).toBeGreaterThan(0);
  expect(endedSpans(proc, ToolExecutionSpan).length).toBeGreaterThan(0);

  expect(eventsOf(proc, AgentExecutionStart).length).toBeGreaterThan(0);
  expect(eventsOf(proc, AgentExecutionEnd).length).toBeGreaterThan(0);
  expect(eventsOf(proc, LlmGenerationRequest).length).toBeGreaterThan(0);
  expect(eventsOf(proc, LlmGenerationResponse).length).toBeGreaterThan(0);
  expect(eventsOf(proc, ToolExecutionRequest).length).toBeGreaterThan(0);
  expect(eventsOf(proc, ToolExecutionResponse).length).toBeGreaterThan(0);
}

/** Port of `_assert_flow_async`. */
function assertFlow(
  proc: RecordingSpanProcessor,
  options?: {
    flowTracingHasLlm?: boolean;
    expectedToolResponseOutputs?: Record<string, unknown>;
  },
): void {
  const hasLlm = options?.flowTracingHasLlm ?? true;
  expect(proc.startedUp).toBe(true);
  expect(proc.shutDown).toBe(true);

  expect(startedSpans(proc, FlowExecutionSpan).length).toBeGreaterThan(0);
  expect(endedSpans(proc, FlowExecutionSpan).length).toBeGreaterThan(0);
  expect(startedSpans(proc, NodeExecutionSpan).length).toBeGreaterThan(0);
  expect(endedSpans(proc, NodeExecutionSpan).length).toBeGreaterThan(0);
  expect(startedSpans(proc, ToolExecutionSpan).length).toBeGreaterThan(0);
  expect(endedSpans(proc, ToolExecutionSpan).length).toBeGreaterThan(0);
  expect(startedSpans(proc, LlmGenerationSpan).length > 0).toBe(hasLlm);
  expect(endedSpans(proc, LlmGenerationSpan).length > 0).toBe(hasLlm);

  expect(eventsOf(proc, FlowExecutionStart).length).toBeGreaterThan(0);
  expect(eventsOf(proc, FlowExecutionEnd).length).toBeGreaterThan(0);
  expect(eventsOf(proc, NodeExecutionStart).length).toBeGreaterThan(0);
  expect(eventsOf(proc, NodeExecutionEnd).length).toBeGreaterThan(0);
  expect(eventsOf(proc, ToolExecutionRequest).length).toBeGreaterThan(0);
  expect(eventsOf(proc, ToolExecutionResponse).length).toBeGreaterThan(0);
  expect(eventsOf(proc, LlmGenerationRequest).length > 0).toBe(hasLlm);
  expect(eventsOf(proc, LlmGenerationResponse).length > 0).toBe(hasLlm);

  if (options?.expectedToolResponseOutputs !== undefined) {
    const toolResponseEvents = eventsOf(proc, ToolExecutionResponse);
    expect(toolResponseEvents).toHaveLength(1);
    expect(toolResponseEvents[0]!.outputs).toEqual(
      options.expectedToolResponseOutputs,
    );
  }
}

/** The weather ServerTool spec used by the agent tests. */
function weatherTool() {
  return createServerTool({
    name: "get_weather",
    description: "Retrieves the weather in a city",
    inputs: [stringProperty({ title: "city" })],
    outputs: [stringProperty({ title: "weather" })],
  });
}

const WEATHER_TOOL_REGISTRY = {
  get_weather: async (input: unknown) =>
    `The weather in ${(input as { city: string }).city} is sunny.`,
};

const WEATHER_QUESTION = {
  messages: [{ role: "user", content: "What's the weather in Agadir?" }],
};

/** Fake responses driving one get_weather tool call then a final answer. */
function weatherResponses(): AIMessage[] {
  return [
    toolCallMessage("get_weather", { city: "Agadir" }),
    new AIMessage("The weather in Agadir is sunny."),
  ];
}

/**
 * The converter seam substitutes whole fake models, so the tests attach the
 * adapter's LLM handler exactly where `convertLlmConfig` attaches it for real
 * models: on the chat model's constructor callbacks.
 */
function fakeModelWithLlmTracing(responses: AIMessage[]) {
  return (llmConfig: LlmConfig): FakeToolCallingChatModel =>
    new FakeToolCallingChatModel({
      responses,
      callbacks: [new AgentSpecLlmCallbackHandler(llmConfig)],
    });
}

async function loadWeatherAgent(): Promise<RunnableGraph> {
  const { agent } = await loadWithFakeLlm(
    makeAgent({ tools: [weatherTool()] }),
    fakeModelWithLlmTracing(weatherResponses()),
    { toolRegistry: WEATHER_TOOL_REGISTRY },
  );
  return agent as unknown as RunnableGraph;
}

/**
 * Streaming fake: one scripted list of AIMessageChunks per model turn,
 * reported to the run manager chunk by chunk (`handleLLMNewToken`) exactly
 * like a real provider model's `_streamResponseChunks`.
 */
class StreamingFakeChatModel extends FakeToolCallingChatModel {
  private readonly turns: AIMessageChunk[][];
  private streamTurnIdx = 0;

  constructor(
    fields: { turns: AIMessageChunk[][]; responses?: AIMessage[] } & BaseChatModelParams,
  ) {
    super({ ...fields, responses: fields.responses ?? [] });
    this.turns = fields.turns;
  }

  override _llmType(): string {
    return "streaming-fake-chat-model";
  }

  override async *_streamResponseChunks(
    _messages: BaseMessage[],
    _options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    const turn =
      this.turns[Math.min(this.streamTurnIdx, this.turns.length - 1)]!;
    this.streamTurnIdx += 1;
    for (const messageChunk of turn) {
      const text =
        typeof messageChunk.content === "string" ? messageChunk.content : "";
      const generationChunk = new ChatGenerationChunk({
        message: messageChunk,
        text,
      });
      await runManager?.handleLLMNewToken(
        text,
        { prompt: 0, completion: 0 },
        undefined,
        undefined,
        undefined,
        { chunk: generationChunk },
      );
      yield generationChunk;
    }
  }
}

function toolCallChunkMessage(
  id: string,
  toolCallChunk: Partial<ToolCallChunk>,
): AIMessageChunk {
  return new AIMessageChunk({
    content: "",
    id,
    tool_call_chunks: [
      { ...toolCallChunk, index: 0, type: "tool_call_chunk" } as ToolCallChunk,
    ],
  });
}

function textChunkMessage(id: string, content: string): AIMessageChunk {
  return new AIMessageChunk({ content, id });
}

/** Two streamed turns: a chunked get_weather tool call, then a text answer. */
function streamingWeatherTurns(): AIMessageChunk[][] {
  return [
    [
      toolCallChunkMessage("msg_1", {
        name: "get_weather",
        args: "",
        id: "call_1",
      }),
      toolCallChunkMessage("msg_1", { args: '{"city":' }),
      toolCallChunkMessage("msg_1", { args: '"Agadir"}' }),
    ],
    [
      textChunkMessage("msg_2", "The weather in Agadir "),
      textChunkMessage("msg_2", "is sunny."),
    ],
  ];
}

/** The `double_tool` flow of the Python async server-tool flow test. */
function doubleToolFlow() {
  const xProp = integerProperty({ title: "x" });
  const resultProp = integerProperty({ title: "result" });
  const serverTool = createServerTool({
    name: "double_tool",
    description: "Doubles the input number",
    inputs: [xProp],
    outputs: [resultProp],
  });
  const startNode = ioStartNode("start", [xProp]);
  const toolNode = createToolNode({ name: "tool", tool: serverTool });
  const endNode = ioEndNode("end", [resultProp]);
  return createFlow({
    name: "flow",
    startNode,
    nodes: [startNode, toolNode, endNode],
    controlFlowConnections: [ctrl(startNode, toolNode), ctrl(toolNode, endNode)],
    dataFlowConnections: [
      dataEdge(startNode, toolNode, "x"),
      dataEdge(toolNode, endNode, "result"),
    ],
    inputs: [xProp],
    outputs: [resultProp],
  });
}

const DOUBLE_TOOL_REGISTRY = {
  double_tool: async (input: unknown) => (input as { x: number }).x * 2,
};

describe("langgraph adapter tracing", () => {
  it("invoke emits agent, LLM and tool spans and events", async () => {
    const agent = await loadWeatherAgent();

    const proc = new RecordingSpanProcessor();
    let response: Record<string, unknown> = {};
    await new Trace({
      name: "langgraph_tracing_async_test",
      spanProcessors: [proc],
    }).run(async () => {
      response = await agent.invoke(WEATHER_QUESTION);
    });

    expect(JSON.stringify(response).toLowerCase()).toContain("sunny");
    assertAgentLlmTool(proc);
  });

  it("stream emits agent, LLM and tool spans and events", async () => {
    const agent = await loadWeatherAgent();

    const proc = new RecordingSpanProcessor();
    let response = "";
    await new Trace({
      name: "langgraph_tracing_async_test",
      spanProcessors: [proc],
    }).run(async () => {
      const stream = await agent.stream(WEATHER_QUESTION, {
        streamMode: "messages",
      });
      for await (const chunk of stream) {
        const [messageChunk] = chunk as [{ content?: unknown }, unknown];
        if (typeof messageChunk?.content === "string") {
          response += messageChunk.content;
        }
      }
    });

    expect(response.toLowerCase()).toContain("sunny");
    assertAgentLlmTool(proc);
  });

  it("agent execution span carries the agent name and start/end payloads", async () => {
    const agent = await loadWeatherAgent();

    const proc = new RecordingSpanProcessor();
    await new Trace({ spanProcessors: [proc] }).run(async () => {
      await agent.invoke(WEATHER_QUESTION);
    });

    const agentSpans = startedSpans(proc, AgentExecutionSpan);
    expect(agentSpans).toHaveLength(1);
    expect(agentSpans[0]!.name).toBe("AgentExecution[test_agent]");
    const startEvents = eventsOf(proc, AgentExecutionStart);
    expect(startEvents).toHaveLength(1);
    // The invocation input state is reported as the start-event inputs.
    expect(startEvents[0]!.inputs).toEqual(WEATHER_QUESTION);
    // No declared agent outputs: the end event reports an empty mapping.
    const endEvents = eventsOf(proc, AgentExecutionEnd);
    expect(endEvents).toHaveLength(1);
    expect(endEvents[0]!.outputs).toEqual({});
  });

  it("maps LangChain roles onto OpenAI roles in the request prompt", async () => {
    const agent = await loadWeatherAgent();

    const proc = new RecordingSpanProcessor();
    await new Trace({ spanProcessors: [proc] }).run(async () => {
      await agent.invoke(WEATHER_QUESTION);
    });

    const requests = eventsOf(proc, LlmGenerationRequest);
    expect(requests).toHaveLength(2);
    // Turn 1: system prompt + user question.
    expect(requests[0]!.prompt.map((m) => m.role)).toEqual(["system", "user"]);
    // Turn 2: the tool loop appended the assistant tool call and tool result.
    expect(requests[1]!.prompt.map((m) => m.role)).toEqual([
      "system",
      "user",
      "assistant",
      "tool",
    ]);
    for (const request of requests) {
      expect(request.llmConfig.name).toBe("test-llm");
      for (const message of request.prompt) {
        expect(message.sender).toBe("");
        expect(typeof message.content).toBe("string");
      }
    }
  });

  it("synthesizes request tools from the model's invocation params", async () => {
    class InvocationParamsFakeModel extends FakeToolCallingChatModel {
      override invocationParams(): Record<string, unknown> {
        return {
          tools: [
            {
              type: "function",
              function: {
                name: "get_weather",
                description: "Retrieves the weather in a city",
                parameters: {
                  type: "object",
                  properties: { city: { type: "string" } },
                },
              },
            },
          ],
        };
      }
    }
    const { agent } = await loadWithFakeLlm(
      makeAgent({ tools: [weatherTool()] }),
      (llmConfig) =>
        new InvocationParamsFakeModel({
          responses: weatherResponses(),
          callbacks: [new AgentSpecLlmCallbackHandler(llmConfig)],
        }),
      { toolRegistry: WEATHER_TOOL_REGISTRY },
    );

    const proc = new RecordingSpanProcessor();
    await new Trace({ spanProcessors: [proc] }).run(async () => {
      await (agent as unknown as RunnableGraph).invoke(WEATHER_QUESTION);
    });

    const request = eventsOf(proc, LlmGenerationRequest)[0]!;
    expect(request.tools).toHaveLength(1);
    const requestTool = request.tools[0]!;
    // ClientTool is the generic Tool carrier for invocation-params tools.
    expect(requestTool.componentType).toBe("ClientTool");
    expect(requestTool.name).toBe("get_weather");
    expect(requestTool.description).toBe("Retrieves the weather in a city");
    expect(requestTool.inputs?.map((input) => input.title)).toEqual(["city"]);
    expect(requestTool.inputs?.[0]?.jsonSchema).toEqual({
      type: "string",
      title: "city",
    });
  });

  it("emits chunk events with tool-call carry-forward while streaming", async () => {
    const { agent } = await loadWithFakeLlm(
      makeAgent({ tools: [weatherTool()] }),
      (llmConfig) =>
        new StreamingFakeChatModel({
          turns: streamingWeatherTurns(),
          callbacks: [new AgentSpecLlmCallbackHandler(llmConfig)],
        }),
      { toolRegistry: WEATHER_TOOL_REGISTRY },
    );

    const proc = new RecordingSpanProcessor();
    await new Trace({ spanProcessors: [proc] }).run(async () => {
      const stream = await (agent as unknown as RunnableGraph).stream(
        WEATHER_QUESTION,
        { streamMode: "messages" },
      );
      for await (const _chunk of stream) {
        // Drain the stream; the assertions read the recorded events.
      }
    });

    const chunkEvents = eventsOf(proc, LlmGenerationChunkReceived);
    const toolCallChunks = chunkEvents.filter(
      (event) => event.toolCalls.length === 1,
    );
    expect(toolCallChunks).toHaveLength(3);
    // The id+name announced by the first chunk carry forward to the
    // args-delta chunks; arguments stay deltas, not accumulations.
    for (const event of toolCallChunks) {
      expect(event.toolCalls[0]!.callId).toBe("call_1");
      expect(event.toolCalls[0]!.toolName).toBe("get_weather");
      expect(event.completionId).toBe("msg_1");
    }
    expect(toolCallChunks.map((event) => event.toolCalls[0]!.arguments)).toEqual(
      ["", '{"city":', '"Agadir"}'],
    );

    // Text chunks of the final turn carry content and no tool calls.
    const textChunks = chunkEvents.filter(
      (event) => event.completionId === "msg_2",
    );
    expect(textChunks.map((event) => event.content)).toEqual([
      "The weather in Agadir ",
      "is sunny.",
    ]);
    for (const event of textChunks) {
      expect(event.toolCalls).toEqual([]);
    }

    // Chunk events share their turn's request id, and the streamed response
    // aggregates the tool call with the full JSON arguments.
    const requests = eventsOf(proc, LlmGenerationRequest);
    expect(requests).toHaveLength(2);
    const firstTurnRequestId = requests[0]!.requestId;
    for (const event of toolCallChunks) {
      expect(event.requestId).toBe(firstTurnRequestId);
    }
    const responses = eventsOf(proc, LlmGenerationResponse);
    expect(responses).toHaveLength(2);
    expect(responses[0]!.requestId).toBe(firstTurnRequestId);
    expect(responses[0]!.completionId).toBe("msg_1");
    expect(responses[0]!.toolCalls).toHaveLength(1);
    expect(responses[0]!.toolCalls[0]!.callId).toBe("call_1");
    expect(responses[0]!.toolCalls[0]!.toolName).toBe("get_weather");
    expect(responses[0]!.toolCalls[0]!.arguments).toBe('{"city":"Agadir"}');
  });

  it("streams tool_call_ids consistent with the executed tool spans", async () => {
    // Port of the sync suite's
    // test_langgraph_agent_emits_tool_calls_and_results_with_consistent_ids.
    const { agent } = await loadWithFakeLlm(
      makeAgent({ tools: [weatherTool()] }),
      (llmConfig) =>
        new StreamingFakeChatModel({
          turns: streamingWeatherTurns(),
          callbacks: [new AgentSpecLlmCallbackHandler(llmConfig)],
        }),
      { toolRegistry: WEATHER_TOOL_REGISTRY },
    );

    const proc = new RecordingSpanProcessor();
    await new Trace({ spanProcessors: [proc] }).run(async () => {
      const stream = await (agent as unknown as RunnableGraph).stream(
        WEATHER_QUESTION,
        { streamMode: "messages" },
      );
      for await (const _chunk of stream) {
        // Drain the stream.
      }
    });

    const streamedToolCallIds = new Set(
      eventsOf(proc, LlmGenerationChunkReceived)
        .filter((event) => event.toolCalls.length === 1)
        .map((event) => event.toolCalls[0]!.callId),
    );
    // LangChain can stream provisional tool_call_ids that get abandoned
    // before execution, so executed ids must be a subset of streamed ids.
    const executedToolCallIds = new Set(
      proc.events
        .map(([, span]) => span)
        .filter((span): span is ToolExecutionSpan => span instanceof ToolExecutionSpan)
        .filter((span) => span.description !== "")
        .map((span) => span.description.replace("tcid__", "")),
    );
    expect(executedToolCallIds.size).toBeGreaterThan(0);
    for (const executedId of executedToolCallIds) {
      expect(streamedToolCallIds.has(executedId)).toBe(true);
    }
  });

  it("tool spans map the ToolMessage output onto the declared outputs", async () => {
    const agent = await loadWeatherAgent();

    const proc = new RecordingSpanProcessor();
    await new Trace({ spanProcessors: [proc] }).run(async () => {
      await agent.invoke(WEATHER_QUESTION);
    });

    const toolSpans = startedSpans(proc, ToolExecutionSpan) as ToolExecutionSpan[];
    expect(toolSpans).toHaveLength(1);
    expect(toolSpans[0]!.name).toBe("ToolExecution[get_weather]");
    // The react-agent tool node runs tools with their tool_call_id, smuggled
    // through the span description for correlation.
    expect(toolSpans[0]!.description).toBe("tcid__call_1");
    expect(toolSpans[0]!.tool.componentType).toBe("ServerTool");

    const requests = eventsOf(proc, ToolExecutionRequest);
    expect(requests).toHaveLength(1);
    expect(requests[0]!.inputs).toEqual({ city: "Agadir" });
    const responses = eventsOf(proc, ToolExecutionResponse);
    expect(responses).toHaveLength(1);
    expect(responses[0]!.outputs).toEqual({
      weather: "The weather in Agadir is sunny.",
    });
    expect(responses[0]!.requestId).toBe(requests[0]!.requestId);
  });

  it("invoke emits flow, node and async server tool events", async () => {
    // Port of test_langgraph_ainvoke_tracing_emits_async_server_tool_events_for_flow.
    const graph = (await loadFlow(doubleToolFlow(), {
      toolRegistry: DOUBLE_TOOL_REGISTRY,
    })) as unknown as RunnableGraph;

    const proc = new RecordingSpanProcessor();
    let response: Record<string, unknown> = {};
    await new Trace({
      name: "langgraph_tracing_async_server_tool_test",
      spanProcessors: [proc],
    }).run(async () => {
      response = await graph.invoke({ inputs: { x: 5 } });
    });

    expect(outputsOf(response)).toEqual({ result: 10 });
    assertFlow(proc, {
      flowTracingHasLlm: false,
      expectedToolResponseOutputs: { result: 10 },
    });

    // Node spans wrap every flow node, named `<NodeType>Execution[<name>]`.
    const nodeSpanNames = startedSpans(proc, NodeExecutionSpan).map(
      (span) => span.name,
    );
    expect(nodeSpanNames).toEqual([
      "StartNodeExecution[start]",
      "ToolNodeExecution[tool]",
      "EndNodeExecution[end]",
    ]);
    expect(endedSpans(proc, NodeExecutionSpan)).toHaveLength(3);

    // The flow execution span and the end event's payload.
    const flowSpans = startedSpans(proc, FlowExecutionSpan);
    expect(flowSpans).toHaveLength(1);
    expect(flowSpans[0]!.name).toBe("FlowExecution[flow]");
    const flowStart = eventsOf(proc, FlowExecutionStart)[0]!;
    expect(flowStart.inputs).toEqual({ inputs: { x: 5 } });
    const flowEnd = eventsOf(proc, FlowExecutionEnd)[0]!;
    expect(flowEnd.outputs).toEqual({ result: 10 });
    expect(flowEnd.branchSelected).toBe(String(detailsOf(response)["branch"]));

    // The tool node's end event carries the mapped outputs and branch.
    const toolNodeEnd = eventsOf(proc, NodeExecutionEnd).find(
      (event) => event.node.name === "tool",
    )!;
    expect(toolNodeEnd.outputs).toEqual({ result: 10 });
  });

  it("stream emits flow events while yielding value chunks", async () => {
    // Port of test_langgraph_astream_tracing_emits_flow_events.
    const graph = (await loadFlow(doubleToolFlow(), {
      toolRegistry: DOUBLE_TOOL_REGISTRY,
    })) as unknown as RunnableGraph;

    const proc = new RecordingSpanProcessor();
    let lastChunk: Record<string, unknown> = {};
    await new Trace({
      name: "langgraph_tracing_async_test",
      spanProcessors: [proc],
    }).run(async () => {
      const stream = await graph.stream(
        { inputs: { x: 5 } },
        { streamMode: "values" },
      );
      for await (const chunk of stream) {
        if (chunk) {
          lastChunk = chunk as Record<string, unknown>;
        }
      }
    });

    expect(outputsOf(lastChunk)).toEqual({ result: 10 });
    assertFlow(proc, {
      flowTracingHasLlm: false,
      expectedToolResponseOutputs: { result: 10 },
    });
  });

  it("collects exact tool inputs and outputs in a flow", async () => {
    // Port of the sync suite's test_langgraph_flow_tracing_collects_tool_inputs.
    const cityProperty = stringProperty({ title: "city" });
    const forecastProperty = objectProperty({
      title: "forecast",
      properties: {
        city: stringProperty({ title: "city" }),
        condition: stringProperty({ title: "condition" }),
      },
    });
    const weatherFlowTool = createServerTool({
      name: "get_weather",
      description: "Retrieves the weather in a city",
      inputs: [cityProperty],
      outputs: [forecastProperty],
    });
    const startNode = ioStartNode("start", [cityProperty]);
    const toolNode = createToolNode({ name: "tool", tool: weatherFlowTool });
    const endNode = ioEndNode("end", [forecastProperty]);
    const flow = createFlow({
      name: "weather_flow",
      startNode,
      nodes: [startNode, toolNode, endNode],
      controlFlowConnections: [
        ctrl(startNode, toolNode),
        ctrl(toolNode, endNode),
      ],
      dataFlowConnections: [
        dataEdge(startNode, toolNode, "city"),
        dataEdge(toolNode, endNode, "forecast"),
      ],
      inputs: [cityProperty],
      outputs: [forecastProperty],
    });
    const graph = (await loadFlow(flow, {
      toolRegistry: {
        get_weather: (input: unknown) => ({
          city: (input as { city: string }).city,
          condition: "sunny",
        }),
      },
    })) as unknown as RunnableGraph;

    const proc = new RecordingSpanProcessor();
    let response: Record<string, unknown> = {};
    await new Trace({
      name: "langgraph_tool_input_trace_test",
      spanProcessors: [proc],
    }).run(async () => {
      response = await graph.invoke({ inputs: { city: "Agadir" } });
    });

    expect(outputsOf(response)).toEqual({
      forecast: { city: "Agadir", condition: "sunny" },
    });
    const toolRequestEvents = eventsOf(proc, ToolExecutionRequest);
    expect(toolRequestEvents).toHaveLength(1);
    expect(toolRequestEvents[0]!.inputs).toEqual({ city: "Agadir" });
    const toolResponseEvents = eventsOf(proc, ToolExecutionResponse);
    expect(toolResponseEvents).toHaveLength(1);
    expect(toolResponseEvents[0]!.outputs).toEqual({
      forecast: { city: "Agadir", condition: "sunny" },
    });
  });

  it("wraps manager-workers runs in an execution span with worker agent spans", async () => {
    const spec = createManagerWorkers({
      name: "Team",
      groupManager: createAgentSpecAgent({
        name: "Coordinator",
        llmConfig: makeLlmConfig({ name: "manager_llm" }),
        systemPrompt: "You coordinate.",
      }),
      workers: [
        createAgentSpecAgent({
          name: "Research Helper",
          llmConfig: makeLlmConfig({ name: "worker_llm" }),
          systemPrompt: "You research.",
          description: "Handles research",
        }),
      ],
    });
    const queues: Record<string, AIMessage[]> = {
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
    };
    const loader = new FakeLlmAgentSpecLoader(
      (llmConfig) =>
        new FakeToolCallingChatModel({
          responses: queues[llmConfig.name]!,
          callbacks: [new AgentSpecLlmCallbackHandler(llmConfig)],
        }),
      { checkpointer: new MemorySaver() },
    );
    const graph = (await loader.loadComponent(spec)) as unknown as RunnableGraph;

    const proc = new RecordingSpanProcessor();
    let response: Record<string, unknown> = {};
    await new Trace({ spanProcessors: [proc] }).run(async () => {
      response = await graph.invoke(
        { messages: [new HumanMessage("Tell me about Saturn.")] },
        threadConfig("mw-tracing-1"),
      );
    });

    const mwSpans = startedSpans(proc, ManagerWorkersExecutionSpan);
    expect(mwSpans).toHaveLength(1);
    expect(mwSpans[0]!.name).toBe("ManagerWorkersExecution[Team]");
    expect(endedSpans(proc, ManagerWorkersExecutionSpan)).toHaveLength(1);
    const mwStart = eventsOf(proc, ManagerWorkersExecutionStart);
    expect(mwStart).toHaveLength(1);
    const mwEnd = eventsOf(proc, ManagerWorkersExecutionEnd);
    expect(mwEnd).toHaveLength(1);
    expect(mwEnd[0]!.outputs).toEqual({ messages: response["messages"] });

    // Workers are invoked through their patched react agents, so the
    // delegated turn is wrapped in an AgentExecutionSpan; every model turn
    // (manager and worker) carries an LlmGenerationSpan.
    const workerAgentSpans = startedSpans(proc, AgentExecutionSpan);
    expect(workerAgentSpans.length).toBeGreaterThan(0);
    expect(workerAgentSpans.map((span) => span.name)).toContain(
      "AgentExecution[Research Helper]",
    );
    expect(startedSpans(proc, LlmGenerationSpan).length).toBeGreaterThan(1);
    expect(eventsOf(proc, LlmGenerationRequest).length).toBeGreaterThan(1);
  });

  it("records ExceptionRaised on the CatchExceptionNode span", async () => {
    const xProp = integerProperty({ title: "x" });
    const yProp = stringProperty({ title: "y", default: "" });
    const flakyTool = createServerTool({
      name: "flaky_tool",
      description: "Raises for negative inputs",
      inputs: [xProp],
      outputs: [yProp],
    });
    const subStart = ioStartNode("sub_start", [xProp]);
    const flakyNode = createToolNode({ name: "flaky_node", tool: flakyTool });
    const subEnd = ioEndNode("sub_end", [yProp]);
    const subflow = createFlow({
      name: "subflow",
      startNode: subStart,
      nodes: [subStart, flakyNode, subEnd],
      controlFlowConnections: [ctrl(subStart, flakyNode), ctrl(flakyNode, subEnd)],
      dataFlowConnections: [
        dataEdge(subStart, flakyNode, "x"),
        dataEdge(flakyNode, subEnd, "y"),
      ],
      inputs: [xProp],
      outputs: [yProp],
    });
    const catchNode = createCatchExceptionNode({ name: "catch", subflow });
    const start = ioStartNode("start", [xProp]);
    const end = ioEndNode("end", [yProp]);
    const errorEnd = ioEndNode("error_end", [], "ERROR");
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
      ],
      inputs: [xProp],
      outputs: [yProp],
    });
    const graph = (await loadFlow(flow, {
      toolRegistry: {
        flaky_tool: (input: unknown) => {
          if ((input as { x: number }).x < 0) {
            throw new Error("x must be non-negative");
          }
          return "ok";
        },
      },
    })) as unknown as RunnableGraph;

    const proc = new RecordingSpanProcessor();
    let response: Record<string, unknown> = {};
    await new Trace({ spanProcessors: [proc] }).run(async () => {
      response = await graph.invoke({ inputs: { x: -1 } });
    });

    expect(detailsOf(response)["branch"]).toBe("ERROR");
    // Two ExceptionRaised events: the failing ToolNode span records the
    // propagating error, and the CatchExceptionNode span records the caught
    // one (Python parity for the async path).
    const exceptionPairs = proc.events.filter(
      ([event]) => event instanceof ExceptionRaised,
    ) as Array<[ExceptionRaised, Span]>;
    expect(exceptionPairs).toHaveLength(2);
    for (const [event, span] of exceptionPairs) {
      expect(event.exceptionMessage).toContain("x must be non-negative");
      expect(span).toBeInstanceOf(NodeExecutionSpan);
    }
    const spanNames = exceptionPairs.map(([, span]) => span.name);
    expect(spanNames).toContain("ToolNodeExecution[flaky_node]");
    expect(spanNames).toContain("CatchExceptionNodeExecution[catch]");
    // The outer flow still completes: every started span ended.
    expect(proc.ends).toHaveLength(proc.starts.length);
  });

  it("runs without a Trace: no processors, no events, unchanged behavior", async () => {
    const agent = await loadWeatherAgent();
    const response = await agent.invoke(WEATHER_QUESTION);
    expect(JSON.stringify(response).toLowerCase()).toContain("sunny");
    expect(getCurrentSpan()).toBeUndefined();
  });

  it("attaches the LLM handler to converter-built chat models", async () => {
    const llmConfig = makeLlmConfig({ name: "traced-llm" });
    const model = await convertLlmConfig(llmConfig);
    const callbacks = (model as { callbacks?: unknown }).callbacks;
    expect(Array.isArray(callbacks)).toBe(true);
    const handler = (callbacks as unknown[]).find(
      (callback) => callback instanceof AgentSpecLlmCallbackHandler,
    ) as AgentSpecLlmCallbackHandler | undefined;
    expect(handler).toBeDefined();
    expect(handler!.llmConfig).toBe(llmConfig);
  });

  it("attaches the tool handler to server tools but not client tools", async () => {
    const serverTool = weatherTool();
    const structuredTool = convertServerTool(serverTool, {
      get_weather: WEATHER_TOOL_REGISTRY.get_weather,
    });
    const serverCallbacks = (structuredTool as { callbacks?: unknown[] })
      .callbacks;
    const serverHandler = serverCallbacks?.find(
      (callback) => callback instanceof AgentSpecToolCallbackHandler,
    ) as AgentSpecToolCallbackHandler | undefined;
    expect(serverHandler).toBeDefined();
    expect(serverHandler!.tool).toBe(serverTool);

    const clientTool = createClientTool({
      name: "ask_user",
      inputs: [stringProperty({ title: "question" })],
    });
    const structuredClientTool = convertClientTool(clientTool);
    const clientCallbacks = (structuredClientTool as { callbacks?: unknown[] })
      .callbacks;
    const clientHandler = clientCallbacks?.find(
      (callback) => callback instanceof AgentSpecToolCallbackHandler,
    );
    expect(clientHandler).toBeUndefined();
  });

  it("attaches the tool handler to loaded MCP tools with a synthesized MCPTool", async () => {
    const transport = createSSETransport({
      name: "my server",
      url: "https://example.com/sse",
    });
    mcpMocks.tools = [
      {
        name: "fooza_tool",
        description: "fooza_tool description",
        schema: {
          title: "fooza_tool",
          type: "object",
          properties: { q: { type: "string" } },
        },
      },
    ];

    const tools = await getOrCreateMcpTools(
      transport,
      convertClientTransport(transport),
      {},
    );

    const callbacks = (tools["fooza_tool"] as { callbacks?: unknown[] })
      .callbacks;
    expect(Array.isArray(callbacks)).toBe(true);
    const handler = callbacks!.find(
      (callback) => callback instanceof AgentSpecToolCallbackHandler,
    ) as AgentSpecToolCallbackHandler | undefined;
    expect(handler).toBeDefined();
    const synthesized = handler!.tool as MCPTool;
    expect(synthesized.componentType).toBe("MCPTool");
    expect(synthesized.name).toBe("fooza_tool");
    expect(synthesized.description).toBe("fooza_tool description");
    expect(synthesized.clientTransport).toEqual(transport);
    expect(synthesized.inputs?.map((input) => input.title)).toEqual(["q"]);
    expect(synthesized.outputs?.map((output) => output.title)).toEqual([
      "tool_output",
    ]);
  });
});

/**
 * Regression tests: `patchWithExecutionSpan` forks a child ambient context
 * per run, so concurrent patched invocations in ONE context (ManagerWorkers
 * dispatching several workers at once, a user-level `Promise.all`) keep
 * isolated span stacks — a span ending first must not pop a still-running
 * sibling from the caller's stack. Python is immune because asyncio tasks
 * copy contextvars per task.
 */
describe("patchWithExecutionSpan parallel isolation", () => {
  interface Deferred {
    promise: Promise<void>;
    resolve: () => void;
  }

  function deferred(): Deferred {
    let resolve!: () => void;
    const promise = new Promise<void>((res) => {
      resolve = res;
    });
    return { promise, resolve };
  }

  /** A fake compiled graph whose run signals `started`, then blocks on `gate`. */
  function gatedGraph(started: Deferred, gate: Deferred) {
    return {
      async invoke(_input: unknown): Promise<Record<string, unknown>> {
        started.resolve();
        await gate.promise;
        return { messages: [] };
      },
      async stream(_input: unknown): Promise<AsyncIterable<unknown>> {
        return (async function* () {
          started.resolve();
          await gate.promise;
          yield ["values", { messages: [] }];
        })();
      },
    };
  }

  function gatedPatchedPair() {
    const started1 = deferred();
    const started2 = deferred();
    const gate1 = deferred();
    const gate2 = deferred();
    const graph1 = patchWithExecutionSpan(gatedGraph(started1, gate1), {
      kind: "agent",
      component: makeAgent({ name: "agent1" }),
    });
    const graph2 = patchWithExecutionSpan(gatedGraph(started2, gate2), {
      kind: "agent",
      component: makeAgent({ name: "agent2" }),
    });
    return { started1, started2, gate1, gate2, graph1, graph2 };
  }

  it("keeps concurrent invokes' span stacks isolated in one ambient context", async () => {
    const { started1, started2, gate1, gate2, graph1, graph2 } =
      gatedPatchedPair();
    const proc = new RecordingSpanProcessor();
    const trace = new Trace({ spanProcessors: [proc] });

    await trace.run(async () => {
      const run1 = graph1.invoke({});
      const run2 = graph2.invoke({});
      await Promise.all([started1.promise, started2.promise]);

      // Finish the first-started run while the second is still in flight.
      gate1.resolve();
      await run1;
      // The caller's ambient stack is untouched: agent1's end must not have
      // popped agent2's still-running span, nor left ended-agent1 as current.
      expect(getCurrentSpan()).toBe(trace.rootSpan);
      const endedSoFar = endedSpans(proc, AgentExecutionSpan);
      expect(endedSoFar).toHaveLength(1);
      expect(endedSoFar[0]!.name).toBe("AgentExecution[agent1]");

      gate2.resolve();
      await run2;
    });

    const agentSpans = startedSpans(proc, AgentExecutionSpan);
    expect(agentSpans).toHaveLength(2);
    // Parallel runs are siblings under the root, never nested under each other.
    for (const span of agentSpans) {
      expect(span.parentSpan).toBe(trace.rootSpan);
    }
    expect(endedSpans(proc, AgentExecutionSpan)).toHaveLength(2);
  });

  it("keeps concurrent streams isolated while consumed from the caller's context", async () => {
    const { started1, started2, gate1, gate2, graph1, graph2 } =
      gatedPatchedPair();
    const proc = new RecordingSpanProcessor();
    const trace = new Trace({ spanProcessors: [proc] });

    await trace.run(async () => {
      const drain = async (graph: typeof graph1): Promise<number> => {
        let chunks = 0;
        for await (const _chunk of await graph.stream({})) {
          chunks += 1;
        }
        return chunks;
      };
      const run1 = drain(graph1);
      const run2 = drain(graph2);
      await Promise.all([started1.promise, started2.promise]);

      gate1.resolve();
      expect(await run1).toBe(1);
      expect(getCurrentSpan()).toBe(trace.rootSpan);
      expect(endedSpans(proc, AgentExecutionSpan)).toHaveLength(1);

      gate2.resolve();
      expect(await run2).toBe(1);
    });

    const agentSpans = startedSpans(proc, AgentExecutionSpan);
    expect(agentSpans).toHaveLength(2);
    for (const span of agentSpans) {
      expect(span.parentSpan).toBe(trace.rootSpan);
    }
    expect(endedSpans(proc, AgentExecutionSpan)).toHaveLength(2);
  });

  it("parents the run's LLM and tool spans under the execution span inside the fork", async () => {
    const agent = await loadWeatherAgent();
    const proc = new RecordingSpanProcessor();
    await new Trace({ spanProcessors: [proc] }).run(async () => {
      await agent.invoke(WEATHER_QUESTION);
    });

    const agentSpans = startedSpans(proc, AgentExecutionSpan);
    expect(agentSpans).toHaveLength(1);
    const llmSpans = startedSpans(proc, LlmGenerationSpan);
    const toolSpans = startedSpans(proc, ToolExecutionSpan);
    expect(llmSpans.length).toBeGreaterThan(0);
    expect(toolSpans.length).toBeGreaterThan(0);
    for (const span of [...llmSpans, ...toolSpans]) {
      expect(span.parentSpan).toBe(agentSpans[0]);
    }
  });
});
