/**
 * Tracing for the LangGraph adapter.
 *
 * Port of `pyagentspec.adapters.langgraph.tracing` (the LLM and tool callback
 * handlers) and `pyagentspec.adapters.langgraph._execution_span` (the
 * graph-level execution-span wrapper). Span and event payloads match the
 * Python adapter so span processors work across both SDKs.
 *
 * Divergences from Python (see the adapter README):
 * - Async-only: Python's sync/async twin callbacks and the
 *   `NotImplementedError` fallback chains collapse into single async methods.
 * - Python fights `copy_context()` snapshots with a run_id-keyed span-stack
 *   singleton (`_SpanStack`); LangChain JS callbacks run inline in the
 *   emitting async context once `awaitHandlers` is set, so the handlers keep
 *   only a run_id -> span registry and call the span APIs directly.
 * - `patchWithExecutionSpan` wraps `invoke`/`stream` through a Proxy (Python
 *   monkey-patches `stream`/`astream` in place, which `invoke` uses
 *   internally). A consequence: the raw compiled graph unwrapped from a
 *   patched react agent (swarm assembly, the ManagerWorkers `__manager__`
 *   node) is NOT patched, so those embedded sub-agent runs emit no
 *   AgentExecutionSpan of their own, while ManagerWorkers workers (invoked
 *   through the patched agent) do.
 * - The `invoke` wrapper builds the end event from the invoke result (Python
 *   folds streamed state chunks, which yields `{}` on the invoke path); the
 *   `stream` wrapper folds `[namespace, state]`-style array chunks exactly
 *   like Python.
 * - Non-string payloads are coerced with `JSON.stringify` where Python uses
 *   `str(...)` (content blocks, tool-call argument objects).
 */
import { BaseCallbackHandler } from "@langchain/core/callbacks/base";
import type {
  HandleLLMNewTokenCallbackFields,
  NewTokenIndices,
} from "@langchain/core/callbacks/base";
import type { Serialized } from "@langchain/core/load/serializable";
import type { AIMessage, AIMessageChunk, BaseMessage } from "@langchain/core/messages";
import { isToolMessage } from "@langchain/core/messages";
import type { ChatGenerationChunk, LLMResult } from "@langchain/core/outputs";
import type { Agent, ManagerWorkers } from "../../agents/index.js";
import type { Flow } from "../../flows/index.js";
import type { LlmConfig } from "../../llms/index.js";
import { createClientTool, type Tool } from "../../tools/index.js";
import { propertyFromJsonSchema, type JsonSchemaValue } from "../../property.js";
import {
  AgentExecutionEnd,
  AgentExecutionSpan,
  AgentExecutionStart,
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
  Message as TracingMessage,
  Span,
  ToolCall as TracingToolCall,
  ToolExecutionRequest,
  ToolExecutionResponse,
  ToolExecutionSpan,
  type Event as TracingEvent,
} from "../../tracing/index.js";
import { isRecordLike } from "../common/index.js";
import { extractOutputsFromInvokeResult } from "./node-execution/agent-node.js";

/** LangChain message types mapped onto OpenAI-style tracing roles. */
const LANGCHAIN_ROLES_TO_OPENAI_ROLES: Readonly<Record<string, string>> = {
  human: "user",
  ai: "assistant",
  tool: "tool",
  system: "system",
};

/**
 * Coerce a payload to a string (port of Python's `_ensure_string`).
 *
 * Strings pass through; `null`/`undefined` raise like Python's `None` check;
 * everything else is JSON-stringified (Python uses `str(...)`, so the exact
 * text of coerced non-string payloads differs across SDKs).
 */
function ensureString(obj: unknown): string {
  if (obj === null || obj === undefined) {
    throw new Error("can only coerce non-string objects to string");
  }
  if (typeof obj === "string") {
    return obj;
  }
  try {
    return JSON.stringify(obj) ?? String(obj);
  } catch {
    throw new Error(`obj is not a valid JSON dict: ${String(obj)}`);
  }
}

/**
 * Python coerces falsy chunk content (`None`, `""`, `[]`) to `""` with
 * `content or ""`; JS truthiness differs for arrays, so the falsy cases are
 * spelled out.
 */
function chunkContentToString(rawContent: unknown): string {
  if (
    rawContent === null ||
    rawContent === undefined ||
    rawContent === "" ||
    (Array.isArray(rawContent) && rawContent.length === 0)
  ) {
    return "";
  }
  return ensureString(rawContent);
}

/**
 * Normalize LangChain callback tool inputs into the mapping expected by trace
 * events (port of Python's `_normalize_tool_inputs`).
 *
 * LangChain JS passes the structured input JSON-stringified where Python
 * receives the structured dict in the `inputs` kwarg, so parsing the string
 * back is the JS equivalent of Python's structured-inputs priority branch;
 * the remaining branches mirror Python's fallbacks for non-dict inputs.
 */
function normalizeToolInputs(
  tool: Tool,
  inputValue: string,
): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(inputValue);
    if (isRecordLike(parsed)) {
      return parsed;
    }
  } catch {
    // Not JSON — fall through to the positional fallbacks.
  }
  if (tool.inputs !== undefined && tool.inputs.length === 1) {
    return { [tool.inputs[0]!.title]: inputValue };
  }
  return { value: inputValue };
}

/**
 * Synthesize the Agent Spec tools reported in an `LlmGenerationRequest` from
 * the OpenAI function-format tool schemas in the model's invocation params
 * (`ClientTool` is used as a generic `Tool` carrier here, like Python).
 */
function toolsFromInvocationParams(
  extraParams: Record<string, unknown> | undefined,
): Tool[] {
  const invocationParamsRaw = extraParams?.["invocation_params"];
  const invocationParams = isRecordLike(invocationParamsRaw)
    ? invocationParamsRaw
    : {};
  const toolSchemas = invocationParams["tools"];
  if (!Array.isArray(toolSchemas)) {
    return [];
  }
  return toolSchemas.map((toolSchema) => {
    // Python indexes tool_schema["function"]["name"] etc. directly and lets
    // a malformed entry raise; mirror with explicit errors.
    const fn = isRecordLike(toolSchema)
      ? toolSchema["function"]
      : undefined;
    if (!isRecordLike(fn) || typeof fn["name"] !== "string") {
      throw new Error(
        "[on_chat_model_start] invocation_params tools entries must be " +
          `OpenAI function-format tool schemas, got: ${JSON.stringify(toolSchema)}`,
      );
    }
    const parameters = isRecordLike(fn["parameters"]) ? fn["parameters"] : {};
    const properties = isRecordLike(parameters["properties"])
      ? (parameters["properties"] as Record<string, JsonSchemaValue>)
      : {};
    return createClientTool({
      name: fn["name"],
      ...(typeof fn["description"] === "string"
        ? { description: fn["description"] }
        : {}),
      inputs: Object.entries(properties).map(([propertyTitle, propertySchema]) =>
        // Python's Property(title=..., json_schema=...) merges the explicit
        // title into the schema.
        propertyFromJsonSchema({ ...propertySchema, title: propertyTitle }),
      ),
    });
  });
}

/** Build an Agent Spec ToolCall from a LangChain tool-call dict (raw OpenAI form included). */
function buildAgentSpecToolCall(toolCall: Record<string, unknown>): TracingToolCall {
  const callId = toolCall["id"];
  if (typeof callId !== "string") {
    throw new Error(
      `Expected tool call to carry a string id, got: ${JSON.stringify(toolCall)}`,
    );
  }
  let payload = toolCall;
  let argsKey = "args";
  if ("function" in toolCall && isRecordLike(toolCall["function"])) {
    payload = toolCall["function"];
    argsKey = "arguments";
  }
  return new TracingToolCall({
    callId,
    toolName: String(payload["name"]),
    arguments: ensureString(payload[argsKey]),
  });
}

/**
 * Extract completion id, content and tool calls from an LLM result (port of
 * Python's `_extract_message_content_and_tool_calls`).
 */
function extractMessageContentAndToolCalls(response: LLMResult): {
  messageId: string | null;
  content: string;
  toolCalls: TracingToolCall[];
} {
  const generations = response.generations ?? [];
  if (generations.length !== 1 || generations[0]!.length !== 1) {
    throw new Error(
      "Expected response to contain one generation and one chat_generation",
    );
  }
  const message = (generations[0]![0] as { message?: AIMessage }).message;
  if (message === undefined) {
    throw new Error(
      "Expected response to contain one generation and one chat_generation",
    );
  }
  const rawContent: unknown = message.content;
  const messageToolCalls = message.tool_calls ?? [];
  const additionalToolCalls = message.additional_kwargs?.["tool_calls"];
  const toolCallsRaw: unknown[] =
    messageToolCalls.length > 0
      ? messageToolCalls
      : Array.isArray(additionalToolCalls)
        ? additionalToolCalls
        : [];
  // NOTE: content can be empty (empty string ""); in that case tool_calls
  // should not be empty.
  if (rawContent === "" && toolCallsRaw.length === 0) {
    throw new Error(
      "Expected tool_calls to not be empty when content is empty. " +
        "This issue is LLM-specific depending on their tool-calling capabilities; " +
        "you may want to try again or switch to another LLM.",
    );
  }
  const content = ensureString(rawContent);
  const toolCalls = toolCallsRaw.map((toolCall) =>
    buildAgentSpecToolCall(toolCall as Record<string, unknown>),
  );
  // If streaming, response_id is not provided; rely on the message id.
  const responseMetadataId = (
    message.response_metadata as Record<string, unknown> | undefined
  )?.["id"];
  const messageId = message.id
    ? message.id
    : typeof responseMetadataId === "string"
      ? responseMetadataId
      : null;
  return { messageId, content, toolCalls };
}

/**
 * Base of the adapter callback handlers: a run_id -> span registry plus
 * Python's `raise_error = True`. `awaitHandlers` is forced on so handlers run
 * inline in the emitting async context — the AsyncLocalStorage span stack
 * stays correct across parallel runs, and every event is delivered before the
 * surrounding `invoke` resolves.
 */
abstract class AgentSpecCallbackHandler extends BaseCallbackHandler {
  /** Spans opened by this handler, keyed by LangChain run id. */
  protected readonly agentspecSpansRegistry = new Map<string, Span>();

  constructor() {
    super();
    this.raiseError = true;
    this.awaitHandlers = true;
  }
}

/** Tool-call-chunk carry-forward record for one streamed message. */
interface MessageInProgress {
  /** The streamed chunk message id. */
  id: string;
  toolCallId?: string;
  toolCallName?: string;
}

/**
 * LangChain callback handler emitting `LlmGenerationSpan`s with
 * request / streamed-chunk / response events for one Agent Spec LLM config
 * (port of Python's `AgentSpecLlmCallbackHandler`). Attached to every chat
 * model the converter creates.
 */
export class AgentSpecLlmCallbackHandler extends AgentSpecCallbackHandler {
  override readonly name = "AgentSpecLlmCallbackHandler";
  readonly llmConfig: LlmConfig;
  /**
   * Tool-call streaming state keyed by run id, used to associate streamed
   * argument deltas with the tool_call_id announced by the first chunk
   * (tool_call_id is not available mid-stream).
   */
  readonly messagesInProcess = new Map<string, MessageInProgress>();

  constructor(llmConfig: LlmConfig) {
    super();
    this.llmConfig = llmConfig;
  }

  override async handleChatModelStart(
    _llm: Serialized,
    messages: BaseMessage[][],
    runId: string,
    _parentRunId?: string,
    extraParams?: Record<string, unknown>,
  ): Promise<void> {
    // Create and start the LLM span for this run.
    const span = new LlmGenerationSpan({ llmConfig: this.llmConfig });
    this.agentspecSpansRegistry.set(runId, span);
    await span.start();

    // This is a list of lists because it can be batched, but we assume it to
    // be a batch of size 1.
    if (messages.length !== 1) {
      throw new Error(
        "[on_chat_model_start] langchain messages is a nested list of list of " +
          "BaseMessage, expected the outer list to have size one but got size " +
          `${messages.length}`,
      );
    }
    const prompt = messages[0]!.map((message) => {
      const messageType = message.getType();
      const role = LANGCHAIN_ROLES_TO_OPENAI_ROLES[messageType];
      if (role === undefined) {
        // Python raises a bare KeyError from the role map here.
        throw new Error(
          `Unsupported LangChain message type '${messageType}' for the tracing role map.`,
        );
      }
      return new TracingMessage({
        content: ensureString(message.content),
        sender: "",
        role,
      });
    });

    const event = new LlmGenerationRequest({
      requestId: runId,
      llmConfig: this.llmConfig,
      llmGenerationConfig: this.llmConfig.defaultGenerationParameters ?? null,
      prompt,
      tools: toolsFromInvocationParams(extraParams),
    });
    await span.addEvent(event);
  }

  override async handleLLMNewToken(
    _token: string,
    _idx: NewTokenIndices,
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
    fields?: HandleLLMNewTokenCallbackFields,
  ): Promise<void> {
    // Streaming only: text chunks and/or tool-call chunks. The first chunk of
    // a tool call carries id and name (empty args); the following chunks
    // carry only argument deltas.
    const chunk = fields?.chunk;
    if (chunk === undefined || chunk === null) {
      throw new Error("[on_llm_new_token] Expected chunk to not be None");
    }
    const span = this.agentspecSpansRegistry.get(runId);
    if (!(span instanceof LlmGenerationSpan)) {
      throw new Error(
        "LLM span not started; on_chat_model_start must run first",
      );
    }
    const chunkMessage = (chunk as ChatGenerationChunk).message as AIMessageChunk;
    // Note: chunk_message.response_metadata.id is not populated mid-stream.
    if (typeof chunkMessage.id !== "string") {
      throw new Error(
        "[on_llm_new_token] Expected chunk_message.id to be a string but got: " +
          typeof chunkMessage.id,
      );
    }
    const messageId = chunkMessage.id;

    let agentspecToolCalls: TracingToolCall[] = [];
    const toolCallChunks = chunkMessage.tool_call_chunks ?? [];
    if (toolCallChunks.length > 0) {
      if (toolCallChunks.length !== 1) {
        throw new Error(
          "[on_llm_new_token] Expected exactly one tool call chunk " +
            `if streaming tool calls, but got: ${JSON.stringify(toolCallChunks)}`,
        );
      }
      const toolCallChunk = toolCallChunks[0]!;
      let toolName = toolCallChunk.name;
      let callId = toolCallChunk.id;
      const toolArgs = toolCallChunk.args;
      if (callId === undefined || callId === null) {
        const currentStream = this.messagesInProcess.get(runId);
        if (currentStream === undefined) {
          // Python raises a bare KeyError from messages_in_process here.
          throw new Error(
            `[on_llm_new_token] No tool call in progress for run_id=${runId}`,
          );
        }
        toolName = currentStream.toolCallName;
        callId = currentStream.toolCallId;
      } else {
        this.messagesInProcess.set(runId, {
          id: messageId,
          toolCallId: callId,
          ...(toolName !== undefined && toolName !== null
            ? { toolCallName: toolName }
            : {}),
        });
      }
      agentspecToolCalls = [
        new TracingToolCall({
          callId: callId ?? "",
          toolName: toolName ?? "",
          // Argument DELTAS, not the accumulated arguments (Python parity).
          arguments: toolArgs || "",
        }),
      ];
    }

    const event = new LlmGenerationChunkReceived({
      requestId: runId,
      completionId: messageId,
      content: chunkContentToString(chunkMessage.content),
      llmConfig: this.llmConfig,
      toolCalls: agentspecToolCalls,
    });
    await span.addEvent(event);
  }

  override async handleLLMEnd(output: LLMResult, runId: string): Promise<void> {
    const span = this.agentspecSpansRegistry.get(runId);
    if (!(span instanceof LlmGenerationSpan)) {
      throw new Error(
        "LLM span not started; on_chat_model_start must run first",
      );
    }
    const { messageId, content, toolCalls } =
      extractMessageContentAndToolCalls(output);
    const event = new LlmGenerationResponse({
      llmConfig: this.llmConfig,
      requestId: runId,
      completionId: messageId,
      content,
      toolCalls,
    });
    await span.addEvent(event);
    await span.end();
    this.agentspecSpansRegistry.delete(runId);
    this.messagesInProcess.delete(runId);
  }
}

/**
 * LangChain callback handler emitting `ToolExecutionSpan`s with
 * request/response events for one Agent Spec tool (port of Python's
 * `AgentSpecToolCallbackHandler`). Attached to server tools, remote tools and
 * loaded MCP tools — NOT to client tools, whose request/response events are
 * the runtime's human-in-the-loop business (Python parity).
 */
export class AgentSpecToolCallbackHandler extends AgentSpecCallbackHandler {
  override readonly name = "AgentSpecToolCallbackHandler";
  readonly tool: Tool;

  constructor(tool: Tool) {
    super();
    this.tool = tool;
  }

  override async handleToolStart(
    _tool: Serialized,
    input: string,
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
    _metadata?: Record<string, unknown>,
    _runName?: string,
    toolCallId?: string,
  ): Promise<void> {
    // Instead of the real tool_call_id, the run_id correlates the tool
    // request with the tool result.
    const requestEvent = new ToolExecutionRequest({
      requestId: runId,
      tool: this.tool,
      inputs: normalizeToolInputs(this.tool, input),
    });
    // Hack (Python parity): transmit the tool_call_id as the span's
    // description so that tool results can be correlated with the streamed
    // LLM tool-call chunks that announced them.
    const tcidString = toolCallId !== undefined ? `tcid__${String(toolCallId)}` : "";
    const toolSpan = new ToolExecutionSpan({
      name: `ToolExecution[${this.tool.name}]`,
      description: tcidString,
      tool: this.tool,
    });
    this.agentspecSpansRegistry.set(runId, toolSpan);
    await toolSpan.start();
    await toolSpan.addEvent(requestEvent);
  }

  /**
   * Python's sync and async `on_tool_end` twins map outputs differently; the
   * port follows the sync variant (declared-outputs title mapping, request_id
   * always the run id), which the Python flow tests pin to exact payloads.
   */
  override async handleToolEnd(output: unknown, runId: string): Promise<void> {
    const toolSpan = this.agentspecSpansRegistry.get(runId);
    if (!(toolSpan instanceof ToolExecutionSpan)) {
      throw new Error(
        `Expected tool_span to be a ToolExecutionSpan but got ${typeof toolSpan}`,
      );
    }

    let outputValue: unknown = output;
    if (isToolMessage(output)) {
      const content: unknown = output.content;
      if (typeof content === "string") {
        try {
          outputValue = JSON.parse(content);
        } catch {
          outputValue = String(content);
        }
      } else {
        outputValue = content;
      }
    }

    let outputs: Record<string, unknown>;
    const declaredOutputs = this.tool.outputs ?? [];
    if (declaredOutputs.length === 1) {
      // Exactly one declared output: use its title.
      outputs = { [declaredOutputs[0]!.title]: outputValue };
    } else if (declaredOutputs.length > 1) {
      // The output should already be a mapping with the right entries; when
      // it is not, something went wrong and no output is reported.
      outputs = isRecordLike(outputValue) ? outputValue : {};
    } else {
      // No declared outputs: the tool has no entries to report.
      outputs = {};
    }

    const responseEvent = new ToolExecutionResponse({
      requestId: runId,
      tool: toolSpan.tool,
      outputs,
    });
    await toolSpan.addEvent(responseEvent);
    await toolSpan.end();
    this.agentspecSpansRegistry.delete(runId);
  }

  override async handleToolError(err: Error, runId: string): Promise<void> {
    try {
      await this.handleToolEnd(null, runId);
    } catch {
      // Python's `finally: raise error` swallows secondary errors from
      // on_tool_end so the original tool error always propagates.
    }
    throw err;
  }
}

/**
 * Which execution span wraps a compiled graph, and the Agent Spec component
 * that span reports on: an `AgentExecutionSpan` for react agents, a
 * `FlowExecutionSpan` for compiled flows and a `ManagerWorkersExecutionSpan`
 * for hierarchical manager-workers graphs.
 */
export type ExecutionSpanTarget =
  | { kind: "agent"; component: Agent }
  | { kind: "flow"; component: Flow }
  | { kind: "manager-workers"; component: ManagerWorkers };

/** The span plus start/end event builders for one execution-span target. */
interface ExecutionSpanFactories {
  makeSpan(): Span;
  makeStartEvent(inputs: Record<string, unknown>): TracingEvent;
  makeEndEvent(result: Record<string, unknown>): TracingEvent;
}

/** Build the span/event factories matching Python's three call sites (§ _execution_span). */
function executionSpanFactories(
  target: ExecutionSpanTarget,
): ExecutionSpanFactories {
  switch (target.kind) {
    case "agent": {
      const agent = target.component;
      return {
        makeSpan: () =>
          new AgentExecutionSpan({ name: `AgentExecution[${agent.name}]`, agent }),
        makeStartEvent: (inputs) => new AgentExecutionStart({ agent, inputs }),
        makeEndEvent: (result) =>
          new AgentExecutionEnd({
            agent,
            outputs: extractOutputsFromInvokeResult(result, agent.outputs ?? []),
          }),
      };
    }
    case "flow": {
      const flow = target.component;
      return {
        makeSpan: () =>
          new FlowExecutionSpan({ name: `FlowExecution[${flow.name}]`, flow }),
        makeStartEvent: (inputs) => new FlowExecutionStart({ flow, inputs }),
        makeEndEvent: (result) => {
          const outputs = result["outputs"];
          const details = result["node_execution_details"];
          const branch = isRecordLike(details) ? details["branch"] : undefined;
          return new FlowExecutionEnd({
            flow,
            outputs: isRecordLike(outputs) ? outputs : {},
            branchSelected: typeof branch === "string" ? branch : "",
          });
        },
      };
    }
    case "manager-workers": {
      const managerworkers = target.component;
      return {
        makeSpan: () =>
          new ManagerWorkersExecutionSpan({
            name: `ManagerWorkersExecution[${managerworkers.name}]`,
            managerworkers,
          }),
        makeStartEvent: (inputs) =>
          new ManagerWorkersExecutionStart({ managerworkers, inputs }),
        makeEndEvent: (result) =>
          new ManagerWorkersExecutionEnd({
            managerworkers,
            outputs: { messages: result["messages"] ?? [] },
          }),
      };
    }
  }
}

/** The invocation input state of a patched call, or `{}` when it isn't a record. */
function invocationInputs(input: unknown): Record<string, unknown> {
  return isRecordLike(input) ? input : {};
}

/**
 * Fold one streamed chunk into the running "last state seen". State arrives
 * as `[namespace, state]`-style tuples (arrays in JS — subgraph, multi-mode
 * and messages streams); other chunk shapes leave the fold untouched
 * (Python parity).
 */
function foldFinalState(chunk: unknown, soFar: unknown): unknown {
  return Array.isArray(chunk) ? chunk[1] : soFar;
}

function isPromiseLike(value: unknown): value is PromiseLike<unknown> {
  return (
    value !== null &&
    (typeof value === "object" || typeof value === "function") &&
    typeof (value as { then?: unknown }).then === "function"
  );
}

/**
 * Wrap a compiled graph (or react agent) so each run is traced inside the
 * execution span named by `target.kind` for `target.component`.
 *
 * Mirrors Python's `patch_with_execution_span` semantics on the async path:
 * the span starts when the run starts, a Start event carries the invocation
 * inputs, streamed chunks are yielded through while the final state chunk is
 * folded, the End event carries the run outputs, and the span always ends —
 * consumer abandonment and mid-run errors included (no ExceptionRaised event,
 * matching Python's `astream` path). A `stream()` promise that rejects before
 * producing the iterable still emits the span with its Start event, like
 * Python's first-`anext` failure.
 */
export function patchWithExecutionSpan<T>(
  graph: T,
  target: ExecutionSpanTarget,
): T {
  const factories = executionSpanFactories(target);

  async function* traceStream(
    iterable: AsyncIterable<unknown>,
    inputs: Record<string, unknown>,
  ): AsyncGenerator<unknown> {
    const span = factories.makeSpan();
    await span.start();
    try {
      await span.addEvent(factories.makeStartEvent(inputs));
      let state: unknown = {};
      for await (const chunk of iterable) {
        yield chunk;
        state = foldFinalState(chunk, state);
      }
      await span.addEvent(
        factories.makeEndEvent(isRecordLike(state) ? state : {}),
      );
    } finally {
      await span.end();
    }
  }

  async function traceFailedStreamStart(
    inputs: Record<string, unknown>,
  ): Promise<void> {
    const span = factories.makeSpan();
    await span.start();
    try {
      await span.addEvent(factories.makeStartEvent(inputs));
    } finally {
      await span.end();
    }
  }

  const wrapInvoke =
    (original: (...args: unknown[]) => unknown, targetObject: object) =>
    async (...args: unknown[]): Promise<unknown> => {
      const span = factories.makeSpan();
      await span.start();
      try {
        await span.addEvent(factories.makeStartEvent(invocationInputs(args[0])));
        const result: unknown = await original.apply(targetObject, args);
        await span.addEvent(
          factories.makeEndEvent(isRecordLike(result) ? result : {}),
        );
        return result;
      } finally {
        await span.end();
      }
    };

  const wrapStream =
    (original: (...args: unknown[]) => unknown, targetObject: object) =>
    (...args: unknown[]): unknown => {
      const inputs = invocationInputs(args[0]);
      const out = original.apply(targetObject, args);
      if (isPromiseLike(out)) {
        return (out as Promise<AsyncIterable<unknown>>).then(
          (iterable) => traceStream(iterable, inputs),
          async (error: unknown) => {
            await traceFailedStreamStart(inputs);
            throw error;
          },
        );
      }
      return traceStream(out as AsyncIterable<unknown>, inputs);
    };

  // Probe-verified wrapping: a Proxy intercepting invoke/stream and binding
  // every other method to the target preserves the full graph surface
  // (builder introspection, options, streamEvents, private-field methods).
  return new Proxy(graph as object, {
    get(targetObject, property) {
      const original: unknown = Reflect.get(targetObject, property, targetObject);
      if (typeof original !== "function") {
        return original;
      }
      if (property === "invoke") {
        return wrapInvoke(
          original as (...args: unknown[]) => unknown,
          targetObject,
        );
      }
      if (property === "stream") {
        return wrapStream(
          original as (...args: unknown[]) => unknown,
          targetObject,
        );
      }
      return (original as (...args: unknown[]) => unknown).bind(targetObject);
    },
  }) as T;
}
