/**
 * Tracing seams for the LangGraph adapter.
 *
 * The Python adapter attaches tracing callbacks and execution spans at three
 * kinds of sites: LLM callbacks on every converted chat model, tool callbacks
 * on converted server/remote/MCP tools, and stream-wrapping execution spans on
 * every compiled agent / flow / manager-workers graph.
 *
 * The TypeScript SDK has no tracing package yet, so these functions are no-op
 * seams: they are invoked from the exact same attachment sites as Python so
 * that a future port of `pyagentspec.tracing` only needs to fill in the
 * implementations here (returning real `BaseCallbackHandler`s and wrapping
 * `stream`/`streamEvents` in execution spans) without touching the converter.
 */
import type { LlmConfig } from "../../llms/index.js";
import type { Tool } from "../../tools/index.js";

/**
 * Build the tracing callbacks to attach to a chat model created for the given
 * Agent Spec LLM config.
 *
 * Python attaches an `AgentSpecLlmCallbackHandler` emitting
 * `LlmGenerationRequest` / `LlmGenerationChunkReceived` /
 * `LlmGenerationResponse` events inside an `LlmGenerationSpan`. No-op until
 * the tracing package is ported.
 */
export function buildLlmCallbacks(_llmConfig: LlmConfig): unknown[] {
  return [];
}

/**
 * Build the tracing callbacks to attach to a LangChain tool created for the
 * given Agent Spec tool.
 *
 * Python attaches an `AgentSpecToolCallbackHandler` emitting
 * `ToolExecutionRequest` / `ToolExecutionResponse` events inside a
 * `ToolExecutionSpan`. No-op until the tracing package is ported.
 */
export function buildToolCallbacks(_tool: Tool): unknown[] {
  return [];
}

/**
 * Wrap a compiled graph (or react agent) so each run is traced inside an
 * execution span.
 *
 * Python monkey-patches `stream`/`astream` to open an
 * `AgentExecutionSpan` / `FlowExecutionSpan` / `ManagerWorkersExecutionSpan`,
 * emit the start event with the invocation inputs, fold the streamed chunks
 * into a final state and emit the end event with the run outputs. Returns the
 * graph unchanged until the tracing package is ported.
 */
export function patchWithExecutionSpan<T>(graph: T): T {
  return graph;
}
