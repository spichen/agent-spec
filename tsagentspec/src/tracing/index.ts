/**
 * Agent Spec tracing package — async-only port of `pyagentspec.tracing`.
 *
 * Class and field names match the Python package so span processors written
 * against either SDK see the same serialized shapes (snake_case wire fields,
 * a `type` discriminant carrying the class name, sensitive payloads masked
 * with `** MASKED **` by default). Python's sync/async bridging machinery has
 * no JS equivalent and is intentionally not ported.
 */
export { PII_MASK, TracingSerializable, type TracingSerializeOptions } from "./base.js";
export { Message, type MessageOptions } from "./message.js";
export { SpanProcessor } from "./span-processor.js";
export { Trace, type TraceOptions } from "./trace.js";
export { getTrace, getCurrentSpan, getActiveSpanStack } from "./context.js";
export {
  Span,
  RootSpan,
  AgentExecutionSpan,
  FlowExecutionSpan,
  LlmGenerationSpan,
  ManagerWorkersExecutionSpan,
  NodeExecutionSpan,
  SwarmExecutionSpan,
  ToolExecutionSpan,
  type SpanOptions,
  type AgentExecutionSpanOptions,
  type FlowExecutionSpanOptions,
  type LlmGenerationSpanOptions,
  type ManagerWorkersExecutionSpanOptions,
  type NodeExecutionSpanOptions,
  type SwarmExecutionSpanOptions,
  type ToolExecutionSpanOptions,
} from "./spans/index.js";
export {
  Event,
  ExceptionRaised,
  exceptionRaisedFromError,
  AgentExecutionStart,
  AgentExecutionEnd,
  FlowExecutionStart,
  FlowExecutionEnd,
  NodeExecutionStart,
  NodeExecutionEnd,
  ManagerWorkersExecutionStart,
  ManagerWorkersExecutionEnd,
  SwarmExecutionStart,
  SwarmExecutionEnd,
  HumanInTheLoopRequest,
  HumanInTheLoopResponse,
  LlmGenerationRequest,
  LlmGenerationResponse,
  LlmGenerationChunkReceived,
  ToolCall,
  ToolExecutionRequest,
  ToolExecutionResponse,
  ToolConfirmationRequest,
  ToolConfirmationResponse,
  ToolExecutionStreamingChunkReceived,
  StateSnapshotEmitted,
  type EventOptions,
  type ExceptionRaisedOptions,
  type AgentExecutionStartOptions,
  type AgentExecutionEndOptions,
  type FlowExecutionStartOptions,
  type FlowExecutionEndOptions,
  type NodeExecutionStartOptions,
  type NodeExecutionEndOptions,
  type ManagerWorkersExecutionStartOptions,
  type ManagerWorkersExecutionEndOptions,
  type SwarmExecutionStartOptions,
  type SwarmExecutionEndOptions,
  type HumanInTheLoopRequestOptions,
  type HumanInTheLoopResponseOptions,
  type LlmGenerationRequestOptions,
  type LlmGenerationResponseOptions,
  type LlmGenerationChunkReceivedOptions,
  type ToolCallOptions,
  type ToolExecutionRequestOptions,
  type ToolExecutionResponseOptions,
  type ToolConfirmationRequestOptions,
  type ToolConfirmationResponseOptions,
  type ToolExecutionStreamingChunkReceivedOptions,
  type StateSnapshotEmittedOptions,
} from "./events/index.js";
