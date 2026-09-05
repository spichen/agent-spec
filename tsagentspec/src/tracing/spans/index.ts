/**
 * Tracing spans barrel (mirrors `pyagentspec.tracing.spans.__init__`).
 */
export { Span, type SpanOptions } from "./span.js";
export { RootSpan } from "./root.js";
export { AgentExecutionSpan, type AgentExecutionSpanOptions } from "./agent.js";
export { FlowExecutionSpan, type FlowExecutionSpanOptions } from "./flow.js";
export { LlmGenerationSpan, type LlmGenerationSpanOptions } from "./llm.js";
export {
  ManagerWorkersExecutionSpan,
  type ManagerWorkersExecutionSpanOptions,
} from "./manager-workers.js";
export { NodeExecutionSpan, type NodeExecutionSpanOptions } from "./node.js";
export { SwarmExecutionSpan, type SwarmExecutionSpanOptions } from "./swarm.js";
export { ToolExecutionSpan, type ToolExecutionSpanOptions } from "./tool.js";
