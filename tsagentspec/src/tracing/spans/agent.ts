/**
 * AgentExecutionSpan (port of `pyagentspec.tracing.spans.agent`).
 */
import type { Agent } from "../../agents/index.js";
import { Span, type SpanOptions } from "./span.js";

export interface AgentExecutionSpanOptions extends SpanOptions {
  /** The Agent being executed */
  agent: Agent;
}

/**
 * Span to represent the execution of an agent. Can be nested when executing sub-agents.
 *
 * - Starts when: agent execution starts
 * - Ends when: the agent execution is completed, and the result is ready to be processed
 */
export class AgentExecutionSpan extends Span {
  agent: Agent;

  override get type(): string {
    return "AgentExecutionSpan";
  }

  constructor(options: AgentExecutionSpanOptions) {
    super(options);
    this.agent = options.agent;
  }
}
