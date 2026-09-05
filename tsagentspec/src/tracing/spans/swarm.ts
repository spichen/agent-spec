/**
 * SwarmExecutionSpan (port of `pyagentspec.tracing.spans.swarm`).
 */
import type { Swarm } from "../../agents/index.js";
import { Span, type SpanOptions } from "./span.js";

export interface SwarmExecutionSpanOptions extends SpanOptions {
  /** The Swarm being executed */
  swarm: Swarm;
}

/**
 * Span to represent the execution of a Swarm. Can be nested when executing sub-agents.
 *
 * - Starts when: swarm pattern execution starts
 * - Ends when: the swarm execution is completed and the result is ready to be processed
 */
export class SwarmExecutionSpan extends Span {
  swarm: Swarm;

  override get type(): string {
    return "SwarmExecutionSpan";
  }

  constructor(options: SwarmExecutionSpanOptions) {
    super(options);
    this.swarm = options.swarm;
  }
}
