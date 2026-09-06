/**
 * FlowExecutionSpan (port of `pyagentspec.tracing.spans.flow`).
 */
import type { Flow } from "../../flows/index.js";
import { Span, type SpanOptions } from "./span.js";

export interface FlowExecutionSpanOptions extends SpanOptions {
  /** The Flow being executed */
  flow: Flow;
}

/**
 * Span that covers the execution of a Flow.
 *
 * - Starts when: the StartNode execution of this flow starts
 * - Ends when: one of the EndNode executions finishes
 */
export class FlowExecutionSpan extends Span {
  flow: Flow;

  override get type(): string {
    return "FlowExecutionSpan";
  }

  constructor(options: FlowExecutionSpanOptions) {
    super(options);
    this.flow = options.flow;
  }
}
