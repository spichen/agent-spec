/**
 * NodeExecutionSpan (port of `pyagentspec.tracing.spans.node`).
 */
import type { Node } from "../../flows/index.js";
import { Span, type SpanOptions } from "./span.js";

export interface NodeExecutionSpanOptions extends SpanOptions {
  /** The Node being executed */
  node: Node;
}

/**
 * Span that covers the execution of a Node.
 *
 * - Starts when: the node execution starts on the given inputs
 * - Ends when: the node execution ends and outputs are ready to be processed
 */
export class NodeExecutionSpan extends Span {
  node: Node;

  override get type(): string {
    return "NodeExecutionSpan";
  }

  constructor(options: NodeExecutionSpanOptions) {
    super(options);
    this.node = options.node;
  }
}
