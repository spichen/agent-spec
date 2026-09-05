/**
 * ToolExecutionSpan (port of `pyagentspec.tracing.spans.tool`).
 */
import type { Tool } from "../../tools/index.js";
import { Span, type SpanOptions } from "./span.js";

export interface ToolExecutionSpanOptions extends SpanOptions {
  /** The Tool being executed */
  tool: Tool;
}

/**
 * Span that covers a tool execution. This does not include client tools.
 *
 * - Starts when: tool execution starts
 * - Ends when: the tool execution is completed and the result is ready to be processed
 */
export class ToolExecutionSpan extends Span {
  tool: Tool;

  override get type(): string {
    return "ToolExecutionSpan";
  }

  constructor(options: ToolExecutionSpanOptions) {
    super(options);
    this.tool = options.tool;
  }
}
