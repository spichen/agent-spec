/**
 * RootSpan (port of `pyagentspec.tracing.spans.root`).
 */
import { Span } from "./span.js";

/**
 * Span that covers a whole Trace.
 *
 * - Starts when: a Trace is started
 * - Ends when: a Trace is closed
 */
export class RootSpan extends Span {
  override get type(): string {
    return "RootSpan";
  }
}
