/**
 * Trace (port of `pyagentspec.tracing.trace.Trace`).
 *
 * Async-only: Python's sync/async context-manager pairs collapse into
 * `start()`/`end()` plus the `run()` helper. Python's
 * `is_async_mode_active` bookkeeping has no JS equivalent and is not ported.
 */
import { runInChildContext, setAmbientTrace, getTrace } from "./context.js";
import type { SpanProcessor } from "./span-processor.js";
import { RootSpan } from "./spans/root.js";
import type { Span } from "./spans/span.js";

export { getTrace };

export interface TraceOptions {
  /** The name of the trace */
  name?: string;
  /** A unique identifier for the trace */
  id?: string;
  /** The list of SpanProcessors active on this trace */
  spanProcessors?: SpanProcessor[];
  /** Whether to call shutdown on span processors when the trace ends */
  shutdownOnExit?: boolean;
  /** The root span of the trace. If not provided, a new RootSpan with default values is used. */
  rootSpan?: Span;
}

/**
 * The root of a collection of Spans.
 *
 * It is used to group together all the spans and events emitted during the
 * execution of an assistant.
 */
export class Trace {
  /** The name of the trace */
  name: string;
  /** A unique identifier for the trace */
  id: string;
  /** The list of SpanProcessors active on this trace */
  spanProcessors: SpanProcessor[];
  /** Whether to call shutdown on span processors when the trace ends */
  shutdownOnExit: boolean;
  /** The root span of the trace */
  readonly rootSpan: Span;

  constructor(options: TraceOptions = {}) {
    this.name = options.name || "Trace";
    this.id = options.id || crypto.randomUUID();
    this.spanProcessors = options.spanProcessors ?? [];
    this.shutdownOnExit = options.shutdownOnExit ?? true;
    this.rootSpan = options.rootSpan ?? new RootSpan();
  }

  /**
   * Start the trace in the current context: register it as the ambient trace,
   * call `startup` on every span processor, then start the root span. Throws
   * if a trace is already active in this context.
   */
  async start(): Promise<void> {
    if (getTrace() !== undefined) {
      throw new Error("A Trace already exists. Cannot create two nested Traces.");
    }
    setAmbientTrace(this);
    for (const spanProcessor of this.spanProcessors) {
      await spanProcessor.startup();
    }
    await this.rootSpan.start();
  }

  /**
   * End the trace: end the root span, clear the ambient trace, and — when
   * `shutdownOnExit` is set — call `shutdown` on every span processor.
   */
  async end(): Promise<void> {
    await this.rootSpan.end();
    setAmbientTrace(undefined);
    if (this.shutdownOnExit) {
      for (const spanProcessor of this.spanProcessors) {
        await spanProcessor.shutdown();
      }
    }
  }

  /**
   * Run `fn` inside this trace — the JS equivalent of Python's
   * `with Trace(...) as trace:` — in a forked ambient context: start the
   * trace, run `fn`, and always end the trace (which, like Python's
   * `__exit__`, does not record exceptions on the root span).
   */
  async run<T>(fn: (trace: this) => Promise<T> | T): Promise<T> {
    return runInChildContext(async () => {
      await this.start();
      try {
        return await fn(this);
      } finally {
        await this.end();
      }
    });
  }
}
