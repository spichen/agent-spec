/**
 * Span base class (port of `pyagentspec.tracing.spans.span.Span`).
 *
 * Async-only: the sync/async twin methods collapse into single async
 * `start`/`end`/`addEvent`. Python's context-manager protocol is provided by
 * {@link Span.run}, which also forks the ambient context so parallel async
 * branches keep isolated span stacks.
 */
import { TracingSerializable, nowNs } from "../base.js";
import {
  appendSpanToActiveStack,
  getCurrentSpan,
  getTrace,
  popSpanFromActiveStack,
  runInChildContext,
} from "../context.js";
import type { Event } from "../events/event.js";
import { exceptionRaisedFromError } from "../events/exception.js";
import type { SpanProcessor } from "../span-processor.js";

export interface SpanOptions {
  /** A unique identifier for the span */
  id?: string;
  /** The name of the span. If not provided, the span class name is used. */
  name?: string;
  /** The description of the span. */
  description?: string;
  /** Metadata related to the span */
  metadata?: Record<string, unknown>;
}

export class Span extends TracingSerializable {
  /** A unique identifier for the span */
  readonly id: string;
  /** The name of the span. Defaults to the span class name. */
  name: string;
  /** The description of the span. */
  description: string;
  /** The timestamp of when the span was started (ns since the Unix epoch) */
  startTime: number | null = null;
  /** The timestamp of when the span was closed (ns since the Unix epoch) */
  endTime: number | null = null;
  /** The list of events recorded in the scope of this span */
  readonly events: Event[] = [];
  /** Metadata related to the span */
  metadata: Record<string, unknown>;
  /** The parent span captured from the ambient context when this span started */
  parentSpan: Span | undefined = undefined;

  private _spanWasAppendedToActiveStack = false;
  private _startedSpanProcessors: SpanProcessor[] = [];

  get type(): string {
    return "Span";
  }

  constructor(options: SpanOptions = {}) {
    super();
    this.id = options.id ?? crypto.randomUUID();
    // Like Python's model_post_init: a falsy name defaults to the class name.
    this.name = options.name || this.type;
    this.description = options.description ?? "";
    this.metadata = options.metadata ?? {};
  }

  /** The list of SpanProcessors to which this Span should be forwarded. */
  private get spanProcessors(): SpanProcessor[] {
    return getTrace()?.spanProcessors ?? [];
  }

  /**
   * Start the span.
   *
   * This includes calling the `onStart` hook of the active SpanProcessors.
   * If any hook throws, the span records an ExceptionRaised event, ends
   * (notifying only the processors that were successfully started), never
   * enters the active stack, and the error is re-thrown.
   */
  async start(): Promise<void> {
    try {
      this.parentSpan = getCurrentSpan();
      this.startTime = nowNs();
      for (const spanProcessor of this.spanProcessors) {
        await spanProcessor.onStart(this);
        // We remember which span processors were started, so that we call
        // onEnd on them only, e.g., when an exception happens.
        this._startedSpanProcessors.push(spanProcessor);
      }
      appendSpanToActiveStack(this);
      this._spanWasAppendedToActiveStack = true;
    } catch (error) {
      // If anything happens during the recording of the start span, we still
      // have to do the work needed to exit the context, including the
      // spanProcessors' onEnd call and removing the span from the active stack.
      await this.recordException(error);
      await this.end();
      throw error;
    }
  }

  /**
   * End the span.
   *
   * This includes calling the `onEnd` hook of the active SpanProcessors.
   * Per-processor errors are caught so every started processor gets its
   * `onEnd`; the first caught error is re-thrown after the loop. The span is
   * always popped from the active stack if it was appended.
   */
  async end(): Promise<void> {
    try {
      const caughtErrors: unknown[] = [];
      this.endTime = nowNs();
      // We call onEnd only on the span processors that were successfully started.
      for (const spanProcessor of this._startedSpanProcessors) {
        try {
          await spanProcessor.onEnd(this);
        } catch (error) {
          caughtErrors.push(error);
        }
      }
      if (caughtErrors.length > 0) {
        throw caughtErrors[0];
      }
    } finally {
      // Whatever happens, we have to pop the span if it is on the active stack.
      if (this._spanWasAppendedToActiveStack) {
        popSpanFromActiveStack();
      }
    }
  }

  /** Add an event to the span and trigger `onEvent` on the started SpanProcessors. */
  async addEvent(event: Event): Promise<void> {
    this.events.push(event);
    for (const spanProcessor of this._startedSpanProcessors) {
      await spanProcessor.onEvent(event, this);
    }
  }

  /** Record a caught error as an ExceptionRaised event on this span. */
  async recordException(error: unknown): Promise<void> {
    await this.addEvent(exceptionRaisedFromError(error));
  }

  /**
   * Run `fn` inside this span — the JS equivalent of Python's
   * `with Span(...) as span:` — in a forked ambient context: start the span,
   * run `fn`, record an ExceptionRaised event if it throws, and always end
   * the span. Parallel `run` branches keep isolated span stacks.
   */
  async run<T>(fn: (span: this) => Promise<T> | T): Promise<T> {
    return runInChildContext(async () => {
      await this.start();
      let caughtError: unknown;
      let didThrow = false;
      try {
        return await fn(this);
      } catch (error) {
        didThrow = true;
        caughtError = error;
        throw error;
      } finally {
        if (didThrow) {
          await this.recordException(caughtError);
        }
        await this.end();
      }
    });
  }
}
