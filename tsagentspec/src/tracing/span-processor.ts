/**
 * SpanProcessor (port of `pyagentspec.tracing.spanprocessor.SpanProcessor`).
 *
 * Python defines sync and async twins for every hook; async-only JS has a
 * single set of hooks that may return either `void` or a `Promise` (the
 * callers await them either way). The `NotImplementedError` async-to-sync
 * fallback chains are therefore not ported.
 */
import type { Event } from "./events/event.js";
import type { Span } from "./spans/span.js";

/**
 * Interface which allows hooks for `Span` start and end method invocations.
 *
 * Aligned with OpenTelemetry APIs. Processors are registered on a `Trace`
 * (`new Trace({ spanProcessors: [...] })`); spans discover them at start time
 * through the ambient trace.
 */
export abstract class SpanProcessor {
  /**
   * Whether this processor masks sensitive information when it serializes
   * spans and events. The tracing core never reads this flag; processors use
   * it themselves when dumping (`span.serialize({ maskSensitiveInformation:
   * this.maskSensitiveInformation })`). Defaults to `true`.
   */
  maskSensitiveInformation: boolean;

  constructor(maskSensitiveInformation: boolean = true) {
    this.maskSensitiveInformation = maskSensitiveInformation;
  }

  /**
   * Called when a `Span` is started.
   *
   * @param span - The span that starts
   */
  abstract onStart(span: Span): void | Promise<void>;

  /**
   * Called when a `Span` is ended.
   *
   * @param span - The span that ends
   */
  abstract onEnd(span: Span): void | Promise<void>;

  /**
   * Called when an `Event` is triggered.
   *
   * @param event - The event that is happening
   * @param span - The span where the event occurs
   */
  abstract onEvent(event: Event, span: Span): void | Promise<void>;

  /** Called when a `Trace` is started. */
  abstract startup(): void | Promise<void>;

  /** Called when a `Trace` is shutdown. */
  abstract shutdown(): void | Promise<void>;
}
