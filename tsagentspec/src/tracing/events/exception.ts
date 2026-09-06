/**
 * ExceptionRaised event (port of `pyagentspec.tracing.events.exception`).
 */
import { Event, type EventOptions } from "./event.js";

export interface ExceptionRaisedOptions extends EventOptions {
  /** Type of the exception */
  exceptionType: string;
  /** Message of the exception (sensitive) */
  exceptionMessage: string;
  /** Stacktrace of the exception (sensitive) */
  exceptionStacktrace?: string;
}

/** This event is recorded whenever an exception occurs. */
export class ExceptionRaised extends Event {
  exceptionType: string;
  exceptionMessage: string;
  exceptionStacktrace: string;

  override get type(): string {
    return "ExceptionRaised";
  }

  constructor(options: ExceptionRaisedOptions) {
    super(options);
    this.exceptionType = options.exceptionType;
    this.exceptionMessage = options.exceptionMessage;
    this.exceptionStacktrace = options.exceptionStacktrace ?? "";
  }
}

/**
 * Build an ExceptionRaised event from a caught value, mirroring how Python's
 * `Span.__exit__` records `exc_type.__name__` / `str(exc_value)` / the
 * formatted traceback (empty when no stacktrace is available).
 */
export function exceptionRaisedFromError(error: unknown): ExceptionRaised {
  if (error instanceof Error) {
    return new ExceptionRaised({
      exceptionType: error.name || "Error",
      exceptionMessage: error.message,
      exceptionStacktrace: error.stack ?? "",
    });
  }
  return new ExceptionRaised({
    exceptionType: "Unknown",
    exceptionMessage: String(error),
    exceptionStacktrace: "",
  });
}
