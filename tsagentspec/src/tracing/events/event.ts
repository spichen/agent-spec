/**
 * Event base class (port of `pyagentspec.tracing.events.event.Event`).
 */
import { TracingSerializable, nowNs } from "../base.js";

export interface EventOptions {
  /** A unique identifier for the event */
  id?: string;
  /** The name of the event. If not provided, the event class name is used. */
  name?: string;
  /** The description of the event. */
  description?: string;
  /** The timestamp of when the event occurred (ns since the Unix epoch) */
  timestamp?: number;
  /** Metadata related to the event */
  metadata?: Record<string, unknown>;
}

export class Event extends TracingSerializable {
  /** A unique identifier for the event */
  readonly id: string;
  /** The name of the event. Defaults to the event class name. */
  name: string;
  /** The description of the event. */
  description: string;
  /** The timestamp of when the event occurred (ns since the Unix epoch) */
  timestamp: number;
  /** Metadata related to the event */
  metadata: Record<string, unknown>;

  get type(): string {
    return "Event";
  }

  constructor(options: EventOptions = {}) {
    super();
    this.id = options.id ?? crypto.randomUUID();
    // Like Python's model_post_init: a falsy name defaults to the class name.
    this.name = options.name || this.type;
    this.description = options.description ?? "";
    this.timestamp = options.timestamp ?? nowNs();
    this.metadata = options.metadata ?? {};
  }
}
