/**
 * Human-in-the-loop events (port of `pyagentspec.tracing.events.humanintheloop`).
 */
import { Event, type EventOptions } from "./event.js";

export interface HumanInTheLoopRequestOptions extends EventOptions {
  /** Identifier of the human-in-the-loop request */
  requestId: string;
  /** The content of the request forwarded to the user (sensitive) */
  content: Record<string, unknown>;
}

/** A human-in-the-loop (HITL) intervention is required. Emitted when the execution is interrupted due to HITL request. */
export class HumanInTheLoopRequest extends Event {
  requestId: string;
  content: Record<string, unknown>;

  override get type(): string {
    return "HumanInTheLoopRequest";
  }

  constructor(options: HumanInTheLoopRequestOptions) {
    super(options);
    this.requestId = options.requestId;
    this.content = options.content;
  }
}

export interface HumanInTheLoopResponseOptions extends EventOptions {
  /** Identifier of the human-in-the-loop request */
  requestId: string;
  /** The content of the response received from the user (sensitive) */
  content: Record<string, unknown>;
}

/** A human-in-the-loop response is provided. Emitted when the execution restarts after HITL response. */
export class HumanInTheLoopResponse extends Event {
  requestId: string;
  content: Record<string, unknown>;

  override get type(): string {
    return "HumanInTheLoopResponse";
  }

  constructor(options: HumanInTheLoopResponseOptions) {
    super(options);
    this.requestId = options.requestId;
    this.content = options.content;
  }
}
