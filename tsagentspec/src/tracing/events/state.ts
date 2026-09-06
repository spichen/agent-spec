/**
 * StateSnapshotEmitted event (port of `pyagentspec.tracing.events.state`).
 */
import { Event, type EventOptions } from "./event.js";

/**
 * Validate that a state snapshot payload can be encoded as strict JSON.
 *
 * Mirrors Python's `json.dumps(payload, allow_nan=False)` check: NaN/Infinity
 * and non-JSON values are rejected. Snapshot payloads are forwarded through
 * JSON-based transports and may later be stored and replayed for
 * resumability; if NaN/Infinity were accepted here, later JSON encoding could
 * silently coerce them (for example to `null`), breaking the expectation that
 * the runtime snapshot payload is carried through unchanged.
 */
function validateJsonSerializablePayload(
  payloadName: string,
  payload: Record<string, unknown> | null,
): void {
  if (payload === null) return;
  if (!isStrictJsonValue(payload)) {
    throw new Error(`${payloadName} must be JSON-serializable`);
  }
}

function isStrictJsonValue(value: unknown): boolean {
  if (value === null) return true;
  switch (typeof value) {
    case "string":
    case "boolean":
      return true;
    case "number":
      return Number.isFinite(value);
    case "object": {
      if (Array.isArray(value)) {
        return value.every(isStrictJsonValue);
      }
      const proto = Object.getPrototypeOf(value);
      if (proto !== Object.prototype && proto !== null) {
        // Class instances, Maps, Dates, ... are not strict-JSON payloads
        // (Python's json.dumps rejects arbitrary objects the same way).
        return false;
      }
      return Object.values(value as Record<string, unknown>).every(isStrictJsonValue);
    }
    default:
      // undefined, function, symbol, bigint
      return false;
  }
}

export interface StateSnapshotEmittedOptions extends EventOptions {
  /** Stable identifier of the logical conversation or thread this snapshot refers to. */
  conversationId: string;
  /**
   * Runtime-defined JSON-serializable snapshot content for the current state
   * (sensitive). This payload may contain opaque runtime-owned state needed
   * for resuming or reconstructing execution later.
   */
  stateSnapshot?: Record<string, unknown> | null;
  /** Developer-defined JSON-serializable state such as UI or application state (sensitive). */
  extraState?: Record<string, unknown> | null;
}

/**
 * A runtime emits a state snapshot for downstream consumers.
 *
 * This event carries a JSON-serializable snapshot of the current logical
 * conversation or thread state. The exact schema of `stateSnapshot` is
 * intentionally runtime-defined.
 */
export class StateSnapshotEmitted extends Event {
  conversationId: string;
  stateSnapshot: Record<string, unknown> | null;
  extraState: Record<string, unknown> | null;

  override get type(): string {
    return "StateSnapshotEmitted";
  }

  constructor(options: StateSnapshotEmittedOptions) {
    super(options);
    this.conversationId = options.conversationId;
    this.stateSnapshot = options.stateSnapshot ?? null;
    this.extraState = options.extraState ?? null;

    if (this.stateSnapshot === null && this.extraState === null) {
      throw new Error("At least one of state_snapshot or extra_state must be provided");
    }
    validateJsonSerializablePayload("state_snapshot", this.stateSnapshot);
    validateJsonSerializablePayload("extra_state", this.extraState);
  }
}
