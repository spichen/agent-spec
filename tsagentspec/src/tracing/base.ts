/**
 * Serialization core for the tracing package.
 *
 * Mirrors Python's `pyagentspec.tracing._basemodel.BaseModelWithSensitiveInfo`:
 * every Span and Event serializes with sensitive payload fields masked by
 * default, embedded Agent Spec components serialized through the regular
 * serialization machinery pinned to the CURRENT spec version (so component
 * level sensitive fields such as `LlmConfig.api_key` are redacted), and a
 * `type` discriminant carrying the class name appended to the dump.
 */
import { CURRENT_VERSION } from "../versioning.js";
import { isComponent } from "../component.js";
import { SerializationContext, camelToSnake } from "../serialization/serialization-context.js";
import { BuiltinsComponentSerializationPlugin } from "../serialization/builtin-serialization-plugin.js";
import { DANGEROUS_KEYS } from "../serialization/types.js";
import { Message, ToolCall } from "./message.js";

/** Placeholder value replacing sensitive fields in masked dumps (Python `_PII_MASK`). */
export const PII_MASK = "** MASKED **";

/** Maximum recursion depth when dumping tracing payloads. */
const MAX_TRACING_DUMP_DEPTH = 100;

/**
 * Sensitive payload fields per tracing type (Python `SensitiveField` markers).
 * Field names are the in-memory camelCase names; they are masked wholesale
 * with {@link PII_MASK} on serialization unless masking is explicitly opted
 * out of. Types without sensitive fields are omitted.
 */
export const TRACING_SENSITIVE_FIELDS: Readonly<Record<string, ReadonlySet<string>>> = {
  ExceptionRaised: new Set(["exceptionMessage", "exceptionStacktrace"]),
  AgentExecutionStart: new Set(["inputs"]),
  AgentExecutionEnd: new Set(["outputs"]),
  FlowExecutionStart: new Set(["inputs"]),
  FlowExecutionEnd: new Set(["outputs"]),
  NodeExecutionStart: new Set(["inputs"]),
  NodeExecutionEnd: new Set(["outputs"]),
  ManagerWorkersExecutionStart: new Set(["inputs"]),
  ManagerWorkersExecutionEnd: new Set(["outputs"]),
  SwarmExecutionStart: new Set(["inputs"]),
  SwarmExecutionEnd: new Set(["outputs"]),
  HumanInTheLoopRequest: new Set(["content"]),
  HumanInTheLoopResponse: new Set(["content"]),
  LlmGenerationRequest: new Set(["prompt"]),
  LlmGenerationResponse: new Set(["content", "toolCalls"]),
  LlmGenerationChunkReceived: new Set(["content", "toolCalls"]),
  ToolExecutionRequest: new Set(["inputs"]),
  ToolExecutionResponse: new Set(["outputs"]),
  ToolExecutionStreamingChunkReceived: new Set(["content"]),
  StateSnapshotEmitted: new Set(["stateSnapshot", "extraState"]),
};

/**
 * Fields holding plain model objects (not components, not user data) whose
 * keys must be converted to snake_case with unset values excluded, matching
 * Python's `LlmGenerationConfig.model_dump(exclude_none=True)`.
 */
const TRACING_MODEL_OBJECT_FIELDS: Readonly<Record<string, ReadonlySet<string>>> = {
  LlmGenerationRequest: new Set(["llmGenerationConfig"]),
};

/**
 * In-memory bookkeeping fields never included in serialized dumps
 * (Python models them as pydantic private attributes).
 */
const EXCLUDED_TRACING_FIELDS: ReadonlySet<string> = new Set(["parentSpan"]);

export interface TracingSerializeOptions {
  /**
   * Whether sensitive payload fields are replaced with {@link PII_MASK}.
   * Defaults to `true`; pass `false` only as an explicit opt-out (mirrors
   * Python's `model_dump(mask_sensitive_information=...)`).
   */
  maskSensitiveInformation?: boolean;
}

/**
 * Base class of all tracing spans and events.
 *
 * Provides {@link serialize}: dumps own fields with snake_case wire names,
 * masks sensitive fields by default, embeds Agent Spec components through the
 * regular serialization context pinned to {@link CURRENT_VERSION}, and appends
 * the `type` discriminant. Do not log or serialize spans/events outside
 * `serialize()` — the in-memory objects hold unmasked payloads.
 */
export abstract class TracingSerializable {
  /** The tracing class name, used as the `type` discriminant on the wire. */
  abstract get type(): string;

  serialize(options?: TracingSerializeOptions): Record<string, unknown> {
    const mask = options?.maskSensitiveInformation ?? true;
    // Fresh context per dump, pinned to the current spec version — mirrors
    // Python's `_TracingSerializationContextImpl` (component-level sensitive
    // fields stay redacted: `includeSensitiveFields` is never set here).
    const context = new SerializationContext(
      [new BuiltinsComponentSerializationPlugin()],
      { targetVersion: CURRENT_VERSION },
    );
    const sensitiveFields = TRACING_SENSITIVE_FIELDS[this.type];
    const modelObjectFields = TRACING_MODEL_OBJECT_FIELDS[this.type];

    const serialized: Record<string, unknown> = {};
    for (const [fieldName, value] of Object.entries(this)) {
      if (fieldName.startsWith("_") || EXCLUDED_TRACING_FIELDS.has(fieldName)) {
        continue;
      }
      const wireName = camelToSnake(fieldName);
      if (mask && sensitiveFields?.has(fieldName)) {
        serialized[wireName] = PII_MASK;
        continue;
      }
      if (
        modelObjectFields?.has(fieldName) &&
        value !== null &&
        typeof value === "object" &&
        !Array.isArray(value)
      ) {
        serialized[wireName] = context.dumpModelObject(
          value as Record<string, unknown>,
          /* excludeNulls */ true,
        );
        continue;
      }
      serialized[wireName] = dumpTracingValue(value, context, mask);
    }
    serialized["type"] = this.type;
    return serialized;
  }
}

/**
 * Dump a single tracing field value.
 *
 * Nested spans/events propagate the masking flag (a deliberate hardening over
 * Python, where nested events inside a span dump bypass the masking override);
 * Message/ToolCall models dump to their fixed wire shapes; components go
 * through the serialization context; plain arrays/objects are carried through
 * with their keys preserved.
 */
function dumpTracingValue(
  value: unknown,
  context: SerializationContext,
  mask: boolean,
  depth = 0,
): unknown {
  if (depth > MAX_TRACING_DUMP_DEPTH) {
    throw new Error(
      `Tracing serialization nesting depth exceeds maximum of ${MAX_TRACING_DUMP_DEPTH}`,
    );
  }
  if (value === null || value === undefined) {
    return null;
  }
  if (value instanceof TracingSerializable) {
    return value.serialize({ maskSensitiveInformation: mask });
  }
  if (value instanceof Message || value instanceof ToolCall) {
    return value.toWireDict();
  }
  if (isComponent(value)) {
    return context.dumpComponentToDict(value);
  }
  if (Array.isArray(value)) {
    return value.map((item) => dumpTracingValue(item, context, mask, depth + 1));
  }
  if (typeof value === "object") {
    const result: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
      if (DANGEROUS_KEYS.has(key)) continue;
      result[key] = dumpTracingValue(item, context, mask, depth + 1);
    }
    return result;
  }
  return value;
}

/**
 * Current timestamp in nanoseconds since the Unix epoch (Python
 * `time.time_ns()`; sub-millisecond digits are always zero in JS).
 */
export function nowNs(): number {
  return Date.now() * 1e6;
}
