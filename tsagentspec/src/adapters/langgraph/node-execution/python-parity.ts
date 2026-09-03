/**
 * Python-semantics value coercion for the LangGraph flow node executors.
 *
 * TS-only emulation layer with no direct Python counterpart: it reproduces
 * the behavior Python gets for free from `json.dumps`, `int()` and `float()`
 * inside `_cast_values_and_add_defaults` (`_node_execution.py`), so casting
 * node values produces byte-identical flow state text across both SDKs.
 */
import type { Property } from "../../../property.js";
import type { NodeOutputs } from "../types.js";

/** Serialize one string the way Python's `json.dumps` does (ensure_ascii). */
function pythonJsonDumpsString(value: string): string {
  let out = '"';
  for (const ch of value) {
    const code = ch.codePointAt(0)!;
    if (ch === '"') out += '\\"';
    else if (ch === "\\") out += "\\\\";
    else if (ch === "\b") out += "\\b";
    else if (ch === "\f") out += "\\f";
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (code < 0x20 || code > 0x7e) {
      if (code > 0xffff) {
        // ensure_ascii escapes astral characters as a surrogate pair.
        const high = 0xd800 + ((code - 0x10000) >> 10);
        const low = 0xdc00 + ((code - 0x10000) & 0x3ff);
        out += `\\u${high.toString(16).padStart(4, "0")}`;
        out += `\\u${low.toString(16).padStart(4, "0")}`;
      } else {
        out += `\\u${code.toString(16).padStart(4, "0")}`;
      }
    } else out += ch;
  }
  return out + '"';
}

/**
 * Serialize a value the way Python's `json.dumps` does with its default
 * arguments: `", "` / `": "` separators, ensure_ascii `\uXXXX` escapes, and
 * `Infinity`/`-Infinity`/`NaN` literals (allow_nan). Used when casting
 * non-string values into `string`-typed properties so the resulting flow
 * state text matches the Python adapter byte-for-byte.
 */
export function pythonJsonDumps(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    if (Number.isFinite(value)) return JSON.stringify(value);
    if (value === Infinity) return "Infinity";
    if (value === -Infinity) return "-Infinity";
    return "NaN";
  }
  if (typeof value === "string") return pythonJsonDumpsString(value);
  if (Array.isArray(value)) {
    return `[${value.map((item) => pythonJsonDumps(item)).join(", ")}]`;
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, v]) => v !== undefined && typeof v !== "function")
      .map(([k, v]) => `${pythonJsonDumpsString(k)}: ${pythonJsonDumps(v)}`);
    return `{${entries.join(", ")}}`;
  }
  return JSON.stringify(value) ?? "null";
}

/** Digit run with Python's underscore separators (`1_000`, not `1__0`). */
const PY_DIGITS = String.raw`\d(?:_?\d)*`;

/** Python `int()` string grammar: optional sign + underscore-separated digits. */
const PYTHON_INT_REGEXP = new RegExp(`^[+-]?${PY_DIGITS}$`);

/** Python `float()` numeric grammar (decimal/scientific, no hex/binary/octal). */
const PYTHON_FLOAT_REGEXP = new RegExp(
  `^[+-]?(?:(?:${PY_DIGITS})?\\.${PY_DIGITS}|${PY_DIGITS}\\.?)(?:[eE][+-]?${PY_DIGITS})?$`,
);

/**
 * Parse a (trimmed) string with Python `float()` semantics: decimal and
 * scientific forms plus `inf`/`infinity`/`nan` (any case, optional sign) and
 * underscore digit separators. Returns `undefined` for anything Python's
 * `float()` rejects (hex/binary/octal literals, `1__0`, empty strings, ...).
 */
function parsePythonFloat(text: string): number | undefined {
  const unsigned = text.toLowerCase().replace(/^[+-]/, "");
  if (unsigned === "inf" || unsigned === "infinity") {
    return text.startsWith("-") ? -Infinity : Infinity;
  }
  if (unsigned === "nan") return NaN;
  if (!PYTHON_FLOAT_REGEXP.test(text)) return undefined;
  const parsed = Number(text.replace(/_/g, ""));
  return Number.isNaN(parsed) ? undefined : parsed;
}

/**
 * Cast the given values to the types declared by the properties and add
 * missing defaults, mirroring Python's `_cast_values_and_add_defaults`:
 * non-strings are `json.dumps`-serialized into `string` properties, numbers
 * become booleans, numeric strings parse into `integer`/`number` properties
 * (an unparsable integer string raises like Python's `int()`; an unparsable
 * number string is left as-is like Python's swallowed `float()` error), and
 * a property with neither value nor default raises. Values for undeclared
 * properties are dropped.
 */
export function castValuesAndAddDefaults(
  valuesDict: Record<string, unknown>,
  properties: Property[],
  nodeName: string,
): NodeOutputs {
  const resultsDict: NodeOutputs = {};
  for (const property of properties) {
    const key = property.title;
    if (Object.hasOwn(valuesDict, key)) {
      let value = valuesDict[key];
      const propertyType = property.type;
      if (propertyType === "string" && typeof value !== "string") {
        value = pythonJsonDumps(value);
      } else if (propertyType === "boolean" && typeof value === "number") {
        value = Boolean(value);
      } else if (propertyType === "integer" && typeof value === "boolean") {
        value = value ? 1 : 0;
      } else if (propertyType === "integer" && typeof value === "number") {
        value = Math.trunc(value);
      } else if (propertyType === "integer" && typeof value === "string") {
        // Python does `int(value.strip())` and re-raises for any unparsable
        // string (its error-message guard never matches `int()`'s text), so
        // an unparsable integer string aborts the flow here too.
        const trimmed = value.trim();
        if (PYTHON_INT_REGEXP.test(trimmed)) {
          value = parseInt(trimmed.replace(/_/g, ""), 10);
        } else {
          // Python raises ValueError with this exact message (repr'd value).
          throw new Error(
            `invalid literal for int() with base 10: ${JSON.stringify(trimmed)}`,
          );
        }
      } else if (propertyType === "number" && typeof value === "boolean") {
        value = value ? 1 : 0;
      } else if (propertyType === "number" && typeof value === "string") {
        // Try converting numeric strings to floats with Python `float()`
        // semantics; if the parse fails, leave the string as-is (Python
        // swallows the `could not convert string to float:` error).
        const parsed = parsePythonFloat(value.trim());
        if (parsed !== undefined) {
          value = parsed;
        }
      }
      resultsDict[key] = value;
    } else if (property.default !== undefined) {
      resultsDict[key] = property.default;
    } else {
      throw new Error(
        `Expected node \`${nodeName}\` to have a value ` +
          `for property \`${property.title}\`, but none was found.`,
      );
    }
  }
  return resultsDict;
}
