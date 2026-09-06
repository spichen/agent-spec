/**
 * Record-shape guards shared by the AgentSpec adapters.
 *
 * Two DIFFERENT contracts live here on purpose — pick the one matching the
 * call site's semantics, they are not interchangeable:
 * - `isPlainRecord` is strict: only plain objects (`Object.prototype` or
 *   `null` prototype). Used where the value is about to be treated as pure
 *   data (body encoding, template recursion) and class instances / Maps must
 *   NOT match.
 * - `isRecordLike` is loose: any non-array object, class instances included.
 *   Used for structural probes over runtime state and third-party objects.
 */

/** Strict record check: plain objects only (prototype-checked). */
export function isPlainRecord(
  value: unknown,
): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  const prototype: unknown = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

/** Loose record check: any non-array object (class instances included). */
export function isRecordLike(
  value: unknown,
): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
