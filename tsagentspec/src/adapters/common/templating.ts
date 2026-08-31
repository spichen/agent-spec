/**
 * Template rendering helpers shared by the AgentSpec adapters.
 *
 * Mirrors `pyagentspec.adapters._utils.render_template` and
 * `render_nested_object_template`: `{{placeholder}}` occurrences whose names
 * appear in `inputs` are substituted, unknown placeholders are left verbatim.
 *
 * Divergence from Python (see the adapter README): Python renders values with
 * `str()`; TypeScript uses `String()` for primitives and `JSON.stringify` for
 * objects/arrays.
 */
import { TEMPLATE_PLACEHOLDER_REGEXP } from "../../templating.js";

/** Render a value for insertion into a template string. */
export function stringifyTemplateValue(value: unknown): string {
  if (typeof value === "object" && value !== null) {
    return JSON.stringify(value);
  }
  return String(value);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null) return false;
  const prototype: unknown = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

/**
 * Render a template string using the given inputs.
 *
 * Placeholders whose names are keys of `inputs` are replaced with the
 * stringified value; unknown placeholders stay verbatim. Non-string templates
 * are stringified and returned unchanged otherwise.
 */
export function renderTemplate(
  template: unknown,
  inputs: Record<string, unknown>,
): string {
  if (typeof template !== "string") {
    return stringifyTemplateValue(template);
  }
  // Fresh regexp instance: the shared exported one is global and stateful.
  const re = new RegExp(TEMPLATE_PLACEHOLDER_REGEXP.source, "g");
  const renderedParts: string[] = [];
  let lastEnd = 0;
  let match: RegExpExecArray | null;
  while ((match = re.exec(template)) !== null) {
    renderedParts.push(template.slice(lastEnd, match.index));
    // Original placeholder text as it appeared in the template, including
    // braces and inner whitespace.
    const fullPlaceholder = match[0];
    // Only the placeholder name, extracted from inside {{ ... }}.
    const inputTitle = match[1];
    if (inputTitle !== undefined && Object.hasOwn(inputs, inputTitle)) {
      renderedParts.push(stringifyTemplateValue(inputs[inputTitle]));
    } else {
      renderedParts.push(fullPlaceholder);
    }
    lastEnd = match.index + fullPlaceholder.length;
  }
  renderedParts.push(template.slice(lastEnd));
  return renderedParts.join("");
}

/**
 * Recursively render `{{placeholder}}` templates inside an arbitrarily nested
 * structure of strings, byte arrays, plain objects, arrays and sets. Object
 * keys and values are both rendered; any other value is returned untouched.
 */
export function renderNestedObjectTemplate(
  object: unknown,
  inputs: Record<string, unknown>,
): unknown {
  if (typeof object === "string") {
    return renderTemplate(object, inputs);
  }
  if (object instanceof Uint8Array) {
    // Python decodes bytes as UTF-8 with errors="replace"; the non-fatal
    // TextDecoder replaces invalid sequences the same way.
    return renderNestedObjectTemplate(
      new TextDecoder("utf-8", { fatal: false }).decode(object),
      inputs,
    );
  }
  if (Array.isArray(object)) {
    return object.map((item) => renderNestedObjectTemplate(item, inputs));
  }
  if (object instanceof Set) {
    return new Set(
      [...object].map((item) => renderNestedObjectTemplate(item, inputs)),
    );
  }
  if (isPlainObject(object)) {
    const rendered: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(object)) {
      rendered[renderTemplate(key, inputs)] = renderNestedObjectTemplate(
        value,
        inputs,
      );
    }
    return rendered;
  }
  return object;
}
