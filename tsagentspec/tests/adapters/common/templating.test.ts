/**
 * Tests for the shared adapter template rendering helpers.
 *
 * Ports `pyagentspec/tests/adapters/test_template_rendering.py` (minus the
 * tuple/tuple-key cases, which have no JS equivalent).
 *
 * Documented divergence exercised here: Python renders values with `str()`;
 * TypeScript uses `String()` for primitives and `JSON.stringify` for
 * objects/arrays.
 */
import { describe, expect, it } from "vitest";
import {
  renderNestedObjectTemplate,
  renderTemplate,
  stringifyTemplateValue,
} from "../../../src/adapters/common/templating.js";

describe("renderTemplate", () => {
  const cases: Array<
    [template: string, inputs: Record<string, unknown>, expected: string]
  > = [
    ["a", {}, "a"],
    ["{{a}}", { a: 1 }, "1"],
    ["{{ a}} {{b }}", { a: 1, b: 2 }, "1 2"],
    ["{{ a} {b }}", { a: 1, b: 2 }, "{{ a} {b }}"],
    ["{{ a}{}{b }}", { a: 1, b: 2 }, "{{ a}{}{b }}"],
    ["{{ a a a a }}", { a: 1 }, "{{ a a a a }}"],
    ["{{a}}{{b}}{{a}}{{a}}", { a: 1, b: 2 }, "1211"],
    ["{{ b{{a}} }}{{b1}}", { a: 1, b: 2, b1: 3 }, "{{ b1 }}3"],
    ["{{{{a}}}}", { a: " b ", b: 2 }, "{{ b }}"],
    // Rendered values are never re-scanned for placeholders.
    ["{{a}}{{b}}", { a: "{{b}}", b: 2 }, "{{b}}2"],
    ["{{a}}{{b}}", { b: 2, a: "{{b}}" }, "{{b}}2"],
    // Input keys are matched literally, never as regular expressions.
    ["{{a}}", { ".*": "b" }, "{{a}}"],
    ["{{a}}", { "a|b": "c" }, "{{a}}"],
    ["{{a}}", { "[abc]": "b" }, "{{a}}"],
    ["{{a}}", { "b)": "b" }, "{{a}}"],
    [
      "Here is the equation: {{a}} plus {{b}} equals {{c}}",
      { a: "{{", b: "}}", plus: "SECRET" },
      "Here is the equation: {{ plus }} equals {{c}}",
    ],
    [
      "{{a}}{{b}}",
      { a: "{{sec", b: "ret}}", secret: "SECRET" },
      "{{secret}}",
    ],
  ];

  it.each(cases)("renders %j with %j", (template, inputs, expected) => {
    expect(renderTemplate(template, inputs)).toBe(expected);
  });

  it("stringifies object and array values with JSON.stringify (TS divergence)", () => {
    expect(renderTemplate("{{a}}", { a: { b: 1 } })).toBe('{"b":1}');
    expect(renderTemplate("{{a}}", { a: [1, "x"] })).toBe('[1,"x"]');
  });

  it("stringifies primitive values with String", () => {
    expect(renderTemplate("{{a}}", { a: true })).toBe("true");
    expect(renderTemplate("{{a}}", { a: null })).toBe("null");
    expect(renderTemplate("{{a}}", { a: 1.5 })).toBe("1.5");
  });

  it("does not resolve placeholders from the object prototype", () => {
    expect(renderTemplate("{{toString}}", {})).toBe("{{toString}}");
    expect(renderTemplate("{{constructor}}", {})).toBe("{{constructor}}");
  });

  it("stringifies non-string templates without rendering", () => {
    expect(renderTemplate(5, {})).toBe("5");
    expect(renderTemplate({ a: "{{x}}" }, { x: 1 })).toBe('{"a":"{{x}}"}');
  });
});

describe("stringifyTemplateValue", () => {
  it("uses String for primitives and JSON.stringify for objects", () => {
    expect(stringifyTemplateValue("s")).toBe("s");
    expect(stringifyTemplateValue(3)).toBe("3");
    expect(stringifyTemplateValue({ a: 1 })).toBe('{"a":1}');
    expect(stringifyTemplateValue([1, 2])).toBe("[1,2]");
  });
});

describe("renderNestedObjectTemplate", () => {
  const cases: Array<[template: unknown, inputs: Record<string, unknown>, expected: unknown]> = [
    ["a", {}, "a"],
    ["{{a}}", { a: 1 }, "1"],
    ["{{ a}} {{b }}", { a: 1, b: 2 }, "1 2"],
    [
      { "{{a}}": "{{a}}{{b}}" },
      { a: "{{b}}", b: 2 },
      { "{{b}}": "{{b}}2" },
    ],
    [
      { "{{a}}": { "{{a}}{{b}}": { "{{b}}": "{{a}}" } } },
      { a: "{{b}}", b: 2 },
      { "{{b}}": { "{{b}}2": { "2": "{{b}}" } } },
    ],
    [
      [{ "id_{{a}}": "v{{b}}" }, { inner: { k: "{{c}}" } }],
      { a: 1, b: 2, c: 3 },
      [{ id_1: "v2" }, { inner: { k: "3" } }],
    ],
    [
      { "{{a}}": [{ "{{b}}": "v{{c}}" }] },
      { a: "A", b: "B", c: "C" },
      { A: [{ B: "vC" }] },
    ],
    [
      { l1: [{ l2: [{ l3: "x {{x}}" }] }], "k{{y}}": "v" },
      { x: "X", y: "Y" },
      { l1: [{ l2: [{ l3: "x X" }] }], kY: "v" },
    ],
    [
      ["pre {{p}}", { mid: ["{{p}}", { deep: "d{{d}}" }] }, "suf {{s}}"],
      { p: "P", d: "D", s: "S" },
      ["pre P", { mid: ["P", { deep: "dD" }] }, "suf S"],
    ],
    [
      { mix: [null, 0, "{{z}}", { inner: [true, "{{z}}"] }] },
      { z: "Z" },
      { mix: [null, 0, "Z", { inner: [true, "Z"] }] },
    ],
  ];

  it.each(cases)("renders nested %j", (template, inputs, expected) => {
    expect(renderNestedObjectTemplate(template, inputs)).toEqual(expected);
  });

  it("renders inside sets", () => {
    expect(
      renderNestedObjectTemplate({ set: new Set(["a", "s{{x}}"]) }, { x: "X" }),
    ).toEqual({ set: new Set(["a", "sX"]) });
  });

  it("decodes Uint8Array as UTF-8 before rendering (Python bytes)", () => {
    const bytes = new TextEncoder().encode("b{{bb}}");
    expect(
      renderNestedObjectTemplate({ bytes: [bytes, { k: "{{bb2}}" }] }, {
        bb: "BB",
        bb2: "B2",
      }),
    ).toEqual({ bytes: ["bBB", { k: "B2" }] });
  });

  it("leaves non-plain objects untouched", () => {
    const date = new Date(0);
    expect(renderNestedObjectTemplate(date, { a: 1 })).toBe(date);
    expect(renderNestedObjectTemplate(42, { a: 1 })).toBe(42);
  });
});
