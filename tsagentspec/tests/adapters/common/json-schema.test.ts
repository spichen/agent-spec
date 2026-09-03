/**
 * Tests for the shared JSON-schema helpers.
 *
 * `jsonSchemasHaveSameType` is the SDK's canonical port of
 * `pyagentspec.property.json_schemas_have_same_type` (in `src/property.ts`,
 * re-exported through the adapter common barrel);
 * `buildJsonSchemaFromProperties` builds LangChain tool argument schemas from
 * AgentSpec properties (defaults excluded from `required`, mirroring the
 * Python generated pydantic models).
 */
import { describe, expect, it } from "vitest";
import type { JsonSchemaValue } from "../../../src/index.js";
import { integerProperty, stringProperty } from "../../../src/index.js";
import { jsonSchemasHaveSameType } from "../../../src/property.js";
import { buildJsonSchemaFromProperties } from "../../../src/adapters/common/json-schema.js";

describe("jsonSchemasHaveSameType", () => {
  it("matches identical basic types and rejects different ones", () => {
    expect(
      jsonSchemasHaveSameType({ type: "integer" }, { type: "integer" }),
    ).toBe(true);
    expect(
      jsonSchemasHaveSameType({ type: "integer" }, { type: "string" }),
    ).toBe(false);
  });

  it("ignores non-type keys such as title and description", () => {
    expect(
      jsonSchemasHaveSameType(
        { title: "a", type: "integer", description: "x" },
        { title: "b", type: "integer" },
      ),
    ).toBe(true);
  });

  it("treats anyOf and type lists as equivalent unions, order-insensitively", () => {
    const anyOf: JsonSchemaValue = {
      anyOf: [{ type: "string" }, { type: "integer" }],
    };
    const typeList: JsonSchemaValue = { type: ["integer", "string"] };
    expect(jsonSchemasHaveSameType(anyOf, typeList)).toBe(true);
    expect(jsonSchemasHaveSameType(typeList, anyOf)).toBe(true);
    expect(
      jsonSchemasHaveSameType(anyOf, { type: ["integer", "boolean"] }),
    ).toBe(false);
  });

  it("compares array item types", () => {
    expect(
      jsonSchemasHaveSameType(
        { type: "array", items: { type: "string" } },
        { type: "array", items: { type: "string" } },
      ),
    ).toBe(true);
    expect(
      jsonSchemasHaveSameType(
        { type: "array", items: { type: "string" } },
        { type: "array", items: { type: "integer" } },
      ),
    ).toBe(false);
    // Missing items on one side compares against {}.
    expect(
      jsonSchemasHaveSameType(
        { type: "array" },
        { type: "array", items: { type: "string" } },
      ),
    ).toBe(false);
  });

  it("compares object property sets and their types", () => {
    const a: JsonSchemaValue = {
      type: "object",
      properties: { x: { type: "integer" }, y: { type: "string" } },
    };
    expect(
      jsonSchemasHaveSameType(a, {
        type: "object",
        properties: { y: { type: "string" }, x: { type: "integer" } },
      }),
    ).toBe(true);
    expect(
      jsonSchemasHaveSameType(a, {
        type: "object",
        properties: { x: { type: "integer" } },
      }),
    ).toBe(false);
    expect(
      jsonSchemasHaveSameType(a, {
        type: "object",
        properties: { x: { type: "integer" }, y: { type: "boolean" } },
      }),
    ).toBe(false);
  });

  it("compares additionalProperties strictly when boolean", () => {
    expect(
      jsonSchemasHaveSameType(
        { type: "object", additionalProperties: false },
        { type: "object", additionalProperties: false },
      ),
    ).toBe(true);
    expect(
      jsonSchemasHaveSameType(
        { type: "object", additionalProperties: false },
        { type: "object" },
      ),
    ).toBe(false);
    expect(
      jsonSchemasHaveSameType(
        { type: "object", additionalProperties: { type: "string" } },
        { type: "object", additionalProperties: { type: "integer" } },
      ),
    ).toBe(false);
  });

  it("throws on allOf and oneOf", () => {
    expect(() =>
      jsonSchemasHaveSameType({ allOf: [] }, { type: "string" }),
    ).toThrow("Support for schemas using allOf is not implemented.");
    expect(() =>
      jsonSchemasHaveSameType({ type: "string" }, { oneOf: [] }),
    ).toThrow("Support for schemas using oneOf is not implemented.");
  });

  it("throws when a union has more than 100 member types", () => {
    const big: JsonSchemaValue = {
      anyOf: Array.from({ length: 101 }, () => ({ type: "string" })),
    };
    expect(() => jsonSchemasHaveSameType(big, { type: "string" })).toThrow(
      "The schema is the union of more than 100 types.",
    );
  });
});

describe("buildJsonSchemaFromProperties", () => {
  it("builds an object schema with required for default-less properties", () => {
    const schema = buildJsonSchemaFromProperties("myToolArgs", [
      integerProperty({ title: "x" }),
      stringProperty({ title: "y" }),
    ]);
    expect(schema).toEqual({
      title: "myToolArgs",
      type: "object",
      properties: {
        x: { title: "x", type: "integer" },
        y: { title: "y", type: "string" },
      },
      required: ["x", "y"],
    });
  });

  it("includes defaults and drops defaulted properties from required", () => {
    const schema = buildJsonSchemaFromProperties("args", [
      integerProperty({ title: "x" }),
      integerProperty({ title: "n", default: 3 }),
    ]);
    expect(schema["required"]).toEqual(["x"]);
    expect(
      (schema["properties"] as Record<string, JsonSchemaValue>)["n"],
    ).toEqual({ title: "n", type: "integer", default: 3 });
  });

  it("omits required entirely when every property has a default", () => {
    const schema = buildJsonSchemaFromProperties("args", [
      stringProperty({ title: "s", default: "hello" }),
    ]);
    expect("required" in schema).toBe(false);
  });

  it("copies the property description into the schema when missing there", () => {
    // Hand-built property whose jsonSchema carries no description.
    const schema = buildJsonSchemaFromProperties("args", [
      {
        title: "s",
        description: "a string input",
        jsonSchema: { title: "s", type: "string" },
        default: undefined,
        type: "string",
      },
    ]);
    expect(
      (schema["properties"] as Record<string, JsonSchemaValue>)["s"],
    ).toEqual({
      title: "s",
      type: "string",
      description: "a string input",
    });
  });

  it("keeps an existing schema description over the property description", () => {
    const schema = buildJsonSchemaFromProperties("args", [
      {
        title: "s",
        description: "property description",
        jsonSchema: { title: "s", type: "string", description: "schema wins" },
        default: undefined,
        type: "string",
      },
    ]);
    expect(
      (schema["properties"] as Record<string, JsonSchemaValue>)["s"]?.[
        "description"
      ],
    ).toBe("schema wins");
  });

  it("builds an empty schema for no properties", () => {
    expect(buildJsonSchemaFromProperties("args", [])).toEqual({
      title: "args",
      type: "object",
      properties: {},
    });
  });
});
