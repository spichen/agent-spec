/**
 * JSON-schema helpers shared by the AgentSpec adapters.
 *
 * `jsonSchemasHaveSameType` ports `pyagentspec.property.json_schemas_have_same_type`;
 * `buildJsonSchemaFromProperties` builds an object schema from AgentSpec
 * properties, suitable as a LangChain tool argument schema.
 */
import type { JsonSchemaValue, Property } from "../../property.js";

const MAX_JSON_SCHEMA_UNION_TYPE_ALLOWED_LENGTH = 100;

/**
 * Normalization merges the basic types and anyOf for a schema and returns a
 * list containing all the schemas.
 */
function normalizeJsonSchemaUnionTypes(
  schema: JsonSchemaValue,
): JsonSchemaValue[] {
  const jsonSchemaType = schema["type"] ?? [];
  const jsonSchemaTypes: unknown[] = Array.isArray(jsonSchemaType)
    ? jsonSchemaType
    : [jsonSchemaType];

  const allTypes: JsonSchemaValue[] = [
    ...((schema["anyOf"] as JsonSchemaValue[] | undefined) ?? []),
  ];
  for (const type of jsonSchemaTypes) {
    if (type === "array") {
      // If one of the basic types is array, we put the items definition in it
      allTypes.push({ type: "array", items: schema["items"] ?? {} });
    } else if (type === "object") {
      // If one of the basic types is object, we put the properties definition in it
      allTypes.push({
        type: "object",
        properties: schema["properties"] ?? {},
        additionalProperties: schema["additionalProperties"] ?? false,
      });
    } else {
      // Normally we just carry over the basic type
      allTypes.push({ type });
    }
  }

  if (allTypes.length > MAX_JSON_SCHEMA_UNION_TYPE_ALLOWED_LENGTH) {
    throw new Error(
      `The schema is the union of more than ${MAX_JSON_SCHEMA_UNION_TYPE_ALLOWED_LENGTH}` +
        " types. This is not supported. Please consider simplifying the type definition or" +
        " using 'Any'.",
    );
  }
  return allTypes;
}

/** Check if the two schemas define the same type. */
export function jsonSchemasHaveSameType(
  jsonSchemaA: JsonSchemaValue,
  jsonSchemaB: JsonSchemaValue,
): boolean {
  if ("allOf" in jsonSchemaA || "allOf" in jsonSchemaB) {
    throw new Error("Support for schemas using allOf is not implemented.");
  }
  if ("oneOf" in jsonSchemaA || "oneOf" in jsonSchemaB) {
    throw new Error("Support for schemas using oneOf is not implemented.");
  }

  // Basic types must match
  if (
    "anyOf" in jsonSchemaA ||
    Array.isArray(jsonSchemaA["type"]) ||
    "anyOf" in jsonSchemaB ||
    Array.isArray(jsonSchemaB["type"])
  ) {
    // We need to combine anyOf and the list of types specified in type.
    // We normalize them to other json schemas, so that we can compare them
    // afterward using this method.
    const aTypeList = normalizeJsonSchemaUnionTypes(jsonSchemaA);
    const bTypeList = normalizeJsonSchemaUnionTypes(jsonSchemaB);
    // We make sure that the sets of possible types overlap correctly (same
    // elements). We cannot check the length directly, as the same type could
    // be repeated.
    for (const aType of aTypeList) {
      if (!bTypeList.some((bType) => jsonSchemasHaveSameType(aType, bType))) {
        return false;
      }
    }
    for (const bType of bTypeList) {
      if (!aTypeList.some((aType) => jsonSchemasHaveSameType(aType, bType))) {
        return false;
      }
    }
    // We flattened everything in the anyOf, so no need to go on with the checks
    return true;
  }
  if (jsonSchemaA["type"] !== jsonSchemaB["type"]) {
    return false;
  }

  // If it's an array, the items type must match
  if ("items" in jsonSchemaA || "items" in jsonSchemaB) {
    if (
      !jsonSchemasHaveSameType(
        (jsonSchemaA["items"] as JsonSchemaValue | undefined) ?? {},
        (jsonSchemaB["items"] as JsonSchemaValue | undefined) ?? {},
      )
    ) {
      return false;
    }
  }

  // If it's an object, the set of properties must match, and their types must match too
  if ("properties" in jsonSchemaA || "properties" in jsonSchemaB) {
    const aProperties = (jsonSchemaA["properties"] ?? {}) as Record<
      string,
      JsonSchemaValue
    >;
    const bProperties = (jsonSchemaB["properties"] ?? {}) as Record<
      string,
      JsonSchemaValue
    >;
    const aKeys = Object.keys(aProperties).sort();
    const bKeys = Object.keys(bProperties).sort();
    if (
      aKeys.length !== bKeys.length ||
      aKeys.some((key, index) => key !== bKeys[index])
    ) {
      return false;
    }
    for (const propertyName of aKeys) {
      if (
        !jsonSchemasHaveSameType(
          aProperties[propertyName]!,
          bProperties[propertyName]!,
        )
      ) {
        return false;
      }
    }
  }

  if (
    "additionalProperties" in jsonSchemaA ||
    "additionalProperties" in jsonSchemaB
  ) {
    const aAdditionalProperties = jsonSchemaA["additionalProperties"] ?? {};
    const bAdditionalProperties = jsonSchemaB["additionalProperties"] ?? {};
    // If any of the two additional properties is a boolean, check strict equality
    if (
      typeof aAdditionalProperties === "boolean" ||
      typeof bAdditionalProperties === "boolean"
    ) {
      return aAdditionalProperties === bAdditionalProperties;
    }
    if (
      !jsonSchemasHaveSameType(
        aAdditionalProperties as JsonSchemaValue,
        bAdditionalProperties as JsonSchemaValue,
      )
    ) {
      return false;
    }
  }
  return true;
}

/**
 * Build an object JSON schema from AgentSpec properties, suitable as a
 * LangChain tool argument schema. Each property contributes its own
 * `jsonSchema` (with its default included when set); properties without a
 * default are listed as required.
 */
export function buildJsonSchemaFromProperties(
  name: string,
  properties: Property[],
): JsonSchemaValue {
  const schemaProperties: Record<string, JsonSchemaValue> = {};
  const required: string[] = [];
  for (const property of properties) {
    const propertySchema: JsonSchemaValue = { ...property.jsonSchema };
    if (
      property.description !== undefined &&
      propertySchema["description"] === undefined
    ) {
      propertySchema["description"] = property.description;
    }
    if (property.default !== undefined) {
      propertySchema["default"] = property.default;
    } else {
      required.push(property.title);
    }
    schemaProperties[property.title] = propertySchema;
  }

  const schema: JsonSchemaValue = {
    title: name,
    type: "object",
    properties: schemaProperties,
  };
  if (required.length > 0) {
    schema["required"] = required;
  }
  return schema;
}
