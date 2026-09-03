/**
 * JSON-schema helpers shared by the AgentSpec adapters.
 *
 * `buildJsonSchemaFromProperties` builds an object schema from AgentSpec
 * properties, suitable as a LangChain tool argument schema. Schema *type
 * comparison* lives in the SDK's canonical property layer:
 * `jsonSchemasHaveSameType` in `src/property.ts` (re-exported through
 * `common/index.ts`).
 */
import type { JsonSchemaValue, Property } from "../../property.js";

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
