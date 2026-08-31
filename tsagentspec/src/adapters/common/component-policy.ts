/**
 * Component allow/block policy helpers used while loading Agent Spec
 * configurations. Port of `pyagentspec.serialization.componentpolicy`.
 *
 * The TypeScript SDK has no Component class hierarchy, so policy entries are
 * componentType strings: concrete component type names (exact match,
 * distance 0), the SDK's `AbstractComponentType` group names (group match,
 * distance 1), or the `"Component"`/`"ComponentWithIO"` wildcards
 * (distance 2). When allow and block entries both match, the closest match
 * wins; block entries win same-distance ties. Names that resolve to neither a
 * known concrete type nor a group match only that exact serialized
 * componentType (distance 0), like unresolved names in Python.
 */
import type { ComponentBase } from "../../component.js";
import { getChildrenFromFieldValue } from "../../serialization/referencing.js";
import { OPAQUE_FIELDS } from "../../serialization/types.js";

/** A single policy entry: a concrete or abstract componentType name. */
export type ComponentPolicyEntry = string;

/** Policy input: a single componentType name or an iterable of names. */
export type ComponentPolicyInput =
  | ComponentPolicyEntry
  | Iterable<ComponentPolicyEntry>;

const CONCRETE_MATCH_DISTANCE = 0;
const ABSTRACT_GROUP_MATCH_DISTANCE = 1;
const WILDCARD_MATCH_DISTANCE = 2;

// Concrete members of each AgenticComponentUnion entry (src/agents/index.ts).
const AGENTIC_COMPONENT_TYPES = [
  "Agent",
  "Swarm",
  "ManagerWorkers",
  "RemoteAgent",
  "A2AAgent",
  "SpecializedAgent",
];

// Concrete members of NodeUnion (src/flows/nodes/index.ts).
const NODE_TYPES = [
  "StartNode",
  "EndNode",
  "LlmNode",
  "ToolNode",
  "AgentNode",
  "FlowNode",
  "BranchingNode",
  "MapNode",
  "ParallelMapNode",
  "ParallelFlowNode",
  "ApiNode",
  "InputMessageNode",
  "OutputMessageNode",
  "CatchExceptionNode",
];

// Concrete members of ToolUnion (src/tools/index.ts).
const TOOL_TYPES = ["ServerTool", "ClientTool", "RemoteTool", "BuiltinTool", "MCPTool"];

/**
 * Membership of the SDK's abstract component groups, keyed by
 * `AbstractComponentType` name (src/component.ts). Hardcoded from the SDK's
 * discriminated unions:
 * - AgenticComponentUnion (src/agents/index.ts)
 * - NodeUnion (src/flows/nodes/index.ts)
 * - ToolUnion (src/tools/index.ts)
 * - LlmConfigUnion (src/llms/index.ts)
 * - ToolBoxUnion (src/tools/toolbox.ts)
 * - OciClientConfigUnion (src/llms/oci-client-config.ts)
 * - ClientTransportUnion (src/mcp/client-transport.ts)
 * - SupportedDatastoresSchema (src/transforms/message-transform.ts)
 * - MessageTransformUnion (src/transforms/message-transform.ts)
 */
const ABSTRACT_COMPONENT_GROUPS: Record<string, ReadonlySet<string>> = {
  AgenticComponent: new Set(AGENTIC_COMPONENT_TYPES),
  Node: new Set(NODE_TYPES),
  Tool: new Set(TOOL_TYPES),
  LlmConfig: new Set([
    "OpenAiCompatibleConfig",
    "OllamaConfig",
    "VllmConfig",
    "OpenAiConfig",
    "OciGenAiConfig",
  ]),
  ToolBox: new Set(["MCPToolBox"]),
  OciClientConfig: new Set([
    "OciClientConfigWithApiKey",
    "OciClientConfigWithInstancePrincipal",
    "OciClientConfigWithResourcePrincipal",
    "OciClientConfigWithSecurityToken",
  ]),
  ClientTransport: new Set([
    "StdioTransport",
    "SSETransport",
    "SSEmTLSTransport",
    "StreamableHTTPTransport",
    "StreamableHTTPmTLSTransport",
    "RemoteTransport",
  ]),
  Datastore: new Set([
    "InMemoryCollectionDatastore",
    "OracleDatabaseDatastore",
    "PostgresDatabaseDatastore",
  ]),
  MessageTransform: new Set([
    "MessageSummarizationTransform",
    "ConversationSummarizationTransform",
  ]),
};

/**
 * Every builtin componentType extending ComponentWithIOSchema (schemas built
 * on ComponentWithIOSchema / ToolBaseSchema / NodeBaseSchema across src/).
 */
const COMPONENT_WITH_IO_TYPES: ReadonlySet<string> = new Set([
  ...AGENTIC_COMPONENT_TYPES,
  ...NODE_TYPES,
  ...TOOL_TYPES,
  "MCPToolSpec",
  "Flow",
  "AgentSpecializationParameters",
]);

function normalizeComponentTypes(
  componentTypes: ComponentPolicyInput | undefined,
): string[] | undefined {
  if (componentTypes === undefined) {
    return undefined;
  }
  const entries: unknown[] =
    typeof componentTypes === "string"
      ? [componentTypes]
      : typeof (componentTypes as Iterable<string>)[Symbol.iterator] ===
          "function"
        ? [...componentTypes]
        : (() => {
            throw new Error(
              "`allowed_components` and `blocked_components` entries must be component " +
                `type names or Component classes, got ${String(componentTypes)}.`,
            );
          })();
  const normalizedComponentTypes: string[] = [];
  for (const componentType of entries) {
    if (typeof componentType !== "string") {
      throw new Error(
        "`allowed_components` and `blocked_components` entries must be component " +
          `type names or Component classes, got ${String(componentType)}.`,
      );
    }
    normalizedComponentTypes.push(componentType);
  }
  return normalizedComponentTypes;
}

/** Return the distance of the most specific matching policy entry. */
function getBestPolicyMatchDistance(
  componentType: string,
  policyEntries: readonly string[],
): number | undefined {
  let bestDistance: number | undefined;
  for (const entry of policyEntries) {
    let distance: number | undefined;
    if (entry === componentType) {
      distance = CONCRETE_MATCH_DISTANCE;
    } else if (entry === "Component") {
      // Matches every component, including unknown plugin-defined types.
      distance = WILDCARD_MATCH_DISTANCE;
    } else if (entry === "ComponentWithIO") {
      distance = COMPONENT_WITH_IO_TYPES.has(componentType)
        ? WILDCARD_MATCH_DISTANCE
        : undefined;
    } else if (ABSTRACT_COMPONENT_GROUPS[entry]?.has(componentType)) {
      distance = ABSTRACT_GROUP_MATCH_DISTANCE;
    }
    if (distance !== undefined && (bestDistance === undefined || distance < bestDistance)) {
      bestDistance = distance;
    }
  }
  return bestDistance;
}

/**
 * Allow/block policy for component types loaded from Agent Spec
 * configurations.
 *
 * If no allow list is given, all component types are allowed unless a
 * matching block entry applies; with an allow list, only matching component
 * types are allowed. When both allow and block entries match, the closest
 * hierarchy match wins and block entries win same-distance ties.
 */
export class ComponentLoadPolicy {
  /** Normalized allow-list entries, or undefined when no allow list is set. */
  readonly allowedComponents?: readonly string[];
  /** Normalized block-list entries (empty when none). */
  readonly blockedComponents: readonly string[];

  constructor(
    allowedComponents?: ComponentPolicyInput,
    blockedComponents?: ComponentPolicyInput,
  ) {
    this.allowedComponents = normalizeComponentTypes(allowedComponents);
    this.blockedComponents = normalizeComponentTypes(blockedComponents) ?? [];
  }

  /** Raise if the component type is disallowed by the policy. */
  validateComponentType(componentType: string): void {
    const blockedMatchDistance = getBestPolicyMatchDistance(
      componentType,
      this.blockedComponents,
    );
    const allowedMatchDistance =
      this.allowedComponents !== undefined
        ? getBestPolicyMatchDistance(componentType, this.allowedComponents)
        : undefined;

    if (
      blockedMatchDistance !== undefined &&
      (allowedMatchDistance === undefined ||
        blockedMatchDistance <= allowedMatchDistance)
    ) {
      throw new Error(
        `Loading Agent Spec component type \`${componentType}\` is in the block list.`,
      );
    }
    if (this.allowedComponents !== undefined && allowedMatchDistance === undefined) {
      throw new Error(
        `Loading Agent Spec component type \`${componentType}\` is not in the allow list.`,
      );
    }
  }

  /** Raise if the component is disallowed by the policy. */
  validateComponent(component: ComponentBase): void {
    this.validateComponentType(component.componentType);
  }

  /** Validate a constructed component and all nested child components. */
  validateComponentTree(component: ComponentBase): void {
    const componentsToCheck: ComponentBase[] = [component];
    const visitedComponents = new Set<ComponentBase>();
    while (componentsToCheck.length > 0) {
      const currentComponent = componentsToCheck.pop()!;
      if (visitedComponents.has(currentComponent)) {
        continue;
      }
      visitedComponents.add(currentComponent);

      this.validateComponent(currentComponent);
      const fields = currentComponent as unknown as Record<string, unknown>;
      for (const [fieldName, fieldValue] of Object.entries(fields)) {
        if (fieldName === "id" || fieldName === "componentType") continue;
        // Opaque fields hold user-controlled data; the structural isComponent
        // check would false-positive on plain dicts there (Python's
        // isinstance check cannot, since deserialized opaque data never
        // becomes Component instances).
        if (OPAQUE_FIELDS.has(fieldName)) continue;
        componentsToCheck.push(...getChildrenFromFieldValue(fieldValue));
      }
    }
  }
}
