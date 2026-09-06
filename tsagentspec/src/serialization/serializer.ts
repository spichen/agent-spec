/**
 * AgentSpecSerializer - main entry point for component serialization.
 *
 * Provides toJson and toYaml methods.
 */
import YAML from "yaml";

import type { AgentSpecVersion } from "../versioning.js";
import type { ComponentBase } from "../component.js";
import type { ComponentSerializationPlugin } from "./serialization-plugin.js";
import { SerializationContext } from "./serialization-context.js";
import { BuiltinsComponentSerializationPlugin } from "./builtin-serialization-plugin.js";
import type { SerializedDict, DisaggregatedComponentsDict } from "./types.js";

/**
 * Configuration of the components to disaggregate upon serialization,
 * mirroring Python's `DisaggregatedComponentsConfigT`. Each item is either:
 *
 * - a `ComponentBase`: disaggregated under its own id, or
 * - a `[ComponentBase, string]` pair: disaggregated under the custom id. The
 *   custom id is applied only as the serialization-time mapping key (the
 *   `$referenced_components` registry key and the `$component_ref` target);
 *   the component itself keeps its own `id` everywhere it is serialized.
 */
export type DisaggregatedComponentsConfig = ReadonlyArray<
  ComponentBase | readonly [ComponentBase, string]
>;

export class AgentSpecSerializer {
  private plugins: ComponentSerializationPlugin[];

  constructor(plugins?: ComponentSerializationPlugin[]) {
    this.plugins = [...(plugins ?? [])];

    // Always add the builtin plugin at the end
    this.plugins.push(new BuiltinsComponentSerializationPlugin());

    // Validate no plugin collisions by building a context
    new SerializationContext(this.plugins);
  }

  /** Serialize a component to a plain dict (internal) */
  private _toDict(
    component: ComponentBase,
    options?: {
      agentspecVersion?: AgentSpecVersion;
      disaggregatedComponents?: DisaggregatedComponentsConfig;
      exportDisaggregatedComponents?: boolean;
      camelCase?: boolean;
      includeSensitiveFields?: boolean;
    },
  ): SerializedDict | [SerializedDict, DisaggregatedComponentsDict] {
    const opts = options ?? {};
    const disaggregated = opts.disaggregatedComponents ?? [];
    const exportDisag = opts.exportDisaggregatedComponents ?? false;
    const useCamelCase = opts.camelCase ?? false;
    const includeSensitive = opts.includeSensitiveFields ?? false;

    if (includeSensitive) {
      console.warn(
        "includeSensitiveFields=true was set. Serialized output may contain " +
          "unredacted sensitive values; do not log, commit, or share it unless " +
          "those values are intended to be exposed.",
      );
    }

    // Normalize the disaggregated config to [component, mappedId] pairs and
    // build the id mapping (component id -> registry key). Like Python, a
    // custom id is only the serialization-time mapping key: the component
    // keeps its own id inside its serialized dump.
    const convertedConfig: Array<readonly [ComponentBase, string]> = [];
    const componentsIdMapping = new Map<string, string>();
    for (const entry of disaggregated) {
      if (Array.isArray(entry)) {
        if (entry.length !== 2 || typeof entry[1] !== "string") {
          throw new Error(
            `Invalid disaggregated_components entry: ${JSON.stringify(entry)}`,
          );
        }
        const [disagComponent, mappedId] = entry as readonly [
          ComponentBase,
          string,
        ];
        convertedConfig.push([disagComponent, mappedId]);
        componentsIdMapping.set(disagComponent.id, mappedId);
      } else {
        const disagComponent = entry as ComponentBase;
        convertedConfig.push([disagComponent, disagComponent.id]);
        componentsIdMapping.set(disagComponent.id, disagComponent.id);
      }
    }

    // Serialize disaggregated components separately
    const disaggregatedDict: Record<string, SerializedDict> = {};
    for (const [disag, mappedId] of convertedConfig) {
      if (disag === component) {
        throw new Error("Cannot disaggregate the root component");
      }
      const disagCtx = new SerializationContext(this.plugins, {
        targetVersion: opts.agentspecVersion,
        camelCase: useCamelCase,
        includeSensitiveFields: includeSensitive,
      });
      const dump = disagCtx.saveToDict(disag, opts.agentspecVersion);
      disaggregatedDict[mappedId] = dump;
    }

    // Serialize the main component
    const resolvedComponents = new Map<string, SerializedDict>();
    for (const [id, dump] of Object.entries(disaggregatedDict)) {
      resolvedComponents.set(id, dump);
    }
    const mainCtx = new SerializationContext(this.plugins, {
      targetVersion: opts.agentspecVersion,
      resolvedComponents,
      componentsIdMapping,
      camelCase: useCamelCase,
      includeSensitiveFields: includeSensitive,
    });
    const mainDump = mainCtx.saveToDict(component, opts.agentspecVersion);

    if (!exportDisag) {
      return mainDump;
    }

    return [
      mainDump,
      { $referenced_components: disaggregatedDict },
    ];
  }

  /** Serialize a component to a JSON string */
  toJson(
    component: ComponentBase,
    options?: {
      agentspecVersion?: AgentSpecVersion;
      disaggregatedComponents?: DisaggregatedComponentsConfig;
      exportDisaggregatedComponents?: boolean;
      indent?: number;
      camelCase?: boolean;
      includeSensitiveFields?: boolean;
    },
  ): string | [string, string] {
    const indent = options?.indent ?? 2;
    const result = this._toDict(component, options);

    if (Array.isArray(result)) {
      return [
        JSON.stringify(result[0], null, indent),
        JSON.stringify(result[1], null, indent),
      ];
    }
    return JSON.stringify(result, null, indent);
  }

  /** Serialize a component to a YAML string */
  toYaml(
    component: ComponentBase,
    options?: {
      agentspecVersion?: AgentSpecVersion;
      disaggregatedComponents?: DisaggregatedComponentsConfig;
      exportDisaggregatedComponents?: boolean;
      camelCase?: boolean;
      includeSensitiveFields?: boolean;
    },
  ): string | [string, string] {
    const result = this._toDict(component, options);

    if (Array.isArray(result)) {
      return [
        YAML.stringify(result[0], { sortMapEntries: false }),
        YAML.stringify(result[1], { sortMapEntries: false }),
      ];
    }
    return YAML.stringify(result, { sortMapEntries: false });
  }
}
