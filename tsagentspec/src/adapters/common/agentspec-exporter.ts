/**
 * Framework-agnostic exporter converting runtime objects to Agent Spec
 * configurations. Port of
 * `pyagentspec.adapters._agentspecexporter.AdapterAgnosticAgentSpecExporter`.
 */
import type { ComponentBase } from "../../component.js";
import { AgentSpecSerializer } from "../../serialization/index.js";
import type {
  ComponentSerializationPlugin,
  DisaggregatedComponentsConfig,
} from "../../serialization/index.js";
import type { AgentSpecVersion } from "../../versioning.js";
import type { RuntimeToAgentSpecConverter } from "./converters.js";

/**
 * Runtime components/fields to disaggregate upon serialization. Each item can
 * be a runtime component (disaggregated using its converted component id) or
 * a `[runtimeComponent, customId]` pair (disaggregated using the custom id).
 */
export type RuntimeDisaggregatedComponentsConfig = ReadonlyArray<
  unknown | readonly [unknown, string]
>;

/** Options for the export methods. */
export interface ExportOptions {
  /** The Agent Spec version to serialize the component at. */
  agentspecVersion?: AgentSpecVersion;
  /**
   * Components/fields to disaggregate upon serialization. Components listed
   * here are disaggregated even if `exportDisaggregatedComponents` is false.
   */
  disaggregatedComponents?: RuntimeDisaggregatedComponentsConfig;
  /** Whether to export the disaggregated components. Defaults to false. */
  exportDisaggregatedComponents?: boolean;
}

/** Serialized dictionary form of a component. */
export type ExportedDict = Record<string, unknown>;

/** Helper class to convert runtime objects to Agent Spec configurations. */
export abstract class AdapterAgnosticAgentSpecExporter {
  /** Serialization plugins passed to the serializer. */
  readonly plugins: ComponentSerializationPlugin[];

  constructor(plugins?: ComponentSerializationPlugin[]) {
    this.plugins = plugins ?? [];
  }

  /** Converter used to convert runtime components to Agent Spec components. */
  abstract get runtimeToAgentSpecConverter(): RuntimeToAgentSpecConverter;

  /**
   * Transform the given runtime component into the respective Agent Spec JSON
   * representation. Returns `[main, referenced]` JSON strings when
   * `exportDisaggregatedComponents` is true.
   */
  toJson(
    runtimeComponent: unknown,
    options?: ExportOptions,
  ): string | [string, string] {
    return this._export(
      (serializer, agentspecAssistant, serializerOptions) =>
        serializer.toJson(agentspecAssistant, serializerOptions),
      runtimeComponent,
      options,
    );
  }

  /**
   * Transform the given runtime component into the respective Agent Spec YAML
   * representation. Returns `[main, referenced]` YAML strings when
   * `exportDisaggregatedComponents` is true.
   */
  toYaml(
    runtimeComponent: unknown,
    options?: ExportOptions,
  ): string | [string, string] {
    return this._export(
      (serializer, agentspecAssistant, serializerOptions) =>
        serializer.toYaml(agentspecAssistant, serializerOptions),
      runtimeComponent,
      options,
    );
  }

  /**
   * Transform the given runtime component into the respective Agent Spec
   * dictionary. Returns `[main, referenced]` dictionaries when
   * `exportDisaggregatedComponents` is true.
   */
  toDict(
    runtimeComponent: unknown,
    options?: ExportOptions,
  ): ExportedDict | [ExportedDict, ExportedDict] {
    return this._export(
      (
        serializer,
        agentspecAssistant,
        serializerOptions,
      ): ExportedDict | [ExportedDict, ExportedDict] => {
        // The TS AgentSpecSerializer has no public toDict, so the dictionary
        // form is derived from the JSON serialization.
        const json = serializer.toJson(agentspecAssistant, serializerOptions);
        if (Array.isArray(json)) {
          return [
            JSON.parse(json[0]) as ExportedDict,
            JSON.parse(json[1]) as ExportedDict,
          ];
        }
        return JSON.parse(json) as ExportedDict;
      },
      runtimeComponent,
      options,
    );
  }

  /**
   * Transform the given runtime component into the respective AgentSpec
   * component.
   */
  toComponent(runtimeComponent: unknown): ComponentBase {
    return this.runtimeToAgentSpecConverter.convert(runtimeComponent);
  }

  /**
   * Common implementation of the export methods. Each public method passes
   * the closure that serializes the converted component to its output form.
   */
  protected _export<SerializedT>(
    serialize: (
      serializer: AgentSpecSerializer,
      agentspecAssistant: ComponentBase,
      serializerOptions: {
        agentspecVersion?: AgentSpecVersion;
        disaggregatedComponents?: DisaggregatedComponentsConfig;
        exportDisaggregatedComponents: boolean;
      },
    ) => SerializedT,
    runtimeComponent: unknown,
    options?: ExportOptions,
  ): SerializedT {
    const serializer = new AgentSpecSerializer(this.plugins);

    const [convertedDisagComponents, referencedComponents] =
      options?.disaggregatedComponents !== undefined
        ? this._convertDisaggregatedConfig(options.disaggregatedComponents)
        : [undefined, undefined];
    const agentspecAssistant = this.runtimeToAgentSpecConverter.convert(
      runtimeComponent,
      referencedComponents,
    );
    const serializerOptions = {
      agentspecVersion: options?.agentspecVersion,
      disaggregatedComponents: convertedDisagComponents,
      exportDisaggregatedComponents:
        options?.exportDisaggregatedComponents ?? false,
    };
    return serialize(serializer, agentspecAssistant, serializerOptions);
  }

  /**
   * Convert the runtime disaggregated-components config into Agent Spec
   * components, accumulating the shared referenced-objects registry that is
   * then passed into the root conversion (so disaggregated components share
   * identity with references inside the root component).
   */
  protected _convertDisaggregatedConfig(
    runtimeDisagConfig: RuntimeDisaggregatedComponentsConfig,
  ): [DisaggregatedComponentsConfig, Map<string, ComponentBase>] {
    const agentspecDisaggregatedComponents: Array<
      ComponentBase | readonly [ComponentBase, string]
    > = [];
    const referencedComponents = new Map<string, ComponentBase>();
    for (const disagConfig of runtimeDisagConfig) {
      const pair =
        Array.isArray(disagConfig) &&
        disagConfig.length === 2 &&
        typeof disagConfig[1] === "string"
          ? (disagConfig as unknown as readonly [unknown, string])
          : undefined;
      const runtimeComponent = pair !== undefined ? pair[0] : disagConfig;
      const agentspecComponent = this.runtimeToAgentSpecConverter.convert(
        runtimeComponent,
        referencedComponents,
      );
      // Mirroring Python, the converted component keeps its own id everywhere
      // (root tree and disaggregated registry); a custom id is passed through
      // as a [component, customId] pair and applied by the serializer only as
      // the serialization-time mapping key.
      agentspecDisaggregatedComponents.push(
        pair !== undefined
          ? ([agentspecComponent, pair[1]] as const)
          : agentspecComponent,
      );
    }
    return [agentspecDisaggregatedComponents, referencedComponents];
  }
}
