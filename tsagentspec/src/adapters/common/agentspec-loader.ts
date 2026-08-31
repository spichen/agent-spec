/**
 * Framework-agnostic loader for Agent Spec configurations. Port of
 * `pyagentspec.adapters._agentspecloader.AdapterAgnosticAgentSpecLoader`.
 *
 * This base class centralizes plugin-aware deserialization, the component
 * load policy, and support for disaggregated components
 * (`importOnlyReferencedComponents`). Subclasses supply the two converters.
 *
 * Divergence from Python: the TS `AgentSpecDeserializer` has no
 * allowed/blocked component parameters, so the policy is enforced on the
 * deserialized component tree (`validateComponentTree`) before conversion,
 * not at parse time.
 */
import type { ComponentBase } from "../../component.js";
import { AgentSpecDeserializer } from "../../serialization/index.js";
import type {
  ComponentDeserializationPlugin,
  ComponentsRegistry,
} from "../../serialization/index.js";
import {
  ComponentLoadPolicy,
  type ComponentPolicyInput,
} from "./component-policy.js";
import type {
  AgentSpecToRuntimeConverter,
  RuntimeToAgentSpecConverter,
} from "./converters.js";

/** Components blocked by default: stdio MCP transports run local processes. */
const DEFAULT_BLOCKED_COMPONENTS: readonly string[] = ["StdioTransport"];

/** Constructor options for the adapter-agnostic loader base. */
export interface AdapterAgnosticAgentSpecLoaderOptions {
  /** Registry mapping tool names to runtime implementations. */
  toolRegistry?: Record<string, unknown>;
  /** Deserialization plugins; builtins are used when omitted. */
  plugins?: ComponentDeserializationPlugin[];
  /**
   * Component type names allowed to load. When omitted, all component types
   * are allowed unless blocked.
   */
  allowedComponents?: ComponentPolicyInput;
  /**
   * Component type names blocked from loading. When omitted, `StdioTransport`
   * is blocked by default.
   */
  blockedComponents?: ComponentPolicyInput;
}

/** Per-call options for the load methods. */
export interface LoadOptions {
  /**
   * Registry mapping ids to runtime components/values. Entries are converted
   * back to Agent Spec components to resolve references during
   * deserialization; if a conversion fails, the given value is used as-is.
   */
  componentsRegistry?: Record<string, unknown>;
  /**
   * When true, load only the referenced/disaggregated components and return a
   * dictionary mapping component id to runtime components/values. These can
   * be used as the `componentsRegistry` when loading the main configuration.
   */
  importOnlyReferencedComponents?: boolean;
}

/** Convert serialized Agent Spec into adapter runtime components. */
export abstract class AdapterAgnosticAgentSpecLoader {
  /** Registry mapping tool names to runtime implementations. */
  readonly toolRegistry: Record<string, unknown>;
  /** Deserialization plugins passed to the deserializer (builtins if unset). */
  readonly plugins?: ComponentDeserializationPlugin[];
  /** The allow/block policy applied to every loaded component tree. */
  readonly componentLoadPolicy: ComponentLoadPolicy;
  /** Normalized allow-list entries, or undefined when no allow list is set. */
  readonly allowedComponents?: readonly string[];
  /** Normalized block-list entries. */
  readonly blockedComponents: readonly string[];

  constructor(options?: AdapterAgnosticAgentSpecLoaderOptions) {
    const opts = options ?? {};
    this.plugins = opts.plugins;
    this.toolRegistry = opts.toolRegistry ?? {};
    this.componentLoadPolicy = new ComponentLoadPolicy(
      opts.allowedComponents,
      opts.blockedComponents ?? DEFAULT_BLOCKED_COMPONENTS,
    );
    this.allowedComponents = this.componentLoadPolicy.allowedComponents;
    this.blockedComponents = this.componentLoadPolicy.blockedComponents;
  }

  /** Converter used to convert Agent Spec components to runtime components. */
  abstract get agentspecToRuntimeConverter(): AgentSpecToRuntimeConverter;

  /** Converter used to convert runtime components to Agent Spec components. */
  abstract get runtimeToAgentSpecConverter(): RuntimeToAgentSpecConverter;

  /**
   * Transform the given Agent Spec YAML into runtime components, with support
   * for disaggregated configurations.
   */
  async loadYaml(
    serializedAssistant: string,
    options?: LoadOptions,
  ): Promise<unknown> {
    return this._load("yaml", serializedAssistant, options);
  }

  /**
   * Transform the given Agent Spec JSON into runtime components, with support
   * for disaggregated configurations.
   */
  async loadJson(
    serializedAssistant: string,
    options?: LoadOptions,
  ): Promise<unknown> {
    return this._load("json", serializedAssistant, options);
  }

  /**
   * Transform the given Agent Spec dictionary into runtime components, with
   * support for disaggregated configurations.
   */
  async loadDict(
    serializedAssistant: Record<string, unknown>,
    options?: LoadOptions,
  ): Promise<unknown> {
    return this._load("dict", serializedAssistant, options);
  }

  /**
   * Convert an Agent Spec component into a runtime component after validating
   * it against the component load policy.
   *
   * Subclasses may override this method to pass adapter-specific parameters
   * into their converter (e.g., checkpointers).
   */
  async loadComponent(agentspecComponent: ComponentBase): Promise<unknown> {
    this.componentLoadPolicy.validateComponentTree(agentspecComponent);
    return this.agentspecToRuntimeConverter.convert(
      agentspecComponent,
      this.toolRegistry,
    );
  }

  /**
   * Convert a runtime components registry into an Agent Spec registry so that
   * references can be resolved during deserialization. Values that fail to
   * convert are kept as-is with a warning (mirrors Python).
   */
  protected _convertComponentRegistry(
    runtimeComponentRegistry: Record<string, unknown>,
  ): ComponentsRegistry {
    const converter = this.runtimeToAgentSpecConverter;
    const convertedRegistry: ComponentsRegistry = new Map();
    for (const [customId, runtimeComponentOrValue] of Object.entries(
      runtimeComponentRegistry,
    )) {
      try {
        convertedRegistry.set(
          customId,
          converter.convert(runtimeComponentOrValue),
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.warn(
          `Failed to convert runtime component ${customId} with exception \`${message}\`. Fallback to given value.`,
        );
        // Mirrors Python: unconvertible registry values are used as-is when
        // resolving references.
        convertedRegistry.set(customId, runtimeComponentOrValue as ComponentBase);
      }
    }
    return convertedRegistry;
  }

  /** Common implementation of the load methods. */
  protected async _load(
    loader: "yaml" | "json" | "dict",
    serializedAssistant: string | Record<string, unknown>,
    options?: LoadOptions,
  ): Promise<unknown> {
    const deserializer = new AgentSpecDeserializer(this.plugins);
    let deserialize: (deserializeOptions: {
      componentsRegistry?: ComponentsRegistry;
      importOnlyReferencedComponents?: boolean;
    }) => ComponentBase | Record<string, ComponentBase>;
    if (loader === "yaml") {
      deserialize = (deserializeOptions) =>
        deserializer.fromYaml(serializedAssistant as string, deserializeOptions);
    } else if (loader === "json") {
      deserialize = (deserializeOptions) =>
        deserializer.fromJson(serializedAssistant as string, deserializeOptions);
    } else if (loader === "dict") {
      // The TS AgentSpecDeserializer has no public dict entry point;
      // round-trip through JSON.
      const json = JSON.stringify(serializedAssistant);
      deserialize = (deserializeOptions) =>
        deserializer.fromJson(json, deserializeOptions);
    } else {
      throw new Error(
        `Unsupported loader type: \`${String(loader)}\`. Expected \`dict\`, \`json\`, or \`yaml\`.`,
      );
    }

    const convertedRegistry =
      options?.componentsRegistry !== undefined
        ? this._convertComponentRegistry(options.componentsRegistry)
        : undefined;

    if (options?.importOnlyReferencedComponents) {
      // Loading the disaggregated components
      const referencedComponentsDict = deserialize({
        componentsRegistry: convertedRegistry,
        importOnlyReferencedComponents: true,
      }) as Record<string, ComponentBase>;
      const runtimeComponents: Record<string, unknown> = {};
      for (const [componentId, agentspecComponent] of Object.entries(
        referencedComponentsDict,
      )) {
        runtimeComponents[componentId] =
          await this.loadComponent(agentspecComponent);
      }
      return runtimeComponents;
    }

    const agentspecComponent = deserialize({
      componentsRegistry: convertedRegistry,
      importOnlyReferencedComponents: false,
    }) as ComponentBase;
    return this.loadComponent(agentspecComponent);
  }
}
