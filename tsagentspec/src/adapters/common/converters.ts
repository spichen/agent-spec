/**
 * Converter interfaces shared by the adapter loaders and exporters. Port of
 * the converter Protocols in `pyagentspec.adapters._agentspecloader`.
 */
import type { ComponentBase } from "../../component.js";

/**
 * Adapter-specific AgentSpec -> runtime converter used by loaders.
 *
 * Conversion is async in the TypeScript adapters (dynamic imports, MCP
 * loading); `options` carries adapter-specific parameters such as
 * checkpointers or conversion caches.
 */
export interface AgentSpecToRuntimeConverter<
  TOptions = Record<string, unknown>,
> {
  convert(
    agentspecComponent: ComponentBase,
    toolRegistry: Record<string, unknown>,
    options?: TOptions,
  ): Promise<unknown>;
}

/**
 * Adapter-specific runtime -> AgentSpec converter used by loaders and
 * exporters. `referencedObjects` memoizes converted components by runtime
 * object reference so shared runtime objects become single referenced
 * components in the output.
 */
export interface RuntimeToAgentSpecConverter {
  convert(
    runtimeComponent: unknown,
    referencedObjects?: Map<string, ComponentBase>,
  ): ComponentBase;
}
