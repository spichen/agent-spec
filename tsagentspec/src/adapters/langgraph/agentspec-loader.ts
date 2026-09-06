/**
 * Public loader converting Agent Spec configurations into LangGraph objects.
 *
 * Port of `pyagentspec.adapters.langgraph.agentspecloader.AgentSpecLoader` on
 * top of the adapter-agnostic loader base: plugin-aware deserialization, the
 * component load policy (`StdioTransport` blocked by default), disaggregated
 * configurations, and the LangGraph-specific checkpointer / config /
 * middleware options threaded into every conversion.
 */
import type { RunnableConfig } from "@langchain/core/runnables";
import type { BaseCheckpointSaver } from "@langchain/langgraph";
import type { ComponentBase } from "../../component.js";
import {
  AdapterAgnosticAgentSpecLoader,
  type AdapterAgnosticAgentSpecLoaderOptions,
} from "../common/index.js";
import { LangGraphToAgentSpecConverter } from "./agentspec-converter.js";
import { AgentSpecToLangGraphConverter } from "./langgraph-converter.js";
import type { ToolRegistry } from "./types.js";

/** Constructor options for the LangGraph `AgentSpecLoader`. */
export interface AgentSpecLoaderOptions
  extends AdapterAgnosticAgentSpecLoaderOptions {
  /**
   * Tool implementations keyed by tool name: LangChain structured tools or
   * plain (sync or async) functions. Narrows the adapter-agnostic
   * `Record<string, unknown>` to the LangGraph registry contract.
   */
  toolRegistry?: ToolRegistry;
  /**
   * LangGraph checkpointer wired into created graphs; enables features that
   * require one (e.g., client tools and tool confirmation interrupts).
   */
  checkpointer?: BaseCheckpointSaver;
  /** RunnableConfig passed to created runnables/graphs. */
  config?: RunnableConfig;
  /**
   * LangChain agent middleware instances forwarded verbatim to
   * `createAgent({middleware})` when compiling an Agent Spec `Agent` into a
   * react graph. Order is preserved — index 0 is the outermost middleware.
   * When omitted or empty, the middleware option is not passed at all.
   */
  middleware?: unknown[];
}

/**
 * Helper class to convert Agent Spec configurations into LangGraph objects.
 *
 * Loading is async: `loadYaml` / `loadJson` / `loadDict` return promises of
 * the converted runtime component (or, with
 * `importOnlyReferencedComponents: true`, a record mapping component ids to
 * runtime components).
 */
export class AgentSpecLoader extends AdapterAgnosticAgentSpecLoader {
  /** The LangGraph registry contract for the base loader's registry field. */
  declare readonly toolRegistry: ToolRegistry;
  /** Checkpointer wired into created graphs. */
  readonly checkpointer?: BaseCheckpointSaver;
  /** RunnableConfig passed to created runnables/graphs. */
  readonly config?: RunnableConfig;
  private readonly middleware: unknown[];

  constructor(options?: AgentSpecLoaderOptions) {
    super(options);
    this.checkpointer = options?.checkpointer;
    this.config = options?.config;
    this.middleware = [...(options?.middleware ?? [])];
  }

  /** Converter used to convert Agent Spec components to LangGraph components. */
  get agentspecToRuntimeConverter(): AgentSpecToLangGraphConverter {
    return new AgentSpecToLangGraphConverter();
  }

  /** Converter used to convert LangGraph components to Agent Spec components. */
  get runtimeToAgentSpecConverter(): LangGraphToAgentSpecConverter {
    return new LangGraphToAgentSpecConverter();
  }

  /**
   * Convert an Agent Spec component into a LangGraph component after
   * validating it against the component load policy, threading the loader's
   * checkpointer, config and middleware into the conversion.
   */
  override async loadComponent(
    agentspecComponent: ComponentBase,
  ): Promise<unknown> {
    this.componentLoadPolicy.validateComponentTree(agentspecComponent);
    return this.agentspecToRuntimeConverter.convert(
      agentspecComponent,
      this.toolRegistry,
      {
        checkpointer: this.checkpointer,
        config: this.config,
        middleware: this.middleware,
      },
    );
  }
}
