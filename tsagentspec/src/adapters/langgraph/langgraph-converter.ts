/**
 * AgentSpec -> LangGraph converter.
 *
 * Port of `pyagentspec.adapters.langgraph._langgraphconverter
 * .AgentSpecToLangGraphConverter`: dispatches on the component type and
 * assembles langchain `createAgent` react agents, `@langchain/langgraph-swarm`
 * swarms, hierarchical ManagerWorkers graphs and Flow StateGraphs.
 *
 * Runtime contracts (state keys, node names, error-message text) mirror the
 * Python adapter exactly so specs behave the same across both SDKs.
 *
 * Divergences from Python (see the adapter README):
 * - Conversion is async end to end (dynamic imports, MCP loading).
 * - An Agent converts to a langchain `ReactAgent` instance rather than a bare
 *   compiled graph: it is directly invocable, and its `options` property is
 *   the sanctioned source for the exporter. Call sites needing the compiled
 *   graph (swarm assembly, the ManagerWorkers `__manager__` node) unwrap
 *   `.graph`.
 * - Declared agent inputs extend the react-agent state through a zod object
 *   schema (`Annotation.Root` is silently ignored by the JS `createAgent`);
 *   the langchain JS agent state has no `remaining_steps` channel, so no such
 *   key is added.
 * - `patchWithExecutionSpan` wraps the compiled graph through a Proxy at the
 *   same graph-compilation sites as Python (which monkey-patches
 *   stream/astream in place); LLM/tool tracing callbacks are attached where
 *   Python attaches them (see `tracing.ts` for the divergences).
 * - Python's "async interrupts on Python < 3.11" load-time warning has no JS
 *   equivalent and is not ported.
 */
import { ToolMessage } from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import type { BaseCheckpointSaver } from "@langchain/langgraph";
import { ToolInvocationError, createAgent, toolStrategy } from "langchain";
import { z } from "zod";
import type { Agent, ManagerWorkers, Swarm } from "../../agents/index.js";
import { HandoffMode } from "../../agents/index.js";
import { type ComponentBase, isComponent } from "../../component.js";
import type { AgentNode, Flow, Node } from "../../flows/index.js";
import type { LlmConfig } from "../../llms/index.js";
import type { ClientTransport, MCPTool } from "../../mcp/index.js";
import type { Property } from "../../property.js";
import type { MCPToolBox, Tool } from "../../tools/index.js";
import {
  CLIENT_TRANSPORT_TYPES,
  LLM_CONFIG_TYPES,
  NODE_TYPES,
  buildJsonSchemaFromProperties,
  importOptionalPeer,
  isRecordLike,
} from "../common/index.js";
import { isCompiledGraphLike } from "./graph-introspection.js";
import { compileFlow } from "./langgraph-converter-flow.js";
import { convertLlmConfig as convertLlmConfigToChatModel } from "./llm.js";
import {
  ManagerWorkersNodeExecutor,
  compileManagerWorkers,
} from "./manager-workers.js";
import { convertClientTransport, convertMcpTool, convertMcpToolbox } from "./mcp.js";
import type { NodeExecutor } from "./node-execution.js";
import {
  AgentNodeExecutor,
  ApiNodeExecutor,
  BranchingNodeExecutor,
  CatchExceptionNodeExecutor,
  EndNodeExecutor,
  FlowNodeExecutor,
  InputMessageNodeExecutor,
  LlmNodeExecutor,
  MapNodeExecutor,
  OutputMessageNodeExecutor,
  StartNodeExecutor,
  ToolNodeExecutor,
} from "./node-execution.js";
import {
  convertClientTool,
  convertRemoteTool,
  convertServerTool,
  ensureCheckpointerAndValidToolConfig,
} from "./tools.js";
import { patchWithExecutionSpan } from "./tracing.js";
import type {
  ConvertOptions,
  InvocableGraph,
  ToolRegistry,
} from "./types.js";

/** Conversion parameters threaded through every recursive conversion. */
interface ConversionContext {
  toolRegistry: ToolRegistry;
  convertedComponents: Map<string, unknown>;
  checkpointer?: BaseCheckpointSaver;
  config: RunnableConfig;
  middleware: unknown[];
}

/**
 * Unwrap a langchain `ReactAgent` to its compiled graph; compiled graphs (and
 * anything else) pass through unchanged.
 */
function resolveCompiledGraph(agentOrGraph: unknown): unknown {
  if (isCompiledGraphLike(agentOrGraph)) {
    return agentOrGraph;
  }
  if (
    isRecordLike(agentOrGraph) &&
    isCompiledGraphLike(agentOrGraph["graph"])
  ) {
    return agentOrGraph["graph"];
  }
  return agentOrGraph;
}

/**
 * Assert that a converted subflow is an invocable compiled graph. One home
 * for the validation the subflow executor constructors rely on; each call
 * site keeps its exact (Python-parity) error text.
 */
function assertInvocableGraph(
  value: unknown,
  errorMessage: string,
): asserts value is InvocableGraph {
  if (!isCompiledGraphLike(value)) {
    throw new Error(errorMessage);
  }
}

/**
 * Python-parity tool error handling for react-agent tool nodes.
 *
 * LangChain JS's default ToolNode handler converts EVERY tool error into a
 * ToolMessage fed back to the model, while langgraph Python's default (which
 * the Python adapter relies on) only does so for tool-input validation errors
 * and re-raises everything else. The Agent Spec runtime contracts depend on
 * the Python semantics: a rejected `requiresConfirmation` interrupt must
 * raise `Tool '<name>' was denied by the user (reason: ...).` out of
 * `invoke`, and malformed confirmation resume payloads must raise their
 * validation error. Returning `undefined` from a custom handler makes the
 * ToolNode re-throw the error; GraphInterrupts are always re-thrown before
 * the handler applies, so client-tool/confirmation interrupts still work.
 */
function pythonParityToolErrorHandler(
  error: unknown,
  toolCall: { id?: string; name: string },
): ToolMessage | undefined {
  if (ToolInvocationError.isInstance(error)) {
    return new ToolMessage({
      content: error.message,
      tool_call_id: toolCall.id ?? "",
      name: toolCall.name,
    });
  }
  return undefined;
}

/**
 * Install the Python-parity tool error handler on a react agent's `tools`
 * node. `createAgent` exposes no tool-error-handling option, so the compiled
 * graph's ToolNode (reachable at `graph.builder.nodes["tools"].runnable`, a
 * probed-stable surface the exporter also relies on) is patched in place.
 * Agents without tools have no `tools` node and are left untouched.
 */
function applyPythonToolErrorSemantics(reactAgent: unknown): void {
  const graph = (
    reactAgent as {
      graph?: {
        builder?: { nodes?: Record<string, { runnable?: unknown } | undefined> };
      };
    }
  ).graph;
  const runnable = graph?.builder?.nodes?.["tools"]?.runnable as
    | { handleToolErrors?: unknown }
    | undefined;
  if (runnable !== undefined && "handleToolErrors" in runnable) {
    runnable.handleToolErrors = pythonParityToolErrorHandler;
  }
}

/**
 * Convert Agent Spec components into LangGraph runtime components.
 *
 * `convert` memoizes by component id in the per-call `convertedComponents`
 * map, which doubles as a seam for pre-seeding already-converted fakes in
 * tests. `convertLlmConfig` is `protected` so tests can substitute fake chat
 * models by subclassing.
 */
export class AgentSpecToLangGraphConverter {
  /**
   * Convert the given Agent Spec component into the corresponding LangGraph
   * component.
   *
   * When no `config` is given, a `{configurable: {thread_id}}` config with a
   * random thread id is defaulted if a checkpointer is present (else an empty
   * config), mirroring Python.
   */
  async convert(
    agentspecComponent: ComponentBase,
    toolRegistry: ToolRegistry,
    options?: ConvertOptions,
  ): Promise<unknown> {
    const checkpointer = options?.checkpointer;
    let config = options?.config;
    if (config === undefined) {
      config =
        checkpointer !== undefined
          ? { configurable: { thread_id: crypto.randomUUID() } }
          : {};
    }
    const context: ConversionContext = {
      toolRegistry,
      convertedComponents: options?.convertedComponents ?? new Map<string, unknown>(),
      checkpointer,
      config,
      middleware: [...(options?.middleware ?? [])],
    };
    return this.convertWithContext(agentspecComponent, context);
  }

  /** Memoized conversion entry used for every nested component. */
  protected async convertWithContext(
    agentspecComponent: ComponentBase,
    context: ConversionContext,
  ): Promise<unknown> {
    if (!context.convertedComponents.has(agentspecComponent.id)) {
      context.convertedComponents.set(
        agentspecComponent.id,
        await this.convertComponent(agentspecComponent, context),
      );
    }
    return context.convertedComponents.get(agentspecComponent.id);
  }

  /** Dispatch a single (uncached) component conversion by component type. */
  protected async convertComponent(
    agentspecComponent: ComponentBase,
    context: ConversionContext,
  ): Promise<unknown> {
    if (!isComponent(agentspecComponent)) {
      throw new Error(
        "Expected object of type 'pyagentspec.component.Component'," +
          ` but got ${typeof agentspecComponent} instead`,
      );
    }
    const componentType = agentspecComponent.componentType;
    switch (componentType) {
      case "Agent":
        return this.createReactAgent(agentspecComponent as Agent, context);
      case "Swarm":
        return this.convertSwarm(agentspecComponent as Swarm, context);
      case "ManagerWorkers":
        return this.compileManagerWorkersGraph(
          agentspecComponent as ManagerWorkers,
          context,
        );
      case "MCPTool": {
        const mcpTool = agentspecComponent as MCPTool;
        ensureCheckpointerAndValidToolConfig(mcpTool, context.checkpointer);
        const connection = await this.convertWithContext(
          mcpTool.clientTransport,
          context,
        );
        return convertMcpTool(
          mcpTool,
          context.toolRegistry,
          connection as Parameters<typeof convertMcpTool>[2],
        );
      }
      case "MCPToolBox": {
        const mcpToolbox = agentspecComponent as MCPToolBox;
        const connection = await this.convertWithContext(
          mcpToolbox.clientTransport,
          context,
        );
        return convertMcpToolbox(
          mcpToolbox,
          context.toolRegistry,
          connection as Parameters<typeof convertMcpToolbox>[2],
        );
      }
      case "ServerTool": {
        const serverTool = agentspecComponent as Extract<Tool, { componentType: "ServerTool" }>;
        ensureCheckpointerAndValidToolConfig(serverTool, context.checkpointer);
        return convertServerTool(serverTool, context.toolRegistry);
      }
      case "ClientTool": {
        const clientTool = agentspecComponent as Extract<Tool, { componentType: "ClientTool" }>;
        ensureCheckpointerAndValidToolConfig(clientTool, context.checkpointer);
        return convertClientTool(clientTool);
      }
      case "RemoteTool": {
        const remoteTool = agentspecComponent as Extract<Tool, { componentType: "RemoteTool" }>;
        ensureCheckpointerAndValidToolConfig(remoteTool, context.checkpointer);
        return convertRemoteTool(remoteTool);
      }
      case "Flow":
        return this.convertFlow(agentspecComponent as Flow, context);
      default:
        // Membership tests over the SDK-derived component families, so a new
        // union member can never silently miss its dispatch group.
        if (LLM_CONFIG_TYPES.has(componentType)) {
          return this.convertLlmConfig(agentspecComponent as LlmConfig);
        }
        if (CLIENT_TRANSPORT_TYPES.has(componentType)) {
          return convertClientTransport(agentspecComponent as ClientTransport);
        }
        if (NODE_TYPES.has(componentType)) {
          return this.convertNode(
            agentspecComponent as unknown as Node,
            context,
          );
        }
        throw new Error(
          `The Agent Spec type '${componentType}' is not yet supported for conversion.`,
        );
    }
  }

  /**
   * Create the LangChain chat model for an Agent Spec LLM configuration.
   *
   * Protected so tests can subclass the converter and substitute fake chat
   * models; the default implementation delegates to `convertLlmConfig` from
   * `llm.js`.
   */
  protected async convertLlmConfig(llmConfig: LlmConfig): Promise<unknown> {
    return convertLlmConfigToChatModel(llmConfig);
  }

  /** Build the zod state schema extending the agent state with declared inputs. */
  private buildAgentStateSchema(inputs: Property[]): z.ZodTypeAny {
    // The JS createAgent has no runtime-validated required/optional split and
    // silently ignores Annotation.Root schemas; a zod object with optional
    // keys adds the channels so declared inputs round-trip through invoke().
    const shape: Record<string, z.ZodTypeAny> = {};
    for (const property of inputs) {
      shape[property.title] = z.unknown().optional();
    }
    return z.object(shape);
  }

  /**
   * Assemble a langchain react agent from an Agent Spec Agent, mirroring
   * Python's `_create_react_agent_with_given_info`: converted model and
   * tools, tool-strategy structured output for declared outputs (with the
   * structured-output sentence appended to the system prompt), extended state
   * for declared inputs, middleware forwarded only when non-empty.
   *
   * `overrides` carries the per-call-site signal: a replacement system prompt
   * (a flow step's rendered template, a ManagerWorkers roster), extra
   * LangGraph-native tools prepended to the converted ones (swarm handoffs,
   * delegation tools), and `dropDeclaredInputs` for prompts that already have
   * the declared inputs baked in.
   */
  protected async createReactAgent(
    agent: Agent,
    context: ConversionContext,
    overrides?: {
      systemPrompt?: string;
      extraLangGraphTools?: unknown[];
      dropDeclaredInputs?: boolean;
    },
  ): Promise<unknown> {
    const model = await this.convertWithContext(agent.llmConfig, context);
    const langgraphTools: unknown[] = [...(overrides?.extraLangGraphTools ?? [])];
    for (const agentspecTool of agent.tools ?? []) {
      langgraphTools.push(await this.convertWithContext(agentspecTool, context));
    }
    for (const toolbox of agent.toolboxes ?? []) {
      const toolboxTools = (await this.convertWithContext(
        toolbox,
        context,
      )) as unknown[];
      langgraphTools.push(...toolboxTools);
    }

    const inputs = overrides?.dropDeclaredInputs ? [] : (agent.inputs ?? []);
    const outputs = agent.outputs ?? [];
    let systemPrompt = overrides?.systemPrompt ?? agent.systemPrompt;
    let responseFormat: unknown;
    if (outputs.length > 0) {
      // Explicitly use the tool strategy instead of letting LangChain select
      // a provider strategy: OpenAI-compatible models do not necessarily
      // support provider-native structured output.
      responseFormat = toolStrategy(
        buildJsonSchemaFromProperties("AgentOutputModel", outputs) as {
          type: "object";
          [key: string]: unknown;
        },
      );
      systemPrompt =
        `${systemPrompt}\n\n` +
        "After using the available tools, provide the final result by calling the " +
        "structured output tool. Do not respond with a plain-text final answer.";
    }

    const createAgentParams: Record<string, unknown> = {
      name: agent.name,
      model,
      tools: langgraphTools,
      systemPrompt,
    };
    if (context.checkpointer !== undefined) {
      createAgentParams["checkpointer"] = context.checkpointer;
    }
    if (responseFormat !== undefined) {
      createAgentParams["responseFormat"] = responseFormat;
    }
    if (inputs.length > 0) {
      createAgentParams["stateSchema"] = this.buildAgentStateSchema(inputs);
    }
    if (context.middleware.length > 0) {
      createAgentParams["middleware"] = context.middleware;
    }
    const reactAgent = createAgent(
      createAgentParams as unknown as Parameters<typeof createAgent>[0],
    );
    applyPythonToolErrorSemantics(reactAgent);
    return patchWithExecutionSpan(reactAgent, {
      kind: "agent",
      component: agent,
    });
  }

  private async convertSwarm(
    swarm: Swarm,
    context: ConversionContext,
  ): Promise<unknown> {
    if (swarm.handoff === HandoffMode.NEVER) {
      // We cannot control what langgraph-swarm does internally in terms of
      // conversation sharing, so NEVER is not really supported.
      throw new Error(
        "Handoff mode NEVER is not supported for conversion in LangGraph adapter",
      );
    }
    // LangGraph distinguishes agents by name; relationships are tuples of
    // (fromAgent, toAgent) and we assume to get only agents in relationships.
    const agentsByName = new Map<string, Record<string, unknown>>();
    for (const relationship of swarm.relationships) {
      for (const participant of relationship) {
        agentsByName.set(String(participant["name"]), participant);
      }
    }
    for (const participant of agentsByName.values()) {
      // Handoff is performed with tools, so only Agents can take part.
      if (participant["componentType"] !== "Agent") {
        throw new Error(
          "Only Agents are supported as part of a Swarm in the LangGraph " +
            `adapter, received ${String(participant["componentType"])} instead.`,
        );
      }
      // Convert the agents even though the converted graphs are re-created
      // below with handoff tools, so they land in the converted-components
      // cache in case they are used in other places.
      await this.convertWithContext(
        participant as unknown as ComponentBase,
        context,
      );
    }
    const handoffs = new Map<string, string[]>();
    for (const agentName of agentsByName.keys()) {
      handoffs.set(agentName, []);
    }
    for (const [fromAgent, toAgent] of swarm.relationships) {
      handoffs.get(String(fromAgent["name"]))?.push(String(toAgent["name"]));
    }

    const swarmModule = await importOptionalPeer(
      () => import("@langchain/langgraph-swarm"),
      "@langchain/langgraph-swarm",
      "convert Swarm components",
      "remove Swarms from the spec.",
    );
    // Re-create the agents with the additional handoff tools.
    const langgraphAgents: unknown[] = [];
    for (const participant of agentsByName.values()) {
      const agent = participant as unknown as Agent;
      const reactAgent = await this.createReactAgent(agent, context, {
        extraLangGraphTools: (handoffs.get(agent.name) ?? []).map(
          (toAgentName) =>
            swarmModule.createHandoffTool({ agentName: toAgentName }),
        ),
      });
      langgraphAgents.push(resolveCompiledGraph(reactAgent));
    }
    const workflow = swarmModule.createSwarm({
      agents: langgraphAgents as never,
      defaultActiveAgent: String(swarm.firstAgent["name"]),
    });
    return workflow.compile({
      ...(context.checkpointer !== undefined
        ? { checkpointer: context.checkpointer }
        : {}),
      name: swarm.name,
    });
  }

  /**
   * Compile a ManagerWorkers into its hierarchical graph. When
   * `systemPromptOverride` is set (a ManagerWorkers flow step with rendered
   * inputs), it replaces the group manager's system prompt and the manager's
   * declared inputs are dropped (they are baked into the prompt).
   */
  private async compileManagerWorkersGraph(
    managerWorkers: ManagerWorkers,
    context: ConversionContext,
    systemPromptOverride?: string,
  ): Promise<unknown> {
    return compileManagerWorkers(managerWorkers, {
      checkpointer: context.checkpointer,
      ...(systemPromptOverride !== undefined
        ? { systemPrompt: systemPromptOverride }
        : {}),
      compileManagerAgent: async (rosterSystemPrompt, delegationTools) => {
        // compileManagerWorkers already validated the group manager type.
        const managerAgent = managerWorkers.groupManager as unknown as Agent;
        const reactAgent = await this.createReactAgent(managerAgent, context, {
          systemPrompt: rosterSystemPrompt,
          extraLangGraphTools: delegationTools,
          // A prompt override means the declared inputs are already baked
          // into the rendered prompt.
          dropDeclaredInputs: systemPromptOverride !== undefined,
        });
        return resolveCompiledGraph(reactAgent);
      },
      convertWorker: (worker) =>
        this.convertWithContext(worker as unknown as ComponentBase, context),
    });
  }

  /**
   * Compile a Flow into a LangGraph StateGraph (see
   * `langgraph-converter-flow.ts`); node executors are built through the
   * memoized recursive conversion.
   */
  private async convertFlow(
    flow: Flow,
    context: ConversionContext,
  ): Promise<unknown> {
    return compileFlow(flow, {
      convertNode: async (node: Node) =>
        (await this.convertWithContext(
          node as unknown as ComponentBase,
          context,
        )) as NodeExecutor,
      checkpointer: context.checkpointer,
    });
  }

  /** Build the node executor for one flow node. */
  protected async convertNode(
    node: Node,
    context: ConversionContext,
  ): Promise<NodeExecutor> {
    switch (node.componentType) {
      case "StartNode":
        return new StartNodeExecutor(node);
      case "EndNode":
        return new EndNodeExecutor(node);
      case "ToolNode": {
        const convertedTool = await this.convertWithContext(node.tool, context);
        return new ToolNodeExecutor(node, convertedTool);
      }
      case "LlmNode": {
        const chatModel = await this.convertWithContext(node.llmConfig, context);
        return new LlmNodeExecutor(node, chatModel);
      }
      case "AgentNode":
        return this.convertAgentNode(node, context);
      case "BranchingNode":
        return new BranchingNodeExecutor(node);
      case "ApiNode":
        return new ApiNodeExecutor(node);
      case "FlowNode": {
        const subflow = await this.convertWithContext(
          node.subflow as unknown as ComponentBase,
          context,
        );
        assertInvocableGraph(
          subflow,
          "FlowNodeExecutor can only initialize FlowNode",
        );
        return new FlowNodeExecutor(node, subflow, context.config);
      }
      case "CatchExceptionNode": {
        const subflow = await this.convertWithContext(
          node.subflow as unknown as ComponentBase,
          context,
        );
        assertInvocableGraph(
          subflow,
          "Internal error: CatchExceptionNodeExecutor expects `subflow` " +
            `to be a CompiledStateGraph, was ${typeof subflow}`,
        );
        return new CatchExceptionNodeExecutor(node, subflow, context.config);
      }
      case "InputMessageNode":
        return new InputMessageNodeExecutor(node);
      case "OutputMessageNode":
        return new OutputMessageNodeExecutor(node);
      case "MapNode": {
        const subflow = await this.convertWithContext(
          node.subflow as unknown as ComponentBase,
          context,
        );
        assertInvocableGraph(
          subflow,
          "MapNodeExecutor can only be initialized with MapNode",
        );
        return new MapNodeExecutor(node, subflow);
      }
      default:
        throw new Error(
          `The AgentSpec component of type ${node.componentType} is not yet supported for conversion`,
        );
    }
  }

  /**
   * Build the executor for an AgentNode: a `ManagerWorkersNodeExecutor` when
   * the node's agent is a ManagerWorkers, else an `AgentNodeExecutor`. The
   * executor receives a compile factory taking the rendered system prompt
   * (executors render templates against node inputs and cache per rendered
   * prompt).
   */
  private convertAgentNode(
    node: AgentNode,
    context: ConversionContext,
  ): NodeExecutor {
    const agentComponent = node.agent;
    if (agentComponent.componentType === "ManagerWorkers") {
      return new ManagerWorkersNodeExecutor(
        node,
        (renderedSystemPrompt: string) =>
          this.compileManagerWorkersGraph(
            agentComponent,
            context,
            renderedSystemPrompt,
          ),
        context.config,
      );
    }
    // The executor's (Python-faithful) guard rejects anything but a plain
    // Agent before the factory ever runs, so the factory can assume one.
    const compileAgentFactory = (
      renderedSystemPrompt: string,
    ): Promise<unknown> =>
      this.createReactAgent(agentComponent as Agent, context, {
        systemPrompt: renderedSystemPrompt,
      });
    return new AgentNodeExecutor(node, compileAgentFactory, context.config);
  }
}
