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
 * - No tracing callbacks/spans are attached; `patchWithExecutionSpan` is a
 *   no-op seam invoked at the same sites as Python.
 * - Python's "async interrupts on Python < 3.11" load-time warning has no JS
 *   equivalent and is not ported.
 */
import { ToolMessage, type BaseMessage } from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import type { BaseCheckpointSaver } from "@langchain/langgraph";
import { Annotation, START, StateGraph } from "@langchain/langgraph";
import { ToolInvocationError, createAgent, toolStrategy } from "langchain";
import { z } from "zod";
import type { Agent, ManagerWorkers, Swarm } from "../../agents/index.js";
import { HandoffMode } from "../../agents/index.js";
import { type ComponentBase, isComponent } from "../../component.js";
import type {
  AgentNode,
  ControlFlowEdge,
  DataFlowEdge,
  Flow,
  Node,
} from "../../flows/index.js";
import { DEFAULT_NEXT_BRANCH, createDataFlowEdge } from "../../flows/index.js";
import type { LlmConfig } from "../../llms/index.js";
import type { ClientTransport, MCPTool } from "../../mcp/index.js";
import type { Property } from "../../property.js";
import type { MCPToolBox, Tool, ToolBox } from "../../tools/index.js";
import type { ComponentWithIO } from "../../component.js";
import { buildJsonSchemaFromProperties, jsonSchemasHaveSameType } from "../common/index.js";
import { convertLlmConfig as convertLlmConfigToChatModel } from "./llm.js";
import {
  ManagerWorkersNodeExecutor,
  compileManagerWorkers,
} from "./manager-workers.js";
import { convertClientTransport, convertMcpTool, convertMcpToolbox } from "./mcp.js";
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
  FlowState,
  NextNodeInputs,
  NodeExecutionDetails,
  NodeOutputs,
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

/** Inputs of the react-agent assembly helper. */
interface ReactAgentInfo {
  name: string;
  systemPrompt: string;
  agent: Agent;
  llmConfig: LlmConfig;
  tools: Tool[];
  toolboxes: ToolBox[];
  inputs: Property[];
  outputs: Property[];
  additionalLangGraphTools?: unknown[];
}

/** The structural surface of a flow node executor used by the converter. */
interface NodeExecutorLike {
  attachEdge(edge: DataFlowEdge): void;
  call(state: FlowState, config: RunnableConfig): Promise<Partial<FlowState>>;
}

interface EndNodeExecutorLike extends NodeExecutorLike {
  setFlowOutputs(flowOutputs: Property[]): void;
}

interface MapNodeExecutorLike extends NodeExecutorLike {
  setInputsToIterate(inputsToIterate: string[]): void;
}

/** Loosely-typed StateGraph surface for graphs with dynamic node names. */
interface DynamicStateGraph {
  addNode(key: string, action: unknown): DynamicStateGraph;
  addEdge(start: string, end: string): DynamicStateGraph;
  addConditionalEdges(
    source: string,
    path: (state: FlowState) => string,
    pathMap?: Record<string, string>,
  ): DynamicStateGraph;
  compile(options?: {
    checkpointer?: BaseCheckpointSaver;
    name?: string;
  }): unknown;
}

type LangGraphSwarmModule = typeof import("@langchain/langgraph-swarm");

async function importLangGraphSwarmModule(): Promise<LangGraphSwarmModule> {
  try {
    return await import("@langchain/langgraph-swarm");
  } catch (error) {
    throw new Error(
      "@langchain/langgraph-swarm is required to convert Swarm components. " +
        "Install it (e.g., npm install @langchain/langgraph-swarm) or remove Swarms from the spec.",
      { cause: error },
    );
  }
}

/**
 * Unwrap a langchain `ReactAgent` to its compiled graph; compiled graphs (and
 * anything else) pass through unchanged.
 */
function resolveCompiledGraph(agentOrGraph: unknown): unknown {
  if (typeof agentOrGraph === "object" && agentOrGraph !== null) {
    const candidate = agentOrGraph as {
      lg_is_pregel?: unknown;
      graph?: { lg_is_pregel?: unknown };
    };
    if (candidate.lg_is_pregel === true) {
      return agentOrGraph;
    }
    if (
      typeof candidate.graph === "object" &&
      candidate.graph !== null &&
      candidate.graph.lg_is_pregel === true
    ) {
      return candidate.graph;
    }
  }
  return agentOrGraph;
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

/** Duck-type check for a compiled LangGraph graph. */
function isCompiledGraph(value: unknown): boolean {
  return (
    typeof value === "object" &&
    value !== null &&
    (value as { lg_is_pregel?: unknown }).lg_is_pregel === true
  );
}

/** A last-value channel with an initial default. */
function lastValueChannel<T>(defaultValue: () => T) {
  return Annotation<T>({
    reducer: (_current: T, update: T) => update,
    default: defaultValue,
  });
}

function findPropertyByTitle(
  properties: Property[],
  title: string,
  context: string,
): Property {
  const property = properties.find((candidate) => candidate.title === title);
  if (property === undefined) {
    throw new Error(`Property \`${title}\` was not found in ${context}.`);
  }
  return property;
}

const NODE_COMPONENT_TYPES = new Set([
  "StartNode",
  "EndNode",
  "ToolNode",
  "LlmNode",
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
]);

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
        return this.convertAgent(agentspecComponent as Agent, context);
      case "Swarm":
        return this.convertSwarm(agentspecComponent as Swarm, context);
      case "ManagerWorkers":
        return this.compileManagerWorkersGraph(
          agentspecComponent as ManagerWorkers,
          context,
        );
      case "OpenAiConfig":
      case "OpenAiCompatibleConfig":
      case "VllmConfig":
      case "OllamaConfig":
      case "OciGenAiConfig":
        return this.convertLlmConfig(agentspecComponent as LlmConfig);
      case "StdioTransport":
      case "SSETransport":
      case "SSEmTLSTransport":
      case "StreamableHTTPTransport":
      case "StreamableHTTPmTLSTransport":
      case "RemoteTransport":
        return convertClientTransport(agentspecComponent as ClientTransport);
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
        if (NODE_COMPONENT_TYPES.has(componentType)) {
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
   * Assemble a langchain react agent from Agent Spec information, mirroring
   * Python's `_create_react_agent_with_given_info`: converted model and
   * tools, tool-strategy structured output for declared outputs (with the
   * structured-output sentence appended to the system prompt), extended state
   * for declared inputs, middleware forwarded only when non-empty.
   */
  protected async createReactAgentWithGivenInfo(
    info: ReactAgentInfo,
    context: ConversionContext,
  ): Promise<unknown> {
    const model = await this.convertWithContext(info.llmConfig, context);
    const langgraphTools: unknown[] = [...(info.additionalLangGraphTools ?? [])];
    for (const agentspecTool of info.tools) {
      langgraphTools.push(await this.convertWithContext(agentspecTool, context));
    }
    for (const toolbox of info.toolboxes) {
      const toolboxTools = (await this.convertWithContext(
        toolbox,
        context,
      )) as unknown[];
      langgraphTools.push(...toolboxTools);
    }

    let systemPrompt = info.systemPrompt;
    let responseFormat: unknown;
    if (info.outputs.length > 0) {
      // Explicitly use the tool strategy instead of letting LangChain select
      // a provider strategy: OpenAI-compatible models do not necessarily
      // support provider-native structured output.
      responseFormat = toolStrategy(
        buildJsonSchemaFromProperties("AgentOutputModel", info.outputs) as {
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
      name: info.name,
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
    if (info.inputs.length > 0) {
      createAgentParams["stateSchema"] = this.buildAgentStateSchema(info.inputs);
    }
    if (context.middleware.length > 0) {
      createAgentParams["middleware"] = context.middleware;
    }
    const reactAgent = createAgent(
      createAgentParams as unknown as Parameters<typeof createAgent>[0],
    );
    applyPythonToolErrorSemantics(reactAgent);
    return patchWithExecutionSpan(reactAgent);
  }

  private async convertAgent(
    agent: Agent,
    context: ConversionContext,
  ): Promise<unknown> {
    return this.createReactAgentWithGivenInfo(
      {
        name: agent.name,
        systemPrompt: agent.systemPrompt,
        agent,
        llmConfig: agent.llmConfig,
        tools: agent.tools,
        toolboxes: agent.toolboxes,
        inputs: agent.inputs ?? [],
        outputs: agent.outputs ?? [],
      },
      context,
    );
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

    const swarmModule = await importLangGraphSwarmModule();
    // Re-create the agents with the additional handoff tools.
    const langgraphAgents: unknown[] = [];
    for (const participant of agentsByName.values()) {
      const agent = participant as unknown as Agent;
      const reactAgent = await this.createReactAgentWithGivenInfo(
        {
          name: agent.name,
          systemPrompt: agent.systemPrompt,
          agent,
          llmConfig: agent.llmConfig,
          tools: agent.tools,
          toolboxes: agent.toolboxes,
          inputs: agent.inputs ?? [],
          outputs: agent.outputs ?? [],
          additionalLangGraphTools: (handoffs.get(agent.name) ?? []).map(
            (toAgentName) =>
              swarmModule.createHandoffTool({ agentName: toAgentName }),
          ),
        },
        context,
      );
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
        const reactAgent = await this.createReactAgentWithGivenInfo(
          {
            name: managerAgent.name,
            systemPrompt: rosterSystemPrompt,
            agent: managerAgent,
            llmConfig: managerAgent.llmConfig,
            tools: managerAgent.tools ?? [],
            toolboxes: managerAgent.toolboxes ?? [],
            inputs:
              systemPromptOverride !== undefined
                ? []
                : (managerAgent.inputs ?? []),
            outputs: managerAgent.outputs ?? [],
            additionalLangGraphTools: delegationTools,
          },
          context,
        );
        return resolveCompiledGraph(reactAgent);
      },
      convertWorker: (worker) =>
        this.convertWithContext(worker as unknown as ComponentBase, context),
    });
  }

  private async convertFlow(
    flow: Flow,
    context: ConversionContext,
  ): Promise<unknown> {
    // The input/output schemas must reference the SAME channel instances as
    // the state schema, or StateGraph rejects them as conflicting channels.
    const inputsChannel = lastValueChannel<NextNodeInputs>(() => ({}));
    const outputsChannel = lastValueChannel<NodeOutputs>(() => ({}));
    const messagesChannel = lastValueChannel<BaseMessage[]>(() => []);
    const nodeExecutionDetailsChannel = lastValueChannel<NodeExecutionDetails>(
      () => ({}),
    );
    const graphBuilder = new StateGraph({
      state: Annotation.Root({
        inputs: inputsChannel,
        outputs: outputsChannel,
        messages: messagesChannel,
        node_execution_details: nodeExecutionDetailsChannel,
      }),
      input: Annotation.Root({
        inputs: inputsChannel,
        messages: messagesChannel,
      }),
      output: Annotation.Root({
        outputs: outputsChannel,
        messages: messagesChannel,
        node_execution_details: nodeExecutionDetailsChannel,
      }),
    }) as unknown as DynamicStateGraph;

    graphBuilder.addEdge(START, String(flow.startNode["id"]));

    const flowNodes = flow.nodes as unknown as Node[];
    const nodeExecutors = new Map<string, NodeExecutorLike>();
    for (const node of flowNodes) {
      nodeExecutors.set(
        node.id,
        (await this.convertWithContext(
          node as unknown as ComponentBase,
          context,
        )) as NodeExecutorLike,
      );
    }

    // Tell the MapNodes which inputs they should iterate over, based on the
    // type of the outputs they are connected to; give EndNodes the flow
    // outputs to reshape their result. Mirroring Python, only explicitly
    // declared data-flow connections take part in MapNode iteration wiring.
    for (const node of flowNodes) {
      if (node.componentType === "MapNode") {
        const inputsToIterate: string[] = [];
        for (const dataFlowEdge of flow.dataFlowConnections ?? []) {
          if (String(dataFlowEdge.destinationNode["id"]) !== node.id) {
            continue;
          }
          const sourceProperty = findPropertyByTitle(
            (dataFlowEdge.sourceNode["outputs"] as Property[] | undefined) ?? [],
            dataFlowEdge.sourceOutput,
            `the outputs of node \`${String(dataFlowEdge.sourceNode["name"])}\``,
          );
          const innerFlowInputProperty = findPropertyByTitle(
            (node.subflow["inputs"] as Property[] | undefined) ?? [],
            dataFlowEdge.destinationInput.replace("iterated_", ""),
            `the inputs of the subflow of MapNode \`${node.name}\``,
          );
          // Compare against an array-of-inner-input schema, like Python's
          // ListProperty(item_type=inner).json_schema (titles are ignored by
          // the comparison).
          if (
            jsonSchemasHaveSameType(sourceProperty.jsonSchema, {
              type: "array",
              items: innerFlowInputProperty.jsonSchema,
            })
          ) {
            inputsToIterate.push(dataFlowEdge.destinationInput);
          }
        }
        (nodeExecutors.get(node.id) as MapNodeExecutorLike).setInputsToIterate(
          inputsToIterate,
        );
      } else if (node.componentType === "EndNode") {
        (nodeExecutors.get(node.id) as EndNodeExecutorLike).setFlowOutputs(
          flow.outputs ?? [],
        );
      }
    }

    for (const [nodeId, nodeExecutor] of nodeExecutors) {
      // Graph node names are the AgentSpec node ids.
      graphBuilder.addNode(nodeId, (state: FlowState, config: RunnableConfig) =>
        nodeExecutor.call(state, config),
      );
    }

    let dataFlowConnections: DataFlowEdge[];
    if (flow.dataFlowConnections === undefined) {
      // Manually create data flow connections if they are not given in the
      // flow: one edge per matching-title (source output, destination input)
      // pair. This is the conversion recommended by the Agent Spec language
      // specification.
      dataFlowConnections = [];
      for (const sourceNode of flowNodes) {
        for (const destinationNode of flowNodes) {
          for (const sourceOutput of sourceNode.outputs ?? []) {
            for (const destinationInput of destinationNode.inputs ?? []) {
              if (sourceOutput.title === destinationInput.title) {
                dataFlowConnections.push(
                  createDataFlowEdge({
                    name: `${sourceNode.name}-${destinationNode.name}-${sourceOutput.title}`,
                    sourceNode: sourceNode as unknown as ComponentWithIO,
                    sourceOutput: sourceOutput.title,
                    destinationNode: destinationNode as unknown as ComponentWithIO,
                    destinationInput: destinationInput.title,
                  }),
                );
              }
            }
          }
        }
      }
    } else {
      dataFlowConnections = flow.dataFlowConnections;
    }

    for (const dataFlowEdge of dataFlowConnections) {
      // Flow validation guarantees every edge endpoint is a node of the flow.
      nodeExecutors
        .get(String(dataFlowEdge.sourceNode["id"]))!
        .attachEdge(dataFlowEdge);
    }

    this.addConditionalEdgesToGraph(flow.controlFlowConnections, graphBuilder);

    const compiledGraph = graphBuilder.compile(
      context.checkpointer !== undefined
        ? { checkpointer: context.checkpointer }
        : {},
    );
    return patchWithExecutionSpan(compiledGraph);
  }

  /** Add one conditional edge per source node, routing on the last branch. */
  private addConditionalEdgesToGraph(
    controlFlowConnections: ControlFlowEdge[],
    graphBuilder: DynamicStateGraph,
  ): void {
    const controlFlow = new Map<string, Record<string, string>>();
    for (const controlFlowEdge of controlFlowConnections) {
      const sourceNodeId = String(controlFlowEdge.fromNode["id"]);
      let mapping = controlFlow.get(sourceNodeId);
      if (mapping === undefined) {
        mapping = {};
        controlFlow.set(sourceNodeId, mapping);
      }
      // Python's `from_branch or DEFAULT_NEXT_BRANCH`: an empty-string
      // branch coerces to the default branch too, not just null/undefined.
      const branchName = controlFlowEdge.fromBranch || DEFAULT_NEXT_BRANCH;
      mapping[branchName] = String(controlFlowEdge.toNode["id"]);
    }
    for (const [sourceNodeId, controlFlowMapping] of controlFlow) {
      graphBuilder.addConditionalEdges(
        sourceNodeId,
        (state: FlowState) =>
          state.node_execution_details?.branch ?? DEFAULT_NEXT_BRANCH,
        controlFlowMapping,
      );
    }
  }

  /** Build the node executor for one flow node. */
  protected async convertNode(
    node: Node,
    context: ConversionContext,
  ): Promise<unknown> {
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
        if (!isCompiledGraph(subflow)) {
          throw new Error("FlowNodeExecutor can only initialize FlowNode");
        }
        return new FlowNodeExecutor(node, subflow, context.config);
      }
      case "CatchExceptionNode": {
        const subflow = await this.convertWithContext(
          node.subflow as unknown as ComponentBase,
          context,
        );
        if (!isCompiledGraph(subflow)) {
          throw new Error(
            "Internal error: CatchExceptionNodeExecutor expects `subflow` " +
              `to be a CompiledStateGraph, was ${typeof subflow}`,
          );
        }
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
        if (!isCompiledGraph(subflow)) {
          throw new Error("MapNodeExecutor can only be initialized with MapNode");
        }
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
  private convertAgentNode(node: AgentNode, context: ConversionContext): unknown {
    if (node.agent.componentType === "ManagerWorkers") {
      const managerWorkers = node.agent as ManagerWorkers;
      return new ManagerWorkersNodeExecutor(
        node,
        (renderedSystemPrompt: string) =>
          this.compileManagerWorkersGraph(
            managerWorkers,
            context,
            renderedSystemPrompt,
          ),
        context.config,
      );
    }
    const agentComponent = node.agent;
    const compileAgentFactory = async (
      renderedSystemPrompt: string,
    ): Promise<unknown> => {
      if (agentComponent.componentType !== "Agent") {
        throw new Error(
          "AgentNodeExecutor can only be used with AgentSpecAgent agents",
        );
      }
      const agent = agentComponent as Agent;
      return this.createReactAgentWithGivenInfo(
        {
          name: agent.name,
          systemPrompt: renderedSystemPrompt,
          agent,
          llmConfig: agent.llmConfig,
          tools: agent.tools,
          toolboxes: agent.toolboxes,
          inputs: agent.inputs ?? [],
          outputs: agent.outputs ?? [],
        },
        context,
      );
    };
    return new AgentNodeExecutor(node, compileAgentFactory, context.config);
  }
}
