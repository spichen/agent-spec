/**
 * ManagerWorkers compilation for the LangGraph adapter.
 *
 * Port of `pyagentspec.adapters.langgraph._managerworkers` and
 * `_managerworkers_node`: a hierarchical StateGraph where a react-agent
 * manager delegates tasks to worker subgraphs through synthetic
 * `__delegate_to__<worker>` tools.
 *
 * The delegation protocol is visible on purpose: `__delegate_to__<worker>`
 * calls stream like any other tool call. Consumers that would rather not
 * render them can filter on `isDelegationToolName`.
 *
 * Runtime contracts (node names, tool names, Send payload keys, roster text)
 * mirror the Python adapter exactly so specs behave the same across SDKs.
 */
import type { BaseMessage } from "@langchain/core/messages";
import { HumanMessage, ToolMessage } from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import type { StructuredToolInterface } from "@langchain/core/tools";
import { tool } from "@langchain/core/tools";
import type { BaseCheckpointSaver } from "@langchain/langgraph";
import {
  Command,
  END,
  MessagesAnnotation,
  START,
  Send,
  StateGraph,
  getCurrentTaskInput,
} from "@langchain/langgraph";
import type { ManagerWorkers } from "../../agents/index.js";
import type { AgentNode } from "../../flows/index.js";
import { renderTemplate } from "../common/index.js";
import { AgentNodeExecutor } from "./node-execution.js";
import { patchWithExecutionSpan } from "./tracing.js";
import type {
  DynamicStateGraph,
  ExecuteOutput,
  InvocableGraph,
  NodeOutputs,
} from "./types.js";

/**
 * Prefix of the synthetic `__delegate_to__<worker>` tool names the manager's
 * LLM uses to address a worker. The dunder prefix, like the delegation keys
 * below, keeps it from colliding with a real tool named
 * `delegate_to_<something>`.
 */
export const DELEGATE_TOOL_PREFIX = "__delegate_to__";

// Cannot collide with a worker node name: normalizeIdentifier strips leading
// and trailing underscores, so no normalized name ever starts with one.
const MANAGER_NODE_KEY = "__manager__";

// Keys of the per-delegation `Send` payload: the task to run, and the
// tool_call_id the worker's reply must answer. Routing per delegation
// (instead of off shared state) lets one manager turn delegate to several
// workers at once.
const DELEGATE_TASK_KEY = "__delegate_task__";
const DELEGATE_CALL_ID_KEY = "__delegate_tool_call_id__";

/** True for the synthetic `__delegate_to__<worker>` tool names a manager emits. */
export function isDelegationToolName(name: unknown): name is string {
  return typeof name === "string" && name.startsWith(DELEGATE_TOOL_PREFIX);
}

/** Lowercase, collapse non-alphanumerics to underscores, strip leading/trailing ones. */
function normalizeIdentifier(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

/**
 * Normalize a worker name into a LangGraph node identifier.
 *
 * The LLM has to emit `__delegate_to__<node_name>` reliably as a tool name,
 * so node names stay ASCII identifiers. Falls back to the normalized
 * component id when the name slugifies to nothing.
 */
function safeNodeName(name: string, fallbackId: string): string {
  return normalizeIdentifier(name) || normalizeIdentifier(fallbackId) || "worker";
}

/** Read `messages` off a state object, defensively copied to an array. */
function messagesOf(state: unknown): unknown[] {
  if (typeof state === "object" && state !== null) {
    const messages = (state as { messages?: unknown }).messages;
    if (Array.isArray(messages)) {
      return [...messages];
    }
  }
  return [];
}

/**
 * Append an `Available workers:` block listing `- <name>: <description>`.
 *
 * Descriptions are flattened to one line each, since the LLM routes off the
 * block's one-line-per-worker shape.
 */
function appendWorkersRoster(
  systemPrompt: string,
  entries: [string, string][],
): string {
  if (entries.length === 0) {
    return systemPrompt;
  }
  const lines = entries.map(
    ([name, description]) => `- ${name}: ${description.replace(/\s+/g, " ").trim()}`,
  );
  const roster = "Available workers:\n" + lines.join("\n");
  return systemPrompt ? `${systemPrompt}\n\n${roster}` : roster;
}

/**
 * Build the `__delegate_to__<worker>` tool the manager's LLM emits to route
 * to a worker.
 *
 * Executing the tool is only how the call escapes the react subgraph: its
 * body surfaces the subgraph messages to the parent with
 * `Command({graph: Command.PARENT})` and no `goto`. Routing stays in the edge
 * built by `makeManagerRouter`; a `goto` here would collapse several
 * same-turn delegations into one parent Command and leave the other
 * `tool_call_id`s unanswered.
 */
function makeWorkerDelegationTool(
  workerNodeName: string,
): StructuredToolInterface {
  const toolName = `${DELEGATE_TOOL_PREFIX}${workerNodeName}`;
  const description =
    `Delegate a task to the ${workerNodeName} worker and receive its response. ` +
    `Use this when the task fits the worker's described capability.`;
  return tool(
    async () => {
      // The task (and the tool_call_id) are declared for the LLM-facing
      // schema; the routing edge recovers both off the surfaced AIMessage's
      // tool_calls. The addMessages reducer dedupes by id, so re-surfacing
      // messages is a no-op. `getCurrentTaskInput` is the JS equivalent of
      // Python's InjectedState.
      //
      // JS divergence from Python (which returns an update-only parent
      // command): `goto: END` is REQUIRED here. `Command#goto` defaults to
      // `[]`, and langchain's ToolNode folds any parent command whose goto is
      // an array of Sends (the empty array included) into a goto-only
      // command, dropping the update — the manager's messages would never
      // reach the parent graph. A truthy non-Send goto keeps the command
      // intact end to end; END is harmless as a routed destination because
      // the delegation Sends emitted by the router create their own tasks
      // (verified against @langchain/langgraph 1.4.13 / langchain 1.5.10).
      const state = getCurrentTaskInput();
      return new Command({
        graph: Command.PARENT,
        goto: END,
        update: { messages: messagesOf(state) },
      });
    },
    {
      name: toolName,
      description,
      schema: {
        type: "object",
        properties: { task: { type: "string" } },
        required: ["task"],
      },
    },
  ) as StructuredToolInterface;
}

interface ToolCallLike {
  name?: unknown;
  args?: Record<string, unknown>;
  id?: unknown;
}

/**
 * Build the conditional edge routing the parent graph off the manager's last
 * AIMessage: one `Send` per `__delegate_to__<worker>` tool call, or `END`
 * when it emitted none.
 *
 * Every delegation gets its own `Send`, so each tool_call_id is answered
 * independently; an unanswered one breaks the manager's next-turn
 * tool-call/result sequence. Plain tool calls already ran inside the react
 * loop; that includes a real tool whose name merely starts with the prefix,
 * which is why a suffix that is not a worker node is not routed.
 */
function makeManagerRouter(
  workerNodeNames: string[],
): (state: Record<string, unknown>) => Send[] | string {
  const knownWorkers = new Set(workerNodeNames);
  return (state: Record<string, unknown>): Send[] | string => {
    const messages = Array.isArray(state["messages"]) ? state["messages"] : [];
    const last = messages[messages.length - 1] as
      | { tool_calls?: ToolCallLike[] }
      | undefined;
    const sends: Send[] = [];
    for (const toolCall of last?.tool_calls ?? []) {
      const name = toolCall?.name;
      if (!isDelegationToolName(name)) {
        continue;
      }
      const workerNodeName = name.slice(DELEGATE_TOOL_PREFIX.length);
      if (!knownWorkers.has(workerNodeName)) {
        continue;
      }
      const args = toolCall.args ?? {};
      const task = args["task"];
      sends.push(
        new Send(workerNodeName, {
          [DELEGATE_TASK_KEY]: (typeof task === "string" ? task : "") || "",
          [DELEGATE_CALL_ID_KEY]:
            (typeof toolCall.id === "string" ? toolCall.id : "") || "",
        }),
      );
    }
    return sends.length > 0 ? sends : END;
  };
}

/**
 * Wrap a worker subgraph as a node of the ManagerWorkers parent graph.
 *
 * Hierarchical rather than shared-state like a Swarm: each run is handed only
 * the manager's chosen task, and the worker's answer comes back as a
 * ToolMessage so the manager's react loop sees a well-formed tool response on
 * its next turn. The worker receives this node's ambient run config
 * explicitly, which streams its token events under the worker node's
 * checkpoint namespace.
 */
function wrapWorkerForSubgraph(
  workerGraph: unknown,
  _workerNodeName: string,
): (
  state: Record<string, unknown>,
  config: RunnableConfig,
) => Promise<Record<string, unknown>> {
  return async function workerNode(
    state: Record<string, unknown>,
    config: RunnableConfig,
  ): Promise<Record<string, unknown>> {
    const task = state[DELEGATE_TASK_KEY];
    const input = {
      messages: [
        new HumanMessage({
          content: (typeof task === "string" ? task : "") || "",
        }),
      ],
    };
    const result = await (workerGraph as InvocableGraph).invoke(input, config);
    const messages = Array.isArray(result?.["messages"])
      ? (result["messages"] as { content?: unknown }[])
      : [];
    const lastMessage = messages[messages.length - 1];
    const content = lastMessage?.content ?? "";
    const callId = state[DELEGATE_CALL_ID_KEY];
    return {
      messages: [
        new ToolMessage({
          content: (content as string) || "",
          tool_call_id: (typeof callId === "string" ? callId : "") || "",
        }),
      ],
    };
  };
}

/** Options for `compileManagerWorkers`. */
export interface CompileManagerWorkersOptions {
  /** Checkpointer wired into the compiled parent graph. */
  checkpointer?: BaseCheckpointSaver;
  /**
   * Overrides the group manager's system prompt (before the roster is
   * appended). Used by `ManagerWorkersNodeExecutor` to bake rendered flow
   * inputs into the prompt.
   */
  systemPrompt?: string;
  /**
   * Compiles the group-manager Agent into a graph usable as the
   * `__manager__` node, given the roster-augmented system prompt and the
   * delegation tools to append. Provided by the converter (which owns react
   * agent assembly).
   */
  compileManagerAgent: (
    rosterSystemPrompt: string,
    delegationTools: StructuredToolInterface[],
  ) => Promise<unknown>;
  /**
   * Converts one worker agentic component into an invocable graph. Provided
   * by the converter (recursive conversion, memoized by component id).
   */
  convertWorker: (worker: Record<string, unknown>) => Promise<unknown>;
}

/**
 * Compile a `ManagerWorkers` into a hierarchical LangGraph.
 *
 * Topology:
 *
 *                     ┌─ __delegate_to__w1 ─→ worker_1 ─┐
 *     START → manager ┤                                 ├→ manager (loop)
 *                     └─ __delegate_to__w2 ─→ worker_2 ─┘
 *                             │
 *                             └─ no tool_call ─→ END
 *
 * The manager is a react-agent holding one synthetic `__delegate_to__<worker>`
 * tool per worker. A conditional edge routes each delegation to its worker,
 * which runs in an isolated message context and answers with a `ToolMessage`
 * matched to the pending delegation id. Workers are converted recursively and
 * wired in as subgraph nodes.
 */
export async function compileManagerWorkers(
  managerWorkers: ManagerWorkers,
  options: CompileManagerWorkersOptions,
): Promise<unknown> {
  const groupManager = managerWorkers.groupManager;
  if (groupManager["componentType"] !== "Agent") {
    // Delegation is routed off the manager's tool_calls, so the manager needs
    // a chat-LLM; a Flow, Swarm or nested ManagerWorkers gives nothing to
    // route on.
    throw new Error(
      `ManagerWorkers.group_manager must be an Agent for LangGraph ` +
        `conversion; got ${String(groupManager["componentType"])}.`,
    );
  }

  const namedWorkers: [string, Record<string, unknown>][] =
    managerWorkers.workers.map((worker) => [
      safeNodeName(String(worker["name"] ?? ""), String(worker["id"] ?? "")),
      worker,
    ]);
  const workerNodeNames = namedWorkers.map(([nodeName]) => nodeName);
  if (new Set(workerNodeNames).size !== workerNodeNames.length) {
    throw new Error(
      "ManagerWorkers worker names collide after normalization: " +
        `${JSON.stringify(workerNodeNames)}. Give each worker a unique name.`,
    );
  }

  // The roster tells the LLM which delegation tool maps to which worker.
  const basePrompt =
    options.systemPrompt ?? String(groupManager["systemPrompt"] ?? "");
  const rosterPrompt = appendWorkersRoster(
    basePrompt,
    namedWorkers.map(([nodeName, worker]) => [
      nodeName,
      String(worker["description"] ?? ""),
    ]),
  );

  // The delegation tools execute inside the react loop: their
  // Command({graph: PARENT}) is how the call escapes the subgraph so the
  // conditional edge below can route on it.
  const delegationTools = workerNodeNames.map((nodeName) =>
    makeWorkerDelegationTool(nodeName),
  );
  const managerGraph = await options.compileManagerAgent(
    rosterPrompt,
    delegationTools,
  );

  // Manager and workers all go in as compiled subgraph nodes, which is what
  // makes LangGraph stream them with `subgraph: true`.
  const builder = new StateGraph(
    MessagesAnnotation,
  ) as unknown as DynamicStateGraph;
  builder.addNode(MANAGER_NODE_KEY, managerGraph);
  for (const [nodeName, worker] of namedWorkers) {
    const workerGraph = await options.convertWorker(worker);
    builder.addNode(nodeName, wrapWorkerForSubgraph(workerGraph, nodeName));
    // Workers always loop back to the manager.
    builder.addEdge(nodeName, MANAGER_NODE_KEY);
  }

  builder.addEdge(START, MANAGER_NODE_KEY);
  const pathMap: Record<string, string> = {};
  for (const nodeName of workerNodeNames) {
    pathMap[nodeName] = nodeName;
  }
  pathMap[END] = END;
  // The path map covers every worker plus END, so langgraph can validate the
  // routing statically.
  builder.addConditionalEdges(
    MANAGER_NODE_KEY,
    makeManagerRouter(workerNodeNames),
    pathMap,
  );

  const compiledGraph = builder.compile({
    ...(options.checkpointer !== undefined
      ? { checkpointer: options.checkpointer }
      : {}),
    name: managerWorkers.name,
  });
  return patchWithExecutionSpan(compiledGraph, {
    kind: "manager-workers",
    component: managerWorkers,
  });
}

/**
 * Executes an `AgentNode` whose agent is a `ManagerWorkers`.
 *
 * The hierarchical graph runs over `MessagesState`, which can carry neither
 * structured inputs inward nor a `structured_response` outward. Inputs are
 * therefore rendered into the group-manager's system prompt before compiling,
 * and the manager's final message is the node's single string output.
 *
 * Mirroring Python, only the parent's two template-method hooks are
 * overridden: `prepareAgentAndInputs` (compile the hierarchical graph, run it
 * on messages alone) and `formatAgentResult` (single-string output). The
 * compile callback, the rendered-prompt cache and `withDrivingMessage` are
 * the inherited ones.
 */
export class ManagerWorkersNodeExecutor extends AgentNodeExecutor {
  private readonly managerWorkers: ManagerWorkers;

  constructor(
    node: AgentNode,
    compileManagerWorkers: (renderedSystemPrompt: string) => Promise<unknown>,
    config: RunnableConfig,
  ) {
    super(node, compileManagerWorkers, config);
    if (node.agent.componentType !== "ManagerWorkers") {
      throw new Error(
        "ManagerWorkersNodeExecutor requires an AgentNode holding a ManagerWorkers",
      );
    }
    this.managerWorkers = node.agent;
    // Anything but a single string output cannot be honored (see class
    // docstring); raising here fails at conversion time rather than mid-run.
    const outputs = node.outputs ?? [];
    if (outputs.length > 0 && (outputs.length !== 1 || outputs[0]!.type !== "string")) {
      throw new Error(
        "A ManagerWorkers flow step supports a single string output; " +
          `node \`${node.name}\` declares ${JSON.stringify(outputs.map((o) => o.title))}.`,
      );
    }
  }

  /**
   * Compile the `ManagerWorkers` with the node inputs rendered into the
   * group-manager's system prompt, cached by rendered prompt (the same
   * `agentsCache` key `AgentNodeExecutor` uses for its react-agent cache).
   */
  private async createManagerWorkersWithGivenInputValues(
    inputs: NodeOutputs,
  ): Promise<unknown> {
    const groupManager = this.managerWorkers.groupManager;
    const systemPrompt = renderTemplate(
      String(groupManager["systemPrompt"] ?? ""),
      inputs,
    );
    let graph = this.agentsCache.get(systemPrompt);
    if (graph === undefined) {
      graph = await this.compileAgent(systemPrompt);
      this.agentsCache.set(systemPrompt, graph);
    }
    return graph;
  }

  protected override async prepareAgentAndInputs(
    inputs: NodeOutputs,
    messages: BaseMessage[],
  ): Promise<[InvocableGraph, Record<string, unknown>]> {
    // Inputs were baked into the group-manager's prompt, so this graph runs
    // on messages alone rather than the react-agent's remaining_steps state.
    const graph = await this.createManagerWorkersWithGivenInputValues(inputs);
    return [
      graph as InvocableGraph,
      { messages: this.withDrivingMessage(messages) },
    ];
  }

  protected override formatAgentResult(
    result: Record<string, unknown>,
  ): ExecuteOutput {
    const nodeOutputs = this.node.outputs ?? [];
    if (nodeOutputs.length === 0) {
      return super.formatAgentResult(result);
    }
    const resultMessages = Array.isArray(result["messages"])
      ? (result["messages"] as { content?: unknown }[])
      : [];
    const lastMessage = resultMessages[resultMessages.length - 1];
    // The constructor already rejected any shape but a single string output.
    return [{ [nodeOutputs[0]!.title]: lastMessage?.content }, {}];
  }
}
