/**
 * Shared types for the LangGraph adapter.
 *
 * Runtime contracts (state keys, node names, interrupt payloads) mirror the Python
 * `pyagentspec.adapters.langgraph` adapter exactly so that specs behave the same
 * across both SDKs.
 */
import type { BaseMessage, BaseMessageLike } from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import type { StructuredToolInterface } from "@langchain/core/tools";
import type { BaseCheckpointSaver, Send } from "@langchain/langgraph";

/** Execution metadata produced by every flow node step. */
export interface NodeExecutionDetails {
  should_finish?: boolean;
  branch?: string;
  generated_messages?: BaseMessageLike[];
}

/** Outputs produced by a node execution, keyed by output property title. */
export type NodeOutputs = Record<string, unknown>;

/**
 * Pending inputs for downstream nodes: nodeId -> {inputTitle: value}.
 * Flow-level inputs are stored under plain string keys (consumed by the StartNode).
 */
export type NextNodeInputs = Record<string, unknown>;

/** State schema of a compiled AgentSpec Flow graph (keys mirror the Python adapter). */
export interface FlowState {
  inputs: NextNodeInputs;
  outputs: NodeOutputs;
  messages: BaseMessage[];
  node_execution_details: NodeExecutionDetails;
}

/** Result of a node executor: outputs plus execution details. */
export type ExecuteOutput = [NodeOutputs, NodeExecutionDetails];

/**
 * A compiled graph / react agent surface: everything invocable.
 */
export interface InvocableGraph {
  invoke(
    input: unknown,
    config?: RunnableConfig,
  ): Promise<Record<string, unknown>>;
}

/**
 * Loosely-typed StateGraph surface for graphs with dynamic node names (the
 * `StateGraph` generics cannot express node sets decided at conversion time).
 * The `path` signature is the superset of the adapter's routing functions:
 * flow conditional edges route on `FlowState`, the ManagerWorkers router
 * returns per-delegation `Send`s off a plain messages state.
 */
export interface DynamicStateGraph {
  addNode(key: string, action: unknown): DynamicStateGraph;
  addEdge(start: string, end: string): DynamicStateGraph;
  addConditionalEdges(
    source: string,
    path: (state: FlowState & Record<string, unknown>) => Send[] | string,
    pathMap?: Record<string, string>,
  ): DynamicStateGraph;
  compile(options?: {
    checkpointer?: BaseCheckpointSaver;
    name?: string;
  }): unknown;
}

/** A plain (sync or async) tool implementation: receives the parsed input object. */
export type ToolImplementation = (input: unknown, config?: unknown) => unknown;

/**
 * Registry mapping tool names to runtime implementations: a LangChain structured
 * tool or a plain (sync or async) function. MCP tools are cached here under
 * `${clientTransportId}::${toolName}` keys.
 */
export type ToolRegistry = Record<
  string,
  StructuredToolInterface | ToolImplementation
>;

/** Options threaded through AgentSpec-to-LangGraph conversion. */
export interface ConvertOptions {
  /** Per-call conversion cache keyed by component id; pre-seed to inject fakes. */
  convertedComponents?: Map<string, unknown>;
  checkpointer?: BaseCheckpointSaver;
  config?: RunnableConfig;
  /** LangChain agent middleware, forwarded to createAgent in order (index 0 outermost). */
  middleware?: unknown[];
}
