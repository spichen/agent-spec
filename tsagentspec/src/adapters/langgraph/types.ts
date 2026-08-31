/**
 * Shared types for the LangGraph adapter.
 *
 * Runtime contracts (state keys, node names, interrupt payloads) mirror the Python
 * `pyagentspec.adapters.langgraph` adapter exactly so that specs behave the same
 * across both SDKs.
 */
import type { BaseMessage, BaseMessageLike } from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import type { BaseCheckpointSaver } from "@langchain/langgraph";

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
 * Registry mapping tool names to runtime implementations: a LangChain structured
 * tool or a plain (sync or async) function. MCP tools are cached here under
 * `${clientTransportId}::${toolName}` keys.
 */
export type ToolRegistry = Record<string, unknown>;

/** Options threaded through AgentSpec-to-LangGraph conversion. */
export interface ConvertOptions {
  /** Per-call conversion cache keyed by component id; pre-seed to inject fakes. */
  convertedComponents?: Map<string, unknown>;
  checkpointer?: BaseCheckpointSaver;
  config?: RunnableConfig;
  /** LangChain agent middleware, forwarded to createAgent in order (index 0 outermost). */
  middleware?: unknown[];
}
