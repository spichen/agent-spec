/**
 * Base class of the LangGraph flow node executors.
 *
 * Port of `pyagentspec.adapters.langgraph._node_execution.NodeExecutor`.
 *
 * Runtime contracts (state keys, branch names, error-message text) mirror the
 * Python adapter exactly so specs behave the same across both SDKs.
 *
 * Divergences from Python (see the adapter README):
 * - Execution is async-only (no sync `__call__` / thread offloading).
 * - Executors never mutate the incoming state: they return updated copies
 *   with the same accumulate semantics as Python's in-place mutation.
 * - Executors receive their collaborators from the converter (converted
 *   tools, chat models, compiled subgraphs, agent compile factories) instead
 *   of importing the converter, so there are no module cycles.
 * - Node execution spans/events are not emitted (tracing is a no-op seam).
 */
import type { BaseMessage } from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import { addMessages } from "@langchain/langgraph";
import type { DataFlowEdge } from "../../../flows/index.js";
import { DEFAULT_NEXT_BRANCH } from "../../../flows/index.js";
import type { Property } from "../../../property.js";
import { isRecordLike } from "../../common/index.js";
import type {
  ExecuteOutput,
  FlowState,
  NextNodeInputs,
  NodeExecutionDetails,
  NodeOutputs,
} from "../types.js";
import { castValuesAndAddDefaults } from "./python-parity.js";

/** The structural surface of an Agent Spec flow node used by the executors. */
export interface FlowNodeLike {
  id: string;
  name: string;
  inputs?: Property[];
  outputs?: Property[];
}

/**
 * Base class of the flow node executors.
 *
 * `call` is the LangGraph node function: it selects this node's pending
 * inputs from the state, casts them against the declared input properties,
 * executes the node, and returns the updated flow state (accumulated inputs
 * routing table, cast outputs, merged messages and execution details).
 */
export abstract class NodeExecutor<
  TNode extends FlowNodeLike = FlowNodeLike,
> {
  protected readonly node: TNode;
  protected readonly edges: DataFlowEdge[] = [];

  constructor(node: TNode) {
    this.node = node;
  }

  /** Attach a data-flow edge whose source is this node. */
  attachEdge(edge: DataFlowEdge): void {
    this.edges.push(edge);
  }

  /** Execute this node against the current flow state (LangGraph node fn). */
  async call(state: FlowState, _config?: RunnableConfig): Promise<FlowState> {
    const inputs = this.getInputs(state);
    const [outputs, executionDetails] = await this._execute(
      inputs,
      state.messages ?? [],
    );
    return this.updateStatus(outputs, executionDetails, state);
  }

  /** Execute the node with the given cast inputs; returns outputs + details. */
  protected abstract _execute(
    inputs: NodeOutputs,
    messages: BaseMessage[],
  ): Promise<ExecuteOutput>;

  /**
   * Retrieve the inputs for this node (the `state.inputs` entries keyed by
   * this node's id), adding default values when missing and casting to the
   * declared types.
   */
  protected getInputs(state: FlowState): NodeOutputs {
    const nodeInputs = state.inputs?.[this.node.id];
    const ioInputs: Record<string, unknown> = isRecordLike(nodeInputs)
      ? { ...nodeInputs }
      : {};
    return castValuesAndAddDefaults(
      ioInputs,
      this.node.inputs ?? [],
      this.node.name,
    );
  }

  /**
   * Fold the node outputs and execution details into the flow state: cast the
   * outputs, route them along the attached data-flow edges into the pending
   * inputs of downstream nodes (accumulating into a copy of the previous
   * routing table), default the execution details, and merge generated
   * messages via LangGraph's `addMessages`.
   */
  protected updateStatus(
    outputs: NodeOutputs,
    executionDetails: NodeExecutionDetails,
    previousState: FlowState,
  ): FlowState {
    const castOutputs = castValuesAndAddDefaults(
      outputs,
      this.node.outputs ?? [],
      this.node.name,
    );
    const nextNodeInputs: NextNodeInputs = { ...(previousState.inputs ?? {}) };
    for (const edge of this.edges) {
      const destinationNodeId = String(edge.destinationNode["id"]);
      const existing = nextNodeInputs[destinationNodeId];
      const destinationInputs: Record<string, unknown> = isRecordLike(existing)
        ? { ...existing }
        : {};
      if (!Object.hasOwn(castOutputs, edge.sourceOutput)) {
        // Python raises a bare KeyError here.
        throw new Error(
          `Node \`${this.node.name}\` produced no output ` +
            `\`${edge.sourceOutput}\` required by data-flow edge \`${edge.name}\`.`,
        );
      }
      destinationInputs[edge.destinationInput] = castOutputs[edge.sourceOutput];
      nextNodeInputs[destinationNodeId] = destinationInputs;
    }

    const details: NodeExecutionDetails = {
      branch: executionDetails.branch ?? DEFAULT_NEXT_BRANCH,
      generated_messages: executionDetails.generated_messages ?? [],
      should_finish: executionDetails.should_finish ?? false,
    };
    return {
      inputs: nextNodeInputs,
      outputs: castOutputs,
      messages: addMessages(
        previousState.messages ?? [],
        details.generated_messages ?? [],
      ),
      node_execution_details: details,
    };
  }
}
