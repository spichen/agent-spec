/**
 * Executors for the structural flow nodes: Start, End, Branching,
 * InputMessage and OutputMessage.
 *
 * Port of the matching executors in
 * `pyagentspec.adapters.langgraph._node_execution`. Runtime contracts (state
 * keys, branch names, interrupt payloads, error-message text) mirror the
 * Python adapter exactly so specs behave the same across both SDKs — the
 * InputMessageNode interrupts the graph with the same empty-string payload.
 */
import type { BaseMessage } from "@langchain/core/messages";
import { interrupt } from "@langchain/langgraph";
import type {
  BranchingNode,
  EndNode,
  InputMessageNode,
  OutputMessageNode,
  StartNode,
} from "../../../flows/index.js";
import {
  DEFAULT_BRANCH,
  DEFAULT_INPUT_MESSAGE_OUTPUT,
} from "../../../flows/index.js";
import type { Property } from "../../../property.js";
import { renderTemplate } from "../../common/index.js";
import type {
  ExecuteOutput,
  FlowState,
  NodeExecutionDetails,
  NodeOutputs,
} from "../types.js";
import { NodeExecutor } from "./executor.js";
import { castValuesAndAddDefaults } from "./python-parity.js";

/**
 * Executes a StartNode: consumes the flow-level invocation inputs (plain
 * string keys at the top level of `state.inputs`) and passes them through as
 * outputs, flowing to downstream nodes along the data edges.
 */
export class StartNodeExecutor extends NodeExecutor<StartNode> {
  protected override getInputs(state: FlowState): NodeOutputs {
    // At StartNode time the state inputs hold the flow's initial call inputs
    // as plain `{inputName: value}` keys (no node-id nesting): consume all of
    // them (they are removed from the state in updateStatus below).
    const ioInputs: Record<string, unknown> = { ...(state.inputs ?? {}) };
    return castValuesAndAddDefaults(
      ioInputs,
      this.node.inputs ?? [],
      this.node.name,
    );
  }

  protected override updateStatus(
    outputs: NodeOutputs,
    executionDetails: NodeExecutionDetails,
    previousState: FlowState,
  ): FlowState {
    // Python pops the consumed flow-level inputs out of the state; the
    // non-mutating equivalent is starting the routing table from scratch.
    return super.updateStatus(outputs, executionDetails, {
      ...previousState,
      inputs: {},
    });
  }

  protected async _execute(
    inputs: NodeOutputs,
    _messages: BaseMessage[],
  ): Promise<ExecuteOutput> {
    return [inputs, {}];
  }
}

/**
 * Executes an EndNode: passes its inputs through as outputs, reshapes them to
 * the flow's declared outputs, and marks the run finished on the node's
 * branch.
 */
export class EndNodeExecutor extends NodeExecutor<EndNode> {
  private flowOutputs: Property[] = [];

  /** Give the executor the flow outputs used to reshape the final state. */
  setFlowOutputs(flowOutputs: Property[]): void {
    this.flowOutputs = flowOutputs;
  }

  protected async _execute(
    inputs: NodeOutputs,
    _messages: BaseMessage[],
  ): Promise<ExecuteOutput> {
    return [inputs, { branch: this.node.branchName, should_finish: true }];
  }

  protected override updateStatus(
    outputs: NodeOutputs,
    executionDetails: NodeExecutionDetails,
    previousState: FlowState,
  ): FlowState {
    const newState = super.updateStatus(
      outputs,
      executionDetails,
      previousState,
    );
    const nodeOutputs = newState.outputs;
    const filteredOutputs: NodeOutputs = {};
    for (const property of this.flowOutputs) {
      filteredOutputs[property.title] = Object.hasOwn(
        nodeOutputs,
        property.title,
      )
        ? nodeOutputs[property.title]
        : property.default;
    }
    for (const [propertyName, propertyValue] of Object.entries(nodeOutputs)) {
      if (propertyValue === undefined) {
        throw new Error(
          `EndNode \`${this.node.name}\` exited without any value generated for property \`${propertyName}\``,
        );
      }
    }
    return { ...newState, outputs: filteredOutputs };
  }
}

/**
 * Executes a BranchingNode: reads its first input and selects the branch its
 * mapping points to (the `default` branch when the value is unmapped).
 */
export class BranchingNodeExecutor extends NodeExecutor<BranchingNode> {
  constructor(node: BranchingNode) {
    super(node);
    if (!node.inputs || node.inputs.length === 0) {
      throw new Error("BranchingNode requires at least one input");
    }
  }

  protected async _execute(
    inputs: NodeOutputs,
    _messages: BaseMessage[],
  ): Promise<ExecuteOutput> {
    const nodeInputs = this.node.inputs ?? [];
    const inputBranchPropTitle = nodeInputs[0]!.title;
    const inputBranchName = Object.hasOwn(inputs, inputBranchPropTitle)
      ? inputs[inputBranchPropTitle]
      : DEFAULT_BRANCH;
    const selectedBranch =
      typeof inputBranchName === "string" &&
      Object.hasOwn(this.node.mapping, inputBranchName)
        ? this.node.mapping[inputBranchName]!
        : DEFAULT_BRANCH;
    return [{}, { branch: selectedBranch }];
  }
}

/**
 * Executes an InputMessageNode: interrupts the graph with an empty-string
 * payload; the resume value becomes both the node output and a new user
 * message.
 */
export class InputMessageNodeExecutor extends NodeExecutor<InputMessageNode> {
  protected async _execute(
    _inputs: NodeOutputs,
    _messages: BaseMessage[],
  ): Promise<ExecuteOutput> {
    const response = interrupt("");
    const outputs = this.node.outputs ?? [];
    const outputName =
      outputs.length > 0 ? outputs[0]!.title : DEFAULT_INPUT_MESSAGE_OUTPUT;
    return [
      { [outputName]: response },
      {
        generated_messages: [
          { role: "user", content: response as string },
        ],
      },
    ];
  }
}

/**
 * Executes an OutputMessageNode: renders the node's message template against
 * the inputs and emits it as an assistant message.
 */
export class OutputMessageNodeExecutor extends NodeExecutor<OutputMessageNode> {
  protected async _execute(
    inputs: NodeOutputs,
    _messages: BaseMessage[],
  ): Promise<ExecuteOutput> {
    const message = renderTemplate(this.node.message, inputs);
    return [
      {},
      { generated_messages: [{ role: "assistant", content: message }] },
    ];
  }
}
