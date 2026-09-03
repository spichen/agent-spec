/**
 * Executors for the subflow-holding nodes: FlowNode, CatchExceptionNode and
 * MapNode.
 *
 * Port of the matching executors in
 * `pyagentspec.adapters.langgraph._node_execution`. Runtime contracts (state
 * keys, branch names, error-message text) mirror the Python adapter exactly
 * so specs behave the same across both SDKs.
 *
 * Divergence from Python (see the adapter README): node execution
 * spans/events are not emitted (tracing is a no-op seam), so the
 * CatchExceptionNode emits no ExceptionRaised event on error.
 */
import type { BaseMessage } from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import type {
  CatchExceptionNode,
  FlowNode,
  MapNode,
} from "../../../flows/index.js";
import {
  CAUGHT_EXCEPTION_BRANCH,
  DEFAULT_NEXT_BRANCH,
} from "../../../flows/index.js";
import type { Property } from "../../../property.js";
import { isRecordLike, stringifyTemplateValue } from "../../common/index.js";
import type {
  ExecuteOutput,
  InvocableGraph,
  NodeExecutionDetails,
  NodeOutputs,
} from "../types.js";
import { NodeExecutor } from "./executor.js";

/**
 * Executes a FlowNode: invokes the compiled subflow with this node's inputs
 * and messages; the subflow's outputs become the node outputs and its
 * terminating EndNode branch propagates as this node's branch.
 */
export class FlowNodeExecutor extends NodeExecutor<FlowNode> {
  private readonly subflow: InvocableGraph;
  private readonly config: RunnableConfig;

  constructor(node: FlowNode, subflow: InvocableGraph, config: RunnableConfig) {
    super(node);
    this.subflow = subflow;
    this.config = config;
  }

  protected async _execute(
    inputs: NodeOutputs,
    messages: BaseMessage[],
  ): Promise<ExecuteOutput> {
    const flowOutput = await this.subflow.invoke(
      { messages, inputs },
      this.config,
    );
    const details = flowOutput["node_execution_details"] as
      | NodeExecutionDetails
      | undefined;
    return [
      (flowOutput["outputs"] ?? {}) as NodeOutputs,
      { branch: details?.branch ?? DEFAULT_NEXT_BRANCH },
    ];
  }
}

/**
 * Executes a CatchExceptionNode: invokes the compiled subflow; on success the
 * subflow outputs pass through with `caught_exception_info: null`, and on
 * error the subflow's declared output defaults are emitted with the error
 * message on the `caught_exception_branch`.
 */
export class CatchExceptionNodeExecutor extends NodeExecutor<CatchExceptionNode> {
  private readonly subflow: InvocableGraph;
  private readonly config: RunnableConfig;

  constructor(
    node: CatchExceptionNode,
    subflow: InvocableGraph,
    config: RunnableConfig,
  ) {
    super(node);
    this.subflow = subflow;
    this.config = config;
  }

  protected async _execute(
    inputs: NodeOutputs,
    messages: BaseMessage[],
  ): Promise<ExecuteOutput> {
    try {
      const flowOutput = await this.subflow.invoke(
        { messages, inputs },
        this.config,
      );
      const outputs: NodeOutputs = isRecordLike(flowOutput["outputs"])
        ? { ...(flowOutput["outputs"] as NodeOutputs) }
        : {};
      // As per the spec, when the subflow runs without error
      // `caught_exception_info` is null.
      outputs["caught_exception_info"] = null;
      const details = flowOutput["node_execution_details"] as
        | NodeExecutionDetails
        | undefined;
      return [outputs, { branch: details?.branch ?? DEFAULT_NEXT_BRANCH }];
    } catch (error) {
      // Python emits an ExceptionRaised event on the current node span here;
      // tracing is a no-op seam in the TS adapter, so nothing is emitted.
      const defaultOutputs: NodeOutputs = {};
      const subflowOutputs =
        (this.node.subflow["outputs"] as Property[] | undefined) ?? [];
      for (const property of subflowOutputs) {
        // Use default value for subflow outputs when exception occurs
        defaultOutputs[property.title] = property.default;
      }
      defaultOutputs["caught_exception_info"] =
        error instanceof Error ? error.message : String(error);
      return [defaultOutputs, { branch: CAUGHT_EXCEPTION_BRANCH }];
    }
  }
}

/**
 * Executes a MapNode: iterates the compiled subflow over the `iterated_`
 * inputs the converter selected (broadcasting the others) and appends each
 * run's subflow outputs into the node's `collected_` outputs.
 */
export class MapNodeExecutor extends NodeExecutor<MapNode> {
  private readonly subflow: InvocableGraph;
  private inputsToIterate: string[] = [];

  constructor(node: MapNode, subflow: InvocableGraph) {
    super(node);
    if (!node.inputs || node.inputs.length === 0) {
      throw new Error("MapNode has no inputs");
    }
    // Mirroring Python, the subflow runs are not passed the ambient config.
    this.subflow = subflow;
  }

  /** Set which inputs to iterate over (decided by the converter). */
  setInputsToIterate(inputsToIterate: string[]): void {
    this.inputsToIterate = inputsToIterate;
  }

  private prepareIterations(inputs: NodeOutputs): {
    subflowInputsList: Record<string, unknown>[];
    outputs: Record<string, unknown[]>;
  } {
    const outputs: Record<string, unknown[]> = {};
    for (const output of this.node.outputs ?? []) {
      outputs[output.title] = [];
    }

    if (this.inputsToIterate.length === 0) {
      throw new Error("MapNode has no inputs to iterate");
    }

    let numInputsToIterate: number | undefined;
    for (const inputName of this.inputsToIterate) {
      const iterable = inputs[inputName];
      const size =
        Array.isArray(iterable) || typeof iterable === "string"
          ? iterable.length
          : undefined;
      if (size === undefined) {
        // Python raises a TypeError from `len()` here; the adapter names the
        // node and the offending input instead.
        throw new Error(
          `MapNode \`${this.node.name}\` cannot iterate over input ` +
            `\`${inputName}\`: ${stringifyTemplateValue(iterable)} has no length`,
        );
      }
      if (numInputsToIterate === undefined) {
        numInputsToIterate = size;
      } else if (size !== numInputsToIterate) {
        throw new Error(
          `Found inputs to iterate with different sizes (${stringifyTemplateValue(iterable)} and ${numInputsToIterate})`,
        );
      }
    }
    if (numInputsToIterate === undefined) {
      throw new Error(
        "MapNode inputs_to_iterate did not match any provided inputs",
      );
    }

    const subflowInputsList: Record<string, unknown>[] = [];
    for (let i = 0; i < numInputsToIterate; i += 1) {
      const subInputs: Record<string, unknown> = {};
      for (const inputProperty of this.node.inputs ?? []) {
        const title = inputProperty.title;
        // Note: Python strips every `iterated_` occurrence here (str.replace
        // with no count), not just the prefix.
        const subflowInputName = title.replaceAll("iterated_", "");
        if (this.inputsToIterate.includes(title)) {
          const collection = inputs[title];
          subInputs[subflowInputName] = Array.isArray(collection)
            ? collection[i]
            : typeof collection === "string"
              ? collection[i]
              : undefined;
        } else {
          subInputs[subflowInputName] = inputs[title];
        }
      }
      subflowInputsList.push(subInputs);
    }
    return { subflowInputsList, outputs };
  }

  private accumulateOutputs(
    outputs: Record<string, unknown[]>,
    subflowOutputs: Record<string, unknown>,
  ): void {
    for (const [outputName, outputValue] of Object.entries(subflowOutputs)) {
      const collectedOutputName = `collected_${outputName}`;
      // Not all outputs might be exposed: keep only those the node declares.
      const collected = outputs[collectedOutputName];
      if (collected !== undefined) {
        collected.push(outputValue);
      }
    }
  }

  protected async _execute(
    inputs: NodeOutputs,
    messages: BaseMessage[],
  ): Promise<ExecuteOutput> {
    const { subflowInputsList, outputs } = this.prepareIterations(inputs);
    for (const subflowInputs of subflowInputsList) {
      const subflowResult = await this.subflow.invoke({
        inputs: subflowInputs,
        messages,
      });
      const subflowOutputs = subflowResult["outputs"];
      if (isRecordLike(subflowOutputs)) {
        this.accumulateOutputs(outputs, subflowOutputs);
      }
    }
    return [outputs, {}];
  }
}
