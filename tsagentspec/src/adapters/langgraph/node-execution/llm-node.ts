/**
 * LlmNode executor for the LangGraph adapter.
 *
 * Port of `pyagentspec.adapters.langgraph._node_execution.LlmNodeExecutor`.
 * Runtime contracts (output naming, error-message text) mirror the Python
 * adapter exactly so specs behave the same across both SDKs.
 */
import type { BaseMessage } from "@langchain/core/messages";
import type { LlmNode } from "../../../flows/index.js";
import type { Property } from "../../../property.js";
import { isRecordLike, renderTemplate } from "../../common/index.js";
import type { ExecuteOutput, NodeOutputs } from "../types.js";
import { NodeExecutor } from "./executor.js";

/** The chat-model surface the LlmNodeExecutor relies on. */
interface ChatModelLike {
  invoke(input: unknown, config?: unknown): Promise<unknown>;
  withStructuredOutput?(schema: Record<string, unknown>): {
    invoke(input: unknown, config?: unknown): Promise<unknown>;
  };
}

/**
 * Executes an LlmNode: renders the prompt template against the inputs and
 * invokes the chat model, using structured output whenever the declared
 * outputs are anything but a single string.
 */
export class LlmNodeExecutor extends NodeExecutor<LlmNode> {
  private readonly llm: ChatModelLike;
  /** Present exactly when the declared outputs require structured generation. */
  private readonly structuredLlm:
    | { invoke(input: unknown, config?: unknown): Promise<unknown> }
    | undefined;

  constructor(node: LlmNode, llm: unknown) {
    super(node);
    if (
      typeof llm !== "object" ||
      llm === null ||
      typeof (llm as { invoke?: unknown }).invoke !== "function"
    ) {
      throw new Error("Llm can only be initialized with a BaseChatModel");
    }
    this.llm = llm as ChatModelLike;

    const nodeOutputs = node.outputs ?? [];
    const requiresStructuredGeneration = !(
      nodeOutputs.length === 1 && nodeOutputs[0]!.type === "string"
    );
    if (requiresStructuredGeneration) {
      if (typeof this.llm.withStructuredOutput !== "function") {
        throw new Error(
          "Llm can only be initialized with a BaseChatModel supporting withStructuredOutput",
        );
      }
      const jsonSchema: Record<string, unknown> = {
        // Title is required by langgraph
        title: "structured_output",
        type: "object",
        properties: Object.fromEntries(
          nodeOutputs.map((output) => [output.title, output.jsonSchema]),
        ),
      };
      this.structuredLlm = this.llm.withStructuredOutput(jsonSchema);
    } else {
      this.structuredLlm = undefined;
    }
  }

  private buildInvokeInputs(inputs: NodeOutputs): unknown[] {
    const renderedPrompt = renderTemplate(this.node.promptTemplate, inputs);
    return [{ role: "user", content: renderedPrompt }];
  }

  private formatStructuredOutput(
    nodeOutputs: Property[],
    generatedRaw: unknown,
  ): NodeOutputs {
    if (!isRecordLike(generatedRaw)) {
      throw new Error(
        `Expected structured LLM to return a dict, got ${typeof generatedRaw}`,
      );
    }
    let generatedOutput: NodeOutputs = generatedRaw;
    // LangGraph sometimes flattens a 1-property nested object; rebuild if needed
    if (
      nodeOutputs.length === 1 &&
      nodeOutputs[0]!.title !== Object.keys(generatedOutput)[0]
    ) {
      generatedOutput = { [nodeOutputs[0]!.title]: generatedOutput };
    }
    return generatedOutput;
  }

  private formatUnstructuredOutput(
    nodeOutputs: Property[],
    generatedMessage: unknown,
  ): NodeOutputs {
    const outputName =
      nodeOutputs.length > 0 ? nodeOutputs[0]!.title : "generated_text";
    if (
      typeof generatedMessage !== "object" ||
      generatedMessage === null ||
      !("content" in generatedMessage)
    ) {
      throw new Error(
        "generated_message should not be a dict when not doing structured generation",
      );
    }
    return {
      [outputName]: (generatedMessage as { content?: unknown }).content,
    };
  }

  protected async _execute(
    inputs: NodeOutputs,
    _messages: BaseMessage[],
  ): Promise<ExecuteOutput> {
    const invokeInputs = this.buildInvokeInputs(inputs);
    const nodeOutputs = this.node.outputs ?? [];
    if (this.structuredLlm !== undefined) {
      const generatedRaw = await this.structuredLlm.invoke(invokeInputs);
      return [this.formatStructuredOutput(nodeOutputs, generatedRaw), {}];
    }
    const generatedMessage = await this.llm.invoke(invokeInputs);
    return [this.formatUnstructuredOutput(nodeOutputs, generatedMessage), {}];
  }
}
