/**
 * ToolNode executor for the LangGraph adapter.
 *
 * Port of `pyagentspec.adapters.langgraph._node_execution.ToolNodeExecutor`.
 * Runtime contracts (output mapping, error-message text) mirror the Python
 * adapter exactly so specs behave the same across both SDKs.
 *
 * Divergence from Python (see the adapter README): JS has no tuple type, so
 * arrays map positionally onto multiple declared tool-node outputs where
 * Python only accepts tuples.
 */
import type { BaseMessage } from "@langchain/core/messages";
import type { ToolNode } from "../../../flows/index.js";
import { isRecordLike, stringifyTemplateValue } from "../../common/index.js";
import type { ExecuteOutput, InvocableGraph, NodeOutputs } from "../types.js";
import { NodeExecutor } from "./executor.js";

/** True for a list of MCP-style content blocks (text / image / file). */
function isMcpContentBlocksList(items: unknown[]): boolean {
  // Empty lists are ambiguous; treat them as non-MCP to avoid false positives
  if (items.length === 0) {
    return false;
  }
  for (const element of items) {
    if (!isRecordLike(element)) {
      return false;
    }
    const blockType = element["type"];
    if (blockType !== "text" && blockType !== "image" && blockType !== "file") {
      return false;
    }
    if (blockType === "text") {
      if (typeof element["text"] !== "string") {
        return false;
      }
    } else if (
      !("base64" in element) &&
      !("url" in element) &&
      !("file_id" in element)
    ) {
      return false;
    }
  }
  return true;
}

/** Extract the payload of one MCP content block. */
function extractValueFromContentBlock(block: Record<string, unknown>): unknown {
  const blockType = block["type"];
  if (blockType === "text") {
    return block["text"];
  }
  if (blockType === "image" || blockType === "file") {
    if ("base64" in block) {
      return block["base64"];
    }
    if ("url" in block) {
      return block["url"];
    }
    if ("file_id" in block) {
      return block["file_id"];
    }
    throw new Error(
      `No payload found in ${blockType} block: ${JSON.stringify(block)}`,
    );
  }
  throw new Error(
    `Unsupported message content block type: ${String(blockType)}`,
  );
}

/**
 * Executes a ToolNode: invokes the converted LangChain tool with the node
 * inputs and maps the raw tool output onto the node's declared output
 * properties (MCP content-block lists map positionally; dicts are filtered;
 * arrays map positionally onto multiple outputs).
 */
export class ToolNodeExecutor extends NodeExecutor<ToolNode> {
  private readonly toolCallable: InvocableGraph;

  constructor(node: ToolNode, tool: unknown) {
    super(node);
    if (
      typeof tool !== "object" ||
      tool === null ||
      typeof (tool as { invoke?: unknown }).invoke !== "function"
    ) {
      throw new Error(
        `ToolNodeExecutor expected a LangChain StructuredTool, but got ${typeof tool}.`,
      );
    }
    this.toolCallable = tool as InvocableGraph;
  }

  /** Best-effort mapping of raw tool outputs to the declared node outputs. */
  private formatToolResult(toolOutput: unknown): ExecuteOutput {
    const nodeOutputProperties = this.node.outputs ?? [];
    let mapped: NodeOutputs;
    if (Array.isArray(toolOutput) && isMcpContentBlocksList(toolOutput)) {
      const extractedValues = (toolOutput as Record<string, unknown>[]).map(
        (block) => extractValueFromContentBlock(block),
      );
      mapped = {};
      nodeOutputProperties.forEach((property, i) => {
        if (i >= extractedValues.length) {
          // Python raises a bare IndexError ("list index out of range") here.
          throw new Error(
            `Tool node \`${this.node.name}\` returned ${extractedValues.length} ` +
              `content block(s) but declares ${nodeOutputProperties.length} ` +
              `outputs; no value for output \`${property.title}\`.`,
          );
        }
        mapped[property.title] = extractedValues[i];
      });
    } else if (nodeOutputProperties.length === 1) {
      // The tool returns a dict with a single key being the node's output
      // property's title: use it as-is to avoid double-wrapping.
      const onlyTitle = nodeOutputProperties[0]!.title;
      if (
        isRecordLike(toolOutput) &&
        Object.keys(toolOutput).length === 1 &&
        Object.hasOwn(toolOutput, onlyTitle)
      ) {
        mapped = toolOutput;
      } else {
        mapped = { [onlyTitle]: toolOutput };
      }
    } else if (isRecordLike(toolOutput)) {
      // The node emits multiple outputs: filter the tool output.
      mapped = {};
      for (const property of nodeOutputProperties) {
        if (Object.hasOwn(toolOutput, property.title)) {
          mapped[property.title] = toolOutput[property.title];
        }
      }
    } else if (Array.isArray(toolOutput)) {
      // Multiple outputs from an array (Python: tuple): map positionally.
      mapped = {};
      nodeOutputProperties.forEach((property, i) => {
        if (i >= toolOutput.length) {
          // Python raises a bare IndexError ("tuple index out of range") here.
          throw new Error(
            `Tool node \`${this.node.name}\` returned ${toolOutput.length} ` +
              `value(s) but declares ${nodeOutputProperties.length} ` +
              `outputs; no value for output \`${property.title}\`.`,
          );
        }
        mapped[property.title] = toolOutput[i];
      });
    } else {
      throw new Error(
        `Unsupported multi-output mapping for tool_output: ${stringifyTemplateValue(toolOutput)}` +
          `(declared_outputs=${nodeOutputProperties.length}).`,
      );
    }
    return [mapped, {}];
  }

  protected async _execute(
    inputs: NodeOutputs,
    _messages: BaseMessage[],
  ): Promise<ExecuteOutput> {
    const toolOutput = await this.toolCallable.invoke(inputs);
    return this.formatToolResult(toolOutput);
  }
}
