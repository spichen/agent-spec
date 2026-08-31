/**
 * Flow node executors for the LangGraph adapter.
 *
 * Port of `pyagentspec.adapters.langgraph._node_execution`: one executor per
 * Agent Spec flow node type, each turning the shared flow state into node
 * inputs, executing the node, and folding outputs / routing details back into
 * the state.
 *
 * Runtime contracts (state keys, branch names, interrupt payloads,
 * error-message text) mirror the Python adapter exactly so specs behave the
 * same across both SDKs.
 *
 * Divergences from Python (see the adapter README):
 * - Execution is async-only (no sync `__call__` / thread offloading).
 * - Executors never mutate the incoming state: they return updated copies
 *   with the same accumulate semantics as Python's in-place mutation.
 * - Executors receive their collaborators from the converter (converted
 *   tools, chat models, compiled subgraphs, agent compile factories) instead
 *   of importing the converter, so there are no module cycles.
 * - Node execution spans/events are not emitted (tracing is a no-op seam).
 * - JS has no tuple type: arrays map positionally onto multiple declared
 *   tool-node outputs where Python only accepts tuples.
 * - The react-agent invoke payload adds no `remaining_steps` /
 *   `structured_response` keys: the langchain JS agent state has neither
 *   channel (structured output lands in `structuredResponse`).
 */
import type { BaseMessage } from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import { addMessages, interrupt } from "@langchain/langgraph";
import type {
  AgentNode,
  ApiNode,
  BranchingNode,
  CatchExceptionNode,
  EndNode,
  FlowNode,
  InputMessageNode,
  LlmNode,
  MapNode,
  OutputMessageNode,
  StartNode,
  ToolNode,
} from "../../flows/index.js";
import {
  CAUGHT_EXCEPTION_BRANCH,
  DEFAULT_BRANCH,
  DEFAULT_INPUT_MESSAGE_OUTPUT,
  DEFAULT_NEXT_BRANCH,
} from "../../flows/index.js";
import type { DataFlowEdge } from "../../flows/index.js";
import type { Property } from "../../property.js";
import {
  fetchWithAdapterDefaults,
  maybeWarnAboutUnrestrictedTemplatedUrl,
  renderNestedObjectTemplate,
  renderTemplate,
  stringifyTemplateValue,
  validateUrlAgainstAllowList,
} from "../common/index.js";
import type {
  ExecuteOutput,
  FlowState,
  NextNodeInputs,
  NodeExecutionDetails,
  NodeOutputs,
} from "./types.js";

/** The structural surface of an Agent Spec flow node used by the executors. */
interface FlowNodeLike {
  id: string;
  name: string;
  inputs?: Property[];
  outputs?: Property[];
}

/** A compiled graph / react agent surface: everything invocable. */
interface InvocableGraph {
  invoke(
    input: unknown,
    config?: RunnableConfig,
  ): Promise<Record<string, unknown>>;
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Serialize one string the way Python's `json.dumps` does (ensure_ascii). */
function pythonJsonDumpsString(value: string): string {
  let out = '"';
  for (const ch of value) {
    const code = ch.codePointAt(0)!;
    if (ch === '"') out += '\\"';
    else if (ch === "\\") out += "\\\\";
    else if (ch === "\b") out += "\\b";
    else if (ch === "\f") out += "\\f";
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (code < 0x20 || code > 0x7e) {
      if (code > 0xffff) {
        // ensure_ascii escapes astral characters as a surrogate pair.
        const high = 0xd800 + ((code - 0x10000) >> 10);
        const low = 0xdc00 + ((code - 0x10000) & 0x3ff);
        out += `\\u${high.toString(16).padStart(4, "0")}`;
        out += `\\u${low.toString(16).padStart(4, "0")}`;
      } else {
        out += `\\u${code.toString(16).padStart(4, "0")}`;
      }
    } else out += ch;
  }
  return out + '"';
}

/**
 * Serialize a value the way Python's `json.dumps` does with its default
 * arguments: `", "` / `": "` separators, ensure_ascii `\uXXXX` escapes, and
 * `Infinity`/`-Infinity`/`NaN` literals (allow_nan). Used when casting
 * non-string values into `string`-typed properties so the resulting flow
 * state text matches the Python adapter byte-for-byte.
 */
export function pythonJsonDumps(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    if (Number.isFinite(value)) return JSON.stringify(value);
    if (value === Infinity) return "Infinity";
    if (value === -Infinity) return "-Infinity";
    return "NaN";
  }
  if (typeof value === "string") return pythonJsonDumpsString(value);
  if (Array.isArray(value)) {
    return `[${value.map((item) => pythonJsonDumps(item)).join(", ")}]`;
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, v]) => v !== undefined && typeof v !== "function")
      .map(([k, v]) => `${pythonJsonDumpsString(k)}: ${pythonJsonDumps(v)}`);
    return `{${entries.join(", ")}}`;
  }
  return JSON.stringify(value) ?? "null";
}

/** Digit run with Python's underscore separators (`1_000`, not `1__0`). */
const PY_DIGITS = String.raw`\d(?:_?\d)*`;

/** Python `int()` string grammar: optional sign + underscore-separated digits. */
const PYTHON_INT_REGEXP = new RegExp(`^[+-]?${PY_DIGITS}$`);

/** Python `float()` numeric grammar (decimal/scientific, no hex/binary/octal). */
const PYTHON_FLOAT_REGEXP = new RegExp(
  `^[+-]?(?:(?:${PY_DIGITS})?\\.${PY_DIGITS}|${PY_DIGITS}\\.?)(?:[eE][+-]?${PY_DIGITS})?$`,
);

/**
 * Parse a (trimmed) string with Python `float()` semantics: decimal and
 * scientific forms plus `inf`/`infinity`/`nan` (any case, optional sign) and
 * underscore digit separators. Returns `undefined` for anything Python's
 * `float()` rejects (hex/binary/octal literals, `1__0`, empty strings, ...).
 */
function parsePythonFloat(text: string): number | undefined {
  const unsigned = text.toLowerCase().replace(/^[+-]/, "");
  if (unsigned === "inf" || unsigned === "infinity") {
    return text.startsWith("-") ? -Infinity : Infinity;
  }
  if (unsigned === "nan") return NaN;
  if (!PYTHON_FLOAT_REGEXP.test(text)) return undefined;
  const parsed = Number(text.replace(/_/g, ""));
  return Number.isNaN(parsed) ? undefined : parsed;
}

/**
 * Cast the given values to the types declared by the properties and add
 * missing defaults, mirroring Python's `_cast_values_and_add_defaults`:
 * non-strings are `json.dumps`-serialized into `string` properties, numbers
 * become booleans, numeric strings parse into `integer`/`number` properties
 * (an unparsable integer string raises like Python's `int()`; an unparsable
 * number string is left as-is like Python's swallowed `float()` error), and
 * a property with neither value nor default raises. Values for undeclared
 * properties are dropped.
 */
export function castValuesAndAddDefaults(
  valuesDict: Record<string, unknown>,
  properties: Property[],
  nodeName: string,
): NodeOutputs {
  const resultsDict: NodeOutputs = {};
  for (const property of properties) {
    const key = property.title;
    if (Object.hasOwn(valuesDict, key)) {
      let value = valuesDict[key];
      const propertyType = property.type;
      if (propertyType === "string" && typeof value !== "string") {
        value = pythonJsonDumps(value);
      } else if (propertyType === "boolean" && typeof value === "number") {
        value = Boolean(value);
      } else if (propertyType === "integer" && typeof value === "boolean") {
        value = value ? 1 : 0;
      } else if (propertyType === "integer" && typeof value === "number") {
        value = Math.trunc(value);
      } else if (propertyType === "integer" && typeof value === "string") {
        // Python does `int(value.strip())` and re-raises for any unparsable
        // string (its error-message guard never matches `int()`'s text), so
        // an unparsable integer string aborts the flow here too.
        const trimmed = value.trim();
        if (PYTHON_INT_REGEXP.test(trimmed)) {
          value = parseInt(trimmed.replace(/_/g, ""), 10);
        } else {
          // Python raises ValueError with this exact message (repr'd value).
          throw new Error(
            `invalid literal for int() with base 10: ${JSON.stringify(trimmed)}`,
          );
        }
      } else if (propertyType === "number" && typeof value === "boolean") {
        value = value ? 1 : 0;
      } else if (propertyType === "number" && typeof value === "string") {
        // Try converting numeric strings to floats with Python `float()`
        // semantics; if the parse fails, leave the string as-is (Python
        // swallows the `could not convert string to float:` error).
        const parsed = parsePythonFloat(value.trim());
        if (parsed !== undefined) {
          value = parsed;
        }
      }
      resultsDict[key] = value;
    } else if (property.default !== undefined) {
      resultsDict[key] = property.default;
    } else {
      throw new Error(
        `Expected node \`${nodeName}\` to have a value ` +
          `for property \`${property.title}\`, but none was found.`,
      );
    }
  }
  return resultsDict;
}

/**
 * Extract the outputs of an agent invoke result for the expected output
 * properties, merging (in increasing priority) property defaults, the
 * structured response, and top-level result entries. Reads the langchain JS
 * `structuredResponse` key, falling back to Python's `structured_response`.
 */
export function extractOutputsFromInvokeResult(
  result: Record<string, unknown>,
  expectedOutputs: Property[],
): NodeOutputs {
  const outputs: NodeOutputs = {};
  for (const output of expectedOutputs) {
    if (output.default !== undefined) {
      outputs[output.title] = output.default;
    }
  }
  const structuredResponse =
    result["structuredResponse"] ?? result["structured_response"];
  if (isPlainRecord(structuredResponse)) {
    Object.assign(outputs, structuredResponse);
  }
  for (const output of expectedOutputs) {
    if (Object.hasOwn(result, output.title)) {
      outputs[output.title] = result[output.title];
    }
  }
  return outputs;
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
    const ioInputs: Record<string, unknown> = isPlainRecord(nodeInputs)
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
      const destinationInputs: Record<string, unknown> = isPlainRecord(existing)
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

/** True for a list of MCP-style content blocks (text / image / file). */
function isMcpContentBlocksList(items: unknown[]): boolean {
  // Empty lists are ambiguous; treat them as non-MCP to avoid false positives
  if (items.length === 0) {
    return false;
  }
  for (const element of items) {
    if (!isPlainRecord(element)) {
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
        isPlainRecord(toolOutput) &&
        Object.keys(toolOutput).length === 1 &&
        Object.hasOwn(toolOutput, onlyTitle)
      ) {
        mapped = toolOutput;
      } else {
        mapped = { [onlyTitle]: toolOutput };
      }
    } else if (isPlainRecord(toolOutput)) {
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

/**
 * Executes an AgentNode holding a plain Agent: renders the agent's system
 * prompt against the node inputs, compiles (and caches) a react agent per
 * rendered prompt through the converter-provided factory, and invokes it on
 * the flow messages.
 */
export class AgentNodeExecutor extends NodeExecutor<AgentNode> {
  private readonly compileAgent: (
    renderedSystemPrompt: string,
  ) => Promise<unknown>;
  protected readonly config: RunnableConfig;
  /** Compiled agents cached by rendered system prompt. */
  private readonly agentsCache = new Map<string, unknown>();

  constructor(
    node: AgentNode,
    compileAgent: (renderedSystemPrompt: string) => Promise<unknown>,
    config: RunnableConfig,
  ) {
    super(node);
    this.compileAgent = compileAgent;
    this.config = config;
  }

  private async createReactAgentWithGivenInputValues(
    inputs: NodeOutputs,
  ): Promise<InvocableGraph> {
    if (this.node.agent.componentType !== "Agent") {
      throw new Error(
        "AgentNodeExecutor can only be used with AgentSpecAgent agents",
      );
    }
    const agentComponent = this.node.agent as { systemPrompt?: unknown };
    const systemPrompt = renderTemplate(
      String(agentComponent.systemPrompt ?? ""),
      inputs,
    );
    let agent = this.agentsCache.get(systemPrompt);
    if (agent === undefined) {
      agent = await this.compileAgent(systemPrompt);
      this.agentsCache.set(systemPrompt, agent);
    }
    return agent as InvocableGraph;
  }

  /** LangGraph's agent expects at least one user message to drive execution. */
  protected withDrivingMessage(messages: BaseMessage[]): unknown[] {
    return messages.length > 0 ? messages : [{ role: "user", content: "" }];
  }

  /** Map an agent invoke result onto the node outputs (or a chat message). */
  protected formatAgentResult(result: Record<string, unknown>): ExecuteOutput {
    const nodeOutputs = this.node.outputs ?? [];
    if (nodeOutputs.length === 0) {
      const messages = Array.isArray(result["messages"])
        ? (result["messages"] as { content?: unknown }[])
        : [];
      const generatedMessage = messages[messages.length - 1];
      return [
        {},
        {
          generated_messages: [
            {
              role: "assistant",
              content: (generatedMessage?.content ?? "") as string,
            },
          ],
        },
      ];
    }
    return [extractOutputsFromInvokeResult(result, nodeOutputs), {}];
  }

  protected async _execute(
    inputs: NodeOutputs,
    messages: BaseMessage[],
  ): Promise<ExecuteOutput> {
    const agent = await this.createReactAgentWithGivenInputValues(inputs);
    const preparedInputs: Record<string, unknown> = {
      ...inputs,
      messages: this.withDrivingMessage(messages),
    };
    const result = await agent.invoke(preparedInputs, this.config);
    return this.formatAgentResult(result);
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
  private readonly requiresStructuredGeneration: boolean;
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
    this.requiresStructuredGeneration = !(
      nodeOutputs.length === 1 && nodeOutputs[0]!.type === "string"
    );
    if (this.requiresStructuredGeneration) {
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
    if (!isPlainRecord(generatedRaw)) {
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
    if (this.requiresStructuredGeneration) {
      if (this.structuredLlm === undefined) {
        throw new Error("Structured LLM was not initialized");
      }
      const generatedRaw = await this.structuredLlm.invoke(invokeInputs);
      return [this.formatStructuredOutput(nodeOutputs, generatedRaw), {}];
    }
    const generatedMessage = await this.llm.invoke(invokeInputs);
    return [this.formatUnstructuredOutput(nodeOutputs, generatedMessage), {}];
  }
}

/**
 * Executes an ApiNode: renders `{{placeholder}}` templates in the URL, data,
 * headers and query params against the inputs, performs the HTTP request and
 * returns the parsed JSON response body as the node output.
 */
export class ApiNodeExecutor extends NodeExecutor<ApiNode> {
  constructor(node: ApiNode) {
    super(node);
    // The TS SDK ApiNode has no urlAllowList field yet: the helpers are
    // invoked with `undefined` (i.e. allow), matching the documented
    // divergence, so the templated-URL warning fires per the Python rules.
    maybeWarnAboutUnrestrictedTemplatedUrl(
      node.url,
      undefined,
      `ApiNode \`${node.name}\``,
    );
  }

  private buildRequest(inputs: NodeOutputs): {
    url: string;
    init: RequestInit;
  } {
    const apiNode = this.node;
    const apiNodeData = renderNestedObjectTemplate(apiNode.data, inputs);
    const apiNodeHeaders: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(apiNode.headers)) {
      apiNodeHeaders[renderTemplate(key, inputs)] = renderNestedObjectTemplate(
        value,
        inputs,
      );
    }
    const apiNodeQueryParams: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(apiNode.queryParams)) {
      apiNodeQueryParams[renderTemplate(key, inputs)] =
        renderNestedObjectTemplate(value, inputs);
    }
    const apiNodeUrl = renderTemplate(apiNode.url, inputs);

    const contentTypeHeader =
      apiNodeHeaders["Content-Type"] ?? apiNodeHeaders["content-type"];
    const expectUrlencodedFormData =
      typeof contentTypeHeader === "string" &&
      contentTypeHeader.includes("application/x-www-form-urlencoded");

    const requestHeaders: Record<string, string> = {};
    for (const [key, value] of Object.entries(apiNodeHeaders)) {
      requestHeaders[key] =
        typeof value === "string" ? value : stringifyTemplateValue(value);
    }
    const callerSetContentType = Object.keys(requestHeaders).some(
      (key) => key.toLowerCase() === "content-type",
    );

    const method = apiNode.httpMethod;
    const methodUpper = method.toUpperCase();
    // fetch forbids request bodies on GET/HEAD (Python's httpx sends them).
    const methodAllowsBody = methodUpper !== "GET" && methodUpper !== "HEAD";
    if (!methodAllowsBody) {
      const hasDeclaredBody =
        apiNodeData !== undefined &&
        apiNodeData !== null &&
        apiNodeData !== "" &&
        !(isPlainRecord(apiNodeData) && Object.keys(apiNodeData).length === 0);
      if (hasDeclaredBody) {
        // Forced divergence from Python: warn instead of silently dropping.
        console.warn(
          `ApiNode \`${apiNode.name}\` declares request data for HTTP method ` +
            `${methodUpper}, but fetch forbids request bodies on GET/HEAD: ` +
            `the declared body is not sent (the Python adapter sends it).`,
        );
      }
    }

    let body: string | URLSearchParams | Uint8Array | undefined;
    if (methodAllowsBody) {
      if (expectUrlencodedFormData && isPlainRecord(apiNodeData)) {
        const form = new URLSearchParams();
        for (const [key, value] of Object.entries(apiNodeData)) {
          form.append(
            key,
            typeof value === "string" ? value : stringifyTemplateValue(value),
          );
        }
        body = form;
      } else if (typeof apiNodeData === "string") {
        body = apiNodeData;
      } else if (apiNodeData instanceof Uint8Array) {
        body = apiNodeData;
      } else if (apiNodeData !== undefined && apiNodeData !== null) {
        body = JSON.stringify(apiNodeData);
        if (!callerSetContentType) {
          requestHeaders["Content-Type"] = "application/json";
        }
      }
    }

    // Kept as the seam for allow-list enforcement: the TS SDK ApiNode has no
    // urlAllowList field yet, so this always allows.
    validateUrlAgainstAllowList(apiNodeUrl, undefined);

    const searchParams = new URLSearchParams();
    for (const [key, value] of Object.entries(apiNodeQueryParams)) {
      if (Array.isArray(value)) {
        for (const item of value) {
          searchParams.append(
            key,
            item == null ? "" : stringifyTemplateValue(item),
          );
        }
      } else {
        searchParams.append(
          key,
          value == null ? "" : stringifyTemplateValue(value),
        );
      }
    }
    const query = searchParams.toString();
    const requestUrl =
      query.length > 0
        ? `${apiNodeUrl}${apiNodeUrl.includes("?") ? "&" : "?"}${query}`
        : apiNodeUrl;

    return {
      url: requestUrl,
      init: {
        method,
        headers: requestHeaders,
        ...(body !== undefined ? { body } : {}),
      },
    };
  }

  protected async _execute(
    inputs: NodeOutputs,
    _messages: BaseMessage[],
  ): Promise<ExecuteOutput> {
    const { url, init } = this.buildRequest(inputs);
    // Redirects are not followed and the request times out after the shared
    // default, matching Python's httpx defaults (see fetchWithAdapterDefaults).
    const response = await fetchWithAdapterDefaults(
      url,
      init,
      `ApiNode \`${this.node.name}\``,
    );
    // Python parses the JSON body regardless of the HTTP status (a 3xx
    // response returned without following included).
    const responseJson = (await response.json()) as unknown;
    return [responseJson as NodeOutputs, {}];
  }
}

/**
 * Executes a FlowNode: invokes the compiled subflow with this node's inputs
 * and messages; the subflow's outputs become the node outputs and its
 * terminating EndNode branch propagates as this node's branch.
 */
export class FlowNodeExecutor extends NodeExecutor<FlowNode> {
  private readonly subflow: InvocableGraph;
  private readonly config: RunnableConfig;

  constructor(node: FlowNode, subflow: unknown, config: RunnableConfig) {
    super(node);
    this.subflow = subflow as InvocableGraph;
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
    subflow: unknown,
    config: RunnableConfig,
  ) {
    super(node);
    this.subflow = subflow as InvocableGraph;
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
      const outputs: NodeOutputs = isPlainRecord(flowOutput["outputs"])
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

  constructor(node: MapNode, subflow: unknown, _config: RunnableConfig) {
    super(node);
    if (!node.inputs || node.inputs.length === 0) {
      throw new Error("MapNode has no inputs");
    }
    // Mirroring Python, the subflow runs are not passed the ambient config.
    this.subflow = subflow as InvocableGraph;
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
        throw new Error(
          `Found inputs to iterate with different sizes (${stringifyTemplateValue(iterable)} and ${String(numInputsToIterate)})`,
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
      if (isPlainRecord(subflowOutputs)) {
        this.accumulateOutputs(outputs, subflowOutputs);
      }
    }
    return [outputs, {}];
  }
}
