/**
 * Tool conversion for the LangGraph adapter.
 *
 * Port of the tool sections of
 * `pyagentspec.adapters.langgraph._langgraphconverter`: ServerTool /
 * ClientTool / RemoteTool conversion, the confirmation-interrupt machinery,
 * and the checkpointer requirements for interrupting tools.
 *
 * Runtime contracts (interrupt payload shapes, error-message text) mirror the
 * Python adapter exactly so specs behave the same across both SDKs.
 *
 * Divergences from Python (see the adapter README):
 * - JS tools take a single input object, so there is no positional-args path:
 *   client tool interrupts always carry `inputs: { args: [], kwargs }`, and
 *   Python's "Args are not supported, please only use kwargs" branch cannot
 *   trigger.
 * - Python's sync `func` / async `coroutine` pair collapses into one function
 *   (JS is async-native).
 * - Interpolated values in mirrored error/interrupt messages are rendered with
 *   `JSON.stringify` instead of Python's `repr`.
 * - No tracing callbacks are attached (tracing is a no-op seam in v1).
 */
import type { StructuredToolInterface } from "@langchain/core/tools";
import { isStructuredTool, tool } from "@langchain/core/tools";
import type { BaseCheckpointSaver } from "@langchain/langgraph";
import { interrupt } from "@langchain/langgraph";
import type { JsonSchemaValue, Property } from "../../property.js";
import type {
  ClientTool,
  RemoteTool,
  ServerTool,
  Tool,
} from "../../tools/index.js";
import {
  buildJsonSchemaFromProperties,
  createRemoteToolFunc,
} from "../common/index.js";
import type { ToolRegistry } from "./types.js";

const ALLOWED_DECISIONS = ["approve", "reject"];

/** A tool implementation function: receives the parsed input object. */
export type ToolFunction = (input: unknown, config?: unknown) => unknown;

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Merge each declared input property's default into the tool-call input when
 * the key is absent, mirroring the default injection performed by Python's
 * pydantic argument models (langchain JS does not apply JSON-schema
 * defaults). Applied before the confirmation wrapper so confirmation and
 * client-tool interrupt payloads carry the defaults, like Python.
 */
function applyInputDefaults(
  input: Record<string, unknown>,
  properties: Property[],
): Record<string, unknown> {
  const record: Record<string, unknown> = { ...input };
  for (const property of properties) {
    if (
      property.default !== undefined &&
      !Object.hasOwn(record, property.title)
    ) {
      record[property.title] = property.default;
    }
  }
  return record;
}

/**
 * Build the JSON-schema argument schema for an AgentSpec tool from its input
 * properties (the schema is titled `${toolName}Args`, mirroring Python's
 * generated pydantic model name).
 */
export function buildArgsSchema(
  toolName: string,
  properties: Property[],
): JsonSchemaValue {
  return buildJsonSchemaFromProperties(`${toolName}Args`, properties);
}

/**
 * Validate the checkpointer requirements of a tool: tools with
 * `requiresConfirmation` and all ClientTools interrupt at runtime, which
 * requires a checkpointer.
 */
export function ensureCheckpointerAndValidToolConfig(
  agentspecTool: Tool,
  checkpointer: BaseCheckpointSaver | undefined,
): void {
  const toolName = agentspecTool.name;
  if (agentspecTool.requiresConfirmation && checkpointer == null) {
    throw new Error(
      `A Checkpointer is required for tool '${toolName}' because requires_confirmation=True`,
    );
  } else if (
    agentspecTool.componentType === "ClientTool" &&
    checkpointer == null
  ) {
    throw new Error(
      `A Checkpointer is required when using ClientTool '${toolName}'.`,
    );
  }
}

/**
 * Interrupt the graph asking the user to approve or reject a tool execution.
 *
 * The interrupt payload and the resume shape are aligned with the langchain
 * human-in-the-loop docs and mirror the Python adapter exactly. Returns a
 * `[approved, reason]` tuple.
 */
export function confirmToolUse(
  toolName: string,
  toolArguments: Record<string, unknown>,
): [boolean, string] {
  const confirmationPayload = {
    action_requests: [
      {
        name: toolName,
        arguments: toolArguments,
        description: `Tool execution pending approval\n\nTool: ${toolName}\nArgs: ${JSON.stringify(toolArguments)}`,
      },
    ],
    review_configs: [
      {
        action_name: toolName,
        allowed_decisions: ALLOWED_DECISIONS,
        description:
          'Please resume with {"decisions": [{"type": "approve"}]}  # or "reject" ' +
          'with an optional "reason" for rejected tool calls.',
      },
    ],
  };
  const response = interrupt<typeof confirmationPayload, unknown>(
    confirmationPayload,
  );
  if (!isPlainRecord(response) || !("decisions" in response)) {
    throw new Error(
      `Tool confirmation result for tool ${toolName} is not valid, should be a ` +
        `dict with a 'decisions' key, was ${JSON.stringify(response)} of type ${typeof response}.`,
    );
  }
  const decisions = response["decisions"];
  const decisionList = Array.isArray(decisions) ? decisions : [];
  if (decisionList.length !== 1) {
    throw new Error(
      `Tool confirmation result for tool ${toolName} is not valid, decisions ` +
        `should be of length 1, was of length ${decisionList.length}`,
    );
  }
  const decision: unknown = decisionList[0];
  if (
    !isPlainRecord(decision) ||
    !("type" in decision) ||
    typeof decision["type"] !== "string" ||
    !ALLOWED_DECISIONS.includes(decision["type"])
  ) {
    throw new Error(
      `Tool confirmation result for tool ${toolName} is not valid, ` +
        `decision should be in ['approve', 'reject'], was ${JSON.stringify(decision)}.`,
    );
  }
  const reason =
    decision["reason"] !== undefined
      ? String(decision["reason"])
      : "No reason was provided.";
  return [decision["type"] === "approve", reason];
}

/**
 * Wrap a tool function so that it first interrupts for confirmation (when
 * required). A rejected confirmation throws
 * `Tool '<name>' was denied by the user (reason: <reason>).`.
 */
export function confirmThen(
  func: ToolFunction,
  toolName: string,
  requiresConfirmation: boolean,
): ToolFunction {
  if (!requiresConfirmation) {
    return func;
  }
  return function confirmedToolFunction(
    input: unknown,
    config?: unknown,
  ): unknown {
    const confirmationArguments = isPlainRecord(input)
      ? input
      : { args: [input] };
    const [confirmed, reason] = confirmToolUse(toolName, confirmationArguments);
    if (!confirmed) {
      throw new Error(
        `Tool '${toolName}' was denied by the user (reason: ${reason}).`,
      );
    }
    return func(input, config);
  };
}

/**
 * Convert an AgentSpec ServerTool into a LangChain structured tool using its
 * implementation from the tool registry.
 *
 * The registry value may be a LangChain structured tool (its name,
 * description and schema are reused; it must expose a callable `func`) or a
 * plain (sync or async) function (name, description and argument schema come
 * from the AgentSpec tool). `requiresConfirmation` wraps the implementation
 * with a confirmation interrupt.
 */
export function convertServerTool(
  agentspecServerTool: ServerTool,
  toolRegistry: ToolRegistry,
): StructuredToolInterface {
  const toolName = agentspecServerTool.name;
  // Own-keys membership like Python's dict: `in` would walk the prototype
  // chain and let names like "constructor" resolve to inherited functions.
  if (!Object.hasOwn(toolRegistry, toolName)) {
    throw new Error(
      `The Agent Spec representation includes a tool '${toolName}' ` +
        `but this tool does not appear in the tool registry`,
    );
  }
  const toolObj = toolRegistry[toolName];
  const toolDescription = agentspecServerTool.description ?? "";
  const requiresConfirmation = agentspecServerTool.requiresConfirmation;

  if (isStructuredTool(toolObj as StructuredToolInterface)) {
    // A LangChain tool instance from the registry: reuse its name,
    // description and schema; wrap its implementation function. Python's
    // StructuredTool-vs-other-BaseTool split collapses here since every
    // LangChain JS tool exposes the same surface.
    const registeredTool = toolObj as StructuredToolInterface & {
      func?: unknown;
    };
    const registeredFunc = registeredTool.func;
    if (typeof registeredFunc !== "function") {
      throw new Error(
        `Unsupported tool type for '${toolName}': StructuredTool has neither func nor coroutine.`,
      );
    }
    if (registeredTool.schema == null) {
      throw new Error(
        `Unsupported tool type for '${toolName}': StructuredTool has no args_schema.`,
      );
    }
    // Bridge the calling conventions: a registered tool's `func` is invoked
    // as `(input, runManager, parentConfig)`, while the wrapper created by
    // `tool()` below invokes our function as `(input, config)`.
    const registeredCallable: ToolFunction = (input, config) =>
      (
        registeredFunc as (
          input: unknown,
          runManager?: unknown,
          parentConfig?: unknown,
        ) => unknown
      )(input, undefined, config);
    const wrapped = confirmThen(
      registeredCallable,
      toolName,
      requiresConfirmation,
    );
    return tool(wrapped as (input: unknown) => unknown, {
      name: registeredTool.name,
      description: registeredTool.description,
      schema: registeredTool.schema,
    }) as StructuredToolInterface;
  }
  if (typeof toolObj === "function") {
    const toolInputs = agentspecServerTool.inputs ?? [];
    const wrapped = confirmThen(
      toolObj as ToolFunction,
      toolName,
      requiresConfirmation,
    );
    const withDefaults: ToolFunction = (input, config) =>
      wrapped(
        isPlainRecord(input) ? applyInputDefaults(input, toolInputs) : input,
        config,
      );
    return tool(withDefaults as (input: unknown) => unknown, {
      name: toolName,
      description: toolDescription,
      schema: buildArgsSchema(toolName, toolInputs),
    }) as StructuredToolInterface;
  }
  throw new Error(
    `Unsupported tool type for '${toolName}': ${typeof toolObj}. ` +
      `Expected callable, StructuredTool, or supported BaseTool.`,
  );
}

/**
 * Convert an AgentSpec ClientTool into a LangChain structured tool whose
 * implementation interrupts the graph with a `client_tool_request` payload;
 * the resume value is returned as the tool result.
 */
export function convertClientTool(
  agentspecClientTool: ClientTool,
): StructuredToolInterface {
  const toolName = agentspecClientTool.name;
  const toolDescription = agentspecClientTool.description ?? "";
  const requiresConfirmation = agentspecClientTool.requiresConfirmation;

  const clientToolFunc = (kwargs: unknown): unknown => {
    const kwargsRecord = applyInputDefaults(
      isPlainRecord(kwargs) ? kwargs : {},
      agentspecClientTool.inputs ?? [],
    );
    if (requiresConfirmation) {
      const [confirmed, reason] = confirmToolUse(toolName, kwargsRecord);
      if (!confirmed) {
        throw new Error(
          `Tool '${toolName}' was denied by the user (reason: ${reason}).`,
        );
      }
    }
    const toolRequest = {
      type: "client_tool_request",
      name: toolName,
      description: toolDescription,
      inputs: {
        args: [] as unknown[],
        kwargs: kwargsRecord,
      },
    };
    return interrupt(toolRequest);
  };

  // Note: no tool execution callback is attached, matching Python.
  return tool(clientToolFunc, {
    name: toolName,
    description: toolDescription,
    schema: buildArgsSchema(toolName, agentspecClientTool.inputs ?? []),
  }) as StructuredToolInterface;
}

/**
 * Convert an AgentSpec RemoteTool into a LangChain structured tool wrapping
 * the shared remote-tool fetch executor, with confirmation wrapping when
 * `requiresConfirmation` is set.
 */
export function convertRemoteTool(
  agentspecRemoteTool: RemoteTool,
): StructuredToolInterface {
  const toolName = agentspecRemoteTool.name;
  const toolDescription = agentspecRemoteTool.description ?? "";
  const toolInputs = agentspecRemoteTool.inputs ?? [];
  const remoteToolFunc = createRemoteToolFunc(agentspecRemoteTool);
  const wrapped = confirmThen(
    (input: unknown) =>
      remoteToolFunc(isPlainRecord(input) ? input : {}),
    toolName,
    agentspecRemoteTool.requiresConfirmation,
  );
  const withDefaults: ToolFunction = (input, config) =>
    wrapped(
      applyInputDefaults(isPlainRecord(input) ? input : {}, toolInputs),
      config,
    );
  return tool(withDefaults as (input: unknown) => unknown, {
    name: toolName,
    description: toolDescription,
    schema: buildArgsSchema(toolName, toolInputs),
  }) as StructuredToolInterface;
}
