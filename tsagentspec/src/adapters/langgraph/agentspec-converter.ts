/**
 * LangGraph -> Agent Spec converter.
 *
 * Port of `pyagentspec.adapters.langgraph._agentspecconverter
 * .LangGraphToAgentSpecConverter`: LangChain structured tools become
 * ServerTools, chat models become LLM configs, langchain react agents become
 * Agents, and any other StateGraph becomes a Flow.
 *
 * Divergences from Python (see the adapter README) — Python leans on CPython
 * closure introspection that has no JS equivalent:
 * - React agents export from the langchain `ReactAgent` instance (its public
 *   `options` retains model / systemPrompt / tools). A bare compiled agent
 *   graph does NOT retain them (private fields) and is rejected with a clear
 *   error instead of Python's closure digging.
 * - Swarm graphs are rejected: the compiled per-agent graphs do not retain
 *   their models/prompts, so a faithful Swarm export is unreachable in JS.
 * - MCP tools load as ServerTools: the MCP connection lives in a JS closure
 *   that cannot be introspected, so Python's MCPTool recovery is skipped.
 * - The TS SDK LlmConfigs have no `retryPolicy`, so ChatOpenAI retry/timeout
 *   settings are not exported.
 * - OciGenAiConfig export is not supported (no langchain-oci JS package).
 */
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { StructuredToolInterface } from "@langchain/core/tools";
import { isStructuredTool } from "@langchain/core/tools";
import { toJsonSchema } from "@langchain/core/utils/json_schema";
import type { Agent } from "../../agents/index.js";
import { createAgent as createAgentSpecAgent } from "../../agents/index.js";
import type { ComponentBase } from "../../component.js";
import type { LlmConfig } from "../../llms/index.js";
import {
  OpenAIAPIType,
  createOllamaConfig,
  createOpenAiCompatibleConfig,
  createOpenAiConfig,
} from "../../llms/index.js";
import type { JsonSchemaValue, Property } from "../../property.js";
import type { Tool } from "../../tools/index.js";
import { createServerTool } from "../../tools/index.js";
import type { RuntimeToAgentSpecConverter } from "../common/index.js";
import { isRecordLike } from "../common/index.js";
import { langgraphGraphConvertToAgentSpec } from "./agentspec-converter-flow.js";
import {
  getGraphBuilder,
  isCompiledGraphLike,
  isStateGraphLike,
  stateSchemaKeys,
} from "./graph-introspection.js";

/**
 * True for a LangChain structured tool. `isStructuredTool` alone only tests
 * `lc_namespace` (which every LC serializable carries), so the check is
 * strengthened with the tool surface: a string name, a schema and `invoke`.
 */
function isLangChainStructuredTool(
  value: unknown,
): value is StructuredToolInterface {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as {
    name?: unknown;
    schema?: unknown;
    invoke?: unknown;
  };
  return (
    isStructuredTool(value as StructuredToolInterface) &&
    typeof candidate.name === "string" &&
    "schema" in candidate &&
    typeof candidate.invoke === "function"
  );
}

// JS has no `id()`: emulate stable per-object identities with a WeakMap.
const objectIdentities = new WeakMap<object, number>();
let nextObjectIdentity = 1;

/**
 * Runtime-object identity key used for exporter memoization, mirroring
 * Python's `_get_obj_reference` (`<classname>/<id>`).
 */
function getObjectReference(runtimeComponent: unknown): string {
  if (
    (typeof runtimeComponent !== "object" || runtimeComponent === null) &&
    typeof runtimeComponent !== "function"
  ) {
    return `${typeof runtimeComponent}/${String(runtimeComponent)}`;
  }
  const target = runtimeComponent as object;
  let identity = objectIdentities.get(target);
  if (identity === undefined) {
    identity = nextObjectIdentity;
    nextObjectIdentity += 1;
    objectIdentities.set(target, identity);
  }
  const constructorName =
    (target as { constructor?: { name?: string } }).constructor?.name ??
    "object";
  return `${constructorName.toLowerCase()}/${identity}`;
}

/** The `ReactAgent` surface used for export (langchain `createAgent` result). */
interface ReactAgentLike {
  options: Record<string, unknown>;
  graph?: { name?: unknown };
}

/** True for a langchain `ReactAgent` instance (the `createAgent` result). */
function isReactAgentInstance(value: unknown): value is ReactAgentLike {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as {
    constructor?: { name?: string };
    options?: unknown;
    graph?: unknown;
  };
  if (!isRecordLike(candidate.options)) {
    return false;
  }
  if (candidate.constructor?.name === "ReactAgent") {
    return true;
  }
  const graph = candidate.graph as { invoke?: unknown } | undefined;
  return typeof graph?.invoke === "function";
}

/** True for a graph compiled by langchain's `createAgent` (fingerprint). */
function isReactAgentGraph(value: unknown): boolean {
  if (!isStateGraphLike(value)) {
    return false;
  }
  const builder = getGraphBuilder(value);
  return isRecordLike(builder.nodes) && "model_request" in builder.nodes;
}

/** True for a graph built by `@langchain/langgraph-swarm`'s `createSwarm`. */
function isSwarmGraph(value: unknown): boolean {
  if (!isStateGraphLike(value)) {
    return false;
  }
  const builder = getGraphBuilder(value);
  if (!(stateSchemaKeys(builder) ?? []).includes("activeAgent")) {
    return false;
  }
  const startBranches = builder.branches?.["__start__"];
  if (!isRecordLike(startBranches)) {
    return false;
  }
  const nodeSpecs = Object.values(builder.nodes ?? {});
  return (
    nodeSpecs.length > 0 &&
    nodeSpecs.every((spec) => isCompiledGraphLike(spec.runnable))
  );
}

/** Build an Agent Spec property from one tool-argument JSON schema. */
function toolArgumentProperty(
  argumentTitle: string,
  argumentSchema: JsonSchemaValue,
): Property {
  const jsonSchema: JsonSchemaValue = { ...argumentSchema };
  if (
    typeof jsonSchema["title"] !== "string" ||
    (jsonSchema["title"] as string).length === 0
  ) {
    jsonSchema["title"] = argumentTitle;
  }
  if (!("type" in jsonSchema) && !("anyOf" in jsonSchema)) {
    // Python exports an untyped Property here (its Property model accepts a
    // bare `{title}` schema); the TS SDK Property requires `type` or
    // `anyOf`, so synthesize the closest representable "any" union instead
    // of failing the whole export.
    jsonSchema["anyOf"] = [
      { type: "object" },
      { type: "array" },
      { type: "string" },
      { type: "number" },
      { type: "integer" },
      { type: "boolean" },
      { type: "null" },
    ];
  }
  return {
    jsonSchema,
    title: argumentTitle,
    description: jsonSchema["description"] as string | undefined,
    default: jsonSchema["default"],
    type: jsonSchema["type"] as string | string[] | undefined,
  };
}

/**
 * Convert LangGraph/LangChain runtime components into Agent Spec components.
 *
 * Shared runtime objects (the same model or tool instance used twice) become
 * single referenced components via the `referencedObjects` memoization map.
 */
export class LangGraphToAgentSpecConverter
  implements RuntimeToAgentSpecConverter
{
  /** Convert the given LangGraph object into an Agent Spec component. */
  convert(
    runtimeComponent: unknown,
    referencedObjects?: Map<string, ComponentBase>,
  ): ComponentBase {
    const registry = referencedObjects ?? new Map<string, ComponentBase>();
    // Reuse the same object multiple times to exploit the referencing system.
    const objectReference = getObjectReference(runtimeComponent);
    const cached = registry.get(objectReference);
    if (cached !== undefined) {
      return cached;
    }
    const converted = this._convert(runtimeComponent, registry);
    registry.set(objectReference, converted);
    return converted;
  }

  /** Dispatch one (uncached) conversion on the runtime component's shape. */
  protected _convert(
    langgraphComponent: unknown,
    referencedObjects: Map<string, ComponentBase>,
  ): ComponentBase {
    // The chat-model check comes first: langchain's `isStructuredTool` only
    // tests `lc_namespace`, which every LC serializable (models included)
    // carries.
    if (langgraphComponent instanceof BaseChatModel) {
      return this.baseChatModelConvertToAgentSpec(langgraphComponent);
    }
    if (isLangChainStructuredTool(langgraphComponent)) {
      return this.langgraphAnyToolToAgentSpecTool(langgraphComponent);
    }
    if (isReactAgentInstance(langgraphComponent)) {
      return this.reactAgentConvertToAgentSpec(
        langgraphComponent,
        referencedObjects,
      );
    }
    if (isSwarmGraph(langgraphComponent)) {
      throw new Error(
        "Exporting a LangGraph swarm is not supported by the TypeScript " +
          "adapter: the compiled per-agent graphs do not retain their chat " +
          "model or system prompt.",
      );
    }
    if (isReactAgentGraph(langgraphComponent)) {
      throw new Error(
        "Exporting a compiled agent graph is not supported by the TypeScript " +
          "adapter: the compiled graph does not retain its chat model or " +
          "system prompt. Export the langchain ReactAgent instance (the " +
          "createAgent result) instead.",
      );
    }
    if (isStateGraphLike(langgraphComponent)) {
      return langgraphGraphConvertToAgentSpec(
        this,
        langgraphComponent,
        referencedObjects,
      );
    }
    throw new Error(
      `Conversion for ${String(langgraphComponent)} not implemented yet`,
    );
  }

  /**
   * Convert a LangChain structured tool into an Agent Spec ServerTool.
   *
   * Python additionally recovers MCPTool transports from the tool coroutine's
   * closure; that is unreachable in JS, so MCP-loaded tools export as plain
   * ServerTools (documented divergence).
   */
  protected langgraphAnyToolToAgentSpecTool(
    tool: StructuredToolInterface,
  ): ComponentBase {
    const toolSchema = toJsonSchema(
      (tool as { schema: Parameters<typeof toJsonSchema>[0] }).schema,
    ) as JsonSchemaValue;
    const argumentSchemas = isRecordLike(toolSchema["properties"])
      ? (toolSchema["properties"] as Record<string, JsonSchemaValue>)
      : {};
    const inputs = Object.entries(argumentSchemas).map(
      ([argumentTitle, argumentSchema]) =>
        toolArgumentProperty(argumentTitle, argumentSchema),
    );
    return createServerTool({
      name: tool.name,
      description: tool.description ?? "",
      inputs,
    });
  }

  /** Convert a LangChain chat model into the closest Agent Spec LLM config. */
  protected baseChatModelConvertToAgentSpec(model: BaseChatModel): LlmConfig {
    const llmType =
      typeof (model as unknown as { _llmType?: () => string })._llmType ===
      "function"
        ? (model as unknown as { _llmType: () => string })._llmType()
        : "";
    const constructorName = model.constructor?.name ?? "";

    if (llmType === "ollama" || constructorName === "ChatOllama") {
      const ollamaModel = model as unknown as {
        model?: string;
        baseUrl?: string;
      };
      const modelId = ollamaModel.model ?? "";
      return createOllamaConfig({
        name: modelId,
        url: ollamaModel.baseUrl ?? "",
        modelId,
      });
    }
    if (llmType === "openai" || constructorName === "ChatOpenAI") {
      const openAiModel = model as unknown as {
        model?: string;
        useResponsesApi?: boolean;
        clientConfig?: { baseURL?: string };
        fields?: { configuration?: { baseURL?: string } };
      };
      const modelName = openAiModel.model ?? "";
      const apiType = openAiModel.useResponsesApi
        ? OpenAIAPIType.RESPONSES
        : OpenAIAPIType.CHAT_COMPLETIONS;
      const baseUrl =
        openAiModel.clientConfig?.baseURL ??
        openAiModel.fields?.configuration?.baseURL ??
        "";
      // Note: the TS SDK LlmConfigs have no retryPolicy, so ChatOpenAI
      // maxRetries/timeout are not exported (documented divergence).
      if (baseUrl.startsWith("https://api.openai.com")) {
        return createOpenAiConfig({
          name: modelName,
          modelId: modelName,
          apiType,
        });
      }
      return createOpenAiCompatibleConfig({
        name: modelName,
        url: baseUrl,
        modelId: modelName,
        apiType,
      });
    }
    throw new Error(
      `The LLM instance provided is of an unsupported type \`${constructorName || llmType}\`.`,
    );
  }

  /**
   * Convert a langchain `ReactAgent` into an Agent Spec Agent using its
   * public `options` (the JS-native replacement for Python's closure
   * introspection on the compiled graph).
   */
  protected reactAgentConvertToAgentSpec(
    reactAgent: ReactAgentLike,
    referencedObjects: Map<string, ComponentBase>,
  ): Agent {
    const options = reactAgent.options;
    const optionsName = options["name"];
    const graphName = reactAgent.graph?.name;
    const agentName =
      typeof optionsName === "string" && optionsName.length > 0
        ? optionsName
        : typeof graphName === "string" &&
            graphName.length > 0 &&
            graphName !== "LangGraph"
          ? graphName
          : "LangGraph Agent";

    const model = options["model"];
    if (typeof model === "string") {
      throw new Error(
        "Exporting an agent created from a model identifier string is not " +
          "supported; pass a chat model instance to createAgent instead.",
      );
    }
    const llmConfig = this.convert(model, referencedObjects) as LlmConfig;

    const systemPromptRaw = options["systemPrompt"];
    let systemPrompt = "";
    if (typeof systemPromptRaw === "string") {
      systemPrompt = systemPromptRaw;
    } else if (
      isRecordLike(systemPromptRaw) ||
      (typeof systemPromptRaw === "object" && systemPromptRaw !== null)
    ) {
      const content = (systemPromptRaw as { content?: unknown }).content;
      systemPrompt = content === undefined ? "" : String(content);
    }

    const optionTools = Array.isArray(options["tools"])
      ? (options["tools"] as unknown[])
      : [];
    const tools = optionTools.map(
      (langgraphTool) =>
        this.convert(langgraphTool, referencedObjects) as Tool,
    );

    return createAgentSpecAgent({
      name: agentName,
      llmConfig,
      systemPrompt,
      tools,
    });
  }
}
