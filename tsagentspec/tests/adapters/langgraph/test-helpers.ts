/**
 * Shared test infrastructure for the LangGraph adapter test suite.
 *
 * Ports the mechanisms the Python tests rely on (see
 * `pyagentspec/tests/adapters/langgraph/`):
 * - `FakeToolCallingChatModel`: queued AIMessage responses consumed one per
 *   model call, with a self-returning `bindTools` so it drives `createAgent`
 *   tool loops (JS equivalent of `FakeMessagesListChatModel` + patched
 *   `bind_tools`).
 * - `FakeLlmAgentSpecLoader` / `loadWithFakeLlm`: the converter injection
 *   seam — a loader whose converter overrides the protected
 *   `convertLlmConfig` hook (JS equivalent of patching
 *   `_llm_convert_to_langgraph`), supporting one fake for all LLM configs or
 *   one per LLM config name.
 * - `installMockFetch`: global fetch stub for RemoteTool tests (JS equivalent
 *   of patching `httpx.request`).
 * - Spec builder helpers (`makeLlmConfig`, `makeAgent`) and interrupt/resume
 *   helpers matching the Python test command shapes.
 */
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import {
  BaseChatModel,
  type BaseChatModelParams,
  type BindToolsInput,
} from "@langchain/core/language_models/chat_models";
import { AIMessage, type BaseMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import { Command } from "@langchain/langgraph";
import {
  createAgent as createAgentSpecAgent,
  createVllmConfig,
} from "../../../src/index.js";
import type {
  Agent,
  ComponentBase,
  LlmConfig,
  Property,
  Tool,
  ToolBox,
  VllmConfig,
} from "../../../src/index.js";
import {
  AgentSpecLoader,
  type AgentSpecLoaderOptions,
} from "../../../src/adapters/langgraph/agentspec-loader.js";
import { AgentSpecToLangGraphConverter } from "../../../src/adapters/langgraph/langgraph-converter.js";

/**
 * Fake chat model returning queued AIMessages verbatim (tool_calls included).
 *
 * The queue index advances one message per `_generate` call and clamps on the
 * last response. `bindTools` records the bound tools and returns `this`, which
 * is what makes the fake work through `createAgent`'s binding flow.
 */
export class FakeToolCallingChatModel extends BaseChatModel {
  responses: AIMessage[];
  idx = 0;
  /** Tools bound by the agent (last `bindTools` call wins). */
  bound: BindToolsInput[] = [];
  /** The message lists received by each `_generate` call. */
  calls: BaseMessage[][] = [];

  constructor(fields: { responses: AIMessage[] } & BaseChatModelParams) {
    super(fields);
    this.responses = fields.responses;
  }

  _llmType(): string {
    return "fake-tool-calling-chat-model";
  }

  bindTools(tools: BindToolsInput[]): this {
    this.bound = tools;
    return this;
  }

  async _generate(
    messages: BaseMessage[],
    _options?: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    if (this.responses.length === 0) {
      throw new Error("FakeToolCallingChatModel has no queued responses.");
    }
    this.calls.push(messages);
    const message = this.responses[Math.min(this.idx, this.responses.length - 1)]!;
    this.idx += 1;
    return {
      generations: [
        {
          text: typeof message.content === "string" ? message.content : "",
          message,
        },
      ],
      llmOutput: {},
    };
  }
}

/** Build an AIMessage carrying a single tool call. */
export function toolCallMessage(
  name: string,
  args: Record<string, unknown>,
  id = "call_1",
): AIMessage {
  return new AIMessage({
    content: "",
    tool_calls: [{ name, args, id, type: "tool_call" }],
  });
}

/** Default LLM config used by the spec builder helpers. */
export function makeLlmConfig(overrides?: {
  name?: string;
  url?: string;
  modelId?: string;
  id?: string;
}): VllmConfig {
  return createVllmConfig({
    name: overrides?.name ?? "test-llm",
    url: overrides?.url ?? "http://localhost:8000",
    modelId: overrides?.modelId ?? "fake-model",
    ...(overrides?.id !== undefined ? { id: overrides.id } : {}),
  });
}

/** Build an Agent Spec Agent with sensible defaults for loader tests. */
export function makeAgent(overrides?: {
  name?: string;
  systemPrompt?: string;
  llmConfig?: LlmConfig;
  tools?: Tool[];
  toolboxes?: ToolBox[];
  inputs?: Property[];
  outputs?: Property[];
}): Agent {
  return createAgentSpecAgent({
    name: overrides?.name ?? "test_agent",
    systemPrompt: overrides?.systemPrompt ?? "You are a helpful agent.",
    llmConfig: overrides?.llmConfig ?? makeLlmConfig(),
    ...(overrides?.tools !== undefined ? { tools: overrides.tools } : {}),
    ...(overrides?.toolboxes !== undefined
      ? { toolboxes: overrides.toolboxes }
      : {}),
    ...(overrides?.inputs !== undefined ? { inputs: overrides.inputs } : {}),
    ...(overrides?.outputs !== undefined ? { outputs: overrides.outputs } : {}),
  });
}

/**
 * Fake responses source for `FakeLlmAgentSpecLoader`:
 * - a single response queue shared by every LLM config,
 * - a per-LLM-config-name map of response queues,
 * - or a factory receiving the LLM config and returning any chat model.
 */
export type FakeLlmResponses =
  | AIMessage[]
  | Record<string, AIMessage[]>
  | ((llmConfig: LlmConfig) => unknown);

/** Converter whose protected LLM hook delegates to the fake loader. */
class FakeLlmConverter extends AgentSpecToLangGraphConverter {
  constructor(private readonly loader: FakeLlmAgentSpecLoader) {
    super();
  }

  protected override async convertLlmConfig(
    llmConfig: LlmConfig,
  ): Promise<unknown> {
    return this.loader.resolveFakeModel(llmConfig);
  }
}

/**
 * AgentSpecLoader whose converter substitutes fake chat models for every LLM
 * config — the TS equivalent of patching `_llm_convert_to_langgraph` in the
 * Python tests. Fakes are cached per LLM config name for later inspection.
 */
export class FakeLlmAgentSpecLoader extends AgentSpecLoader {
  /** Fake models created so far, keyed by LLM config name. */
  readonly fakeModels = new Map<string, FakeToolCallingChatModel>();
  /** Every LLM config routed through the conversion seam, in order. */
  readonly convertedLlmConfigs: LlmConfig[] = [];
  private readonly responses: FakeLlmResponses;

  constructor(responses: FakeLlmResponses, options?: AgentSpecLoaderOptions) {
    super(options);
    this.responses = responses;
  }

  override get agentspecToRuntimeConverter(): AgentSpecToLangGraphConverter {
    return new FakeLlmConverter(this);
  }

  /** Resolve (creating and caching if needed) the fake model for a config. */
  resolveFakeModel(llmConfig: LlmConfig): unknown {
    this.convertedLlmConfigs.push(llmConfig);
    if (typeof this.responses === "function") {
      return this.responses(llmConfig);
    }
    const responseQueue = Array.isArray(this.responses)
      ? this.responses
      : this.responses[llmConfig.name];
    if (responseQueue === undefined) {
      throw new Error(
        `No fake responses configured for LLM config '${llmConfig.name}'.`,
      );
    }
    let model = this.fakeModels.get(llmConfig.name);
    if (model === undefined) {
      model = new FakeToolCallingChatModel({ responses: responseQueue });
      this.fakeModels.set(llmConfig.name, model);
    }
    return model;
  }

  /** The single created fake model, or the one for the given config name. */
  getFakeModel(llmName?: string): FakeToolCallingChatModel {
    if (llmName !== undefined) {
      const model = this.fakeModels.get(llmName);
      if (model === undefined) {
        throw new Error(`No fake model was created for LLM config '${llmName}'.`);
      }
      return model;
    }
    const models = [...this.fakeModels.values()];
    if (models.length !== 1) {
      throw new Error(
        `Expected exactly one fake model, found ${models.length}. Pass the LLM config name.`,
      );
    }
    return models[0]!;
  }
}

/** Structural surface of a loaded langchain ReactAgent used by the tests. */
export interface LoadedReactAgent {
  options: {
    name?: string;
    systemPrompt?: string;
    middleware?: unknown[];
    responseFormat?: unknown;
    tools?: unknown[];
    [key: string]: unknown;
  };
  graph: {
    lg_is_pregel?: boolean;
    name?: string;
    getName(): string;
    builder: {
      nodes: Record<string, unknown>;
      channels: Record<string, unknown>;
    };
  };
  invoke(input: unknown, config?: unknown): Promise<Record<string, unknown>>;
}

/**
 * Load an in-memory Agent Spec component with fake LLMs injected at the
 * converter seam. Returns the loaded runtime object (typed as a react agent
 * for convenience) together with the loader for fake-model inspection.
 */
export async function loadWithFakeLlm(
  spec: ComponentBase,
  responses: FakeLlmResponses,
  options?: AgentSpecLoaderOptions,
): Promise<{ agent: LoadedReactAgent; loader: FakeLlmAgentSpecLoader }> {
  const loader = new FakeLlmAgentSpecLoader(responses, options);
  const agent = (await loader.loadComponent(spec)) as LoadedReactAgent;
  return { agent, loader };
}

/** One recorded call observed by the mock fetch installed by `installMockFetch`. */
export interface RecordedFetchCall {
  url: string;
  init: RequestInit | undefined;
}

/** Controller returned by `installMockFetch`. */
export interface MockFetchController {
  /** The calls received so far, in order. */
  calls: RecordedFetchCall[];
  /** Restore the original global fetch. */
  restore(): void;
}

/**
 * Replace `globalThis.fetch` with a recording mock for RemoteTool tests.
 *
 * The handler receives the URL and request init; it may return a `Response`
 * directly, or any JSON-able value (sync or async) which is wrapped in a 200
 * JSON response. Always call `restore()` (e.g. in `afterEach`/`finally`).
 */
export function installMockFetch(
  handler: (url: string, init?: RequestInit) => unknown,
): MockFetchController {
  const originalFetch = globalThis.fetch;
  const calls: RecordedFetchCall[] = [];
  const mockedFetch = async (
    input: string | URL | Request,
    init?: RequestInit,
  ): Promise<Response> => {
    const url =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.toString()
          : input.url;
    calls.push({ url, init });
    const result = await handler(url, init);
    if (result instanceof Response) {
      return result;
    }
    return new Response(JSON.stringify(result), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };
  globalThis.fetch = mockedFetch as typeof fetch;
  return {
    calls,
    restore(): void {
      globalThis.fetch = originalFetch;
    },
  };
}

/** RunnableConfig pinning a thread id (checkpointed runs). */
export function threadConfig(
  threadId: string,
): { configurable: { thread_id: string } } {
  return { configurable: { thread_id: threadId } };
}

/** The `__interrupt__` entries of an invoke result (empty when none). */
export function getInterrupts(
  result: Record<string, unknown>,
): Array<{ id?: string; value?: unknown }> {
  return (
    (result["__interrupt__"] as Array<{ id?: string; value?: unknown }>) ?? []
  );
}

/** Resume command approving a pending tool confirmation. */
export function approveCommand(): Command {
  return new Command({ resume: { decisions: [{ type: "approve" }] } });
}

/** Resume command rejecting a pending tool confirmation. */
export function rejectCommand(reason?: string): Command {
  return new Command({
    resume: {
      decisions: [
        { type: "reject", ...(reason !== undefined ? { reason } : {}) },
      ],
    },
  });
}
