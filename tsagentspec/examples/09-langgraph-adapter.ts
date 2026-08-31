/**
 * Example 9: LangGraph adapter
 *
 * Demonstrates the two directions of the LangGraph adapter:
 * - AgentSpecLoader: Agent Spec YAML -> runnable LangGraph agent
 * - AgentSpecExporter: hand-built LangGraph StateGraph -> Agent Spec YAML
 *
 * The example runs fully offline: instead of a real LLM provider it injects a
 * fake tool-calling chat model through the loader's `convertedComponents`
 * conversion seam (pre-seeded runtime components, keyed by component id).
 */
import {
  createAgent,
  createServerTool,
  createVllmConfig,
  stringProperty,
  AgentSpecSerializer,
  type ComponentBase,
} from "agentspec";
import {
  AgentSpecExporter,
  AgentSpecLoader,
  type AgentSpecLoaderOptions,
} from "agentspec/adapters/langgraph";
import {
  BaseChatModel,
  type BindToolsInput,
} from "@langchain/core/language_models/chat_models";
import { AIMessage, type BaseMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import { Annotation, StateGraph, START, END } from "@langchain/langgraph";

// =============================================
// Build an Agent spec with a ServerTool
// =============================================

const weatherTool = createServerTool({
  name: "get_weather",
  description: "Returns the current weather for a city",
  inputs: [stringProperty({ title: "city" })],
  outputs: [stringProperty({ title: "weather" })],
});

const llmConfig = createVllmConfig({
  name: "vllm-model",
  url: "http://localhost:8000",
  modelId: "llama-3-70b",
});

const agentSpec = createAgent({
  name: "weather_agent",
  llmConfig,
  systemPrompt: "You are a helpful weather assistant.",
  tools: [weatherTool],
});

// Serialize to YAML — this is the portable Agent Spec configuration.
const yaml = new AgentSpecSerializer().toYaml(agentSpec) as string;
console.log("--- Agent Spec (YAML) ---");
console.log(yaml);

// =============================================
// A minimal fake tool-calling chat model
// =============================================

// Returns queued AIMessages one per model call; `bindTools` returns `this` so
// the model flows through `createAgent`'s tool-binding. Offline stand-in for
// ChatOpenAI/ChatOllama — real runs would skip this and let the adapter build
// the chat model from the LLM config.

class FakeChatModel extends BaseChatModel {
  private index = 0;

  constructor(private readonly responses: AIMessage[]) {
    super({});
  }

  _llmType(): string {
    return "fake-chat-model";
  }

  bindTools(_tools: BindToolsInput[]): this {
    return this;
  }

  async _generate(_messages: BaseMessage[]): Promise<ChatResult> {
    const message =
      this.responses[Math.min(this.index, this.responses.length - 1)]!;
    this.index += 1;
    return { generations: [{ text: "", message }], llmOutput: {} };
  }
}

const fakeModel = new FakeChatModel([
  // First model turn: call the tool.
  new AIMessage({
    content: "",
    tool_calls: [
      {
        name: "get_weather",
        args: { city: "Agadir" },
        id: "call_1",
        type: "tool_call",
      },
    ],
  }),
  // Second model turn: final answer.
  new AIMessage("It is sunny in Agadir — enjoy the beach!"),
]);

// =============================================
// Load the YAML into a runnable LangGraph agent
// =============================================

// `convertedComponents` maps component ids to already-converted runtime
// objects; pre-seeding the LLM config's id substitutes the fake model.

class OfflineAgentSpecLoader extends AgentSpecLoader {
  constructor(
    private readonly prebuilt: ReadonlyMap<string, unknown>,
    options?: AgentSpecLoaderOptions,
  ) {
    super(options);
  }

  override async loadComponent(component: ComponentBase): Promise<unknown> {
    this.componentLoadPolicy.validateComponentTree(component);
    return this.agentspecToRuntimeConverter.convert(
      component,
      this.toolRegistry,
      {
        convertedComponents: new Map(this.prebuilt),
        checkpointer: this.checkpointer,
        config: this.config,
      },
    );
  }
}

const loader = new OfflineAgentSpecLoader(
  new Map([[llmConfig.id, fakeModel]]),
  {
    // ServerTool implementations are looked up here by tool name.
    toolRegistry: {
      get_weather: (input: unknown) =>
        `The weather in ${(input as { city: string }).city} is sunny.`,
    },
  },
);

const agent = (await loader.loadYaml(yaml)) as {
  invoke(input: unknown): Promise<Record<string, unknown>>;
};

const result = await agent.invoke({
  messages: [{ role: "user", content: "What is the weather in Agadir?" }],
});

console.log("--- Conversation ---");
for (const message of result["messages"] as BaseMessage[]) {
  const toolCalls =
    message instanceof AIMessage ? (message.tool_calls ?? []) : [];
  const text =
    toolCalls.length > 0
      ? toolCalls
          .map((call) => `tool call: ${call.name}(${JSON.stringify(call.args)})`)
          .join(", ")
      : String(message.content);
  console.log(`[${message.getType()}] ${text}`);
}

// =============================================
// Export a hand-built StateGraph to Agent Spec
// =============================================

const SummaryState = Annotation.Root({
  text: Annotation<string>,
  summary: Annotation<string>,
});

const graph = new StateGraph(SummaryState)
  .addNode("summarize", (state: typeof SummaryState.State) => ({
    summary: `Summary of: ${state.text}`,
  }))
  .addEdge(START, "summarize")
  .addEdge("summarize", END)
  .compile({ name: "Summarizer" });

const exporter = new AgentSpecExporter();
console.log("--- Exported StateGraph (YAML) ---");
console.log(exporter.toYaml(graph) as string);
