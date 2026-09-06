/**
 * AgentNode flow execution tests for the LangGraph adapter.
 *
 * Mirrors `pyagentspec/tests/adapters/langgraph/flows/test_agentnode.py` with
 * fake chat models injected at the converter seam; all tests run offline.
 */
import { describe, expect, it } from "vitest";
import { AIMessage } from "@langchain/core/messages";
import {
  createAgentNode,
  createFlow,
  stringProperty,
  type Flow,
  type LlmConfig,
} from "../../../../src/index.js";
import { AgentSpecToLangGraphConverter } from "../../../../src/adapters/langgraph/langgraph-converter.js";
import {
  FakeToolCallingChatModel,
  ctrl,
  dataEdge,
  ioEndNode,
  ioStartNode,
  loadWithFakeLlm,
  makeAgent,
  messagesOf,
  outputsOf,
  toolCallMessage,
  type CompiledFlow,
} from "../test-helpers.js";

describe("AgentNode in a flow", () => {
  const nationality = stringProperty({ title: "nationality" });
  const car = stringProperty({ title: "car" });

  function buildAgentFlow(): Flow {
    const agentSpec = makeAgent({
      name: "agent",
      systemPrompt: "What is the fastest {{nationality}} car?",
      inputs: [nationality],
      outputs: [car],
    });
    const agentNode = createAgentNode({ name: "agent_node", agent: agentSpec });
    const start = ioStartNode("start", [nationality]);
    const end = ioEndNode("end", [car]);
    return createFlow({
      name: "flow",
      startNode: start,
      nodes: [start, agentNode, end],
      controlFlowConnections: [ctrl(start, agentNode), ctrl(agentNode, end)],
      dataFlowConnections: [
        dataEdge(start, agentNode, "nationality"),
        dataEdge(agentNode, end, "car"),
      ],
      outputs: [car],
    });
  }

  it("renders the system prompt from node inputs and extracts declared outputs", async () => {
    const { agent, loader } = await loadWithFakeLlm(buildAgentFlow(), [
      toolCallMessage("AgentOutputModel", { car: "Ferrari 296" }),
    ]);

    const result = await agent.invoke({ inputs: { nationality: "italian" } });
    expect(outputsOf(result)).toEqual({ car: "Ferrari 296" });

    // The compiled react agent received the RENDERED system prompt (langchain
    // v1 normalizes the prompt into a content-blocks array).
    const fakeModel = loader.getFakeModel();
    const systemMessage = fakeModel.calls[0]![0]!;
    expect(systemMessage.getType()).toBe("system");
    const systemText = JSON.stringify(systemMessage.content);
    expect(systemText).toContain("What is the fastest italian car?");
    // No placeholder survives rendering.
    expect(systemText).not.toContain("{{");
  });

  it("emits the agent's answer as an assistant message when the node declares no outputs", async () => {
    const chatAgent = makeAgent({ name: "chat_agent", systemPrompt: "Say hi." });
    const agentNode = createAgentNode({ name: "agent_node", agent: chatAgent });
    const start = ioStartNode("start");
    const end = ioEndNode("end");
    const flow = createFlow({
      name: "flow",
      startNode: start,
      nodes: [start, agentNode, end],
      controlFlowConnections: [ctrl(start, agentNode), ctrl(agentNode, end)],
    });

    const { agent } = await loadWithFakeLlm(flow, [new AIMessage("Ciao!")]);
    const result = await agent.invoke({ inputs: {} });

    const messages = messagesOf(result);
    expect(messages).toHaveLength(1);
    expect(messages[0]!.getType()).toBe("ai");
    expect(messages[0]!.content).toBe("Ciao!");
    expect(outputsOf(result)).toEqual({});
  });

  it("caches the compiled agent per rendered system prompt across invokes", async () => {
    /** Converter that counts react-agent compilations and injects a fake LLM. */
    class CountingFakeConverter extends AgentSpecToLangGraphConverter {
      compileCount = 0;

      constructor(private readonly model: unknown) {
        super();
      }

      protected override async convertLlmConfig(
        _llmConfig: LlmConfig,
      ): Promise<unknown> {
        return this.model;
      }

      protected override async createReactAgent(
        agent: unknown,
        context: unknown,
        overrides?: unknown,
      ): Promise<unknown> {
        this.compileCount += 1;
        return super.createReactAgent(
          agent as never,
          context as never,
          overrides as never,
        );
      }
    }

    const fakeModel = new FakeToolCallingChatModel({
      responses: [toolCallMessage("AgentOutputModel", { car: "Ferrari" })],
    });
    const converter = new CountingFakeConverter(fakeModel);
    const graph = (await converter.convert(buildAgentFlow(), {})) as CompiledFlow;

    // Compilation is lazy: nothing is compiled at load time.
    expect(converter.compileCount).toBe(0);

    let result = await graph.invoke({ inputs: { nationality: "italian" } });
    expect(outputsOf(result)["car"]).toBe("Ferrari");
    expect(converter.compileCount).toBe(1);

    // Same rendered prompt: the cached agent is reused.
    result = await graph.invoke({ inputs: { nationality: "italian" } });
    expect(converter.compileCount).toBe(1);

    // A different rendered prompt compiles a new agent.
    result = await graph.invoke({ inputs: { nationality: "french" } });
    expect(outputsOf(result)["car"]).toBe("Ferrari");
    expect(converter.compileCount).toBe(2);
  });
});
