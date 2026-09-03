/**
 * LlmNode flow execution tests for the LangGraph adapter.
 *
 * Mirrors `pyagentspec/tests/adapters/langgraph/flows/test_llmnode.py` with a
 * duck-typed chat-model fake injected at the converter seam; all tests run
 * offline.
 */
import { describe, expect, it } from "vitest";
import { AIMessage } from "@langchain/core/messages";
import {
  createFlow,
  createLlmNode,
  integerProperty,
  objectProperty,
  stringProperty,
  type Flow,
  type Property,
} from "../../../../src/index.js";
import {
  ctrl,
  dataEdge,
  ioEndNode,
  ioStartNode,
  loadWithFakeLlm,
  makeLlmConfig,
  outputsOf,
} from "../test-helpers.js";

/** Duck-typed chat-model fake for LlmNode tests. */
function makeChatModelFake(opts: {
  reply?: string;
  structured?: Record<string, unknown>;
}) {
  const captured = {
    prompts: [] as unknown[],
    structuredSchemas: [] as Record<string, unknown>[],
  };
  const model = {
    invoke: async (input: unknown) => {
      captured.prompts.push(input);
      return new AIMessage(opts.reply ?? "");
    },
    withStructuredOutput: (schema: Record<string, unknown>) => {
      captured.structuredSchemas.push(schema);
      return {
        invoke: async (input: unknown) => {
          captured.prompts.push(input);
          if (opts.structured === undefined) {
            throw new Error("No structured response configured.");
          }
          return opts.structured;
        },
      };
    },
  };
  return { model, captured };
}

describe("LlmNode", () => {
  const nationality = stringProperty({ title: "nationality" });
  const car = stringProperty({ title: "car" });

  function buildLlmFlow(outputs: Property[]): Flow {
    const llmNode = createLlmNode({
      name: "llm_node",
      llmConfig: makeLlmConfig(),
      promptTemplate:
        "Answer in one short sentence. What is the fastest {{nationality}} car?",
      inputs: [nationality],
      outputs,
    });
    const start = ioStartNode("start", [nationality]);
    const end = ioEndNode("end", outputs);
    return createFlow({
      name: "flow",
      startNode: start,
      nodes: [start, llmNode, end],
      controlFlowConnections: [ctrl(start, llmNode), ctrl(llmNode, end)],
      dataFlowConnections: [
        dataEdge(start, llmNode, "nationality"),
        ...outputs.map((prop) => dataEdge(llmNode, end, prop.title)),
      ],
      outputs,
    });
  }

  it("unstructured: a single string output takes the message content of the rendered prompt call", async () => {
    const { model, captured } = makeChatModelFake({ reply: "The Ferrari." });
    const { agent } = await loadWithFakeLlm(buildLlmFlow([car]), () => model);

    const result = await agent.invoke({ inputs: { nationality: "italian" } });
    expect(outputsOf(result)).toEqual({ car: "The Ferrari." });

    // The prompt template was rendered against the node inputs.
    expect(captured.structuredSchemas).toHaveLength(0);
    expect(captured.prompts).toHaveLength(1);
    const promptMessages = captured.prompts[0] as Array<{
      role: string;
      content: string;
    }>;
    expect(promptMessages).toEqual([
      {
        role: "user",
        content:
          "Answer in one short sentence. What is the fastest italian car?",
      },
    ]);
  });

  it("structured: multiple outputs use withStructuredOutput with the built JSON schema", async () => {
    const rating = integerProperty({ title: "rating" });
    const { model, captured } = makeChatModelFake({
      structured: { car: "Ferrari", rating: 9 },
    });
    const { agent } = await loadWithFakeLlm(
      buildLlmFlow([car, rating]),
      () => model,
    );

    const result = await agent.invoke({ inputs: { nationality: "italian" } });
    expect(outputsOf(result)).toEqual({ car: "Ferrari", rating: 9 });

    expect(captured.structuredSchemas).toHaveLength(1);
    expect(captured.structuredSchemas[0]).toEqual({
      title: "structured_output",
      type: "object",
      properties: {
        car: car.jsonSchema,
        rating: rating.jsonSchema,
      },
    });
  });

  it("structured: a flattened single-property result is rewrapped under the declared title", async () => {
    const wrapped = objectProperty({ title: "wrapped", properties: {} });
    const { model } = makeChatModelFake({ structured: { inner: 1 } });
    const { agent } = await loadWithFakeLlm(
      buildLlmFlow([wrapped]),
      () => model,
    );

    const result = await agent.invoke({ inputs: { nationality: "italian" } });
    expect(outputsOf(result)).toEqual({ wrapped: { inner: 1 } });
  });
});
