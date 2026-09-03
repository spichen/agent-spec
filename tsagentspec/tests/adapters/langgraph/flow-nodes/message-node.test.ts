/**
 * Input/OutputMessageNode flow execution tests for the LangGraph adapter.
 *
 * Mirrors `pyagentspec/tests/adapters/langgraph/flows/test_inputmessagenode.py`
 * and `test_outputmessagenode.py` (merged: each suite is a single small
 * scenario); all tests run offline.
 */
import { describe, expect, it } from "vitest";
import { Command, MemorySaver } from "@langchain/langgraph";
import {
  createFlow,
  createInputMessageNode,
  createOutputMessageNode,
  stringProperty,
} from "../../../../src/index.js";
import {
  ctrl,
  dataEdge,
  getInterrupts,
  ioEndNode,
  ioStartNode,
  loadFlow,
  messagesOf,
  outputsOf,
  threadConfig,
} from "../test-helpers.js";

describe("InputMessageNode", () => {
  it("interrupts with an empty payload; the resume value becomes the output and a user message", async () => {
    const customInput = stringProperty({ title: "custom_input" });
    const inputMessageNode = createInputMessageNode({
      name: "input_message",
      outputs: [customInput],
    });
    const start = ioStartNode("start");
    const end = ioEndNode("end", [customInput]);
    const flow = createFlow({
      name: "flow",
      startNode: start,
      nodes: [start, inputMessageNode, end],
      controlFlowConnections: [
        ctrl(start, inputMessageNode),
        ctrl(inputMessageNode, end),
      ],
      dataFlowConnections: [dataEdge(inputMessageNode, end, "custom_input")],
      outputs: [customInput],
    });

    const graph = await loadFlow(flow, { checkpointer: new MemorySaver() });
    const config = threadConfig("1");

    const first = await graph.invoke({}, config);
    const interrupts = getInterrupts(first);
    expect(interrupts).toHaveLength(1);
    expect(interrupts[0]!.value).toBe("");

    const result = await graph.invoke(new Command({ resume: "3" }), config);
    expect(outputsOf(result)).toEqual({ custom_input: "3" });

    const messages = messagesOf(result);
    expect(messages).toHaveLength(1);
    expect(messages[0]!.getType()).toBe("human");
    expect(messages[0]!.content).toBe("3");
  });
});

describe("OutputMessageNode", () => {
  it("emits the rendered template as an assistant message", async () => {
    const customInput = stringProperty({ title: "custom_input" });
    const outputMessageNode = createOutputMessageNode({
      name: "output_message",
      message: "Hey {{custom_input}}",
      inputs: [customInput],
    });
    const start = ioStartNode("start", [customInput]);
    const end = ioEndNode("end");
    const flow = createFlow({
      name: "flow",
      startNode: start,
      nodes: [start, outputMessageNode, end],
      controlFlowConnections: [
        ctrl(start, outputMessageNode),
        ctrl(outputMessageNode, end),
      ],
      dataFlowConnections: [dataEdge(start, outputMessageNode, "custom_input")],
      inputs: [customInput],
    });

    const graph = await loadFlow(flow);
    const result = await graph.invoke({ inputs: { custom_input: "custom" } });

    expect(result).toHaveProperty("outputs");
    const messages = messagesOf(result);
    expect(messages).toHaveLength(1);
    expect(messages[0]!.getType()).toBe("ai");
    expect(messages[0]!.content).toBe("Hey custom");
  });
});
