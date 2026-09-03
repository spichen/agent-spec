/**
 * ToolNode flow execution tests for the LangGraph adapter.
 *
 * Mirrors `pyagentspec/tests/adapters/langgraph/flows/test_toolnode.py` with
 * a checkpointer-driven interrupt/resume loop so every test runs offline.
 *
 * Documented divergence exercised here: tuples do not exist in JS, so arrays
 * map positionally onto multiple declared tool-node outputs (Python restricts
 * positional mapping to tuples).
 *
 * Note on node construction: the Python SDK infers the missing IO side of
 * Start/End nodes, so Python specs always carry both sides on the wire; the
 * TS factories default the missing side to `[]`, so these tests pass both
 * sides explicitly (via `ioStartNode`/`ioEndNode`), matching the serialized
 * wire format.
 */
import { describe, expect, it } from "vitest";
import { Command, MemorySaver } from "@langchain/langgraph";
import {
  createClientTool,
  createFlow,
  createToolNode,
  listProperty,
  numberProperty,
  objectProperty,
  stringProperty,
  type Flow,
  type Property,
} from "../../../../src/index.js";
import {
  ctrl,
  dataEdge,
  getInterrupts,
  ioEndNode,
  ioStartNode,
  loadFlow,
  outputsOf,
  threadConfig,
} from "../test-helpers.js";

describe("ToolNode output-mapping matrix", () => {
  /** Python's `_build_flow_with_client_tool`: start -> ClientTool -> end. */
  function buildClientToolFlow(
    inputProp: Property,
    outputProps: Property[],
  ): Flow {
    const start = ioStartNode("start", [inputProp]);
    const clientTool = createClientTool({
      name: "echo_tool",
      description: "Client-side tool used for testing",
      inputs: [inputProp],
      outputs: outputProps,
    });
    const toolNode = createToolNode({ name: "tool", tool: clientTool });
    const end = ioEndNode("end", outputProps);
    return createFlow({
      name: "tool_output_flow",
      startNode: start,
      nodes: [start, toolNode, end],
      controlFlowConnections: [ctrl(start, toolNode), ctrl(toolNode, end)],
      dataFlowConnections: [
        dataEdge(start, toolNode, inputProp.title),
        ...outputProps.map((prop) => dataEdge(toolNode, end, prop.title)),
      ],
    });
  }

  /** Interrupt at the client tool, then resume with the given payload. */
  async function runFlowAndResume(
    flow: Flow,
    resumePayload: unknown,
  ): Promise<Record<string, unknown>> {
    const graph = await loadFlow(flow, { checkpointer: new MemorySaver() });
    const config = threadConfig("t");
    const first = await graph.invoke(
      { inputs: { [flow.inputs![0]!.title]: 123 } },
      config,
    );
    expect(getInterrupts(first)).toHaveLength(1);
    const resumed = await graph.invoke(
      new Command({ resume: resumePayload }),
      config,
    );
    return outputsOf(resumed);
  }

  it("interrupts with the client_tool_request payload and resumes with the value", async () => {
    const inputProp = numberProperty({ title: "input" });
    const outputProp = numberProperty({ title: "input_square" });
    const squareTool = createClientTool({
      name: "square_tool",
      description: "Computes the square of a number",
      inputs: [inputProp],
      outputs: [outputProp],
    });
    const start = ioStartNode("subflow_start", [inputProp]);
    const toolNode = createToolNode({ name: "square_tool_node", tool: squareTool });
    const end = ioEndNode("subflow_end", [outputProp]);
    const flow = createFlow({
      name: "Square number flow",
      startNode: start,
      nodes: [start, toolNode, end],
      controlFlowConnections: [ctrl(start, toolNode), ctrl(toolNode, end)],
      dataFlowConnections: [
        dataEdge(start, toolNode, "input"),
        dataEdge(toolNode, end, "input_square"),
      ],
    });

    const graph = await loadFlow(flow, { checkpointer: new MemorySaver() });
    const config = threadConfig("1");
    const first = await graph.invoke({ inputs: { input: 4 } }, config);
    const interrupts = getInterrupts(first);
    expect(interrupts).toHaveLength(1);
    expect(interrupts[0]!.value).toEqual({
      type: "client_tool_request",
      name: "square_tool",
      description: "Computes the square of a number",
      inputs: { args: [], kwargs: { input: 4 } },
    });

    const resumed = await graph.invoke(new Command({ resume: 16 }), config);
    expect(outputsOf(resumed)["input_square"]).toBe(16);
  });

  it("single ObjectProperty output wraps a multi-key dict under the declared key", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      objectProperty({ title: "out_dict", properties: {} }),
    ]);
    const outputs = await runFlowAndResume(flow, { a: 1, b: 2 });
    expect(outputs).toEqual({ out_dict: { a: 1, b: 2 } });
  });

  it("single ObjectProperty output wraps a single-key dict under the declared key", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      objectProperty({ title: "out_dict", properties: {} }),
    ]);
    const outputs = await runFlowAndResume(flow, { a: 1 });
    expect(outputs).toEqual({ out_dict: { a: 1 } });
  });

  it("single output uses a dict keyed by the declared title as-is", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      objectProperty({ title: "out_dict", properties: {} }),
    ]);
    const outputs = await runFlowAndResume(flow, { out_dict: 1 });
    expect(outputs).toEqual({ out_dict: 1 });
  });

  it("scalar output passes through under the declared key", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      stringProperty({ title: "out_string" }),
    ]);
    const outputs = await runFlowAndResume(flow, "value");
    expect(outputs).toEqual({ out_string: "value" });
  });

  it("multiple outputs filter the dict and defaults fill missing keys", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      numberProperty({ title: "a" }),
      numberProperty({ title: "b", default: 0 }),
    ]);
    const outputs = await runFlowAndResume(flow, { a: 5 });
    expect(outputs).toEqual({ a: 5, b: 0 });
  });

  it("list output maps to a single declared list output", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      listProperty({ title: "out", itemType: numberProperty({ title: "item" }) }),
    ]);
    const outputs = await runFlowAndResume(flow, [1, 2, 3]);
    expect(outputs).toEqual({ out: [1, 2, 3] });
  });

  it("scalar output maps to a single declared number output", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      numberProperty({ title: "out_number" }),
    ]);
    const outputs = await runFlowAndResume(flow, 42);
    expect(outputs).toEqual({ out_number: 42 });
  });

  it("array output onto a single declared string output is stringified", async () => {
    // Python (tuple payload) stringifies via json.dumps -> "[1, 2]"; the TS
    // cast mirrors json.dumps formatting (", " separator, not "[1,2]").
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      stringProperty({ title: "out" }),
    ]);
    const outputs = await runFlowAndResume(flow, [1, 2]);
    expect(outputs).toEqual({ out: "[1, 2]" });
  });

  it("array output shorter than the declared outputs raises like Python", async () => {
    // Python raises IndexError instead of silently mapping undefined.
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      numberProperty({ title: "a" }),
      stringProperty({ title: "b" }),
    ]);
    await expect(runFlowAndResume(flow, [7])).rejects.toThrow(
      "Tool node `tool` returned 1 value(s) but declares 2 outputs; " +
        "no value for output `b`.",
    );
  });

  it("content-block list shorter than the declared outputs raises like Python", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      stringProperty({ title: "text_out" }),
      stringProperty({ title: "image_out" }),
    ]);
    await expect(
      runFlowAndResume(flow, [{ type: "text", text: "hello" }]),
    ).rejects.toThrow(
      "Tool node `tool` returned 1 content block(s) but declares 2 outputs; " +
        "no value for output `image_out`.",
    );
  });

  it("array output maps positionally onto multiple outputs", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      numberProperty({ title: "a" }),
      stringProperty({ title: "b" }),
    ]);
    const outputs = await runFlowAndResume(flow, [7, "ok"]);
    expect(outputs).toEqual({ a: 7, b: "ok" });
  });

  it("mixed array output maps positionally onto number/object/array outputs", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      numberProperty({ title: "num" }),
      objectProperty({ title: "obj", properties: {} }),
      listProperty({ title: "array", itemType: numberProperty({ title: "elem" }) }),
    ]);
    const outputs = await runFlowAndResume(flow, [7, { key: "val" }, [1]]);
    expect(outputs).toEqual({ num: 7, obj: { key: "val" }, array: [1] });
  });

  it("MCP content-block lists extract payloads positionally", async () => {
    const flow = buildClientToolFlow(numberProperty({ title: "x" }), [
      stringProperty({ title: "text_out" }),
      stringProperty({ title: "image_out" }),
    ]);
    const outputs = await runFlowAndResume(flow, [
      { type: "text", text: "hello" },
      { type: "image", base64: "imgdata" },
    ]);
    expect(outputs).toEqual({ text_out: "hello", image_out: "imgdata" });
  });
});
