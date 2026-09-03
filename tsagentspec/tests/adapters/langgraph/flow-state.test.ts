/**
 * Flow state contract tests for the LangGraph adapter.
 *
 * Mirrors the state-shape behaviors of the Python suite
 * (`pyagentspec/tests/adapters/langgraph/flows/`): the `{inputs}` in /
 * `{outputs, messages, node_execution_details}` out contract, StartNode
 * consumption of flow-level inputs (defaults, casting, missing-required
 * error), automatically generated data-flow edges when the flow declares no
 * `dataFlowConnections`, and explicit data-flow edges routing between
 * differently-named properties. All tests run offline.
 *
 * Note on node construction: the Python SDK infers the missing IO side of
 * Start/End nodes (`_get_inferred_inputs/outputs` return `inputs or outputs`),
 * so Python specs always carry both on the wire. The TS factories do not infer
 * (`createStartNode`/`createEndNode` default the missing side to `[]`), so
 * these tests pass both sides explicitly, matching the serialized wire format.
 */
import { describe, expect, it } from "vitest";
import type { BaseMessage } from "@langchain/core/messages";
import {
  createFlow,
  createServerTool,
  createToolNode,
  integerProperty,
  numberProperty,
  stringProperty,
  type Flow,
  type Property,
} from "../../../src/index.js";
import {
  ctrl,
  dataEdge,
  ioEndNode,
  ioStartNode,
  loadFlow,
} from "./test-helpers.js";

/** A start -> end pass-through flow over the given properties. */
function passThroughFlow(props: Property[], endBranchName?: string): Flow {
  const start = ioStartNode("start", props);
  const end = ioEndNode("end", props, endBranchName);
  return createFlow({
    name: "pass_through_flow",
    startNode: start,
    nodes: [start, end],
    controlFlowConnections: [ctrl(start, end)],
    dataFlowConnections: props.map((prop) => dataEdge(start, end, prop.title)),
  });
}

describe("flow state contract", () => {
  it("takes {inputs} in and returns {outputs, messages, node_execution_details}", async () => {
    const graph = await loadFlow(
      passThroughFlow([stringProperty({ title: "x" })]),
    );
    const result = await graph.invoke({ inputs: { x: "v" } });

    expect(result).toHaveProperty("outputs");
    expect(result).toHaveProperty("messages");
    expect(result).toHaveProperty("node_execution_details");
    // `inputs` is internal routing state and must not leak out.
    expect("inputs" in result).toBe(false);

    expect(result["outputs"]).toEqual({ x: "v" });
    expect(result["messages"]).toEqual([]);
    expect(result["node_execution_details"]).toEqual({
      branch: "next",
      generated_messages: [],
      should_finish: true,
    });
  });

  it("exposes a custom EndNode branchName in node_execution_details.branch", async () => {
    const graph = await loadFlow(
      passThroughFlow([stringProperty({ title: "x" })], "DONE"),
    );
    const result = await graph.invoke({ inputs: { x: "v" } });
    expect(
      (result["node_execution_details"] as Record<string, unknown>)["branch"],
    ).toBe("DONE");
    expect(
      (result["node_execution_details"] as Record<string, unknown>)[
        "should_finish"
      ],
    ).toBe(true);
  });

  it("drops flow-level inputs that no start-node property declares", async () => {
    const graph = await loadFlow(
      passThroughFlow([stringProperty({ title: "x" })]),
    );
    const result = await graph.invoke({
      inputs: { x: "v", undeclared: "ignored" },
    });
    expect(result["outputs"]).toEqual({ x: "v" });
  });
});

describe("StartNode input consumption", () => {
  it("applies declared defaults when the invocation omits inputs entirely", async () => {
    const graph = await loadFlow(
      passThroughFlow([stringProperty({ title: "x", default: "fallback" })]),
    );
    const result = await graph.invoke({});
    expect(result["outputs"]).toEqual({ x: "fallback" });
  });

  it("casts values to the declared property types", async () => {
    const graph = await loadFlow(
      passThroughFlow([
        integerProperty({ title: "count" }),
        stringProperty({ title: "text" }),
        numberProperty({ title: "ratio" }),
      ]),
    );
    const result = await graph.invoke({
      inputs: { count: " 5 ", text: 7, ratio: "2.5" },
    });
    // Numeric strings parse into integer/number properties; non-strings are
    // JSON-serialized into string properties.
    expect(result["outputs"]).toEqual({ count: 5, text: "7", ratio: 2.5 });
  });

  it("raises the Python error text for a missing required input", async () => {
    const graph = await loadFlow(
      passThroughFlow([stringProperty({ title: "x" })]),
    );
    await expect(graph.invoke({ inputs: {} })).rejects.toThrow(
      "Expected node `start` to have a value for property `x`, but none was found.",
    );
  });

  it("raises Python's int() error for an unparsable integer string", async () => {
    // Python does `int(value.strip())` and its error-message guard never
    // matches int()'s text, so unparsable integer strings abort the flow.
    const graph = await loadFlow(
      passThroughFlow([integerProperty({ title: "count" })]),
    );
    await expect(graph.invoke({ inputs: { count: "3.5" } })).rejects.toThrow(
      'invalid literal for int() with base 10: "3.5"',
    );
    await expect(graph.invoke({ inputs: { count: "abc" } })).rejects.toThrow(
      'invalid literal for int() with base 10: "abc"',
    );
  });

  it("accepts underscore digit separators in integer strings like int()", async () => {
    const graph = await loadFlow(
      passThroughFlow([integerProperty({ title: "count" })]),
    );
    const result = await graph.invoke({ inputs: { count: "1_000" } });
    expect(result["outputs"]).toEqual({ count: 1000 });
  });

  it("matches Python's float() coercion matrix for number strings", async () => {
    const graph = await loadFlow(
      passThroughFlow([numberProperty({ title: "ratio" })]),
    );
    // Hex/binary/octal literals stay strings (Python float() rejects them
    // and the error is swallowed, leaving the value as-is).
    let result = await graph.invoke({ inputs: { ratio: "0x10" } });
    expect(result["outputs"]).toEqual({ ratio: "0x10" });
    // inf/nan/underscore separators convert like Python's float().
    result = await graph.invoke({ inputs: { ratio: "inf" } });
    expect(result["outputs"]).toEqual({ ratio: Infinity });
    result = await graph.invoke({ inputs: { ratio: "-Infinity" } });
    expect(result["outputs"]).toEqual({ ratio: -Infinity });
    result = await graph.invoke({ inputs: { ratio: "1_000.5" } });
    expect(result["outputs"]).toEqual({ ratio: 1000.5 });
    result = await graph.invoke({ inputs: { ratio: "nan" } });
    expect(result["outputs"]).toEqual({ ratio: NaN });
  });

  it("stringifies non-string values with json.dumps formatting", async () => {
    // Python casts container values into string properties via json.dumps,
    // whose separators include spaces: "[1, 2]", not JSON.stringify's
    // "[1,2]".
    const graph = await loadFlow(
      passThroughFlow([stringProperty({ title: "text" })]),
    );
    let result = await graph.invoke({ inputs: { text: [1, 2] } });
    expect(result["outputs"]).toEqual({ text: "[1, 2]" });
    result = await graph.invoke({ inputs: { text: { a: 1, b: [true, null] } } });
    expect(result["outputs"]).toEqual({ text: '{"a": 1, "b": [true, null]}' });
    // ensure_ascii escapes non-ASCII text.
    result = await graph.invoke({ inputs: { text: ["café"] } });
    expect(result["outputs"]).toEqual({ text: '["caf\\u00e9"]' });
  });
});

describe("control-flow edges", () => {
  it("treats an empty-string fromBranch as the default next branch", async () => {
    // Python coerces a falsy from_branch ("" included) to "next"; the TS
    // adapter must not keep "" as a distinct branch key.
    const prop = stringProperty({ title: "x" });
    const start = ioStartNode("start", [prop]);
    const end = ioEndNode("end", [prop]);
    const flow = createFlow({
      name: "empty_branch_flow",
      startNode: start,
      nodes: [start, end],
      controlFlowConnections: [ctrl(start, end, "")],
      dataFlowConnections: [dataEdge(start, end, "x")],
    });
    const graph = await loadFlow(flow);
    const result = await graph.invoke({ inputs: { x: "v" } });
    expect(result["outputs"]).toEqual({ x: "v" });
  });
});

describe("data-flow edges", () => {
  const doubleToolSpec = createServerTool({
    name: "double_tool",
    description: "Doubles a number",
    inputs: [numberProperty({ title: "x" })],
    outputs: [numberProperty({ title: "y" })],
  });
  const double = (input: unknown): number => (input as { x: number }).x * 2;

  it("auto-generates edges by matching titles when dataFlowConnections is undefined", async () => {
    const start = ioStartNode("start", [numberProperty({ title: "x" })]);
    const toolNode = createToolNode({ name: "double", tool: doubleToolSpec });
    const end = ioEndNode("end", [numberProperty({ title: "y" })]);
    const flow = createFlow({
      name: "auto_edges_flow",
      startNode: start,
      nodes: [start, toolNode, end],
      controlFlowConnections: [ctrl(start, toolNode), ctrl(toolNode, end)],
      // No dataFlowConnections: the adapter wires start.x -> double.x and
      // double.y -> end.y automatically.
    });
    expect(flow.dataFlowConnections).toBeUndefined();

    const graph = await loadFlow(flow, {
      toolRegistry: { double_tool: double },
    });
    const result = await graph.invoke({ inputs: { x: 3 } });
    expect(result["outputs"]).toEqual({ y: 6 });
  });

  it("explicit edges route between differently-named properties", async () => {
    const renamedToolSpec = createServerTool({
      name: "double_tool",
      description: "Doubles a number",
      inputs: [numberProperty({ title: "value" })],
      outputs: [numberProperty({ title: "doubled" })],
    });
    const start = ioStartNode("start", [numberProperty({ title: "x" })]);
    const toolNode = createToolNode({ name: "double", tool: renamedToolSpec });
    const end = ioEndNode("end", [numberProperty({ title: "y" })]);
    const flow = createFlow({
      name: "explicit_edges_flow",
      startNode: start,
      nodes: [start, toolNode, end],
      controlFlowConnections: [ctrl(start, toolNode), ctrl(toolNode, end)],
      dataFlowConnections: [
        dataEdge(start, toolNode, "x", "value"),
        dataEdge(toolNode, end, "doubled", "y"),
      ],
    });

    const graph = await loadFlow(flow, {
      toolRegistry: {
        double_tool: (input: unknown) => (input as { value: number }).value * 2,
      },
    });
    const result = await graph.invoke({ inputs: { x: 4 } });
    expect(result["outputs"]).toEqual({ y: 8 });
  });

  it("accumulates routed values across successive nodes", async () => {
    // start.x flows through TWO chained tool nodes: each node's update must
    // accumulate into (not replace) the pending-inputs routing table.
    const secondToolSpec = createServerTool({
      name: "add_tool",
      description: "Adds two numbers",
      inputs: [numberProperty({ title: "y" }), numberProperty({ title: "x" })],
      outputs: [numberProperty({ title: "sum" })],
    });
    const start = ioStartNode("start", [numberProperty({ title: "x" })]);
    const doubleNode = createToolNode({ name: "double", tool: doubleToolSpec });
    const addNode = createToolNode({ name: "add", tool: secondToolSpec });
    const end = ioEndNode("end", [numberProperty({ title: "sum" })]);
    const flow = createFlow({
      name: "accumulate_flow",
      startNode: start,
      nodes: [start, doubleNode, addNode, end],
      controlFlowConnections: [
        ctrl(start, doubleNode),
        ctrl(doubleNode, addNode),
        ctrl(addNode, end),
      ],
      dataFlowConnections: [
        dataEdge(start, doubleNode, "x"),
        // start routes x directly to the LATER add node: the intermediate
        // double node's state update must keep this pending value alive.
        dataEdge(start, addNode, "x"),
        dataEdge(doubleNode, addNode, "y"),
        dataEdge(addNode, end, "sum"),
      ],
    });

    const graph = await loadFlow(flow, {
      toolRegistry: {
        double_tool: double,
        add_tool: (input: unknown) => {
          const { x, y } = input as { x: number; y: number };
          return x + y;
        },
      },
    });
    const result = await graph.invoke({ inputs: { x: 3 } });
    // double(3) = 6, add(6, 3) = 9
    expect(result["outputs"]).toEqual({ sum: 9 });
    expect(
      (result["messages"] as BaseMessage[]).length,
    ).toBe(0);
  });
});
