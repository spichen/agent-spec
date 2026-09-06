/**
 * MapNode flow execution tests for the LangGraph adapter.
 *
 * Mirrors `pyagentspec/tests/adapters/langgraph/flows/test_mapnode.py`; all
 * tests run offline.
 */
import { describe, expect, it } from "vitest";
import {
  createFlow,
  createMapNode,
  createServerTool,
  createToolNode,
  listProperty,
  numberProperty,
  unionProperty,
  type Flow,
} from "../../../../src/index.js";
import {
  ctrl,
  dataEdge,
  ioEndNode,
  ioStartNode,
  loadFlow,
  outputsOf,
} from "../test-helpers.js";

describe("MapNode", () => {
  function buildSquareSubflow(): Flow {
    const xProp = numberProperty({ title: "input" });
    const xSquareProp = numberProperty({ title: "input_square" });
    const squareTool = createServerTool({
      name: "square_tool",
      description: "Computes the square of a number",
      inputs: [xProp],
      outputs: [xSquareProp],
    });
    const subStart = ioStartNode("subflow_start", [xProp]);
    const toolNode = createToolNode({ name: "square_tool_node", tool: squareTool });
    const subEnd = ioEndNode("subflow_end", [xSquareProp]);
    return createFlow({
      name: "Square number flow",
      startNode: subStart,
      nodes: [subStart, toolNode, subEnd],
      controlFlowConnections: [ctrl(subStart, toolNode), ctrl(toolNode, subEnd)],
      dataFlowConnections: [
        dataEdge(subStart, toolNode, "input"),
        dataEdge(toolNode, subEnd, "input_square"),
      ],
    });
  }

  const iteratedInput = unionProperty({
    title: "iterated_input",
    anyOf: [
      numberProperty({ title: "input" }),
      listProperty({ title: "input", itemType: numberProperty({ title: "item" }) }),
    ],
  });
  const collectedSquare = listProperty({
    title: "collected_input_square",
    itemType: numberProperty({ title: "item" }),
  });
  const squareRegistry = {
    square_tool: (input: unknown) => {
      const { input: value } = input as { input: number };
      return value * value;
    },
  };

  it("iterates the subflow over the list input and collects the outputs", async () => {
    const mapNode = createMapNode({
      name: "square_number_map_node",
      subflow: buildSquareSubflow(),
      inputs: [iteratedInput],
      outputs: [collectedSquare],
    });
    const inputList = listProperty({
      title: "input_list",
      itemType: numberProperty({ title: "item" }),
    });
    const start = ioStartNode("outer_start", [inputList]);
    const end = ioEndNode("outer_end", [collectedSquare]);
    const flow = createFlow({
      name: "flow to square all elements of a list",
      startNode: start,
      nodes: [start, mapNode, end],
      controlFlowConnections: [ctrl(start, mapNode), ctrl(mapNode, end)],
      dataFlowConnections: [
        dataEdge(start, mapNode, "input_list", "iterated_input"),
        dataEdge(mapNode, end, "collected_input_square"),
      ],
    });

    const graph = await loadFlow(flow, { toolRegistry: squareRegistry });
    const result = await graph.invoke({ inputs: { input_list: [1, 2, 3, 4] } });
    expect(outputsOf(result)).toEqual({
      collected_input_square: [1, 4, 9, 16],
    });
  });

  it("raises when iterated inputs have different lengths", async () => {
    const aProp = numberProperty({ title: "a" });
    const bProp = numberProperty({ title: "b" });
    const totalProp = numberProperty({ title: "total" });
    const sumTool = createServerTool({
      name: "sum_tool",
      description: "Adds two numbers",
      inputs: [aProp, bProp],
      outputs: [totalProp],
    });
    const subStart = ioStartNode("sum_start", [aProp, bProp]);
    const toolNode = createToolNode({ name: "sum_tool_node", tool: sumTool });
    const subEnd = ioEndNode("sum_end", [totalProp]);
    const sumSubflow = createFlow({
      name: "sum_subflow",
      startNode: subStart,
      nodes: [subStart, toolNode, subEnd],
      controlFlowConnections: [ctrl(subStart, toolNode), ctrl(toolNode, subEnd)],
      dataFlowConnections: [
        dataEdge(subStart, toolNode, "a"),
        dataEdge(subStart, toolNode, "b"),
        dataEdge(toolNode, subEnd, "total"),
      ],
      inputs: [aProp, bProp],
      outputs: [totalProp],
    });

    const iteratedA = unionProperty({
      title: "iterated_a",
      anyOf: [
        numberProperty({ title: "a" }),
        listProperty({ title: "a", itemType: numberProperty({ title: "item" }) }),
      ],
    });
    const iteratedB = unionProperty({
      title: "iterated_b",
      anyOf: [
        numberProperty({ title: "b" }),
        listProperty({ title: "b", itemType: numberProperty({ title: "item" }) }),
      ],
    });
    const collectedTotal = listProperty({
      title: "collected_total",
      itemType: numberProperty({ title: "item" }),
    });
    const mapNode = createMapNode({
      name: "sum_map_node",
      subflow: sumSubflow,
      inputs: [iteratedA, iteratedB],
      outputs: [collectedTotal],
    });

    const listA = listProperty({
      title: "list_a",
      itemType: numberProperty({ title: "item" }),
    });
    const listB = listProperty({
      title: "list_b",
      itemType: numberProperty({ title: "item" }),
    });
    const start = ioStartNode("outer_start", [listA, listB]);
    const end = ioEndNode("outer_end", [collectedTotal]);
    const flow = createFlow({
      name: "sum_map_flow",
      startNode: start,
      nodes: [start, mapNode, end],
      controlFlowConnections: [ctrl(start, mapNode), ctrl(mapNode, end)],
      dataFlowConnections: [
        dataEdge(start, mapNode, "list_a", "iterated_a"),
        dataEdge(start, mapNode, "list_b", "iterated_b"),
        dataEdge(mapNode, end, "collected_total"),
      ],
    });

    const graph = await loadFlow(flow, {
      toolRegistry: {
        sum_tool: (input: unknown) => {
          const { a, b } = input as { a: number; b: number };
          return a + b;
        },
      },
    });
    await expect(
      graph.invoke({ inputs: { list_a: [1, 2], list_b: [10, 20, 30] } }),
    ).rejects.toThrow("Found inputs to iterate with different sizes");
  });

  it("raises naming the input when an iterated input has no length at runtime", async () => {
    // The converter selects iterated_input statically (list-typed schema),
    // but the runtime value is a scalar: the error names the node and the
    // offending input instead of reusing the size-mismatch text.
    const mapNode = createMapNode({
      name: "square_number_map_node",
      subflow: buildSquareSubflow(),
      inputs: [iteratedInput],
      outputs: [collectedSquare],
    });
    const inputList = listProperty({
      title: "input_list",
      itemType: numberProperty({ title: "item" }),
    });
    const start = ioStartNode("outer_start", [inputList]);
    const end = ioEndNode("outer_end", [collectedSquare]);
    const flow = createFlow({
      name: "flow to square all elements of a list",
      startNode: start,
      nodes: [start, mapNode, end],
      controlFlowConnections: [ctrl(start, mapNode), ctrl(mapNode, end)],
      dataFlowConnections: [
        dataEdge(start, mapNode, "input_list", "iterated_input"),
        dataEdge(mapNode, end, "collected_input_square"),
      ],
    });

    const graph = await loadFlow(flow, { toolRegistry: squareRegistry });
    await expect(graph.invoke({ inputs: { input_list: 7 } })).rejects.toThrow(
      "MapNode `square_number_map_node` cannot iterate over input " +
        "`iterated_input`: 7 has no length",
    );
  });

  it("raises when no data-flow edge selects an input to iterate", async () => {
    const mapNode = createMapNode({
      name: "square_map_scalar",
      subflow: buildSquareSubflow(),
      inputs: [iteratedInput],
      outputs: [collectedSquare],
    });
    // The edge feeds a SCALAR into iterated_input, so the converter finds no
    // list-typed source matching the subflow input and selects nothing.
    const singleX = numberProperty({ title: "single_x" });
    const start = ioStartNode("outer_start", [singleX]);
    const end = ioEndNode("outer_end", [collectedSquare]);
    const flow = createFlow({
      name: "scalar_map_flow",
      startNode: start,
      nodes: [start, mapNode, end],
      controlFlowConnections: [ctrl(start, mapNode), ctrl(mapNode, end)],
      dataFlowConnections: [
        dataEdge(start, mapNode, "single_x", "iterated_input"),
        dataEdge(mapNode, end, "collected_input_square"),
      ],
    });

    const graph = await loadFlow(flow, { toolRegistry: squareRegistry });
    await expect(graph.invoke({ inputs: { single_x: 3 } })).rejects.toThrow(
      "MapNode has no inputs to iterate",
    );
  });
});
