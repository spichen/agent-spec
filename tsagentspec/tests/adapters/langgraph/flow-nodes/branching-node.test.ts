/**
 * BranchingNode flow execution tests for the LangGraph adapter.
 *
 * Mirrors `pyagentspec/tests/adapters/langgraph/flows/test_branchingnode.py`;
 * all tests run offline.
 */
import { describe, expect, it } from "vitest";
import {
  createBranchingNode,
  createFlow,
  stringProperty,
} from "../../../../src/index.js";
import {
  ctrl,
  dataEdge,
  ioEndNode,
  ioStartNode,
  loadFlow,
  outputsOf,
} from "../test-helpers.js";

describe("BranchingNode", () => {
  it("routes on the mapping, falls back to the default branch, and keeps defaults on untaken paths", async () => {
    const customInput = stringProperty({ title: "custom_input" });
    const outputA = stringProperty({ title: "output_a", default: "no_value" });
    const outputB = stringProperty({ title: "output_b", default: "no_value" });
    const branchingNode = createBranchingNode({
      name: "branching",
      mapping: { a: "branch_a", b: "branch_b" },
      inputs: [customInput],
    });
    const start = ioStartNode("start", [customInput]);
    const endA = ioEndNode("end_a", [outputA]);
    const endB = ioEndNode("end_b", [outputB]);
    const endDefault = ioEndNode("end_default");

    const flow = createFlow({
      name: "flow",
      startNode: start,
      nodes: [start, branchingNode, endA, endB, endDefault],
      controlFlowConnections: [
        ctrl(start, branchingNode),
        ctrl(branchingNode, endA, "branch_a"),
        ctrl(branchingNode, endB, "branch_b"),
        ctrl(branchingNode, endDefault, "default"),
      ],
      dataFlowConnections: [
        dataEdge(start, branchingNode, "custom_input"),
        dataEdge(start, endB, "custom_input", "output_b"),
        dataEdge(start, endA, "custom_input", "output_a"),
      ],
      outputs: [outputA, outputB],
    });

    const graph = await loadFlow(flow);

    let result = await graph.invoke({ inputs: { custom_input: "a" } });
    expect(outputsOf(result)).toEqual({ output_a: "a", output_b: "no_value" });
    expect(result).toHaveProperty("messages");

    result = await graph.invoke({ inputs: { custom_input: "b" } });
    expect(outputsOf(result)).toEqual({ output_a: "no_value", output_b: "b" });

    result = await graph.invoke({ inputs: { custom_input: "no_match" } });
    expect(outputsOf(result)).toEqual({
      output_a: "no_value",
      output_b: "no_value",
    });
  });

  it("raises the missing-input error when nothing feeds the branching input", async () => {
    const customInput = stringProperty({ title: "custom_input" });
    const branchingNode = createBranchingNode({
      name: "branching",
      mapping: { a: "branch_a" },
      inputs: [customInput],
    });
    const start = ioStartNode("start");
    const endA = ioEndNode("end_a");
    const endDefault = ioEndNode("end_default");
    const flow = createFlow({
      name: "flow",
      startNode: start,
      nodes: [start, branchingNode, endA, endDefault],
      controlFlowConnections: [
        ctrl(start, branchingNode),
        ctrl(branchingNode, endA, "branch_a"),
        ctrl(branchingNode, endDefault, "default"),
      ],
      dataFlowConnections: [],
    });

    const graph = await loadFlow(flow);
    await expect(graph.invoke({ inputs: {} })).rejects.toThrow(
      "Expected node `branching` to have a value for property `custom_input`, but none was found.",
    );
  });
});
