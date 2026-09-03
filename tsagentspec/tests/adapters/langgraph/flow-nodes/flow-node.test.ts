/**
 * FlowNode (subflow) execution tests for the LangGraph adapter.
 *
 * Mirrors `pyagentspec/tests/adapters/langgraph/flows/test_flownode.py`; all
 * tests run offline.
 */
import { describe, expect, it } from "vitest";
import { createFlow, createFlowNode, stringProperty } from "../../../../src/index.js";
import {
  ctrl,
  dataEdge,
  ioEndNode,
  ioStartNode,
  loadFlow,
  outputsOf,
} from "../test-helpers.js";

describe("FlowNode", () => {
  it("executes the subflow and passes its outputs through", async () => {
    const customProp = stringProperty({ title: "custom_prop" });
    const subStart = ioStartNode("start", [customProp]);
    const subEnd = ioEndNode("end", [customProp]);
    const subflow = createFlow({
      name: "subflow",
      startNode: subStart,
      nodes: [subStart, subEnd],
      controlFlowConnections: [ctrl(subStart, subEnd)],
      dataFlowConnections: [dataEdge(subStart, subEnd, "custom_prop")],
      inputs: [customProp],
      outputs: [customProp],
    });

    const flowNode = createFlowNode({ name: "flow_node", subflow });
    const start = ioStartNode("start", [customProp]);
    const end = ioEndNode("end", [customProp]);
    const flow = createFlow({
      name: "outer",
      startNode: start,
      nodes: [start, flowNode, end],
      controlFlowConnections: [ctrl(start, flowNode), ctrl(flowNode, end)],
      dataFlowConnections: [
        dataEdge(start, flowNode, "custom_prop"),
        dataEdge(flowNode, end, "custom_prop"),
      ],
      inputs: [customProp],
      outputs: [customProp],
    });

    const graph = await loadFlow(flow);
    const result = await graph.invoke({ inputs: { custom_prop: "custom" } });
    expect(result).toHaveProperty("messages");
    expect(outputsOf(result)).toEqual({ custom_prop: "custom" });
  });
});
