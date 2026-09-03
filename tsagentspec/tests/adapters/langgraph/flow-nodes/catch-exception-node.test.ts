/**
 * CatchExceptionNode flow execution tests for the LangGraph adapter.
 *
 * Mirrors `pyagentspec/tests/adapters/langgraph/flows/test_catchexceptionode.py`;
 * all tests run offline.
 */
import { describe, expect, it } from "vitest";
import {
  createCatchExceptionNode,
  createFlow,
  createServerTool,
  createToolNode,
  integerProperty,
  nullProperty,
  stringProperty,
  unionProperty,
  type Flow,
  type Property,
  type ServerTool,
} from "../../../../src/index.js";
import {
  ctrl,
  dataEdge,
  detailsOf,
  ioEndNode,
  ioStartNode,
  loadFlow,
  outputsOf,
} from "../test-helpers.js";

describe("CatchExceptionNode", () => {
  const inp = integerProperty({ title: "x" });
  const outp = stringProperty({ title: "y", default: "" });

  function makeErrorInfoProperty(): Property {
    return unionProperty({
      title: "error_info",
      anyOf: [
        stringProperty({ title: "error_info" }),
        nullProperty({ title: "error_info" }),
      ],
      default: null,
    });
  }

  function buildToolSubflow(tool: ServerTool, endBranch?: string): Flow {
    const subStart = ioStartNode("sub_start", [inp]);
    const toolNode = createToolNode({ name: `${tool.name}_node`, tool });
    const subEnd = ioEndNode("sub_end", [outp], endBranch);
    return createFlow({
      name: `${tool.name}_subflow`,
      startNode: subStart,
      nodes: [subStart, toolNode, subEnd],
      controlFlowConnections: [ctrl(subStart, toolNode), ctrl(toolNode, subEnd)],
      dataFlowConnections: [
        dataEdge(subStart, toolNode, "x"),
        dataEdge(toolNode, subEnd, "y"),
      ],
      inputs: [inp],
      outputs: [outp],
    });
  }

  it("routes exceptions to the caught_exception_branch with default outputs and caught_exception_info", async () => {
    const flakyTool = createServerTool({
      name: "flaky_tool",
      description: "Raises for negative inputs",
      inputs: [inp],
      outputs: [outp],
    });
    const subflow = buildToolSubflow(flakyTool);
    const catchNode = createCatchExceptionNode({ name: "catch", subflow });
    const errorInfo = makeErrorInfoProperty();
    const start = ioStartNode("start", [inp]);
    const end = ioEndNode("end", [outp]);
    const errorEnd = ioEndNode("error_end", [errorInfo], "ERROR");
    const flow = createFlow({
      name: "outer",
      startNode: start,
      nodes: [start, catchNode, end, errorEnd],
      controlFlowConnections: [
        ctrl(start, catchNode),
        ctrl(catchNode, end),
        ctrl(catchNode, errorEnd, "caught_exception_branch"),
      ],
      dataFlowConnections: [
        dataEdge(start, catchNode, "x"),
        dataEdge(catchNode, end, "y"),
        dataEdge(catchNode, errorEnd, "caught_exception_info", "error_info"),
      ],
      inputs: [inp],
      outputs: [outp, errorInfo],
    });

    const graph = await loadFlow(flow, {
      toolRegistry: {
        flaky_tool: (input: unknown) => {
          const { x } = input as { x: number };
          if (x < 0) {
            throw new Error("x must be non-negative");
          }
          return "ok";
        },
      },
    });

    // Case 1: no exception -> subflow output passes through.
    let result = await graph.invoke({ inputs: { x: 1 } });
    expect(outputsOf(result)["y"]).toBe("ok");
    expect(outputsOf(result)["error_info"]).toBeNull();

    // Case 2: exception -> default output value, ERROR end branch, and the
    // exception message routed through caught_exception_info.
    result = await graph.invoke({ inputs: { x: -1 } });
    expect(outputsOf(result)["y"]).toBe("");
    expect(detailsOf(result)["branch"]).toBe("ERROR");
    const caught = outputsOf(result)["error_info"];
    expect(typeof caught).toBe("string");
    expect(String(caught)).toContain("x must be non-negative");
  });

  it("propagates a custom subflow end branch on success with null exception info", async () => {
    const okTool = createServerTool({
      name: "ok_tool",
      description: "Always returns ok",
      inputs: [inp],
      outputs: [outp],
    });
    const subflow = buildToolSubflow(okTool, "OK");
    const catchNode = createCatchExceptionNode({ name: "catch", subflow });
    const errorInfo = makeErrorInfoProperty();
    const start = ioStartNode("start", [inp]);
    const okEnd = ioEndNode("ok_end", [outp, errorInfo]);
    const otherEnd = ioEndNode("other_end");
    const flow = createFlow({
      name: "outer",
      startNode: start,
      nodes: [start, catchNode, okEnd, otherEnd],
      controlFlowConnections: [
        ctrl(start, catchNode),
        ctrl(catchNode, okEnd, "OK"),
        ctrl(catchNode, otherEnd),
      ],
      dataFlowConnections: [
        dataEdge(start, catchNode, "x"),
        dataEdge(catchNode, okEnd, "y"),
        dataEdge(catchNode, okEnd, "caught_exception_info", "error_info"),
      ],
      inputs: [inp],
      outputs: [outp, errorInfo],
    });

    const graph = await loadFlow(flow, {
      toolRegistry: { ok_tool: () => "ok" },
    });
    const result = await graph.invoke({ inputs: { x: 7 } });
    expect(detailsOf(result)["branch"]).toBe("next");
    expect(outputsOf(result)["y"]).toBe("ok");
    expect(outputsOf(result)["error_info"]).toBeNull();
  });

  it("uses the default next branch on success with null exception info", async () => {
    const okTool = createServerTool({
      name: "ok_tool_default",
      description: "Always returns ok",
      inputs: [inp],
      outputs: [outp],
    });
    const subflow = buildToolSubflow(okTool);
    const catchNode = createCatchExceptionNode({ name: "catch", subflow });
    const errorInfo = makeErrorInfoProperty();
    const start = ioStartNode("start", [inp]);
    const nextEnd = ioEndNode("next_end", [outp, errorInfo]);
    const flow = createFlow({
      name: "outer",
      startNode: start,
      nodes: [start, catchNode, nextEnd],
      controlFlowConnections: [ctrl(start, catchNode), ctrl(catchNode, nextEnd)],
      dataFlowConnections: [
        dataEdge(start, catchNode, "x"),
        dataEdge(catchNode, nextEnd, "y"),
        dataEdge(catchNode, nextEnd, "caught_exception_info", "error_info"),
      ],
      inputs: [inp],
      outputs: [outp, errorInfo],
    });

    const graph = await loadFlow(flow, {
      toolRegistry: { ok_tool_default: () => "ok" },
    });
    const result = await graph.invoke({ inputs: { x: 5 } });
    expect(detailsOf(result)["branch"]).toBe("next");
    expect(outputsOf(result)["y"]).toBe("ok");
    expect(outputsOf(result)["error_info"]).toBeNull();
  });
});
