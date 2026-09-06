/**
 * Exporter tests for generic state graphs (LangGraph StateGraph -> Flow).
 *
 * Mirrors the state-graph portion of the Python suite
 * (`pyagentspec/tests/adapters/langgraph/test_langgraph_to_agentspec.py`):
 * plain edges, sink-node END synthesis, distinct input/output/node schemas,
 * conditional edges (including the "condition" name collision), subgraph
 * recursion and the documented conditional-edge rejections. Exercises the
 * `agentspec-converter-flow.ts` pipeline behind `AgentSpecExporter`. All
 * tests run offline: node functions are plain closures, never chat models.
 */
import { describe, expect, it } from "vitest";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { DEFAULT_BRANCH, DEFAULT_INPUT } from "../../../src/index.js";
import type { Flow, Property, ServerTool } from "../../../src/index.js";
import { AgentSpecExporter } from "../../../src/adapters/langgraph/agentspec-exporter.js";

/** Structural view of an exported flow node used by the assertions. */
interface ExportedNodeView {
  id: string;
  componentType: string;
  name: string;
  inputs?: Property[];
  outputs?: Property[];
  tool?: ServerTool;
  mapping?: Record<string, string>;
  branches?: string[];
  subflow?: Flow;
}

function nodesOf(flow: Flow): ExportedNodeView[] {
  return flow.nodes as unknown as ExportedNodeView[];
}

function nodeNamed(flow: Flow, name: string): ExportedNodeView {
  const node = nodesOf(flow).find((candidate) => candidate.name === name);
  if (node === undefined) {
    throw new Error(`Flow has no node named '${name}'.`);
  }
  return node;
}

function controlFlowNames(flow: Flow): string[] {
  return flow.controlFlowConnections.map((edge) => edge.name ?? "");
}

function dataFlowNames(flow: Flow): string[] {
  return (flow.dataFlowConnections ?? []).map((edge) => edge.name ?? "");
}

/** Keys listed by a synthetic `state` property. */
function statePropertyKeys(property: Property): string[] {
  return Object.keys(
    (property.jsonSchema["properties"] as Record<string, unknown>) ?? {},
  );
}

describe("AgentSpecExporter: state graph flows", () => {
  const CodeGenState = Annotation.Root({
    language: Annotation<string>,
    request: Annotation<string>,
    output: Annotation<string>,
  });

  it("converts a linear compiled graph into a Flow", () => {
    const exporter = new AgentSpecExporter();
    const graph = new StateGraph(CodeGenState)
      .addNode("llm_code_gen", () => ({ output: "generated" }))
      .addEdge(START, "llm_code_gen")
      .addEdge("llm_code_gen", END);
    const compiled = graph.compile({ name: "CodeGen Assistant" });

    const flow = exporter.toComponent(compiled) as Flow;

    expect(flow.componentType).toBe("Flow");
    expect(flow.name).toBe("CodeGen Assistant");
    // llm_code_gen + synthesized __start__ + __end__
    expect(flow.nodes).toHaveLength(3);
    expect(
      nodesOf(flow).map((node) => [node.componentType, node.name]),
    ).toEqual([
      ["ToolNode", "llm_code_gen"],
      ["StartNode", "__start__"],
      ["EndNode", "__end__"],
    ]);
    // One ctrl+data pair per LangGraph edge, with the Python edge names.
    expect(controlFlowNames(flow)).toEqual([
      "__start___to_llm_code_gen",
      "llm_code_gen_to___end__",
    ]);
    expect(dataFlowNames(flow)).toEqual([
      "__start___to_llm_code_gen_data_edge",
      "llm_code_gen_to___end___data_edge",
    ]);

    // The synthetic tool mirrors the node, over a single `state` property
    // listing the channel keys.
    const toolNode = nodeNamed(flow, "llm_code_gen");
    expect(toolNode.tool?.componentType).toBe("ServerTool");
    expect(toolNode.tool?.name).toBe("llm_code_gen_tool");
    const toolInput = toolNode.tool?.inputs[0] as Property;
    expect(toolInput.title).toBe("state");
    expect(toolInput.type).toBe("object");
    expect(statePropertyKeys(toolInput)).toEqual([
      "language",
      "request",
      "output",
    ]);

    // Flow inputs/outputs are inferred from the synthesized start/end nodes.
    expect(flow.inputs?.map((input) => input.title)).toEqual(["state"]);
    expect(flow.outputs?.map((output) => output.title)).toEqual(["state"]);
    expect(statePropertyKeys(flow.inputs?.[0] as Property)).toEqual([
      "language",
      "request",
      "output",
    ]);
  });

  it("converts an uncompiled builder into a Flow with the default name", () => {
    const exporter = new AgentSpecExporter();
    const graph = new StateGraph(CodeGenState)
      .addNode("llm_code_gen", () => ({ output: "generated" }))
      .addEdge(START, "llm_code_gen")
      .addEdge("llm_code_gen", END);

    const flow = exporter.toComponent(graph) as Flow;

    expect(flow.componentType).toBe("Flow");
    expect(flow.name).toBe("LangGraph Flow");
  });

  it("synthesizes END edges for sink nodes without outgoing edges", () => {
    const exporter = new AgentSpecExporter();
    const graph = new StateGraph(CodeGenState)
      .addNode("sink", () => ({}))
      .addEdge(START, "sink");

    const flow = exporter.toComponent(graph.compile()) as Flow;

    expect(flow.nodes).toHaveLength(3);
    expect(controlFlowNames(flow)).toEqual([
      "__start___to_sink",
      "sink_to___end__",
    ]);
    expect(dataFlowNames(flow)).toEqual([
      "__start___to_sink_data_edge",
      "sink_to___end___data_edge",
    ]);
  });

  it("converts a graph with distinct input/output/node schemas", () => {
    // Per-node `input` options are the JS equivalent of the Python function
    // annotations the Python adapter introspects.
    const exporter = new AgentSpecExporter();
    const InputSchema = Annotation.Root({ city: Annotation<string> });
    const OutputSchema = Annotation.Root({ response: Annotation<string> });
    const WeatherSchema = Annotation.Root({
      weather_data: Annotation<string>,
    });
    const InternalState = Annotation.Root({
      city: Annotation<string>,
      weather_data: Annotation<string>,
      response: Annotation<string>,
    });
    const graph = new StateGraph({
      state: InternalState,
      input: InputSchema,
      output: OutputSchema,
    })
      .addNode("get_weather", () => ({ weather_data: "sunny" }), {
        input: InputSchema,
      })
      .addNode("llm_node", () => ({ response: "reformulated" }), {
        input: WeatherSchema,
      })
      .addEdge(START, "get_weather")
      .addEdge("get_weather", "llm_node")
      .addEdge("llm_node", END);

    const flow = exporter.toComponent(graph.compile({ name: "Weather Flow" })) as Flow;

    expect(flow.name).toBe("Weather Flow");
    // get_weather + llm_node + __start__ + __end__
    expect(flow.nodes).toHaveLength(4);
    expect(flow.controlFlowConnections).toHaveLength(3);
    expect(flow.dataFlowConnections).toHaveLength(3);
    const startNode = nodeNamed(flow, "__start__");
    const endNode = nodeNamed(flow, "__end__");
    expect(statePropertyKeys(startNode.outputs?.[0] as Property)).toEqual([
      "city",
    ]);
    expect(statePropertyKeys(endNode.outputs?.[0] as Property)).toEqual([
      "response",
    ]);
    const getWeatherNode = nodeNamed(flow, "get_weather");
    expect(statePropertyKeys(getWeatherNode.inputs?.[0] as Property)).toEqual([
      "city",
    ]);
    expect(statePropertyKeys(getWeatherNode.outputs?.[0] as Property)).toEqual([
      "weather_data",
    ]);
  });

  it("expands a conditional edge into a conditional ToolNode plus a BranchingNode", () => {
    const exporter = new AgentSpecExporter();
    const CaseState = Annotation.Root({ sentence: Annotation<string> });
    const graph = new StateGraph(CaseState)
      .addNode("lowercase", () => ({}))
      .addNode("uppercase", () => ({}))
      .addNode("messycase", () => ({}))
      .addConditionalEdges(START, () => "lowercase", {
        lowercase: "lowercase",
        uppercase: "uppercase",
        messycase: "messycase",
      });

    const flow = exporter.toComponent(
      graph.compile({ name: "Casecheck Flow" }),
    ) as Flow;

    expect(flow.name).toBe("Casecheck Flow");
    // 3 case nodes + __start__ + __end__ + conditional node + branching node
    expect(flow.nodes).toHaveLength(7);

    // The conditional ToolNode computes the branch name (LangGraph JS names
    // every conditional branch "condition").
    const conditionalNode = nodeNamed(flow, "condition");
    expect(conditionalNode.componentType).toBe("ToolNode");
    expect(conditionalNode.tool?.name).toBe("condition_tool");
    expect(conditionalNode.tool?.outputs.map((output) => output.title)).toEqual(
      [DEFAULT_INPUT],
    );

    const branchingNode = nodeNamed(flow, "condition_branching_node");
    expect(branchingNode.componentType).toBe("BranchingNode");
    expect(branchingNode.mapping).toEqual({
      lowercase: "lowercase",
      uppercase: "uppercase",
      messycase: "messycase",
    });
    expect(new Set(branchingNode.branches)).toEqual(
      new Set([DEFAULT_BRANCH, "lowercase", "uppercase", "messycase"]),
    );

    // Control edges: source -> conditional -> branching -> per-branch targets
    // plus the default fall-through to END and auto-END edges for the sinks.
    const edgesWithBranch = flow.controlFlowConnections.map((edge) => [
      edge.name,
      edge.fromBranch,
    ]);
    expect(edgesWithBranch).toEqual([
      ["__start___to_condition", undefined],
      ["condition_to_condition_branching_node", undefined],
      ["condition_branching_node_to_lowercase", "lowercase"],
      ["condition_branching_node_to_uppercase", "uppercase"],
      ["condition_branching_node_to_messycase", "messycase"],
      ["condition_branching_node_to___end__", DEFAULT_BRANCH],
      ["lowercase_to___end__", undefined],
      ["uppercase_to___end__", undefined],
      ["messycase_to___end__", undefined],
    ]);

    const dataNames = dataFlowNames(flow);
    expect(dataNames).toContain("__start___to_condition_data_edge");
    expect(dataNames).toContain(
      "condition_to_condition_branching_node_data_edge",
    );
    expect(dataNames).toContain("data___start___to_lowercase");
    const branchingDataEdge = (flow.dataFlowConnections ?? []).find(
      (edge) => edge.name === "condition_to_condition_branching_node_data_edge",
    );
    expect(branchingDataEdge?.sourceOutput).toBe(DEFAULT_INPUT);
    expect(branchingDataEdge?.destinationInput).toBe(DEFAULT_INPUT);
  });

  it("keeps a real node named 'condition' distinct from the synthetic conditional node", () => {
    // LangGraph JS stores every conditional edge's branch under the fixed key
    // "condition"; a user node with that literal name must not be overwritten
    // by the synthetic conditional ToolNode. The synthetic names are suffixed
    // instead (only in the colliding case).
    const exporter = new AgentSpecExporter();
    const CaseState = Annotation.Root({ sentence: Annotation<string> });
    const graph = new StateGraph(CaseState)
      .addNode("condition", () => ({}))
      .addNode("other", () => ({}))
      .addConditionalEdges(START, () => "condition", {
        condition: "condition",
        other: "other",
      });

    const flow = exporter.toComponent(
      graph.compile({ name: "Collision Flow" }),
    ) as Flow;

    // 2 real nodes + __start__ + __end__ + conditional node + branching node.
    expect(flow.nodes).toHaveLength(6);

    const realNode = nodeNamed(flow, "condition");
    expect(realNode.componentType).toBe("ToolNode");
    expect(realNode.tool?.name).toBe("condition_tool");

    const conditionalNode = nodeNamed(flow, "condition_1");
    expect(conditionalNode.componentType).toBe("ToolNode");
    expect(conditionalNode.tool?.name).toBe("condition_1_tool");

    const branchingNode = nodeNamed(flow, "condition_1_branching_node");
    expect(branchingNode.componentType).toBe("BranchingNode");
    expect(branchingNode.mapping).toEqual({
      condition: "condition",
      other: "other",
    });

    // The branch-target edge is wired to the REAL node, not the synthetic one.
    const branchTargetEdge = flow.controlFlowConnections.find(
      (edge) => edge.name === "condition_1_branching_node_to_condition",
    );
    expect(branchTargetEdge?.fromBranch).toBe("condition");
    expect((branchTargetEdge?.toNode as unknown as ExportedNodeView).id).toBe(
      realNode.id,
    );

    // And the real node stays connected downstream (auto edge to END).
    expect(controlFlowNames(flow)).toContain("condition_to___end__");
  });

  it("rejects a conditional edge without a path map", () => {
    const exporter = new AgentSpecExporter();
    const CaseState = Annotation.Root({ sentence: Annotation<string> });
    const graph = new StateGraph(CaseState)
      .addNode("lowercase", () => ({}))
      .addConditionalEdges(START, () => "lowercase");

    expect(() => exporter.toComponent(graph.compile())).toThrow(
      "Mapping for condition not found.\n" +
        "            Make sure to add proper return type hints to the branching function.",
    );
  });

  it("rejects multiple conditional edges with the same source node", () => {
    const exporter = new AgentSpecExporter();
    const CaseState = Annotation.Root({ sentence: Annotation<string> });
    const graph = new StateGraph(CaseState)
      .addNode("node_a", () => ({}))
      .addNode("node_b", () => ({}))
      .addConditionalEdges(START, () => "node_a", { go: "node_a" });
    // LangGraph JS names every conditional branch "condition" and refuses a
    // second one on the same source, so the runtime shape the exporter guards
    // against is reproduced on the builder directly.
    const branches = (
      graph as unknown as {
        branches: Record<string, Record<string, unknown>>;
      }
    ).branches;
    branches[START]!["condition2"] = branches[START]!["condition"]!;

    expect(() => exporter.toComponent(graph)).toThrow(
      "Conversion of multiple conditional edges with the same source node is not yet supported",
    );
  });

  it("converts subgraph nodes into FlowNodes recursively", () => {
    const exporter = new AgentSpecExporter();
    const SubState = Annotation.Root({ foo: Annotation<string> });
    const subgraph = new StateGraph(SubState)
      .addNode("subgraph_node_1", (state) => ({ foo: `hi! ${state.foo}` }))
      .addEdge(START, "subgraph_node_1")
      .compile();
    const parent = new StateGraph(SubState)
      .addNode("node_1", subgraph)
      .addEdge(START, "node_1");
    const compiled = parent.compile({ name: "GraphWithSubgraph" });

    const flow = exporter.toComponent(compiled) as Flow;

    expect(flow.componentType).toBe("Flow");
    expect(flow.name).toBe("GraphWithSubgraph");
    const flowNodes = nodesOf(flow).filter(
      (node) => node.componentType === "FlowNode",
    );
    expect(flowNodes).toHaveLength(1);
    expect(flowNodes[0]!.name).toBe("node_1");

    const subflow = flowNodes[0]!.subflow as Flow;
    expect(subflow.componentType).toBe("Flow");
    // Both levels synthesize __start__/__end__ around their single node.
    expect(flow.nodes).toHaveLength(3);
    expect(subflow.nodes).toHaveLength(3);
    expect(
      nodesOf(subflow).map((node) => [node.componentType, node.name]),
    ).toEqual([
      ["ToolNode", "subgraph_node_1"],
      ["StartNode", "__start__"],
      ["EndNode", "__end__"],
    ]);
    // Explicit edge + implicit edge to END at each level.
    expect(controlFlowNames(flow)).toEqual([
      "__start___to_node_1",
      "node_1_to___end__",
    ]);
    expect(controlFlowNames(subflow as Flow)).toEqual([
      "__start___to_subgraph_node_1",
      "subgraph_node_1_to___end__",
    ]);
  });
});
