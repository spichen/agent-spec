/**
 * Generic LangGraph graph -> Agent Spec Flow conversion.
 *
 * Port of `pyagentspec.adapters.langgraph._agentspec_converter_flow`: every
 * LangGraph node becomes a ToolNode wrapping a synthetic ServerTool (or a
 * FlowNode for compiled-subgraph nodes), plain edges become control+data
 * edges over a single `state` property, conditional edges expand into a
 * conditional ToolNode plus a BranchingNode, and synthetic Start/End nodes
 * plus fall-through END edges complete the flow.
 *
 * Divergence from Python (see the adapter README): Python derives the
 * `state` property schemas from the pydantic/TypedDict state classes; the JS
 * builders expose channel maps instead, so the schemas list the channel keys
 * as untyped properties.
 */
import type { ComponentBase } from "../../component.js";
import type { ControlFlowEdge, DataFlowEdge, Flow } from "../../flows/index.js";
import {
  DEFAULT_BRANCH,
  DEFAULT_INPUT,
  createBranchingNode,
  createControlFlowEdge,
  createDataFlowEdge,
  createEndNode,
  createFlow,
  createFlowNode,
  createStartNode,
  createToolNode,
} from "../../flows/index.js";
import type { ComponentWithIO } from "../../component.js";
import type { JsonSchemaValue, Property } from "../../property.js";
import {
  propertyFromJsonSchema,
  stringProperty,
  unionProperty,
} from "../../property.js";
import { createServerTool } from "../../tools/index.js";
import type { RuntimeToAgentSpecConverter } from "../common/converters.js";
import type { BranchLike, BuilderLike } from "./graph-introspection.js";
import {
  definitionKeys,
  getGraphBuilder,
  isCompiledGraphLike,
  isStateGraphBuilderLike,
  stateSchemaKeys,
} from "./graph-introspection.js";

const START = "__start__";
const END = "__end__";

/** Build the `state` property listing the given state keys. */
function statePropertyFromKeys(keys: string[] | undefined): Property {
  const properties: Record<string, JsonSchemaValue> = {};
  for (const key of keys ?? []) {
    properties[key] = { title: key };
  }
  return propertyFromJsonSchema({
    title: "state",
    type: "object",
    properties,
  });
}

function getStateProperty(builder: BuilderLike): Property {
  return statePropertyFromKeys(stateSchemaKeys(builder));
}

function getInputProperty(builder: BuilderLike): Property {
  return statePropertyFromKeys(
    definitionKeys(builder._inputDefinition) ?? stateSchemaKeys(builder),
  );
}

function getOutputProperty(builder: BuilderLike): Property {
  return statePropertyFromKeys(
    definitionKeys(builder._outputDefinition) ?? stateSchemaKeys(builder),
  );
}

function getNodeInputProperty(
  builder: BuilderLike,
  nodeName: string,
): Property {
  const spec = builder.nodes[nodeName];
  const inputKeys = definitionKeys(spec?.input);
  return statePropertyFromKeys(inputKeys ?? stateSchemaKeys(builder));
}

/** Resolve the property describing data flowing towards the target nodes. */
function resolveOutputProperties(
  builder: BuilderLike,
  targetNodes: string[],
): Property {
  if (targetNodes.length === 0) {
    // Nodes without an explicit outgoing edge are routed to END, so the
    // property is the output schema of the entire graph.
    return getOutputProperty(builder);
  }
  if (targetNodes.length === 1) {
    const nodeName = targetNodes[0]!;
    if (nodeName === START) {
      return getInputProperty(builder);
    }
    if (nodeName === END) {
      return getOutputProperty(builder);
    }
    return getNodeInputProperty(builder, nodeName);
  }
  return unionProperty({
    title: "state",
    anyOf: targetNodes.map((nodeName) =>
      resolveOutputProperties(builder, [nodeName]),
    ),
  });
}

function graphEdges(builder: BuilderLike): [string, string][] {
  return [...(builder.edges ?? [])];
}

function asNode(component: ComponentBase | undefined): ComponentWithIO {
  if (component === undefined) {
    throw new Error("Internal error: referenced LangGraph node was not converted");
  }
  return component as ComponentWithIO;
}

/** Reject graphs with several conditional edges on the same source node. */
function validateConditionalEdgesSupport(builder: BuilderLike): void {
  for (const branchSpecs of Object.values(builder.branches ?? {})) {
    if (Object.keys(branchSpecs).length > 1) {
      throw new Error(
        "Conversion of multiple conditional edges with the same source node is not yet supported",
      );
    }
  }
}

/** Convert one non-subgraph LangGraph node into an Agent Spec ToolNode. */
function langgraphNodeConvertToAgentSpec(
  builder: BuilderLike,
  nodeName: string,
  referencedObjects: Map<string, ComponentBase>,
): ComponentBase {
  const existing = referencedObjects.get(nodeName);
  if (existing !== undefined) {
    const componentType = (existing as { componentType?: unknown })
      .componentType;
    if (typeof componentType !== "string" || !componentType.endsWith("Node")) {
      throw new Error(
        `expected node ${JSON.stringify(existing)} to be of type Node, got: ${String(componentType)}`,
      );
    }
    return existing;
  }

  const inputProperty = getNodeInputProperty(builder, nodeName);
  const targetNodes: string[] = [];
  for (const [from, to] of graphEdges(builder)) {
    if (from !== to && from === nodeName) {
      targetNodes.push(to);
    }
  }
  const outputProperty = resolveOutputProperties(builder, targetNodes);

  const tool = createServerTool({
    name: `${nodeName}_tool`,
    inputs: [inputProperty],
    outputs: [outputProperty],
  });
  const toolNode = createToolNode({
    name: nodeName,
    tool,
    inputs: [inputProperty],
    outputs: [outputProperty],
  });
  referencedObjects.set(nodeName, toolNode);
  return toolNode;
}

/** Create (or reuse) the flow's Start and End nodes. */
function getStartEndNodes(
  builder: BuilderLike,
  referencedObjects: Map<string, ComponentBase>,
): [ComponentBase, ComponentBase] {
  if (!referencedObjects.has(START)) {
    if (!(START in builder.nodes)) {
      referencedObjects.set(
        START,
        createStartNode({
          name: START,
          inputs: [getInputProperty(builder)],
          outputs: [getInputProperty(builder)],
        }),
      );
    } else {
      referencedObjects.set(
        START,
        langgraphNodeConvertToAgentSpec(builder, START, referencedObjects),
      );
    }
  }
  if (!referencedObjects.has(END)) {
    if (!(END in builder.nodes)) {
      referencedObjects.set(
        END,
        createEndNode({
          name: END,
          inputs: [getOutputProperty(builder)],
          outputs: [getOutputProperty(builder)],
        }),
      );
    } else {
      referencedObjects.set(
        END,
        langgraphNodeConvertToAgentSpec(builder, END, referencedObjects),
      );
    }
  }
  return [referencedObjects.get(START)!, referencedObjects.get(END)!];
}

function edgeToControlFlow(
  edge: [string, string],
  referencedObjects: Map<string, ComponentBase>,
): ControlFlowEdge {
  const [from, to] = edge;
  return createControlFlowEdge({
    name: `${from}_to_${to}`,
    fromNode: asNode(referencedObjects.get(from)),
    toNode: asNode(referencedObjects.get(to)),
  });
}

function edgeToDataFlow(
  builder: BuilderLike,
  edge: [string, string],
  referencedObjects: Map<string, ComponentBase>,
): DataFlowEdge {
  const [from, to] = edge;
  const internalStateProperty =
    from === START ? getInputProperty(builder) : getStateProperty(builder);
  const destinationInputProperty = resolveOutputProperties(builder, [to]);
  return createDataFlowEdge({
    name: `${from}_to_${to}_data_edge`,
    sourceNode: asNode(referencedObjects.get(from)),
    sourceOutput: internalStateProperty.title,
    destinationNode: asNode(referencedObjects.get(to)),
    destinationInput: destinationInputProperty.title,
  });
}

/**
 * Derive a unique name for a synthetic (conditional / branching) node.
 *
 * LangGraph JS stores every conditional edge's branch under the fixed default
 * key `"condition"`, so a user graph with a real node of that exact name (or
 * of a derived synthetic name) would collide: the synthetic node would
 * overwrite the real node in the registry and steal its edges. Keep the
 * Python-style base name in the common non-colliding case and suffix `_N`
 * only when a real or already-registered node claims it.
 */
function uniqueSyntheticNodeName(
  baseName: string,
  builder: BuilderLike,
  referencedObjects: Map<string, ComponentBase>,
): string {
  let candidate = baseName;
  let suffix = 1;
  while (
    Object.hasOwn(builder.nodes, candidate) ||
    referencedObjects.has(candidate)
  ) {
    candidate = `${baseName}_${suffix}`;
    suffix += 1;
  }
  return candidate;
}

/** Expand one conditional edge into conditional + branching nodes and edges. */
function branchConvertToAgentSpec(
  sourceNode: string,
  branchSpecs: Record<string, BranchLike>,
  builder: BuilderLike,
  referencedObjects: Map<string, ComponentBase>,
): [ComponentBase[], ControlFlowEdge[], DataFlowEdge[]] {
  const additionalNodes: ComponentBase[] = [];
  const additionalCtrlFlows: ControlFlowEdge[] = [];
  const additionalDataFlows: DataFlowEdge[] = [];

  for (const [branchKey, branchSpec] of Object.entries(branchSpecs)) {
    const ends = branchSpec.ends;
    if (ends === undefined || ends === null) {
      throw new Error(
        `Mapping for ${branchKey} not found.\n` +
          "            Make sure to add proper return type hints to the branching function.",
      );
    }
    const mapping: Record<string, string> = {};
    for (const [branchName, targetNodeName] of Object.entries(ends)) {
      mapping[String(branchName)] = targetNodeName;
    }

    // The synthetic node is named after the branch key (Python-style), unless
    // a real node claims that name — see uniqueSyntheticNodeName.
    const conditionalNodeName = uniqueSyntheticNodeName(
      branchKey,
      builder,
      referencedObjects,
    );

    // Create the conditional node to compute which branch to go to
    const conditionalNodeInput = resolveOutputProperties(builder, [sourceNode]);
    const conditionalNode = createToolNode({
      name: conditionalNodeName,
      tool: createServerTool({
        name: `${conditionalNodeName}_tool`,
        inputs: [conditionalNodeInput],
        outputs: [stringProperty({ title: DEFAULT_INPUT })],
      }),
    });
    additionalNodes.push(conditionalNode);
    referencedObjects.set(conditionalNodeName, conditionalNode);

    // The source node goes to the conditional node to compute which branch
    // to go to.
    additionalCtrlFlows.push(
      createControlFlowEdge({
        name: `${sourceNode}_to_${conditionalNodeName}`,
        fromNode: asNode(referencedObjects.get(sourceNode)),
        toNode: conditionalNode,
      }),
    );
    additionalDataFlows.push(
      createDataFlowEdge({
        name: `${sourceNode}_to_${conditionalNodeName}_data_edge`,
        sourceNode: asNode(referencedObjects.get(sourceNode)),
        sourceOutput: conditionalNodeInput.title,
        destinationNode: conditionalNode,
        destinationInput: conditionalNodeInput.title,
      }),
    );

    // Create the branching node for the current conditional edge
    const branchingNodeName = uniqueSyntheticNodeName(
      `${conditionalNodeName}_branching_node`,
      builder,
      referencedObjects,
    );
    const branchingNode = createBranchingNode({
      name: branchingNodeName,
      mapping,
    });
    additionalNodes.push(branchingNode);
    referencedObjects.set(branchingNodeName, branchingNode);

    additionalCtrlFlows.push(
      createControlFlowEdge({
        name: `${conditionalNodeName}_to_${branchingNodeName}`,
        fromNode: conditionalNode,
        toNode: branchingNode,
      }),
    );
    additionalDataFlows.push(
      createDataFlowEdge({
        name: `${conditionalNodeName}_to_${branchingNodeName}_data_edge`,
        sourceNode: conditionalNode,
        sourceOutput: DEFAULT_INPUT,
        destinationNode: branchingNode,
        destinationInput: DEFAULT_INPUT,
      }),
    );

    // For each different target node, create a control flow edge that goes
    // from the branching node to the target node if from_branch == branch.
    for (const [branchName, targetNodeName] of Object.entries(mapping)) {
      additionalCtrlFlows.push(
        createControlFlowEdge({
          name: `${branchingNodeName}_to_${targetNodeName}`,
          fromNode: branchingNode,
          toNode: asNode(referencedObjects.get(targetNodeName)),
          fromBranch: branchName,
        }),
      );
      additionalDataFlows.push(
        createDataFlowEdge({
          name: `data_${sourceNode}_to_${targetNodeName}`,
          sourceNode: asNode(referencedObjects.get(sourceNode)),
          sourceOutput: resolveOutputProperties(builder, [targetNodeName])
            .title,
          destinationNode: asNode(referencedObjects.get(targetNodeName)),
          destinationInput: resolveOutputProperties(builder, [targetNodeName])
            .title,
        }),
      );
    }

    // Create an edge for the default case, going straight to the end node.
    // This should "in practice" never be reached.
    const defaultEdgeName = `${branchingNodeName}_to_${END}`;
    if (!additionalCtrlFlows.some((flow) => flow.name === defaultEdgeName)) {
      additionalCtrlFlows.push(
        createControlFlowEdge({
          name: defaultEdgeName,
          fromNode: branchingNode,
          toNode: asNode(referencedObjects.get(END)),
          fromBranch: DEFAULT_BRANCH,
        }),
      );
    }
  }

  return [additionalNodes, additionalCtrlFlows, additionalDataFlows];
}

/**
 * Convert a LangGraph StateGraph (builder or compiled) into an Agent Spec
 * Flow of synthetic ToolNodes / FlowNodes plus Start/End nodes and edges.
 */
export function langgraphGraphConvertToAgentSpec(
  converter: RuntimeToAgentSpecConverter,
  graph: unknown,
  referencedObjects: Map<string, ComponentBase>,
): Flow {
  validateConditionalEdgesSupport(getGraphBuilder(graph));
  const flowName = isCompiledGraphLike(graph)
    ? String(graph.name ?? "LangGraph")
    : "LangGraph Flow";
  const builder = getGraphBuilder(graph);

  const nodes: ComponentBase[] = [];
  for (const [nodeName, nodeSpec] of Object.entries(builder.nodes)) {
    if (nodeName === START || nodeName === END) {
      continue;
    }
    const runnable = nodeSpec.runnable;
    if (isCompiledGraphLike(runnable) || isStateGraphBuilderLike(runnable)) {
      // Subgraph nodes convert recursively with a fresh registry, matching
      // Python.
      const subflow = converter.convert(runnable, new Map()) as Flow;
      const flowNode = createFlowNode({
        name: nodeName,
        subflow,
      });
      referencedObjects.set(nodeName, flowNode);
      nodes.push(flowNode);
    } else {
      nodes.push(
        langgraphNodeConvertToAgentSpec(builder, nodeName, referencedObjects),
      );
    }
  }

  const [startNode, endNode] = getStartEndNodes(builder, referencedObjects);
  nodes.push(startNode);
  nodes.push(endNode);

  const controlFlowEdges: ControlFlowEdge[] = [];
  const dataFlowEdges: DataFlowEdge[] = [];
  for (const edge of graphEdges(builder)) {
    controlFlowEdges.push(edgeToControlFlow(edge, referencedObjects));
    dataFlowEdges.push(edgeToDataFlow(builder, edge, referencedObjects));
  }

  for (const [sourceNode, branchSpecs] of Object.entries(
    builder.branches ?? {},
  )) {
    const [additionalNodes, additionalCtrlFlows, additionalDataFlows] =
      branchConvertToAgentSpec(
        sourceNode,
        branchSpecs,
        builder,
        referencedObjects,
      );
    nodes.push(...additionalNodes);
    controlFlowEdges.push(...additionalCtrlFlows);
    dataFlowEdges.push(...additionalDataFlows);
  }

  // Add missing edges towards END for nodes with no outgoing edges.
  for (const agentspecNode of nodes) {
    const nodeName = (agentspecNode as { name?: unknown }).name;
    if (nodeName === START || nodeName === END) {
      continue;
    }
    const hasOutgoing = controlFlowEdges.some(
      (ctrlFlow) =>
        (ctrlFlow.fromNode as { name?: unknown })["name"] === nodeName,
    );
    if (!hasOutgoing) {
      const edge: [string, string] = [String(nodeName), END];
      controlFlowEdges.push(edgeToControlFlow(edge, referencedObjects));
      dataFlowEdges.push(edgeToDataFlow(builder, edge, referencedObjects));
    }
  }

  return createFlow({
    name: flowName,
    startNode,
    nodes,
    controlFlowConnections: controlFlowEdges,
    dataFlowConnections: dataFlowEdges,
  });
}
