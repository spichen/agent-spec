/**
 * Shared fixtures for the tracing test suites — port of
 * pyagentspec/tests/tracing/conftest.py.
 */
import {
  createAgent,
  createControlFlowEdge,
  createDataFlowEdge,
  createEndNode,
  createFlow,
  createLlmNode,
  createManagerWorkers,
  createOpenAiConfig,
  createServerTool,
  createStartNode,
  createSwarm,
  integerProperty,
  stringProperty,
  type Agent,
  type Flow,
  type LlmConfig,
  type LlmNode,
  type ManagerWorkers,
  type ServerTool,
  type Swarm,
} from "../../src/index.js";

export function dummyLlmConfig(): LlmConfig {
  return createOpenAiConfig({ name: "openai", modelId: "gpt-test" });
}

export function dummyAgent(llmConfig: LlmConfig = dummyLlmConfig()): Agent {
  return createAgent({ name: "agent", llmConfig, systemPrompt: "Hello" });
}

export function dummyTool(): ServerTool {
  return createServerTool({
    name: "servertool",
    inputs: [integerProperty({ title: "x" })],
    outputs: [integerProperty({ title: "y" })],
  });
}

export function dummyFlow(llmConfig: LlmConfig = dummyLlmConfig()): Flow {
  const promptProp = stringProperty({ title: "prompt" });
  const llmOutProp = stringProperty({ title: "generated_text" });
  const startNode = createStartNode({
    name: "start",
    inputs: [promptProp],
    outputs: [promptProp],
  });
  const llmNode = createLlmNode({
    name: "llm",
    llmConfig,
    promptTemplate: "{{prompt}}",
    inputs: [promptProp],
    outputs: [llmOutProp],
  });
  const endNode = createEndNode({
    name: "end",
    inputs: [llmOutProp],
    outputs: [llmOutProp],
  });
  const controlFlowEdges = [
    createControlFlowEdge({ name: "s_to_llm", fromNode: startNode, toNode: llmNode }),
    createControlFlowEdge({ name: "llm_to_e", fromNode: llmNode, toNode: endNode }),
  ];
  const dataFlowEdges = [
    createDataFlowEdge({
      name: "prompt_edge",
      sourceNode: startNode,
      sourceOutput: "prompt",
      destinationNode: llmNode,
      destinationInput: "prompt",
    }),
    createDataFlowEdge({
      name: "out_edge",
      sourceNode: llmNode,
      sourceOutput: "generated_text",
      destinationNode: endNode,
      destinationInput: "generated_text",
    }),
  ];
  return createFlow({
    name: "flow",
    startNode,
    nodes: [startNode, llmNode, endNode],
    controlFlowConnections: controlFlowEdges,
    dataFlowConnections: dataFlowEdges,
  });
}

export function dummyNode(llmConfig: LlmConfig = dummyLlmConfig()): LlmNode {
  return createLlmNode({
    name: "llm_node",
    llmConfig,
    promptTemplate: "{{prompt}}",
    inputs: [stringProperty({ title: "prompt" })],
    outputs: [stringProperty({ title: "generated_text" })],
  });
}

export function dummyManagerWorkers(
  llmConfig: LlmConfig = dummyLlmConfig(),
): ManagerWorkers {
  const manager = createAgent({
    name: "manager",
    llmConfig,
    systemPrompt: "You are a manager",
  });
  const worker = createAgent({
    name: "worker",
    llmConfig,
    systemPrompt: "You are a worker",
  });
  return createManagerWorkers({ name: "mw", groupManager: manager, workers: [worker] });
}

export function dummySwarm(llmConfig: LlmConfig = dummyLlmConfig()): Swarm {
  const a1 = createAgent({ name: "a1", llmConfig, systemPrompt: "You are a1" });
  const a2 = createAgent({ name: "a2", llmConfig, systemPrompt: "You are a2" });
  return createSwarm({ name: "sw", firstAgent: a1, relationships: [[a1, a2]] });
}
