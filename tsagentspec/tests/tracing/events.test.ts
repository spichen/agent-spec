/**
 * Port of pyagentspec/tests/tracing/events/test_events.py.
 */
import { describe, expect, it } from "vitest";
import {
  AgentExecutionEnd,
  AgentExecutionStart,
  ExceptionRaised,
  FlowExecutionEnd,
  FlowExecutionStart,
  HumanInTheLoopRequest,
  HumanInTheLoopResponse,
  LlmGenerationChunkReceived,
  LlmGenerationRequest,
  LlmGenerationResponse,
  ManagerWorkersExecutionEnd,
  ManagerWorkersExecutionStart,
  Message,
  NodeExecutionEnd,
  NodeExecutionStart,
  PII_MASK,
  SwarmExecutionEnd,
  SwarmExecutionStart,
  ToolCall,
  ToolConfirmationRequest,
  ToolConfirmationResponse,
  ToolExecutionRequest,
  ToolExecutionResponse,
  ToolExecutionStreamingChunkReceived,
} from "../../src/index.js";
import {
  dummyAgent,
  dummyFlow,
  dummyLlmConfig,
  dummyManagerWorkers,
  dummyNode,
  dummySwarm,
  dummyTool,
} from "./fixtures.js";

describe("tracing events", () => {
  // Exception events
  it("creates and masks ExceptionRaised", () => {
    const event = new ExceptionRaised({
      exceptionType: "ValueError",
      exceptionMessage: "bad",
      exceptionStacktrace: "trace",
    });
    expect(event.exceptionType).toBe("ValueError");
    expect(event.exceptionMessage).toBe("bad");
    expect(typeof event.exceptionStacktrace).toBe("string");
    // Masking behavior
    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });
    expect(masked["exception_message"]).toBe(PII_MASK);
    expect(masked["exception_stacktrace"]).toBe(PII_MASK);
    expect(unmasked["exception_message"]).toBe("bad");
    expect(unmasked["exception_stacktrace"]).toBe("trace");
    expect(masked["type"]).toBe("ExceptionRaised");
  });

  it("masks sensitive fields by default", () => {
    const event = new ExceptionRaised({
      exceptionType: "ValueError",
      exceptionMessage: "bad",
    });
    // Default serialization masks (mirrors mask_sensitive_information=True default)
    expect(event.serialize()["exception_message"]).toBe(PII_MASK);
  });

  // Agent events
  it("creates and masks AgentExecutionStart", () => {
    const agent = dummyAgent();
    const event = new AgentExecutionStart({ agent, inputs: { x: 1 }, name: "custom" });
    expect(event.name).toBe("custom");
    expect(event.agent).toBe(agent);
    expect(event.inputs).toEqual({ x: 1 });
    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });
    expect(masked["inputs"]).toBe(PII_MASK);
    expect(unmasked["inputs"]).toEqual({ x: 1 });
    expect(masked["type"]).toBe("AgentExecutionStart");
  });

  it("creates and masks AgentExecutionEnd", () => {
    const agent = dummyAgent();
    const event = new AgentExecutionEnd({ agent, outputs: { y: 2 } });
    expect(event.agent).toBe(agent);
    expect(event.outputs).toEqual({ y: 2 });
    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });
    expect(masked["outputs"]).toBe(PII_MASK);
    expect(unmasked["outputs"]).toEqual({ y: 2 });
    expect(masked["type"]).toBe("AgentExecutionEnd");
  });

  // Flow events
  it("creates and masks FlowExecutionStart", () => {
    const flow = dummyFlow();
    const event = new FlowExecutionStart({
      flow,
      inputs: { a: 1 },
      name: "flow_start_custom",
    });
    expect(event.name).toBe("flow_start_custom");
    expect(event.flow).toBe(flow);
    expect(event.inputs).toEqual({ a: 1 });
    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });
    expect(masked["inputs"]).toBe(PII_MASK);
    expect(unmasked["inputs"]).toEqual({ a: 1 });
    expect(masked["type"]).toBe("FlowExecutionStart");
  });

  it("creates and masks FlowExecutionEnd", () => {
    const flow = dummyFlow();
    const event = new FlowExecutionEnd({
      flow,
      outputs: { b: 2 },
      branchSelected: "next",
    });
    expect(event.flow).toBe(flow);
    expect(event.outputs).toEqual({ b: 2 });
    expect(event.branchSelected).toBe("next");
    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });
    expect(masked["outputs"]).toBe(PII_MASK);
    expect(unmasked["outputs"]).toEqual({ b: 2 });
    expect(masked["branch_selected"]).toBe("next");
    expect(unmasked["branch_selected"]).toBe("next");
    expect(masked["type"]).toBe("FlowExecutionEnd");
  });

  // HITL events
  it("creates and masks HumanInTheLoopRequest", () => {
    const event = new HumanInTheLoopRequest({
      requestId: "r1",
      content: { question: "ok?" },
    });
    expect(event.requestId).toBe("r1");
    expect(event.content).toEqual({ question: "ok?" });
    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });
    expect(masked["content"]).toBe(PII_MASK);
    expect(unmasked["content"]).toEqual({ question: "ok?" });
    expect(masked["type"]).toBe("HumanInTheLoopRequest");
  });

  it("creates and masks HumanInTheLoopResponse", () => {
    const event = new HumanInTheLoopResponse({
      requestId: "r1",
      content: { answer: "yes" },
    });
    expect(event.requestId).toBe("r1");
    expect(event.content).toEqual({ answer: "yes" });
    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });
    expect(masked["content"]).toBe(PII_MASK);
    expect(unmasked["content"]).toEqual({ answer: "yes" });
    expect(masked["type"]).toBe("HumanInTheLoopResponse");
  });

  // LLM generation events
  it("creates and masks LlmGenerationRequest", () => {
    const llmConfig = dummyLlmConfig();
    const tool = dummyTool();
    const messages = [new Message({ content: "hello", role: "user" })];
    const event = new LlmGenerationRequest({
      llmConfig,
      prompt: messages,
      tools: [tool],
      requestId: "req-1",
    });
    expect(event.llmConfig).toBe(llmConfig);
    expect(event.prompt).toBe(messages);
    expect(event.tools).toEqual([tool]);
    expect(event.requestId).toBe("req-1");
    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });
    expect(masked["prompt"]).toBe(PII_MASK);
    expect(unmasked["prompt"]).toEqual([
      { id: null, content: "hello", sender: null, role: "user" },
    ]);
    const toolDumps = unmasked["tools"] as Array<Record<string, unknown>>;
    expect(toolDumps[0]!["name"]).toBe(tool.name);
    expect(masked["type"]).toBe("LlmGenerationRequest");
  });

  it("creates and masks LlmGenerationResponse", () => {
    const llmConfig = dummyLlmConfig();
    const event = new LlmGenerationResponse({
      llmConfig,
      content: "hi",
      requestId: "req-1",
      completionId: "c-1",
      toolCalls: [new ToolCall({ callId: "a", toolName: "b", arguments: "{'c': 1}" })],
      inputTokens: 10,
      outputTokens: 2,
    });
    expect(event.content).toBe("hi");
    expect(event.requestId).toBe("req-1");
    expect(event.completionId).toBe("c-1");
    expect(event.inputTokens).toBe(10);
    expect(event.outputTokens).toBe(2);
    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });
    expect(masked["content"]).toBe(PII_MASK);
    expect(masked["tool_calls"]).toBe(PII_MASK);
    expect(unmasked["content"]).toBe("hi");
    expect(unmasked["tool_calls"]).toEqual([
      { call_id: "a", tool_name: "b", arguments: "{'c': 1}" },
    ]);
    expect(unmasked["request_id"]).toBe("req-1");
    expect(unmasked["completion_id"]).toBe("c-1");
    expect(unmasked["input_tokens"]).toBe(10);
    expect(unmasked["output_tokens"]).toBe(2);
    expect(masked["type"]).toBe("LlmGenerationResponse");
  });

  it("creates and masks LlmGenerationChunkReceived", () => {
    const llmConfig = dummyLlmConfig();
    const event = new LlmGenerationChunkReceived({
      llmConfig,
      content: "piece",
      toolCalls: [new ToolCall({ callId: "a", toolName: "b", arguments: "{'c': 1}" })],
      requestId: "r",
      completionId: "c",
      outputTokens: 1,
    });
    expect(event.llmConfig).toBe(llmConfig);
    expect(event.content).toBe("piece");
    expect(event.requestId).toBe("r");
    expect(event.outputTokens).toBe(1);
    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });
    expect(masked["content"]).toBe(PII_MASK);
    expect(masked["tool_calls"]).toBe(PII_MASK);
    expect(unmasked["content"]).toBe("piece");
    expect(unmasked["tool_calls"]).toEqual([
      { call_id: "a", tool_name: "b", arguments: "{'c': 1}" },
    ]);
    expect(masked["type"]).toBe("LlmGenerationChunkReceived");
  });

  // Manager-workers events
  it("creates and masks ManagerWorkersExecutionStart", () => {
    const managerworkers = dummyManagerWorkers();
    const event = new ManagerWorkersExecutionStart({
      managerworkers,
      inputs: { foo: "bar" },
    });
    expect(event.managerworkers).toBe(managerworkers);
    expect(event.inputs).toEqual({ foo: "bar" });
    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });
    expect(masked["inputs"]).toBe(PII_MASK);
    expect(unmasked["inputs"]).toEqual({ foo: "bar" });
    expect(masked["type"]).toBe("ManagerWorkersExecutionStart");
  });

  it("creates and masks ManagerWorkersExecutionEnd", () => {
    const managerworkers = dummyManagerWorkers();
    const event = new ManagerWorkersExecutionEnd({
      managerworkers,
      outputs: { foo: "baz" },
    });
    expect(event.managerworkers).toBe(managerworkers);
    expect(event.outputs).toEqual({ foo: "baz" });
    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });
    expect(masked["outputs"]).toBe(PII_MASK);
    expect(unmasked["outputs"]).toEqual({ foo: "baz" });
    expect(masked["type"]).toBe("ManagerWorkersExecutionEnd");
  });

  // Node events
  it("creates and masks NodeExecutionStart", () => {
    const node = dummyNode();
    const event = new NodeExecutionStart({ node, inputs: { v: 3 } });
    expect(event.node).toBe(node);
    expect(event.inputs).toEqual({ v: 3 });
    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });
    expect(masked["inputs"]).toBe(PII_MASK);
    expect(unmasked["inputs"]).toEqual({ v: 3 });
    expect(masked["type"]).toBe("NodeExecutionStart");
  });

  it("creates and masks NodeExecutionEnd", () => {
    const node = dummyNode();
    const event = new NodeExecutionEnd({
      node,
      outputs: { v: 4 },
      branchSelected: "next",
    });
    expect(event.node).toBe(node);
    expect(event.outputs).toEqual({ v: 4 });
    expect(event.branchSelected).toBe("next");
    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });
    expect(masked["outputs"]).toBe(PII_MASK);
    expect(unmasked["outputs"]).toEqual({ v: 4 });
    expect(masked["branch_selected"]).toBe("next");
    expect(unmasked["branch_selected"]).toBe("next");
    expect(masked["type"]).toBe("NodeExecutionEnd");
  });

  // Swarm events
  it("creates and masks SwarmExecutionStart", () => {
    const swarm = dummySwarm();
    const event = new SwarmExecutionStart({ swarm, inputs: { q: "x" } });
    expect(event.swarm).toBe(swarm);
    expect(event.inputs).toEqual({ q: "x" });
    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });
    expect(masked["inputs"]).toBe(PII_MASK);
    expect(unmasked["inputs"]).toEqual({ q: "x" });
    expect(masked["type"]).toBe("SwarmExecutionStart");
  });

  it("creates and masks SwarmExecutionEnd", () => {
    const swarm = dummySwarm();
    const event = new SwarmExecutionEnd({ swarm, outputs: { r: "y" } });
    expect(event.swarm).toBe(swarm);
    expect(event.outputs).toEqual({ r: "y" });
    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });
    expect(masked["outputs"]).toBe(PII_MASK);
    expect(unmasked["outputs"]).toEqual({ r: "y" });
    expect(masked["type"]).toBe("SwarmExecutionEnd");
  });

  // Tool events
  it("creates and masks ToolExecutionRequest", () => {
    const tool = dummyTool();
    const event = new ToolExecutionRequest({ tool, inputs: { x: 1 }, requestId: "t1" });
    expect(event.tool).toBe(tool);
    expect(event.inputs).toEqual({ x: 1 });
    expect(event.requestId).toBe("t1");
    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });
    expect(masked["inputs"]).toBe(PII_MASK);
    expect(unmasked["inputs"]).toEqual({ x: 1 });
    expect(unmasked["request_id"]).toBe("t1");
    expect(masked["type"]).toBe("ToolExecutionRequest");
  });

  it("creates and masks ToolExecutionResponse", () => {
    const tool = dummyTool();
    const event = new ToolExecutionResponse({ tool, outputs: { y: 2 }, requestId: "t1" });
    expect(event.tool).toBe(tool);
    expect(event.outputs).toEqual({ y: 2 });
    expect(event.requestId).toBe("t1");
    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });
    expect(masked["outputs"]).toBe(PII_MASK);
    expect(unmasked["outputs"]).toEqual({ y: 2 });
    expect(unmasked["request_id"]).toBe("t1");
    expect(masked["type"]).toBe("ToolExecutionResponse");
  });

  it("creates and masks ToolExecutionStreamingChunkReceived", () => {
    const tool = dummyTool();
    const event = new ToolExecutionStreamingChunkReceived({
      tool,
      requestId: "t1",
      content: "piece",
    });
    expect(event.tool).toBe(tool);
    expect(event.content).toBe("piece");
    expect(event.requestId).toBe("t1");
    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });
    expect(masked["content"]).toBe(PII_MASK);
    expect(unmasked["content"]).toBe("piece");
    expect(unmasked["request_id"]).toBe("t1");
    expect(masked["type"]).toBe("ToolExecutionStreamingChunkReceived");
  });

  it("creates ToolConfirmationRequest (no sensitive fields)", () => {
    const tool = dummyTool();
    const event = new ToolConfirmationRequest({
      tool,
      requestId: "c1",
      toolExecutionRequestId: "t1",
    });
    expect(event.tool).toBe(tool);
    expect(event.requestId).toBe("c1");
    expect(event.toolExecutionRequestId).toBe("t1");
    // Masking behavior (no sensitive fields)
    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });
    expect(masked).toEqual(unmasked);
    expect(masked["type"]).toBe("ToolConfirmationRequest");
  });

  it("creates ToolConfirmationResponse (no sensitive fields)", () => {
    const tool = dummyTool();
    const event = new ToolConfirmationResponse({
      tool,
      executionConfirmed: true,
      requestId: "c1",
      toolExecutionRequestId: "t1",
    });
    expect(event.tool).toBe(tool);
    expect(event.executionConfirmed).toBe(true);
    expect(event.requestId).toBe("c1");
    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });
    expect(masked).toEqual(unmasked);
    expect(masked["execution_confirmed"]).toBe(true);
  });
});
