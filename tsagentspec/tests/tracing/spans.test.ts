/**
 * Port of pyagentspec/tests/tracing/spans/test_spans.py.
 */
import { describe, expect, it } from "vitest";
import {
  AgentExecutionSpan,
  Event,
  FlowExecutionSpan,
  LlmGenerationSpan,
  ManagerWorkersExecutionSpan,
  NodeExecutionSpan,
  SwarmExecutionSpan,
  ToolExecutionSpan,
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

function dummyEvent(): Event {
  return new Event({ id: "dummy_event_id", name: "dummy_event" });
}

describe("tracing spans", () => {
  it("creates an AgentExecutionSpan", async () => {
    const agent = dummyAgent();
    const span = new AgentExecutionSpan({ agent, name: "custom_agent_span" });
    expect(span.name).toBe("custom_agent_span");
    expect(span.agent).toBe(agent);
    const event = dummyEvent();
    await span.addEvent(event);
    expect(span.events).toHaveLength(1);
    expect(span.events[0]).toBe(event);
    // Masking behavior (no sensitive fields in spans)
    const masked = span.serialize({ maskSensitiveInformation: true });
    const unmasked = span.serialize({ maskSensitiveInformation: false });
    expect(masked).toEqual(unmasked);
    expect(masked["type"]).toBe("AgentExecutionSpan");
  });

  it("creates a FlowExecutionSpan", async () => {
    const flow = dummyFlow();
    const span = new FlowExecutionSpan({ flow, name: "custom_flow_span" });
    expect(span.name).toBe("custom_flow_span");
    expect(span.flow).toBe(flow);
    const event = dummyEvent();
    await span.addEvent(event);
    expect(span.events).toHaveLength(1);
    expect(span.events[0]).toBe(event);
    const masked = span.serialize({ maskSensitiveInformation: true });
    const unmasked = span.serialize({ maskSensitiveInformation: false });
    expect(masked).toEqual(unmasked);
    expect(masked["type"]).toBe("FlowExecutionSpan");
  });

  it("creates an LlmGenerationSpan", async () => {
    const llmConfig = dummyLlmConfig();
    const span = new LlmGenerationSpan({ llmConfig, name: "custom_llm_span" });
    expect(span.name).toBe("custom_llm_span");
    expect(span.llmConfig).toBe(llmConfig);
    const event = dummyEvent();
    await span.addEvent(event);
    expect(span.events).toHaveLength(1);
    expect(span.events[0]).toBe(event);
    const masked = span.serialize({ maskSensitiveInformation: true });
    const unmasked = span.serialize({ maskSensitiveInformation: false });
    expect(masked).toEqual(unmasked);
    expect(masked["type"]).toBe("LlmGenerationSpan");
  });

  it("creates a ManagerWorkersExecutionSpan", async () => {
    const managerworkers = dummyManagerWorkers();
    const span = new ManagerWorkersExecutionSpan({
      managerworkers,
      name: "custom_mw_span",
    });
    expect(span.name).toBe("custom_mw_span");
    expect(span.managerworkers).toBe(managerworkers);
    const event = dummyEvent();
    await span.addEvent(event);
    expect(span.events).toHaveLength(1);
    expect(span.events[0]).toBe(event);
    const masked = span.serialize({ maskSensitiveInformation: true });
    const unmasked = span.serialize({ maskSensitiveInformation: false });
    expect(masked).toEqual(unmasked);
    expect(masked["type"]).toBe("ManagerWorkersExecutionSpan");
  });

  it("creates a NodeExecutionSpan", async () => {
    const node = dummyNode();
    const span = new NodeExecutionSpan({ node, name: "custom_node_span" });
    expect(span.name).toBe("custom_node_span");
    expect(span.node).toBe(node);
    const event = dummyEvent();
    await span.addEvent(event);
    expect(span.events).toHaveLength(1);
    expect(span.events[0]).toBe(event);
    const masked = span.serialize({ maskSensitiveInformation: true });
    const unmasked = span.serialize({ maskSensitiveInformation: false });
    expect(masked).toEqual(unmasked);
    expect(masked["type"]).toBe("NodeExecutionSpan");
  });

  it("creates a SwarmExecutionSpan", async () => {
    const swarm = dummySwarm();
    const span = new SwarmExecutionSpan({ swarm, name: "custom_swarm_span" });
    expect(span.name).toBe("custom_swarm_span");
    expect(span.swarm).toBe(swarm);
    const event = dummyEvent();
    await span.addEvent(event);
    expect(span.events).toHaveLength(1);
    expect(span.events[0]).toBe(event);
    const masked = span.serialize({ maskSensitiveInformation: true });
    const unmasked = span.serialize({ maskSensitiveInformation: false });
    expect(masked).toEqual(unmasked);
    expect(masked["type"]).toBe("SwarmExecutionSpan");
  });

  it("creates a ToolExecutionSpan", async () => {
    const tool = dummyTool();
    const span = new ToolExecutionSpan({ tool, name: "custom_tool_span" });
    expect(span.name).toBe("custom_tool_span");
    expect(span.tool).toBe(tool);
    const event = dummyEvent();
    await span.addEvent(event);
    expect(span.events).toHaveLength(1);
    expect(span.events[0]).toBe(event);
    const masked = span.serialize({ maskSensitiveInformation: true });
    const unmasked = span.serialize({ maskSensitiveInformation: false });
    expect(masked).toEqual(unmasked);
    expect(masked["type"]).toBe("ToolExecutionSpan");
  });

  it("span dumps use snake_case wire names and embed the component", () => {
    const agent = dummyAgent();
    const span = new AgentExecutionSpan({ agent, name: "custom_agent_span" });
    const dump = span.serialize();
    expect(Object.keys(dump)).toEqual(
      expect.arrayContaining([
        "id",
        "name",
        "description",
        "start_time",
        "end_time",
        "events",
        "metadata",
        "agent",
        "type",
      ]),
    );
    expect(dump["start_time"]).toBeNull();
    expect(dump["end_time"]).toBeNull();
    const agentDump = dump["agent"] as Record<string, unknown>;
    expect(agentDump["component_type"]).toBe("Agent");
    expect(agentDump["name"]).toBe("agent");
    // Bookkeeping fields never serialize
    expect(dump).not.toHaveProperty("parentSpan");
    expect(dump).not.toHaveProperty("parent_span");
  });
});
