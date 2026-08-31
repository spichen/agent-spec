/**
 * Tests for the adapter component load policy.
 *
 * Ports `pyagentspec.serialization.componentpolicy` semantics: without an
 * allow list everything is allowed unless blocked; with one, only matching
 * types load. Concrete entries beat abstract group entries beat the
 * `Component` wildcards, and block entries win same-distance ties. The
 * loaders block `StdioTransport` by default.
 */
import { describe, expect, it } from "vitest";
import {
  createAgent,
  createMCPTool,
  createStdioTransport,
  createVllmConfig,
} from "../../../src/index.js";
import { ComponentLoadPolicy } from "../../../src/adapters/common/component-policy.js";
import { AgentSpecLoader } from "../../../src/adapters/langgraph/agentspec-loader.js";

function blockedError(componentType: string): string {
  return `Loading Agent Spec component type \`${componentType}\` is in the block list.`;
}

function notAllowedError(componentType: string): string {
  return `Loading Agent Spec component type \`${componentType}\` is not in the allow list.`;
}

describe("ComponentLoadPolicy.validateComponentType", () => {
  it("allows everything by default", () => {
    const policy = new ComponentLoadPolicy();
    expect(() => policy.validateComponentType("Agent")).not.toThrow();
    expect(() => policy.validateComponentType("StdioTransport")).not.toThrow();
    expect(() => policy.validateComponentType("SomePluginType")).not.toThrow();
  });

  it("blocks a concrete component type with the Python error text", () => {
    const policy = new ComponentLoadPolicy(undefined, ["StdioTransport"]);
    expect(() => policy.validateComponentType("StdioTransport")).toThrow(
      blockedError("StdioTransport"),
    );
    expect(() => policy.validateComponentType("SSETransport")).not.toThrow();
  });

  it("restricts loading to the allow list when one is given", () => {
    const policy = new ComponentLoadPolicy(["Agent"]);
    expect(() => policy.validateComponentType("Agent")).not.toThrow();
    expect(() => policy.validateComponentType("Swarm")).toThrow(
      notAllowedError("Swarm"),
    );
  });

  it("accepts a single string as the policy input", () => {
    const policy = new ComponentLoadPolicy("Agent", "Swarm");
    expect(() => policy.validateComponentType("Agent")).not.toThrow();
    expect(() => policy.validateComponentType("Swarm")).toThrow(
      blockedError("Swarm"),
    );
  });

  it("matches abstract group names against their concrete members", () => {
    const allowPolicy = new ComponentLoadPolicy(["LlmConfig"]);
    expect(() => allowPolicy.validateComponentType("VllmConfig")).not.toThrow();
    expect(() => allowPolicy.validateComponentType("Agent")).toThrow(
      notAllowedError("Agent"),
    );

    const blockPolicy = new ComponentLoadPolicy(undefined, ["Tool"]);
    expect(() => blockPolicy.validateComponentType("ServerTool")).toThrow(
      blockedError("ServerTool"),
    );
    expect(() => blockPolicy.validateComponentType("Agent")).not.toThrow();
  });

  it("concrete allow entry beats an abstract block entry", () => {
    const policy = new ComponentLoadPolicy(["ServerTool"], ["Tool"]);
    expect(() => policy.validateComponentType("ServerTool")).not.toThrow();
    expect(() => policy.validateComponentType("ClientTool")).toThrow(
      blockedError("ClientTool"),
    );
  });

  it("concrete block entry beats an abstract allow entry", () => {
    const policy = new ComponentLoadPolicy(["Tool"], ["ServerTool"]);
    expect(() => policy.validateComponentType("ServerTool")).toThrow(
      blockedError("ServerTool"),
    );
    expect(() => policy.validateComponentType("ClientTool")).not.toThrow();
  });

  it("block wins same-distance ties", () => {
    const concreteTie = new ComponentLoadPolicy(["ServerTool"], ["ServerTool"]);
    expect(() => concreteTie.validateComponentType("ServerTool")).toThrow(
      blockedError("ServerTool"),
    );

    const wildcardTie = new ComponentLoadPolicy(["Component"], ["Component"]);
    expect(() => wildcardTie.validateComponentType("Agent")).toThrow(
      blockedError("Agent"),
    );
  });

  it("Component wildcard matches unknown plugin-defined types", () => {
    const blockAll = new ComponentLoadPolicy(undefined, ["Component"]);
    expect(() => blockAll.validateComponentType("MyPluginType")).toThrow(
      blockedError("MyPluginType"),
    );

    const allowAll = new ComponentLoadPolicy(["Component"]);
    expect(() => allowAll.validateComponentType("MyPluginType")).not.toThrow();
  });

  it("ComponentWithIO wildcard only matches IO component types", () => {
    const policy = new ComponentLoadPolicy(undefined, ["ComponentWithIO"]);
    expect(() => policy.validateComponentType("Agent")).toThrow(
      blockedError("Agent"),
    );
    // LLM configs do not extend ComponentWithIO.
    expect(() => policy.validateComponentType("VllmConfig")).not.toThrow();
  });

  it("an unknown policy name matches only that exact componentType", () => {
    const policy = new ComponentLoadPolicy(undefined, ["SomethingCustom"]);
    expect(() => policy.validateComponentType("SomethingCustom")).toThrow(
      blockedError("SomethingCustom"),
    );
    expect(() => policy.validateComponentType("Agent")).not.toThrow();
  });

  it("rejects non-string policy entries with the Python error text", () => {
    expect(
      () => new ComponentLoadPolicy([123 as unknown as string]),
    ).toThrow(
      "`allowed_components` and `blocked_components` entries must be component " +
        "type names or Component classes, got 123.",
    );
  });
});

describe("ComponentLoadPolicy.validateComponentTree", () => {
  const stdioTransport = createStdioTransport({
    name: "stdio",
    command: "echo",
  });
  const agentWithNestedTransport = createAgent({
    name: "agent",
    systemPrompt: "You are a helpful agent.",
    llmConfig: createVllmConfig({
      name: "llm",
      url: "http://localhost:8000",
      modelId: "m",
    }),
    tools: [
      createMCPTool({ name: "fooza_tool", clientTransport: stdioTransport }),
    ],
  });

  it("catches a blocked component nested deep in the tree", () => {
    const policy = new ComponentLoadPolicy(undefined, ["StdioTransport"]);
    expect(() => policy.validateComponentTree(agentWithNestedTransport)).toThrow(
      blockedError("StdioTransport"),
    );
  });

  it("passes the same tree when nothing is blocked", () => {
    const policy = new ComponentLoadPolicy(undefined, []);
    expect(() =>
      policy.validateComponentTree(agentWithNestedTransport),
    ).not.toThrow();
  });

  it("applies an allow list to every nested component", () => {
    const policy = new ComponentLoadPolicy([
      "Agent",
      "VllmConfig",
      "MCPTool",
      // StdioTransport intentionally missing.
    ]);
    expect(() => policy.validateComponentTree(agentWithNestedTransport)).toThrow(
      notAllowedError("StdioTransport"),
    );
  });
});

describe("loader default policy", () => {
  it("blocks StdioTransport by default", () => {
    const loader = new AgentSpecLoader();
    expect(loader.blockedComponents).toEqual(["StdioTransport"]);
    expect(() =>
      loader.componentLoadPolicy.validateComponentType("StdioTransport"),
    ).toThrow(blockedError("StdioTransport"));
  });

  it("blockedComponents: [] unblocks StdioTransport", () => {
    const loader = new AgentSpecLoader({ blockedComponents: [] });
    expect(loader.blockedComponents).toEqual([]);
    expect(() =>
      loader.componentLoadPolicy.validateComponentType("StdioTransport"),
    ).not.toThrow();
  });
});
