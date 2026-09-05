import { describe, it, expect } from "vitest";
import {
  AgentSpecSerializer,
  AgentSpecVersion,
  createAgent,
  createApiNode,
  createMCPTool,
  createOpenAiCompatibleConfig,
  createRemoteTool,
  createSSETransport,
  createServerTool,
  createBuiltinTool,
  createMCPToolBox,
  createStdioTransport,
  stringProperty,
} from "../../src/index.js";

function makeLlmConfig() {
  return createOpenAiCompatibleConfig({
    name: "test-llm",
    url: "http://localhost:8000",
    modelId: "gpt-4",
  });
}

describe("version-gated field serialization", () => {
  it("should exclude humanInTheLoop for versions before 25.4.2", () => {
    const serializer = new AgentSpecSerializer();
    const agent = createAgent({
      name: "agent",
      llmConfig: makeLlmConfig(),
      systemPrompt: "Hello",
      humanInTheLoop: false,
    });
    const json = serializer.toJson(agent, {
      agentspecVersion: AgentSpecVersion.V25_4_1,
    }) as string;
    const dict = JSON.parse(json);
    expect("human_in_the_loop" in dict).toBe(false);
  });

  it("should include humanInTheLoop for version 25.4.2+", () => {
    const serializer = new AgentSpecSerializer();
    const agent = createAgent({
      name: "agent",
      llmConfig: makeLlmConfig(),
      systemPrompt: "Hello",
      humanInTheLoop: false,
    });
    const json = serializer.toJson(agent, {
      agentspecVersion: AgentSpecVersion.V25_4_2,
    }) as string;
    const dict = JSON.parse(json);
    expect("human_in_the_loop" in dict).toBe(true);
  });

  it("should throw when serializing MCPToolBox at version before 25.4.2", () => {
    const serializer = new AgentSpecSerializer();
    const toolbox = createMCPToolBox({
      name: "toolbox",
      clientTransport: createStdioTransport({
        name: "stdio",
        command: "node",
      }),
    });
    const agent = createAgent({
      name: "agent",
      llmConfig: makeLlmConfig(),
      systemPrompt: "Hello",
      toolboxes: [toolbox],
    });
    expect(() =>
      serializer.toJson(agent, {
        agentspecVersion: AgentSpecVersion.V25_4_1,
      }),
    ).toThrow(/Invalid agentspec_version.*25\.4\.1.*25\.4\.2.*toolbox/);
  });

  it("should exclude requiresConfirmation on tools for versions before 25.4.2", () => {
    const serializer = new AgentSpecSerializer();
    const tool = createServerTool({
      name: "tool",
      inputs: [stringProperty({ title: "q" })],
      requiresConfirmation: true,
    });
    const agent = createAgent({
      name: "agent",
      llmConfig: makeLlmConfig(),
      systemPrompt: "Hello",
      tools: [tool],
    });
    const json = serializer.toJson(agent, {
      agentspecVersion: AgentSpecVersion.V25_4_1,
    }) as string;
    const dict = JSON.parse(json);
    const tools = dict["tools"] as Record<string, unknown>[];
    expect("requires_confirmation" in tools[0]!).toBe(false);
  });

  it("should version-gate BuiltinTool _self fields for versions before 25.4.2", () => {
    const serializer = new AgentSpecSerializer();
    const tool = createBuiltinTool({
      name: "code-exec",
      toolType: "code_execution",
      configuration: { language: "python" },
    });
    const agent = createAgent({
      name: "agent",
      llmConfig: makeLlmConfig(),
      systemPrompt: "Hello",
      tools: [tool],
    });

    // For current version, BuiltinTool should appear with all fields
    const json = serializer.toJson(agent) as string;
    const dictCurrent = JSON.parse(json);
    const toolsCurrent = dictCurrent["tools"] as Record<string, unknown>[];
    const btCurrent = toolsCurrent.find(
      (t) => t["component_type"] === "BuiltinTool",
    );
    expect(btCurrent).toBeDefined();
    expect(btCurrent!["tool_type"]).toBe("code_execution");
  });

  it("should exclude MCPToolBox requiresConfirmation for versions before 26.2.0", () => {
    const serializer = new AgentSpecSerializer();
    const toolbox = createMCPToolBox({
      name: "toolbox",
      clientTransport: createStdioTransport({
        name: "stdio",
        command: "node",
      }),
      requiresConfirmation: true,
    });
    const agent = createAgent({
      name: "agent",
      llmConfig: makeLlmConfig(),
      systemPrompt: "Hello",
      toolboxes: [toolbox],
    });
    const json = serializer.toJson(agent, {
      agentspecVersion: AgentSpecVersion.V25_4_2,
    }) as string;
    const dict = JSON.parse(json);
    const toolboxes = dict["toolboxes"] as Record<string, unknown>[];
    expect("requires_confirmation" in toolboxes[0]!).toBe(false);
  });

  it("should include MCPToolBox requiresConfirmation for version 26.2.0+", () => {
    const serializer = new AgentSpecSerializer();
    const toolbox = createMCPToolBox({
      name: "toolbox",
      clientTransport: createStdioTransport({
        name: "stdio",
        command: "node",
      }),
      requiresConfirmation: true,
    });
    const agent = createAgent({
      name: "agent",
      llmConfig: makeLlmConfig(),
      systemPrompt: "Hello",
      toolboxes: [toolbox],
    });
    const json = serializer.toJson(agent, {
      agentspecVersion: AgentSpecVersion.V26_2_0,
    }) as string;
    const dict = JSON.parse(json);
    const toolboxes = dict["toolboxes"] as Record<string, unknown>[];
    expect("requires_confirmation" in toolboxes[0]!).toBe(true);
  });

  it("should include everything for current version", () => {
    const serializer = new AgentSpecSerializer();
    const tool = createServerTool({
      name: "tool",
      inputs: [stringProperty({ title: "q" })],
      requiresConfirmation: true,
    });
    const agent = createAgent({
      name: "agent",
      llmConfig: makeLlmConfig(),
      systemPrompt: "Hello",
      tools: [tool],
      humanInTheLoop: false,
    });
    const json = serializer.toJson(agent) as string;
    const dict = JSON.parse(json);
    expect("human_in_the_loop" in dict).toBe(true);
    const tools = dict["tools"] as Record<string, unknown>[];
    expect("requires_confirmation" in tools[0]!).toBe(true);
  });

  it("should include RemoteTool urlAllowList and retryPolicy for version 26.1.2+", () => {
    const serializer = new AgentSpecSerializer();
    const tool = createRemoteTool({
      name: "remote",
      url: "https://api.example.com/orders/",
      httpMethod: "GET",
      urlAllowList: ["https://api.example.com/orders/"],
      retryPolicy: { maxAttempts: 3 },
    });
    const json = serializer.toJson(tool, {
      agentspecVersion: AgentSpecVersion.V26_1_2,
    }) as string;
    const dict = JSON.parse(json);
    expect(dict["url_allow_list"]).toEqual(["https://api.example.com/orders/"]);
    expect((dict["retry_policy"] as Record<string, unknown>)["max_attempts"]).toBe(3);
  });

  it("should exclude RemoteTool urlAllowList and retryPolicy for versions before 26.1.2", () => {
    const serializer = new AgentSpecSerializer();
    const tool = createRemoteTool({
      name: "remote",
      url: "https://api.example.com/orders/",
      httpMethod: "GET",
      urlAllowList: ["https://api.example.com/orders/"],
      retryPolicy: { maxAttempts: 3 },
    });
    const json = serializer.toJson(tool, {
      agentspecVersion: AgentSpecVersion.V26_1_0,
    }) as string;
    const dict = JSON.parse(json);
    expect("url_allow_list" in dict).toBe(false);
    expect("retry_policy" in dict).toBe(false);
  });

  it("should gate ApiNode urlAllowList and retryPolicy on 26.1.2", () => {
    const serializer = new AgentSpecSerializer();
    const node = createApiNode({
      name: "api",
      url: "https://api.example.com",
      httpMethod: "GET",
      urlAllowList: ["https://api.example.com/"],
      retryPolicy: { maxAttempts: 3 },
    });

    const current = JSON.parse(
      serializer.toJson(node, {
        agentspecVersion: AgentSpecVersion.V26_1_2,
      }) as string,
    );
    expect(current["url_allow_list"]).toEqual(["https://api.example.com/"]);
    expect((current["retry_policy"] as Record<string, unknown>)["max_attempts"]).toBe(3);

    const old = JSON.parse(
      serializer.toJson(node, {
        agentspecVersion: AgentSpecVersion.V26_1_0,
      }) as string,
    );
    expect("url_allow_list" in old).toBe(false);
    expect("retry_policy" in old).toBe(false);
  });

  it("should gate LLM config retryPolicy on 26.1.2", () => {
    const serializer = new AgentSpecSerializer();
    const agent = createAgent({
      name: "agent",
      llmConfig: createOpenAiCompatibleConfig({
        name: "llm",
        url: "http://localhost:8000",
        modelId: "gpt-4",
        retryPolicy: { maxAttempts: 3 },
      }),
      systemPrompt: "Hello",
    });

    const current = JSON.parse(serializer.toJson(agent) as string);
    const llmDict = current["llm_config"] as Record<string, unknown>;
    expect((llmDict["retry_policy"] as Record<string, unknown>)["max_attempts"]).toBe(3);

    const old = JSON.parse(
      serializer.toJson(agent, {
        agentspecVersion: AgentSpecVersion.V26_1_0,
      }) as string,
    );
    expect("retry_policy" in (old["llm_config"] as Record<string, unknown>)).toBe(false);
  });

  it("should exclude remote transport retryPolicy for versions before 26.1.2", () => {
    const serializer = new AgentSpecSerializer();
    const transport = createSSETransport({
      name: "sse",
      url: "http://localhost/sse",
      retryPolicy: { maxAttempts: 3, initialRetryDelay: 0.25 },
    });
    const json = serializer.toJson(transport, {
      agentspecVersion: AgentSpecVersion.V26_1_0,
    }) as string;
    expect("retry_policy" in JSON.parse(json)).toBe(false);
  });

  it("should gate MCPTool semantic retryPolicy on 26.3.0, not 26.2.0", () => {
    const serializer = new AgentSpecSerializer();
    const tool = createMCPTool({
      name: "mcp-tool",
      clientTransport: createStdioTransport({ name: "stdio", command: "node" }),
      retryPolicy: { maxAttempts: 3, initialRetryDelay: 0.25 },
    });

    const at26_3 = JSON.parse(
      serializer.toJson(tool, {
        agentspecVersion: AgentSpecVersion.V26_3_0,
      }) as string,
    );
    expect((at26_3["retry_policy"] as Record<string, unknown>)["max_attempts"]).toBe(3);

    // The Python threshold is v26_3_0 (there is no 26.2.0 member in Python's
    // current versioning); the field must still be gated out at 26.2.0.
    const at26_2 = JSON.parse(
      serializer.toJson(tool, {
        agentspecVersion: AgentSpecVersion.V26_2_0,
      }) as string,
    );
    expect("retry_policy" in at26_2).toBe(false);

    const at26_1_2 = JSON.parse(
      serializer.toJson(tool, {
        agentspecVersion: AgentSpecVersion.V26_1_2,
      }) as string,
    );
    expect("retry_policy" in at26_1_2).toBe(false);
  });

  it("should gate MCPToolBox semantic retryPolicy on 26.3.0", () => {
    const serializer = new AgentSpecSerializer();
    const toolbox = createMCPToolBox({
      name: "toolbox",
      clientTransport: createStdioTransport({ name: "stdio", command: "node" }),
      retryPolicy: { maxAttempts: 4, initialRetryDelay: 0.5 },
    });

    const at26_3 = JSON.parse(
      serializer.toJson(toolbox, {
        agentspecVersion: AgentSpecVersion.V26_3_0,
      }) as string,
    );
    expect((at26_3["retry_policy"] as Record<string, unknown>)["max_attempts"]).toBe(4);

    const at26_2 = JSON.parse(
      serializer.toJson(toolbox, {
        agentspecVersion: AgentSpecVersion.V26_2_0,
      }) as string,
    );
    expect("retry_policy" in at26_2).toBe(false);
  });

  it("should serialize MCPToolBox without a retryPolicy key when unset", () => {
    const serializer = new AgentSpecSerializer();
    const toolbox = createMCPToolBox({
      name: "toolbox",
      clientTransport: createStdioTransport({ name: "stdio", command: "node" }),
    });
    const json = serializer.toJson(toolbox, {
      agentspecVersion: AgentSpecVersion.V26_1_2,
    }) as string;
    expect("retry_policy" in JSON.parse(json)).toBe(false);
  });

  it("should throw when serializing BuiltinTool at version before 25.4.2", () => {
    const serializer = new AgentSpecSerializer();
    const tool = createBuiltinTool({
      name: "code-exec",
      toolType: "code_execution",
      configuration: { language: "python" },
    });
    const agent = createAgent({
      name: "agent",
      llmConfig: makeLlmConfig(),
      systemPrompt: "Hello",
      tools: [tool],
    });
    expect(() =>
      serializer.toJson(agent, {
        agentspecVersion: AgentSpecVersion.V25_4_1,
      }),
    ).toThrow(/Invalid agentspec_version.*25\.4\.1.*25\.4\.2.*code-exec/);
  });
});
