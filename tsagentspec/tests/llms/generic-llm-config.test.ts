import { describe, it, expect } from "vitest";
import {
  AgentSpecSerializer,
  AgentSpecDeserializer,
  AgentSpecVersion,
  LlmConfigUnion,
  createAgent,
  createLlmConfig,
  type GenericLlmConfig,
  type Agent,
} from "../../src/index.js";

function makeBareLlmConfig(): GenericLlmConfig {
  return createLlmConfig({
    id: "llm",
    name: "generic-llm",
    modelId: "llama-3.3-70b",
    provider: "meta",
    apiProvider: "vllm",
    apiType: "chat_completions",
    url: "http://localhost:8000",
    defaultGenerationParameters: { temperature: 0.5 },
    retryPolicy: { maxAttempts: 3 },
  });
}

describe("Bare LlmConfig component", () => {
  it("should create with all fields", () => {
    const config = makeBareLlmConfig();
    expect(config.componentType).toBe("LlmConfig");
    expect(config.modelId).toBe("llama-3.3-70b");
    expect(config.provider).toBe("meta");
    expect(config.apiProvider).toBe("vllm");
    expect(config.apiType).toBe("chat_completions");
    expect(config.url).toBe("http://localhost:8000");
    expect(config.retryPolicy?.maxAttempts).toBe(3);
    expect(Object.isFrozen(config)).toBe(true);
  });

  it("should create with only the required fields", () => {
    const config = createLlmConfig({ name: "minimal", modelId: "gpt-4o" });
    expect(config.componentType).toBe("LlmConfig");
    expect(config.modelId).toBe("gpt-4o");
    expect(config.provider).toBeUndefined();
    expect(config.apiProvider).toBeUndefined();
    expect(config.apiType).toBeUndefined();
    expect(config.url).toBeUndefined();
    expect(config.retryPolicy).toBeUndefined();
  });

  it("should be accepted by LlmConfigUnion", () => {
    const parsed = LlmConfigUnion.parse(makeBareLlmConfig());
    expect(parsed.componentType).toBe("LlmConfig");
  });

  it("should serialize with snake_case wire names", () => {
    const serializer = new AgentSpecSerializer();
    const json = serializer.toJson(makeBareLlmConfig()) as string;
    const dict = JSON.parse(json);

    expect(dict["component_type"]).toBe("LlmConfig");
    expect(dict["model_id"]).toBe("llama-3.3-70b");
    expect(dict["provider"]).toBe("meta");
    expect(dict["api_provider"]).toBe("vllm");
    expect(dict["api_type"]).toBe("chat_completions");
    expect(dict["url"]).toBe("http://localhost:8000");
    expect(
      (dict["retry_policy"] as Record<string, unknown>)["max_attempts"],
    ).toBe(3);
  });

  it("should redact apiKey from serialized output", () => {
    const serializer = new AgentSpecSerializer();
    const config = createLlmConfig({
      name: "with-key",
      modelId: "gpt-4o",
      apiKey: "sk-secret-value",
    });
    const json = serializer.toJson(config) as string;
    expect("api_key" in JSON.parse(json)).toBe(false);
    expect(json.includes("sk-secret-value")).toBe(false);
  });

  it("should throw when serializing at a version before 26.1.2", () => {
    const serializer = new AgentSpecSerializer();
    expect(() =>
      serializer.toJson(makeBareLlmConfig(), {
        agentspecVersion: AgentSpecVersion.V26_1_0,
      }),
    ).toThrow(/Invalid agentspec_version.*26\.1\.0.*26\.1\.2/);
  });

  it("should serialize successfully at 26.1.2", () => {
    const serializer = new AgentSpecSerializer();
    const json = serializer.toJson(makeBareLlmConfig(), {
      agentspecVersion: AgentSpecVersion.V26_1_2,
    }) as string;
    expect(JSON.parse(json)["agentspec_version"]).toBe("26.1.2");
  });

  it("should round-trip standalone", () => {
    const serializer = new AgentSpecSerializer();
    const deserializer = new AgentSpecDeserializer();
    const config = makeBareLlmConfig();

    const json = serializer.toJson(config) as string;
    const loaded = deserializer.fromJson(json) as GenericLlmConfig;

    expect(loaded).toEqual(config);
  });

  it("should round-trip on an Agent", () => {
    const serializer = new AgentSpecSerializer();
    const deserializer = new AgentSpecDeserializer();
    const agent = createAgent({
      name: "agent",
      llmConfig: makeBareLlmConfig(),
      systemPrompt: "Hello",
    });

    const json = serializer.toJson(agent) as string;
    const loaded = deserializer.fromJson(json) as Agent;

    expect(loaded.llmConfig.componentType).toBe("LlmConfig");
    const llmConfig = loaded.llmConfig as GenericLlmConfig;
    expect(llmConfig.apiProvider).toBe("vllm");
    expect(llmConfig.retryPolicy?.maxAttempts).toBe(3);
  });
});
