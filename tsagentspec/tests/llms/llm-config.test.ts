import { describe, it, expect } from "vitest";
import {
  LlmGenerationConfigSchema,
  OpenAIAPIType,
  createOpenAiCompatibleConfig,
  createOllamaConfig,
  createVllmConfig,
  createOpenAiConfig,
  createOciGenAiConfig,
  createOciClientConfigWithApiKey,
  type RetryPolicy,
} from "../../src/index.js";

describe("LlmGenerationConfig", () => {
  it("should parse with all fields", () => {
    const config = LlmGenerationConfigSchema.parse({
      maxTokens: 1024,
      temperature: 0.7,
      topP: 0.9,
    });
    expect(config.maxTokens).toBe(1024);
    expect(config.temperature).toBe(0.7);
    expect(config.topP).toBe(0.9);
  });

  it("should allow all fields to be optional", () => {
    const config = LlmGenerationConfigSchema.parse({});
    expect(config.maxTokens).toBeUndefined();
    expect(config.temperature).toBeUndefined();
    expect(config.topP).toBeUndefined();
  });

  it("should allow extra fields (passthrough)", () => {
    const config = LlmGenerationConfigSchema.parse({
      maxTokens: 100,
      customParam: "value",
      anotherParam: 42,
    });
    expect(config.maxTokens).toBe(100);
    expect((config as Record<string, unknown>)["customParam"]).toBe("value");
    expect((config as Record<string, unknown>)["anotherParam"]).toBe(42);
  });

  it("should reject non-integer maxTokens", () => {
    expect(() =>
      LlmGenerationConfigSchema.parse({ maxTokens: 10.5 }),
    ).toThrow();
  });
});

describe("OpenAIAPIType", () => {
  it("should define CHAT_COMPLETIONS", () => {
    expect(OpenAIAPIType.CHAT_COMPLETIONS).toBe("chat_completions");
  });

  it("should define RESPONSES", () => {
    expect(OpenAIAPIType.RESPONSES).toBe("responses");
  });
});

describe("retryPolicy on LLM configs", () => {
  // Python declares retry_policy once on the LlmConfig base; every TS config
  // schema must carry it explicitly. Each factory accepts a partial policy
  // and fills the defaults.
  const retryPolicy = { maxAttempts: 3, requestTimeout: 0.5 };

  function expectPolicy(policy: RetryPolicy | undefined) {
    expect(policy?.maxAttempts).toBe(3);
    expect(policy?.requestTimeout).toBe(0.5);
    expect(policy?.initialRetryDelay).toBe(1.0);
    expect(policy?.maxRetryDelay).toBe(8.0);
    expect(policy?.backoffFactor).toBe(2.0);
    expect(policy?.jitter).toBe("full_and_equal_for_throttle");
  }

  it("should be accepted by OpenAiCompatibleConfig", () => {
    const config = createOpenAiCompatibleConfig({
      name: "llm",
      url: "http://localhost:8000",
      modelId: "gpt-4",
      retryPolicy,
    });
    expectPolicy(config.retryPolicy);
  });

  it("should be accepted by OllamaConfig", () => {
    const config = createOllamaConfig({
      name: "ollama",
      url: "http://localhost:11434",
      modelId: "llama3",
      retryPolicy,
    });
    expectPolicy(config.retryPolicy);
  });

  it("should be accepted by VllmConfig", () => {
    const config = createVllmConfig({
      name: "vllm",
      url: "http://localhost:8000",
      modelId: "mistral",
      retryPolicy,
    });
    expectPolicy(config.retryPolicy);
  });

  it("should be accepted by OpenAiConfig", () => {
    const config = createOpenAiConfig({
      name: "openai",
      modelId: "gpt-4o",
      retryPolicy,
    });
    expectPolicy(config.retryPolicy);
  });

  it("should be accepted by OciGenAiConfig", () => {
    const config = createOciGenAiConfig({
      name: "oci-llm",
      modelId: "cohere.command-r-plus",
      compartmentId: "ocid1.compartment.oc1..aaa",
      clientConfig: createOciClientConfigWithApiKey({
        name: "oci-client",
        serviceEndpoint: "https://inference.example.oraclecloud.com",
        authProfile: "DEFAULT",
        authFileLocation: "~/.oci/config",
      }),
      retryPolicy,
    });
    expectPolicy(config.retryPolicy);
  });

  it("should default to undefined when not provided", () => {
    const config = createOpenAiConfig({ name: "openai", modelId: "gpt-4o" });
    expect(config.retryPolicy).toBeUndefined();
  });
});
