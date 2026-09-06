/**
 * LLM config conversion tests for the LangGraph adapter.
 *
 * Mirrors `pyagentspec/tests/adapters/langgraph/llms/test_llm_conversion.py`
 * (URL normalization matrix, ChatOpenAI/ChatOllama mapping, responses-API
 * flag, generation parameter forwarding, the retryPolicy-to-ChatOpenAI
 * mapping and its NotImplementedError paths) and
 * `pyagentspec/tests/adapters/langgraph/test_bare_llmconfig_dispatch.py`
 * (bare LlmConfig api_provider dispatch), plus the OciGenAiConfig rejection.
 * All tests run offline: models are constructed, never invoked.
 *
 * Documented divergences (see the adapter README / llm.ts header):
 * - conversion is async;
 * - the JS ChatOpenAI takes its request timeout in milliseconds, so the
 *   spec's requestTimeout seconds are multiplied by 1000;
 * - OciGenAiConfig is rejected outright (no langchain-oci JS package).
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { ChatOllama } from "@langchain/ollama";
import { ChatOpenAI } from "@langchain/openai";
import {
  OpenAIAPIType,
  createLlmConfig,
  createOciClientConfigWithApiKey,
  createOciGenAiConfig,
  createOllamaConfig,
  createOpenAiCompatibleConfig,
  createOpenAiConfig,
  createVllmConfig,
  type LlmConfig,
  type LlmGenerationConfig,
} from "../../../src/index.js";
import {
  convertLlmConfig,
  prepareOpenAiCompatibleUrl,
} from "../../../src/adapters/langgraph/llm.js";

/** Runtime surface of ChatOpenAI inspected by these tests. */
interface ChatOpenAiProbe {
  model: string;
  apiKey?: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  presencePenalty?: number;
  useResponsesApi: boolean;
  modelKwargs?: Record<string, unknown>;
  clientConfig: { baseURL?: string };
  /** Explicit retry count flows into the async caller. */
  caller: { maxRetries: number };
  /** JS ChatOpenAI request timeout, in milliseconds. */
  timeout?: number;
}

async function convertToChatOpenAi(config: LlmConfig): Promise<ChatOpenAiProbe> {
  const model = await convertLlmConfig(config);
  expect(model).toBeInstanceOf(ChatOpenAI);
  return model as unknown as ChatOpenAiProbe;
}

const DEFAULT_GENERATION_PARAMETERS: LlmGenerationConfig = {
  temperature: 0.2,
  maxTokens: 128,
  topP: 0.8,
};

describe("prepareOpenAiCompatibleUrl", () => {
  const cases: Array<[raw: string, expected: string]> = [
    // Ported from the Python parametrized cases.
    ["localhost:8000", "http://localhost:8000/v1"],
    ["127.0.0.1:5000", "http://127.0.0.1:5000/v1"],
    ["https://api.example.com", "https://api.example.com/v1"],
    ["http://my-host/api/v2", "http://my-host/v1"],
    [" my-host:9999  ", "http://my-host:9999/v1"],
    // Query parameters and fragments are stripped.
    ["http://host:1234/path?query=1#frag", "http://host:1234/v1"],
    ["https://api.example.com?key=value", "https://api.example.com/v1"],
    // An already-normalized URL is preserved.
    ["https://api.example.com/v1", "https://api.example.com/v1"],
  ];

  it.each(cases)("formats %j as %j", (raw, expected) => {
    expect(prepareOpenAiCompatibleUrl(raw)).toBe(expected);
  });
});

describe("convertLlmConfig for OpenAI-compatible configs", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("maps VllmConfig to ChatOpenAI with a normalized base URL", async () => {
    vi.stubEnv("OPENAI_API_KEY", "");
    const model = await convertToChatOpenAi(
      createVllmConfig({
        name: "llm",
        modelId: "meta-llama/Meta-Llama-3.1-8B-Instruct",
        url: "localhost:8000", // missing scheme on purpose
        defaultGenerationParameters: DEFAULT_GENERATION_PARAMETERS,
      }),
    );
    expect(model.model).toBe("meta-llama/Meta-Llama-3.1-8B-Instruct");
    expect(model.clientConfig.baseURL).toBe("http://localhost:8000/v1");
    expect(model.useResponsesApi).toBe(false);
    expect(model.temperature).toBe(0.2);
    expect(model.maxTokens).toBe(128);
    expect(model.topP).toBe(0.8);
  });

  it("maps OpenAiCompatibleConfig to ChatOpenAI with the /v1 base URL", async () => {
    vi.stubEnv("OPENAI_API_KEY", "DUMMY_KEY");
    const model = await convertToChatOpenAi(
      createOpenAiCompatibleConfig({
        name: "oaic",
        modelId: "gpt-4o-mini",
        url: "https://api.compatible",
        defaultGenerationParameters: DEFAULT_GENERATION_PARAMETERS,
      }),
    );
    expect(model.model).toBe("gpt-4o-mini");
    expect(model.clientConfig.baseURL).toBe("https://api.compatible/v1");
    expect(model.maxTokens).toBe(128);
    expect(model.temperature).toBe(0.2);
  });

  it.each([
    [OpenAIAPIType.RESPONSES, true],
    [OpenAIAPIType.CHAT_COMPLETIONS, false],
  ])("sets useResponsesApi for apiType %j", async (apiType, expectedFlag) => {
    vi.stubEnv("OPENAI_API_KEY", "DUMMY_KEY");
    const model = await convertToChatOpenAi(
      createOpenAiCompatibleConfig({
        name: "oaic",
        modelId: "gpt-4o-mini",
        url: "https://api.compatible",
        apiType,
      }),
    );
    expect(model.useResponsesApi).toBe(expectedFlag);
  });

  it("maps OpenAiConfig to ChatOpenAI without a base URL", async () => {
    vi.stubEnv("OPENAI_API_KEY", "DUMMY_KEY");
    const model = await convertToChatOpenAi(
      createOpenAiConfig({
        name: "openai",
        modelId: "gpt-4o-mini",
        apiType: OpenAIAPIType.RESPONSES,
      }),
    );
    expect(model.model).toBe("gpt-4o-mini");
    expect(model.clientConfig.baseURL).toBeUndefined();
    expect(model.useResponsesApi).toBe(true);
  });

  it("does not forward extra generation fields", async () => {
    vi.stubEnv("OPENAI_API_KEY", "DUMMY_KEY");
    const model = await convertToChatOpenAi(
      createOpenAiConfig({
        name: "openai",
        modelId: "gpt-4o-mini",
        defaultGenerationParameters: {
          temperature: 0.2,
          maxTokens: 128,
          topP: 0.8,
          presencePenalty: 1.0,
        } as LlmGenerationConfig,
      }),
    );
    expect(model.temperature).toBe(0.2);
    expect(model.maxTokens).toBe(128);
    expect(model.presencePenalty).toBeUndefined();
    expect(model.modelKwargs ?? {}).not.toHaveProperty("presence_penalty");
  });

  it("leaves generation parameters unset without defaultGenerationParameters", async () => {
    vi.stubEnv("OPENAI_API_KEY", "DUMMY_KEY");
    const model = await convertToChatOpenAi(
      createOpenAiConfig({ name: "openai", modelId: "gpt-4o-mini" }),
    );
    expect(model.temperature).toBeUndefined();
    expect(model.maxTokens).toBeUndefined();
    expect(model.topP).toBeUndefined();
  });

  it("prefers the config api key over the environment", async () => {
    vi.stubEnv("OPENAI_API_KEY", "env-key");
    const model = await convertToChatOpenAi(
      createOpenAiConfig({
        name: "openai",
        modelId: "gpt-4o-mini",
        apiKey: "sk-config",
      }),
    );
    expect(model.apiKey).toBe("sk-config");
  });

  it("falls back to OPENAI_API_KEY when the config has no api key", async () => {
    vi.stubEnv("OPENAI_API_KEY", "env-key");
    const model = await convertToChatOpenAi(
      createOpenAiConfig({ name: "openai", modelId: "gpt-4o-mini" }),
    );
    expect(model.apiKey).toBe("env-key");
  });

  it('falls back to the fake "EMPTY" key when neither is set', async () => {
    // An empty env value falls through, matching Python `or` semantics.
    vi.stubEnv("OPENAI_API_KEY", "");
    const model = await convertToChatOpenAi(
      createOpenAiConfig({ name: "openai", modelId: "gpt-4o-mini" }),
    );
    expect(model.apiKey).toBe("EMPTY");
  });

  it("uses an explicit empty-string api key without substituting the environment key", async () => {
    // The spec's key must never be silently replaced by the developer's
    // environment credential (Python falls back only when api_key is None).
    vi.stubEnv("OPENAI_API_KEY", "env-key");
    const model = await convertToChatOpenAi(
      createOpenAiConfig({
        name: "openai",
        modelId: "gpt-4o-mini",
        apiKey: "",
      }),
    );
    expect(model.apiKey).toBe("");
  });
});

describe("convertLlmConfig for OllamaConfig", () => {
  it("maps generation parameters onto the ChatOllama names", async () => {
    const model = (await convertLlmConfig(
      createOllamaConfig({
        name: "oll",
        modelId: "llama3.1",
        url: "http://ollama.local:11434",
        defaultGenerationParameters: DEFAULT_GENERATION_PARAMETERS,
      }),
    )) as ChatOllama;
    expect(model).toBeInstanceOf(ChatOllama);
    // The Ollama URL is used verbatim (no /v1 normalization).
    expect(model.baseUrl).toBe("http://ollama.local:11434");
    expect(model.model).toBe("llama3.1");
    expect(model.temperature).toBe(0.2);
    expect(model.numPredict).toBe(128);
    expect(model.topP).toBe(0.8);
  });

  it("leaves generation parameters unset without defaultGenerationParameters", async () => {
    const model = (await convertLlmConfig(
      createOllamaConfig({
        name: "oll",
        modelId: "llama3.2",
        url: "http://localhost:11434",
      }),
    )) as ChatOllama;
    expect(model.temperature).toBeUndefined();
    expect(model.numPredict).toBeUndefined();
    expect(model.topP).toBeUndefined();
  });

  it("does not forward extra generation fields", async () => {
    // Only temperature / maxTokens / topP are supported; extra fields in the
    // (passthrough) generation config must not reach the constructed model.
    const model = (await convertLlmConfig(
      createOllamaConfig({
        name: "oll",
        modelId: "llama3.1",
        url: "http://localhost:11434",
        defaultGenerationParameters: {
          temperature: 0.2,
          maxTokens: 128,
          topP: 0.8,
          presencePenalty: 1.0,
        } as LlmGenerationConfig,
      }),
    )) as ChatOllama;
    expect(model.temperature).toBe(0.2);
    expect(model.numPredict).toBe(128);
    expect(model.topP).toBe(0.8);
    expect(
      (model as unknown as { presencePenalty?: number }).presencePenalty,
    ).toBeUndefined();
  });
});

describe("convertLlmConfig retry policy mapping", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("maps maxAttempts and requestTimeout onto ChatOpenAI retries and (ms) timeout", async () => {
    vi.stubEnv("OPENAI_API_KEY", "DUMMY_KEY");
    const model = await convertToChatOpenAi(
      createOpenAiConfig({
        name: "openai",
        modelId: "gpt-4o-mini",
        retryPolicy: { maxAttempts: 3, requestTimeout: 45 },
      }),
    );
    expect(model.caller.maxRetries).toBe(3);
    // The spec's requestTimeout is seconds; the JS ChatOpenAI timeout is ms.
    expect(model.timeout).toBe(45_000);
  });

  it("leaves the timeout unset when the policy has no requestTimeout", async () => {
    vi.stubEnv("OPENAI_API_KEY", "DUMMY_KEY");
    const model = await convertToChatOpenAi(
      createOpenAiConfig({
        name: "openai",
        modelId: "gpt-4o-mini",
        retryPolicy: { maxAttempts: 4 },
      }),
    );
    expect(model.caller.maxRetries).toBe(4);
    expect(model.timeout).toBeUndefined();
  });

  it("applies the retry policy on VllmConfig and OpenAiCompatibleConfig too", async () => {
    vi.stubEnv("OPENAI_API_KEY", "DUMMY_KEY");
    const vllm = await convertToChatOpenAi(
      createVllmConfig({
        name: "llm",
        modelId: "m",
        url: "localhost:8000",
        retryPolicy: { maxAttempts: 7 },
      }),
    );
    expect(vllm.caller.maxRetries).toBe(7);
    const compatible = await convertToChatOpenAi(
      createOpenAiCompatibleConfig({
        name: "oaic",
        modelId: "m",
        url: "https://api.compatible",
        retryPolicy: { maxAttempts: 0, requestTimeout: 0.5 },
      }),
    );
    expect(compatible.caller.maxRetries).toBe(0);
    expect(compatible.timeout).toBe(500);
  });

  it("rejects a policy customizing fields ChatOpenAI cannot express, with the Python text", async () => {
    await expect(
      convertLlmConfig(
        createOpenAiConfig({
          name: "openai",
          modelId: "gpt-4o-mini",
          retryPolicy: { initialRetryDelay: 5 },
        }),
      ),
    ).rejects.toThrow(
      "LangGraph ChatOpenAI conversion supports only " +
        "`RetryPolicy.max_attempts` and `RetryPolicy.request_timeout`. " +
        "This is because the underlying ChatOpenAI/OpenAI client only exposes " +
        "retry count and timeout settings. " +
        "Unsupported retry policy fields: initial_retry_delay",
    );
  });

  it("lists every customized unsupported field, jitter: null included", async () => {
    await expect(
      convertLlmConfig(
        createOpenAiConfig({
          name: "openai",
          modelId: "gpt-4o-mini",
          retryPolicy: {
            initialRetryDelay: 2,
            maxRetryDelay: 16,
            backoffFactor: 3,
            jitter: null,
            serviceErrorRetryOnAny5xx: false,
            recoverableStatuses: { "429": [] },
          },
        }),
      ),
    ).rejects.toThrow(
      "Unsupported retry policy fields: initial_retry_delay, max_retry_delay, " +
        "backoff_factor, jitter, service_error_retry_on_any_5xx, " +
        "recoverable_statuses",
    );
  });

  it("treats a reordered default recoverableStatuses record as the default", async () => {
    // Python compares dicts order-insensitively; the TS deep comparison
    // must match ({"429": [], "409": []} equals the {"409": [], "429": []}
    // default).
    vi.stubEnv("OPENAI_API_KEY", "DUMMY_KEY");
    const model = await convertToChatOpenAi(
      createOpenAiConfig({
        name: "openai",
        modelId: "gpt-4o-mini",
        retryPolicy: {
          maxAttempts: 5,
          recoverableStatuses: { "429": [], "409": [] },
        },
      }),
    );
    expect(model.caller.maxRetries).toBe(5);
  });

  it("rejects a retry policy on OllamaConfig with the Python text", async () => {
    await expect(
      convertLlmConfig(
        createOllamaConfig({
          name: "oll",
          modelId: "llama3.1",
          url: "http://localhost:11434",
          retryPolicy: { maxAttempts: 1 },
        }),
      ),
    ).rejects.toThrow(
      "LangGraph ChatOllama conversion does not support `RetryPolicy`.",
    );
  });

  it("rejects a retry policy on OciGenAiConfig before the unsupported-type error", async () => {
    await expect(
      convertLlmConfig(
        createOciGenAiConfig({
          name: "oci",
          modelId: "meta.llama-3.1-70b-instruct",
          compartmentId: "ocid1.compartment.oc1..x",
          clientConfig: createOciClientConfigWithApiKey({
            name: "client",
            serviceEndpoint: "https://inference.generativeai.example.com",
            authProfile: "DEFAULT",
            authFileLocation: "~/.oci/config",
          }),
          retryPolicy: { maxAttempts: 1 },
        }),
      ),
    ).rejects.toThrow(
      "LangGraph OCI GenAI conversion does not support `RetryPolicy`.",
    );
  });
});

describe("convertLlmConfig for the bare LlmConfig", () => {
  // Ports pyagentspec/tests/adapters/langgraph/test_bare_llmconfig_dispatch.py.
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("openai provider respects api_type responses", async () => {
    vi.stubEnv("OPENAI_API_KEY", "DUMMY_KEY");
    const model = await convertToChatOpenAi(
      createLlmConfig({
        name: "test",
        modelId: "gpt-4o",
        apiProvider: "openai",
        apiType: "responses",
      }),
    );
    expect(model.useResponsesApi).toBe(true);
  });

  it("openai provider defaults to chat completions", async () => {
    vi.stubEnv("OPENAI_API_KEY", "DUMMY_KEY");
    const model = await convertToChatOpenAi(
      createLlmConfig({ name: "test", modelId: "gpt-4o", apiProvider: "openai" }),
    );
    expect(model.model).toBe("gpt-4o");
    expect(model.useResponsesApi).toBe(false);
    expect(model.clientConfig.baseURL).toBeUndefined();
  });

  it("openai provider forwards the base url verbatim (no /v1 normalization) and the api key", async () => {
    const model = await convertToChatOpenAi(
      createLlmConfig({
        name: "test",
        modelId: "gpt-4o",
        apiProvider: "openai",
        url: "https://my-proxy.example.com/v1",
        apiKey: "sk-test-key",
      }),
    );
    expect(model.clientConfig.baseURL).toBe("https://my-proxy.example.com/v1");
    expect(model.apiKey).toBe("sk-test-key");
  });

  it("openai provider adds a scheme to a raw base url", async () => {
    vi.stubEnv("OPENAI_API_KEY", "DUMMY_KEY");
    const model = await convertToChatOpenAi(
      createLlmConfig({
        name: "test",
        modelId: "gpt-4o",
        apiProvider: "openai",
        url: "localhost:8000",
      }),
    );
    expect(model.clientConfig.baseURL).toBe("http://localhost:8000");
  });

  it("openai provider maps the retry policy like the dedicated configs", async () => {
    vi.stubEnv("OPENAI_API_KEY", "DUMMY_KEY");
    const model = await convertToChatOpenAi(
      createLlmConfig({
        name: "test",
        modelId: "gpt-4o",
        apiProvider: "openai",
        retryPolicy: { maxAttempts: 3, requestTimeout: 45 },
      }),
    );
    expect(model.caller.maxRetries).toBe(3);
    expect(model.timeout).toBe(45_000);
  });

  it("rejects unsupported api providers with the Python text", async () => {
    await expect(
      convertLlmConfig(
        createLlmConfig({
          name: "test",
          modelId: "some-model",
          apiProvider: "unsupported_provider",
        }),
      ),
    ).rejects.toThrow(
      "LlmConfig with api_provider='unsupported_provider' is not yet " +
        "supported in langgraph. Consider using a specific LlmConfig " +
        "subclass instead.",
    );
  });
});

describe("convertLlmConfig rejections", () => {
  it("rejects OciGenAiConfig (no langchain-oci package for JS)", async () => {
    const ociConfig = createOciGenAiConfig({
      name: "oci",
      modelId: "meta.llama-3.1-70b-instruct",
      compartmentId: "ocid1.compartment.oc1..x",
      clientConfig: createOciClientConfigWithApiKey({
        name: "client",
        serviceEndpoint: "https://inference.generativeai.example.com",
        authProfile: "DEFAULT",
        authFileLocation: "~/.oci/config",
      }),
    });
    await expect(convertLlmConfig(ociConfig)).rejects.toThrow(
      "The Agent Spec type 'OciGenAiConfig' is not supported by the LangGraph TypeScript adapter yet.",
    );
  });

  it("rejects unknown LLM config component types", async () => {
    const bogus = {
      componentType: "MadeUpConfig",
      name: "x",
    } as unknown as LlmConfig;
    await expect(convertLlmConfig(bogus)).rejects.toThrow(
      "The Agent Spec type 'MadeUpConfig' is not yet supported for conversion.",
    );
  });
});
