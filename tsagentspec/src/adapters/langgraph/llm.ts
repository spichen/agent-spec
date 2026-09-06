/**
 * LLM config conversion for the LangGraph adapter.
 *
 * Port of the `_llm_convert_to_langgraph` section of
 * `pyagentspec.adapters.langgraph._langgraphconverter`.
 *
 * Divergences from Python (see the adapter README):
 * - Conversion is async (chat-model packages are loaded via dynamic import so
 *   they stay optional peer dependencies).
 * - The JS ChatOpenAI takes its request timeout in milliseconds (Python's
 *   client takes seconds), so `RetryPolicy.requestTimeout` (seconds) is
 *   multiplied by 1000.
 * - OciGenAiConfig is not supported (no langchain-oci package for JS).
 */
import type { BaseCallbackHandler } from "@langchain/core/callbacks/base";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { LlmConfig, LlmGenerationConfig } from "../../llms/index.js";
import { OpenAIAPIType } from "../../llms/index.js";
import { RetryPolicySchema, type RetryPolicy } from "../../retry-policy.js";
import { importOptionalPeer } from "../common/index.js";
import { AgentSpecLlmCallbackHandler } from "./tracing.js";

function ensureUrlHasScheme(url: string): string {
  const trimmed = url.trim();
  if (!trimmed.startsWith("http://") && !trimmed.startsWith("https://")) {
    return `http://${trimmed}`;
  }
  return trimmed;
}

/**
 * Correctly format a URL for an OpenAI-compatible server.
 *
 * - Ensures a scheme (http, https) is present, defaulting to 'http'.
 * - Replaces any existing path with exactly '/v1'.
 * - Strips query parameters and fragments.
 *
 * Examples:
 * - "localhost:8000"          -> "http://localhost:8000/v1"
 * - "127.0.0.1:5000"          -> "http://127.0.0.1:5000/v1"
 * - "https://api.example.com" -> "https://api.example.com/v1"
 * - "http://my-host/api/v2"   -> "http://my-host/v1"
 */
export function prepareOpenAiCompatibleUrl(url: string): string {
  const parsed = new URL(ensureUrlHasScheme(url));
  parsed.pathname = "/v1";
  parsed.search = "";
  parsed.hash = "";
  return parsed.toString();
}

/** ChatOpenAI retry/timeout settings derived from an Agent Spec RetryPolicy. */
interface ChatRetryConfig {
  maxRetries?: number;
  timeoutSeconds?: number;
}

/** Default retry policy the supported/unsupported field split compares to. */
const RETRY_POLICY_DEFAULTS: RetryPolicy = RetryPolicySchema.parse({});

/**
 * RetryPolicy fields ChatOpenAI cannot express, with their Python wire names
 * (kept in the error text for cross-SDK parity).
 */
const UNSUPPORTED_CHAT_OPENAI_RETRY_FIELDS: ReadonlyArray<
  [field: keyof RetryPolicy, wireName: string]
> = [
  ["initialRetryDelay", "initial_retry_delay"],
  ["maxRetryDelay", "max_retry_delay"],
  ["backoffFactor", "backoff_factor"],
  ["jitter", "jitter"],
  ["serviceErrorRetryOnAny5xx", "service_error_retry_on_any_5xx"],
  ["recoverableStatuses", "recoverable_statuses"],
];

/**
 * Compare one retry-policy field against its default (Python's `!=`), with
 * order-insensitive deep equality for the `recoverableStatuses` record.
 */
function retryFieldEqualsDefault(value: unknown, defaultValue: unknown): boolean {
  if (
    typeof value === "object" &&
    value !== null &&
    typeof defaultValue === "object" &&
    defaultValue !== null
  ) {
    const actual = value as Record<string, string[]>;
    const expected = defaultValue as Record<string, string[]>;
    const actualKeys = Object.keys(actual).sort();
    const expectedKeys = Object.keys(expected).sort();
    return (
      actualKeys.length === expectedKeys.length &&
      actualKeys.every(
        (key, index) =>
          key === expectedKeys[index] &&
          actual[key]!.length === expected[key]!.length &&
          actual[key]!.every((code, codeIndex) => code === expected[key]![codeIndex]),
      )
    );
  }
  return value === defaultValue;
}

/**
 * Convert Agent Spec retry policy settings into ChatOpenAI keyword arguments.
 *
 * Port of Python's `_retry_policy_convert_to_langgraph`: only `maxAttempts`
 * and `requestTimeout` are representable (the underlying ChatOpenAI/OpenAI
 * client only exposes retry count and timeout settings); any other field set
 * away from its default raises the Python NotImplementedError text.
 */
export function retryPolicyConvertToLanggraph(
  retryPolicy: RetryPolicy | undefined,
): ChatRetryConfig {
  if (retryPolicy == null) {
    return {};
  }

  const unsupportedFields = UNSUPPORTED_CHAT_OPENAI_RETRY_FIELDS.filter(
    ([field]) =>
      !retryFieldEqualsDefault(retryPolicy[field], RETRY_POLICY_DEFAULTS[field]),
  ).map(([, wireName]) => wireName);
  if (unsupportedFields.length > 0) {
    throw new Error(
      "LangGraph ChatOpenAI conversion supports only " +
        "`RetryPolicy.max_attempts` and `RetryPolicy.request_timeout`. " +
        "This is because the underlying ChatOpenAI/OpenAI client only exposes " +
        "retry count and timeout settings. " +
        "Unsupported retry policy fields: " +
        unsupportedFields.join(", "),
    );
  }

  return {
    maxRetries: retryPolicy.maxAttempts,
    ...(retryPolicy.requestTimeout != null
      ? { timeoutSeconds: retryPolicy.requestTimeout }
      : {}),
  };
}

/**
 * Create a ChatOpenAI model without overriding env-based defaults.
 *
 * If no api key is given and the `OPENAI_API_KEY` environment variable is not
 * set, a fake "EMPTY" key is used so servers that require no key still work.
 */
async function createChatOpenAiModel(options: {
  modelId: string;
  useResponsesApi: boolean;
  generationConfig: LlmGenerationConfig;
  retryConfig: ChatRetryConfig;
  callbacks: BaseCallbackHandler[];
  baseUrl?: string;
  apiKey?: string;
}): Promise<BaseChatModel> {
  const { ChatOpenAI } = await importOptionalPeer(
    () => import("@langchain/openai"),
    "@langchain/openai",
    "convert OpenAI-compatible LLM configs",
    "remove them from the spec.",
  );
  // Mirror the Python fallback chain: a MISSING config value falls back to
  // OPENAI_API_KEY -> "EMPTY", but an explicit key (even an empty string) is
  // used as-is — the spec's key must never be silently replaced by the
  // developer's environment credential.
  const apiKey =
    options.apiKey ?? (process.env["OPENAI_API_KEY"] || "EMPTY");
  return new ChatOpenAI({
    model: options.modelId,
    useResponsesApi: options.useResponsesApi,
    apiKey,
    callbacks: options.callbacks,
    temperature: options.generationConfig.temperature,
    maxTokens: options.generationConfig.maxTokens,
    topP: options.generationConfig.topP,
    ...(options.retryConfig.maxRetries !== undefined
      ? { maxRetries: options.retryConfig.maxRetries }
      : {}),
    // The JS ChatOpenAI request timeout is in milliseconds (Python's client
    // takes seconds).
    ...(options.retryConfig.timeoutSeconds !== undefined
      ? { timeout: options.retryConfig.timeoutSeconds * 1000 }
      : {}),
    ...(options.baseUrl !== undefined
      ? { configuration: { baseURL: options.baseUrl } }
      : {}),
  });
}

/**
 * Create the LangChain chat model for the given Agent Spec LLM configuration.
 *
 * VllmConfig / OpenAiCompatibleConfig map to ChatOpenAI with a normalized
 * OpenAI-compatible base URL; OpenAiConfig maps to ChatOpenAI without a base
 * URL; OllamaConfig maps to ChatOllama (rejecting a retry policy, like
 * Python); the bare LlmConfig dispatches on its `apiProvider` string
 * ("openai" maps to ChatOpenAI with a scheme-ensured base URL).
 * OciGenAiConfig is not supported yet.
 */
export async function convertLlmConfig(
  llmConfig: LlmConfig,
): Promise<BaseChatModel> {
  // Only temperature / maxTokens / topP are supported; each use site reads
  // the fields individually, so unset ones simply stay undefined.
  const generationConfig = llmConfig.defaultGenerationParameters ?? {};

  // Every chat model the converter creates carries the Agent Spec LLM
  // tracing handler (Python parity; the unsupported OCI branch is the one
  // Python site without callbacks).
  const callbacks: BaseCallbackHandler[] = [
    new AgentSpecLlmCallbackHandler(llmConfig),
  ];

  switch (llmConfig.componentType) {
    case "VllmConfig":
    case "OpenAiCompatibleConfig":
      return createChatOpenAiModel({
        modelId: llmConfig.modelId,
        baseUrl: prepareOpenAiCompatibleUrl(llmConfig.url),
        apiKey: llmConfig.apiKey,
        useResponsesApi: llmConfig.apiType === OpenAIAPIType.RESPONSES,
        generationConfig,
        retryConfig: retryPolicyConvertToLanggraph(llmConfig.retryPolicy),
        callbacks,
      });
    case "OllamaConfig": {
      if (llmConfig.retryPolicy != null) {
        throw new Error(
          "LangGraph ChatOllama conversion does not support `RetryPolicy`.",
        );
      }
      const { ChatOllama } = await importOptionalPeer(
        () => import("@langchain/ollama"),
        "@langchain/ollama",
        "convert OllamaConfig LLM configs",
        "remove them from the spec.",
      );
      return new ChatOllama({
        baseUrl: llmConfig.url,
        model: llmConfig.modelId,
        callbacks,
        temperature: generationConfig.temperature,
        numPredict: generationConfig.maxTokens,
        topP: generationConfig.topP,
      });
    }
    case "OpenAiConfig":
      return createChatOpenAiModel({
        modelId: llmConfig.modelId,
        apiKey: llmConfig.apiKey,
        useResponsesApi: llmConfig.apiType === OpenAIAPIType.RESPONSES,
        generationConfig,
        retryConfig: retryPolicyConvertToLanggraph(llmConfig.retryPolicy),
        callbacks,
      });
    case "LlmConfig": {
      // Bare LlmConfig — dispatch on the api_provider string, like Python.
      if (llmConfig.apiProvider === "openai") {
        return createChatOpenAiModel({
          modelId: llmConfig.modelId,
          // Scheme-ensured only: unlike the OpenAI-compatible configs, the
          // bare config's URL path is used verbatim (no /v1 normalization).
          ...(llmConfig.url !== undefined
            ? { baseUrl: ensureUrlHasScheme(llmConfig.url) }
            : {}),
          apiKey: llmConfig.apiKey,
          useResponsesApi: llmConfig.apiType === "responses",
          generationConfig,
          retryConfig: retryPolicyConvertToLanggraph(llmConfig.retryPolicy),
          callbacks,
        });
      }
      throw new Error(
        `LlmConfig with api_provider='${llmConfig.apiProvider}' is not yet ` +
          "supported in langgraph. Consider using a specific LlmConfig " +
          "subclass instead.",
      );
    }
    case "OciGenAiConfig":
      // Python rejects the retry policy before attempting the (here
      // unavailable) langchain-oci import, so keep that error precedence.
      if (llmConfig.retryPolicy != null) {
        throw new Error(
          "LangGraph OCI GenAI conversion does not support `RetryPolicy`.",
        );
      }
      throw new Error(
        "The Agent Spec type 'OciGenAiConfig' is not supported by the LangGraph TypeScript adapter yet.",
      );
    default: {
      const componentType = (llmConfig as { componentType: string })
        .componentType;
      throw new Error(
        `The Agent Spec type '${componentType}' is not yet supported for conversion.`,
      );
    }
  }
}
