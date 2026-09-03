/**
 * LLM config conversion for the LangGraph adapter.
 *
 * Port of the `_llm_convert_to_langgraph` section of
 * `pyagentspec.adapters.langgraph._langgraphconverter`.
 *
 * Divergences from Python (see the adapter README):
 * - Conversion is async (chat-model packages are loaded via dynamic import so
 *   they stay optional peer dependencies).
 * - The TS SDK LlmConfig components have no `retryPolicy` field, so the
 *   Python retry-policy-to-ChatOpenAI mapping is not ported.
 * - OciGenAiConfig is not supported (no langchain-oci package for JS).
 * - No tracing callbacks are attached here (tracing is a no-op seam in v1).
 */
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { LlmConfig, LlmGenerationConfig } from "../../llms/index.js";
import { OpenAIAPIType } from "../../llms/index.js";
import { importOptionalPeer } from "../common/index.js";

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
    temperature: options.generationConfig.temperature,
    maxTokens: options.generationConfig.maxTokens,
    topP: options.generationConfig.topP,
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
 * URL; OllamaConfig maps to ChatOllama. OciGenAiConfig is not supported yet.
 */
export async function convertLlmConfig(
  llmConfig: LlmConfig,
): Promise<BaseChatModel> {
  // Only temperature / maxTokens / topP are supported; each use site reads
  // the fields individually, so unset ones simply stay undefined.
  const generationConfig = llmConfig.defaultGenerationParameters ?? {};

  switch (llmConfig.componentType) {
    case "VllmConfig":
    case "OpenAiCompatibleConfig":
      return createChatOpenAiModel({
        modelId: llmConfig.modelId,
        baseUrl: prepareOpenAiCompatibleUrl(llmConfig.url),
        apiKey: llmConfig.apiKey,
        useResponsesApi: llmConfig.apiType === OpenAIAPIType.RESPONSES,
        generationConfig,
      });
    case "OllamaConfig": {
      const { ChatOllama } = await importOptionalPeer(
        () => import("@langchain/ollama"),
        "@langchain/ollama",
        "convert OllamaConfig LLM configs",
        "remove them from the spec.",
      );
      return new ChatOllama({
        baseUrl: llmConfig.url,
        model: llmConfig.modelId,
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
      });
    case "OciGenAiConfig":
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
