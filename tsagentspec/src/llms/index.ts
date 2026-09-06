/**
 * LLM config types barrel export.
 */
import { z } from "zod";
import { LlmConfigSchema } from "./generic-llm-config.js";
import { OpenAiCompatibleConfigSchema } from "./openai-compatible-config.js";
import { OllamaConfigSchema } from "./ollama-config.js";
import { VllmConfigSchema } from "./vllm-config.js";
import { OpenAiConfigSchema } from "./openai-config.js";
import { OciGenAiConfigSchema } from "./oci-genai-config.js";

/**
 * Discriminated union of all LLM config types. The bare LlmConfig component
 * (wire component_type "LlmConfig") is a member; the union keeps the public
 * `LlmConfig` type name, so the bare component's TS type is exported as
 * `GenericLlmConfig`.
 */
export const LlmConfigUnion = z.discriminatedUnion("componentType", [
  LlmConfigSchema,
  OpenAiCompatibleConfigSchema,
  OllamaConfigSchema,
  VllmConfigSchema,
  OpenAiConfigSchema,
  OciGenAiConfigSchema,
]);

export type LlmConfig = z.infer<typeof LlmConfigUnion>;

export {
  LlmGenerationConfigSchema,
  OpenAIAPIType,
  type LlmGenerationConfig,
} from "./llm-config.js";

export {
  LlmConfigSchema,
  createLlmConfig,
  type GenericLlmConfig,
} from "./generic-llm-config.js";

export {
  OpenAiCompatibleConfigSchema,
  createOpenAiCompatibleConfig,
  type OpenAiCompatibleConfig,
} from "./openai-compatible-config.js";

export {
  OllamaConfigSchema,
  createOllamaConfig,
  type OllamaConfig,
} from "./ollama-config.js";

export {
  VllmConfigSchema,
  createVllmConfig,
  type VllmConfig,
} from "./vllm-config.js";

export {
  OpenAiConfigSchema,
  createOpenAiConfig,
  type OpenAiConfig,
} from "./openai-config.js";

export {
  OciGenAiConfigSchema,
  createOciGenAiConfig,
  ServingMode,
  ModelProvider,
  OciAPIType,
  type OciGenAiConfig,
} from "./oci-genai-config.js";

export {
  OciClientConfigUnion,
  OciClientConfigWithApiKeySchema,
  OciClientConfigWithInstancePrincipalSchema,
  OciClientConfigWithResourcePrincipalSchema,
  OciClientConfigWithSecurityTokenSchema,
  createOciClientConfigWithApiKey,
  createOciClientConfigWithInstancePrincipal,
  createOciClientConfigWithResourcePrincipal,
  createOciClientConfigWithSecurityToken,
  type OciClientConfig,
  type OciClientConfigWithApiKey,
  type OciClientConfigWithInstancePrincipal,
  type OciClientConfigWithResourcePrincipal,
  type OciClientConfigWithSecurityToken,
} from "./oci-client-config.js";
