/**
 * Bare LlmConfig — provider-agnostic LLM connection config (Agent Spec >= 26.1.2).
 *
 * Describes any LLM without a dedicated subclass: `apiProvider` selects the
 * serving API a runtime dispatches on to pick a client (e.g. "openai",
 * "oci", "vllm"), `provider` names the model maker (e.g. "meta", "openai",
 * "cohere"), and `apiType` picks the wire protocol (e.g. "chat_completions",
 * "responses"). All three are free-form strings here; the dedicated config
 * components pin them instead.
 *
 * The wire component_type is "LlmConfig" (matching Python's now-concrete
 * base class); the TS type is exported as GenericLlmConfig because the
 * public `LlmConfig` type name is already taken by the union of all LLM
 * config components. Configs ported later (e.g. GeminiConfig,
 * DbmsVectorChainLlmConfig) must also carry `retryPolicy`.
 */
import { z } from "zod";
import { ComponentBaseSchema } from "../component.js";
import { LlmGenerationConfigSchema } from "./llm-config.js";
import { RetryPolicySchema } from "../retry-policy.js";

export const LlmConfigSchema = ComponentBaseSchema.extend({
  componentType: z.literal("LlmConfig"),
  /** Identifier of the model to use, as expected by the selected API provider. */
  modelId: z.string(),
  /** The provider of the model (e.g. "meta", "openai", "cohere"). */
  provider: z.string().optional(),
  /** The API provider used to serve the model (e.g. "openai", "oci", "vllm"). */
  apiProvider: z.string().optional(),
  /** The API format to use (e.g. "chat_completions", "responses"). */
  apiType: z.string().optional(),
  /** URL of the API endpoint (e.g. "https://api.openai.com/v1"). */
  url: z.string().optional(),
  /** Optional API key for the remote LLM model — sensitive, never serialized. */
  apiKey: z.string().optional(),
  /** Parameters used for the generation call of this LLM. */
  defaultGenerationParameters: LlmGenerationConfigSchema.optional(),
  /** Optional retry configuration for remote LLM calls. */
  retryPolicy: RetryPolicySchema.optional(),
});

export type GenericLlmConfig = z.infer<typeof LlmConfigSchema>;

export function createLlmConfig(opts: {
  name: string;
  modelId: string;
  id?: string;
  description?: string;
  metadata?: Record<string, unknown>;
  provider?: string;
  apiProvider?: string;
  apiType?: string;
  url?: string;
  apiKey?: string;
  defaultGenerationParameters?: z.infer<typeof LlmGenerationConfigSchema>;
  retryPolicy?: z.input<typeof RetryPolicySchema>;
}): GenericLlmConfig {
  return Object.freeze(
    LlmConfigSchema.parse({
      ...opts,
      componentType: "LlmConfig" as const,
    }),
  );
}
