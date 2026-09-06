/**
 * Ollama LLM config.
 */
import { z } from "zod";
import { ComponentBaseSchema } from "../component.js";
import { RetryPolicySchema } from "../retry-policy.js";
import { LlmGenerationConfigSchema, OpenAIAPIType } from "./llm-config.js";

export const OllamaConfigSchema = ComponentBaseSchema.extend({
  componentType: z.literal("OllamaConfig"),
  url: z.string(),
  modelId: z.string(),
  apiType: z
    .enum([OpenAIAPIType.CHAT_COMPLETIONS, OpenAIAPIType.RESPONSES])
    .default(OpenAIAPIType.CHAT_COMPLETIONS),
  defaultGenerationParameters: LlmGenerationConfigSchema.optional(),
  apiKey: z.string().optional(),
  retryPolicy: RetryPolicySchema.optional(),
});

export type OllamaConfig = z.infer<typeof OllamaConfigSchema>;

export function createOllamaConfig(opts: {
  name: string;
  url: string;
  modelId: string;
  id?: string;
  description?: string;
  metadata?: Record<string, unknown>;
  apiType?: OpenAIAPIType;
  defaultGenerationParameters?: z.infer<typeof LlmGenerationConfigSchema>;
  apiKey?: string;
  retryPolicy?: z.input<typeof RetryPolicySchema>;
}): OllamaConfig {
  const parsed = OllamaConfigSchema.parse({
    ...opts,
    componentType: "OllamaConfig" as const,
  });
  return Object.freeze(parsed);
}
