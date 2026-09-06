/**
 * LlmGenerationSpan (port of `pyagentspec.tracing.spans.llm`).
 */
import type { LlmConfig } from "../../llms/index.js";
import { Span, type SpanOptions } from "./span.js";

export interface LlmGenerationSpanOptions extends SpanOptions {
  /** The LlmConfig that performs the generation */
  llmConfig: LlmConfig;
}

/**
 * Span that covers the whole LLM generation process.
 *
 * - Starts when: the LLM generation request is received and the LLM call is performed
 * - Ends when: the LLM output was generated, and it's ready to be processed
 */
export class LlmGenerationSpan extends Span {
  llmConfig: LlmConfig;

  override get type(): string {
    return "LlmGenerationSpan";
  }

  constructor(options: LlmGenerationSpanOptions) {
    super(options);
    this.llmConfig = options.llmConfig;
  }
}
