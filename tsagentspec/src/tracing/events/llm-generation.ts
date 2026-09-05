/**
 * LLM generation events (port of `pyagentspec.tracing.events.llmgeneration`).
 */
import type { LlmConfig, LlmGenerationConfig } from "../../llms/index.js";
import type { Tool } from "../../tools/index.js";
import { Message, ToolCall, type ToolCallOptions } from "../message.js";
import { Event, type EventOptions } from "./event.js";

export { ToolCall, type ToolCallOptions };

export interface LlmGenerationRequestOptions extends EventOptions {
  /** The LlmConfig that performs the generation */
  llmConfig: LlmConfig;
  /** The content of the prompt that will be sent to the LLM (sensitive) */
  prompt: Message[];
  /** The list of tools sent as part of the generation request */
  tools: Tool[];
  /** Identifier of the generation request */
  requestId: string;
  /** The LLM configuration used for this LLM call */
  llmGenerationConfig?: LlmGenerationConfig | null;
}

/** An LLM generation request was received. Start of the LlmGenerationSpan. */
export class LlmGenerationRequest extends Event {
  llmConfig: LlmConfig;
  prompt: Message[];
  tools: Tool[];
  requestId: string;
  llmGenerationConfig: LlmGenerationConfig | null;

  override get type(): string {
    return "LlmGenerationRequest";
  }

  constructor(options: LlmGenerationRequestOptions) {
    super(options);
    this.llmConfig = options.llmConfig;
    this.prompt = options.prompt;
    this.tools = options.tools;
    this.requestId = options.requestId;
    this.llmGenerationConfig = options.llmGenerationConfig ?? null;
  }
}

export interface LlmGenerationResponseOptions extends EventOptions {
  /** The LlmConfig that performed the generation */
  llmConfig: LlmConfig;
  /** The content of the response received from the LLM (sensitive) */
  content: string | null;
  /** The list of tool calls that should be performed, received as part of the generation response (sensitive) */
  toolCalls?: ToolCall[];
  /** Identifier of the generation request */
  requestId: string;
  /** The identifier of the completion related to this response */
  completionId?: string | null;
  /** Number of input tokens */
  inputTokens?: number | null;
  /** Number of output tokens */
  outputTokens?: number | null;
}

/** An LLM response was received. End of an LlmGenerationSpan. */
export class LlmGenerationResponse extends Event {
  llmConfig: LlmConfig;
  content: string | null;
  toolCalls: ToolCall[];
  requestId: string;
  completionId: string | null;
  inputTokens: number | null;
  outputTokens: number | null;

  override get type(): string {
    return "LlmGenerationResponse";
  }

  constructor(options: LlmGenerationResponseOptions) {
    super(options);
    this.llmConfig = options.llmConfig;
    this.content = options.content;
    this.toolCalls = options.toolCalls ?? [];
    this.requestId = options.requestId;
    this.completionId = options.completionId ?? null;
    this.inputTokens = options.inputTokens ?? null;
    this.outputTokens = options.outputTokens ?? null;
  }
}

export interface LlmGenerationChunkReceivedOptions extends EventOptions {
  /** The LlmConfig that performs the generation */
  llmConfig: LlmConfig;
  /** The content of the chunk received from the LLM (sensitive) */
  content: string | null;
  /** Identifier of the generation request */
  requestId: string;
  /** The list of tool calls that should be performed, received as part of the generation response chunk (sensitive) */
  toolCalls?: ToolCall[];
  /** The identifier of the completion related to this response chunk */
  completionId?: string | null;
  /** Number of output tokens for this chunk */
  outputTokens?: number | null;
}

/** A chunk of an LLM response was received during streaming generation. */
export class LlmGenerationChunkReceived extends Event {
  llmConfig: LlmConfig;
  content: string | null;
  requestId: string;
  toolCalls: ToolCall[];
  completionId: string | null;
  outputTokens: number | null;

  override get type(): string {
    return "LlmGenerationChunkReceived";
  }

  constructor(options: LlmGenerationChunkReceivedOptions) {
    super(options);
    this.llmConfig = options.llmConfig;
    this.content = options.content;
    this.requestId = options.requestId;
    this.toolCalls = options.toolCalls ?? [];
    this.completionId = options.completionId ?? null;
    this.outputTokens = options.outputTokens ?? null;
  }
}
