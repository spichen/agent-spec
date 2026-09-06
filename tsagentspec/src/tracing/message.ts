/**
 * Plain data models carried by tracing events.
 *
 * `Message` mirrors `pyagentspec.tracing.messages.message.Message`; `ToolCall`
 * mirrors `pyagentspec.tracing.events.llmgeneration.ToolCall` (defined here so
 * the serialization core can reference both without import cycles, and
 * re-exported from `events/llm-generation.ts` to match the Python layout).
 * Neither is an Event: their wire dumps carry no `type` discriminant.
 */

export interface MessageOptions {
  /** Identifier of the message */
  id?: string | null;
  /** Content of the message */
  content: string;
  /** Sender of the message */
  sender?: string | null;
  /** Role of the sender of the message. Typically "user", "assistant", or "system" */
  role: string;
}

/** Model used to specify LLM message details in events and spans */
export class Message {
  id: string | null;
  content: string;
  sender: string | null;
  role: string;

  constructor(options: MessageOptions) {
    this.id = options.id ?? null;
    this.content = options.content;
    this.sender = options.sender ?? null;
    this.role = options.role;
  }

  /** Wire dump matching Python's `Message.model_dump()`. */
  toWireDict(): Record<string, unknown> {
    return {
      id: this.id,
      content: this.content,
      sender: this.sender,
      role: this.role,
    };
  }
}

export interface ToolCallOptions {
  /** Identifier of the tool call */
  callId: string;
  /** The name of the tool that should be called */
  toolName: string;
  /** The values of the arguments that should be passed to the tool, in JSON format */
  arguments: string;
}

/** Model for an LLM tool call. */
export class ToolCall {
  callId: string;
  toolName: string;
  arguments: string;

  constructor(options: ToolCallOptions) {
    this.callId = options.callId;
    this.toolName = options.toolName;
    this.arguments = options.arguments;
  }

  /** Wire dump matching Python's `ToolCall.model_dump()`. */
  toWireDict(): Record<string, unknown> {
    return {
      call_id: this.callId,
      tool_name: this.toolName,
      arguments: this.arguments,
    };
  }
}
