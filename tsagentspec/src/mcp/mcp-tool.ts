/**
 * MCP Tool and MCP ToolSpec.
 */
import { z } from "zod";
import { ComponentWithIOSchema } from "../component.js";
import type { Property } from "../property.js";
import { RetryPolicySchema } from "../retry-policy.js";
import { ToolBaseSchema } from "../tools/tool.js";
import { ClientTransportUnion, type ClientTransport } from "./client-transport.js";

export const MCPToolSchema = ToolBaseSchema.extend({
  componentType: z.literal("MCPTool"),
  clientTransport: ClientTransportUnion,
  /**
   * Optional retry configuration for semantic MCP tool resolution and
   * execution. Only the attempt and backoff fields apply to this semantic
   * retry; transport request timeout and HTTP status retry fields belong to
   * retry policies on remote MCP transports.
   */
  retryPolicy: RetryPolicySchema.optional(),
});

export type MCPTool = z.infer<typeof MCPToolSchema>;

export function createMCPTool(opts: {
  name: string;
  clientTransport: ClientTransport;
  id?: string;
  description?: string;
  metadata?: Record<string, unknown>;
  retryPolicy?: z.input<typeof RetryPolicySchema>;
  inputs?: Property[];
  outputs?: Property[];
  requiresConfirmation?: boolean;
}): MCPTool {
  return Object.freeze(
    MCPToolSchema.parse({
      ...opts,
      componentType: "MCPTool" as const,
    }),
  );
}

export const MCPToolSpecSchema = ComponentWithIOSchema.extend({
  componentType: z.literal("MCPToolSpec"),
  requiresConfirmation: z.boolean().default(false),
});

export type MCPToolSpec = z.infer<typeof MCPToolSpecSchema>;

export function createMCPToolSpec(opts: {
  name: string;
  id?: string;
  description?: string;
  metadata?: Record<string, unknown>;
  inputs?: Property[];
  outputs?: Property[];
  requiresConfirmation?: boolean;
}): MCPToolSpec {
  return Object.freeze(
    MCPToolSpecSchema.parse({
      ...opts,
      componentType: "MCPToolSpec" as const,
    }),
  );
}
