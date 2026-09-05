/**
 * ToolBox and MCPToolBox.
 */
import { z } from "zod";
import { ComponentBaseSchema } from "../component.js";
import { ClientTransportUnion, type ClientTransport } from "../mcp/client-transport.js";
import { MCPToolSpecSchema } from "../mcp/mcp-tool.js";
import { RetryPolicySchema } from "../retry-policy.js";

export const MCPToolBoxSchema = ComponentBaseSchema.extend({
  componentType: z.literal("MCPToolBox"),
  clientTransport: ClientTransportUnion,
  /**
   * Optional retry configuration for semantic MCP toolbox discovery and
   * generated tool execution. Only the attempt and backoff fields apply to
   * this semantic retry; transport request timeout and HTTP status retry
   * fields belong to retry policies on remote MCP transports.
   */
  retryPolicy: RetryPolicySchema.optional(),
  toolFilter: z
    .array(z.union([MCPToolSpecSchema, z.string()]))
    .optional(),
  requiresConfirmation: z.boolean().default(false),
});

export type MCPToolBox = z.infer<typeof MCPToolBoxSchema>;

export function createMCPToolBox(opts: {
  name: string;
  clientTransport: ClientTransport;
  id?: string;
  description?: string;
  metadata?: Record<string, unknown>;
  retryPolicy?: z.input<typeof RetryPolicySchema>;
  toolFilter?: Array<z.infer<typeof MCPToolSpecSchema> | string>;
  requiresConfirmation?: boolean;
}): MCPToolBox {
  return Object.freeze(
    MCPToolBoxSchema.parse({
      ...opts,
      componentType: "MCPToolBox" as const,
    }),
  );
}

export const ToolBoxUnion = z.discriminatedUnion("componentType", [
  MCPToolBoxSchema,
]);

export type ToolBox = z.infer<typeof ToolBoxUnion>;
