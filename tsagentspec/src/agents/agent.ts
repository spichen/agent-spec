/**
 * Agent component.
 */
import { z } from "zod";
import { ComponentWithIOSchema } from "../component.js";
import type { Property } from "../property.js";
import { getPlaceholderPropertiesFromJsonObject } from "../templating.js";
import { LlmConfigUnion, type LlmConfig } from "../llms/index.js";
import { ToolUnion, ToolBoxUnion, type Tool, type ToolBox } from "../tools/index.js";
import {
  MessageTransformUnion,
  type MessageTransform,
} from "../transforms/index.js";

// z.lazy(() => z.record(z.unknown())) breaks the circular dependency
// (Agent -> AgenticComponentUnion -> Agent). Deep validation of sub-agents
// is handled by the createAgent factory and at deserialization time.
const AgenticComponentRef = z.lazy(() => z.record(z.unknown()));

export const AgentSchema = ComponentWithIOSchema.extend({
  componentType: z.literal("Agent"),
  llmConfig: LlmConfigUnion,
  systemPrompt: z.string(),
  tools: z.array(ToolUnion).default([]),
  toolboxes: z.array(ToolBoxUnion).default([]),
  humanInTheLoop: z.boolean().default(true),
  transforms: z.array(MessageTransformUnion).default([]),
  subAgents: z.array(AgenticComponentRef).default([]),
});

export type Agent = z.infer<typeof AgentSchema>;

/** Collect all sub-agent names reachable from a component (depth-first). */
function collectSubAgentNames(
  component: Record<string, unknown>,
  visited: Set<string>,
): void {
  const name = component["name"] as string | undefined;
  if (!name || visited.has(name)) return;
  visited.add(name);
  const children = component["subAgents"] as Record<string, unknown>[] | undefined;
  if (Array.isArray(children)) {
    for (const child of children) {
      collectSubAgentNames(child as Record<string, unknown>, visited);
    }
  }
}

export function createAgent(opts: {
  name: string;
  llmConfig: LlmConfig;
  systemPrompt: string;
  id?: string;
  description?: string;
  metadata?: Record<string, unknown>;
  tools?: Tool[];
  toolboxes?: ToolBox[];
  humanInTheLoop?: boolean;
  transforms?: MessageTransform[];
  subAgents?: Record<string, unknown>[];
  inputs?: Property[];
  outputs?: Property[];
}): Agent {
  // Validate unique sub-agent names
  const subAgents = opts.subAgents ?? [];
  const names = subAgents.map((sa) => sa["name"] as string);
  const duplicates = names.filter((n, i) => names.indexOf(n) !== i);
  if (duplicates.length > 0) {
    const unique = [...new Set(duplicates)].sort();
    throw new Error(
      `Sub-agent names must be unique within a parent agent. Duplicate name(s) found: ${unique.join(", ")}`,
    );
  }

  // Validate no cycles: this agent's name must not appear in the sub-agent hierarchy
  const visited = new Set<string>();
  for (const sa of subAgents) {
    collectSubAgentNames(sa as Record<string, unknown>, visited);
  }
  if (visited.has(opts.name)) {
    throw new Error(
      `Cycle detected in subAgents: agent '${opts.name}' appears in its own sub-agent hierarchy.`,
    );
  }

  const inputs =
    opts.inputs ?? getPlaceholderPropertiesFromJsonObject(opts.systemPrompt);
  const parsed = AgentSchema.parse({
    ...opts,
    inputs,
    outputs: opts.outputs ?? [],
    componentType: "Agent" as const,
  });
  return Object.freeze(parsed);
}
