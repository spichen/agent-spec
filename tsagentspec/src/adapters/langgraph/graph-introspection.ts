/**
 * Structural introspection over LangGraph internals.
 *
 * Single owner of everything the adapter probes off LangGraph runtime
 * objects — the `lg_is_pregel` compiled-graph fingerprint, the StateGraph
 * builder surface, and the state-schema key extraction — shared by both
 * converter directions. These are probed-stable but private surfaces of
 * `@langchain/langgraph`, so a langgraph version bump gets fixed here and
 * nowhere else.
 */
import { isRecordLike } from "../common/index.js";

/** Runtime shape of one LangGraph builder node spec. */
export interface NodeSpecLike {
  runnable?: unknown;
  input?: unknown;
}

/** Runtime shape of one LangGraph conditional-edge branch. */
export interface BranchLike {
  path?: unknown;
  ends?: Record<string, string>;
}

/** Runtime shape of a LangGraph StateGraph builder. */
export interface BuilderLike {
  nodes: Record<string, NodeSpecLike>;
  edges: Iterable<[string, string]>;
  branches?: Record<string, Record<string, BranchLike>>;
  channels?: Record<string, unknown>;
  _schemaDefinition?: unknown;
  _inputDefinition?: unknown;
  _outputDefinition?: unknown;
}

/** Duck-type check for a compiled LangGraph graph (the `lg_is_pregel` probe). */
export function isCompiledGraphLike(
  value: unknown,
): value is { builder: BuilderLike; name?: unknown } {
  return (
    isRecordLike(value) &&
    (value as { lg_is_pregel?: unknown }).lg_is_pregel === true
  );
}

/** Duck-type check for a StateGraph builder. */
export function isStateGraphBuilderLike(value: unknown): value is BuilderLike {
  if (!isRecordLike(value)) {
    return false;
  }
  const candidate = value as {
    nodes?: unknown;
    compile?: unknown;
    addNode?: unknown;
  };
  return (
    isRecordLike(candidate.nodes) &&
    typeof candidate.compile === "function" &&
    typeof candidate.addNode === "function"
  );
}

/** Duck-type check for anything convertible to a Flow (builder or compiled). */
export function isStateGraphLike(value: unknown): boolean {
  return isCompiledGraphLike(value) || isStateGraphBuilderLike(value);
}

/** Normalize a compiled graph or builder to the builder. */
export function getGraphBuilder(graph: unknown): BuilderLike {
  if (isCompiledGraphLike(graph)) {
    return graph.builder;
  }
  return graph as BuilderLike;
}

/**
 * Extract the state-key names of a schema definition: a langgraph channel
 * map, an `Annotation.Root` (via `.spec`) or a zod object (via `.shape`).
 */
export function definitionKeys(definition: unknown): string[] | undefined {
  if (!isRecordLike(definition)) {
    return undefined;
  }
  const spec = (definition as { spec?: unknown }).spec;
  if (isRecordLike(spec)) {
    return Object.keys(spec);
  }
  const shape = (definition as { shape?: unknown }).shape;
  if (isRecordLike(shape)) {
    return Object.keys(shape);
  }
  return Object.keys(definition);
}

/**
 * The builder's state-schema keys, honoring the `_schemaDefinition` (JS
 * builders constructed from Annotation.Root / zod) over the channel map.
 */
export function stateSchemaKeys(builder: BuilderLike): string[] | undefined {
  return (
    definitionKeys(builder._schemaDefinition) ?? definitionKeys(builder.channels)
  );
}
