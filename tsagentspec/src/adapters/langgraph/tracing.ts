/**
 * Tracing seam for the LangGraph adapter.
 *
 * The Python adapter wraps every compiled agent / flow / manager-workers
 * graph in an execution span (patching `stream`/`astream`). The TypeScript
 * SDK has no tracing package yet, so `patchWithExecutionSpan` is an identity
 * seam: it is invoked from the same graph-compilation sites as Python, and
 * receives the Agent Spec component the graph was compiled from, so a future
 * port of `pyagentspec.tracing` only needs to fill in the implementation here
 * without touching the converter.
 *
 * Python's LLM and tool callback handlers are NOT seamed here: no callbacks
 * are attached in this adapter (see the divergence notes in `llm.ts`,
 * `tools.ts` and `mcp.ts`), so a tracing port must add those attachment
 * sites itself.
 */
import type { Agent, ManagerWorkers } from "../../agents/index.js";
import type { Flow } from "../../flows/index.js";

/**
 * Which execution span Python opens around a compiled graph, and the Agent
 * Spec component that span reports on.
 *
 * Python opens an `AgentExecutionSpan` for react agents, a
 * `FlowExecutionSpan` for compiled flows and a `ManagerWorkersExecutionSpan`
 * for hierarchical manager-workers graphs.
 */
export type ExecutionSpanTarget =
  | { kind: "agent"; component: Agent }
  | { kind: "flow"; component: Flow }
  | { kind: "manager-workers"; component: ManagerWorkers };

/**
 * Wrap a compiled graph (or react agent) so each run is traced inside an
 * execution span.
 *
 * Python monkey-patches `stream`/`astream` to open the span named by
 * `target.kind` for `target.component`, emit the start event with the
 * invocation inputs, fold the streamed chunks into a final state and emit the
 * end event with the run outputs. Returns the graph unchanged until the
 * tracing package is ported.
 */
export function patchWithExecutionSpan<T>(
  graph: T,
  _target: ExecutionSpanTarget,
): T {
  return graph;
}
