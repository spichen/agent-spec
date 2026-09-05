/**
 * Ambient trace/span context (port of Python's contextvars machinery).
 *
 * Python keeps two ContextVars: `_TRACE` (the ambient Trace) and
 * `_ACTIVE_SPAN_STACK` (the stack of active spans, copy-on-write). In
 * async-only JS both live in a single AsyncLocalStorage store; when no store
 * is active (plain sequential code without `run()` scoping), a module-level
 * fallback store is used.
 *
 * Mutations replace the stack array on the current store (never mutate it in
 * place), so a forked child context — created by `Trace.run` / `Span.run` —
 * holding its own copied stack stays isolated from parallel branches, while
 * sequential `start()`/`end()` calls in the same context observe each other.
 */
import { AsyncLocalStorage } from "node:async_hooks";
import type { Span } from "./spans/span.js";
import type { Trace } from "./trace.js";

interface TraceContextStore {
  trace: Trace | undefined;
  spanStack: readonly Span[];
}

const storage = new AsyncLocalStorage<TraceContextStore>();

/** Fallback store for code running outside any `run()`-scoped context. */
const moduleStore: TraceContextStore = { trace: undefined, spanStack: [] };

function currentStore(): TraceContextStore {
  return storage.getStore() ?? moduleStore;
}

/**
 * Get the Trace object active in the current context.
 *
 * @returns The active Trace object, or `undefined` when no trace is active.
 */
export function getTrace(): Trace | undefined {
  return currentStore().trace;
}

/**
 * Retrieve the stack of active spans in this context.
 *
 * @returns A copy of the stack of active spans in this context.
 */
export function getActiveSpanStack(): Span[] {
  return [...currentStore().spanStack];
}

/**
 * Retrieve the currently active span in this context.
 *
 * @returns The active span in this context, or `undefined` when none is active.
 */
export function getCurrentSpan(): Span | undefined {
  const spanStack = currentStore().spanStack;
  return spanStack.length > 0 ? spanStack[spanStack.length - 1] : undefined;
}

/** @internal Set (or clear) the ambient trace on the current context. */
export function setAmbientTrace(trace: Trace | undefined): void {
  currentStore().trace = trace;
}

/** @internal Push a span onto the active stack (copy-on-write). */
export function appendSpanToActiveStack(span: Span): void {
  const store = currentStore();
  store.spanStack = [...store.spanStack, span];
}

/** @internal Pop the top span from the active stack (copy-on-write). */
export function popSpanFromActiveStack(): void {
  const store = currentStore();
  store.spanStack = store.spanStack.slice(0, -1);
}

/**
 * @internal Fork a child context seeded with the current trace and a copy of
 * the current span stack, and return a runner that executes functions inside
 * that same forked store on every call. Needed where one logical branch spans
 * multiple resumptions driven from the parent context — e.g. an async
 * generator consumed by the caller: async generator bodies resume in the
 * context of whoever calls `next()`, so each resumption must re-enter the
 * forked store explicitly.
 */
export function forkChildContext(): <T>(fn: () => T) => T {
  const store = currentStore();
  const childStore: TraceContextStore = {
    trace: store.trace,
    spanStack: [...store.spanStack],
  };
  return (fn) => storage.run(childStore, fn);
}

/**
 * @internal Run `fn` in a forked child context seeded with the current trace
 * and a copy of the current span stack. Mutations inside the child (span
 * pushes/pops, trace set/clear) are invisible to the parent and to parallel
 * sibling branches — the JS equivalent of asyncio tasks copying the Python
 * context at creation time.
 */
export function runInChildContext<T>(fn: () => T): T {
  return forkChildContext()(fn);
}
