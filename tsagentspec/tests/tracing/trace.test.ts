/**
 * Port of pyagentspec/tests/tracing/test_tracing.py (async-only: Python's
 * sync/async twin cases collapse onto the single async API, so the
 * sync-vs-async segregation and NotImplementedError-fallback cases are N/A).
 */
import { describe, expect, it } from "vitest";
import {
  Event,
  ExceptionRaised,
  RootSpan,
  Span,
  SpanProcessor,
  Trace,
  getActiveSpanStack,
  getCurrentSpan,
  getTrace,
} from "../../src/index.js";

const UUID_RE =
  /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$/;

function nowNs(): number {
  return Date.now() * 1e6;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

class DummySpanProcessor extends SpanProcessor {
  startedUp = false;
  shutDown = false;
  starts: Span[] = [];
  ends: Span[] = [];
  events: Array<[Event, Span]> = [];

  onStart(span: Span): void {
    this.starts.push(span);
  }

  onEnd(span: Span): void {
    this.ends.push(span);
  }

  onEvent(event: Event, span: Span): void {
    this.events.push([event, span]);
  }

  startup(): void {
    this.startedUp = true;
  }

  shutdown(): void {
    this.shutDown = true;
  }
}

class FailingStartSpanProcessor extends DummySpanProcessor {
  override onStart(span: Span): void {
    this.starts.push(span);
    throw new Error("start failed");
  }
}

class FailingEndSpanProcessor extends DummySpanProcessor {
  constructor(private readonly message: string) {
    super();
  }

  override onEnd(span: Span): void {
    this.ends.push(span);
    throw new Error(this.message);
  }
}

describe("tracing core", () => {
  it("event defaults: name is the class name, uuid id, ns timestamp", () => {
    const before = nowNs();
    const event = new Event();
    const after = nowNs();
    expect(event.name).toBe("Event");
    expect(typeof event.id).toBe("string");
    expect(event.id).toMatch(UUID_RE);
    expect(event.timestamp).toBeGreaterThan(0);
    expect(event.timestamp).toBeGreaterThanOrEqual(before);
    expect(event.timestamp).toBeLessThanOrEqual(after);
  });

  it("span instantiation defaults", () => {
    const span = new Span();
    expect(span.name).toBe("Span");
    expect(typeof span.id).toBe("string");
    expect(span.id).toMatch(UUID_RE);
    expect(span.startTime).toBeNull();
    expect(span.endTime).toBeNull();
    expect(span.events).toEqual([]);
    // Not started, shouldn't be active
    expect(getCurrentSpan()).toBeUndefined();
  });

  it("exception event creation defaults", () => {
    const event = new ExceptionRaised({
      exceptionType: "ValueError",
      exceptionMessage: "bad input",
    });
    expect(event.name).toBe("ExceptionRaised");
    expect(event.exceptionType).toBe("ValueError");
    expect(event.exceptionMessage).toBe("bad input");
    expect(typeof event.exceptionStacktrace).toBe("string");
  });

  it("span start/end updates timestamps and the active stack", async () => {
    const stackLenBefore = getActiveSpanStack().length;
    const span = new Span({ name: "current_span" });
    const beforeSpanStart = nowNs();
    await span.start();
    const afterSpanStartBeforeEnd = nowNs();
    // Span is the current one while active
    expect(getCurrentSpan()).toBe(span);
    expect(span.startTime).not.toBeNull();
    expect(span.startTime!).toBeGreaterThanOrEqual(beforeSpanStart);
    expect(span.startTime!).toBeLessThanOrEqual(afterSpanStartBeforeEnd);
    expect(span.endTime).toBeNull();
    // Active stack grew by 1
    expect(getActiveSpanStack().length).toBe(stackLenBefore + 1);
    await span.end();
    const afterSpanEnd = nowNs();
    // After exit, span is closed and stack restored
    expect(span.endTime).not.toBeNull();
    expect(span.endTime!).toBeGreaterThanOrEqual(afterSpanStartBeforeEnd);
    expect(span.endTime!).toBeLessThanOrEqual(afterSpanEnd);
    expect(getCurrentSpan()).toBeUndefined();
    expect(getActiveSpanStack().length).toBe(stackLenBefore);
  });

  it("span run() scopes the span like Python's context manager", async () => {
    const stackLenBefore = getActiveSpanStack().length;
    const span = new Span({ name: "current_span" });
    await span.run(async (s) => {
      expect(s).toBe(span);
      expect(getCurrentSpan()).toBe(span);
      expect(span.startTime).not.toBeNull();
      expect(span.endTime).toBeNull();
      expect(getActiveSpanStack().length).toBe(stackLenBefore + 1);
    });
    expect(span.endTime).not.toBeNull();
    expect(span.endTime!).toBeGreaterThanOrEqual(span.startTime!);
    expect(getCurrentSpan()).toBeUndefined();
    expect(getActiveSpanStack().length).toBe(stackLenBefore);
  });

  it("nested spans record their parent span", async () => {
    const parent = new Span();
    await parent.run(async () => {
      const child = new Span();
      await child.run(async () => {
        expect(child.parentSpan).toBe(parent);
        expect(getCurrentSpan()).toBe(child);
      });
      // After child exits, current is parent
      expect(getCurrentSpan()).toBe(parent);
    });
    expect(getCurrentSpan()).toBeUndefined();
  });

  it("span processor hooks are called through the span lifecycle", async () => {
    const processor = new DummySpanProcessor();
    const rootSpan = new RootSpan();
    const trace = new Trace({ name: "T1", spanProcessors: [processor], rootSpan });
    let innerSpan: Span | undefined;
    await trace.run(async (t) => {
      // Trace set in context, root span active
      expect(getActiveSpanStack()).toContain(rootSpan);
      expect(getCurrentSpan()).toBe(rootSpan);
      expect(processor.starts).toHaveLength(1);
      expect(processor.starts[0]).toBe(rootSpan);
      expect(getTrace()).toBe(t);
      const span = new Span();
      innerSpan = span;
      await span.run(async () => {
        expect(getActiveSpanStack()).toContain(span);
        // onStart called for processor
        expect(processor.starts).toHaveLength(2);
        expect(processor.starts[1]).toBe(span);
        const event = new Event({ name: "custom_event" });
        await span.addEvent(event);
        // Event recorded both in span and processor
        expect(span.events).toHaveLength(1);
        expect(span.events[0]).toBe(event);
        expect(processor.events).toHaveLength(1);
        expect(processor.events[0]![0]).toBe(event);
        expect(processor.events[0]![1]).toBe(span);
      });
      // onEnd called
      expect(processor.ends).toHaveLength(1);
      expect(processor.ends[0]).toBe(span);
      // Trace lifecycle hooks were invoked
      expect(processor.startedUp).toBe(true);
      expect(getActiveSpanStack()).toContain(rootSpan);
      expect(getCurrentSpan()).toBe(rootSpan);
    });
    expect(processor.ends).toHaveLength(2);
    expect(processor.ends[0]).toBe(innerSpan);
    expect(processor.ends[1]).toBe(rootSpan);
    expect(processor.shutDown).toBe(true);
    // After exiting the trace, no active trace
    expect(getTrace()).toBeUndefined();
  });

  it("trace startup/shutdown are called and nested traces are rejected", async () => {
    const processor = new DummySpanProcessor();
    const trace = new Trace({ spanProcessors: [processor] });
    await trace.run(async () => {
      expect(processor.startedUp).toBe(true);
      expect(getTrace()).toBeDefined();
      // Nested Trace not allowed
      await expect(new Trace().run(async () => undefined)).rejects.toThrow(
        "A Trace already exists. Cannot create two nested Traces.",
      );
      await expect(new Trace().start()).rejects.toThrow("A Trace already exists");
    });
    expect(processor.shutDown).toBe(true);
  });

  it("a span records an ExceptionRaised event when its body throws", async () => {
    const span = new Span();
    await new Trace().run(async () => {
      await expect(
        span.run(async () => {
          throw new Error("boom");
        }),
      ).rejects.toThrow("boom");
    });
    // After the exception, the span ended and contains an ExceptionRaised event
    const exceptionEvent = span.events.find(
      (event): event is ExceptionRaised => event instanceof ExceptionRaised,
    );
    expect(exceptionEvent).toBeDefined();
    expect(exceptionEvent!.exceptionType).toBe("Error");
    expect(exceptionEvent!.exceptionMessage).toBe("boom");
    expect(exceptionEvent!.exceptionStacktrace).toContain("Error: boom");
    expect(span.endTime).not.toBeNull();
  });

  it("a failure in a processor onStart triggers cleanup", async () => {
    const successfulProcessor = new DummySpanProcessor();
    const failingProcessor = new FailingStartSpanProcessor();

    const trace = new Trace({ spanProcessors: [successfulProcessor] });
    await trace.run(async (t) => {
      // Spans read the live processor list from the ambient trace at start time
      t.spanProcessors = [successfulProcessor, failingProcessor];
      await expect(new Span({ name: "startup-failure" }).start()).rejects.toThrow(
        "start failed",
      );

      const failedSpan = successfulProcessor.ends[0];
      expect(failedSpan).toBeInstanceOf(Span);
      // Only the successfully started processor got onEnd
      expect(successfulProcessor.ends).toHaveLength(1);
      expect(failingProcessor.ends).toHaveLength(0);
      // The failed span never entered the active stack
      expect(getCurrentSpan()).toBe(t.rootSpan);
      expect(getActiveSpanStack()).not.toContain(failedSpan);
      // The successful processor saw the ExceptionRaised event
      expect(
        successfulProcessor.events.some(
          ([event, span]) => event instanceof ExceptionRaised && span === failedSpan,
        ),
      ).toBe(true);
      // Restore so the root span end is clean
      t.spanProcessors = [successfulProcessor];
    });
  });

  it("end() notifies every started processor, rethrows the first error, and pops the stack", async () => {
    const failingFirst = new FailingEndSpanProcessor("end failed first");
    const failingSecond = new FailingEndSpanProcessor("end failed second");
    const normal = new DummySpanProcessor();

    const trace = new Trace();
    await trace.run(async (t) => {
      // Register after the root span started, so only the inner span sees them
      t.spanProcessors = [failingFirst, normal, failingSecond];
      const span = new Span();
      await span.start();
      await expect(span.end()).rejects.toThrow("end failed first");
      // Every started processor received onEnd despite the failures
      expect(failingFirst.ends).toHaveLength(1);
      expect(normal.ends).toHaveLength(1);
      expect(failingSecond.ends).toHaveLength(1);
      // The span was still popped from the active stack
      expect(getCurrentSpan()).toBe(t.rootSpan);
      expect(getActiveSpanStack()).not.toContain(span);
    });
  });

  it("addEvent only notifies processors that were started with the span", async () => {
    const earlyProcessor = new DummySpanProcessor();
    const lateProcessor = new DummySpanProcessor();
    const trace = new Trace({ spanProcessors: [earlyProcessor] });
    await trace.run(async (t) => {
      const span = new Span();
      await span.start();
      // Registered after the span started: receives no events from it
      t.spanProcessors = [earlyProcessor, lateProcessor];
      await span.addEvent(new Event({ name: "custom_event" }));
      expect(earlyProcessor.events).toHaveLength(1);
      expect(lateProcessor.events).toHaveLength(0);
      await span.end();
      t.spanProcessors = [earlyProcessor];
    });
  });

  it("trace.run returns the callback result and ends the trace on error", async () => {
    const processor = new DummySpanProcessor();
    const trace = new Trace({ spanProcessors: [processor] });
    const result = await trace.run(async () => 42);
    expect(result).toBe(42);
    expect(processor.shutDown).toBe(true);

    const failingTraceProcessor = new DummySpanProcessor();
    const failingTrace = new Trace({ spanProcessors: [failingTraceProcessor] });
    await expect(
      failingTrace.run(async () => {
        throw new Error("trace body failed");
      }),
    ).rejects.toThrow("trace body failed");
    // The trace still ended: root span closed, processors shut down, context clear
    expect(failingTrace.rootSpan.endTime).not.toBeNull();
    expect(failingTraceProcessor.shutDown).toBe(true);
    expect(getTrace()).toBeUndefined();
  });

  it("shutdownOnExit=false skips processor shutdown", async () => {
    const processor = new DummySpanProcessor();
    const trace = new Trace({ spanProcessors: [processor], shutdownOnExit: false });
    await trace.run(async () => undefined);
    expect(processor.startedUp).toBe(true);
    expect(processor.shutDown).toBe(false);
  });

  it("keeps parallel async branches isolated", async () => {
    const processor = new DummySpanProcessor();
    const rootSpan = new RootSpan();
    const trace = new Trace({ spanProcessors: [processor], rootSpan });

    await trace.run(async () => {
      const branch = async (label: string, delayMs: number): Promise<Span> => {
        const outer = new Span({ name: `${label}-outer` });
        await outer.run(async () => {
          await sleep(delayMs);
          // Only this branch's span is visible on top of the root span
          expect(getCurrentSpan()).toBe(outer);
          expect(outer.parentSpan).toBe(rootSpan);
          const inner = new Span({ name: `${label}-inner` });
          await inner.run(async () => {
            await sleep(delayMs);
            expect(getCurrentSpan()).toBe(inner);
            expect(inner.parentSpan).toBe(outer);
            expect(getActiveSpanStack().map((s) => s.name)).toEqual([
              "RootSpan",
              `${label}-outer`,
              `${label}-inner`,
            ]);
          });
          await sleep(delayMs);
          expect(getCurrentSpan()).toBe(outer);
        });
        return outer;
      };

      const [spanA, spanB] = await Promise.all([branch("a", 15), branch("b", 5)]);
      // Both branches are gone; the root span is current again
      expect(getCurrentSpan()).toBe(rootSpan);
      expect(getActiveSpanStack()).toEqual([rootSpan]);
      // Both branch spans started and ended, with the root as parent
      expect(spanA.parentSpan).toBe(rootSpan);
      expect(spanB.parentSpan).toBe(rootSpan);
      expect(spanA.endTime).not.toBeNull();
      expect(spanB.endTime).not.toBeNull();
    });

    // All 5 spans (root + 2 per branch) were started and ended exactly once
    expect(processor.starts).toHaveLength(5);
    expect(processor.ends).toHaveLength(5);
    expect(getTrace()).toBeUndefined();
  });
});
