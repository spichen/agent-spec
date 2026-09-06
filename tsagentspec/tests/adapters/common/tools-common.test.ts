/**
 * Tests for the shared templated-HTTP-request assembly and retry engine.
 *
 * The request-body matrix (JSON / urlencoded form / raw string / GET-HEAD
 * drop) is exercised end-to-end through the ApiNode flow tests; this file
 * pins the per-caller record-guard contract of `buildTemplatedHttpRequest`
 * (the strict default of the RemoteTool path versus the loose `isRecordLike`
 * guard the ApiNode executor passes, preserving each call site's
 * pre-unification behavior for non-plain data objects) and the retry-engine
 * internals ported from `pyagentspec.adapters._tools_common`: Retry-After
 * parsing, the four jitter modes, TLS-failure detection and the
 * exponential-backoff / total-elapsed-cap behavior of `requestWithRetry`
 * (the adapter-level retry matrix lives in the LangGraph remote-tools and
 * api-node suites).
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { RetryPolicySchema, type RetryPolicy } from "../../../src/index.js";
import { isRecordLike } from "../../../src/adapters/common/guards.js";
import {
  MAX_HTTP_ATTEMPTS_PER_CALL,
  MAX_HTTP_REQUEST_TIMEOUT_MS,
  MAX_RETRY_AFTER_SECONDS,
  MIN_RETRY_DELAY_SECONDS,
  buildTemplatedHttpRequest,
  clampRequestTimeoutMs,
  computeWaitSeconds,
  getRetryAfterSeconds,
  isNonRetryableLocalError,
  isTlsOrCertError,
  raiseForStatusWhenPolicySet,
  redactUrlQuery,
  requestWithRetry,
} from "../../../src/adapters/common/tools-common.js";

class InstancePayload {
  a = "1";
}

describe("buildTemplatedHttpRequest record guard", () => {
  const urlencodedSpec = {
    url: "https://api.example.com/form",
    httpMethod: "POST",
    data: new InstancePayload(),
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    queryParams: {},
  };

  it("strict default: a class-instance body is JSON-stringified (RemoteTool path)", () => {
    const { init } = buildTemplatedHttpRequest(urlencodedSpec, {});
    expect(init.body).toBe('{"a":"1"}');
  });

  it("loose isRecordLike: a class-instance body is form-encoded (ApiNode path)", () => {
    const { init } = buildTemplatedHttpRequest(urlencodedSpec, {}, {
      isRecord: isRecordLike,
    });
    expect(init.body).toBeInstanceOf(URLSearchParams);
    expect(String(init.body)).toBe("a=1");
  });

  it("GET empty-body check follows the guard for a keyless class instance", () => {
    const spec = {
      url: "https://api.example.com/plain",
      httpMethod: "GET",
      data: new (class {})(),
      headers: {},
      queryParams: {},
    };
    // Strict: not a record, so the instance counts as a declared body that
    // fetch cannot send on GET.
    expect(buildTemplatedHttpRequest(spec, {}).bodyDropped).toBe(true);
    // Loose: a keyless record counts as empty, so nothing is dropped.
    expect(
      buildTemplatedHttpRequest(spec, {}, { isRecord: isRecordLike })
        .bodyDropped,
    ).toBe(false);
  });
});

describe("buildTemplatedHttpRequest url allow list", () => {
  const spec = {
    url: "https://{{host}}/api/value",
    httpMethod: "GET",
    headers: {},
    queryParams: {},
    urlAllowList: ["https://allowed.example.com/api/"],
  };

  it("validates the rendered URL against the spec's allow list", () => {
    expect(() =>
      buildTemplatedHttpRequest(spec, { host: "blocked.example.com" }),
    ).toThrow("Requested URL is not in allowed list");
    expect(
      buildTemplatedHttpRequest(spec, { host: "allowed.example.com" }).url,
    ).toBe("https://allowed.example.com/api/value");
  });
});

function makeRetryPolicy(
  overrides: Parameters<typeof RetryPolicySchema.parse>[0] = {},
): RetryPolicy {
  return RetryPolicySchema.parse(overrides);
}

describe("getRetryAfterSeconds", () => {
  it("parses numeric values as seconds, capped at 30", () => {
    expect(getRetryAfterSeconds("5")).toBe(5);
    expect(getRetryAfterSeconds("0.5")).toBe(0.5);
    expect(getRetryAfterSeconds("45")).toBe(MAX_RETRY_AFTER_SECONDS);
  });

  it("parses HTTP-dates as the (never negative) seconds until then, capped at 30", () => {
    const nowMs = Date.parse("Wed, 21 Oct 2015 07:28:00 GMT");
    expect(
      getRetryAfterSeconds("Wed, 21 Oct 2015 07:28:10 GMT", nowMs),
    ).toBe(10);
    // A date in the past clamps to 0 rather than a negative wait.
    expect(
      getRetryAfterSeconds("Wed, 21 Oct 2015 07:27:00 GMT", nowMs),
    ).toBe(0);
    expect(
      getRetryAfterSeconds("Wed, 21 Oct 2015 07:38:00 GMT", nowMs),
    ).toBe(MAX_RETRY_AFTER_SECONDS);
  });

  it("returns null for absent or unparsable values", () => {
    expect(getRetryAfterSeconds(null)).toBeNull();
    expect(getRetryAfterSeconds("soon")).toBeNull();
  });

  // The malformed-header edges below deliberately diverge from Python — see
  // the getRetryAfterSeconds docstring and the adapter README.

  it("treats a negative numeric value (invalid per RFC 9110) as an absent header", () => {
    // Python returns -5 and lets time.sleep(-5) raise, failing the call.
    // Returning null here makes the caller fall back to the configured
    // backoff: honoring the value as a zero-second wait would let a hostile
    // server erase the operator's backoff and drive the retry loop at speed.
    expect(getRetryAfterSeconds("-5")).toBeNull();
    expect(getRetryAfterSeconds("-0.1")).toBeNull();
  });

  it("keeps a legitimate zero-second Retry-After", () => {
    expect(getRetryAfterSeconds("0")).toBe(0);
  });

  it("treats infinite numeric values as unparsable (Python's float() caps them at 30)", () => {
    expect(getRetryAfterSeconds("Infinity")).toBeNull();
    expect(getRetryAfterSeconds("-Infinity")).toBeNull();
    expect(getRetryAfterSeconds("inf")).toBeNull();
  });

  it("accepts ISO 8601 dates (Python's HTTP-date parser rejects them)", () => {
    const nowMs = Date.parse("2015-10-21T07:28:00Z");
    expect(getRetryAfterSeconds("2015-10-21T07:28:10Z", nowMs)).toBe(10);
  });
});

describe("computeWaitSeconds jitter modes", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("null jitter returns the bare exponential backoff, capped at maxRetryDelay", () => {
    const policy = makeRetryPolicy({
      initialRetryDelay: 1,
      backoffFactor: 2,
      maxRetryDelay: 8,
      jitter: null,
    });
    expect(computeWaitSeconds(policy, 0, null)).toBe(1);
    expect(computeWaitSeconds(policy, 1, null)).toBe(2);
    expect(computeWaitSeconds(policy, 2, null)).toBe(4);
    expect(computeWaitSeconds(policy, 3, null)).toBe(8);
    expect(computeWaitSeconds(policy, 10, null)).toBe(8);
  });

  it("equal jitter randomizes the upper half of the backoff", () => {
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const policy = makeRetryPolicy({ jitter: "equal", initialRetryDelay: 4 });
    // base = 4; equal jitter = base/2 + rand * base/2 = 2 + 0.5 * 2.
    expect(computeWaitSeconds(policy, 0, null)).toBe(3);
  });

  it("full jitter randomizes the whole backoff", () => {
    vi.spyOn(Math, "random").mockReturnValue(0.25);
    const policy = makeRetryPolicy({ jitter: "full", initialRetryDelay: 4 });
    expect(computeWaitSeconds(policy, 0, null)).toBe(1);
  });

  it("full_and_equal_for_throttle applies equal jitter to 4xx and full jitter otherwise", () => {
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const policy = makeRetryPolicy({ initialRetryDelay: 4 });
    // 429 (throttle): equal jitter — 2 + 0.5 * 2.
    expect(computeWaitSeconds(policy, 0, 429)).toBe(3);
    // 503 / transport error (no status): full jitter — 0.5 * 4.
    expect(computeWaitSeconds(policy, 0, 503)).toBe(2);
    expect(computeWaitSeconds(policy, 0, null)).toBe(2);
  });

  it("decorrelated jitter adds up to one second, capped at maxRetryDelay", () => {
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const policy = makeRetryPolicy({
      jitter: "decorrelated",
      initialRetryDelay: 4,
    });
    expect(computeWaitSeconds(policy, 0, null)).toBe(4.5);
    const capped = makeRetryPolicy({
      jitter: "decorrelated",
      initialRetryDelay: 8,
      maxRetryDelay: 8,
    });
    expect(computeWaitSeconds(capped, 0, null)).toBe(8);
  });
});

describe("isTlsOrCertError", () => {
  it("detects TLS failures on the error itself and through the cause chain", () => {
    expect(
      isTlsOrCertError(new Error("certificate verify failed: self signed")),
    ).toBe(true);
    // undici wraps the TLS failure in `TypeError: fetch failed` with the
    // real error on `cause` (Python walks `__cause__` the same way).
    expect(
      isTlsOrCertError(
        new TypeError("fetch failed", {
          cause: Object.assign(new Error("self-signed certificate"), {
            code: "DEPTH_ZERO_SELF_SIGNED_CERT",
          }),
        }),
      ),
    ).toBe(true);
    expect(
      isTlsOrCertError(
        new TypeError("fetch failed", {
          cause: new Error(
            "Hostname/IP does not match certificate's altnames",
          ),
        }),
      ),
    ).toBe(true);
  });

  it("does not flag ordinary transport errors", () => {
    expect(isTlsOrCertError(new TypeError("fetch failed"))).toBe(false);
    expect(
      isTlsOrCertError(
        new TypeError("fetch failed", {
          cause: Object.assign(new Error("connect ECONNREFUSED 127.0.0.1"), {
            code: "ECONNREFUSED",
          }),
        }),
      ),
    ).toBe(false);
  });
});

describe("requestWithRetry backoff and elapsed cap", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  function install503Fetch(): { calls: number[] } {
    const calls: number[] = [];
    globalThis.fetch = (async () => {
      calls.push(Date.now());
      return new Response('{"error": "busy"}', {
        status: 503,
        headers: { "Content-Type": "application/json" },
      });
    }) as typeof fetch;
    return { calls };
  }

  it("sleeps the exponential backoff between attempts (fake timers)", async () => {
    vi.useFakeTimers();
    const { calls } = install503Fetch();
    const policy = makeRetryPolicy({
      maxAttempts: 3,
      initialRetryDelay: 1,
      backoffFactor: 2,
      maxRetryDelay: 8,
      jitter: null,
    });

    const started = Date.now();
    const responsePromise = requestWithRetry(policy, "https://x/", {}, "T");
    await vi.advanceTimersByTimeAsync(1000 + 2000 + 4000);
    const response = await responsePromise;

    expect(response.status).toBe(503);
    // Attempts at t=0, +1s, +3s, +7s: backoff of 1s, 2s, 4s between them.
    expect(calls.map((timestampMs) => timestampMs - started)).toEqual([
      0, 1000, 3000, 7000,
    ]);
  });

  it("stops retrying once the 600s elapsed budget is exhausted", async () => {
    vi.useFakeTimers();
    const calls: number[] = [];
    globalThis.fetch = (async () => {
      calls.push(Date.now());
      // Simulate a slow failing service: by the time the response arrives,
      // the total elapsed budget is spent, so no further retry is scheduled
      // even though attempts remain.
      vi.setSystemTime(Date.now() + 601_000);
      return new Response('{"error": "busy"}', {
        status: 503,
        headers: { "Content-Type": "application/json" },
      });
    }) as typeof fetch;
    const policy = makeRetryPolicy({
      maxAttempts: 5,
      initialRetryDelay: 0,
      maxRetryDelay: 0,
    });

    const response = await requestWithRetry(policy, "https://x/", {}, "T");

    expect(response.status).toBe(503);
    expect(calls).toHaveLength(1);
  });
});

describe("retry engine bounds against untrusted specs", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("clamps a per-attempt timeout to a value a timer can express", () => {
    // AbortSignal.timeout throws a RangeError for these, which the retry loop
    // would otherwise treat as a transient transport failure.
    expect(clampRequestTimeoutMs(Infinity)).toBe(MAX_HTTP_REQUEST_TIMEOUT_MS);
    expect(clampRequestTimeoutMs(1e308)).toBe(MAX_HTTP_REQUEST_TIMEOUT_MS);
    // (2^31, 2^32) ms silently degrades to a 1ms timer in Node.
    expect(clampRequestTimeoutMs(3_000_000_000)).toBe(
      MAX_HTTP_REQUEST_TIMEOUT_MS,
    );
    expect(clampRequestTimeoutMs(0.4)).toBe(1);
    expect(clampRequestTimeoutMs(5000)).toBe(5000);
    expect(() => AbortSignal.timeout(clampRequestTimeoutMs(Infinity))).not.toThrow();
  });

  it("rejects a non-finite requestTimeout at the schema boundary", () => {
    expect(() => RetryPolicySchema.parse({ requestTimeout: Infinity })).toThrow();
    expect(() => RetryPolicySchema.parse({ requestTimeout: NaN })).toThrow();
    expect(RetryPolicySchema.parse({ requestTimeout: 30 }).requestTimeout).toBe(30);
  });

  it("does not retry an out-of-range timer error", () => {
    // A RangeError is never a transport failure; retrying it spins the loop
    // without producing any network traffic.
    expect(isNonRetryableLocalError(new RangeError("out of range"))).toBe(true);
    // A bare TypeError stays retryable: test doubles and non-undici fetch
    // implementations raise those for simulated network errors.
    expect(isNonRetryableLocalError(new TypeError("temporary failure"))).toBe(
      false,
    );
  });

  it("caps the number of attempts however large maxAttempts is", async () => {
    vi.useFakeTimers();
    let calls = 0;
    globalThis.fetch = vi.fn(async () => {
      calls += 1;
      return new Response("{}", { status: 503 });
    }) as typeof fetch;
    const policy = makeRetryPolicy({
      maxAttempts: 1_000_000_000,
      initialRetryDelay: 0,
      maxRetryDelay: 0,
      jitter: null,
      serviceErrorRetryOnAny5xx: true,
    });

    const pending = requestWithRetry(policy, "https://x/", {}, "T");
    await vi.runAllTimersAsync();
    await pending;

    expect(calls).toBe(MAX_HTTP_ATTEMPTS_PER_CALL);
  });

  it("floors the wait between attempts when the spec configures zero delays", async () => {
    vi.useFakeTimers();
    const waits: number[] = [];
    const setTimeoutSpy = vi
      .spyOn(globalThis, "setTimeout")
      .mockImplementation(((fn: () => void, ms?: number) => {
        waits.push(ms ?? 0);
        fn();
        return 0 as unknown as ReturnType<typeof setTimeout>;
      }) as typeof setTimeout);
    globalThis.fetch = vi.fn(
      async () => new Response("{}", { status: 503 }),
    ) as typeof fetch;
    const policy = makeRetryPolicy({
      maxAttempts: 3,
      initialRetryDelay: 0,
      maxRetryDelay: 0,
      jitter: null,
      serviceErrorRetryOnAny5xx: true,
    });

    await requestWithRetry(policy, "https://x/", {}, "T");

    expect(waits.length).toBeGreaterThan(0);
    for (const wait of waits) {
      expect(wait).toBeGreaterThanOrEqual(MIN_RETRY_DELAY_SECONDS * 1000);
    }
    setTimeoutSpy.mockRestore();
  });

  it("falls back to backoff when a server sends a negative Retry-After", async () => {
    vi.useFakeTimers();
    const waits: number[] = [];
    const setTimeoutSpy = vi
      .spyOn(globalThis, "setTimeout")
      .mockImplementation(((fn: () => void, ms?: number) => {
        waits.push(ms ?? 0);
        fn();
        return 0 as unknown as ReturnType<typeof setTimeout>;
      }) as typeof setTimeout);
    globalThis.fetch = vi.fn(
      async () =>
        new Response("{}", { status: 503, headers: { "Retry-After": "-1" } }),
    ) as typeof fetch;
    const policy = makeRetryPolicy({
      maxAttempts: 2,
      initialRetryDelay: 1,
      maxRetryDelay: 8,
      jitter: null,
      serviceErrorRetryOnAny5xx: true,
    });

    await requestWithRetry(policy, "https://x/", {}, "T");

    // The configured 1s backoff applies; the hostile header does not erase it.
    expect(waits[0]).toBeGreaterThanOrEqual(1000);
    setTimeoutSpy.mockRestore();
  });

  it("treats a TLS handshake failure as non-retryable, like a bad certificate", () => {
    const handshakeFailure = new TypeError("fetch failed", {
      cause: Object.assign(new Error("wrong version number"), {
        code: "ERR_SSL_WRONG_VERSION_NUMBER",
      }),
    });
    expect(isTlsOrCertError(handshakeFailure)).toBe(true);
  });
});

describe("error messages do not leak query secrets", () => {
  it("redacts the query string from a non-2xx tool error", () => {
    const policy = makeRetryPolicy({});
    const response = new Response("nope", { status: 401 });

    expect(() =>
      raiseForStatusWhenPolicySet(
        policy,
        response,
        "RemoteTool `leak`",
        "https://api.example.com/v1/x?api_key=SECRET&token=ALSO_SECRET",
      ),
    ).toThrow(/https:\/\/api\.example\.com\/v1\/x\?<redacted>/);

    try {
      raiseForStatusWhenPolicySet(
        policy,
        response,
        "RemoteTool `leak`",
        "https://api.example.com/v1/x?api_key=SECRET",
      );
    } catch (error) {
      expect((error as Error).message).not.toContain("SECRET");
    }
  });

  it("leaves a URL without a query string intact", () => {
    expect(redactUrlQuery("https://api.example.com/v1/x")).toBe(
      "https://api.example.com/v1/x",
    );
  });
});
