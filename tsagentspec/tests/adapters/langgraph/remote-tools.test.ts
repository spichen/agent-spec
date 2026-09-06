/**
 * RemoteTool / ClientTool conversion tests for the LangGraph adapter.
 *
 * Mirrors the RemoteTool sections of
 * `pyagentspec/tests/adapters/langgraph/test_tools.py` and the retry-policy
 * matrix of `pyagentspec/tests/adapters/test_remote_tool_retry_policy_cases.py`
 * with a mocked global fetch (JS equivalent of patching `httpx.request`):
 * template rendering in url/data/headers/queryParams, body routing
 * (urlencoded form vs raw string vs JSON), retry-policy behavior (per-tool
 * timeouts, transport-error and recoverable-status retries, Retry-After,
 * TLS no-retry, raise-for-status), URL allow-list enforcement, the
 * confirmation-interrupt machinery and the ClientTool interrupt protocol.
 *
 * Documented divergences exercised here (see tools.ts / tools-common.ts):
 * - fetch forbids GET/HEAD bodies, so none is sent for those methods;
 * - confirmation `Args:` strings use JSON.stringify (Python uses str(dict)).
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import type { StructuredToolInterface } from "@langchain/core/tools";
import {
  Annotation,
  Command,
  MemorySaver,
  START,
  StateGraph,
} from "@langchain/langgraph";
import {
  createClientTool,
  createRemoteTool,
  createServerTool,
  integerProperty,
  stringProperty,
  type JsonSchemaValue,
} from "../../../src/index.js";
import { DEFAULT_HTTP_REQUEST_TIMEOUT_MS } from "../../../src/adapters/common/tools-common.js";
import {
  confirmThen,
  convertClientTool,
  convertRemoteTool,
  ensureCheckpointerAndValidToolConfig,
} from "../../../src/adapters/langgraph/tools.js";
import {
  approveCommand,
  getInterrupts,
  installMockFetch,
  rejectCommand,
  threadConfig,
  type MockFetchController,
} from "./test-helpers.js";

let mockFetch: MockFetchController | undefined;

afterEach(() => {
  mockFetch?.restore();
  mockFetch = undefined;
  vi.restoreAllMocks();
  vi.useRealTimers();
});

function headersOf(init: RequestInit | undefined): Record<string, string> {
  return (init?.headers ?? {}) as Record<string, string>;
}

/** Compile a one-node graph that invokes the tool with the given arguments. */
function makeToolCallGraph(
  langchainTool: StructuredToolInterface,
  args: Record<string, unknown>,
) {
  const state = Annotation.Root({ result: Annotation<unknown>() });
  return new StateGraph(state)
    .addNode("call", async () => ({ result: await langchainTool.invoke(args) }))
    .addEdge(START, "call")
    .compile({ checkpointer: new MemorySaver() });
}

describe("convertRemoteTool template rendering", () => {
  it("renders nested data, url path, header keys and values", async () => {
    mockFetch = installMockFetch((url) => {
      const city = decodeURIComponent(url.split("/").pop() ?? "");
      return { weather: `sunny in ${city}` };
    });
    const remoteTool = createRemoteTool({
      name: "forecast_weather",
      description: "Returns a forecast of the weather for the chosen city",
      url: "https://weatherforecast.example/api/forecast/{{city}}",
      httpMethod: "POST",
      data: {
        location: {
          city: "{{city}}",
          coordinates: { lat: "{{lat}}", lon: "{{lon}}" },
        },
        meta: ["requested_by:{{user}}", { note: "hello{{suffix}}" }],
        raw: "binary-{{bin_suffix}}",
      },
      headers: { "X-{{header_key}}": "{{user}}" },
    });

    const langchainTool = convertRemoteTool(remoteTool);
    const result = await langchainTool.invoke({
      city: "Agadir",
      lat: "30.4",
      lon: "-9.6",
      user: "alice",
      suffix: "world",
      bin_suffix: "blob",
      header_key: "Caller",
    });

    expect(mockFetch.calls).toHaveLength(1);
    const call = mockFetch.calls[0]!;
    expect(call.url).toBe(
      "https://weatherforecast.example/api/forecast/Agadir",
    );
    expect(call.init?.method).toBe("POST");
    // Templated header keys and values are both rendered; JSON content type
    // is added automatically for object bodies.
    expect(headersOf(call.init)).toEqual({
      "X-Caller": "alice",
      "Content-Type": "application/json",
    });
    expect(JSON.parse(call.init?.body as string)).toEqual({
      location: {
        city: "Agadir",
        coordinates: { lat: "30.4", lon: "-9.6" },
      },
      meta: ["requested_by:alice", { note: "helloworld" }],
      raw: "binary-blob",
    });
    expect(result).toEqual({ weather: "sunny in Agadir" });
  });

  it("leaves unknown placeholders verbatim", async () => {
    mockFetch = installMockFetch(() => ({ ok: true }));
    const remoteTool = createRemoteTool({
      name: "partial",
      description: "d",
      url: "https://example.com/api/{{known}}/{{unknown}}",
      httpMethod: "POST",
      data: { note: "hello{{unknown}}" },
      inputs: [stringProperty({ title: "known" })],
    });

    await convertRemoteTool(remoteTool).invoke({ known: "k" });

    const call = mockFetch.calls[0]!;
    expect(call.url).toBe("https://example.com/api/k/{{unknown}}");
    expect(JSON.parse(call.init?.body as string)).toEqual({
      note: "hello{{unknown}}",
    });
  });
});

describe("convertRemoteTool body routing", () => {
  it("sends an urlencoded form body for object data with the urlencoded content type", async () => {
    mockFetch = installMockFetch(() => ({ ok: true }));
    const remoteTool = createRemoteTool({
      name: "form_tool",
      description: "d",
      url: "https://example.com/api/form",
      httpMethod: "POST",
      data: { value: "{{v1}}", listofvalues: ["a", "{{v2}}", "c"] },
      headers: {
        header1: "{{h1}}",
        "Content-Type": "application/x-www-form-urlencoded",
      },
    });

    await convertRemoteTool(remoteTool).invoke({
      v1: "test1",
      v2: "test2",
      h1: "test4",
    });

    const call = mockFetch.calls[0]!;
    expect(headersOf(call.init)).toEqual({
      header1: "test4",
      "Content-Type": "application/x-www-form-urlencoded",
    });
    const body = call.init?.body as URLSearchParams;
    expect(body).toBeInstanceOf(URLSearchParams);
    expect(body.get("value")).toBe("test1");
    // Non-string form values are JSON stringified.
    expect(body.get("listofvalues")).toBe('["a","test2","c"]');
  });

  it("sends a raw string body verbatim after rendering", async () => {
    mockFetch = installMockFetch(() => ({ ok: true }));
    const remoteTool = createRemoteTool({
      name: "send_raw",
      description: "Sends a raw string body",
      url: "https://example.com/api/raw",
      httpMethod: "POST",
      data: "request body for city: {{city}} with note: {{note}}",
    });

    await convertRemoteTool(remoteTool).invoke({
      city: "Agadir",
      note: "urgent",
    });

    const call = mockFetch.calls[0]!;
    expect(call.init?.body).toBe(
      "request body for city: Agadir with note: urgent",
    );
    // No JSON content type for raw bodies.
    expect(headersOf(call.init)).toEqual({});
  });

  it("sends a JSON body for array data", async () => {
    mockFetch = installMockFetch(() => ({ ok: true }));
    const remoteTool = createRemoteTool({
      name: "process_array",
      description: "Processes a JSON array body",
      url: "https://example.com/api/process",
      httpMethod: "POST",
      data: ["forecast", { location: "{{city}}", temp: "{{temp}}" }],
    });

    await convertRemoteTool(remoteTool).invoke({ city: "Agadir", temp: "25" });

    const call = mockFetch.calls[0]!;
    expect(headersOf(call.init)["Content-Type"]).toBe("application/json");
    expect(JSON.parse(call.init?.body as string)).toEqual([
      "forecast",
      { location: "Agadir", temp: "25" },
    ]);
  });

  it("does not send a body on GET and appends query parameters", async () => {
    mockFetch = installMockFetch(() => ({ ok: true }));
    const remoteTool = createRemoteTool({
      name: "get_tool",
      description: "d",
      url: "https://example.com/api/echo/{{u1}}",
      httpMethod: "GET",
      data: { ignored: "{{u1}}" },
      queryParams: { param: "{{p1}}" },
    });

    await convertRemoteTool(remoteTool).invoke({ u1: "u_seg", p1: "test3" });

    const call = mockFetch.calls[0]!;
    expect(call.url).toBe("https://example.com/api/echo/u_seg?param=test3");
    expect(call.init?.method).toBe("GET");
    expect(call.init?.body).toBeUndefined();
  });

  it("appends query params with & when the url already has a query, repeating array values", async () => {
    mockFetch = installMockFetch(() => ({ ok: true }));
    const remoteTool = createRemoteTool({
      name: "query_tool",
      description: "d",
      url: "https://example.com/api?x=1",
      httpMethod: "GET",
      queryParams: { tags: ["a", "{{t}}"], n: 3 },
    });

    await convertRemoteTool(remoteTool).invoke({ t: "b" });

    expect(mockFetch.calls[0]!.url).toBe(
      "https://example.com/api?x=1&tags=a&tags=b&n=3",
    );
  });

  it("stringifies non-string header values", async () => {
    mockFetch = installMockFetch(() => ({ ok: true }));
    const remoteTool = createRemoteTool({
      name: "num_header",
      description: "d",
      url: "https://example.com/api",
      httpMethod: "GET",
      headers: { "X-Num": 42 },
    });

    await convertRemoteTool(remoteTool).invoke({});

    expect(headersOf(mockFetch.calls[0]!.init)["X-Num"]).toBe("42");
  });
});

describe("convertRemoteTool responses", () => {
  it("returns the parsed JSON response body", async () => {
    mockFetch = installMockFetch(() => ({ processed_city: "Agadir" }));
    const remoteTool = createRemoteTool({
      name: "echo",
      description: "d",
      url: "https://example.com/api",
      httpMethod: "POST",
      data: { x: "{{x}}" },
    });

    await expect(
      convertRemoteTool(remoteTool).invoke({ x: "1" }),
    ).resolves.toEqual({ processed_city: "Agadir" });
  });

  it("parses and returns the JSON body of non-2xx responses like Python", async () => {
    // Python without a retry policy returns response.json() for every
    // status, so the agent sees error payloads as the tool result instead of
    // an aborted run (with a policy, a final error status raises — see the
    // retry-policy suite below).
    mockFetch = installMockFetch(
      () =>
        new Response('{"error": "bad date range"}', {
          status: 422,
          headers: { "Content-Type": "application/json" },
        }),
    );
    const remoteTool = createRemoteTool({
      name: "failing",
      description: "d",
      url: "https://example.com/api",
      httpMethod: "GET",
    });

    await expect(convertRemoteTool(remoteTool).invoke({})).resolves.toEqual({
      error: "bad date range",
    });
  });

  it("does not follow redirects and parses a 3xx body like any other status", async () => {
    // Python's httpx does not follow redirects (follow_redirects defaults to
    // False): a 3xx comes back as the response instead of triggering a second
    // request to the Location target (redirect-based egress / auth-header
    // forwarding on untrusted spec config). undici's redirect: "manual"
    // returns the 3xx response with its body intact, which the no-retry path
    // then parses like any other status.
    mockFetch = installMockFetch(
      () =>
        new Response('{"error": "moved"}', {
          status: 302,
          headers: {
            "Content-Type": "application/json",
            Location: "https://attacker.example/exfil",
          },
        }),
    );
    const remoteTool = createRemoteTool({
      name: "redirecting",
      description: "d",
      url: "https://example.com/api",
      httpMethod: "GET",
    });

    await expect(convertRemoteTool(remoteTool).invoke({})).resolves.toEqual({
      error: "moved",
    });
    expect(mockFetch.calls).toHaveLength(1);
    expect(mockFetch.calls[0]!.init?.redirect).toBe("manual");
  });

  it("attaches the default httpx-parity timeout and names the tool on a timeout abort", async () => {
    // Python's httpx applies a 5s default timeout; without a
    // RetryPolicy.requestTimeout override the exported constant is the knob
    // and a timeout abort maps to an Error naming the tool.
    expect(DEFAULT_HTTP_REQUEST_TIMEOUT_MS).toBe(5000);
    mockFetch = installMockFetch(() => {
      throw new DOMException(
        "The operation was aborted due to timeout",
        "TimeoutError",
      );
    });
    const remoteTool = createRemoteTool({
      name: "slow",
      description: "d",
      url: "https://example.com/api",
      httpMethod: "GET",
    });

    await expect(convertRemoteTool(remoteTool).invoke({})).rejects.toThrow(
      `RemoteTool \`slow\` HTTP request timed out after ${DEFAULT_HTTP_REQUEST_TIMEOUT_MS}ms.`,
    );
    expect(mockFetch.calls).toHaveLength(1);
    expect(mockFetch.calls[0]!.init?.signal).toBeInstanceOf(AbortSignal);
  });

  it("injects declared input defaults into the rendered request like Python", async () => {
    // Python's pydantic args model fills Property defaults before the tool
    // func renders the URL; langchain JS applies no JSON-schema defaults, so
    // the adapter injects them itself.
    mockFetch = installMockFetch(() => ({ ok: true }));
    const remoteTool = createRemoteTool({
      name: "search",
      description: "d",
      url: "https://api.example.com/search?q={{query}}&limit={{limit}}",
      httpMethod: "GET",
      inputs: [
        stringProperty({ title: "query" }),
        integerProperty({ title: "limit", default: 10 }),
      ],
    });

    await convertRemoteTool(remoteTool).invoke({ query: "abc" });

    expect(mockFetch.calls[0]!.url).toBe(
      "https://api.example.com/search?q=abc&limit=10",
    );
  });

  it("validates the call arguments against the inferred schema", async () => {
    mockFetch = installMockFetch(() => ({ ok: true }));
    const remoteTool = createRemoteTool({
      name: "strict",
      description: "d",
      url: "https://example.com/api/{{city}}",
      httpMethod: "GET",
    });

    await expect(convertRemoteTool(remoteTool).invoke({})).rejects.toThrow(
      "Received tool input did not match expected schema",
    );
    expect(mockFetch.calls).toHaveLength(0);
  });
});

describe("convertRemoteTool retry policy", () => {
  // Ports pyagentspec/tests/adapters/test_remote_tool_retry_policy_cases.py
  // through the LangGraph wrapper (Python's adapters subclass the shared
  // case class the same way).
  function makeRetryTool(
    retryPolicy?: Parameters<typeof createRemoteTool>[0]["retryPolicy"],
  ) {
    return createRemoteTool({
      name: "retry_service",
      description: "A remote service with retry policy",
      url: "https://example.com/api",
      httpMethod: "GET",
      ...(retryPolicy !== undefined ? { retryPolicy } : {}),
    });
  }

  function jsonResponse(
    body: unknown,
    status: number,
    headers: Record<string, string> = {},
  ): Response {
    return new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json", ...headers },
    });
  }

  it.each([
    ["a policy with requestTimeout", { maxAttempts: 0, requestTimeout: 300 }, 300_000],
    ["a policy without requestTimeout", { maxAttempts: 0 }, 5000],
    ["no policy", undefined, 5000],
  ] as const)(
    "passes the per-tool request timeout for %s",
    async (_label, retryPolicy, expectedTimeoutMs) => {
      const timeoutSpy = vi.spyOn(AbortSignal, "timeout");
      mockFetch = installMockFetch(() => ({ result: "ok" }));

      await expect(
        convertRemoteTool(makeRetryTool(retryPolicy)).invoke({}),
      ).resolves.toEqual({ result: "ok" });

      expect(timeoutSpy).toHaveBeenCalledTimes(1);
      expect(timeoutSpy).toHaveBeenCalledWith(expectedTimeoutMs);
    },
  );

  it("retries transport errors and applies the timeout override to every attempt", async () => {
    const timeoutSpy = vi.spyOn(AbortSignal, "timeout");
    let call = 0;
    mockFetch = installMockFetch(() => {
      call += 1;
      if (call <= 2) {
        throw new TypeError("temporary failure");
      }
      return { result: "ok" };
    });
    const remoteTool = makeRetryTool({
      maxAttempts: 2,
      requestTimeout: 0.5,
      initialRetryDelay: 0,
      maxRetryDelay: 0,
    });

    await expect(convertRemoteTool(remoteTool).invoke({})).resolves.toEqual({
      result: "ok",
    });
    expect(mockFetch.calls).toHaveLength(3);
    expect(timeoutSpy.mock.calls.map(([ms]) => ms)).toEqual([500, 500, 500]);
  });

  it("retries a 5xx service error under the default recoverable rules", async () => {
    let call = 0;
    mockFetch = installMockFetch(() => {
      call += 1;
      return call === 1
        ? jsonResponse({ error: "busy" }, 503)
        : { result: "ok" };
    });
    const remoteTool = makeRetryTool({
      maxAttempts: 1,
      initialRetryDelay: 0,
      maxRetryDelay: 0,
    });

    await expect(convertRemoteTool(remoteTool).invoke({})).resolves.toEqual({
      result: "ok",
    });
    expect(mockFetch.calls).toHaveLength(2);
  });

  it("retries a configured status only when the body carries a recoverable error code", async () => {
    let call = 0;
    mockFetch = installMockFetch(() => {
      call += 1;
      return call === 1
        ? jsonResponse({ code: "TooManyRequests", message: "throttled" }, 429)
        : { result: "ok" };
    });
    const remoteTool = makeRetryTool({
      maxAttempts: 1,
      initialRetryDelay: 0,
      maxRetryDelay: 0,
      serviceErrorRetryOnAny5xx: false,
      recoverableStatuses: { "429": ["TooManyRequests"] },
    });

    await expect(convertRemoteTool(remoteTool).invoke({})).resolves.toEqual({
      result: "ok",
    });
    expect(mockFetch.calls).toHaveLength(2);
  });

  it("honors Retry-After delays, capped at 30 seconds", async () => {
    vi.useFakeTimers();
    let call = 0;
    mockFetch = installMockFetch(() => {
      call += 1;
      return call === 1
        ? jsonResponse({ error: "throttled" }, 429, { "Retry-After": "45" })
        : { result: "ok" };
    });
    const remoteTool = makeRetryTool({
      maxAttempts: 1,
      initialRetryDelay: 0,
      maxRetryDelay: 0,
      recoverableStatuses: { "429": [] },
    });

    let settled = false;
    const resultPromise = convertRemoteTool(remoteTool)
      .invoke({})
      .finally(() => {
        settled = true;
      });
    // The 45s Retry-After is capped at 30s: one millisecond earlier the
    // retry has not fired yet.
    await vi.advanceTimersByTimeAsync(29_999);
    expect(settled).toBe(false);
    expect(mockFetch.calls).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    await expect(resultPromise).resolves.toEqual({ result: "ok" });
    expect(mockFetch.calls).toHaveLength(2);
  });

  it("raises for the final status after recoverable retries are exhausted", async () => {
    let call = 0;
    mockFetch = installMockFetch(() => {
      call += 1;
      return call === 1
        ? jsonResponse({ error: "busy" }, 503)
        : jsonResponse({ error: "still busy" }, 503);
    });
    const remoteTool = makeRetryTool({
      maxAttempts: 1,
      initialRetryDelay: 0,
      maxRetryDelay: 0,
    });

    await expect(convertRemoteTool(remoteTool).invoke({})).rejects.toThrow(
      "RemoteTool `retry_service` HTTP request failed with status '503' " +
        "for url 'https://example.com/api'.",
    );
    expect(mockFetch.calls).toHaveLength(2);
  });

  it("raises immediately for a non-retryable status when a policy is set", async () => {
    // 400/401/403/422 (and 501) are never retried, and with a policy the
    // error status raises instead of flowing back as the tool result.
    mockFetch = installMockFetch(() => jsonResponse({ error: "bad" }, 400));
    const remoteTool = makeRetryTool({
      maxAttempts: 2,
      initialRetryDelay: 0,
      maxRetryDelay: 0,
    });

    await expect(convertRemoteTool(remoteTool).invoke({})).rejects.toThrow(
      /400/,
    );
    expect(mockFetch.calls).toHaveLength(1);
  });

  it("does not retry TLS certificate failures", async () => {
    // undici wraps the TLS failure in `TypeError: fetch failed` with the
    // real error on `cause` (Python walks `__cause__` the same way).
    mockFetch = installMockFetch(() => {
      throw new TypeError("fetch failed", {
        cause: Object.assign(
          new Error("unable to verify the first certificate"),
          { code: "UNABLE_TO_VERIFY_LEAF_SIGNATURE" },
        ),
      });
    });
    const remoteTool = makeRetryTool({
      maxAttempts: 2,
      initialRetryDelay: 0,
      maxRetryDelay: 0,
    });

    await expect(convertRemoteTool(remoteTool).invoke({})).rejects.toThrow(
      "fetch failed",
    );
    expect(mockFetch.calls).toHaveLength(1);
  });

  it("rethrows the transport error once retries are exhausted", async () => {
    mockFetch = installMockFetch(() => {
      throw new TypeError("temporary failure");
    });
    const remoteTool = makeRetryTool({
      maxAttempts: 1,
      initialRetryDelay: 0,
      maxRetryDelay: 0,
    });

    await expect(convertRemoteTool(remoteTool).invoke({})).rejects.toThrow(
      "temporary failure",
    );
    expect(mockFetch.calls).toHaveLength(2);
  });
});

describe("convertRemoteTool url allow list", () => {
  // Ports the allow-list tests of
  // pyagentspec/tests/adapters/langgraph/test_tools.py.
  function makeAllowListTool(urlAllowList?: string[]) {
    return createRemoteTool({
      name: "lookup",
      description: "Looks up remote data",
      url: "https://{{host}}/api/value",
      httpMethod: "GET",
      inputs: [stringProperty({ title: "host" })],
      ...(urlAllowList !== undefined ? { urlAllowList } : {}),
    });
  }

  it("rejects a rendered URL outside the allow list without calling fetch", async () => {
    mockFetch = installMockFetch(() => ({ ok: true }));
    const langchainTool = convertRemoteTool(
      makeAllowListTool(["https://allowed.example.com/api/"]),
    );

    await expect(
      langchainTool.invoke({ host: "blocked.example.com" }),
    ).rejects.toThrow(
      "Requested URL is not in allowed list. Please contact the application " +
        "administrator to help adding your URL to the list.",
    );
    expect(mockFetch.calls).toHaveLength(0);
  });

  it("allows a rendered URL matching an allow-list entry", async () => {
    mockFetch = installMockFetch(() => ({ ok: true }));
    const langchainTool = convertRemoteTool(
      makeAllowListTool(["https://allowed.example.com/api/"]),
    );

    await expect(
      langchainTool.invoke({ host: "allowed.example.com" }),
    ).resolves.toEqual({ ok: true });
    expect(mockFetch.calls[0]!.url).toBe("https://allowed.example.com/api/value");
  });

  it("suppresses the templated-destination warning when an allow list is configured", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    convertRemoteTool(makeAllowListTool(["https://allowed.example.com/api/"]));
    expect(warnSpy).not.toHaveBeenCalled();

    convertRemoteTool(makeAllowListTool());
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining(
        "RemoteTool `lookup` uses placeholders in the URL destination",
      ),
    );
  });
});

describe("requiresConfirmation interrupt machinery", () => {
  function makeConfirmedRemoteTool() {
    return createRemoteTool({
      name: "remote_echo",
      description: "Echo",
      url: "https://example.com/echo",
      httpMethod: "POST",
      data: { x: "{{x}}" },
      inputs: [integerProperty({ title: "x" })],
      requiresConfirmation: true,
    });
  }

  it("interrupts with the exact confirmation payload and executes on approve", async () => {
    mockFetch = installMockFetch((_url, init) => ({
      ok: true,
      body: JSON.parse(init?.body as string) as unknown,
    }));
    const graph = makeToolCallGraph(
      convertRemoteTool(makeConfirmedRemoteTool()),
      { x: 3 },
    );
    const config = threadConfig("rt1");

    const first = await graph.invoke({}, config);
    const interrupts = getInterrupts(first);
    expect(interrupts).toHaveLength(1);
    expect(interrupts[0]!.value).toEqual({
      action_requests: [
        {
          name: "remote_echo",
          arguments: { x: 3 },
          description:
            'Tool execution pending approval\n\nTool: remote_echo\nArgs: {"x":3}',
        },
      ],
      review_configs: [
        {
          action_name: "remote_echo",
          allowed_decisions: ["approve", "reject"],
          description:
            'Please resume with {"decisions": [{"type": "approve"}]}  # or "reject" ' +
            'with an optional "reason" for rejected tool calls.',
        },
      ],
    });
    // The HTTP request must not run before approval.
    expect(mockFetch.calls).toHaveLength(0);

    const resumed = await graph.invoke(approveCommand(), config);
    expect(mockFetch.calls).toHaveLength(1);
    expect(resumed["result"]).toEqual({ ok: true, body: { x: "3" } });
  });

  it("throws the denial error on reject and never calls fetch", async () => {
    mockFetch = installMockFetch(() => ({ ok: true }));
    const graph = makeToolCallGraph(
      convertRemoteTool(makeConfirmedRemoteTool()),
      { x: 3 },
    );
    const config = threadConfig("rt2");

    await graph.invoke({}, config);
    await expect(graph.invoke(rejectCommand("no"), config)).rejects.toThrow(
      "Tool 'remote_echo' was denied by the user (reason: no).",
    );
    expect(mockFetch.calls).toHaveLength(0);
  });

  it("uses a default reason when the rejection has none", async () => {
    mockFetch = installMockFetch(() => ({ ok: true }));
    const graph = makeToolCallGraph(
      convertRemoteTool(makeConfirmedRemoteTool()),
      { x: 3 },
    );
    const config = threadConfig("rt3");

    await graph.invoke({}, config);
    await expect(graph.invoke(rejectCommand(), config)).rejects.toThrow(
      "Tool 'remote_echo' was denied by the user (reason: No reason was provided.).",
    );
  });

  it.each([
    [
      "a non-dict resume value",
      "nope",
      "Tool confirmation result for tool remote_echo is not valid, should be " +
        `a dict with a 'decisions' key, was "nope" of type string.`,
    ],
    [
      "an empty decisions list",
      { decisions: [] },
      "Tool confirmation result for tool remote_echo is not valid, decisions " +
        "should be of length 1, was of length 0",
    ],
    [
      "two decisions",
      { decisions: [{ type: "approve" }, { type: "approve" }] },
      "Tool confirmation result for tool remote_echo is not valid, decisions " +
        "should be of length 1, was of length 2",
    ],
    [
      "an unknown decision type",
      { decisions: [{ type: "maybe" }] },
      "Tool confirmation result for tool remote_echo is not valid, decision " +
        `should be in ['approve', 'reject'], was {"type":"maybe"}.`,
    ],
  ])("raises the Python validation error for %s", async (_label, resume, message) => {
    mockFetch = installMockFetch(() => ({ ok: true }));
    const graph = makeToolCallGraph(
      convertRemoteTool(makeConfirmedRemoteTool()),
      { x: 3 },
    );
    const config = threadConfig(`rt-${_label}`);

    await graph.invoke({}, config);
    await expect(
      graph.invoke(new Command({ resume }), config),
    ).rejects.toThrow(message);
    expect(mockFetch.calls).toHaveLength(0);
  });

  it("confirmThen returns the function unchanged without requiresConfirmation", () => {
    const func = (input: unknown): unknown => input;
    expect(confirmThen(func, "t", false)).toBe(func);
    expect(confirmThen(func, "t", true)).not.toBe(func);
  });
});

describe("ensureCheckpointerAndValidToolConfig", () => {
  it("requires a checkpointer for tools with requiresConfirmation", () => {
    const serverTool = createServerTool({
      name: "double_tool",
      description: "Doubles input",
      inputs: [integerProperty({ title: "x" })],
      requiresConfirmation: true,
    });
    expect(() =>
      ensureCheckpointerAndValidToolConfig(serverTool, undefined),
    ).toThrow(
      "A Checkpointer is required for tool 'double_tool' because requires_confirmation=True",
    );
    expect(() =>
      ensureCheckpointerAndValidToolConfig(serverTool, new MemorySaver()),
    ).not.toThrow();
  });

  it("requires a checkpointer for every ClientTool", () => {
    const clientTool = createClientTool({
      name: "client_double",
      description: "Client doubles the number",
      inputs: [integerProperty({ title: "x" })],
    });
    expect(() =>
      ensureCheckpointerAndValidToolConfig(clientTool, undefined),
    ).toThrow("A Checkpointer is required when using ClientTool 'client_double'.");
    expect(() =>
      ensureCheckpointerAndValidToolConfig(clientTool, new MemorySaver()),
    ).not.toThrow();
  });

  it("accepts confirmation-free server tools without a checkpointer", () => {
    const serverTool = createServerTool({
      name: "double_tool",
      description: "Doubles input",
      inputs: [integerProperty({ title: "x" })],
    });
    expect(() =>
      ensureCheckpointerAndValidToolConfig(serverTool, undefined),
    ).not.toThrow();
  });
});

describe("convertClientTool interrupt protocol", () => {
  it("builds the args schema from the AgentSpec inputs", () => {
    const clientTool = createClientTool({
      name: "client_double",
      description: "Client doubles the number",
      inputs: [integerProperty({ title: "x" })],
    });
    const langchainTool = convertClientTool(clientTool);
    expect(langchainTool.name).toBe("client_double");
    expect(langchainTool.description).toBe("Client doubles the number");
    expect(langchainTool.schema as JsonSchemaValue).toEqual({
      title: "client_doubleArgs",
      type: "object",
      properties: { x: { title: "x", type: "integer" } },
      required: ["x"],
    });
  });

  it("interrupts with the client_tool_request payload and returns the resume value", async () => {
    const clientTool = createClientTool({
      name: "client_double",
      description: "Client doubles the number",
      inputs: [integerProperty({ title: "x" })],
    });
    const graph = makeToolCallGraph(convertClientTool(clientTool), { x: 7 });
    const config = threadConfig("ct1");

    const first = await graph.invoke({}, config);
    const interrupts = getInterrupts(first);
    expect(interrupts).toHaveLength(1);
    expect(interrupts[0]!.value).toEqual({
      type: "client_tool_request",
      name: "client_double",
      description: "Client doubles the number",
      inputs: { args: [], kwargs: { x: 7 } },
    });

    const resumed = await graph.invoke(new Command({ resume: 14 }), config);
    expect(resumed["result"]).toBe(14);
  });

  it("includes declared input defaults in the client_tool_request kwargs", async () => {
    // Python's pydantic validation injects defaults before the interrupt
    // payload is built, so the client sees the defaulted arguments too.
    const clientTool = createClientTool({
      name: "client_double",
      description: "Client doubles the number",
      inputs: [
        integerProperty({ title: "x" }),
        integerProperty({ title: "factor", default: 2 }),
      ],
    });
    const graph = makeToolCallGraph(convertClientTool(clientTool), { x: 7 });
    const config = threadConfig("ct-defaults");

    const first = await graph.invoke({}, config);
    const interrupts = getInterrupts(first);
    expect(interrupts).toHaveLength(1);
    expect(interrupts[0]!.value).toEqual({
      type: "client_tool_request",
      name: "client_double",
      description: "Client doubles the number",
      inputs: { args: [], kwargs: { x: 7, factor: 2 } },
    });
  });

  it("confirms first, then interrupts for client execution (two interrupts)", async () => {
    const clientTool = createClientTool({
      name: "client_double",
      description: "Client doubles the number",
      inputs: [integerProperty({ title: "x" })],
      requiresConfirmation: true,
    });
    const graph = makeToolCallGraph(convertClientTool(clientTool), { x: 7 });
    const config = threadConfig("ct2");

    // 1. confirmation interrupt
    const first = await graph.invoke({}, config);
    const confirmPayload = getInterrupts(first)[0]!.value as Record<
      string,
      unknown
    >;
    const actionRequests = confirmPayload["action_requests"] as Array<
      Record<string, unknown>
    >;
    expect(actionRequests[0]!["name"]).toBe("client_double");
    expect(actionRequests[0]!["arguments"]).toEqual({ x: 7 });

    // 2. approve -> client_tool_request interrupt
    const second = await graph.invoke(approveCommand(), config);
    const clientRequest = getInterrupts(second)[0]!.value as Record<
      string,
      unknown
    >;
    expect(clientRequest["type"]).toBe("client_tool_request");
    expect(clientRequest["name"]).toBe("client_double");
    expect(clientRequest["inputs"]).toEqual({ args: [], kwargs: { x: 7 } });

    // 3. resume with the client-side result
    const resumed = await graph.invoke(new Command({ resume: 14 }), config);
    expect(resumed["result"]).toBe(14);
  });

  it("rejecting the confirmation raises and never requests client execution", async () => {
    const clientTool = createClientTool({
      name: "client_double",
      description: "Client doubles the number",
      inputs: [integerProperty({ title: "x" })],
      requiresConfirmation: true,
    });
    const graph = makeToolCallGraph(convertClientTool(clientTool), { x: 7 });
    const config = threadConfig("ct3");

    await graph.invoke({}, config);
    await expect(graph.invoke(rejectCommand("no"), config)).rejects.toThrow(
      "Tool 'client_double' was denied by the user (reason: no).",
    );
  });
});
