/**
 * ApiNode flow execution tests for the LangGraph adapter.
 *
 * Mirrors `pyagentspec/tests/adapters/langgraph/flows/test_apinode.py`
 * (allow-list enforcement included) with a mocked fetch so every test runs
 * offline.
 *
 * Documented divergence exercised here: the node's `retryPolicy` drives the
 * shared retry engine and raises for a final error status (Python's
 * ApiNodeExecutor performs a single plain request, keeping the field
 * representation-only).
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  createApiNode,
  createFlow,
  stringProperty,
  type ComponentWithIO,
  type Flow,
  type Property,
} from "../../../../src/index.js";
import { DEFAULT_HTTP_REQUEST_TIMEOUT_MS } from "../../../../src/adapters/common/tools-common.js";
import {
  ctrl,
  dataEdge,
  installMockFetch,
  ioEndNode,
  ioStartNode,
  loadFlow,
  outputsOf,
  type MockFetchController,
} from "../test-helpers.js";

describe("ApiNode", () => {
  let mockFetch: MockFetchController | undefined;

  afterEach(() => {
    mockFetch?.restore();
    mockFetch = undefined;
    vi.restoreAllMocks();
  });

  function buildApiFlow(
    apiNode: Record<string, unknown>,
    inputProps: Property[],
    outputProps: Property[],
  ): Flow {
    const start = ioStartNode("start", inputProps);
    const end = ioEndNode("end", outputProps);
    return createFlow({
      name: "api_flow",
      startNode: start,
      nodes: [start, apiNode, end],
      controlFlowConnections: [ctrl(start, apiNode), ctrl(apiNode, end)],
      dataFlowConnections: [
        ...inputProps.map((prop) =>
          dataEdge(start, apiNode as unknown as ComponentWithIO, prop.title),
        ),
        ...outputProps.map((prop) =>
          dataEdge(apiNode as unknown as ComponentWithIO, end, prop.title),
        ),
      ],
      inputs: inputProps,
      outputs: outputProps,
    });
  }

  it("GET: templates the URL, query params and headers, and maps the JSON response", async () => {
    // Templated URL destination without an allow list warns per Python rules.
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const inputProps = [
      stringProperty({ title: "host" }),
      stringProperty({ title: "order_id" }),
      stringProperty({ title: "flag" }),
      stringProperty({ title: "token" }),
    ];
    const status = stringProperty({ title: "status" });
    const apiNode = createApiNode({
      name: "api",
      url: "https://{{host}}/orders/{{order_id}}",
      httpMethod: "GET",
      queryParams: { verbose: "{{flag}}" },
      headers: { "X-Auth": "Bearer {{token}}" },
      inputs: inputProps,
      outputs: [status],
    });
    const flow = buildApiFlow(apiNode, inputProps, [status]);
    const graph = await loadFlow(flow);
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining("ApiNode `api` uses placeholders in the URL destination"),
    );

    mockFetch = installMockFetch(() => ({ status: "ok" }));
    const result = await graph.invoke({
      inputs: {
        host: "allowed.example.com",
        order_id: "123",
        flag: "yes",
        token: "tok-1",
      },
    });

    expect(outputsOf(result)).toEqual({ status: "ok" });
    expect(mockFetch.calls).toHaveLength(1);
    expect(mockFetch.calls[0]!.url).toBe(
      "https://allowed.example.com/orders/123?verbose=yes",
    );
    const init = mockFetch.calls[0]!.init!;
    expect(init.method).toBe("GET");
    expect((init.headers as Record<string, string>)["X-Auth"]).toBe(
      "Bearer tok-1",
    );
    expect(init.body).toBeUndefined();
  });

  it("GET: warns when declared request data is dropped (fetch forbids GET bodies)", async () => {
    // Python's httpx sends the body on GET; fetch cannot, so the adapter
    // must at least warn instead of silently discarding the declared data.
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const inputProps = [stringProperty({ title: "term" })];
    const echo = stringProperty({ title: "echo" });
    const apiNode = createApiNode({
      name: "api",
      url: "https://api.example.com/search",
      httpMethod: "GET",
      data: { q: "{{term}}" },
      inputs: inputProps,
      outputs: [echo],
    });
    const flow = buildApiFlow(apiNode, inputProps, [echo]);
    const graph = await loadFlow(flow);

    mockFetch = installMockFetch(() => ({ echo: "done" }));
    await graph.invoke({ inputs: { term: "boots" } });

    expect(mockFetch.calls[0]!.init!.body).toBeUndefined();
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining(
        "ApiNode `api` declares request data for HTTP method GET",
      ),
    );
  });

  it("GET: does not warn about a dropped body for the default empty data", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const echo = stringProperty({ title: "echo" });
    const apiNode = createApiNode({
      name: "api",
      url: "https://api.example.com/plain",
      httpMethod: "GET",
      outputs: [echo],
    });
    const flow = buildApiFlow(apiNode, [], [echo]);
    const graph = await loadFlow(flow);

    mockFetch = installMockFetch(() => ({ echo: "done" }));
    await graph.invoke({ inputs: {} });

    expect(warnSpy).not.toHaveBeenCalledWith(
      expect.stringContaining("declares request data"),
    );
  });

  it("POST: templated dict data is sent as a JSON body with a JSON content type", async () => {
    const inputProps = [
      stringProperty({ title: "order_id" }),
      stringProperty({ title: "tag" }),
    ];
    const echo = stringProperty({ title: "echo" });
    const apiNode = createApiNode({
      name: "api",
      url: "https://api.example.com/orders",
      httpMethod: "POST",
      data: { order: { id: "{{order_id}}" }, tags: ["{{tag}}", "static"] },
      inputs: inputProps,
      outputs: [echo],
    });
    const flow = buildApiFlow(apiNode, inputProps, [echo]);
    const graph = await loadFlow(flow);

    mockFetch = installMockFetch(() => ({ echo: "done" }));
    const result = await graph.invoke({
      inputs: { order_id: "777", tag: "blue" },
    });

    expect(outputsOf(result)).toEqual({ echo: "done" });
    const init = mockFetch.calls[0]!.init!;
    expect(init.method).toBe("POST");
    expect(
      (init.headers as Record<string, string>)["Content-Type"],
    ).toBe("application/json");
    expect(JSON.parse(String(init.body))).toEqual({
      order: { id: "777" },
      tags: ["blue", "static"],
    });
  });

  it("POST: an urlencoded content type sends dict data as a form body and templates header keys", async () => {
    const inputProps = [
      stringProperty({ title: "a" }),
      stringProperty({ title: "key_name" }),
      stringProperty({ title: "key_val" }),
    ];
    const echo = stringProperty({ title: "echo" });
    const apiNode = createApiNode({
      name: "api",
      url: "https://api.example.com/form",
      httpMethod: "POST",
      data: { a: "{{a}}", b: "static" },
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        "X-{{key_name}}": "{{key_val}}",
      },
      inputs: inputProps,
      outputs: [echo],
    });
    const flow = buildApiFlow(apiNode, inputProps, [echo]);
    const graph = await loadFlow(flow);

    mockFetch = installMockFetch(() => ({ echo: "done" }));
    const result = await graph.invoke({
      inputs: { a: "1", key_name: "Trace", key_val: "on" },
    });

    expect(outputsOf(result)).toEqual({ echo: "done" });
    const init = mockFetch.calls[0]!.init!;
    const headers = init.headers as Record<string, string>;
    expect(headers["Content-Type"]).toBe("application/x-www-form-urlencoded");
    expect(headers["X-Trace"]).toBe("on");
    expect(init.body).toBeInstanceOf(URLSearchParams);
    expect(String(init.body)).toBe("a=1&b=static");
  });

  it("POST: an empty-string Content-Type falls through to the lowercase header (Python `or` parity)", async () => {
    // Python looks the content type up with `get("Content-Type") or
    // get("content-type")`: an empty-string uppercase header is falsy, so
    // the lowercase urlencoded header wins and dict data goes out as a form
    // body (a `??` lookup would stop at the empty string and send JSON).
    const echo = stringProperty({ title: "echo" });
    const apiNode = createApiNode({
      name: "api",
      url: "https://api.example.com/form",
      httpMethod: "POST",
      data: { a: "1" },
      headers: {
        "Content-Type": "",
        "content-type": "application/x-www-form-urlencoded",
      },
      outputs: [echo],
    });
    const flow = buildApiFlow(apiNode, [], [echo]);
    const graph = await loadFlow(flow);

    mockFetch = installMockFetch(() => ({ echo: "done" }));
    const result = await graph.invoke({ inputs: {} });

    expect(outputsOf(result)).toEqual({ echo: "done" });
    const init = mockFetch.calls[0]!.init!;
    expect(init.body).toBeInstanceOf(URLSearchParams);
    expect(String(init.body)).toBe("a=1");
  });

  it("does not follow redirects: a 3xx response body maps to the node outputs like any status", async () => {
    // Python's httpx does not follow redirects (follow_redirects defaults to
    // False) and parses the returned 3xx body like any other status; the
    // adapter uses redirect: "manual" so undici returns the 3xx response
    // itself instead of requesting the Location target.
    const echo = stringProperty({ title: "echo" });
    const apiNode = createApiNode({
      name: "api",
      url: "https://api.example.com/redirecting",
      httpMethod: "GET",
      outputs: [echo],
    });
    const flow = buildApiFlow(apiNode, [], [echo]);
    const graph = await loadFlow(flow);

    mockFetch = installMockFetch(
      () =>
        new Response('{"echo": "from-redirect-response"}', {
          status: 302,
          headers: {
            "Content-Type": "application/json",
            Location: "https://attacker.example/exfil",
          },
        }),
    );
    const result = await graph.invoke({ inputs: {} });

    expect(outputsOf(result)).toEqual({ echo: "from-redirect-response" });
    expect(mockFetch.calls).toHaveLength(1);
    expect(mockFetch.calls[0]!.init!.redirect).toBe("manual");
  });

  it("attaches the default httpx-parity timeout and names the node on a timeout abort", async () => {
    const echo = stringProperty({ title: "echo" });
    const apiNode = createApiNode({
      name: "api",
      url: "https://api.example.com/slow",
      httpMethod: "GET",
      outputs: [echo],
    });
    const flow = buildApiFlow(apiNode, [], [echo]);
    const graph = await loadFlow(flow);

    mockFetch = installMockFetch(() => {
      throw new DOMException(
        "The operation was aborted due to timeout",
        "TimeoutError",
      );
    });

    await expect(graph.invoke({ inputs: {} })).rejects.toThrow(
      `ApiNode \`api\` HTTP request timed out after ${DEFAULT_HTTP_REQUEST_TIMEOUT_MS}ms.`,
    );
    expect(mockFetch.calls).toHaveLength(1);
    expect(mockFetch.calls[0]!.init!.signal).toBeInstanceOf(AbortSignal);
  });

  it("enforces the url allow list on the rendered URL and suppresses the templated warning", async () => {
    // Ports test_apinode_rejects_rendered_url_outside_allow_list (and the
    // allowed-URL half of test_apinode_can_be_imported_and_executed).
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const inputProps = [
      stringProperty({ title: "host" }),
      stringProperty({ title: "order_id" }),
    ];
    const status = stringProperty({ title: "status" });
    const apiNode = createApiNode({
      name: "api",
      url: "https://{{host}}/orders/{{order_id}}",
      httpMethod: "GET",
      urlAllowList: ["https://allowed.example.com/orders/"],
      inputs: inputProps,
      outputs: [status],
    });
    const flow = buildApiFlow(apiNode, inputProps, [status]);
    const graph = await loadFlow(flow);
    // The configured allow list suppresses the templated-destination warning.
    expect(warnSpy).not.toHaveBeenCalled();

    mockFetch = installMockFetch(() => ({ status: "ok" }));
    const result = await graph.invoke({
      inputs: { host: "allowed.example.com", order_id: "123" },
    });
    expect(outputsOf(result)).toEqual({ status: "ok" });
    expect(mockFetch.calls[0]!.url).toBe(
      "https://allowed.example.com/orders/123",
    );

    await expect(
      graph.invoke({ inputs: { host: "blocked.example.com", order_id: "123" } }),
    ).rejects.toThrow("Requested URL is not in allowed list");
    expect(mockFetch.calls).toHaveLength(1);
  });

  it("retries per the node's retryPolicy and overrides the request timeout", async () => {
    const timeoutSpy = vi.spyOn(AbortSignal, "timeout");
    const status = stringProperty({ title: "status" });
    const apiNode = createApiNode({
      name: "api",
      url: "https://api.example.com/orders",
      httpMethod: "GET",
      retryPolicy: {
        maxAttempts: 1,
        requestTimeout: 0.25,
        initialRetryDelay: 0,
        maxRetryDelay: 0,
      },
      outputs: [status],
    });
    const flow = buildApiFlow(apiNode, [], [status]);
    const graph = await loadFlow(flow);

    let call = 0;
    mockFetch = installMockFetch(() => {
      call += 1;
      return call === 1
        ? new Response('{"error": "busy"}', {
            status: 503,
            headers: { "Content-Type": "application/json" },
          })
        : { status: "ok" };
    });
    const result = await graph.invoke({ inputs: {} });

    expect(outputsOf(result)).toEqual({ status: "ok" });
    expect(mockFetch.calls).toHaveLength(2);
    expect(timeoutSpy.mock.calls.map(([ms]) => ms)).toEqual([250, 250]);
  });

  it("raises for a final error status when a retryPolicy is configured", async () => {
    const status = stringProperty({ title: "status" });
    const apiNode = createApiNode({
      name: "api",
      url: "https://api.example.com/orders",
      httpMethod: "GET",
      retryPolicy: { maxAttempts: 0 },
      outputs: [status],
    });
    const flow = buildApiFlow(apiNode, [], [status]);
    const graph = await loadFlow(flow);

    mockFetch = installMockFetch(
      () =>
        new Response('{"error": "busy"}', {
          status: 503,
          headers: { "Content-Type": "application/json" },
        }),
    );

    await expect(graph.invoke({ inputs: {} })).rejects.toThrow(
      "ApiNode `api` HTTP request failed with status '503' " +
        "for url 'https://api.example.com/orders'.",
    );
    expect(mockFetch.calls).toHaveLength(1);
  });

  it("POST: string data is sent as a raw body without forcing a content type", async () => {
    const inputProps = [stringProperty({ title: "val" })];
    const echo = stringProperty({ title: "echo" });
    const apiNode = createApiNode({
      name: "api",
      url: "https://api.example.com/raw",
      httpMethod: "POST",
      data: "payload={{val}}",
      inputs: inputProps,
      outputs: [echo],
    });
    const flow = buildApiFlow(apiNode, inputProps, [echo]);
    const graph = await loadFlow(flow);

    mockFetch = installMockFetch(() => ({ echo: "done" }));
    const result = await graph.invoke({ inputs: { val: "hello" } });

    expect(outputsOf(result)).toEqual({ echo: "done" });
    const init = mockFetch.calls[0]!.init!;
    expect(init.body).toBe("payload=hello");
    const headerKeys = Object.keys(init.headers as Record<string, string>);
    expect(
      headerKeys.some((key) => key.toLowerCase() === "content-type"),
    ).toBe(false);
  });
});
