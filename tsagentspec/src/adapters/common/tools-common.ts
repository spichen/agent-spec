/**
 * Shared templated-HTTP-request assembly and RemoteTool execution helpers.
 * Port of `pyagentspec.adapters._tools_common._create_remote_tool_func`; the
 * request assembly (`buildTemplatedHttpRequest`) is also the one Python
 * spells out a second time in `ApiNodeExecutor` (`_node_execution.py`) and is
 * shared here with the LangGraph ApiNode executor.
 *
 * Divergences from Python (see the adapter README):
 * - The TS SDK RemoteTool has no `retryPolicy`, so a single fetch attempt is
 *   performed (no retry/jitter/Retry-After machinery). Like Python without a
 *   retry policy, the response body is parsed and returned regardless of the
 *   HTTP status.
 * - The TS SDK RemoteTool has no `urlAllowList` field, so the allow-list
 *   helpers are invoked with `undefined` (i.e. allow) and the templated-URL
 *   warning fires per the Python rules.
 * - `fetch` forbids request bodies on GET/HEAD, so no body is sent for those
 *   methods (reported via `bodyDropped`).
 *
 * Python-parity network behavior (NOT divergences): redirects are not
 * followed and requests time out after `DEFAULT_HTTP_REQUEST_TIMEOUT_MS`,
 * matching httpx's `follow_redirects=False` and 5s-timeout defaults — see
 * `fetchWithAdapterDefaults`.
 */
import type { RemoteTool } from "../../tools/remote-tool.js";
import { isPlainRecord } from "./guards.js";
import {
  renderNestedObjectTemplate,
  renderTemplate,
  stringifyTemplateValue,
} from "./templating.js";
import {
  maybeWarnAboutUnrestrictedTemplatedUrl,
  validateUrlAgainstAllowList,
} from "./url-validation.js";

/**
 * Default timeout for RemoteTool / ApiNode HTTP requests, in milliseconds.
 *
 * Mirrors the 5-second default timeout httpx applies to every request made by
 * the Python adapter. The TS SDK has no `RetryPolicy.requestTimeout` field yet
 * (Python reads a per-tool override from there), so this constant is the only
 * knob for the request timeout.
 */
export const DEFAULT_HTTP_REQUEST_TIMEOUT_MS = 5000;

/**
 * Perform one `fetch` with the adapter's Python-parity network behavior:
 *
 * - Redirects are NOT followed (`redirect: "manual"`): Node's undici then
 *   returns the 3xx response itself (status/headers/body intact), matching
 *   httpx's `follow_redirects=False` default. Following redirects on
 *   untrusted spec config would enable redirect-based egress and forward
 *   custom auth headers to redirect targets.
 * - The request aborts after `DEFAULT_HTTP_REQUEST_TIMEOUT_MS` (httpx's
 *   default timeout); the abort is rethrown as an Error naming the requester
 *   and the timeout.
 */
export async function fetchWithAdapterDefaults(
  url: string,
  init: RequestInit,
  requesterDescription: string,
): Promise<Response> {
  try {
    return await fetch(url, {
      ...init,
      redirect: "manual",
      signal: AbortSignal.timeout(DEFAULT_HTTP_REQUEST_TIMEOUT_MS),
    });
  } catch (error) {
    // AbortSignal.timeout aborts with a DOMException named "TimeoutError".
    if (
      typeof error === "object" &&
      error !== null &&
      (error as { name?: unknown }).name === "TimeoutError"
    ) {
      throw new Error(
        `${requesterDescription} HTTP request timed out after ` +
          `${DEFAULT_HTTP_REQUEST_TIMEOUT_MS}ms.`,
      );
    }
    throw error;
  }
}

/**
 * Render `{{placeholder}}` templates in both the keys and the values of a
 * record (header/query-param maps), like Python's dict comprehensions over
 * `render_template(k)` / `render_nested_object_template(v)`.
 */
export function renderRecord(
  record: Record<string, unknown>,
  kwargs: Record<string, unknown>,
): Record<string, unknown> {
  const rendered: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(record)) {
    rendered[renderTemplate(key, kwargs)] = renderNestedObjectTemplate(
      value,
      kwargs,
    );
  }
  return rendered;
}

/**
 * The structural surface shared by the AgentSpec `RemoteTool` and `ApiNode`
 * components: a templated HTTP request specification.
 */
export interface TemplatedHttpRequestSpec {
  url: string;
  httpMethod: string;
  data?: unknown;
  headers: Record<string, unknown>;
  queryParams: Record<string, unknown>;
}

/**
 * Assemble one HTTP request from a templated spec and the call inputs:
 * renders `{{placeholder}}` templates in the URL, data, headers and query
 * parameters, stringifies header values, validates the rendered URL against
 * the allow list (a seam — the TS SDK has no `urlAllowList` field yet, so
 * this always allows), encodes the body (an urlencoded form for dict data
 * under an urlencoded content type, raw strings/bytes verbatim, JSON
 * otherwise — adding the JSON content type unless the caller set one), and
 * appends the rendered query parameters to the URL.
 *
 * Mirrors the request assembly Python spells out identically in
 * `_create_remote_tool_func` (`_tools_common.py`) and `ApiNodeExecutor`
 * (`_node_execution.py`).
 *
 * `bodyDropped` reports the one fetch-forced divergence: `fetch` forbids
 * request bodies on GET/HEAD (Python's httpx sends them), so declared
 * non-empty data is not sent for those methods and the flag is returned for
 * the caller to surface (the ApiNode executor warns; the RemoteTool path
 * keeps Python's silence).
 */
export function buildTemplatedHttpRequest(
  spec: TemplatedHttpRequestSpec,
  inputs: Record<string, unknown>,
): { url: string; init: RequestInit; bodyDropped: boolean } {
  const renderedData = renderNestedObjectTemplate(spec.data, inputs);
  const renderedHeaders = renderRecord(spec.headers, inputs);
  const renderedQueryParams = renderRecord(spec.queryParams, inputs);
  const renderedUrl = renderTemplate(spec.url, inputs);

  // Falsy (`||`) coalescing on purpose, matching Python's
  // `headers.get("Content-Type") or headers.get("content-type")`: an
  // empty-string `Content-Type` falls through to the lowercase header.
  const contentTypeHeader =
    renderedHeaders["Content-Type"] || renderedHeaders["content-type"];
  const expectUrlencodedFormData =
    typeof contentTypeHeader === "string" &&
    contentTypeHeader.includes("application/x-www-form-urlencoded");

  const requestHeaders: Record<string, string> = {};
  for (const [key, value] of Object.entries(renderedHeaders)) {
    requestHeaders[key] =
      typeof value === "string" ? value : stringifyTemplateValue(value);
  }
  const callerSetContentType = Object.keys(requestHeaders).some(
    (key) => key.toLowerCase() === "content-type",
  );

  const method = spec.httpMethod;
  const methodUpper = method.toUpperCase();
  // fetch forbids request bodies on GET/HEAD (Python's httpx sends them).
  const methodAllowsBody = methodUpper !== "GET" && methodUpper !== "HEAD";
  const hasDeclaredBody =
    renderedData !== undefined &&
    renderedData !== null &&
    renderedData !== "" &&
    !(isPlainRecord(renderedData) && Object.keys(renderedData).length === 0);

  let body: string | URLSearchParams | Uint8Array | undefined;
  if (methodAllowsBody) {
    if (expectUrlencodedFormData && isPlainRecord(renderedData)) {
      const form = new URLSearchParams();
      for (const [key, value] of Object.entries(renderedData)) {
        form.append(
          key,
          typeof value === "string" ? value : stringifyTemplateValue(value),
        );
      }
      body = form;
    } else if (typeof renderedData === "string") {
      body = renderedData;
    } else if (renderedData instanceof Uint8Array) {
      body = renderedData;
    } else if (renderedData !== undefined && renderedData !== null) {
      body = JSON.stringify(renderedData);
      if (!callerSetContentType) {
        requestHeaders["Content-Type"] = "application/json";
      }
    }
  }

  // Kept as the seam for allow-list enforcement: neither the TS SDK
  // RemoteTool nor the ApiNode has a urlAllowList field yet, so this always
  // allows.
  validateUrlAgainstAllowList(renderedUrl, undefined);

  const searchParams = new URLSearchParams();
  for (const [key, value] of Object.entries(renderedQueryParams)) {
    if (Array.isArray(value)) {
      for (const item of value) {
        searchParams.append(
          key,
          item == null ? "" : stringifyTemplateValue(item),
        );
      }
    } else {
      searchParams.append(
        key,
        value == null ? "" : stringifyTemplateValue(value),
      );
    }
  }
  const query = searchParams.toString();
  const requestUrl =
    query.length > 0
      ? `${renderedUrl}${renderedUrl.includes("?") ? "&" : "?"}${query}`
      : renderedUrl;

  return {
    url: requestUrl,
    init: {
      method,
      headers: requestHeaders,
      ...(body !== undefined ? { body } : {}),
    },
    bodyDropped: !methodAllowsBody && hasDeclaredBody,
  };
}

/**
 * Create the execution function for an AgentSpec RemoteTool.
 *
 * The returned function renders `{{placeholder}}` templates in the URL, data,
 * headers and query parameters using the call kwargs, validates the rendered
 * URL, performs a single `fetch`, and returns the parsed JSON response body.
 *
 * Note: `requiresConfirmation` wrapping is applied by the framework-specific
 * adapter layer (e.g. the LangGraph adapter), not here.
 */
export function createRemoteToolFunc(
  remoteTool: RemoteTool,
): (kwargs: Record<string, unknown>) => Promise<unknown> {
  maybeWarnAboutUnrestrictedTemplatedUrl(
    remoteTool.url,
    undefined,
    `RemoteTool \`${remoteTool.name}\``,
  );

  return async function remoteToolFunc(
    kwargs: Record<string, unknown>,
  ): Promise<unknown> {
    const { url, init } = buildTemplatedHttpRequest(remoteTool, kwargs);
    const response = await fetchWithAdapterDefaults(
      url,
      init,
      `RemoteTool \`${remoteTool.name}\``,
    );
    // Python (with no retry policy — the only state the TS RemoteTool can
    // express) parses and returns the JSON body for every status, so error
    // responses flow back to the agent as the tool result. Redirects are not
    // followed (see fetchWithAdapterDefaults), so a 3xx body parses here too.
    return (await response.json()) as unknown;
  };
}
