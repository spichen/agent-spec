/**
 * Shared RemoteTool execution helper. Port of
 * `pyagentspec.adapters._tools_common._create_remote_tool_func`.
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
 *   methods.
 *
 * Python-parity network behavior (NOT divergences): redirects are not
 * followed and requests time out after `DEFAULT_HTTP_REQUEST_TIMEOUT_MS`,
 * matching httpx's `follow_redirects=False` and 5s-timeout defaults — see
 * `fetchWithAdapterDefaults`.
 */
import type { RemoteTool } from "../../tools/remote-tool.js";
import {
  renderNestedObjectTemplate,
  renderTemplate,
  stringifyTemplateValue,
} from "./templating.js";
import {
  maybeWarnAboutUnrestrictedTemplatedUrl,
  validateUrlAgainstAllowList,
} from "./url-validation.js";

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  const prototype: unknown = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

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

function renderRecord(
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
    const remoteToolData = renderNestedObjectTemplate(remoteTool.data, kwargs);
    const remoteToolHeaders = renderRecord(remoteTool.headers, kwargs);
    const remoteToolQueryParams = renderRecord(remoteTool.queryParams, kwargs);
    const remoteToolUrl = renderTemplate(remoteTool.url, kwargs);

    const contentTypeHeader =
      remoteToolHeaders["Content-Type"] || remoteToolHeaders["content-type"];
    const expectUrlencodedFormData =
      typeof contentTypeHeader === "string" &&
      contentTypeHeader.includes("application/x-www-form-urlencoded");

    const requestHeaders: Record<string, string> = {};
    for (const [key, value] of Object.entries(remoteToolHeaders)) {
      requestHeaders[key] =
        typeof value === "string" ? value : stringifyTemplateValue(value);
    }
    const callerSetContentType = Object.keys(requestHeaders).some(
      (key) => key.toLowerCase() === "content-type",
    );

    const method = remoteTool.httpMethod;
    const methodUpper = method.toUpperCase();
    const methodAllowsBody = methodUpper !== "GET" && methodUpper !== "HEAD";

    let body: string | URLSearchParams | Uint8Array | undefined;
    if (methodAllowsBody) {
      if (expectUrlencodedFormData && isPlainRecord(remoteToolData)) {
        const form = new URLSearchParams();
        for (const [key, value] of Object.entries(remoteToolData)) {
          form.append(
            key,
            typeof value === "string" ? value : stringifyTemplateValue(value),
          );
        }
        body = form;
      } else if (typeof remoteToolData === "string") {
        body = remoteToolData;
      } else if (remoteToolData instanceof Uint8Array) {
        body = remoteToolData;
      } else if (remoteToolData !== undefined && remoteToolData !== null) {
        body = JSON.stringify(remoteToolData);
        if (!callerSetContentType) {
          requestHeaders["Content-Type"] = "application/json";
        }
      }
    }

    // Kept as the seam for allow-list enforcement: the TS SDK RemoteTool has
    // no urlAllowList field yet, so this always allows.
    validateUrlAgainstAllowList(remoteToolUrl, undefined);

    const searchParams = new URLSearchParams();
    for (const [key, value] of Object.entries(remoteToolQueryParams)) {
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
        ? `${remoteToolUrl}${remoteToolUrl.includes("?") ? "&" : "?"}${query}`
        : remoteToolUrl;

    const response = await fetchWithAdapterDefaults(
      requestUrl,
      {
        method,
        headers: requestHeaders,
        ...(body !== undefined ? { body } : {}),
      },
      `RemoteTool \`${remoteTool.name}\``,
    );
    // Python (with no retry policy — the only state the TS RemoteTool can
    // express) parses and returns the JSON body for every status, so error
    // responses flow back to the agent as the tool result. Redirects are not
    // followed (see fetchWithAdapterDefaults), so a 3xx body parses here too.
    return (await response.json()) as unknown;
  };
}
