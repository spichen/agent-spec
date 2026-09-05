/**
 * Shared templated-HTTP-request assembly, retry engine, and RemoteTool
 * execution helpers. Port of `pyagentspec.adapters._tools_common`
 * (`_create_remote_tool_func` and the `_request_with_retry` engine); the
 * request assembly (`buildTemplatedHttpRequest`) is also the one Python
 * spells out a second time in `ApiNodeExecutor` (`_node_execution.py`) and is
 * shared here with the LangGraph ApiNode executor, which also reuses the
 * retry engine (Python's ApiNodeExecutor performs a single plain request).
 *
 * Divergences from Python (see the adapter README):
 * - `fetch` forbids request bodies on GET/HEAD, so no body is sent for those
 *   methods (reported via `bodyDropped`).
 * - Jitter randomness uses `Math.random()` (Python uses `SystemRandom`), and
 *   TLS-failure detection extends Python's message patterns with Node's TLS
 *   error texts (case-insensitively).
 *
 * Python-parity network behavior (NOT divergences): redirects are not
 * followed and requests time out after `DEFAULT_HTTP_REQUEST_TIMEOUT_MS`
 * unless the retry policy configures `requestTimeout`, matching httpx's
 * `follow_redirects=False` and 5s-timeout defaults — see
 * `fetchWithAdapterDefaults`.
 */
import type { RetryPolicy } from "../../retry-policy.js";
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
 * the Python adapter. A configured `RetryPolicy.requestTimeout` (seconds)
 * overrides it per tool/node, like Python's `httpx.Timeout` override.
 */
export const DEFAULT_HTTP_REQUEST_TIMEOUT_MS = 5000;

/** Cap (seconds) on the total time spent across retry attempts. */
export const DEFAULT_TOTAL_ELAPSED_TIME_SECONDS = 600;

/** Cap (seconds) applied to server-provided `Retry-After` delays. */
export const MAX_RETRY_AFTER_SECONDS = 30;

/**
 * Perform one `fetch` with the adapter's Python-parity network behavior:
 *
 * - Redirects are NOT followed (`redirect: "manual"`): Node's undici then
 *   returns the 3xx response itself (status/headers/body intact), matching
 *   httpx's `follow_redirects=False` default. Following redirects on
 *   untrusted spec config would enable redirect-based egress and forward
 *   custom auth headers to redirect targets.
 * - The request aborts after `timeoutMs` (defaulting to httpx's 5s default
 *   timeout); the abort is rethrown as an Error naming the requester and the
 *   timeout.
 */
export async function fetchWithAdapterDefaults(
  url: string,
  init: RequestInit,
  requesterDescription: string,
  timeoutMs: number = DEFAULT_HTTP_REQUEST_TIMEOUT_MS,
): Promise<Response> {
  try {
    return await fetch(url, {
      ...init,
      redirect: "manual",
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (error) {
    // AbortSignal.timeout aborts with a DOMException named "TimeoutError".
    if (
      typeof error === "object" &&
      error !== null &&
      (error as { name?: unknown }).name === "TimeoutError"
    ) {
      throw new Error(
        `${requesterDescription} HTTP request timed out after ${timeoutMs}ms.`,
      );
    }
    throw error;
  }
}

/**
 * Message-text patterns identifying TLS/certificate validation failures
 * (matched case-insensitively over each error's name, code and message).
 *
 * Ports Python's `_is_tls_or_cert_error` pattern list and extends it with
 * the texts Node/undici produce for the same failures.
 */
const TLS_ERROR_PATTERNS: readonly string[] = [
  "certificate_verify_failed",
  "certificate verify failed",
  "hostname",
  "self signed certificate",
  "self-signed certificate",
  "unable to verify the first certificate",
  "unable to get local issuer certificate",
  "certificate has expired",
  "altname",
];

/**
 * Return whether an error chain represents a TLS/certificate validation
 * failure. Such failures are never retried: retrying cannot fix a bad
 * certificate, and hammering a possibly-MITMed endpoint is undesirable.
 *
 * Like Python (which unwraps `__cause__`), the JS `cause` chain is walked and
 * each error's name, code and message are matched against
 * `TLS_ERROR_PATTERNS` (undici wraps the TLS error in a generic
 * `TypeError: fetch failed` whose `cause` carries the real failure).
 */
export function isTlsOrCertError(error: unknown): boolean {
  let current: unknown = error;
  const seen = new Set<unknown>();
  while (typeof current === "object" && current !== null && !seen.has(current)) {
    seen.add(current);
    const { message, code, name } = current as {
      message?: unknown;
      code?: unknown;
      name?: unknown;
    };
    const haystack = [name, code, message]
      .filter((part): part is string => typeof part === "string")
      .join(" ")
      .toLowerCase();
    if (TLS_ERROR_PATTERNS.some((pattern) => haystack.includes(pattern))) {
      return true;
    }
    current = (current as { cause?: unknown }).cause;
  }
  return false;
}

/** HTTP statuses never retried, regardless of `recoverableStatuses`. */
const NON_RETRYABLE_STATUSES = new Set([400, 401, 403, 422]);

/**
 * Return a response body string suitable for retry-code matching (Python's
 * `_get_response_error_text`). Reads from a clone so the original body stays
 * readable in case the response is returned to the caller after a
 * "not retryable" decision.
 */
async function getResponseErrorText(response: Response): Promise<string> {
  try {
    return await response.clone().text();
  } catch {
    return "";
  }
}

/**
 * Return whether an HTTP error response should be retried under the policy.
 * Port of Python's `_is_retryable_http_error`.
 */
async function isRetryableHttpError(
  retryPolicy: RetryPolicy,
  response: Response,
): Promise<boolean> {
  const statusCode = response.status;
  // Agent Spec says runtimes SHOULD NOT retry auth/authz or validation
  // errors, but does not explicitly define precedence against
  // `recoverable_statuses`. We interpret that non-retryable guidance as
  // taking precedence (matching Python).
  if (NON_RETRYABLE_STATUSES.has(statusCode)) {
    return false;
  }
  // Agent Spec defines `service_error_retry_on_any_5xx` as excluding HTTP 501.
  if (statusCode === 501) {
    return false;
  }

  const statusKey = String(statusCode);
  const retryCodes = Object.hasOwn(retryPolicy.recoverableStatuses, statusKey)
    ? retryPolicy.recoverableStatuses[statusKey]
    : undefined;
  if (retryCodes !== undefined) {
    if (retryCodes.length === 0) {
      return true;
    }
    const loweredResponseText = (
      await getResponseErrorText(response)
    ).toLowerCase();
    return retryCodes.some((code) =>
      loweredResponseText.includes(code.toLowerCase()),
    );
  }

  if (
    retryPolicy.serviceErrorRetryOnAny5xx &&
    statusCode >= 500 &&
    statusCode < 600
  ) {
    return true;
  }
  return false;
}

/**
 * Parse and cap a `Retry-After` header value (Python's
 * `_get_retry_after_seconds`): a numeric value is taken as seconds, an
 * HTTP-date as the seconds until that instant (never negative); both are
 * capped at `MAX_RETRY_AFTER_SECONDS`. Returns null for absent or unparsable
 * values.
 */
export function getRetryAfterSeconds(
  retryAfterValue: string | null,
  nowMs?: number,
): number | null {
  if (retryAfterValue === null) {
    return null;
  }
  const trimmed = retryAfterValue.trim();
  if (trimmed !== "") {
    const numericValue = Number(trimmed);
    if (Number.isFinite(numericValue)) {
      return Math.min(numericValue, MAX_RETRY_AFTER_SECONDS);
    }
  }
  const retryAfterDateMs = Date.parse(retryAfterValue);
  if (Number.isNaN(retryAfterDateMs)) {
    return null;
  }
  const currentMs = nowMs ?? Date.now();
  return Math.min(
    Math.max(0, (retryAfterDateMs - currentMs) / 1000),
    MAX_RETRY_AFTER_SECONDS,
  );
}

/**
 * Compute exponential backoff with the configured jitter strategy (Python's
 * `_compute_wait_seconds`). `full_and_equal_for_throttle` applies equal
 * jitter to 4xx throttling responses and full jitter otherwise.
 */
export function computeWaitSeconds(
  retryPolicy: RetryPolicy,
  attemptNum: number,
  statusCode: number | null,
): number {
  const base = Math.min(
    retryPolicy.initialRetryDelay * retryPolicy.backoffFactor ** attemptNum,
    retryPolicy.maxRetryDelay,
  );

  const jitter = retryPolicy.jitter;
  if (jitter == null) {
    return base;
  }
  if (jitter === "equal") {
    return base / 2 + Math.random() * (base / 2);
  }
  if (jitter === "full") {
    return Math.random() * base;
  }
  if (
    jitter === "full_and_equal_for_throttle" &&
    statusCode !== null &&
    statusCode >= 400 &&
    statusCode < 500
  ) {
    return base / 2 + Math.random() * (base / 2);
  }
  if (jitter === "full_and_equal_for_throttle") {
    return Math.random() * base;
  }
  if (jitter === "decorrelated") {
    return Math.min(base + Math.random(), retryPolicy.maxRetryDelay);
  }
  return base;
}

/**
 * Compute the bounded delay before the next retry attempt (Python's
 * `_compute_wait_before_next_attempt`): a parsable `Retry-After` wins over
 * the backoff computation, and the result is clamped to the time remaining
 * under `DEFAULT_TOTAL_ELAPSED_TIME_SECONDS`. Returns null when the elapsed
 * budget is exhausted (the caller then gives up retrying).
 */
function computeWaitBeforeNextAttempt(
  retryPolicy: RetryPolicy,
  attemptNum: number,
  statusCode: number | null,
  retryAfterValue: string | null,
  timeStartedMs: number,
): number | null {
  const waitTimeSeconds =
    getRetryAfterSeconds(retryAfterValue) ??
    computeWaitSeconds(retryPolicy, attemptNum, statusCode);

  const remainingSeconds =
    DEFAULT_TOTAL_ELAPSED_TIME_SECONDS - (Date.now() - timeStartedMs) / 1000;
  if (remainingSeconds <= 0) {
    return null;
  }
  return Math.min(waitTimeSeconds, remainingSeconds);
}

function sleepSeconds(seconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, seconds * 1000));
}

/**
 * Execute an HTTP request with retry-policy handling. Port of Python's
 * `_request_with_retry`.
 *
 * Without a policy, a single `fetch` is performed with the default timeout.
 * With a policy: up to `maxAttempts` retries are attempted after the initial
 * request, transport errors are retried unless they are TLS/certificate
 * failures, error responses are retried per `recoverableStatuses` /
 * `serviceErrorRetryOnAny5xx` (with response-body error-code matching),
 * delays honor `Retry-After` (numeric or HTTP-date, capped at
 * `MAX_RETRY_AFTER_SECONDS`) or exponential backoff with the configured
 * jitter, the total retry time is capped at
 * `DEFAULT_TOTAL_ELAPSED_TIME_SECONDS`, and `requestTimeout` (seconds)
 * overrides the per-attempt timeout.
 *
 * Like Python, an error response that is out of retries (or not retryable)
 * is RETURNED, not thrown — `raiseForStatusWhenPolicySet` restores the
 * with-policy error behavior at the call sites.
 */
export async function requestWithRetry(
  retryPolicy: RetryPolicy | undefined,
  url: string,
  init: RequestInit,
  requesterDescription: string,
): Promise<Response> {
  if (retryPolicy == null) {
    return fetchWithAdapterDefaults(url, init, requesterDescription);
  }

  const timeoutMs =
    retryPolicy.requestTimeout != null
      ? retryPolicy.requestTimeout * 1000
      : DEFAULT_HTTP_REQUEST_TIMEOUT_MS;
  const totalAttempts = retryPolicy.maxAttempts + 1;
  const timeStartedMs = Date.now();

  for (let attemptNum = 0; attemptNum < totalAttempts; attemptNum += 1) {
    let response: Response;
    try {
      response = await fetchWithAdapterDefaults(
        url,
        init,
        requesterDescription,
        timeoutMs,
      );
    } catch (error) {
      if (isTlsOrCertError(error) || attemptNum >= totalAttempts - 1) {
        throw error;
      }
      const waitTimeSeconds = computeWaitBeforeNextAttempt(
        retryPolicy,
        attemptNum,
        null,
        null,
        timeStartedMs,
      );
      if (waitTimeSeconds === null) {
        throw error;
      }
      await sleepSeconds(waitTimeSeconds);
      continue;
    }

    if (response.ok) {
      return response;
    }

    // The last attempt's response is returned before any retryability check
    // so its body is never consumed by error-text matching (Python computes
    // the text first; httpx responses are re-readable, fetch bodies are not).
    if (attemptNum >= totalAttempts - 1) {
      return response;
    }
    if (!(await isRetryableHttpError(retryPolicy, response))) {
      return response;
    }

    const waitTimeSeconds = computeWaitBeforeNextAttempt(
      retryPolicy,
      attemptNum,
      response.status,
      response.headers.get("retry-after"),
      timeStartedMs,
    );
    if (waitTimeSeconds === null) {
      return response;
    }
    // Python closes the response before sleeping; cancel the unread body so
    // the connection is released.
    try {
      await response.body?.cancel();
    } catch {
      // The body may already be disturbed (e.g. by a consumed clone).
    }
    await sleepSeconds(waitTimeSeconds);
  }

  throw new Error("Request failed after retry attempts were exhausted.");
}

/**
 * Throw for a non-2xx response when a retry policy is configured, mirroring
 * Python's `response.raise_for_status()` call that runs only with a policy
 * set. Without a policy the caller keeps the parse-any-status behavior
 * (error payloads flow back as the tool/node result).
 */
export function raiseForStatusWhenPolicySet(
  retryPolicy: RetryPolicy | undefined,
  response: Response,
  requesterDescription: string,
  url: string,
): void {
  if (retryPolicy == null || response.ok) {
    return;
  }
  const statusText =
    response.statusText.length > 0 ? ` ${response.statusText}` : "";
  throw new Error(
    `${requesterDescription} HTTP request failed with status ` +
      `'${response.status}${statusText}' for url '${url}'.`,
  );
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
 * components: a templated HTTP request specification with an optional URL
 * allow list.
 */
export interface TemplatedHttpRequestSpec {
  url: string;
  httpMethod: string;
  data?: unknown;
  headers: Record<string, unknown>;
  queryParams: Record<string, unknown>;
  urlAllowList?: string[] | undefined;
}

/**
 * Assemble one HTTP request from a templated spec and the call inputs:
 * renders `{{placeholder}}` templates in the URL, data, headers and query
 * parameters, stringifies header values, validates the rendered URL against
 * the spec's `urlAllowList` (throwing the Python rejection error when the
 * rendered URL matches no entry), encodes the body (an urlencoded form for
 * dict data under an urlencoded content type, raw strings/bytes verbatim,
 * JSON otherwise — adding the JSON content type unless the caller set one),
 * and appends the rendered query parameters to the URL.
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
 *
 * `options.isRecord` decides which rendered data values count as a record for
 * the urlencoded-form encoding and the GET/HEAD empty-body check. It defaults
 * to the strict `isPlainRecord` (the RemoteTool path's historical guard); the
 * ApiNode executor passes the loose `isRecordLike`, preserving each caller's
 * pre-unification behavior when a full `{{placeholder}}` substitution renders
 * `data` to a non-plain object such as a class instance.
 */
export function buildTemplatedHttpRequest(
  spec: TemplatedHttpRequestSpec,
  inputs: Record<string, unknown>,
  options: {
    isRecord?: (value: unknown) => value is Record<string, unknown>;
  } = {},
): { url: string; init: RequestInit; bodyDropped: boolean } {
  const isRecord = options.isRecord ?? isPlainRecord;
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
    !(isRecord(renderedData) && Object.keys(renderedData).length === 0);

  let body: string | URLSearchParams | Uint8Array | undefined;
  if (methodAllowsBody) {
    if (expectUrlencodedFormData && isRecord(renderedData)) {
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

  // Enforced on the fully rendered URL, before the request goes out (query
  // parameters are appended below but do not participate in matching).
  validateUrlAgainstAllowList(renderedUrl, spec.urlAllowList);

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
 * URL against the tool's `urlAllowList`, performs the request under the
 * tool's `retryPolicy` (a single `fetch` without one), and returns the
 * parsed JSON response body.
 *
 * Note: `requiresConfirmation` wrapping is applied by the framework-specific
 * adapter layer (e.g. the LangGraph adapter), not here.
 */
export function createRemoteToolFunc(
  remoteTool: RemoteTool,
): (kwargs: Record<string, unknown>) => Promise<unknown> {
  // A configured allow list suppresses the templated-destination warning,
  // exactly like Python.
  maybeWarnAboutUnrestrictedTemplatedUrl(
    remoteTool.url,
    remoteTool.urlAllowList,
    `RemoteTool \`${remoteTool.name}\``,
  );

  return async function remoteToolFunc(
    kwargs: Record<string, unknown>,
  ): Promise<unknown> {
    const { url, init } = buildTemplatedHttpRequest(remoteTool, kwargs);
    const response = await requestWithRetry(
      remoteTool.retryPolicy,
      url,
      init,
      `RemoteTool \`${remoteTool.name}\``,
    );
    // With a retry policy Python raises for a final error status; without
    // one it parses and returns the JSON body for every status, so error
    // responses flow back to the agent as the tool result. Redirects are not
    // followed (see fetchWithAdapterDefaults), so a 3xx body parses here too.
    raiseForStatusWhenPolicySet(
      remoteTool.retryPolicy,
      response,
      `RemoteTool \`${remoteTool.name}\``,
      url,
    );
    return (await response.json()) as unknown;
  };
}
