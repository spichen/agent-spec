/**
 * ApiNode executor for the LangGraph adapter.
 *
 * Port of `pyagentspec.adapters.langgraph._node_execution.ApiNodeExecutor`.
 * The request assembly is shared with the RemoteTool path through
 * `buildTemplatedHttpRequest` (Python spells it out identically at both
 * sites).
 *
 * Divergences from Python (see the adapter README):
 * - `fetch` forbids request bodies on GET/HEAD (Python's httpx sends them):
 *   the declared body is not sent for those methods and a warning is emitted
 *   instead of silently dropping it.
 * - The node's `retryPolicy` drives the shared retry engine (and raises for
 *   a final error status): Python's ApiNodeExecutor performs a single plain
 *   request, leaving the ApiNode retry policy representation-only.
 *
 * Python-parity network behavior (NOT divergences): redirects are not
 * followed and requests time out after the shared httpx-parity default
 * (overridden by `retryPolicy.requestTimeout`) — see
 * `fetchWithAdapterDefaults`. The node's `urlAllowList` is enforced on the
 * rendered URL and suppresses the templated-destination warning, like
 * Python.
 */
import type { BaseMessage } from "@langchain/core/messages";
import type { ApiNode } from "../../../flows/index.js";
import {
  buildTemplatedHttpRequest,
  isRecordLike,
  maybeWarnAboutUnrestrictedTemplatedUrl,
  raiseForStatusWhenPolicySet,
  requestWithRetry,
} from "../../common/index.js";
import type { ExecuteOutput, NodeOutputs } from "../types.js";
import { NodeExecutor } from "./executor.js";

/**
 * Executes an ApiNode: renders `{{placeholder}}` templates in the URL, data,
 * headers and query params against the inputs, performs the HTTP request and
 * returns the parsed JSON response body as the node output.
 */
export class ApiNodeExecutor extends NodeExecutor<ApiNode> {
  constructor(node: ApiNode) {
    super(node);
    // A configured allow list suppresses the templated-destination warning,
    // exactly like Python.
    maybeWarnAboutUnrestrictedTemplatedUrl(
      node.url,
      node.urlAllowList,
      `ApiNode \`${node.name}\``,
    );
  }

  protected async _execute(
    inputs: NodeOutputs,
    _messages: BaseMessage[],
  ): Promise<ExecuteOutput> {
    // The loose record guard preserves this executor's historical semantics:
    // when a full `{{placeholder}}` substitution renders `data` to a
    // non-plain object (class instance, Map), it still counts as a record
    // for the urlencoded-form encoding and the GET/HEAD empty-body check
    // (the RemoteTool path keeps the strict default).
    const { url, init, bodyDropped } = buildTemplatedHttpRequest(
      this.node,
      inputs,
      { isRecord: isRecordLike },
    );
    if (bodyDropped) {
      // Forced divergence from Python: warn instead of silently dropping.
      console.warn(
        `ApiNode \`${this.node.name}\` declares request data for HTTP method ` +
          `${this.node.httpMethod.toUpperCase()}, but fetch forbids request bodies on GET/HEAD: ` +
          `the declared body is not sent (the Python adapter sends it).`,
      );
    }
    // Redirects are not followed and the request times out after the shared
    // default unless retryPolicy.requestTimeout overrides it, matching
    // Python's httpx defaults (see fetchWithAdapterDefaults). The node's
    // retryPolicy drives the shared retry engine.
    const response = await requestWithRetry(
      this.node.retryPolicy,
      url,
      init,
      `ApiNode \`${this.node.name}\``,
    );
    // With a retry policy a final error status raises; without one Python
    // parses the JSON body regardless of the HTTP status (a 3xx response
    // returned without following included).
    raiseForStatusWhenPolicySet(
      this.node.retryPolicy,
      response,
      `ApiNode \`${this.node.name}\``,
      url,
    );
    const responseJson = (await response.json()) as unknown;
    return [responseJson as NodeOutputs, {}];
  }
}
