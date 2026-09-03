/**
 * ApiNode executor for the LangGraph adapter.
 *
 * Port of `pyagentspec.adapters.langgraph._node_execution.ApiNodeExecutor`.
 * The request assembly is shared with the RemoteTool path through
 * `buildTemplatedHttpRequest` (Python spells it out identically at both
 * sites).
 *
 * Divergences from Python (see the adapter README):
 * - The TS SDK ApiNode has no `urlAllowList` field, so the allow-list helpers
 *   are invoked with `undefined` (i.e. allow) and the templated-URL warning
 *   fires per the Python rules.
 * - `fetch` forbids request bodies on GET/HEAD (Python's httpx sends them):
 *   the declared body is not sent for those methods and a warning is emitted
 *   instead of silently dropping it.
 *
 * Python-parity network behavior (NOT divergences): redirects are not
 * followed and requests time out after the shared httpx-parity default — see
 * `fetchWithAdapterDefaults`.
 */
import type { BaseMessage } from "@langchain/core/messages";
import type { ApiNode } from "../../../flows/index.js";
import {
  buildTemplatedHttpRequest,
  fetchWithAdapterDefaults,
  maybeWarnAboutUnrestrictedTemplatedUrl,
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
    // The TS SDK ApiNode has no urlAllowList field yet: the helpers are
    // invoked with `undefined` (i.e. allow), matching the documented
    // divergence, so the templated-URL warning fires per the Python rules.
    maybeWarnAboutUnrestrictedTemplatedUrl(
      node.url,
      undefined,
      `ApiNode \`${node.name}\``,
    );
  }

  protected async _execute(
    inputs: NodeOutputs,
    _messages: BaseMessage[],
  ): Promise<ExecuteOutput> {
    const { url, init, bodyDropped } = buildTemplatedHttpRequest(
      this.node,
      inputs,
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
    // default, matching Python's httpx defaults (see fetchWithAdapterDefaults).
    const response = await fetchWithAdapterDefaults(
      url,
      init,
      `ApiNode \`${this.node.name}\``,
    );
    // Python parses the JSON body regardless of the HTTP status (a 3xx
    // response returned without following included).
    const responseJson = (await response.json()) as unknown;
    return [responseJson as NodeOutputs, {}];
  }
}
