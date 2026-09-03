/**
 * Tests for the shared templated-HTTP-request assembly.
 *
 * The request-body matrix (JSON / urlencoded form / raw string / GET-HEAD
 * drop) is exercised end-to-end through the ApiNode flow tests; this file
 * pins the per-caller record-guard contract of `buildTemplatedHttpRequest`:
 * the strict default (the RemoteTool path) versus the loose `isRecordLike`
 * guard the ApiNode executor passes, preserving each call site's
 * pre-unification behavior for non-plain data objects.
 */
import { describe, expect, it } from "vitest";
import { isRecordLike } from "../../../src/adapters/common/guards.js";
import { buildTemplatedHttpRequest } from "../../../src/adapters/common/tools-common.js";

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
