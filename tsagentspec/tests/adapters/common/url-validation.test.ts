/**
 * Tests for the shared URL validation and allow-list helpers.
 *
 * Ports the behavior of `pyagentspec.adapters._url_validation`: matching
 * considers scheme + netloc exactly and path as a prefix, query/fragment are
 * ignored, and templated URL destinations without an allow list warn (via
 * `console.warn` here, `warnings.warn` in Python).
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  getUrlDestinationPlaceholderNames,
  getUrlMatchParts,
  matchesAllowListEntry,
  maybeWarnAboutUnrestrictedTemplatedUrl,
  validateUrlAgainstAllowList,
} from "../../../src/adapters/common/url-validation.js";

const ALLOW_LIST_ERROR =
  "Requested URL is not in allowed list. " +
  "Please contact the application administrator to help adding your URL to the list.";

describe("getUrlMatchParts", () => {
  it("splits scheme, netloc and path", () => {
    expect(getUrlMatchParts("https://example.com/api/x")).toEqual([
      "https",
      "example.com",
      "/api/x",
    ]);
  });

  it("defaults the path to /", () => {
    expect(getUrlMatchParts("http://example.com")).toEqual([
      "http",
      "example.com",
      "/",
    ]);
  });

  it("keeps non-default ports and userinfo in the netloc", () => {
    expect(getUrlMatchParts("http://user:pw@example.com:8080/p")).toEqual([
      "http",
      "user:pw@example.com:8080",
      "/p",
    ]);
    expect(getUrlMatchParts("http://user@example.com/p")).toEqual([
      "http",
      "user@example.com",
      "/p",
    ]);
  });
});

describe("matchesAllowListEntry", () => {
  it("matches exact scheme + host with path prefix", () => {
    expect(
      matchesAllowListEntry(
        "https://allowed.example.com/api/value",
        "https://allowed.example.com/api/",
      ),
    ).toBe(true);
  });

  it("ignores query parameters and fragments", () => {
    expect(
      matchesAllowListEntry(
        "https://allowed.example.com/api/value?x=1&y=2#frag",
        "https://allowed.example.com/api/",
      ),
    ).toBe(true);
  });

  it("rejects a different host", () => {
    expect(
      matchesAllowListEntry(
        "https://blocked.example.com/api/value",
        "https://allowed.example.com/api/",
      ),
    ).toBe(false);
  });

  it("rejects a different scheme", () => {
    expect(
      matchesAllowListEntry(
        "http://allowed.example.com/api/value",
        "https://allowed.example.com/api/",
      ),
    ).toBe(false);
  });

  it("rejects a different port", () => {
    expect(
      matchesAllowListEntry(
        "https://allowed.example.com:8443/api/value",
        "https://allowed.example.com/api/",
      ),
    ).toBe(false);
  });

  it("rejects a path outside the pattern prefix", () => {
    expect(
      matchesAllowListEntry(
        "https://allowed.example.com/other/value",
        "https://allowed.example.com/api/",
      ),
    ).toBe(false);
  });
});

describe("validateUrlAgainstAllowList", () => {
  it("allows everything when no allow list is configured", () => {
    expect(() =>
      validateUrlAgainstAllowList("https://anything.example.com/x", undefined),
    ).not.toThrow();
  });

  it("passes when any entry matches", () => {
    expect(() =>
      validateUrlAgainstAllowList("https://allowed.example.com/api/value", [
        "https://other.example.com/",
        "https://allowed.example.com/api/",
      ]),
    ).not.toThrow();
  });

  it("throws the Python error text on mismatch", () => {
    expect(() =>
      validateUrlAgainstAllowList("https://blocked.example.com/api/value", [
        "https://allowed.example.com/api/",
      ]),
    ).toThrow(ALLOW_LIST_ERROR);
  });

  it("throws on an empty allow list", () => {
    expect(() =>
      validateUrlAgainstAllowList("https://allowed.example.com/api/value", []),
    ).toThrow(ALLOW_LIST_ERROR);
  });
});

describe("getUrlDestinationPlaceholderNames", () => {
  it("finds placeholders in the host and port", () => {
    expect(
      getUrlDestinationPlaceholderNames("https://{{host}}:{{port}}/api"),
    ).toEqual(["host", "port"]);
  });

  it("finds placeholders in the scheme", () => {
    expect(getUrlDestinationPlaceholderNames("{{scheme}}://x.com/api")).toEqual([
      "scheme",
    ]);
  });

  it("ignores placeholders in path, query and fragment", () => {
    expect(
      getUrlDestinationPlaceholderNames(
        "https://example.com/{{path}}?q={{query}}#{{frag}}",
      ),
    ).toEqual([]);
  });

  it("ignores placeholders in the userinfo", () => {
    expect(
      getUrlDestinationPlaceholderNames("https://{{user}}@{{host}}/x"),
    ).toEqual(["host"]);
  });

  it("returns sorted unique names", () => {
    expect(
      getUrlDestinationPlaceholderNames("https://{{b}}.{{a}}.{{b}}/x"),
    ).toEqual(["a", "b"]);
  });
});

describe("maybeWarnAboutUnrestrictedTemplatedUrl", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("warns with the component name when the destination is templated and no allow list is set", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    maybeWarnAboutUnrestrictedTemplatedUrl(
      "https://{{host}}/api/value",
      undefined,
      "RemoteTool `lookup`",
    );
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy).toHaveBeenCalledWith(
      "RemoteTool `lookup` uses placeholders in the URL destination (`host`) " +
        "but no `url_allow_list` is configured. Keep the base URL developer-controlled and " +
        "template only path, query, or body values when possible.",
    );
  });

  it("does not warn when an allow list is configured (even an empty one)", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    maybeWarnAboutUnrestrictedTemplatedUrl(
      "https://{{host}}/api/value",
      [],
      "RemoteTool `lookup`",
    );
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it("does not warn when placeholders only appear outside the destination", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    maybeWarnAboutUnrestrictedTemplatedUrl(
      "https://example.com/{{path}}?q={{query}}",
      undefined,
      "RemoteTool `lookup`",
    );
    expect(warnSpy).not.toHaveBeenCalled();
  });
});
