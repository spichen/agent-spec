/**
 * Helpers for URL validation and optional allow-list handling in HTTP-based
 * components. Port of `pyagentspec.adapters._url_validation`.
 */
import { getPlaceholdersFromString } from "../../templating.js";

/**
 * Return the URL parts used for allow-list matching: scheme, netloc
 * (userinfo + host + port) and path (defaulting to "/").
 *
 * Matching intentionally considers only scheme, authority, and path. Query
 * parameters, URL params, and fragments are ignored.
 *
 * Python normalizes via pydantic `AnyUrl`; here the WHATWG `URL` parser is
 * used. Behavioral difference: WHATWG drops explicit default ports
 * (`http://x:80` -> host `x`) while `AnyUrl` keeps them, so matching is
 * slightly more lenient here when a URL or pattern spells out the scheme's
 * default port.
 */
export function getUrlMatchParts(
  url: string,
): [scheme: string, netloc: string, path: string] {
  const parsed = new URL(url);
  const scheme = parsed.protocol.endsWith(":")
    ? parsed.protocol.slice(0, -1)
    : parsed.protocol;
  const userinfo =
    parsed.username !== "" || parsed.password !== ""
      ? `${parsed.username}${parsed.password !== "" ? `:${parsed.password}` : ""}@`
      : "";
  const netloc = `${userinfo}${parsed.host}`;
  const path = parsed.pathname !== "" ? parsed.pathname : "/";
  return [scheme, netloc, path];
}

/** Check whether a URL matches one allow-list entry. */
export function matchesAllowListEntry(url: string, pattern: string): boolean {
  const [urlScheme, urlNetloc, urlPath] = getUrlMatchParts(url);
  const [patternScheme, patternNetloc, patternPath] = getUrlMatchParts(pattern);
  return (
    urlScheme === patternScheme &&
    urlNetloc === patternNetloc &&
    urlPath.startsWith(patternPath)
  );
}

/**
 * Return placeholders used in the URL destination part.
 *
 * The destination is limited to the scheme, host, and port. Placeholders
 * appearing only in path, query, or fragment are ignored.
 */
export function getUrlDestinationPlaceholderNames(url: string): string[] {
  const schemeSeparator = url.indexOf("://");
  let schemePart: string;
  let remainder: string;
  if (schemeSeparator !== -1) {
    schemePart = url.slice(0, schemeSeparator);
    remainder = url.slice(schemeSeparator + 3);
  } else {
    schemePart = "";
    remainder = url;
  }

  const authorityEndPositions = [
    remainder.indexOf("/"),
    remainder.indexOf("?"),
    remainder.indexOf("#"),
  ].filter((pos) => pos !== -1);
  const authorityEnd =
    authorityEndPositions.length > 0
      ? Math.min(...authorityEndPositions)
      : remainder.length;
  const authority = remainder.slice(0, authorityEnd);
  const atIndex = authority.lastIndexOf("@");
  const hostport = atIndex !== -1 ? authority.slice(atIndex + 1) : authority;
  return [
    ...new Set([
      ...getPlaceholdersFromString(schemePart),
      ...getPlaceholdersFromString(hostport),
    ]),
  ].sort();
}

/** Warn when a templated URL destination is used without an allow list. */
export function maybeWarnAboutUnrestrictedTemplatedUrl(
  url: string,
  urlAllowList: string[] | undefined,
  componentName: string,
): void {
  if (urlAllowList !== undefined) {
    return;
  }

  const placeholderNames = getUrlDestinationPlaceholderNames(url);
  if (placeholderNames.length === 0) {
    return;
  }

  const variableList = placeholderNames.map((name) => `\`${name}\``).join(", ");
  console.warn(
    `${componentName} uses placeholders in the URL destination (${variableList}) ` +
      "but no `url_allow_list` is configured. Keep the base URL developer-controlled and " +
      "template only path, query, or body values when possible.",
  );
}

/** Validate a URL against an optional allow list. */
export function validateUrlAgainstAllowList(
  url: string,
  urlAllowList: string[] | undefined,
): void {
  if (urlAllowList === undefined) {
    return;
  }

  if (urlAllowList.some((pattern) => matchesAllowListEntry(url, pattern))) {
    return;
  }

  throw new Error(
    "Requested URL is not in allowed list. " +
      "Please contact the application administrator to help adding your URL to the list.",
  );
}
