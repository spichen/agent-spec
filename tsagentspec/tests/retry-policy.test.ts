import { describe, it, expect } from "vitest";
import { RetryPolicySchema, RetryJitter } from "../src/index.js";

describe("RetryPolicy", () => {
  it("should apply the Python defaults when parsing an empty object", () => {
    const policy = RetryPolicySchema.parse({});
    expect(policy.maxAttempts).toBe(2);
    expect(policy.requestTimeout).toBeNull();
    expect(policy.initialRetryDelay).toBe(1.0);
    expect(policy.maxRetryDelay).toBe(8.0);
    expect(policy.backoffFactor).toBe(2.0);
    expect(policy.jitter).toBe(RetryJitter.FULL_AND_EQUAL_FOR_THROTTLE);
    expect(policy.serviceErrorRetryOnAny5xx).toBe(true);
    expect(policy.recoverableStatuses).toEqual({ "409": [], "429": [] });
  });

  it("should always carry all eight fields after parsing", () => {
    // Python's model_dump emits every field (None as null); the parsed
    // object must therefore materialize all keys, defaults included.
    const policy = RetryPolicySchema.parse({ maxAttempts: 3 });
    expect(Object.keys(policy).sort()).toEqual(
      [
        "backoffFactor",
        "initialRetryDelay",
        "jitter",
        "maxAttempts",
        "maxRetryDelay",
        "recoverableStatuses",
        "requestTimeout",
        "serviceErrorRetryOnAny5xx",
      ].sort(),
    );
  });

  it("should accept a full configuration", () => {
    const policy = RetryPolicySchema.parse({
      maxAttempts: 5,
      requestTimeout: 0.5,
      initialRetryDelay: 0.25,
      maxRetryDelay: 30,
      backoffFactor: 3,
      jitter: RetryJitter.DECORRELATED,
      serviceErrorRetryOnAny5xx: false,
      recoverableStatuses: { "408": [], "429": ["Retry-After"] },
    });
    expect(policy.maxAttempts).toBe(5);
    expect(policy.requestTimeout).toBe(0.5);
    expect(policy.jitter).toBe("decorrelated");
    expect(policy.serviceErrorRetryOnAny5xx).toBe(false);
    expect(policy.recoverableStatuses).toEqual({
      "408": [],
      "429": ["Retry-After"],
    });
  });

  it("should reject unknown fields (extra='forbid')", () => {
    expect(() =>
      RetryPolicySchema.parse({ maxAttmpts: 7 }),
    ).toThrow(/[Uu]nrecognized key/);
  });

  it.each([
    ["maxAttempts", -1],
    ["maxAttempts", 1.5],
    ["requestTimeout", 0.0],
    ["requestTimeout", -1.0],
    ["initialRetryDelay", -0.1],
    ["maxRetryDelay", -0.1],
    ["backoffFactor", 0.0],
    ["backoffFactor", -1.0],
  ])("should reject invalid numeric value %s=%s", (fieldName, value) => {
    expect(() => RetryPolicySchema.parse({ [fieldName]: value })).toThrow();
  });

  it("should reject maxRetryDelay lower than initialRetryDelay", () => {
    expect(() =>
      RetryPolicySchema.parse({ initialRetryDelay: 2.0, maxRetryDelay: 1.0 }),
    ).toThrow(
      "`max_retry_delay` must be greater than or equal to `initial_retry_delay`.",
    );
  });

  it("should enforce the delay bound against defaults too", () => {
    // initialRetryDelay 10 against the default maxRetryDelay of 8 must fail.
    expect(() =>
      RetryPolicySchema.parse({ initialRetryDelay: 10 }),
    ).toThrow(
      "`max_retry_delay` must be greater than or equal to `initial_retry_delay`.",
    );
  });

  it("should accept an explicit null jitter (no jitter)", () => {
    const policy = RetryPolicySchema.parse({ jitter: null });
    expect(policy.jitter).toBeNull();
  });
});
