/**
 * Retry configuration shared across networked components.
 *
 * Agent Spec treats RetryPolicy as a non-Component configuration object
 * (like LlmGenerationConfig): it has no id/name/componentType and is nested
 * inline in the components that carry it.
 */
import { z } from "zod";

/** Jitter methods for retry backoff */
export const RetryJitter = {
  EQUAL: "equal",
  FULL: "full",
  FULL_AND_EQUAL_FOR_THROTTLE: "full_and_equal_for_throttle",
  DECORRELATED: "decorrelated",
} as const;

export type RetryJitter = (typeof RetryJitter)[keyof typeof RetryJitter];

/** Parsed retry policy (all defaults applied). */
export interface RetryPolicy {
  maxAttempts: number;
  requestTimeout: number | null;
  initialRetryDelay: number;
  maxRetryDelay: number;
  backoffFactor: number;
  jitter: RetryJitter | null;
  serviceErrorRetryOnAny5xx: boolean;
  recoverableStatuses: Record<string, string[]>;
}

/** Retry policy input (every field defaulted, so all are optional). */
export interface RetryPolicyInput {
  maxAttempts?: number;
  requestTimeout?: number | null;
  initialRetryDelay?: number;
  maxRetryDelay?: number;
  backoffFactor?: number;
  jitter?: RetryJitter | null;
  serviceErrorRetryOnAny5xx?: boolean;
  recoverableStatuses?: Record<string, string[]>;
}

// The explicit annotation keeps the schema's declaration-emit type small:
// the ZodEffects-of-strict-object type is referenced by many component
// schemas and would otherwise blow up their inferred types.
export const RetryPolicySchema: z.ZodType<
  RetryPolicy,
  z.ZodTypeDef,
  RetryPolicyInput
> = z
  .object({
    /** Maximum number of retries (not counting the initial attempt). */
    maxAttempts: z.number().int().min(0).default(2),
    /** Per-attempt timeout in seconds (fractional values allowed). */
    requestTimeout: z.number().gt(0).nullish().default(null),
    /** Base delay (seconds) used for exponential backoff. */
    initialRetryDelay: z.number().min(0).default(1.0),
    /** Cap (seconds) on the backoff delay between two retries. */
    maxRetryDelay: z.number().min(0).default(8.0),
    /** Back-off factor controlling how retry delays grow between attempts. */
    backoffFactor: z.number().gt(0).default(2.0),
    /** Method to add randomness to the retry time (null disables jitter). */
    jitter: z
      .enum([
        RetryJitter.EQUAL,
        RetryJitter.FULL,
        RetryJitter.FULL_AND_EQUAL_FOR_THROTTLE,
        RetryJitter.DECORRELATED,
      ])
      .nullish()
      .default(RetryJitter.FULL_AND_EQUAL_FOR_THROTTLE),
    /** Whether to retry on all 5xx errors (network errors, except 501). */
    serviceErrorRetryOnAny5xx: z.boolean().default(true),
    /**
     * Additional statuses considered recoverable. Keys are HTTP status
     * strings (Agent Spec configurations are JSON, so object keys are
     * strings); they are emitted verbatim on the wire.
     */
    recoverableStatuses: z
      .record(z.array(z.string()))
      .default({ "409": [], "429": [] }),
  })
  .strict() // mirrors Python's extra="forbid"
  .superRefine((val, ctx) => {
    if (val.maxRetryDelay < val.initialRetryDelay) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message:
          "`max_retry_delay` must be greater than or equal to `initial_retry_delay`.",
      });
    }
  });

/**
 * camelCase -> exact wire key overrides for RetryPolicy fields whose Python
 * wire names the generic camelToSnake converter cannot produce
 * ("serviceErrorRetryOnAny5xx" would become "service_error_retry_on_any5xx"
 * instead of Python's "service_error_retry_on_any_5xx"). snakeToCamel
 * already maps the wire name back correctly, so only serialization needs it.
 */
export const RETRY_POLICY_WIRE_KEY_OVERRIDES: Readonly<
  Record<string, string>
> = {
  serviceErrorRetryOnAny5xx: "service_error_retry_on_any_5xx",
};
