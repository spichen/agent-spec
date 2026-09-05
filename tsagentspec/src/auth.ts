/**
 * Auth configuration components (Agent Spec >= 26.1.2).
 *
 * Port of pyagentspec/auth.py: OAuthConfig / OAuthClientConfig are
 * Components; OAuthEndpoints and PKCEPolicy are plain model objects.
 */
import { z } from "zod";
import { ComponentBaseSchema } from "./component.js";

/**
 * Explicit OAuth endpoint configuration — non-Component model object.
 *
 * Use when endpoint discovery is not available or not desired.
 */
export const OAuthEndpointsSchema = z.object({
  /** Authorization endpoint where the user agent is redirected for login and consent. */
  authorizationEndpoint: z.string(),
  /** Token endpoint where authorization codes (and refresh tokens) are exchanged. */
  tokenEndpoint: z.string(),
  /** Optional endpoint for refresh token requests (token endpoint reused when absent). */
  refreshEndpoint: z.string().nullish(),
  /** Optional endpoint for token revocation. */
  revocationEndpoint: z.string().nullish(),
  /** Optional OIDC UserInfo endpoint. */
  userinfoEndpoint: z.string().nullish(),
});

export type OAuthEndpoints = z.infer<typeof OAuthEndpointsSchema>;

/** PKCE challenge methods */
export const PKCEMethod = {
  PLAIN: "plain",
  S256: "S256",
} as const;

export type PKCEMethod = (typeof PKCEMethod)[keyof typeof PKCEMethod];

/**
 * Policy configuration for Proof Key for Code Exchange (PKCE) —
 * non-Component model object.
 */
export const PKCEPolicySchema = z.object({
  /** If true, the runtime must refuse to proceed when PKCE cannot be used. */
  required: z.boolean().default(true),
  /** PKCE challenge method. Defaults to "S256". */
  method: z.enum([PKCEMethod.PLAIN, PKCEMethod.S256]).default(PKCEMethod.S256),
});

export type PKCEPolicy = z.infer<typeof PKCEPolicySchema>;

/** How the runtime selects OAuth scopes */
export const ScopePolicy = {
  USE_CHALLENGE_OR_SUPPORTED: "use_challenge_or_supported",
  FIXED: "fixed",
} as const;

export type ScopePolicy = (typeof ScopePolicy)[keyof typeof ScopePolicy];

/**
 * OAuth client identity / registration configuration (Component).
 *
 * Supports pre-registered clients (static clientId/clientSecret), Client ID
 * Metadata Documents (URL-formatted client id), and dynamic client
 * registration (RFC 7591).
 */
export const OAuthClientConfigSchema = ComponentBaseSchema.extend({
  componentType: z.literal("OAuthClientConfig"),
  /** Strategy used to obtain client identity. */
  type: z.enum([
    "pre_registered",
    "client_id_metadata_document",
    "dynamic_registration",
  ]),
  /** OAuth client identifier (pre-registered clients) — sensitive. */
  clientId: z.string().optional(),
  /** OAuth client secret (confidential pre-registered clients) — sensitive. */
  clientSecret: z.string().optional(),
  /** Token endpoint authentication method (e.g. "client_secret_basic"). */
  tokenEndpointAuthMethod: z.string().optional(),
  /** HTTPS URL used as the OAuth client_id for Client ID Metadata Documents — sensitive. */
  clientIdMetadataUrl: z.string().optional(),
  /** Optional dynamic registration endpoint. */
  registrationEndpoint: z.string().optional(),
});

export type OAuthClientConfig = z.infer<typeof OAuthClientConfigSchema>;

/**
 * Configure OAuth-based authentication for a tool or transport (Component).
 *
 * Supports discovery-based configuration (via `issuer`) and explicit
 * endpoints (via `endpoints`).
 */
export const OAuthConfigSchema = ComponentBaseSchema.extend({
  componentType: z.literal("OAuthConfig"),
  /** Authorization server issuer URL used for discovery (OIDC or RFC 8414). */
  issuer: z.string().optional(),
  /** Explicit OAuth endpoints, used directly instead of discovery when set. */
  endpoints: OAuthEndpointsSchema.optional(),
  /** OAuth client identity / registration configuration. */
  client: OAuthClientConfigSchema,
  /** Redirect (callback) URI registered with the authorization server. */
  redirectUri: z.string(),
  /** Requested scopes, space-delimited string or list of scope strings. */
  scopes: z.union([z.string(), z.array(z.string())]).optional(),
  /** How the runtime selects scopes. */
  scopePolicy: z
    .enum([ScopePolicy.USE_CHALLENGE_OR_SUPPORTED, ScopePolicy.FIXED])
    .optional(),
  /** PKCE policy; authorization code flows should typically require S256. */
  pkce: PKCEPolicySchema.optional(),
  /** Optional resource indicator value (RFC 8707). */
  resource: z.string().optional(),
});

export type OAuthConfig = z.infer<typeof OAuthConfigSchema>;

/** Union of auth configs (single member today, mirrors abstract AuthConfig). */
export type AuthConfig = OAuthConfig;

/**
 * Discriminated union of auth configs. Single member today — it mirrors
 * Python's abstract AuthConfig base so future auth schemes slot in here.
 * The explicit annotation keeps the declaration-emit type of the component
 * schemas embedding it (all remote MCP transports) small.
 */
export const AuthConfigUnion: z.ZodType<
  AuthConfig,
  z.ZodTypeDef,
  z.input<typeof OAuthConfigSchema>
> = z.discriminatedUnion("componentType", [OAuthConfigSchema]);

export function createOAuthClientConfig(opts: {
  name: string;
  type: "pre_registered" | "client_id_metadata_document" | "dynamic_registration";
  id?: string;
  description?: string;
  metadata?: Record<string, unknown>;
  clientId?: string;
  clientSecret?: string;
  tokenEndpointAuthMethod?: string;
  clientIdMetadataUrl?: string;
  registrationEndpoint?: string;
}): OAuthClientConfig {
  return Object.freeze(
    OAuthClientConfigSchema.parse({
      ...opts,
      componentType: "OAuthClientConfig" as const,
    }),
  );
}

export function createOAuthConfig(opts: {
  name: string;
  client: OAuthClientConfig;
  redirectUri: string;
  id?: string;
  description?: string;
  metadata?: Record<string, unknown>;
  issuer?: string;
  endpoints?: OAuthEndpoints;
  scopes?: string | string[];
  scopePolicy?: ScopePolicy;
  pkce?: z.input<typeof PKCEPolicySchema>;
  resource?: string;
}): OAuthConfig {
  return Object.freeze(
    OAuthConfigSchema.parse({
      ...opts,
      componentType: "OAuthConfig" as const,
    }),
  );
}
