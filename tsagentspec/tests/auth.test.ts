import { describe, it, expect } from "vitest";
import {
  AgentSpecSerializer,
  AgentSpecDeserializer,
  AgentSpecVersion,
  AuthConfigUnion,
  PKCEMethod,
  ScopePolicy,
  createOAuthClientConfig,
  createOAuthConfig,
  createSSETransport,
  type OAuthConfig,
  type SSETransport,
} from "../src/index.js";

function makeOAuthConfig(): OAuthConfig {
  return createOAuthConfig({
    id: "oauth",
    name: "OAuth",
    issuer: "https://issuer.example.com",
    endpoints: {
      authorizationEndpoint: "https://issuer.example.com/auth",
      tokenEndpoint: "https://issuer.example.com/token",
    },
    client: createOAuthClientConfig({
      id: "client",
      name: "OAuthClientConfig",
      type: "pre_registered",
      clientId: "client_id",
      clientSecret: "client_secret",
    }),
    redirectUri: "https://app.example.com/callback",
    scopes: ["openid", "profile"],
    pkce: { required: true, method: PKCEMethod.S256 },
  });
}

function makeTransportWithOAuth(): SSETransport {
  return createSSETransport({
    id: "transport",
    name: "SSETransport",
    url: "https://mcp.example.com",
    auth: makeOAuthConfig(),
  });
}

describe("OAuthClientConfig", () => {
  it("should create a pre-registered client", () => {
    const client = createOAuthClientConfig({
      name: "client",
      type: "pre_registered",
      clientId: "id",
      clientSecret: "secret",
      tokenEndpointAuthMethod: "client_secret_basic",
    });
    expect(client.componentType).toBe("OAuthClientConfig");
    expect(client.type).toBe("pre_registered");
    expect(client.clientId).toBe("id");
    expect(client.clientSecret).toBe("secret");
    expect(client.tokenEndpointAuthMethod).toBe("client_secret_basic");
    expect(Object.isFrozen(client)).toBe(true);
  });

  it("should create a dynamic-registration client", () => {
    const client = createOAuthClientConfig({
      name: "client",
      type: "dynamic_registration",
      registrationEndpoint: "https://issuer.example.com/register",
    });
    expect(client.type).toBe("dynamic_registration");
    expect(client.registrationEndpoint).toBe(
      "https://issuer.example.com/register",
    );
  });

  it("should reject an unknown client type", () => {
    expect(() =>
      createOAuthClientConfig({
        name: "client",
        type: "implicit" as unknown as "pre_registered",
      }),
    ).toThrow();
  });
});

describe("OAuthConfig", () => {
  it("should create with endpoints, client, scopes, and pkce", () => {
    const oauth = makeOAuthConfig();
    expect(oauth.componentType).toBe("OAuthConfig");
    expect(oauth.issuer).toBe("https://issuer.example.com");
    expect(oauth.endpoints?.authorizationEndpoint).toBe(
      "https://issuer.example.com/auth",
    );
    expect(oauth.client.type).toBe("pre_registered");
    expect(oauth.redirectUri).toBe("https://app.example.com/callback");
    expect(oauth.scopes).toEqual(["openid", "profile"]);
    expect(oauth.pkce).toEqual({ required: true, method: "S256" });
    expect(Object.isFrozen(oauth)).toBe(true);
  });

  it("should default pkce required/method when given an empty policy", () => {
    const oauth = createOAuthConfig({
      name: "OAuth",
      client: createOAuthClientConfig({ name: "c", type: "pre_registered" }),
      redirectUri: "https://app.example.com/callback",
      pkce: {},
    });
    expect(oauth.pkce).toEqual({ required: true, method: PKCEMethod.S256 });
  });

  it("should accept scopes as a space-delimited string", () => {
    const oauth = createOAuthConfig({
      name: "OAuth",
      client: createOAuthClientConfig({ name: "c", type: "pre_registered" }),
      redirectUri: "https://app.example.com/callback",
      scopes: "openid profile",
      scopePolicy: ScopePolicy.FIXED,
    });
    expect(oauth.scopes).toBe("openid profile");
    expect(oauth.scopePolicy).toBe("fixed");
  });

  it("should be accepted by AuthConfigUnion", () => {
    const oauth = makeOAuthConfig();
    const parsed = AuthConfigUnion.parse(oauth);
    expect(parsed.componentType).toBe("OAuthConfig");
  });
});

describe("OAuth serialization on remote transports", () => {
  it("should serialize the transport with a nested OAuthConfig", () => {
    const serializer = new AgentSpecSerializer();
    const json = serializer.toJson(makeTransportWithOAuth()) as string;
    const dict = JSON.parse(json);

    expect(dict["component_type"]).toBe("SSETransport");
    const auth = dict["auth"] as Record<string, unknown>;
    expect(auth["component_type"]).toBe("OAuthConfig");
    expect(auth["issuer"]).toBe("https://issuer.example.com");
    expect(auth["redirect_uri"]).toBe("https://app.example.com/callback");
    expect(auth["scopes"]).toEqual(["openid", "profile"]);
    expect(auth["endpoints"]).toEqual({
      authorization_endpoint: "https://issuer.example.com/auth",
      token_endpoint: "https://issuer.example.com/token",
    });
    expect(auth["pkce"]).toEqual({ required: true, method: "S256" });
    const client = auth["client"] as Record<string, unknown>;
    expect(client["component_type"]).toBe("OAuthClientConfig");
    expect(client["type"]).toBe("pre_registered");
  });

  it("should redact the client secrets from serialized output", () => {
    const serializer = new AgentSpecSerializer();
    const json = serializer.toJson(makeTransportWithOAuth()) as string;
    const dict = JSON.parse(json);
    const client = (dict["auth"] as Record<string, unknown>)[
      "client"
    ] as Record<string, unknown>;

    expect("client_id" in client).toBe(false);
    expect("client_secret" in client).toBe(false);
    expect("client_id_metadata_url" in client).toBe(false);
    expect(json.includes("client_secret")).toBe(false);
  });

  it("should round-trip the transport with auth intact", () => {
    const serializer = new AgentSpecSerializer();
    const deserializer = new AgentSpecDeserializer();
    const transport = makeTransportWithOAuth();

    const json = serializer.toJson(transport) as string;
    const loaded = deserializer.fromJson(json) as SSETransport;

    expect(loaded.componentType).toBe("SSETransport");
    expect(loaded.auth).toBeDefined();
    expect(loaded.auth!.componentType).toBe("OAuthConfig");
    expect(loaded.auth!.client.type).toBe("pre_registered");
    expect(loaded.auth!.endpoints).toEqual({
      authorizationEndpoint: "https://issuer.example.com/auth",
      tokenEndpoint: "https://issuer.example.com/token",
    });
    expect(loaded.auth!.pkce).toEqual({ required: true, method: "S256" });
    expect(loaded.auth!.scopes).toEqual(["openid", "profile"]);
  });

  it("should serialize stably (serialize -> parse -> re-serialize)", () => {
    const serializer = new AgentSpecSerializer();
    const deserializer = new AgentSpecDeserializer();
    const transport = makeTransportWithOAuth();

    const json = serializer.toJson(transport) as string;
    const reserialized = serializer.toJson(
      deserializer.fromJson(json) as SSETransport,
    ) as string;
    expect(reserialized).toBe(json);
  });

  it("should throw when serializing at a version before 26.1.2", () => {
    const serializer = new AgentSpecSerializer();
    expect(() =>
      serializer.toJson(makeTransportWithOAuth(), {
        agentspecVersion: AgentSpecVersion.V26_1_0,
      }),
    ).toThrow(/Invalid agentspec_version.*26\.1\.0.*26\.1\.2/);
  });

  it("should serialize successfully at 26.1.2", () => {
    const serializer = new AgentSpecSerializer();
    const json = serializer.toJson(makeTransportWithOAuth(), {
      agentspecVersion: AgentSpecVersion.V26_1_2,
    }) as string;
    const dict = JSON.parse(json);
    expect(dict["agentspec_version"]).toBe("26.1.2");
    expect((dict["auth"] as Record<string, unknown>)["component_type"]).toBe(
      "OAuthConfig",
    );
  });
});
