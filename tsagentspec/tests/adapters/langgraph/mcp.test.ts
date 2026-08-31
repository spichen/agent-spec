/**
 * MCP conversion tests for the LangGraph adapter.
 *
 * Mirrors the offline-able behaviors of
 * `pyagentspec/tests/adapters/langgraph/mcp/test_mcp.py` and the MCP sections
 * of `_langgraphconverter.py` — transport mapping, the
 * `${transport.id}::${toolName}` registry cache, and MCPTool / MCPToolBox
 * conversion with `toolFilter` validation. `@langchain/mcp-adapters` is
 * mocked (vi.mock), so no MCP server or subprocess is ever started.
 *
 * Documented divergences exercised here (see mcp.ts header):
 * - mTLS transports are rejected (JS connections have no client-cert options);
 * - `sessionParameters.readTimeoutSeconds` maps to the stdio connection's
 *   `defaultToolTimeout` (Python wires it as the MCP session's per-request
 *   read timeout, stdio only);
 * - a missing MCPTool name raises a descriptive Error (Python raises a bare
 *   KeyError).
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  createMCPTool,
  createMCPToolBox,
  createMCPToolSpec,
  createRemoteTransport,
  createSSETransport,
  createSSEmTLSTransport,
  createStdioTransport,
  createStreamableHTTPTransport,
  createStreamableHTTPmTLSTransport,
  integerProperty,
  stringProperty,
  type ClientTransport,
} from "../../../src/index.js";
import {
  convertClientTransport,
  convertMcpTool,
  convertMcpToolbox,
  getOrCreateMcpTools,
} from "../../../src/adapters/langgraph/mcp.js";
import type { ToolRegistry } from "../../../src/adapters/langgraph/types.js";

const mocks = vi.hoisted(() => ({
  getTools: vi.fn<(...servers: string[]) => Promise<unknown[]>>(),
  constructorConfigs: [] as unknown[],
}));

vi.mock("@langchain/mcp-adapters", () => ({
  MultiServerMCPClient: class {
    constructor(config: unknown) {
      mocks.constructorConfigs.push(config);
    }

    getTools(...servers: string[]): Promise<unknown[]> {
      return mocks.getTools(...servers);
    }
  },
}));

/** Minimal structural stand-in for a loaded LangChain MCP tool. */
function fakeMcpTool(
  name: string,
  properties: Record<string, Record<string, unknown>> = {},
): { name: string; description: string; schema: Record<string, unknown> } {
  return {
    name,
    description: `${name} description`,
    schema: { title: name, type: "object", properties },
  };
}

function makeSseTransport(): ClientTransport {
  return createSSETransport({
    name: "my server",
    url: "https://example.com/sse",
  });
}

beforeEach(() => {
  mocks.getTools.mockReset();
  mocks.constructorConfigs.length = 0;
});

describe("convertClientTransport", () => {
  it("maps StdioTransport with command, args, env and cwd", () => {
    const transport = createStdioTransport({
      name: "stdio server",
      command: "uv",
      args: ["run", "server.py"],
      env: { API_KEY: "secret" },
      cwd: "/srv/mcp",
      sessionParameters: { readTimeoutSeconds: 42 },
    });
    // readTimeoutSeconds maps to the per-tool-call timeout, mirroring the
    // per-request read timeout Python passes via stdio session_kwargs.
    expect(convertClientTransport(transport)).toEqual({
      transport: "stdio",
      command: "uv",
      args: ["run", "server.py"],
      env: { API_KEY: "secret" },
      cwd: "/srv/mcp",
      defaultToolTimeout: 42000,
    });
  });

  it("omits env and cwd from stdio connections when unset", () => {
    const transport = createStdioTransport({
      name: "stdio server",
      command: "echo",
    });
    // The SDK defaults readTimeoutSeconds to 60 like Python's spec default.
    expect(convertClientTransport(transport)).toEqual({
      transport: "stdio",
      command: "echo",
      args: [],
      defaultToolTimeout: 60000,
    });
  });

  it("maps SSETransport with url and headers", () => {
    const transport = createSSETransport({
      name: "sse server",
      url: "https://example.com/sse",
      headers: { Authorization: "Bearer token" },
    });
    expect(convertClientTransport(transport)).toEqual({
      transport: "sse",
      url: "https://example.com/sse",
      headers: { Authorization: "Bearer token" },
    });
  });

  it("omits headers from remote connections when unset", () => {
    expect(convertClientTransport(makeSseTransport())).toEqual({
      transport: "sse",
      url: "https://example.com/sse",
    });
  });

  it("maps StreamableHTTPTransport to an http connection", () => {
    const transport = createStreamableHTTPTransport({
      name: "http server",
      url: "https://example.com/mcp",
      headers: { "X-Tenant": "t1" },
    });
    expect(convertClientTransport(transport)).toEqual({
      transport: "http",
      url: "https://example.com/mcp",
      headers: { "X-Tenant": "t1" },
    });
  });

  it("rejects mTLS transports", () => {
    const sseMtls = createSSEmTLSTransport({
      name: "mtls sse",
      url: "https://example.com/sse",
      keyFile: "client.key",
      certFile: "client.crt",
      caFile: "ca.crt",
    });
    expect(() => convertClientTransport(sseMtls)).toThrow(
      "The Agent Spec type 'SSEmTLSTransport' is not supported by the LangGraph TypeScript adapter yet.",
    );

    const httpMtls = createStreamableHTTPmTLSTransport({
      name: "mtls http",
      url: "https://example.com/mcp",
      keyFile: "client.key",
      certFile: "client.crt",
      caFile: "ca.crt",
    });
    expect(() => convertClientTransport(httpMtls)).toThrow(
      "The Agent Spec type 'StreamableHTTPmTLSTransport' is not supported by the LangGraph TypeScript adapter yet.",
    );
  });

  it("rejects unsupported transport types with the Python error text", () => {
    const remoteTransport = createRemoteTransport({
      name: "remote",
      url: "https://example.com/agent",
    });
    expect(() => convertClientTransport(remoteTransport)).toThrow(
      "Agent Spec ClientTransport 'RemoteTransport' is not supported yet.",
    );
  });
});

describe("getOrCreateMcpTools registry cache", () => {
  it("loads tools once and caches them under `${transport.id}::${toolName}`", async () => {
    const transport = makeSseTransport();
    const connection = convertClientTransport(transport);
    const registry: ToolRegistry = {};
    const fooza = fakeMcpTool("fooza_tool");
    const zwak = fakeMcpTool("zwak");
    mocks.getTools.mockResolvedValue([fooza, zwak]);

    const tools = await getOrCreateMcpTools(transport, connection, registry);

    expect(mocks.constructorConfigs).toEqual([
      { mcpServers: { [transport.id]: connection } },
    ]);
    expect(mocks.getTools).toHaveBeenCalledExactlyOnceWith(transport.id);
    expect(tools).toEqual({ fooza_tool: fooza, zwak: zwak });
    expect(registry).toEqual({
      [`${transport.id}::fooza_tool`]: fooza,
      [`${transport.id}::zwak`]: zwak,
    });
  });

  it("returns cached tools without reconnecting on later calls", async () => {
    const transport = makeSseTransport();
    const connection = convertClientTransport(transport);
    const registry: ToolRegistry = {};
    mocks.getTools.mockResolvedValue([fakeMcpTool("fooza_tool")]);

    const first = await getOrCreateMcpTools(transport, connection, registry);
    const second = await getOrCreateMcpTools(transport, connection, registry);

    expect(second).toEqual(first);
    expect(mocks.getTools).toHaveBeenCalledTimes(1);
    expect(mocks.constructorConfigs).toHaveLength(1);
  });

  it("returns pre-seeded registry entries without connecting at all", async () => {
    const transport = makeSseTransport();
    const cached = fakeMcpTool("fooza_tool");
    const registry: ToolRegistry = {
      [`${transport.id}::fooza_tool`]: cached,
      "unrelated::other_tool": fakeMcpTool("other_tool"),
    };

    const tools = await getOrCreateMcpTools(
      transport,
      convertClientTransport(transport),
      registry,
    );

    expect(tools).toEqual({ fooza_tool: cached });
    expect(mocks.getTools).not.toHaveBeenCalled();
    expect(mocks.constructorConfigs).toHaveLength(0);
  });

  it("raises the duplicate-tool error when a key appears while loading", async () => {
    const transport = makeSseTransport();
    const registry: ToolRegistry = {};
    const fooza = fakeMcpTool("fooza_tool");
    // Simulate a concurrent registration racing the load: the key exists by
    // the time the loaded tools are committed to the registry.
    mocks.getTools.mockImplementation(async () => {
      registry[`${transport.id}::fooza_tool`] = fakeMcpTool("fooza_tool");
      return [fooza];
    });

    await expect(
      getOrCreateMcpTools(transport, convertClientTransport(transport), registry),
    ).rejects.toThrow(
      "Trying to add the same tool twice; this might happen " +
        "when the tool is declared as both a standalone MCPTool and part of a MCPToolBox",
    );
  });

  it("raises when a loaded tool has no name", async () => {
    const transport = makeSseTransport();
    mocks.getTools.mockResolvedValue([{ description: "nameless" }]);

    await expect(
      getOrCreateMcpTools(transport, convertClientTransport(transport), {}),
    ).rejects.toThrow("Loaded a tool without a name attribute or __name__.");
  });
});

describe("convertMcpTool", () => {
  it("returns the exposed tool with the MCPTool's name", async () => {
    const transport = makeSseTransport();
    const fooza = fakeMcpTool("fooza_tool");
    mocks.getTools.mockResolvedValue([fooza, fakeMcpTool("zwak")]);
    const mcpTool = createMCPTool({
      name: "fooza_tool",
      clientTransport: transport,
    });

    const converted = await convertMcpTool(mcpTool, {});

    expect(converted).toBe(fooza);
  });

  it("raises when the named tool is not exposed by the server", async () => {
    const transport = makeSseTransport();
    mocks.getTools.mockResolvedValue([fakeMcpTool("zwak")]);
    const mcpTool = createMCPTool({
      name: "missing_tool",
      clientTransport: transport,
    });

    await expect(convertMcpTool(mcpTool, {})).rejects.toThrow(
      "MCP tool 'missing_tool' was not found in the tools exposed " +
        `by the MCP server for transport '${transport.id}'.`,
    );
  });

  it("shares the registry cache with a toolbox on the same transport", async () => {
    const transport = makeSseTransport();
    const fooza = fakeMcpTool("fooza_tool");
    const zwak = fakeMcpTool("zwak");
    mocks.getTools.mockResolvedValue([fooza, zwak]);
    const registry: ToolRegistry = {};

    const standalone = await convertMcpTool(
      createMCPTool({ name: "fooza_tool", clientTransport: transport }),
      registry,
    );
    const toolboxTools = await convertMcpToolbox(
      createMCPToolBox({ name: "box", clientTransport: transport }),
      registry,
    );

    // A single connection serves both conversions.
    expect(mocks.getTools).toHaveBeenCalledTimes(1);
    expect(standalone).toBe(fooza);
    expect(toolboxTools).toEqual([fooza, zwak]);
  });
});

describe("convertMcpToolbox", () => {
  it("returns all exposed tools when no filter is set", async () => {
    const transport = makeSseTransport();
    const fooza = fakeMcpTool("fooza_tool");
    const bwip = fakeMcpTool("bwip_tool");
    const zwak = fakeMcpTool("zwak");
    mocks.getTools.mockResolvedValue([fooza, bwip, zwak]);

    const tools = await convertMcpToolbox(
      createMCPToolBox({ name: "box", clientTransport: transport }),
      {},
    );

    expect(tools).toEqual([fooza, bwip, zwak]);
  });

  it("filters by name and spec entries, preserving the filter order", async () => {
    const transport = makeSseTransport();
    const fooza = fakeMcpTool("fooza_tool");
    const bwip = fakeMcpTool("bwip_tool");
    const zbuk = fakeMcpTool("zbuk_tool", {
      a: { title: "a", type: "integer" },
      b: { title: "b", type: "integer" },
    });
    mocks.getTools.mockResolvedValue([fooza, bwip, zbuk]);
    const toolbox = createMCPToolBox({
      name: "drop_box",
      clientTransport: transport,
      toolFilter: [
        createMCPToolSpec({
          name: "zbuk_tool",
          description: "something",
          inputs: [integerProperty({ title: "a" }), integerProperty({ title: "b" })],
        }),
        "bwip_tool",
      ],
    });

    const tools = await convertMcpToolbox(toolbox, {});

    expect(tools).toEqual([zbuk, bwip]);
  });

  it("raises a sorted Missing tools error for unknown filter names", async () => {
    const transport = makeSseTransport();
    mocks.getTools.mockResolvedValue([fakeMcpTool("fooza_tool")]);
    const toolbox = createMCPToolBox({
      name: "box",
      clientTransport: transport,
      toolFilter: [
        "z_missing",
        "fooza_tool",
        createMCPToolSpec({ name: "a_missing" }),
      ],
    });

    await expect(convertMcpToolbox(toolbox, {})).rejects.toThrow(
      "Missing tools: a_missing, z_missing",
    );
  });

  it("raises Missing tools for filter names that collide with Object.prototype keys", async () => {
    // Membership must be own-keys only: "constructor" must not resolve to
    // the inherited Object function and dodge the missing-tools error.
    const transport = makeSseTransport();
    mocks.getTools.mockResolvedValue([fakeMcpTool("fooza_tool")]);
    const toolbox = createMCPToolBox({
      name: "box",
      clientTransport: transport,
      toolFilter: ["constructor"],
    });

    await expect(convertMcpToolbox(toolbox, {})).rejects.toThrow(
      "Missing tools: constructor",
    );
  });

  it("accepts a spec whose input schemas match the remote tool", async () => {
    const transport = makeSseTransport();
    const zbuk = fakeMcpTool("zbuk_tool", {
      a: { title: "A", type: "integer" },
      b: { title: "b", type: "integer" },
    });
    mocks.getTools.mockResolvedValue([zbuk]);
    const toolbox = createMCPToolBox({
      name: "box",
      clientTransport: transport,
      toolFilter: [
        createMCPToolSpec({
          name: "zbuk_tool",
          inputs: [integerProperty({ title: "a" }), integerProperty({ title: "b" })],
        }),
      ],
    });

    await expect(convertMcpToolbox(toolbox, {})).resolves.toEqual([zbuk]);
  });

  it("does not reject a per-property type mismatch (bug-compatible with Python)", async () => {
    // Both SDKs pass {name: schema} maps to jsonSchemasHaveSameType, which
    // inspects only JSON-schema keys (type/items/properties/...), so the
    // property-level types are never actually compared. Mirrored verbatim
    // from Python for wire compatibility.
    const transport = makeSseTransport();
    const zbuk = fakeMcpTool("zbuk_tool", {
      a: { title: "a", type: "string" },
    });
    mocks.getTools.mockResolvedValue([zbuk]);
    const toolbox = createMCPToolBox({
      name: "box",
      clientTransport: transport,
      toolFilter: [
        createMCPToolSpec({
          name: "zbuk_tool",
          inputs: [integerProperty({ title: "a" })],
        }),
      ],
    });

    await expect(convertMcpToolbox(toolbox, {})).resolves.toEqual([zbuk]);
  });

  it("raises the Input descriptors mismatch error when the schema maps differ in type", async () => {
    // A property named like a JSON-schema keyword ("items" here) makes the
    // schema maps genuinely comparable, so the mismatch branch fires — in
    // Python exactly the same way.
    const transport = makeSseTransport();
    const zbuk = fakeMcpTool("zbuk_tool", {
      items: { title: "items", type: "string" },
    });
    mocks.getTools.mockResolvedValue([zbuk]);
    const spec = createMCPToolSpec({
      name: "zbuk_tool",
      inputs: [
        {
          title: "items",
          description: undefined,
          default: undefined,
          type: "integer",
          jsonSchema: { title: "items", type: "integer" },
        },
      ],
    });
    const toolbox = createMCPToolBox({
      name: "box",
      clientTransport: transport,
      toolFilter: [spec],
    });

    await expect(convertMcpToolbox(toolbox, {})).rejects.toThrow(
      "Input descriptors mismatch for tool 'zbuk_tool'.",
    );
  });

  it("raises when the remote tool schema is not a plain object", async () => {
    const transport = makeSseTransport();
    mocks.getTools.mockResolvedValue([
      { name: "zbuk_tool", description: "d", schema: undefined },
    ]);
    const toolbox = createMCPToolBox({
      name: "box",
      clientTransport: transport,
      toolFilter: [
        createMCPToolSpec({
          name: "zbuk_tool",
          inputs: [stringProperty({ title: "a" })],
        }),
      ],
    });

    await expect(convertMcpToolbox(toolbox, {})).rejects.toThrow(
      "Expected Langchain StructuredTool.args_schema to be a dict but got undefined",
    );
  });
});
