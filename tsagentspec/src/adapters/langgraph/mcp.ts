/**
 * MCP tool conversion for the LangGraph adapter.
 *
 * Port of the MCP sections of
 * `pyagentspec.adapters.langgraph._langgraphconverter` (transport mapping,
 * tool loading and registry caching, MCPTool / MCPToolBox conversion) on top
 * of `@langchain/mcp-adapters`.
 *
 * Divergences from Python (see the adapter README):
 * - mTLS transports (SSEmTLSTransport, StreamableHTTPmTLSTransport) are not
 *   supported: `@langchain/mcp-adapters` connections have no TLS client-cert
 *   options (Python passes an httpx client factory).
 * - `sessionParameters.readTimeoutSeconds` maps to the stdio connection's
 *   `defaultToolTimeout` (per tool call): Python passes it as the MCP
 *   session's per-request read timeout via stdio `session_kwargs` only, so
 *   only the tool-call path is covered on both sides.
 * - Tools are loaded through a `MultiServerMCPClient` that keeps its
 *   connection open for the lifetime of the loaded tools (Python opens a
 *   fresh MCP session per tool call).
 * - No tracing callbacks are attached to loaded tools (tracing is a no-op
 *   seam in v1).
 */
import type { StructuredToolInterface } from "@langchain/core/tools";
import type { Connection } from "@langchain/mcp-adapters";
import type { ClientTransport, MCPTool, MCPToolSpec } from "../../mcp/index.js";
import type { JsonSchemaValue } from "../../property.js";
import type { MCPToolBox } from "../../tools/index.js";
import { importOptionalPeer, jsonSchemasHaveSameType } from "../common/index.js";
import type { ToolRegistry } from "./types.js";

/**
 * Convert an AgentSpec MCP client transport into a `@langchain/mcp-adapters`
 * connection.
 *
 * StdioTransport maps to a stdio connection, SSETransport to an "sse"
 * connection and StreamableHTTPTransport to an "http" connection (static
 * headers included). mTLS transports are not supported yet.
 */
export function convertClientTransport(
  agentspecTransport: ClientTransport,
): Connection {
  switch (agentspecTransport.componentType) {
    case "StdioTransport":
      return {
        transport: "stdio",
        command: agentspecTransport.command,
        args: agentspecTransport.args,
        ...(agentspecTransport.env !== undefined
          ? { env: agentspecTransport.env }
          : {}),
        ...(agentspecTransport.cwd !== undefined
          ? { cwd: agentspecTransport.cwd }
          : {}),
        // Python wires readTimeoutSeconds as the MCP session's per-request
        // read timeout (stdio only); defaultToolTimeout is the JS analogue
        // for the tool-call path.
        ...(agentspecTransport.sessionParameters?.readTimeoutSeconds !==
        undefined
          ? {
              defaultToolTimeout:
                agentspecTransport.sessionParameters.readTimeoutSeconds * 1000,
            }
          : {}),
      };
    case "SSETransport":
      return {
        transport: "sse",
        url: agentspecTransport.url,
        ...(agentspecTransport.headers !== undefined
          ? { headers: agentspecTransport.headers }
          : {}),
      };
    case "StreamableHTTPTransport":
      return {
        transport: "http",
        url: agentspecTransport.url,
        ...(agentspecTransport.headers !== undefined
          ? { headers: agentspecTransport.headers }
          : {}),
      };
    case "SSEmTLSTransport":
    case "StreamableHTTPmTLSTransport":
      throw new Error(
        `The Agent Spec type '${agentspecTransport.componentType}' is not supported by the LangGraph TypeScript adapter yet.`,
      );
    default: {
      const componentType = (agentspecTransport as { componentType: string })
        .componentType;
      throw new Error(
        `Agent Spec ClientTransport '${componentType}' is not supported yet.`,
      );
    }
  }
}

function getSessionToolsFromToolRegistry(
  toolRegistry: ToolRegistry,
  connPrefix: string,
): Record<string, StructuredToolInterface> {
  const sessionTools: Record<string, StructuredToolInterface> = {};
  for (const [key, value] of Object.entries(toolRegistry)) {
    if (key.startsWith(connPrefix)) {
      sessionTools[key.slice(connPrefix.length)] =
        value as StructuredToolInterface;
    }
  }
  return sessionTools;
}

function addSessionToolsToRegistry(
  toolRegistry: ToolRegistry,
  tools: StructuredToolInterface[],
  connPrefix: string,
): void {
  // Prepare a staged mapping so we can insert all-or-nothing.
  const staged: Record<string, StructuredToolInterface> = {};
  for (const loadedTool of tools) {
    const toolName = (loadedTool as { name?: unknown }).name;
    if (typeof toolName !== "string" || toolName.length === 0) {
      throw new Error("Loaded a tool without a name attribute or __name__.");
    }
    const key = `${connPrefix}${toolName}`;
    // Do not overwrite an existing entry if present.
    if (Object.hasOwn(toolRegistry, key)) {
      throw new Error(
        "Trying to add the same tool twice; this might happen " +
          "when the tool is declared as both a standalone MCPTool and part of a MCPToolBox",
      );
    }
    staged[key] = loadedTool;
  }
  // Commit staged entries.
  Object.assign(toolRegistry, staged);
}

/**
 * Load the MCP tools exposed by a client transport and cache them in the
 * tool registry under `${clientTransport.id}::${toolName}` keys.
 *
 * If any tools are already present in the registry for this transport, they
 * are returned without reloading; otherwise all tools are loaded and inserted
 * atomically. Returns a map of tool name to LangChain tool.
 */
export async function getOrCreateMcpTools(
  clientTransport: ClientTransport,
  connection: Connection,
  toolRegistry: ToolRegistry,
): Promise<Record<string, StructuredToolInterface>> {
  const connPrefix = `${clientTransport.id}::`;
  const existing = getSessionToolsFromToolRegistry(toolRegistry, connPrefix);
  if (Object.keys(existing).length > 0) {
    return existing;
  }

  const { MultiServerMCPClient } = await importOptionalPeer(
    () => import("@langchain/mcp-adapters"),
    "@langchain/mcp-adapters",
    "preload MCP tools",
    "remove MCP tools from the spec.",
  );
  const serverName = clientTransport.id;
  // The client stays referenced by the loaded tools; it is intentionally not
  // closed here (closing it would break later tool invocations).
  const client = new MultiServerMCPClient({
    mcpServers: { [serverName]: connection },
  });
  const tools = await client.getTools(serverName);

  addSessionToolsToRegistry(toolRegistry, tools, connPrefix);

  return getSessionToolsFromToolRegistry(toolRegistry, connPrefix);
}

/** Shallow-copy a JSON schema, lowercasing its `title` when present. */
function normalizeTitle(schema: JsonSchemaValue): JsonSchemaValue {
  const out: JsonSchemaValue = { ...schema };
  if (typeof out["title"] === "string") {
    out["title"] = out["title"].toLowerCase();
  }
  return out;
}

function areMcpToolSpecAndLangchainSchemasEqual(
  mcpSpec: MCPToolSpec,
  langchainTool: StructuredToolInterface,
): boolean {
  const argsSchema = (langchainTool as { schema?: unknown }).schema;
  if (
    typeof argsSchema !== "object" ||
    argsSchema === null ||
    Array.isArray(argsSchema)
  ) {
    throw new Error(
      `Expected Langchain StructuredTool.args_schema to be a dict but got ${typeof argsSchema}`,
    );
  }
  const agentspecJsonSchemas: JsonSchemaValue = {};
  for (const input of mcpSpec.inputs ?? []) {
    agentspecJsonSchemas[String(input.jsonSchema["title"])] = normalizeTitle(
      input.jsonSchema,
    );
  }
  const remoteProperties =
    ((argsSchema as JsonSchemaValue)["properties"] as
      | Record<string, JsonSchemaValue>
      | undefined) ?? {};
  const langchainJsonSchemas: JsonSchemaValue = {};
  for (const [key, value] of Object.entries(remoteProperties)) {
    langchainJsonSchemas[key] = normalizeTitle(value);
  }
  return jsonSchemasHaveSameType(agentspecJsonSchemas, langchainJsonSchemas);
}

/**
 * Convert an AgentSpec MCPTool: load (or reuse from the registry cache) the
 * tools exposed by its transport and return the one with the tool's name.
 *
 * An already-converted connection may be passed to reuse the converter's
 * memoized transport conversion.
 */
export async function convertMcpTool(
  agentspecMcpTool: MCPTool,
  toolRegistry: ToolRegistry,
  connection?: Connection,
): Promise<StructuredToolInterface> {
  const resolvedConnection =
    connection ?? convertClientTransport(agentspecMcpTool.clientTransport);
  const exposedTools = await getOrCreateMcpTools(
    agentspecMcpTool.clientTransport,
    resolvedConnection,
    toolRegistry,
  );
  const exposedTool = exposedTools[agentspecMcpTool.name];
  if (exposedTool === undefined) {
    // Python raises a bare KeyError here.
    throw new Error(
      `MCP tool '${agentspecMcpTool.name}' was not found in the tools exposed ` +
        `by the MCP server for transport '${agentspecMcpTool.clientTransport.id}'.`,
    );
  }
  return exposedTool;
}

/**
 * Convert an AgentSpec MCPToolBox into the list of LangChain tools exposed by
 * its transport, applying the toolbox's `toolFilter`.
 *
 * Filter entries may be tool names or MCPToolSpec objects; specs are
 * validated against the remote tool's argument schema. Without a filter, all
 * tools are returned. An already-converted connection may be passed to reuse
 * the converter's memoized transport conversion.
 */
export async function convertMcpToolbox(
  agentspecMcpToolbox: MCPToolBox,
  toolRegistry: ToolRegistry,
  connection?: Connection,
): Promise<StructuredToolInterface[]> {
  const resolvedConnection =
    connection ?? convertClientTransport(agentspecMcpToolbox.clientTransport);
  const remoteTools = await getOrCreateMcpTools(
    agentspecMcpToolbox.clientTransport,
    resolvedConnection,
    toolRegistry,
  );
  // Normalize filter to name -> MCPToolSpec | null (null when the filter
  // entry is a plain string).
  const filterMap = new Map<string, MCPToolSpec | null>();
  for (const filterEntry of agentspecMcpToolbox.toolFilter ?? []) {
    if (typeof filterEntry === "string") {
      filterMap.set(filterEntry, null);
    } else {
      filterMap.set(filterEntry.name, filterEntry);
    }
  }
  // If no filter provided, return all tools.
  if (filterMap.size === 0) {
    return Object.values(remoteTools);
  }
  // Find missing by name first (own-keys membership like Python's dict, so
  // filter names like "constructor" cannot resolve to inherited functions).
  const missing = [...filterMap.keys()]
    .filter((name) => !Object.hasOwn(remoteTools, name))
    .sort();
  if (missing.length > 0) {
    throw new Error("Missing tools: " + missing.join(", "));
  }
  // Validate specs (when provided) and collect tools in filter order.
  for (const [name, spec] of filterMap) {
    const remoteTool = remoteTools[name]!;
    if (
      spec !== null &&
      !areMcpToolSpecAndLangchainSchemasEqual(spec, remoteTool)
    ) {
      throw new Error(
        `Input descriptors mismatch for tool '${spec.name}'.\n` +
          `Local: ${JSON.stringify(spec)}\n` +
          `Remote: ${JSON.stringify((remoteTool as { schema?: unknown }).schema)}`,
      );
    }
  }
  return [...filterMap.keys()].map((name) => remoteTools[name]!);
}
