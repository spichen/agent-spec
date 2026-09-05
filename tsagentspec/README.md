# agentspec TypeScript SDK

TypeScript SDK for the [Oracle Open Agent Specification](https://github.com/oracle/agent-spec).

## Supported component types

These types round-trip correctly between JSON/YAML and TypeScript objects:

- **Agents**: `Agent`, `Swarm`, `ManagerWorkers`, `RemoteAgent`, `SpecializedAgent`, `A2AAgent`
- **Flows**: `Flow`, `StartNode`, `EndNode`, `LlmNode`, `ToolNode`, `AgentNode`, `FlowNode`, `BranchingNode`, `MapNode`, `ParallelMapNode`, `ParallelFlowNode`, `ApiNode`, `InputMessageNode`, `OutputMessageNode`, `CatchExceptionNode`
- **Tools**: `ServerTool`, `ClientTool`, `RemoteTool`, `BuiltinTool`, `MCPTool`
- **LLM configs**: `OpenAiCompatibleConfig`, `OllamaConfig`, `VllmConfig`, `OpenAiConfig`, `OciGenAiConfig`
- **MCP**: `MCPToolBox`, `StdioTransport`, `SSETransport`, `StreamableHTTPTransport` (and mTLS variants)
- **Datastores**: `InMemoryCollectionDatastore`, `OracleDatabaseDatastore`, `PostgresDatabaseDatastore`
- **Other**: `ControlFlowEdge`, `DataFlowEdge`, `MessageSummarizationTransform`, `ConversationSummarizationTransform`

### Fixture compatibility

`tests/repo-fixtures.test.ts` is the CI contract: a curated list of 61 configs known to round-trip with the current SDK. Not every example or historical config file in the repo is covered here. Files using types outside the list above (e.g. `howto_swarm`, `howto_a2aagent`) are not yet supported.

## Installation

This package is not published to npm. Install from source:

```bash
git clone https://github.com/oracle/agent-spec.git
cd agent-spec/tsagentspec
npm ci
npm run build
```

To consume it from another local project, add it as a local path dependency in that project's `package.json`:

```json
"dependencies": {
  "agentspec": "file:../agent-spec/tsagentspec"
}
```

## Usage

See the [examples](./examples/README.md) directory.

## LangGraph adapter

The `agentspec/adapters/langgraph` subpath converts Agent Spec configurations into runnable [LangGraph JS](https://langchain-ai.github.io/langgraphjs/) objects and back, mirroring the Python `pyagentspec.adapters.langgraph` adapter.

### Installation

The LangChain packages are optional peer dependencies of this SDK; install the ones your configurations need:

| Packages | Needed for |
|---|---|
| `langchain`, `@langchain/langgraph`, `@langchain/core` | always (loader/exporter core) |
| `@langchain/openai` | `OpenAiConfig`, `OpenAiCompatibleConfig`, `VllmConfig` |
| `@langchain/ollama` | `OllamaConfig` |
| `@langchain/mcp-adapters` | `MCPTool`, `MCPToolBox` |
| `@langchain/langgraph-swarm` | `Swarm` |

```bash
npm install langchain @langchain/langgraph @langchain/core
# plus, depending on the components your specs use:
npm install @langchain/openai @langchain/ollama @langchain/mcp-adapters @langchain/langgraph-swarm
```

### Quickstart

Load an Agent Spec configuration and invoke the resulting LangGraph object:

```ts
import { AgentSpecLoader } from "agentspec/adapters/langgraph";

const loader = new AgentSpecLoader({
  toolRegistry: {
    // ServerTool implementations, keyed by tool name.
    get_weather: (input: unknown) =>
      `It is sunny in ${(input as { city: string }).city}.`,
  },
});

const agent = (await loader.loadYaml(yamlText)) as {
  invoke(input: unknown): Promise<Record<string, unknown>>;
};
const result = await agent.invoke({
  messages: [{ role: "user", content: "What is the weather in Agadir?" }],
});
```

All load methods (`loadYaml`, `loadJson`, `loadDict`, `loadComponent`) are async. `AgentSpecLoader` also accepts a `checkpointer` (required for `ClientTool` and `requiresConfirmation` interrupts), a `config` (RunnableConfig), agent `middleware`, deserialization `plugins`, and an `allowedComponents`/`blockedComponents` load policy (`StdioTransport` is blocked by default).

### Exporter

Convert LangGraph objects back into Agent Spec configurations:

```ts
import { AgentSpecExporter } from "agentspec/adapters/langgraph";

const exporter = new AgentSpecExporter();
const yaml = exporter.toYaml(compiledGraph) as string; // also: toJson, toDict, toComponent
```

`createAgent(...)` agents export as `Agent`, compiled `@langchain/langgraph-swarm` graphs as `Swarm`, LangChain structured tools as `ServerTool`, `ChatOpenAI`/`ChatOllama` models as LLM configs, and any other `StateGraph` (compiled or not) as a `Flow`.

### Supported components

| Agent Spec | LangGraph runtime |
|---|---|
| `Agent` | `createAgent` react agent (structured outputs via the tool strategy) |
| `Swarm` | `@langchain/langgraph-swarm` `createSwarm` with handoff tools |
| `ManagerWorkers` | hierarchical `StateGraph` (`__manager__` node plus delegation tools) |
| `Flow` | `StateGraph` supporting `StartNode`, `EndNode`, `LlmNode`, `ToolNode`, `AgentNode`, `BranchingNode`, `ApiNode`, `FlowNode`, `CatchExceptionNode`, `InputMessageNode`, `OutputMessageNode`, `MapNode` |
| `ServerTool` | tool implementation resolved from the `toolRegistry` |
| `ClientTool` | LangGraph interrupt (`client_tool_request` payload) |
| `RemoteTool` | `fetch`-based HTTP tool |
| `MCPTool`, `MCPToolBox` | `@langchain/mcp-adapters` tools (SSE and Streamable HTTP transports) |
| `OpenAiConfig`, `OpenAiCompatibleConfig`, `VllmConfig` | `ChatOpenAI` (with `retryPolicy` mapped to retries/timeout) |
| `LlmConfig` (bare, `api_provider: "openai"`) | `ChatOpenAI` |
| `OllamaConfig` | `ChatOllama` (rejects `retryPolicy`, like Python) |

`ParallelMapNode` and `ParallelFlowNode` are not supported and raise an error.

### Divergences from the Python adapter

- The loader and converter APIs are async (`Promise`-based); Python is sync-first.
- `RemoteTool` and `ApiNode` requests honor the spec's `RetryPolicy` (attempts, backoff with all four jitter modes, `Retry-After` with the 30s cap, recoverable statuses with response-body code matching, per-request `requestTimeout` override, no retry on TLS failures) and enforce `urlAllowList` on every rendered URL; a configured allow list suppresses the templated-URL warning, like Python. `ApiNode` retries diverge from Python, whose executor performs a single plain request. Requests do not follow redirects and default to a 5-second timeout (`DEFAULT_HTTP_REQUEST_TIMEOUT_MS`), matching httpx's defaults.
- When exporting a LangGraph graph whose conditional edge collides with a real node literally named `condition`, the synthetic conditional/branching node names are suffixed (`condition_1`, ...) so the real node keeps its edges; the Python-style names are used otherwise.
- MCP transport `auth` and `retryPolicy` are representation-only (as in Python): they survive load → export untouched but are not wired into the MCP connection.
- `OciGenAiConfig` is not supported (no `langchain-oci` package for JS).
- The MCP mTLS transports (`SSEmTLSTransport`, `StreamableHTTPmTLSTransport`) are not supported.
- Tracing is a no-op seam only; no execution spans or events are emitted yet.

See [examples/09-langgraph-adapter.ts](./examples/09-langgraph-adapter.ts) for a complete offline round trip.

## License

UPL-1.0 or Apache-2.0 — see [LICENSE-UPL.txt](../LICENSE-UPL.txt) and [LICENSE-APACHE.txt](../LICENSE-APACHE.txt).
