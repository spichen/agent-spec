# agentspec TypeScript SDK

TypeScript SDK for the [Oracle Open Agent Specification](https://github.com/oracle/agent-spec).

## Supported component types

These types round-trip correctly between JSON/YAML and TypeScript objects:

- **Agents**: `Agent`, `Swarm`, `ManagerWorkers`, `RemoteAgent`, `SpecializedAgent`, `A2AAgent`
- **Flows**: `Flow`, `StartNode`, `EndNode`, `LlmNode`, `ToolNode`, `AgentNode`, `FlowNode`, `BranchingNode`, `MapNode`, `ParallelMapNode`, `ParallelFlowNode`, `ApiNode`, `InputMessageNode`, `OutputMessageNode`, `CatchExceptionNode`
- **Tools**: `ServerTool`, `ClientTool`, `RemoteTool`, `BuiltinTool`, `MCPTool`
- **LLM configs**: `OpenAiCompatibleConfig`, `OllamaConfig`, `VllmConfig`, `OpenAiConfig`, `OciGenAiConfig`, bare `LlmConfig`
- **MCP**: `MCPToolBox`, `StdioTransport`, `SSETransport`, `StreamableHTTPTransport` (and mTLS variants)
- **Auth**: `OAuthConfig`, `OAuthClientConfig` (the `auth` field on remote MCP transports)
- **Datastores**: `InMemoryCollectionDatastore`, `OracleDatabaseDatastore`, `PostgresDatabaseDatastore`
- **Other**: `ControlFlowEdge`, `DataFlowEdge`, `MessageSummarizationTransform`, `ConversationSummarizationTransform`

Non-component configuration objects nested in the above — `RetryPolicy` (on `RemoteTool`, `ApiNode`, MCP tools/transports, and the LLM configs) and `urlAllowList` (on `RemoteTool` and `ApiNode`) — round-trip with the same version-gated serialization as pyagentspec.

### Fixture compatibility

`tests/repo-fixtures.test.ts` is the CI contract: a curated list of 61 configs known to round-trip with the current SDK. Not every example or historical config file in the repo is covered here. Files using types outside the list above (e.g. `howto_swarm`, `howto_a2aagent`) are not yet supported.

### Divergences from the Python SDK

- `OAuthClientConfig` secrets (`client_id`, `client_secret`, `client_id_metadata_url`) are redacted from serialized output like other sensitive fields. pyagentspec declares them as `SensitiveField`s, but an annotation bug (`Optional[SensitiveField[str]]` buries the marker inside the Union, so pydantic never lifts it into the field metadata) means Python currently exports the plain values; this SDK follows the declared intent.

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
| `@langchain/openai` | `OpenAiConfig`, `OpenAiCompatibleConfig`, `VllmConfig`, bare `LlmConfig` |
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

### Tracing

Loaded graphs emit Agent Spec traces. Register a `SpanProcessor` by running the graph inside a `Trace`:

```ts
import { SpanProcessor, Trace } from "agentspec";
import type { Event, Span } from "agentspec";

class ConsoleSpanProcessor extends SpanProcessor {
  onStart(span: Span) {}
  onEnd(span: Span) { console.log("span", span.serialize()); }
  onEvent(event: Event, span: Span) { console.log("event", event.serialize()); }
  startup() {}
  shutdown() {}
}

await new Trace({ spanProcessors: [new ConsoleSpanProcessor()] }).run(() =>
  agent.invoke({ messages: [{ role: "user", content: "..." }] }),
);
```

Without an ambient `Trace`, nothing is emitted and behavior is unchanged.

### Divergences from the Python adapter

- The loader and converter APIs are async (`Promise`-based); Python is sync-first.
- `RemoteTool` and `ApiNode` requests honor the spec's `RetryPolicy` (attempts, backoff with all four jitter modes, `Retry-After` with the 30s cap, recoverable statuses with response-body code matching, per-request `requestTimeout` override, no retry on TLS failures) and enforce `urlAllowList` on every rendered URL; a configured allow list suppresses the templated-URL warning, like Python. `ApiNode` retries diverge from Python, whose executor performs a single plain request. Requests do not follow redirects and default to a 5-second timeout (`DEFAULT_HTTP_REQUEST_TIMEOUT_MS`), matching httpx's defaults. Jitter randomness uses `Math.random()` (Python uses `SystemRandom`), and malformed `Retry-After` headers are handled defensively rather than byte-matching Python's edge behavior: a negative numeric value (invalid per RFC 9110) is treated as an absent header so the configured backoff applies, where Python lets it crash the call in `time.sleep`; `inf`/`Infinity` fall back to jittered backoff where Python caps the wait; and some date formats Python rejects (e.g. ISO 8601) are accepted.
- Because spec files are untrusted input, the retry engine is bounded in ways Python's is not. Python caps only total elapsed time, which permits hundreds of thousands of requests from a single call when the configured delays are zero. Here, one call makes at most `MAX_HTTP_ATTEMPTS_PER_CALL` (100) HTTP attempts, consecutive attempts are separated by at least `MIN_RETRY_DELAY_SECONDS` (50ms) however low `initialRetryDelay`/`maxRetryDelay` are set (or however a server sets `Retry-After`), and `requestTimeout` must be finite and is clamped to a delay a timer can express (`MAX_HTTP_REQUEST_TIMEOUT_MS`) — Python's httpx reads a non-finite timeout as "no timeout", which no JS timer can represent. TLS *handshake* failures (e.g. an `https://` URL aimed at a plain-HTTP port) are treated as non-retryable alongside certificate-validation failures; Python retries them.
- Errors thrown for a non-2xx response redact the URL's query string, which routinely carries credentials in templated specs. Python's `raise_for_status` embeds the full URL, and this error reaches both the model (as the tool result) and the logs.
- When exporting a LangGraph graph whose conditional edge collides with a real node literally named `condition`, the synthetic conditional/branching node names are suffixed (`condition_1`, ...) so the real node keeps its edges; the Python-style names are used otherwise.
- MCP transport `auth` and `retryPolicy` are representation-only (as in Python): they survive load → export untouched but are not wired into the MCP connection.
- `OciGenAiConfig` is not supported (no `langchain-oci` package for JS).
- The MCP mTLS transports (`SSEmTLSTransport`, `StreamableHTTPmTLSTransport`) are not supported.
- Tracing is emitted with Python-parity span/event payloads: loaded graphs are wrapped in execution spans (`AgentExecutionSpan`, `FlowExecutionSpan`, `ManagerWorkersExecutionSpan`), and converter-built chat models and tools emit LLM-generation and tool-execution spans and events. Divergences:
  - The tracing API is async-only; Python's sync/async twin callbacks collapse into single async handlers.
  - A raw compiled graph unwrapped from a patched react agent (swarm assembly, the ManagerWorkers `__manager__` node) is not patched, so those embedded sub-agent runs emit no `AgentExecutionSpan` of their own; ManagerWorkers workers, invoked through the patched agent, do.
  - Tool-end events follow Python's sync `on_tool_end` payload mapping (declared-outputs title mapping, `request_id` always the LangChain run id) — the variant Python's flow tests pin to exact payloads. Python routes runs under an event loop (`ainvoke`/`astream`) to its async twin, which maps a non-dict `ToolMessage` payload to `{"output": ...}` and takes `request_id` from the message's `tool_call_id` when present, so async-Python traces differ from TS (and from sync-Python) traces on those fields.
  - The `invoke` wrapper builds the execution-span end event from the invoke result (Python folds streamed state chunks, which yields `{}` on the invoke path).
  - Non-string trace payloads are coerced with `JSON.stringify` where Python uses `str(...)`.
  - Parallel isolation comes from forking an `AsyncLocalStorage` child context per patched run and per flow-node span (Python relies on asyncio tasks copying `contextvars` per task).

See [examples/09-langgraph-adapter.ts](./examples/09-langgraph-adapter.ts) for a complete offline round trip.

## License

UPL-1.0 or Apache-2.0 — see [LICENSE-UPL.txt](../LICENSE-UPL.txt) and [LICENSE-APACHE.txt](../LICENSE-APACHE.txt).
