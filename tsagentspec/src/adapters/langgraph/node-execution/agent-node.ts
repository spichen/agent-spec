/**
 * AgentNode executor for the LangGraph adapter.
 *
 * Port of `pyagentspec.adapters.langgraph._node_execution.AgentNodeExecutor`.
 * Runtime contracts (state keys, error-message text) mirror the Python
 * adapter exactly so specs behave the same across both SDKs.
 *
 * Divergence from Python (see the adapter README): the react-agent invoke
 * payload adds no `remaining_steps` / `structured_response` keys — the
 * langchain JS agent state has neither channel (structured output lands in
 * `structuredResponse`).
 */
import type { BaseMessage } from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import type { AgentNode } from "../../../flows/index.js";
import type { Property } from "../../../property.js";
import { isRecordLike, renderTemplate } from "../../common/index.js";
import type { ExecuteOutput, InvocableGraph, NodeOutputs } from "../types.js";
import { NodeExecutor } from "./executor.js";

/**
 * Extract the outputs of an agent invoke result for the expected output
 * properties, merging (in increasing priority) property defaults, the
 * structured response, and top-level result entries. Reads the langchain JS
 * `structuredResponse` key, falling back to Python's `structured_response`.
 */
export function extractOutputsFromInvokeResult(
  result: Record<string, unknown>,
  expectedOutputs: Property[],
): NodeOutputs {
  const outputs: NodeOutputs = {};
  for (const output of expectedOutputs) {
    if (output.default !== undefined) {
      outputs[output.title] = output.default;
    }
  }
  const structuredResponse =
    result["structuredResponse"] ?? result["structured_response"];
  if (isRecordLike(structuredResponse)) {
    Object.assign(outputs, structuredResponse);
  }
  for (const output of expectedOutputs) {
    if (Object.hasOwn(result, output.title)) {
      outputs[output.title] = result[output.title];
    }
  }
  return outputs;
}

/**
 * Executes an AgentNode holding a plain Agent: renders the agent's system
 * prompt against the node inputs, compiles (and caches) a react agent per
 * rendered prompt through the converter-provided factory, and invokes it on
 * the flow messages.
 *
 * `_execute` is a template method, mirroring Python: prepare (compile the
 * cached agent, shape the invoke payload) then invoke then format. Subclasses
 * (`ManagerWorkersNodeExecutor`) override only `prepareAgentAndInputs` and
 * `formatAgentResult`, sharing the compile callback, the rendered-prompt
 * cache and `withDrivingMessage`.
 */
export class AgentNodeExecutor extends NodeExecutor<AgentNode> {
  /** Compiles a runnable graph for one rendered system prompt (converter-provided). */
  protected readonly compileAgent: (
    renderedSystemPrompt: string,
  ) => Promise<unknown>;
  protected readonly config: RunnableConfig;
  /** Compiled agents cached by rendered system prompt. */
  protected readonly agentsCache = new Map<string, unknown>();

  constructor(
    node: AgentNode,
    compileAgent: (renderedSystemPrompt: string) => Promise<unknown>,
    config: RunnableConfig,
  ) {
    super(node);
    this.compileAgent = compileAgent;
    this.config = config;
  }

  private async createReactAgentWithGivenInputValues(
    inputs: NodeOutputs,
  ): Promise<InvocableGraph> {
    const agentComponent = this.node.agent;
    if (agentComponent.componentType !== "Agent") {
      throw new Error(
        "AgentNodeExecutor can only be used with AgentSpecAgent agents",
      );
    }
    const systemPrompt = renderTemplate(agentComponent.systemPrompt, inputs);
    let agent = this.agentsCache.get(systemPrompt);
    if (agent === undefined) {
      agent = await this.compileAgent(systemPrompt);
      this.agentsCache.set(systemPrompt, agent);
    }
    return agent as InvocableGraph;
  }

  /** LangGraph's agent expects at least one user message to drive execution. */
  protected withDrivingMessage(messages: BaseMessage[]): unknown[] {
    return messages.length > 0 ? messages : [{ role: "user", content: "" }];
  }

  /** Compile (or reuse) the agent for the inputs and shape its invoke payload. */
  protected async prepareAgentAndInputs(
    inputs: NodeOutputs,
    messages: BaseMessage[],
  ): Promise<[InvocableGraph, Record<string, unknown>]> {
    const agent = await this.createReactAgentWithGivenInputValues(inputs);
    const preparedInputs: Record<string, unknown> = {
      ...inputs,
      messages: this.withDrivingMessage(messages),
    };
    return [agent, preparedInputs];
  }

  /** Map an agent invoke result onto the node outputs (or a chat message). */
  protected formatAgentResult(result: Record<string, unknown>): ExecuteOutput {
    const nodeOutputs = this.node.outputs ?? [];
    if (nodeOutputs.length === 0) {
      const messages = Array.isArray(result["messages"])
        ? (result["messages"] as { content?: unknown }[])
        : [];
      const generatedMessage = messages[messages.length - 1];
      return [
        {},
        {
          generated_messages: [
            {
              role: "assistant",
              content: (generatedMessage?.content ?? "") as string,
            },
          ],
        },
      ];
    }
    return [extractOutputsFromInvokeResult(result, nodeOutputs), {}];
  }

  protected async _execute(
    inputs: NodeOutputs,
    messages: BaseMessage[],
  ): Promise<ExecuteOutput> {
    const [agent, preparedInputs] = await this.prepareAgentAndInputs(
      inputs,
      messages,
    );
    const result = await agent.invoke(preparedInputs, this.config);
    return this.formatAgentResult(result);
  }
}
