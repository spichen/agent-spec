/**
 * Public exporter converting LangGraph objects into Agent Spec
 * configurations.
 *
 * Port of `pyagentspec.adapters.langgraph.agentspecexporter.AgentSpecExporter`
 * on top of the adapter-agnostic exporter base: serialization plugins,
 * disaggregated configurations, and the LangGraph-specific runtime converter.
 */
import { AdapterAgnosticAgentSpecExporter } from "../common/index.js";
import { LangGraphToAgentSpecConverter } from "./agentspec-converter.js";

/**
 * Helper class to convert LangGraph components (react agents, chat models,
 * structured tools, state graphs) into Agent Spec configurations via
 * `toJson` / `toYaml` / `toDict` / `toComponent`.
 */
export class AgentSpecExporter extends AdapterAgnosticAgentSpecExporter {
  /** Converter used to convert LangGraph components to Agent Spec components. */
  get runtimeToAgentSpecConverter(): LangGraphToAgentSpecConverter {
    return new LangGraphToAgentSpecConverter();
  }
}
