/**
 * Shared adapter common layer barrel (internal use; not a package export).
 *
 * Framework-agnostic building blocks for AgentSpec adapters: template
 * rendering, URL validation, the component load policy, JSON-schema helpers,
 * RemoteTool execution, converter interfaces, and the loader/exporter base
 * classes.
 */
export {
  renderTemplate,
  renderNestedObjectTemplate,
  stringifyTemplateValue,
} from "./templating.js";
export {
  getUrlMatchParts,
  matchesAllowListEntry,
  getUrlDestinationPlaceholderNames,
  maybeWarnAboutUnrestrictedTemplatedUrl,
  validateUrlAgainstAllowList,
} from "./url-validation.js";
export {
  ComponentLoadPolicy,
  type ComponentPolicyEntry,
  type ComponentPolicyInput,
} from "./component-policy.js";
export {
  jsonSchemasHaveSameType,
  buildJsonSchemaFromProperties,
} from "./json-schema.js";
export {
  DEFAULT_HTTP_REQUEST_TIMEOUT_MS,
  buildTemplatedHttpRequest,
  createRemoteToolFunc,
  fetchWithAdapterDefaults,
  renderRecord,
  type TemplatedHttpRequestSpec,
} from "./tools-common.js";
export type {
  AgentSpecToRuntimeConverter,
  RuntimeToAgentSpecConverter,
} from "./converters.js";
export {
  AdapterAgnosticAgentSpecLoader,
  type AdapterAgnosticAgentSpecLoaderOptions,
  type LoadOptions,
} from "./agentspec-loader.js";
export {
  AdapterAgnosticAgentSpecExporter,
  type ExportOptions,
  type ExportedDict,
  type RuntimeDisaggregatedComponentsConfig,
} from "./agentspec-exporter.js";
