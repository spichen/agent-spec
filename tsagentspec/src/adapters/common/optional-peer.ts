/**
 * Optional-peer-dependency import helper shared by the AgentSpec adapters.
 *
 * Adapter integrations (chat models, MCP, swarm assembly) live behind
 * optional peer dependencies loaded via dynamic `import()`. Call sites keep
 * the `import("...")` literal inside the `load` thunk so bundlers and TS can
 * still analyze it; this helper only owns the shared failure message shape.
 */

/**
 * Await `load()`, rethrowing an import failure as an actionable error naming
 * the missing package, what it is needed for, and how to proceed.
 */
export async function importOptionalPeer<T>(
  load: () => Promise<T>,
  packageName: string,
  purpose: string,
  hint: string,
): Promise<T> {
  try {
    return await load();
  } catch (error) {
    throw new Error(
      `${packageName} is required to ${purpose}. ` +
        `Install it (e.g., npm install ${packageName}) or ${hint}`,
      { cause: error },
    );
  }
}
