/**
 * ManagerWorkersExecutionSpan (port of `pyagentspec.tracing.spans.managerworkers`).
 */
import type { ManagerWorkers } from "../../agents/index.js";
import { Span, type SpanOptions } from "./span.js";

export interface ManagerWorkersExecutionSpanOptions extends SpanOptions {
  /** The ManagerWorkers being executed */
  managerworkers: ManagerWorkers;
}

/**
 * Span to represent the execution of a ManagerWorkers. Can be nested when executing sub-agents.
 *
 * - Starts when: manager-workers pattern execution starts
 * - Ends when: the manager-workers execution is completed and the result is ready to be processed
 */
export class ManagerWorkersExecutionSpan extends Span {
  managerworkers: ManagerWorkers;

  override get type(): string {
    return "ManagerWorkersExecutionSpan";
  }

  constructor(options: ManagerWorkersExecutionSpanOptions) {
    super(options);
    this.managerworkers = options.managerworkers;
  }
}
