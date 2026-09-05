/**
 * Port of pyagentspec/tests/tracing/events/test_state_snapshot_emitted.py.
 */
import { describe, expect, it } from "vitest";
import { PII_MASK, StateSnapshotEmitted } from "../../src/index.js";

describe("StateSnapshotEmitted", () => {
  it("creates and masks the snapshot payloads", () => {
    const event = new StateSnapshotEmitted({
      conversationId: "conversation-123",
      stateSnapshot: { conversation: { messages: [] } },
      extraState: { ui: { active_tab: "plan" } },
      name: "snapshot",
    });

    expect(event.name).toBe("snapshot");
    expect(event.conversationId).toBe("conversation-123");
    expect(event.stateSnapshot).toEqual({ conversation: { messages: [] } });
    expect(event.extraState).toEqual({ ui: { active_tab: "plan" } });

    const masked = event.serialize({ maskSensitiveInformation: true });
    const unmasked = event.serialize({ maskSensitiveInformation: false });

    expect(masked["state_snapshot"]).toBe(PII_MASK);
    expect(masked["extra_state"]).toBe(PII_MASK);
    expect(masked["conversation_id"]).toBe("conversation-123");
    expect(masked["type"]).toBe("StateSnapshotEmitted");

    expect(unmasked["state_snapshot"]).toEqual({ conversation: { messages: [] } });
    expect(unmasked["extra_state"]).toEqual({ ui: { active_tab: "plan" } });
  });

  it.each([
    [{ conversation: { messages: [] } }, null],
    [null, { ui: { active_tab: "plan" } }],
  ] as Array<
    [Record<string, unknown> | null, Record<string, unknown> | null]
  >)("allows either payload alone (%#)", (stateSnapshot, extraState) => {
    const event = new StateSnapshotEmitted({
      conversationId: "conversation-123",
      stateSnapshot,
      extraState,
    });

    expect(event.stateSnapshot).toEqual(stateSnapshot);
    expect(event.extraState).toEqual(extraState);
  });

  it("requires at least one payload", () => {
    expect(
      () => new StateSnapshotEmitted({ conversationId: "conversation-123" }),
    ).toThrow("At least one of state_snapshot or extra_state must be provided");
  });

  it.each([
    [
      { stateSnapshot: { conversation: { opaque: () => "not-json" } } },
      "state_snapshot must be JSON-serializable",
    ],
    [
      { extraState: { ui: { opaque: () => "not-json" } } },
      "extra_state must be JSON-serializable",
    ],
    [
      { stateSnapshot: { value: Number.NaN } },
      "state_snapshot must be JSON-serializable",
    ],
    [
      { extraState: { value: Number.POSITIVE_INFINITY } },
      "extra_state must be JSON-serializable",
    ],
    [
      { stateSnapshot: { when: new Date(0) } },
      "state_snapshot must be JSON-serializable",
    ],
  ] as Array<[Record<string, Record<string, unknown>>, string]>)(
    "rejects non-JSON-serializable payloads (%#)",
    (payloads, expectedMessage) => {
      expect(
        () =>
          new StateSnapshotEmitted({ conversationId: "conversation-123", ...payloads }),
      ).toThrow(expectedMessage);
    },
  );

  it("accepts a realistic runtime resumable payload", () => {
    const stateSnapshot = {
      runtime: "my-agent-runtime",
      schema_version: 1,
      conversation_state: '{"type":"FlowConversation","version":1}',
      conversation: {
        id: "runtime-conversation-123",
        messages: [],
      },
      execution: {
        status: null,
        status_handled: false,
      },
    };
    const event = new StateSnapshotEmitted({
      conversationId: "conversation-123",
      stateSnapshot,
      extraState: { ui: { active_tab: "plan" } },
    });

    expect(event.conversationId).toBe("conversation-123");
    expect(event.stateSnapshot).toEqual(stateSnapshot);
    expect(event.stateSnapshot!["conversation_state"]).toBe(
      '{"type":"FlowConversation","version":1}',
    );
    expect(event.extraState).toEqual({ ui: { active_tab: "plan" } });
  });
});
