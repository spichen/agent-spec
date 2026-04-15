# Copyright © 2025 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

from typing import Any, Dict

from pydantic import SerializeAsAny

from pyagentspec.agenticcomponent import AgenticComponent
from pyagentspec.sensitive_field import SensitiveField
from pyagentspec.tracing.events.event import Event


class SubAgentExecutionStart(Event):
    """A parent agent is starting delegation to a sub-agent. Emitted when a SubAgentExecutionSpan starts."""

    sub_agent: SerializeAsAny[AgenticComponent]
    "The sub-agent being executed"

    inputs: SensitiveField[Dict[str, Any]]
    "The inputs passed to the sub-agent, one per property defined in the sub-agent's inputs"


class SubAgentExecutionEnd(Event):
    """A sub-agent has finished execution and returned a result to the parent. Emitted when a SubAgentExecutionSpan ends."""

    sub_agent: SerializeAsAny[AgenticComponent]
    "The sub-agent that was executed"

    outputs: SensitiveField[Dict[str, Any]]
    "The outputs returned by the sub-agent, one per property defined in the sub-agent's outputs"
