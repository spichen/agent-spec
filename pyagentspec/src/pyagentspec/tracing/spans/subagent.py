# Copyright © 2025 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

from pydantic import SerializeAsAny

from pyagentspec.agenticcomponent import AgenticComponent
from pyagentspec.tracing.spans.span import Span


class SubAgentExecutionSpan(Span):
    """
    Span to represent the execution of a sub-agent delegated from a parent Agent.

    - Starts when: the parent agent invokes a sub-agent
    - Ends when: the sub-agent execution completes and the result is returned to the parent
    """

    sub_agent: SerializeAsAny[AgenticComponent]
    "The sub-agent being executed"
