# Copyright © 2025 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""The Pydantic deserialization plugin must SURFACE the collected validation
errors, not mask them.

Regression: collected errors were rebuilt into ``pydantic_core.InitErrorDetails``
with no ``ctx``. ``ValidationError.from_exception_data`` requires
``ctx={"error": <exc>}`` for ``value_error`` (and other builtins require their own
ctx keys), so a collected ``value_error`` — e.g. any component ``model_validator``
that raises ``ValueError`` — made ``from_exception_data`` itself raise
``TypeError: 'error' required in context``, hiding the real cause.
"""

import pytest
from pydantic import ValidationError
from pydantic_core import ValidationError as CoreValidationError

from pyagentspec.serialization.pydanticdeserializationplugin import (
    PydanticComponentDeserializationPlugin,
)
from pyagentspec.validation_helpers import PyAgentSpecErrorDetails


def _raise(line_errors):
    """Emulate the plugin's re-raise from a list of collected errors."""
    return CoreValidationError.from_exception_data(
        title="SomeComponent",
        line_errors=[
            PydanticComponentDeserializationPlugin._to_line_error(e) for e in line_errors
        ],
    )


def test_value_error_is_surfaced_not_masked() -> None:
    """A collected ``value_error`` re-raises cleanly and keeps its message."""
    collected = [
        PyAgentSpecErrorDetails(
            type="value_error",
            msg="The AgentNode component expected a property titled `evidence_status`.",
            loc=("outputs",),
        )
    ]

    # Must NOT raise TypeError("'error' required in context") while building.
    err = _raise(collected)

    assert isinstance(err, (ValidationError, CoreValidationError))
    [detail] = err.errors()
    assert detail["type"] == "value_error"
    assert detail["loc"] == ("outputs",)
    # The message is rendered verbatim — no added/doubled "Value error, " prefix.
    assert detail["msg"] == collected[0].msg


def test_original_error_type_is_preserved() -> None:
    """The reconstructed error keeps its original ``type`` (not flattened to
    ``value_error``) and its message verbatim — never crashes on a ctx key it
    would otherwise have to synthesise."""
    collected = [
        PyAgentSpecErrorDetails(type="missing", msg="Field required", loc=("name",))
    ]

    [detail] = _raise(collected).errors()
    assert detail["type"] == "missing"
    assert detail["msg"] == "Field required"


def test_message_with_braces_is_not_interpreted() -> None:
    """Messages carrying JSON (literal ``{`` / ``}``) survive verbatim — the
    message must not be treated as a format template."""
    msg = 'Invalid schema {"type": "object", "required": true} for {x}'
    [detail] = _raise(
        [PyAgentSpecErrorDetails(type="value_error", msg=msg, loc=("s",))]
    ).errors()
    assert detail["msg"] == msg


def test_multiple_errors_all_surface() -> None:
    collected = [
        PyAgentSpecErrorDetails(type="value_error", msg="first problem", loc=("a",)),
        PyAgentSpecErrorDetails(type="missing", msg="second problem", loc=("b",)),
    ]

    err = _raise(collected)

    msgs = {d["msg"] for d in err.errors()}
    assert msgs == {"first problem", "second problem"}
    assert err.error_count() == 2
