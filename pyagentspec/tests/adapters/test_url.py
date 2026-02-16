# Copyright © 2026 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""Tests for the shared OpenAI-compatible URL normalisation helper."""

import pytest

from pyagentspec.adapters._url import prepare_openai_compatible_url


@pytest.mark.parametrize(
    "input_url,expected",
    [
        # No path → /v1 appended
        ("localhost:8000", "http://localhost:8000/v1"),
        ("127.0.0.1:5000", "http://127.0.0.1:5000/v1"),
        ("http://localhost:8000", "http://localhost:8000/v1"),
        ("https://api.example.com", "https://api.example.com/v1"),
        ("  http://localhost:8000  ", "http://localhost:8000/v1"),
        # Explicit path → preserved as-is
        ("https://api.example.com/v2/beta", "https://api.example.com/v2/beta"),
        ("http://my-host/api/v2", "http://my-host/api/v2"),
        ("http://my-host/v1", "http://my-host/v1"),
        ("https://azure.openai.com/openai/deployments/gpt4", "https://azure.openai.com/openai/deployments/gpt4"),
    ],
)
def test_prepare_openai_compatible_url(input_url: str, expected: str) -> None:
    assert prepare_openai_compatible_url(input_url) == expected


def test_https_url_preserves_scheme() -> None:
    """Regression: https:// URLs must not be corrupted to http://https://..."""
    result = prepare_openai_compatible_url("https://api.openai.com")
    assert result.startswith("https://")
    assert result == "https://api.openai.com/v1"
