# Copyright © 2025 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""Shared URL normalisation for OpenAI-compatible endpoints."""

from urllib.parse import urlparse, urlunparse


def prepare_openai_compatible_url(url: str) -> str:
    """
    Normalise a URL for an OpenAI-compatible server.

    - Ensures a scheme (http, https) is present, defaulting to ``http``.
    - If no meaningful path is provided (empty or ``/``), appends ``/v1``.
    - If the caller already supplies a path, it is preserved as-is so that
      non-OpenAI providers (Azure, Anthropic, custom gateways, …) work
      without their paths being silently overwritten.

    Examples:
        - "localhost:8000"                        -> "http://localhost:8000/v1"
        - "127.0.0.1:5000"                       -> "http://127.0.0.1:5000/v1"
        - "https://api.example.com"               -> "https://api.example.com/v1"
        - "https://api.example.com/v2/beta"       -> "https://api.example.com/v2/beta"
        - "http://my-host/custom/path"            -> "http://my-host/custom/path"
    """
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    parsed = urlparse(url)
    if not parsed.path or parsed.path == "/":
        parsed = parsed._replace(path="/v1")
    return str(urlunparse(parsed))
