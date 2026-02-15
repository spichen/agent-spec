# Copyright © 2025 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""Shared URL normalisation for OpenAI-compatible endpoints."""

from urllib.parse import urlparse, urlunparse


def prepare_openai_compatible_url(url: str) -> str:
    """
    Correctly formats a URL for an OpenAI-compatible server.

    This function is robust and handles multiple formats:
    - Ensures a scheme (http, https) is present, defaulting to 'http'.
    - Replaces any existing path with exactly '/v1'.

    Examples:
        - "localhost:8000"            -> "http://localhost:8000/v1"
        - "127.0.0.1:5000"           -> "http://127.0.0.1:5000/v1"
        - "https://api.example.com"  -> "https://api.example.com/v1"
        - "http://my-host/api/v2"    -> "http://my-host/v1"
    """
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    parsed_url = urlparse(url)
    v1_url_parts = parsed_url._replace(path="/v1", params="", query="", fragment="")
    return str(urlunparse(v1_url_parts))
