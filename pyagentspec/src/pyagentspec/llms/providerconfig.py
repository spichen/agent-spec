# Copyright © 2025 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""This module defines the provider configuration for generic LLM configs."""

from typing import Any, Dict, Optional

from pydantic import BaseModel


class ProviderConfig(BaseModel):
    """
    Configuration for an LLM provider.

    The ``type`` field identifies the provider (e.g. ``"openai"``, ``"anthropic"``,
    ``"aws_bedrock"``). Extra fields are accepted for provider-specific options
    such as ``region``, ``project_id``, ``deployment_name``, etc.
    """

    type: str
    """Provider identifier (e.g. ``"openai"``, ``"anthropic"``, ``"aws_bedrock"``)"""

    endpoint: Optional[str] = None
    """Base URL override for the provider endpoint"""

    api_protocol: Optional[str] = None
    """Wire protocol override (e.g. ``"openai_chat_completions"``)"""

    api_version: Optional[str] = None
    """API version string"""

    model_config = {"extra": "allow"}

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(*args, **kwargs)
