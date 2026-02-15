# Copyright © 2025 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""This module defines the auth configuration for generic LLM configs."""

from typing import Any, Dict, Optional

from pydantic import BaseModel


class AuthConfig(BaseModel):
    """
    Configuration for LLM authentication.

    The ``type`` field identifies the auth mechanism (e.g. ``"api_key"``,
    ``"iam_role"``, ``"gcp_service_account"``). Extra fields are accepted
    for auth-type-specific options.
    """

    type: str
    """Auth mechanism identifier (e.g. ``"api_key"``, ``"iam_role"``)"""

    credential_ref: Optional[str] = None
    """Primary credential or environment variable reference"""

    model_config = {"extra": "allow"}

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(*args, **kwargs)
