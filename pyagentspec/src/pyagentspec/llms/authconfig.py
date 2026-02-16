# Copyright © 2025 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""This module defines the auth configuration for generic LLM configs."""

import os
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
    """Reference to a credential. If the value matches an environment variable
    name, :meth:`resolve_credential` returns the variable's value; otherwise
    the literal string is returned."""

    def resolve_credential(self) -> str:
        """Resolve ``credential_ref`` to an actual credential value.

        Resolution order:
        1. If ``credential_ref`` is ``None`` or empty, return ``""``.
        2. If an environment variable with that exact name exists, return its value.
        3. Otherwise treat ``credential_ref`` as the literal credential.
        """
        if not self.credential_ref:
            return ""
        return os.environ.get(self.credential_ref, self.credential_ref)

    model_config = {"extra": "allow"}

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(*args, **kwargs)
