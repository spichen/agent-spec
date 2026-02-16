# Copyright © 2025 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""This module defines the auth configuration for generic LLM configs."""

import os
from typing import Any, Dict, Optional

from pydantic import BaseModel


_ENV_PREFIX = "$env:"


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
    """Reference to a credential.

    Use the ``$env:`` prefix to resolve an environment variable at runtime::

        credential_ref="$env:OPENAI_API_KEY"   # reads os.environ["OPENAI_API_KEY"]

    Without the prefix the value is treated as a literal credential::

        credential_ref="sk-abc123"             # used as-is
    """

    def resolve_credential(self) -> str:
        """Resolve ``credential_ref`` to an actual credential value.

        Resolution order:
        1. If ``credential_ref`` is ``None`` or empty, return ``""``.
        2. If ``credential_ref`` starts with ``$env:``, look up the named
           environment variable.  Raises :class:`ValueError` when the
           variable is not set.
        3. Otherwise return ``credential_ref`` as a literal credential.
        """
        if not self.credential_ref:
            return ""
        if self.credential_ref.startswith(_ENV_PREFIX):
            var_name = self.credential_ref[len(_ENV_PREFIX) :]
            value = os.environ.get(var_name)
            if value is None:
                raise ValueError(
                    f"Environment variable '{var_name}' referenced by "
                    f"credential_ref is not set"
                )
            return value
        return self.credential_ref

    model_config = {"extra": "allow"}

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(*args, **kwargs)
