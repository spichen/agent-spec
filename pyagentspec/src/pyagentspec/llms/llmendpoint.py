# Copyright © 2025 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""This module defines the LLM endpoint model for fallback routing."""

from typing import Any, Dict, Optional

from pydantic import BaseModel

from pyagentspec.llms.authconfig import AuthConfig
from pyagentspec.llms.providerconfig import ProviderConfig


class LlmEndpoint(BaseModel):
    """
    An LLM endpoint for fallback routing.

    Each endpoint specifies a model, provider, and optional auth override,
    allowing fallback across different providers or models.
    """

    model_id: str
    """Model identifier for this endpoint"""

    provider: ProviderConfig
    """Provider configuration for this endpoint"""

    auth: Optional[AuthConfig] = None
    """Optional auth override for this endpoint"""

    provider_extensions: Optional[Dict[str, Any]] = None
    """Non-portable escape hatch for provider-specific options"""
