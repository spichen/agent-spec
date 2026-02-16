# Copyright © 2025 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""This module defines the generic LLM configuration component."""

from typing import Any, Dict, List, Optional

from pyagentspec.llms.authconfig import AuthConfig
from pyagentspec.llms.generationconfig import GenerationConfig
from pyagentspec.llms.llmconfig import LlmConfig
from pyagentspec.llms.llmendpoint import LlmEndpoint
from pyagentspec.llms.providerconfig import ProviderConfig
from pyagentspec.sensitive_field import SensitiveField
from pyagentspec.versioning import AgentSpecVersionEnum


class GenericLlmConfig(LlmConfig):
    """
    A generic, provider-agnostic LLM configuration.

    Works for any provider via the ``provider.type`` string discriminator.
    Supports authentication, extended generation parameters, fallback routing,
    and provider-specific extensions.
    """

    model_id: str
    """Primary model identifier"""

    provider: ProviderConfig
    """Primary provider configuration"""

    auth: SensitiveField[Optional[AuthConfig]] = None
    """Optional authentication configuration. Excluded from exports and replaced
    by a ``$component_ref`` reference during serialization."""

    default_generation_parameters: Optional[GenerationConfig] = None
    """Extended generation parameters (overrides parent type)"""

    fallback: Optional[List[LlmEndpoint]] = None
    """Ordered list of fallback endpoints"""

    provider_extensions: Optional[Dict[str, Any]] = None
    """Non-portable escape hatch for provider-specific options"""

    def _infer_min_agentspec_version_from_configuration(self) -> AgentSpecVersionEnum:
        return AgentSpecVersionEnum.v26_2_0

    def _versioned_model_fields_to_exclude(
        self, agentspec_version: AgentSpecVersionEnum
    ) -> set[str]:
        fields_to_exclude = set()
        if agentspec_version < AgentSpecVersionEnum.v26_2_0:
            fields_to_exclude.update({
                "model_id", "provider", "auth",
                "default_generation_parameters", "fallback",
                "provider_extensions",
            })
        return fields_to_exclude
