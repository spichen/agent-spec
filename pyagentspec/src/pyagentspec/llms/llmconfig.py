# Copyright © 2025 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""This module defines the base class for all LLM configuration component."""

from typing import Any, Optional

from pyagentspec.component import Component
from pyagentspec.llms.llmgenerationconfig import LlmGenerationConfig
from pyagentspec.versioning import AgentSpecVersionEnum


class LlmConfig(Component):
    """
    A LLM configuration defines how to connect to a LLM to do generation requests.

    This class can be used directly with the ``provider``, ``api_provider``, and ``api_type``
    fields to describe any LLM without a dedicated subclass. Concrete subclasses provide
    additional configuration for specific LLM providers.
    """

    model_id: str
    """ID of the model to use"""

    provider: Optional[str] = None
    """The provider of the model (e.g. 'meta', 'openai', 'cohere')."""

    api_provider: Optional[str] = None
    """The API provider used to serve the model (e.g. 'openai', 'oci', 'vllm')."""

    api_type: Optional[str] = None
    """The API protocol to use (e.g. 'chat_completions', 'responses')."""

    default_generation_parameters: Optional[LlmGenerationConfig] = None
    """Parameters used for the generation call of this LLM"""

    def _versioned_model_fields_to_exclude(
        self, agentspec_version: AgentSpecVersionEnum
    ) -> set[str]:
        fields_to_exclude = set()
        # Only exclude the generic fields for bare LlmConfig instances.
        # Subclasses may shadow these fields (e.g. OciGenAiConfig has its own
        # provider field that pre-dates v26_2_0) and manage their own exclusion rules.
        if type(self) is LlmConfig and agentspec_version < AgentSpecVersionEnum.v26_2_0:
            fields_to_exclude.add("provider")
            fields_to_exclude.add("api_provider")
            fields_to_exclude.add("api_type")
        return fields_to_exclude

    def _infer_min_agentspec_version_from_configuration(self) -> AgentSpecVersionEnum:
        parent_min_version = super()._infer_min_agentspec_version_from_configuration()
        current_object_min_version = self.min_agentspec_version
        # Only bump version for bare LlmConfig instances — subclasses set these fields
        # as class-level defaults (implied by component_type) and handle versioning themselves.
        if type(self) is LlmConfig:
            if (
                self.provider is not None
                or self.api_provider is not None
                or self.api_type is not None
            ):
                current_object_min_version = AgentSpecVersionEnum.v26_2_0
        return max(current_object_min_version, parent_min_version)
