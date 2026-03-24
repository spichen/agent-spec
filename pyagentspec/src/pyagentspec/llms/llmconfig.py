# Copyright © 2025 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""This module defines the base class for all LLM configuration component."""

from typing import ClassVar, Optional

from pyagentspec.component import Component
from pyagentspec.llms.llmgenerationconfig import LlmGenerationConfig
from pyagentspec.sensitive_field import SensitiveField
from pyagentspec.versioning import AgentSpecVersionEnum


class LlmConfig(Component):
    """
    A LLM configuration defines how to connect to a LLM to do generation requests.

    This class can be used directly with the ``provider``, ``api_provider``, and ``api_type``
    fields to describe any LLM without a dedicated subclass. Concrete subclasses provide
    additional configuration for specific LLM providers.
    """

    _include_subclasses_in_schema: ClassVar[bool] = True

    model_id: str
    """ID of the model to use"""

    provider: Optional[str] = None
    """The provider of the model (e.g. 'meta', 'openai', 'cohere')."""

    api_provider: Optional[str] = None
    """The API provider used to serve the model (e.g. 'openai', 'oci', 'vllm')."""

    api_type: Optional[str] = None
    """The API protocol to use (e.g. 'chat_completions', 'responses')."""

    base_url: Optional[str] = None
    """Base URL of the API endpoint (e.g. 'https://api.openai.com/v1')."""

    api_key: SensitiveField[Optional[str]] = None
    """An optional API key for the remote LLM model. If specified, the value of the api_key will be
       excluded and replaced by a reference when exporting the configuration."""

    default_generation_parameters: Optional[LlmGenerationConfig] = None
    """Parameters used for the generation call of this LLM"""

    def _versioned_model_fields_to_exclude(
        self, agentspec_version: AgentSpecVersionEnum
    ) -> set[str]:
        fields_to_exclude = super()._versioned_model_fields_to_exclude(agentspec_version)
        if type(self) is not LlmConfig:
            # Subclasses have their own URL fields (e.g. `url`) and don't use base_url.
            fields_to_exclude.add("base_url")
        return fields_to_exclude

    def _infer_min_agentspec_version_from_configuration(self) -> AgentSpecVersionEnum:
        parent_min_version = super()._infer_min_agentspec_version_from_configuration()
        # Bare LlmConfig is a v26_2_0 feature — it was abstract before.
        # Subclasses handle their own versioning independently.
        if type(self) is LlmConfig:
            return max(AgentSpecVersionEnum.v26_2_0, parent_min_version)
        return parent_min_version
