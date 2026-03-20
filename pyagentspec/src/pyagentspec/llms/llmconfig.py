# Copyright © 2025 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""This module defines the base class for all LLM configuration component."""

from typing import Optional

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

    def _infer_min_agentspec_version_from_configuration(self) -> AgentSpecVersionEnum:
        parent_min_version = super()._infer_min_agentspec_version_from_configuration()
        # Bare LlmConfig is a v26_2_0 feature — it was abstract before.
        # Subclasses handle their own versioning independently.
        if type(self) is LlmConfig:
            return max(AgentSpecVersionEnum.v26_2_0, parent_min_version)
        return parent_min_version
