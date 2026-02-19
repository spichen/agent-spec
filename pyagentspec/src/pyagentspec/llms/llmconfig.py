# Copyright © 2025 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""This module defines the base class for all LLM configuration component."""

from typing import Any, ClassVar, Optional, Type, Union

from pydantic import TypeAdapter
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaMode, JsonSchemaValue

from pyagentspec.component import Component
from pyagentspec.llms.llmgenerationconfig import LlmGenerationConfig
from pyagentspec.versioning import AgentSpecVersionEnum

DEFAULT_REF_TEMPLATE = "#/$defs/{model}"


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

    default_generation_parameters: Optional[LlmGenerationConfig] = None
    """Parameters used for the generation call of this LLM"""

    @classmethod
    def model_json_schema(
        cls: Type["LlmConfig"],
        by_alias: bool = True,
        ref_template: str = DEFAULT_REF_TEMPLATE,
        schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
        mode: JsonSchemaMode = "validation",
        only_core_components: bool = False,
        **kwargs: Any,
    ) -> JsonSchemaValue:
        if cls is LlmConfig:
            # Include self + all subclasses in the schema, since LlmConfig is
            # a concrete class that also serves as the base for all LLM configs.
            all_subclasses = cls._get_all_subclasses(only_core_components=only_core_components)
            adapter = TypeAdapter(Union[tuple([cls, *all_subclasses])])  # type: ignore
            json_schema = adapter.json_schema(by_alias=by_alias, mode=mode)

            from pyagentspec.component import (
                _add_agentspec_version_field,
                _add_references,
                replace_abstract_models_and_hierarchical_definitions,
            )

            json_schema = replace_abstract_models_and_hierarchical_definitions(
                json_schema, mode, only_core_components=only_core_components, by_alias=by_alias
            )
            json_schema = _add_references(json_schema, cls.__name__)
            json_schema = _add_agentspec_version_field(json_schema)
            return json_schema
        return super().model_json_schema(
            by_alias=by_alias,
            ref_template=ref_template,
            schema_generator=schema_generator,
            mode=mode,
            only_core_components=only_core_components,
            **kwargs,
        )

    def _infer_min_agentspec_version_from_configuration(self) -> AgentSpecVersionEnum:
        parent_min_version = super()._infer_min_agentspec_version_from_configuration()
        # Bare LlmConfig is a v26_2_0 feature — it was abstract before.
        # Subclasses handle their own versioning independently.
        if type(self) is LlmConfig:
            return max(AgentSpecVersionEnum.v26_2_0, parent_min_version)
        return parent_min_version
