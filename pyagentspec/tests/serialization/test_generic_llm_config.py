# Copyright © 2025 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

import pytest
import yaml

from pyagentspec.agent import Agent
from pyagentspec.llms import (
    AuthConfig,
    GenerationConfig,
    GenericLlmConfig,
    LlmConfig,
    LlmEndpoint,
    LlmGenerationConfig,
    OpenAiConfig,
    ProviderConfig,
    VllmConfig,
)
from pyagentspec.serialization import AgentSpecDeserializer, AgentSpecSerializer

from .conftest import assert_serialized_representations_are_equal


def test_generic_llm_config_is_llm_config() -> None:
    config = GenericLlmConfig(
        id="test",
        name="test",
        model_id="gpt-4",
        provider=ProviderConfig(type="openai"),
    )
    assert isinstance(config, LlmConfig)


def test_generation_config_is_llm_generation_config() -> None:
    gen_config = GenerationConfig(temperature=0.7, top_k=50)
    assert isinstance(gen_config, LlmGenerationConfig)


def test_minimal_config_serialization_round_trip() -> None:
    config = GenericLlmConfig(
        id="minimal",
        name="minimal-llm",
        model_id="gpt-4",
        provider=ProviderConfig(type="openai"),
    )
    serializer = AgentSpecSerializer()
    serialized = serializer.to_yaml(config)
    assert "component_type: GenericLlmConfig" in serialized
    assert "model_id: gpt-4" in serialized

    deserialized = AgentSpecDeserializer().from_yaml(serialized)
    assert config == deserialized


def test_full_config_serialization_round_trip() -> None:
    config = GenericLlmConfig(
        id="full",
        name="full-llm",
        model_id="claude-3-opus",
        provider=ProviderConfig(
            type="anthropic",
            endpoint="https://api.anthropic.com",
            api_version="2024-01",
        ),
        auth=AuthConfig(type="api_key", credential_ref="ANTHROPIC_API_KEY"),
        default_generation_parameters=GenerationConfig(
            max_tokens=1024,
            temperature=0.7,
            top_k=40,
            stop_sequences=["END"],
            frequency_penalty=0.5,
        ),
        fallback=[
            LlmEndpoint(
                model_id="gpt-4",
                provider=ProviderConfig(type="openai"),
                auth=AuthConfig(type="api_key", credential_ref="OPENAI_API_KEY"),
            ),
        ],
        provider_extensions={"custom_header": "x-custom-value"},
    )

    serializer = AgentSpecSerializer()
    serialized = serializer.to_yaml(config)

    deserialized = AgentSpecDeserializer().from_yaml(
        serialized,
        components_registry={
            "full.auth": AuthConfig(type="api_key", credential_ref="ANTHROPIC_API_KEY"),
        },
    )
    assert config == deserialized


def test_auth_excluded_from_exports() -> None:
    config = GenericLlmConfig(
        id="auth-test",
        name="auth-llm",
        model_id="gpt-4",
        provider=ProviderConfig(type="openai"),
        auth=AuthConfig(type="api_key", credential_ref="MY_KEY"),
    )
    serializer = AgentSpecSerializer()
    serialized = serializer.to_yaml(config)
    parsed = yaml.safe_load(serialized)

    assert parsed["auth"] == {"$component_ref": "auth-test.auth"}
    assert "api_key" not in str(parsed["auth"])


def test_auth_restored_from_components_registry() -> None:
    auth = AuthConfig(type="api_key", credential_ref="MY_KEY")
    config = GenericLlmConfig(
        id="restore-test",
        name="restore-llm",
        model_id="gpt-4",
        provider=ProviderConfig(type="openai"),
        auth=auth,
    )
    serializer = AgentSpecSerializer()
    serialized = serializer.to_yaml(config)

    deserialized = AgentSpecDeserializer().from_yaml(
        serialized,
        components_registry={"restore-test.auth": auth},
    )
    assert deserialized.auth == auth


def test_provider_config_extra_fields_pass_through() -> None:
    provider = ProviderConfig(
        type="aws_bedrock",
        region="us-east-1",  # type: ignore
        deployment_name="my-deployment",  # type: ignore
    )
    config = GenericLlmConfig(
        id="extra-test",
        name="extra-llm",
        model_id="anthropic.claude-v2",
        provider=provider,
    )
    serializer = AgentSpecSerializer()
    serialized = serializer.to_yaml(config)
    assert "region: us-east-1" in serialized
    assert "deployment_name: my-deployment" in serialized

    deserialized = AgentSpecDeserializer().from_yaml(serialized)
    assert deserialized.provider.region == "us-east-1"  # type: ignore
    assert deserialized.provider.deployment_name == "my-deployment"  # type: ignore


def test_generic_llm_config_in_agent() -> None:
    llm_config = GenericLlmConfig(
        id="agent-llm",
        name="agent-llm",
        model_id="gpt-4",
        provider=ProviderConfig(type="openai"),
    )
    agent = Agent(
        id="agent1",
        name="test-agent",
        llm_config=llm_config,
        system_prompt="You are a helpful assistant.",
    )
    serializer = AgentSpecSerializer()
    serialized = serializer.to_yaml(agent)
    assert "GenericLlmConfig" in serialized

    deserialized = AgentSpecDeserializer().from_yaml(serialized)
    assert deserialized.llm_config == llm_config


def test_fallback_endpoints_round_trip() -> None:
    config = GenericLlmConfig(
        id="fallback-test",
        name="fallback-llm",
        model_id="primary-model",
        provider=ProviderConfig(type="openai"),
        fallback=[
            LlmEndpoint(
                model_id="fallback-1",
                provider=ProviderConfig(type="anthropic"),
            ),
            LlmEndpoint(
                model_id="fallback-2",
                provider=ProviderConfig(type="aws_bedrock", region="us-west-2"),  # type: ignore
                provider_extensions={"timeout": 30},
            ),
        ],
    )
    serializer = AgentSpecSerializer()
    serialized = serializer.to_yaml(config)

    deserialized = AgentSpecDeserializer().from_yaml(serialized)
    assert len(deserialized.fallback) == 2
    assert deserialized.fallback[0].model_id == "fallback-1"
    assert deserialized.fallback[1].provider_extensions == {"timeout": 30}


def test_provider_extensions_round_trip() -> None:
    config = GenericLlmConfig(
        id="ext-test",
        name="ext-llm",
        model_id="gpt-4",
        provider=ProviderConfig(type="openai"),
        provider_extensions={"custom_key": "custom_value", "nested": {"a": 1}},
    )
    serializer = AgentSpecSerializer()
    serialized = serializer.to_yaml(config)

    deserialized = AgentSpecDeserializer().from_yaml(serialized)
    assert deserialized.provider_extensions == {"custom_key": "custom_value", "nested": {"a": 1}}


def test_min_agentspec_version_is_v26_2_0() -> None:
    config = GenericLlmConfig(
        id="version-test",
        name="version-llm",
        model_id="gpt-4",
        provider=ProviderConfig(type="openai"),
    )
    assert config.min_agentspec_version.value == "26.2.0"


def test_v1_classes_still_work() -> None:
    vllm_config = VllmConfig(
        id="vllm-v1",
        name="vllm-test",
        model_id="model1",
        url="http://some.where",
        default_generation_parameters=LlmGenerationConfig(temperature=0.3),
    )
    openai_config = OpenAiConfig(
        id="openai-v1",
        name="openai-test",
        model_id="gpt-4",
        api_key="secret",
    )

    serializer = AgentSpecSerializer()

    serialized_vllm = serializer.to_yaml(vllm_config)
    deserialized_vllm = AgentSpecDeserializer().from_yaml(serialized_vllm)
    assert_serialized_representations_are_equal(
        serialized_vllm, serializer.to_yaml(deserialized_vllm)
    )

    serialized_openai = serializer.to_yaml(openai_config)
    deserialized_openai = AgentSpecDeserializer().from_yaml(
        serialized_openai,
        components_registry={"openai-v1.api_key": "secret"},
    )
    assert openai_config == deserialized_openai


def test_resolve_credential_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_API_KEY", "secret-from-env")
    auth = AuthConfig(type="api_key", credential_ref="$env:MY_API_KEY")
    assert auth.resolve_credential() == "secret-from-env"


def test_resolve_credential_env_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_VAR", raising=False)
    auth = AuthConfig(type="api_key", credential_ref="$env:MISSING_VAR")
    with pytest.raises(ValueError, match="MISSING_VAR"):
        auth.resolve_credential()


def test_resolve_credential_literal() -> None:
    auth = AuthConfig(type="api_key", credential_ref="sk-abc123")
    assert auth.resolve_credential() == "sk-abc123"


def test_resolve_credential_none() -> None:
    auth = AuthConfig(type="api_key")
    assert auth.resolve_credential() == ""


def test_llm_endpoint_auth_is_sensitive() -> None:
    from pydantic.fields import FieldInfo

    from pyagentspec.sensitive_field import is_sensitive_field

    field: FieldInfo = LlmEndpoint.model_fields["auth"]
    assert is_sensitive_field(field)
