# Copyright © 2025 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""This module defines the extended generation configuration for generic LLM configs."""

from typing import Any, Dict, List, Optional

from pyagentspec.llms.llmgenerationconfig import LlmGenerationConfig


class GenerationConfig(LlmGenerationConfig):
    """
    Extended generation configuration for generic LLM configs.

    Inherits ``max_tokens``, ``temperature``, ``top_p``, and ``extra="allow"``
    from :class:`LlmGenerationConfig`, and adds additional typed fields.
    """

    top_k: Optional[int] = None
    """Top-k sampling parameter"""

    stop_sequences: Optional[List[str]] = None
    """Sequences that stop generation when produced"""

    seed: Optional[int] = None
    """Random seed for reproducible generation"""

    frequency_penalty: Optional[float] = None
    """Penalizes tokens based on their frequency in the generated text"""

    presence_penalty: Optional[float] = None
    """Penalizes tokens based on whether they have appeared in the generated text"""

    response_format: Optional[str] = None
    """Desired response format (e.g. ``"json"``, ``"text"``)"""

    json_schema: Optional[Dict[str, Any]] = None
    """JSON schema for structured output when response_format is ``"json"``"""
