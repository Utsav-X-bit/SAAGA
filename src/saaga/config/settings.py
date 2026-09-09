"""
Configuration management and Pydantic v2 settings schemas for SAAGA.
===================================================================
Provides type-safe configuration for models, providers, runtime limits,
memory backends, and experiment reproduction parameters.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelConfig(BaseModel):
    """Configuration for an individual model backend in the SAAGA pipeline.

    Attributes:
        provider: Provider identifier ('openai', 'vllm', 'hf', 'ollama', 'mock').
        model_id: Model name, checkpoint tag, or local file directory.
        api_base: Optional base URL for HTTP endpoints.
        api_key: Optional authorization token or secret.
        temperature: Sampling temperature (higher = more exploratory).
        top_p: Nucleus sampling probability cutoff.
        max_tokens: Maximum tokens to generate per completion.
        gpu_memory_utilization: Fraction of GPU VRAM allocated for in-process engines.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    provider: str = Field(
        default="openai",
        description="Backend provider engine ('openai', 'vllm', 'hf', 'ollama', 'mock')",
    )
    model_id: str = Field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        description="Model identifier, repo path, or local directory",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Base URL for remote API backends (e.g. http://localhost:8000/v1)",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API token or authentication key",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability",
    )
    max_tokens: int = Field(
        default=1024,
        gt=0,
        description="Maximum new generation tokens",
    )
    gpu_memory_utilization: float = Field(
        default=0.90,
        ge=0.1,
        le=1.0,
        description="GPU VRAM reservation ratio for in-process engines (vLLM)",
    )

    @classmethod
    def coerce(cls, value: Union[str, dict[str, Any], ModelConfig]) -> ModelConfig:
        """Coerce a string model_id, dictionary, or existing instance into ModelConfig."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(model_id=value)
        if isinstance(value, dict):
            return cls(**value)
        raise ValueError(f"Cannot coerce value of type {type(value)} to ModelConfig: {value}")


class SAAGAConfig(BaseModel):
    """Global configuration for a SAAGA red teaming experiment or benchmark run.

    Attributes:
        victim_model: Configuration for the victim LLM being evaluated.
        planner_model: Configuration for the tactical attack planner model.
        generator_model: Configuration for the prompt mutation/generation model.
        max_attempts: Maximum attack iterations per scenario before termination.
        enable_mutation_fallback: Whether to activate mutation fallback on repeated failures.
        rag_enabled: Whether to query RAG for past successful attack templates.
        kb_enabled: Whether to consult the empirical strategy knowledge base.
        output_dir: Directory where execution traces and summaries are stored.
        seed: Random seed for deterministic reproducibility.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    victim_model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            provider="openai",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            temperature=0.0,
        ),
        description="Target model configuration",
    )
    planner_model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            provider="openai",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            temperature=0.7,
        ),
        description="Attack planner model configuration",
    )
    generator_model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            provider="openai",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            temperature=0.8,
        ),
        description="Attack generator model configuration",
    )
    max_attempts: int = Field(
        default=10,
        gt=0,
        description="Maximum turns per challenge scenario",
    )
    enable_mutation_fallback: bool = Field(
        default=True,
        description="Enable multi-armed mutation fallback on near-miss attempts",
    )
    rag_enabled: bool = Field(
        default=True,
        description="Query past successful vectors via RAG",
    )
    kb_enabled: bool = Field(
        default=True,
        description="Incorporate empirical strategy effectiveness matrix",
    )
    output_dir: str = Field(
        default="results",
        description="Root directory for output traces and run artifacts",
    )
    seed: int = Field(
        default=42,
        description="Random seed for repeatable stochastic runs",
    )

    @field_validator("victim_model", "planner_model", "generator_model", mode="before")
    @classmethod
    def _validate_model_config(cls, v: Any) -> ModelConfig:
        return ModelConfig.coerce(v)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> SAAGAConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to .yaml or .yml configuration file.

        Returns:
            Instantiated and validated SAAGAConfig.
        """
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

        return cls(**raw_data)

    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
        """Serialize configuration to a YAML string, optionally saving to a file.

        Args:
            path: Optional file destination.

        Returns:
            YAML string representation.
        """
        data = self.model_dump(mode="json")
        yaml_str = yaml.dump(data, sort_keys=False, default_flow_style=False)

        if path is not None:
            dest = Path(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "w", encoding="utf-8") as f:
                f.write(yaml_str)

        return yaml_str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SAAGAConfig:
        """Create config from dictionary."""
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary representation."""
        return self.model_dump()

    @classmethod
    def from_env(cls, prefix: str = "SAAGA_") -> SAAGAConfig:
        """Construct configuration by reading environment variables.

        Supported environment variables:
          - {prefix}MAX_ATTEMPTS: int
          - {prefix}ENABLE_MUTATION_FALLBACK: bool ('true'/'1'/'yes')
          - {prefix}RAG_ENABLED: bool
          - {prefix}KB_ENABLED: bool
          - {prefix}OUTPUT_DIR: str
          - {prefix}SEED: int
          - {prefix}VICTIM_MODEL / {prefix}VICTIM_MODEL_ID: str
          - {prefix}VICTIM_PROVIDER: str
          - {prefix}VICTIM_API_BASE: str
          - {prefix}VICTIM_API_KEY: str
          - {prefix}PLANNER_MODEL / {prefix}PLANNER_MODEL_ID: str
          - {prefix}PLANNER_PROVIDER: str
          - {prefix}PLANNER_API_BASE: str
          - {prefix}PLANNER_API_KEY: str
          - {prefix}GENERATOR_MODEL / {prefix}GENERATOR_MODEL_ID: str
          - {prefix}GENERATOR_PROVIDER: str
          - {prefix}GENERATOR_API_BASE: str
          - {prefix}GENERATOR_API_KEY: str

        Returns:
            Configured SAAGAConfig instance.
        """
        def get_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key)
            if val is None:
                return default
            return val.lower().strip() in ("1", "true", "yes", "on")

        def get_int(key: str, default: int) -> int:
            val = os.environ.get(key)
            if val is None:
                return default
            try:
                return int(val.strip())
            except ValueError:
                return default

        def build_sub_model(sub_prefix: str, default_model: str, default_temp: float) -> ModelConfig:
            model_id = (
                os.environ.get(f"{prefix}{sub_prefix}_MODEL_ID")
                or os.environ.get(f"{prefix}{sub_prefix}_MODEL")
                or default_model
            )
            provider = os.environ.get(f"{prefix}{sub_prefix}_PROVIDER", "openai")
            api_base = os.environ.get(f"{prefix}{sub_prefix}_API_BASE")
            api_key = os.environ.get(f"{prefix}{sub_prefix}_API_KEY")
            return ModelConfig(
                provider=provider,
                model_id=model_id,
                api_base=api_base,
                api_key=api_key,
                temperature=default_temp,
            )

        victim = build_sub_model("VICTIM", "meta-llama/Meta-Llama-3-8B-Instruct", 0.0)
        planner = build_sub_model("PLANNER", "meta-llama/Meta-Llama-3-8B-Instruct", 0.7)
        generator = build_sub_model("GENERATOR", "meta-llama/Meta-Llama-3-8B-Instruct", 0.8)

        max_attempts = get_int(f"{prefix}MAX_ATTEMPTS", 10)
        enable_fallback = get_bool(f"{prefix}ENABLE_MUTATION_FALLBACK", True)
        rag_enabled = get_bool(f"{prefix}RAG_ENABLED", True)
        kb_enabled = get_bool(f"{prefix}KB_ENABLED", True)
        output_dir = os.environ.get(f"{prefix}OUTPUT_DIR", "results")
        seed = get_int(f"{prefix}SEED", 42)

        return cls(
            victim_model=victim,
            planner_model=planner,
            generator_model=generator,
            max_attempts=max_attempts,
            enable_mutation_fallback=enable_fallback,
            rag_enabled=rag_enabled,
            kb_enabled=kb_enabled,
            output_dir=output_dir,
            seed=seed,
        )
