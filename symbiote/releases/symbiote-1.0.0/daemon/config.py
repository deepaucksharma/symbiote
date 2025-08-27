"""Configuration management for Symbiote."""

from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from .compat import BaseModel, Field, field_validator, logger, Config as ConfigStub


class HotkeysConfig(BaseModel):
    capture: str = "Ctrl+Shift+Space"
    search: str = "Ctrl+Shift+F"


class IndicesConfig(BaseModel):
    fts: bool = True
    vector: bool = True
    analytics: bool = True


class EmbeddingConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"
    dim: int = 384
    chunk_tokens: int = 512
    overlap_tokens: int = 64


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "qwen2.5:3b"
    max_tokens: int = 256
    temperature: float = 0.2


class PrivacyConfig(BaseModel):
    allow_cloud: bool = False
    redaction_default: bool = True
    mask_pii_default: bool = True


class PerformanceConfig(BaseModel):
    max_search_workers: int = 3
    synthesis_interval_min: int = 5
    memory_budget_mb: int = 1500


class SynthesisConfig(BaseModel):
    interval_min: int = 5
    suggest_link_threshold: float = 0.70
    max_themes: int = 3
    
    @field_validator('suggest_link_threshold')
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("suggest_link_threshold must be between 0 and 1")
        return v


class Config(BaseModel):
    """Main configuration for Symbiote daemon."""
    
    vault_path: Path
    hotkeys: HotkeysConfig = Field(default_factory=HotkeysConfig)
    indices: IndicesConfig = Field(default_factory=IndicesConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    
    @field_validator('vault_path')
    @classmethod
    def validate_vault_path(cls, v: Path) -> Path:
        if isinstance(v, str):
            v = Path(v)
        v = v.expanduser().resolve()
        if not v.exists():
            logger.warning(f"Vault path does not exist, will create: {v}")
            v.mkdir(parents=True, exist_ok=True)
        return v
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """Load configuration from YAML file."""
        if config_path is None:
            # Try default locations
            candidates = [
                Path("symbiote.yaml"),
                Path.home() / ".config" / "symbiote" / "config.yaml",
                Path("/etc/symbiote/config.yaml"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    config_path = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"No config file found. Searched: {[str(c) for c in candidates]}"
                )
        
        logger.info(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    def save(self, config_path: Path) -> None:
        """Save configuration to YAML file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.safe_dump(self.model_dump(), f, default_flow_style=False)


# Use stub config if pydantic isn't available
if ConfigStub is not None:
    Config = ConfigStub