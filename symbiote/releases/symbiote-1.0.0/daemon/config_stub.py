"""
Simplified configuration stub that works without pydantic.
Used when pydantic is not available.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import yaml


class HotkeysConfig:
    def __init__(self, capture: str = "Ctrl+Shift+Space", search: str = "Ctrl+Shift+F"):
        self.capture = capture
        self.search = search


class IndicesConfig:
    def __init__(self, fts: bool = True, vector: bool = True, analytics: bool = True):
        self.fts = fts
        self.vector = vector
        self.analytics = analytics


class EmbeddingConfig:
    def __init__(self, model: str = "all-MiniLM-L6-v2", dim: int = 384, 
                 chunk_tokens: int = 512, overlap_tokens: int = 64):
        self.model = model
        self.dim = dim
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens


class LLMConfig:
    def __init__(self, provider: str = "ollama", model: str = "qwen2.5:3b",
                 max_tokens: int = 256, temperature: float = 0.2):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature


class PrivacyConfig:
    def __init__(self, allow_cloud: bool = False, redaction_default: bool = True,
                 mask_pii_default: bool = True):
        self.allow_cloud = allow_cloud
        self.redaction_default = redaction_default
        self.mask_pii_default = mask_pii_default


class PerformanceConfig:
    def __init__(self, max_search_workers: int = 3, synthesis_interval_min: int = 5,
                 memory_budget_mb: int = 1500):
        self.max_search_workers = max_search_workers
        self.synthesis_interval_min = synthesis_interval_min
        self.memory_budget_mb = memory_budget_mb


class SynthesisConfig:
    def __init__(self, interval_min: int = 5, suggest_link_threshold: float = 0.70,
                 max_themes: int = 3):
        self.interval_min = interval_min
        self.suggest_link_threshold = suggest_link_threshold
        self.max_themes = max_themes
        
        # Validate threshold
        if not 0 <= suggest_link_threshold <= 1:
            raise ValueError("suggest_link_threshold must be between 0 and 1")


class ConfigStub:
    """Simplified configuration that works without pydantic."""
    
    def __init__(self, vault_path: str, **kwargs):
        # Required field
        self.vault_path = Path(vault_path).expanduser().resolve()
        
        # Create vault path if it doesn't exist
        if not self.vault_path.exists():
            print(f"Creating vault path: {self.vault_path}")
            self.vault_path.mkdir(parents=True, exist_ok=True)
        
        # Optional configuration sections
        self.hotkeys = HotkeysConfig(**kwargs.get('hotkeys', {}))
        self.indices = IndicesConfig(**kwargs.get('indices', {}))
        self.embedding = EmbeddingConfig(**kwargs.get('embedding', {}))
        self.llm = LLMConfig(**kwargs.get('llm', {}))
        self.privacy = PrivacyConfig(**kwargs.get('privacy', {}))
        self.performance = PerformanceConfig(**kwargs.get('performance', {}))
        self.synthesis = SynthesisConfig(**kwargs.get('synthesis', {}))
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "ConfigStub":
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
                # Create minimal default config
                print("No config file found, using defaults")
                return cls(vault_path="~/symbiote-vault")
        
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    def save(self, config_path: Path) -> None:
        """Save configuration to YAML file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'vault_path': str(self.vault_path),
            'hotkeys': {
                'capture': self.hotkeys.capture,
                'search': self.hotkeys.search
            },
            'indices': {
                'fts': self.indices.fts,
                'vector': self.indices.vector,
                'analytics': self.indices.analytics
            },
            'embedding': {
                'model': self.embedding.model,
                'dim': self.embedding.dim,
                'chunk_tokens': self.embedding.chunk_tokens,
                'overlap_tokens': self.embedding.overlap_tokens
            },
            'llm': {
                'provider': self.llm.provider,
                'model': self.llm.model,
                'max_tokens': self.llm.max_tokens,
                'temperature': self.llm.temperature
            },
            'privacy': {
                'allow_cloud': self.privacy.allow_cloud,
                'redaction_default': self.privacy.redaction_default,
                'mask_pii_default': self.privacy.mask_pii_default
            },
            'performance': {
                'max_search_workers': self.performance.max_search_workers,
                'synthesis_interval_min': self.performance.synthesis_interval_min,
                'memory_budget_mb': self.performance.memory_budget_mb
            },
            'synthesis': {
                'interval_min': self.synthesis.interval_min,
                'suggest_link_threshold': self.synthesis.suggest_link_threshold,
                'max_themes': self.synthesis.max_themes
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False)