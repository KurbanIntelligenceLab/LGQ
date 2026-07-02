"""
Base class for model configurations.
"""

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from quantization.model import ModelConfig, QuantizerConfig, UnifiedAutoEncoder


@dataclass
class LossOutput:
    """Standardized loss output for all models."""
    total_loss: torch.Tensor
    rec_loss: torch.Tensor
    aux_loss: torch.Tensor  # commit_loss, entropy_loss, vq_loss, etc.
    aux_loss_name: str  # Name for logging
    active_codes: int
    perplexity: Optional[float] = None
    recon: Optional[torch.Tensor] = None  # Reconstruction for perceptual/GAN losses


class BaseModelConfig(ABC):
    """Base class for model-specific configurations."""
    
    name: str = "base"
    
    @staticmethod
    @abstractmethod
    def add_model_args(parser: ArgumentParser) -> None:
        """Add model-specific arguments to the parser."""
        pass
    
    @staticmethod
    @abstractmethod
    def create_quantizer_config(args: Namespace) -> QuantizerConfig:
        """Create quantizer config from parsed args."""
        pass
    
    @classmethod
    def create_model(cls, args: Namespace, model_config: ModelConfig) -> UnifiedAutoEncoder:
        """Create model instance from args and unified config."""
        quantizer_config = cls.create_quantizer_config(args)
        return UnifiedAutoEncoder(cls.name, model_config, quantizer_config)
    
    @staticmethod
    @abstractmethod
    def compute_loss(model: nn.Module, x: torch.Tensor) -> LossOutput:
        """Compute loss and return standardized output."""
        pass
    
    @staticmethod
    def get_model_info(args: Namespace) -> str:
        """Return model-specific info string for logging."""
        return ""
