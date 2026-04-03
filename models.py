"""
Neural Network Models for MCF7 Signaling Surrogate Modeling.

This module provides MLP and Residual MLP architectures for learning
phosphoprotein dynamics from the HPN-DREAM MCF7 dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from dataclasses import dataclass, field


# MCF7 Dataset Constants (defaults, actual values determined from data)
MCF7_NUM_STIMULI = 8      # EGF, FGF1, HGF, IGF1, Insulin, NRG1, PBS, Serum
MCF7_NUM_INHIBITORS = 3   # GSK690693, GSK690693_GSK1120212, PD173074
MCF7_NUM_CELLLINE = 1     # MCF7
MCF7_NUM_TIME = 1         # DA:ALL
MCF7_NUM_PROTEINS = 41    # Phosphoprotein measurements (updated from actual data)

MCF7_INPUT_SIZE = MCF7_NUM_CELLLINE + MCF7_NUM_STIMULI + MCF7_NUM_INHIBITORS + MCF7_NUM_TIME  # 13
MCF7_OUTPUT_SIZE = MCF7_NUM_PROTEINS  # 41


@dataclass
class MLPConfig:
    """Configuration for MLP model."""
    input_size: int = MCF7_INPUT_SIZE
    output_size: int = MCF7_OUTPUT_SIZE
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.0


@dataclass
class ResidualMLPConfig:
    """Configuration for Residual MLP model."""
    input_size: int = MCF7_INPUT_SIZE
    output_size: int = MCF7_OUTPUT_SIZE
    hidden_dim: int = 128
    num_blocks: int = 3
    dropout: float = 0.0


class MLP(nn.Module):
    """
    Multi-layer Perceptron for MCF7 signaling surrogate modeling.
    
    Predicts phosphoprotein levels given treatment conditions and time.
    
    Parameters
    ----------
    input_size : int
        Input dimension (default: 13 = 1 cell line + 8 stimuli + 3 inhibitors + 1 time)
    output_size : int
        Output dimension (default: 41 phosphoproteins)
    hidden_sizes : List[int]
        List of hidden layer sizes
    dropout : float
        Dropout probability for regularization
    """
    
    def __init__(
        self, 
        input_size: int = MCF7_INPUT_SIZE, 
        output_size: int = MCF7_OUTPUT_SIZE, 
        hidden_sizes: Optional[List[int]] = None, 
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes or [256, 128]
        self.dropout_rate = dropout
        
        # Build network layers
        layers = []
        prev_size = input_size

        for hidden_size in self.hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.GELU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, input_size) -> (batch, output_size)"""
        return self.network(x)
    
    def get_hidden_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get output of last hidden layer before final projection."""
        for layer in self.network[:-1]:
            x = layer(x)
        return x

    def get_first_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Get first hidden layer representation."""
        x = self.network[0](x)  # First linear
        if len(self.network) > 1 and isinstance(self.network[1], nn.GELU):
            x = self.network[1](x)
        return x
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        return (f"MLP(in={self.input_size}, out={self.output_size}, "
                f"hidden={self.hidden_sizes}, dropout={self.dropout_rate})")


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm."""
    
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x + self.dropout(F.gelu(self.linear(x))))


class ResidualMLP(nn.Module):
    """
    Residual MLP for MCF7 signaling surrogate modeling.
    
    Architecture: Input -> Linear -> GELU -> [ResidualBlock] x N -> Linear -> Output
    
    Parameters
    ----------
    input_size : int
        Input dimension (default: 13)
    output_size : int
        Output dimension (default: 41)
    hidden_dim : int
        Hidden dimension for all layers
    num_blocks : int
        Number of residual blocks
    dropout : float
        Dropout probability
    """
    
    def __init__(
        self, 
        input_size: int = MCF7_INPUT_SIZE, 
        output_size: int = MCF7_OUTPUT_SIZE, 
        hidden_dim: int = 128, 
        num_blocks: int = 3,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout_rate = dropout
        
        self.input_proj = nn.Linear(input_size, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, input_size) -> (batch, output_size)"""
        x = F.gelu(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)
    
    def get_hidden_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get output after all residual blocks."""
        x = F.gelu(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        return x

    def get_first_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Get first hidden layer representation."""
        return F.gelu(self.input_proj(x))
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        return (f"ResidualMLP(in={self.input_size}, out={self.output_size}, "
                f"hidden={self.hidden_dim}, blocks={self.num_blocks})")


def create_model(
    model_type: str = 'mlp',
    input_size: int = MCF7_INPUT_SIZE,
    output_size: int = MCF7_OUTPUT_SIZE,
    **kwargs
) -> nn.Module:
    """
    Factory function to create MCF7 signaling models.
    
    Parameters
    ----------
    model_type : str
        'mlp' or 'residual_mlp'
    input_size : int
        Input dimension (default: 13)
    output_size : int
        Output dimension (default: 41)
    **kwargs
        Additional model arguments
        
    Returns
    -------
    nn.Module
        Instantiated model
    """
    model_type = model_type.lower()
    
    if model_type == 'mlp':
        return MLP(input_size, output_size, **kwargs)
    elif model_type in ('residual_mlp', 'resmlp', 'res_mlp'):
        return ResidualMLP(input_size, output_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'mlp' or 'residual_mlp'")


if __name__ == "__main__":
    print("=" * 50)
    print("MCF7 Signaling Model Demo")
    print(f"Input: {MCF7_INPUT_SIZE} (treatments + time)")
    print(f"Output: {MCF7_OUTPUT_SIZE} (phosphoproteins)")
    print("=" * 50)
    
    # Test batch
    x = torch.randn(32, MCF7_INPUT_SIZE)
    
    # MLP
    mlp = MLP(hidden_sizes=[256, 128], dropout=0.1)
    print(f"\n{mlp}")
    print(f"Parameters: {mlp.count_parameters():,}")
    print(f"Output shape: {mlp(x).shape}")
    
    # ResidualMLP
    res_mlp = ResidualMLP(hidden_dim=128, num_blocks=3, dropout=0.1)
    print(f"\n{res_mlp}")
    print(f"Parameters: {res_mlp.count_parameters():,}")
    print(f"Output shape: {res_mlp(x).shape}")
    
    print("\n✓ All tests passed!")