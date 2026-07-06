import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-layer Perceptron for Fisher-KPP model."""
    def __init__(self, input_size=101, output_size=100, hidden_sizes=[512], dropout=0.0):
        """
        Parameters
        ----------
        input_size : int
            Input dimension (default: 101 = time + 100 initial concentrations)
        output_size : int
            Output dimension (default: 100 grid point values)
        hidden_sizes : list
            List of hidden layer sizes
        dropout : float
            Dropout probability
        """
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def get_hidden_representation(self, x):
        """Get the output of the last hidden layer before the final output layer."""
        for layer in self.network[:-1]:
            x = layer(x)
        return x

    def get_first_hidden(self, x):
        """Get first hidden layer representation."""
        x = self.network[0](x)  # First linear layer
        if len(self.network) > 1 and isinstance(self.network[1], (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.GELU)):
            x = self.network[1](x)  # First activation
        return x


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm for deep networks."""
    def __init__(self, dim, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.ln = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        residual = x
        out = F.gelu(self.linear(x))
        out = self.dropout(out)
        out = self.ln(residual + out)
        return out


class ResidualMLP(nn.Module):
    """Residual MLP with LayerNorm for Fisher-KPP model."""
    def __init__(self, input_size=101, output_size=100, hidden_dim=128, num_blocks=3, 
                 dropout=0.0):
        """
        Parameters
        ----------
        input_size : int
            Input dimension (default: 101 = time + 100 initial concentrations)
        output_size : int
            Output dimension (default: 100 grid point values)
        hidden_dim : int
            Hidden dimension for all layers
        num_blocks : int
            Number of residual blocks
        dropout : float
            Dropout probability
        """
        super(ResidualMLP, self).__init__()
        
        self.input_proj = nn.Linear(input_size, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = F.gelu(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)
        return x
    
    def get_hidden_representation(self, x):
        """Get the output of the last hidden layer before the final output layer."""
        x = self.input_proj(x)
        x = F.gelu(x)
        for block in self.blocks:
            x = block(x)
        return x

    def get_first_hidden(self, x):
        """Get first hidden layer representation (after input projection)."""
        x = self.input_proj(x)
        x = F.gelu(x)
        return x


if __name__ == "__main__":
    # Example usage for Fisher-KPP with 100 grid points
    # Input: time (1) + initial condition (100) = 101
    # Output: solution at 100 grid points
    
    model = MLP(input_size=101, output_size=100, hidden_sizes=[512, 256])
    x = torch.randn(10, 101)  # Batch of 10 samples
    output = model(x)
    print(f"MLP output shape: {output.shape}")  # Should be (10, 100)
    
    model_res = ResidualMLP(input_size=101, output_size=100, hidden_dim=128, num_blocks=3)
    output_res = model_res(x)
    print(f"ResidualMLP output shape: {output_res.shape}")  # Should be (10, 100)