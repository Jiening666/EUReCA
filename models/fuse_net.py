import torch
import torch.nn as nn

class AdaLNBlock(nn.Module):
    def __init__(self, emb_dim, conditioning_dim):
        """
        Args:
            emb_dim: The hidden dimension of your main input (x).
            conditioning_dim: The dimension of the signal you are regressing on (y).
        """
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim, elementwise_affine=False)
        
        # We regress two parameters: scale (gamma) and shift (beta)
        # Often initialized to zero so the block starts as a standard identity identity-ish mapping
        self.regressor = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_dim),
            nn.SiLU(),  # Sigmoid Linear Unit (common in modern generative models)
            nn.Linear(hidden_dim, 2 * emb_dim)
        )
        
        # Zero-initialize the final layer so it starts as a standard identity-like norm
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def forward(self, x, condition):
        """
        x: [Batch, Sequence, emb_dim]
        condition: [Batch, conditioning_dim]
        """
        # 1. Project the conditioning signal
        # [Batch, 2 * emb_dim]
        params = self.regressor(condition) 
        
        # 2. Split into scale (gamma) and shift (beta)
        # Reshape for broadcasting: [Batch, 1, emb_dim]
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        # 3. Apply Norm and Scale/Shift
        # We use (1 + gamma) so that if gamma is 0, the scale is 1
        x = self.norm(x)
        return x * (1 + gamma) + beta