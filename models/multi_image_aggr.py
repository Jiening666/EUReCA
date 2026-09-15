import torch
import torch.nn as nn


class ViewWeightAggregator(nn.Module):
    """
    O(N) scored aggregation — no attention matrix.

    Learns a per-view importance scalar from each view's own features, then
    does a softmax-weighted sum.  No O(N²) cross-view attention cost.

    Suitable for high-resolution (mid / fine) skip-connection aggregation
    where the coarse level already performs full cross-view reasoning.
    With N~25 views and 131 K fine voxels this runs as a single matrix
    multiply with no chunking or gradient checkpointing needed.
    """
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        self.score_proj = nn.Linear(embed_dim, 1, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):          # x: (B, N, C)
        w = self.score_proj(x).softmax(dim=1)   # (B, N, 1)
        return self.fc((x * w).sum(dim=1))       # (B, C)


class MaxPoolAggregator(nn.Module):
    def __init__(self, embed_dim=768, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, N, embed_dim)
        Returns:
            combined_feature: Tensor of shape (Batch, embed_dim)
        """
        # Max pooling over the N views dimension
        return x.max(dim=1).values


class MultiImageAggregator(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim

        # CLS token that absorbs information from all N images
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer: O(L²) attention is fine — sequence length is just N+1 ≈ 25,
        # so the attention matrix is ~25×25 per sample (negligible memory).
        # Mamba was slower here because its pscan overhead dominates at short L,
        # and pscan's (B, L_padded, d_inner, d_state) tensors caused OOM during
        # gradient-checkpoint recomputation at large chunk sizes.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        

        # Optional: Final MLP to process the combined feature
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, N, embed_dim)
        Returns:
            combined_feature: Tensor of shape (Batch, embed_dim)
        """
        # # 1. Expand CLS token to match batch size
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, C)

        # 2. Concatenate CLS token with the image features
        # New shape: (B, N + 1, C)
        x = torch.cat((cls_tokens, x), dim=1)

        # 3. Pass through Transformer
        # All tokens (images + CLS) attend to each other
        x = self.transformer(x)

        # 4. Extract only the CLS token (the first element)
        combined_feature = x[:, 0]

        return self.fc(combined_feature)

        # # Attention pooling: mean of views as self-conditioned query
        # q = self.to_q(x.mean(dim=1, keepdim=True))  # (B, 1, C)
        # k = self.to_k(x)                             # (B, N, C)
        # v = self.to_v(x)                             # (B, N, C)
        # attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, 1, N)
        # attn = attn.softmax(dim=-1)
        # out = (attn @ v).squeeze(1)                  # (B, C)
        # return self.out_proj(out)

# class MultiImageAggregator(nn.Module):
#     def __init__(self, embed_dim=768, num_heads=8, num_layers=2,
#                  num_queries=4, dropout=0.1):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.num_queries = num_queries

#         # Learned query tokens — replace the unstable mean query.
#         # Multiple queries let different "experts" attend to different aspects;
#         # we fuse them at the end. This is the Perceiver/DETR pattern.
#         self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)

#         # Per-view quality gate: soft scalar in (0,1) per view from the view's
#         # own feature. Down-weights bad views BEFORE attention so they can't
#         # poison the keys/values, but never zeroes them out completely.
#         self.view_gate = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim // 4),
#             nn.GELU(),
#             nn.Linear(embed_dim // 4, 1),
#         )

#         # Iterative cross-attention blocks. Each block: queries attend to views,
#         # then self-attend to each other, with residuals + LayerNorm.
#         self.layers = nn.ModuleList([
#             _AggregatorBlock(embed_dim, num_heads, dropout)
#             for _ in range(num_layers)
#         ])

#         # Per-query LayerNorm before flattening (standard transformer pattern).
#         self.fuse_norm_q = nn.LayerNorm(embed_dim)
#         # LayerNorm for the statistical pools (mean / min / max over views).
#         #self.fuse_norm_stats = nn.LayerNorm(embed_dim)

#         # Fuse: [K query tokens] ++ [mean] ++ [min] ++ [max]  ->  single vector.
#         # Total tokens fused = num_queries + 3 statistical pools.
#         fuse_in = embed_dim * (num_queries + 3)
#         self.fuse = nn.Sequential(
#             nn.Linear(fuse_in, embed_dim * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(embed_dim * 2, embed_dim),
#         )

#     def forward(self, x):
#         """
#         x: (B, N, C)  view features  (no masking — always N consistent views)
#         returns: (B, C)
#         """
#         B, N, C = x.shape

#         # 1. Per-view soft gating (never hard-zero, just downweight bad views).
#         #gate = torch.sigmoid(self.view_gate(x))  # (B, N, 1)
#         ##x_gated = x * gate

#         # 2. Iteratively refine learned queries against the gated views.
#         q = self.queries.expand(B, -1, -1)  # (B, K, C)
#         for layer in self.layers:
#             #q = layer(q, x_gated)
#             q = layer(q, x)

#         # 3. Statistical pools over the ORIGINAL (un-gated) views — these are
#         #    consistent reference features that don't depend on the gate
#         #    learning the right thing. Cheap, robust, complementary to attention.
#         x_mean = x.mean(dim=1)                   # (B, C)
#         x_min  = x.amin(dim=1)                   # (B, C)
#         x_max  = x.amax(dim=1)                   # (B, C)
#         stats = torch.stack([x_mean, x_min, x_max], dim=1)  # (B, 3, C)

#         # 4. Normalize each token group, then concat and fuse.
#         q = self.fuse_norm_q(q)                  # (B, K, C)
#         #stats = self.fuse_norm_stats(stats)      # (B, 3, C)
#         fused_in = torch.cat([q, stats], dim=1)  # (B, K+3, C)
#         return self.fuse(fused_in.flatten(1))    # (B, C)


# class _AggregatorBlock(nn.Module):
#     """Cross-attn (queries <- views) + self-attn (queries <-> queries) + FFN."""
#     def __init__(self, embed_dim, num_heads, dropout):
#         super().__init__()
#         self.norm_q1 = nn.LayerNorm(embed_dim)
#         self.norm_kv = nn.LayerNorm(embed_dim)
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim, num_heads, dropout=dropout, batch_first=True
#         )
#         self.norm_q2 = nn.LayerNorm(embed_dim)
#         self.self_attn = nn.MultiheadAttention(
#             embed_dim, num_heads, dropout=dropout, batch_first=True
#         )
#         self.norm_q3 = nn.LayerNorm(embed_dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim * 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(embed_dim * 4, embed_dim),
#             nn.Dropout(dropout),
#         )

#     def forward(self, q, kv):
#         # Cross-attention: queries pull info from views (pre-norm + residual)
#         q_n = self.norm_q1(q)
#         kv_n = self.norm_kv(kv)
#         q = q + self.cross_attn(q_n, kv_n, kv_n, need_weights=False)[0]
#         # Self-attention among queries (lets them specialize / not duplicate)
#         q_n = self.norm_q2(q)
#         q = q + self.self_attn(q_n, q_n, q_n, need_weights=False)[0]
#         # FFN
#         q = q + self.ffn(self.norm_q3(q))
#         return q