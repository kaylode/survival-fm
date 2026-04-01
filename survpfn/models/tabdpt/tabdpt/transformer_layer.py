import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import GELU, LayerNorm, Linear


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer use in TabDPT."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int) -> None:
        """
        Args:
            embed_dim (int): Dimension of the embedding.
            num_heads (int): Number of attention heads.
            ff_dim (int): Dimension of the feed-forward network.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_norm = LayerNorm(embed_dim)
        self.ff_norm = LayerNorm(embed_dim)
        self.ff = nn.Sequential(Linear(embed_dim, ff_dim), GELU(), Linear(ff_dim, embed_dim))
        self.q_norm = LayerNorm(self.head_dim)
        self.k_norm = LayerNorm(self.head_dim)

    def forward(self, x: torch.Tensor, eval_pos: int, **kwargs) -> torch.Tensor:
        """
        Args:
            x (torch.tensor): Input tensor of shape (L, B, D) where B is batch size, L is sequence length, and D is embedding dimension.
            eval_pos (int): Evaluation position used for slicing attention keys and values.
        Returns:
            torch.tensor: Output tensor of the same shape as input.
        """
        # switch to (B, L, D) for attention computation
        x = x.transpose(0, 1)
        B, L, _ = x.size()

        # Normalize the input
        h = self.attn_norm(x)

        # project to query, key, and value with linear layers
        q = self.q_proj(h)
        # slice the key and value projections to the evaluation position
        k, v = self.kv_proj(h[:, :eval_pos]).chunk(2, dim=-1)

        # reshape and transpose for multi-head attention
        # q: (B, L, D) -> (B, L, num_heads, head_dim)
        # k: (B, eval_pos, D) -> (B, num_heads, eval_pos, head_dim)
        # v: (B, eval_pos, D) -> (B, num_heads, eval_pos, head_dim)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, eval_pos, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, eval_pos, self.num_heads, self.head_dim).transpose(1, 2)

        # apply layer normalization to query and key
        q, k = self.q_norm(q), self.k_norm(k)

        # compute scaled dot-product attention
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        attn = self.out_proj(attn.reshape(B, L, self.num_heads * self.head_dim))

        # residual connection and feed-forward network
        x = x + attn
        x = x + self.ff(self.ff_norm(x))

        # back to (L, B, D) for output
        return x.transpose(0, 1)
