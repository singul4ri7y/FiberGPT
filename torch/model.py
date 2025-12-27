from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FiberGPTConfig:
    # GPT2 and GPT3 vocabulary size is 50257, rounded to 64-element boundary
    # for efficiency.
    n_vocab: int = 50304
    context_length: int = 2048
    n_layer: int = 16
    n_head: int = 16
    n_embd: int = 768
    n_hidden: int = n_embd
    ffwd_dim_multiplier: int = 4
    norm_eps: float = 1e-5
    drop_prob: float = 0.0


def _apply_rope(q: torch.Tensor, k: torch.Tensor, rope_cache: torch.Tensor):
    ''' Applies RoPE rotation to query and key. '''

    # cache shape for reshaping later
    q_shape, k_shape = q.shape, k.shape
    q_dtype, k_dtype = q.dtype, k.dtype

    # Get complex domain representation of 2D block vectors
    # (B, T, nh, hs // 2)
    q = torch.view_as_complex(q.float().view(*q.shape[:-1], -1, 2))
    k = torch.view_as_complex(k.float().view(*q.shape[:-1], -1, 2))

    # Make RoPE cache broadcastable.
    rope_cache = rope_cache.unsqueeze(1)

    # Apply rotation in complex plane
    q = torch.view_as_real(q * rope_cache).to(q_dtype)
    k = torch.view_as_real(k * rope_cache).to(k_dtype)

    return q.view(q_shape), k.view(k_shape)


class CausalAttention(nn.Module):
    ''' Casual multi-head attention. '''

    def __init__(self, config: FiberGPTConfig):
        super().__init__()

        # Generally number of features in key, query and values are equal to
        # embedding dimension (or d_model). So, embedding dimension should be
        # perfectly divisible by the number of heads.
        assert config.n_embd % config.n_head == 0
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        
        # Fused linear transformation of key, query and value in the column
        # axis.
        self.attn_qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Dropouts
        self.drop_prob = config.drop_prob
        self.proj_drop = nn.Dropout(self.drop_prob)

    def forward(self, x: torch.Tensor, rope_cache: torch.Tensor):
        B, T, C = x.size()

        qkv = self.attn_qkv(x)
        # Extract query, key and value from fused linear transformation
        q, k, v = qkv.split(self.n_embd, dim=-1)  # (B, T, C), where C = n_embd
        
        # Consider each head is concatenated in column axis. So, we might end up
        # with something like this:
        #   [[ H1, H1, H1, H2, H2, H2 ],
        #    [ H1, H1, H1, H2, H2, H2 ],
        #    [ H1, H1, H1, H2, H2, H2 ]]
        #
        # Consider single batch element and T = 3.
        #
        # Which we want to convert to this:
        #   [[[ H1, H1, H1 ],
        #     [ H1, H1, H1 ],
        #     [ H1, H1, H1 ]],
        #
        #    [[ H2, H2, H2 ],
        #     [ H2, H2, H2 ],
        #     [ H2, H2, H2 ]]]
        #
        # Where H1 and H2 are first and second head elements respectively. Viewing
        # q/k/v as (B, T, nh, hs), where nh = n_head, hs = head_size, will yield
        # something like this:
        #   [[[ H1, H1, H1 ],
        #     [ H2, H2, H2 ]],
        #
        #    [[ H1, H1, H1 ],
        #     [ H2, H2, H2 ]],
        #
        #    [[ H1, H1, H1 ],
        #     [ H2, H2, H2 ]]]
        #
        # So, for the final result, we need to swap 2nd and 3rd dimensions.
        # The final tensor shape should be (B, nh, T, hs).
        q = q.view(B, T, self.n_head, self.head_size)
        k = k.view(B, T, self.n_head, self.head_size)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Apply RoPE before transposition.
        q, k = _apply_rope(q, k, rope_cache)
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        # Apply attention
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.drop_prob if self.training else 0,
            is_causal=True
        )  # (B, nh, T, hs)

        # Re-assemble the output.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_drop(self.proj(y))
    

class FeedForward(nn.Module):
    ''' Feed forward layer in decoder transformer block. '''

    def __init__(self, config: FiberGPTConfig):
        super().__init__()
    
        hidden_dim = int(config.n_hidden * config.ffwd_dim_multiplier)

        # One hidden layer and one projection layer
        self.hl = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor):
        return self.proj(F.silu(self.hl(x)))


class TransformerBlock(nn.Module):
    ''' A single transformer decoder block. '''

    def __init__(self, config: FiberGPTConfig):
        super().__init__()
    
        self.attn = CausalAttention(config)
        self.attn_norm = nn.RMSNorm(config.n_embd, eps=config.norm_eps)
        self.ffwd = FeedForward(config)
        self.ffwd_norm = nn.RMSNorm(config.n_embd, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, rope_cache: torch.Tensor):
        h = x + self.attn(self.attn_norm(x), rope_cache)
        return h + self.ffwd(self.ffwd_norm(h))


class FiberGPT(nn.Module):
    ''' The FiberGPT language model. '''

    def __init__(self, config: FiberGPTConfig):
        super().__init__()

        self.token_embedding = nn.Embedding(config.n_vocab, config.n_embd)

        # Compute rope cache
        self.context_length = config.context_length
        self.register_buffer('rope_cache', self._compute_rope_cache(
            config.n_embd // config.n_head,
            config.context_length
        ))

        # Decoder blocks
        self.dec_blocks = nn.ModuleList(
            [ TransformerBlock(config) for _ in range(config.n_layer) ]
        )

        # Projection
        self.norm = nn.RMSNorm(config.n_embd, eps=config.norm_eps)
        self.proj = nn.Linear(config.n_embd, config.n_vocab, bias=False)

    def _compute_rope_cache(
        self,
        head_size: int,
        context_length: int,
        base: int = 100000,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        ''' Precompute rotation cache of Rotary Positional Embedding. '''

        if device is None:
            device = self.token_embedding.weight.device

        # Gists of RoPE is:
        # [[ x1' ], = [[ cos(m * theta), -sin(m * theta) ],  [[ x1 ]
        #  [ x2' ]]    [ sin(m * theta),  cos(m * theta) ]]   [ x2 ]]
        #
        # For a 2D positional embedding vector.
        
        # theta = 1 / (base ** (2 * i / head_dim)),
        # where i ∈ { 0, ..., (d_model / 2) - 1 },
        # signifying ith vector element.
        two_times_i = torch.arange(
            0, head_size, 2,
            dtype=torch.float32,
            device=device
        )
        theta = 1.0 / (base ** (two_times_i / head_size))

        # RoPE is applied as 2D vector rotations on blocks of 2 elements for nD
        # embedding vectors:
        # [[ x1' ],   [[ cos(m * theta1), -sin(m * theta1),               0,                0 ], [[ x1 ],
        #  [ x2' ], =  [ sin(m * theta1),  cos(m * theta1),               0,                0 ],  [ x2 ],
        #  [ x3' ],    [               0,                0, cos(m * theta2), -sin(m * theta2) ],  [ x3 ],
        #  [ x4' ]]    [               0,                0, sin(m * theta2),  cos(m * theta2) ]]  [ x4 ]]
        #
        # Where m ∈ { 0, 1 }, which can also be written as follows:
        # [[ x1' ],   [[ cos(m * theta1) ],   [[ x1 ],   [[ sin(m * theta1) ],   [[ -x2 ],
        #  [ x2' ], =  [ cos(m * theta1) ], *  [ x2 ], +  [ sin(m * theta1) ], *  [  x1 ], 
        #  [ x3' ],    [ cos(m * theta2) ],    [ x3 ],    [ sin(m * theta2) ],    [ -x4 ],
        #  [ x4' ]]    [ cos(m * theta2) ]]    [ x4 ]]    [ sin(m * theta2) ]]    [  x3 ]]
        #
        # The repetition of the theta can be obtained by reshaping to (..., 1)
        # and braodcasting.
        
        # Compute m * theta, where `m` is the position index.
        m_theta = torch.outer(torch.arange(
            context_length,
            dtype=torch.float32,
            device=device
        ), theta)

        # An efficient way to apply and compute the rotation is using the
        # complex domain. Converting the `m_theta` to polar representation
        # in complex domain will compute re ^ (i * m_theta), where r = 1.
        #
        # Now, viewing the query and key as complex numbers, which will take
        # span of 2-element blocks and view them as complex numbers. Hence,
        # multiplying the e ^ (i * m_theta) will rotate the 2D complex vector
        # blocks.

        return torch.polar(torch.ones_like(m_theta), m_theta)  # complex64
    
    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.LongTensor = None,
        loss_reduction: str = 'mean'
    ) -> torch.Tensor:
        B, T = tokens.shape
        assert T <= self.context_length, 'Tokens exceeding max context length'

        # Truncate RoPE cache if need be.
        rope_cache = self.rope_cache
        if T < self.context_length:
            rope_cache = rope_cache[:T]
        
        x = self.token_embedding(tokens)

        # Propagate decoder blocks
        for blk in self.dec_blocks:
            x = blk(x, rope_cache)

        # projection
        logits = self.proj(self.norm(x))

        if targets is not None:
            return F.cross_entropy(
                logits.flatten(1),
                targets.flatten(0),
                reduction=loss_reduction
            )
        else:
            return logits

