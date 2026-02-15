import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torch.optim import AdamW
from dataclasses import dataclass
from typing import Optional, Tuple, List
from utils.tokenizer import n_vocab
from utils.attention import flash_attn_func
from utils.muon import Muon, DistributedMuon
from utils.common import is_dist_requested


@dataclass
class FiberGPTConfig:
    n_vocab: int = n_vocab
    context_length: int = 1024
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024
    freq_base: int = 10000
    n_hidden: int = n_embd
    ffwd_dim_multiplier: int = 4
    norm_eps: float = 1e-9
    weight_tying: bool = True
    use_value_residuals: bool = True
    # Sliding window attention pattern string
    # S = Small, M = Medium (half context), L = Large (full context)
    # Patterns are repeated throughout the layers.
    window_pattern: str = 'LMML'


def rms_norm(x: torch.Tensor, eps: float):
    '''
    RMS Normalization without parameters. Generally, RMSNorm parameter seldom
    gets trained.
    '''

    return F.rms_norm(x, (x.shape[-1],), None, eps)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, rope_cache: torch.Tensor):
    ''' Applies RoPE rotation to query and key. '''

    # cache shape for reshaping later
    q_shape, k_shape = q.shape, k.shape
    q_dtype, k_dtype = q.dtype, k.dtype

    # Get complex domain representation of 2D block vectors
    # (B, T, nh, hs // 2)
    q = torch.view_as_complex(q.float().view(*q.shape[:-1], -1, 2))
    k = torch.view_as_complex(k.float().view(*k.shape[:-1], -1, 2))

    # Make RoPE cache broadcastable.
    rope_cache = rope_cache.unsqueeze(1)

    # Apply rotation in complex plane
    q = torch.view_as_real(q * rope_cache).to(q_dtype)
    k = torch.view_as_real(k * rope_cache).to(k_dtype)

    return q.view(q_shape), k.view(k_shape)


class CausalAttention(nn.Module):
    ''' Causal multi-head attention. '''

    def __init__(self, config: FiberGPTConfig):
        super().__init__()

        # Generally number of features in key, query and values are equal to
        # embedding dimension (or d_model). So, embedding dimension should be
        # perfectly divisible by the number of heads.
        assert config.n_embd % config.n_head == 0
        self.eps = config.norm_eps
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head

        # Fused linear transformation of key, query and value in the column
        # axis.
        self.attn_qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        ve: Optional[torch.Tensor] = None,
        window_size: Tuple[int] = (-1, -1)
    ):
        B, T, C = x.size()

        qkv = self.attn_qkv(x)
        # Extract query, key and value from fused linear transformation
        q, k, v = qkv.chunk(3, dim=-1)  # (B, T, C), where C = n_embd

        # Apply value residual if available.
        if ve is not None:
            v = v + ve

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
        v = v.view(B, T, self.n_head, self.head_size)

        # Apply RoPE before transposition.
        q, k = _apply_rope(q, k, rope_cache)
        q, k = rms_norm(q, self.eps), rms_norm(k, self.eps)  # QK norm

        # Apply attention
        y = flash_attn_func(
            q, k, v,
            causal=True, window_size=window_size
        )  # (B, T, nh, hs)

        # Re-assemble the output.
        y = y.contiguous().view(B, T, C)
        return self.proj(y)


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

        self.eps = config.norm_eps
        self.attn = CausalAttention(config)
        self.ffwd = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        ve: Optional[torch.Tensor] = None,
        window_size: Tuple[int] = (-1, -1)
    ):
        h = x + self.attn(rms_norm(x, self.eps), rope_cache, ve, window_size)
        return h + self.ffwd(rms_norm(h, self.eps))


class FiberGPT(nn.Module):
    ''' The FiberGPT language model. '''

    def __init__(self, config: FiberGPTConfig):
        super().__init__()

        # Cache required configs
        self.context_length = config.context_length
        self.eps = config.norm_eps

        self.token_embedding = nn.Embedding(config.n_vocab, config.n_embd)

        # Compute rope cache
        self.register_buffer('rope_cache', self._compute_rope_cache(
            config.n_embd // config.n_head,
            config.context_length,
            base = config.freq_base
        ), persistent=False)

        # Window sizes for sliding window attention
        self.window_sizes = self._compute_window_sizes(config)

        # Inspired by ResFormer. Value residual learning, but with value
        # embeddings and gate network.
        if config.use_value_residuals:
            self.value_embedding = nn.Embedding(config.n_vocab, config.n_embd)

            # Gate projection with bias seems to perform better
            self.v_gate = nn.Linear(config.n_embd, config.n_embd, bias=False)
        else:
            self.value_embedding = None

        # Decoder blocks
        self.dec_blocks = nn.ModuleList(
            [ TransformerBlock(config) for _ in range(config.n_layer) ]
        )

        # Projection
        self.proj = nn.Linear(config.n_embd, config.n_vocab, bias=False)

        # Weight tying
        if config.weight_tying:
            self.proj.weight = self.token_embedding.weight

        # Initialize parameters
        self._init_params(config)

    @torch.no_grad()
    def _init_params(self, config: FiberGPTConfig):
        # Embedding
        I.normal_(self.token_embedding.weight, mean=0.0, std=config.n_embd ** -0.5)
        if not config.weight_tying:
            I.normal_(self.token_embedding.weight, mean=0.0, std=1.0)
            I.normal_(self.proj.weight, mean=0.0, std=0.001)

        # Blocks
        # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        s = (3 / config.n_embd) ** 0.5

        for block in self.dec_blocks:
            # Attention
            I.uniform_(block.attn.attn_qkv.weight, -s, s)
            I.zeros_(block.attn.proj.weight)

            # MLP
            I.uniform_(block.ffwd.hl.weight, -s, s)
            I.zeros_(block.ffwd.proj.weight)

        if config.use_value_residuals:
            I.uniform_(self.value_embedding.weight, -s, s)
            I.uniform_(self.v_gate.weight, -s, s)

        # Move all embeddings to bfloat16.
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                m.weight.data = m.weight.bfloat16()

    def _compute_rope_cache(
        self,
        head_size: int,
        context_length: int,
        base: int = 10000,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        ## TODO: Keep RoPE cache in bfloat16

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

    def _compute_window_sizes(self, config: FiberGPTConfig):
        '''
        Compute per layer window sizes.

        S = Small, 1/4 of the context length
        M = Medium, half the context length
        L = Large, full context length
        '''

        # Layer count should be divisible by pattern char count
        pattern = config.window_pattern.upper()
        assert config.n_layer % len(pattern) == 0
        assert all(c in 'SML' for c in pattern), f'Invalid window pattern {pattern}'

        windows = {
            'L': (config.context_length, 0),
            'M': (config.context_length // 2, 0),
            'S': (config.context_length // 4, 0)
        }

        window_sizes = []
        for i in range(config.n_layer):
            window_sizes.append(windows[pattern[i % len(pattern)]])

        return window_sizes

    def _device(self):
        return self.token_embedding.weight.device

    def total_params(self):
        ''' Return the total number of parameters. '''

        params = 0
        for p in self.parameters():
            params += p.numel()

        return params

    def setup_optimizers(
        self,
        embedding_lr: float = 0.05, proj_lr: float = 6e-3,
        matrix_lr: float = 0.02,
        adam_betas=(0.8, 0.95),
        muon_weight_decay: float = 0.01,
        muon_momentum: float = 0.95
    ):
        embedding_params = list(self.token_embedding.parameters())
        matrix_params = list(self.dec_blocks.parameters())
        value_params = list(self.value_embedding.parameters())
        gate_params = list(self.v_gate.parameters())

        # If weights are tied
        if self.token_embedding.weight is self.proj.weight:
            adamw_param_groups = [
                dict(params=embedding_params + value_params, lr=embedding_lr)
            ]
        else:
            adamw_param_groups = [
                dict(params=embedding_params + value_params, lr=embedding_lr),
                dict(params=self.proj.parameters(), lr=proj_lr)
            ]

        optim_adamw = AdamW(
            adamw_param_groups,
            betas=adam_betas,
            weight_decay=0.0
        )

        # TODO: Sort matrix parameters in terms of sizes
        MuonFactory = DistributedMuon if is_dist_requested() else Muon
        optim_muon = MuonFactory(
            matrix_params + gate_params,
            lr=matrix_lr,
            weight_decay=muon_weight_decay,
            momentum=muon_momentum
        )

        # Set initial learning rate.
        for group in optim_adamw.param_groups:
            group['initial_lr'] = group['lr']
        for group in optim_muon.param_groups:
            group['initial_lr'] = group['lr']

        return optim_adamw, optim_muon

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
        ve = None
        if self.value_embedding is not None:
            ve = self.value_embedding(tokens)

            # In Learnable-ResFormer, the lambdas get trained, but all value
            # embedding feature (or vector dimension) is scaled with the same
            # lambda. But applying residuals using gate network (with sigmoid)
            # scales each features individually.
            #
            # Here, 1 -> Don't scale the feature, >1 -> Scale up feature, make it
            # dominant, <1 -> Scale down feature, make it less dominant.
            gate = 2 * F.sigmoid(self.v_gate(x))  # range (0, 2)
            ve = gate * ve

        for i, block in enumerate(self.dec_blocks):
            x = block(x, rope_cache, ve, self.window_sizes[i])

        # projection
        logits = self.proj(rms_norm(x, self.eps))

        if targets is not None:
            return F.cross_entropy(
                logits.view(B * T, -1),
                targets.view(B * T).long(),
                reduction=loss_reduction
            )
        else:
            return logits

    @torch.inference_mode()
    def generate(
        self,
        tokens: List,
        max_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        seed: int = 42
    ):
        ''' Autoregressive streaming interface. '''

        assert isinstance(tokens, list), 'Given tokens must be a list!'

        device = self._device()

        rng = None
        if temperature > 0.0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        # Token indicies (with batch dimension)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)

        for _ in range(max_tokens):
            # Cap tokens upto context length
            idx = idx[:, -self.context_length:]
            logits = self.forward(idx)  # (B, T, n_vocab)

            # We are interested in the very last word
            logits = logits[:, -1, :]

            # Apply Top-K
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                # Zero out the probabilties not top-N during softmax.
                logits[logits < values[:, [-1]]] = float('-inf')

            # Apply temperature
            if temperature > 0:
                logits /= temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # append
            idx = torch.concat([idx, next_token], dim=1)

            yield next_token.item()
