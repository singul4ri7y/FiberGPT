# Flash Attention 3 interface, fallback to standard SDPA if not available

import torch
import torch.nn.functional as F
from typing import Tuple


def _load_fa3():
    '''
    Try loading Flash Attention 3 (for Hopper+ GPUs only). Other variants of
    flash attention such as FA2 for A100 GPUs are available through the SDPA
    interface of PyTorch.
    '''

    if not torch.cuda.is_available():
        return None

    try:
        major, _ = torch.cuda.get_device_capability()

        # FA3 kernels are designed for H100/H200/GH200 GPUs only, for CUDA
        # version 9.x.
        if major != 9:
            return None

        import flash_attn_interface
        return flash_attn_interface
    except:
        pass


_fa3 = _load_fa3()
fa3_available = _fa3 is not None


def _sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    window_size: Tuple[int],
    enable_gqa: bool = False
):
    ''' PyTorch SDPA wrapper with sliding window support. '''

    # q, k and v are (B, nh, T, hs)
    T_q = q.shape[2]
    T_k = k.shape[2]

    # FiberGPT is a decoder only model, so we only care about causal sliding
    # widnow.
    window = window_size[0]

    # Use full context, mostly true during training.
    if (window < 0 or window >= T_q) and T_q == T_k:
        return F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True, enable_gqa=enable_gqa
        )

    # Single token generation, especially when KV cache is used.
    if T_q == 1:
        # Truncate key and value allowing upto `window` tokens from the current
        # token
        if window > 0 and window < T_k:
            k = k[:, :, -window:, :]
            v = v[:, :, -window:, :]

        return F.scaled_dot_product_attention(
            q, k, v,
            is_causal=False, enable_gqa=enable_gqa
        )

    # I genuinely do not know who figured this trick out. Grabbed this code
    # from NanoChat by Karpathy. Cool way to mimic the tril mask for N amount
    # of last queries.
    device = q.device
    # Add dimension at the beginning for broadcasting.
    row_idx = torch.arange(T_k, device=device).unsqueeze(0)
    col_idx = (T_k - T_q) + torch.arange(T_q, device=device).unsqueeze(1)
    # Should create a lower-triangular matrix upto N queries.
    mask = row_idx <= col_idx

    # This trick is even cooler.
    # Imagine, row_idx = [[ 1, 2, 3, 4 ]], shape = (1, 4)
    # and col_idx = [[1], [2], [3], [4]], shape = (4, 1)
    # Let's consider window = 2
    #
    # Now, computing (col_idx - row_idx) will yield:
    # [[ 0, -1, -2, -3],
    #  [ 1,  0, -1, -2],
    #  [ 2,  1,  0, -1],
    #  [ 3,  2,  1,  0]]
    #
    # And ((col_idx - row_idx) <= window) will yield:
    # [[1, 1, 1, 1],
    #  [1, 1, 1, 1],
    #  [1, 1, 1, 1],
    #  [0, 1, 1, 1]]
    #
    # Finally, performing a bitwise AND with the current mask, which already
    # represents a lower-triangular matrix:
    # [[1, 0, 0, 0],
    #  [1, 1, 0, 0],
    #  [1, 1, 1, 0],
    #  [0, 1, 1, 1]]
    #
    # Which is our causal sliding window attention mask.
    if window >= 0 and window < T_k:
        mask = mask & ((col_idx - row_idx) <= window)

    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=mask,
        enable_gqa=enable_gqa
    )


def flash_attn_func(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    causal: bool = False,
    window_size: Tuple[int] = (-1, -1)
):
    ''' Flash attention for training (w/o KV cache). '''

    if fa3_available:
        return _fa3.flash_attn_func(
            q, k, v,
            causal=causal,
            window_size=window_size
        )

    # Torch SDPA fallback
    # (B, T, nh, hs) -> (B, nh, T, hs)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    enable_gqa = q.shape[1] != k.shape[1]
    return _sdpa(
        q, k, v,
        window_size=window_size,
        enable_gqa=enable_gqa
    ).transpose(1, 2)
