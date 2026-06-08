"""
FlashAttention backends for the current CuPy-based GQA implementation.

Supported now:
  - cupy_flash: exact, memory-efficient forward + backward using tiled CuPy matmul.
  - triton_flash_fwd: Triton forward kernel + CuPy tiled backward fallback.

Assumptions:
  - dropout_rate == 0
  - causal self-attention
  - GQA layout:
      Q: [B, KVH, R, T, D] contiguous
      K: [B, KVH, T, D] contiguous
      V: [B, KVH, T, D] contiguous
  - pad_mask is your current additive mask, usually [B,1,1,T], values 0 or -inf.

This file is intentionally independent from ZeRO3.
"""

import math
import cupy as cp


def _as_f32(x):
    return x.astype(cp.float32, copy=False)


def _pad_bias_block(pad_mask, ks, ke, B, dtype=cp.float32):
    """Return additive pad bias with shape [B, 1, 1, 1, BN]."""
    if pad_mask is None:
        return 0.0

    # Current model.make_mask returns [B,1,1,T].
    if pad_mask.ndim == 4:
        return pad_mask[:, None, :, :, ks:ke].astype(dtype)
    if pad_mask.ndim == 5:
        return pad_mask[:, :, :, :, ks:ke].astype(dtype)
    if pad_mask.ndim == 2:
        return pad_mask[:, None, None, None, ks:ke].astype(dtype)

    raise ValueError(f"Unsupported pad_mask shape: {pad_mask.shape}")


def _causal_bias(qs, qe, ks, ke, dtype=cp.float32):
    q_idx = cp.arange(qs, qe, dtype=cp.int32)[:, None]
    k_idx = cp.arange(ks, ke, dtype=cp.int32)[None, :]
    mask = k_idx > q_idx
    return cp.where(mask, -cp.inf, 0.0).astype(dtype)  # [BM, BN]


def cupy_flash_gqa_forward(Q, K, V, pad_mask, scale, block_m=64, block_n=64):
    """
    Exact tiled causal GQA attention forward.

    Args:
        Q: [B, KVH, R, T, D]
        K: [B, KVH, T, D]
        V: [B, KVH, T, D]
        pad_mask: additive mask, e.g. [B,1,1,T]
        scale: scalar, 1/sqrt(D)

    Returns:
        O:   [B, KVH, R, T, D], same dtype as Q
        LSE: [B, KVH, R, T], float32, logsumexp per query row
    """
    B, KVH, R, T, D = Q.shape
    out_dtype = Q.dtype
    scale_f = cp.float32(float(scale))

    O = cp.empty_like(Q)
    LSE = cp.empty((B, KVH, R, T), dtype=cp.float32)

    for qs in range(0, T, block_m):
        qe = min(qs + block_m, T)
        BM = qe - qs

        q = _as_f32(Q[:, :, :, qs:qe, :])  # [B,KVH,R,BM,D]
        m = cp.full((B, KVH, R, BM), -cp.inf, dtype=cp.float32)
        l = cp.zeros((B, KVH, R, BM), dtype=cp.float32)
        acc = cp.zeros((B, KVH, R, BM, D), dtype=cp.float32)

        # causal: keys beyond qe cannot be used by this query block, so stop at qe
        for ks in range(0, qe, block_n):
            ke = min(ks + block_n, T)

            k = _as_f32(K[:, :, ks:ke, :])[:, :, None, :, :]  # [B,KVH,1,BN,D]
            v = _as_f32(V[:, :, ks:ke, :])[:, :, None, :, :]  # [B,KVH,1,BN,D]

            score = cp.matmul(q * scale_f, k.transpose(0, 1, 2, 4, 3))  # [B,KVH,R,BM,BN]
            score += _causal_bias(qs, qe, ks, ke)[None, None, None, :, :]
            score += _pad_bias_block(pad_mask, ks, ke, B)

            m_new = cp.maximum(m, cp.max(score, axis=-1))
            alpha = cp.exp(m - m_new)
            p = cp.exp(score - m_new[..., None])

            acc = acc * alpha[..., None] + cp.matmul(p, v)
            l = l * alpha + cp.sum(p, axis=-1)
            m = m_new

            del score, p, k, v

        O[:, :, :, qs:qe, :] = (acc / (l[..., None] + cp.float32(1e-9))).astype(out_dtype)
        LSE[:, :, :, qs:qe] = m + cp.log(l + cp.float32(1e-9))

    return O, LSE


def cupy_flash_gqa_backward(Q, K, V, dO, O, LSE, pad_mask, scale,
                            block_m=32, block_n=64):
    """
    Exact tiled causal GQA attention backward.

    Args:
        Q:   [B,KVH,R,T,D]
        K/V: [B,KVH,T,D]
        dO:  [B,KVH,R,T,D]
        O:   [B,KVH,R,T,D]
        LSE: [B,KVH,R,T]

    Returns:
        dQ: [B,KVH,R,T,D], same dtype as Q
        dK: [B,KVH,T,D], same dtype as K
        dV: [B,KVH,T,D], same dtype as V
    """
    B, KVH, R, T, D = Q.shape
    out_dtype = Q.dtype
    scale_f = cp.float32(float(scale))

    dQ = cp.zeros(Q.shape, dtype=cp.float32)
    dK = cp.zeros(K.shape, dtype=cp.float32)
    dV = cp.zeros(V.shape, dtype=cp.float32)

    # D_i = dot(dO_i, O_i), shape [B,KVH,R,T]
    delta = cp.sum(_as_f32(dO) * _as_f32(O), axis=-1)

    for qs in range(0, T, block_m):
        qe = min(qs + block_m, T)

        q = _as_f32(Q[:, :, :, qs:qe, :])
        do = _as_f32(dO[:, :, :, qs:qe, :])
        lse = LSE[:, :, :, qs:qe]
        delta_q = delta[:, :, :, qs:qe]

        dQ_acc = cp.zeros(q.shape, dtype=cp.float32)

        # causal: keys beyond qe cannot contribute to this query block
        for ks in range(0, qe, block_n):
            ke = min(ks + block_n, T)

            k = _as_f32(K[:, :, ks:ke, :])[:, :, None, :, :]  # [B,KVH,1,BN,D]
            v = _as_f32(V[:, :, ks:ke, :])[:, :, None, :, :]  # [B,KVH,1,BN,D]

            score = cp.matmul(q * scale_f, k.transpose(0, 1, 2, 4, 3))
            score += _causal_bias(qs, qe, ks, ke)[None, None, None, :, :]
            score += _pad_bias_block(pad_mask, ks, ke, B)

            P = cp.exp(score - lse[..., None])  # [B,KVH,R,BM,BN]

            # dV shared over R, so sum over n_rep axis later
            dV_part = cp.matmul(P.transpose(0, 1, 2, 4, 3), do)  # [B,KVH,R,BN,D]
            dV[:, :, ks:ke, :] += cp.sum(dV_part, axis=2)

            dP = cp.matmul(do, v.transpose(0, 1, 2, 4, 3))
            dS = P * (dP - delta_q[..., None])

            dQ_acc += cp.matmul(dS, k) * scale_f

            dK_part = cp.matmul(dS.transpose(0, 1, 2, 4, 3), q) * scale_f
            dK[:, :, ks:ke, :] += cp.sum(dK_part, axis=2)

            del score, P, dP, dS, dV_part, dK_part, k, v

        dQ[:, :, :, qs:qe, :] = dQ_acc

    return dQ.astype(out_dtype), dK.astype(K.dtype), dV.astype(V.dtype)


# ======================================================================================
# Optional Triton forward backend.
# This is forward only. For training, combine it with cupy_flash_gqa_backward above.
# ======================================================================================

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def _triton_flash_gqa_fwd_kernel(Q, K, V, O, LSE, VALID,
                                     B: tl.constexpr, KVH: tl.constexpr, R: tl.constexpr,
                                     T: tl.constexpr, D: tl.constexpr,
                                     SCALE: tl.constexpr,
                                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                     BLOCK_D: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_brh = tl.program_id(1)

        r = pid_brh % R
        tmp = pid_brh // R
        h = tmp % KVH
        b = tmp // KVH

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        q_base = (((b * KVH + h) * R + r) * T + offs_m[:, None]) * D + offs_d[None, :]
        q = tl.load(Q + q_base, mask=(offs_m[:, None] < T) & (offs_d[None, :] < D), other=0.0)

        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.full((BLOCK_M,), 0.0, tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        valid_len = tl.load(VALID + b)

        for start_n in range(0, T, BLOCK_N):
            k_idx = start_n + offs_n
            k_base = ((b * KVH + h) * T + k_idx[:, None]) * D + offs_d[None, :]
            k = tl.load(K + k_base, mask=(k_idx[:, None] < T) & (offs_d[None, :] < D), other=0.0)
            v = tl.load(V + k_base, mask=(k_idx[:, None] < T) & (offs_d[None, :] < D), other=0.0)

            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * SCALE

            causal = k_idx[None, :] <= offs_m[:, None]
            valid = k_idx[None, :] < valid_len
            q_valid = offs_m[:, None] < T
            qk = tl.where(causal & valid & q_valid, qk, -float("inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            alpha = tl.exp(m_i - m_ij)

            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij

        out = acc / l_i[:, None]
        o_base = (((b * KVH + h) * R + r) * T + offs_m[:, None]) * D + offs_d[None, :]
        tl.store(O + o_base, out, mask=(offs_m[:, None] < T) & (offs_d[None, :] < D))

        lse_base = (((b * KVH + h) * R + r) * T + offs_m)
        tl.store(LSE + lse_base, m_i + tl.log(l_i), mask=offs_m < T)


def triton_flash_gqa_forward(Q, K, V, valid_lens, scale,
                             block_m=64, block_n=64, block_d=None):
    """
    Triton forward-only FlashAttention for contiguous CuPy arrays.

    If your Triton version cannot accept CuPy arrays directly, use the CuPy backend
    or wrap arrays through torch zero-copy. Keep this function behind a runtime flag.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("triton is not installed")

    B, KVH, R, T, D = Q.shape
    if block_d is None:
        # head_dim=192 in your current model, so use 256 and mask D.
        block_d = 1 << int(math.ceil(math.log2(D)))

    O = cp.empty_like(Q)
    LSE = cp.empty((B, KVH, R, T), dtype=cp.float32)

    grid = (triton.cdiv(T, block_m), B * KVH * R)
    _triton_flash_gqa_fwd_kernel[grid](
        Q, K, V, O, LSE, valid_lens,
        B, KVH, R, T, D,
        float(scale),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=8,
        num_stages=3,
    )
    return O, LSE


def triton_is_available():
    return _TRITON_AVAILABLE