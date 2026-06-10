import cupy as cp
import cupyx.scipy.special as cpx
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.functions import sigmoid,softmax,place_enco,total_clip_grads,xavier_initialization,init_weight

try:
    from model.flash_attention import (
        cupy_flash_gqa_forward,
        cupy_flash_gqa_backward,
        triton_flash_gqa_forward,
        triton_is_available,
    )
except Exception:
    # flash_attention_backends.py が未配置、またはTriton/CuPy環境差でimportに失敗した場合でも、
    # 既存CuPy attentionへfallbackできるようにする。
    cupy_flash_gqa_forward = None
    cupy_flash_gqa_backward = None
    triton_flash_gqa_forward = None
    triton_is_available = None
import time
import wandb
import pickle

#RoPE部分
rope_inplace_kernel = cp.ElementwiseKernel(
    'T cos, T sin',      # 入力を T (float16 or float32) にする
    'T x1, T x2',        # 入出力
    '''
    float v1 = (float)x1;
    float v2 = (float)x2;
    float c = (float)cos;
    float s = (float)sin;
    
    float rot1 = v1 * c - v2 * s;
    float rot2 = v2 * c + v1 * s;

    x1 = (T)rot1;
    x2 = (T)rot2;
    ''',
    'rope_inplace_kernel'
)


def apply_rotary_pos_emb_inplace(x, cos, sin):
    """
    メモリを節約するためにxを直接書き換えます(In-place)。
    x: [batch, head, time, dim]
    cos, sin: [time, dim//2]
    return: x (書き換え後の参照)
    """
    # 次元分割（Viewを作成するだけでメモリコピーは発生しません）
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    
    # カーネル実行 (x1, x2の中身が直接書き換わります)
    # cos, sinは自動的にBroadcastされます
    rope_inplace_kernel(cos, sin, x1, x2)
    return x

# precompute_freqs_cis は計算回数が少ないため、そのままでOKです。
# ただし、float16時の精度落ちを防ぐため計算はfloat32で行い、最後にキャストするのが安全です。
def precompute_freqs_cis(dim, end, theta=10000.0, dtype=cp.float32):
    freqs = 1.0 / (theta ** (cp.arange(0, dim, 2)[: (dim // 2)].astype(cp.float32) / dim))
    t = cp.arange(end).astype(cp.float32)
    emb = cp.outer(t, freqs)
    # cos, sinの生成
    return cp.cos(emb).astype(dtype), cp.sin(emb).astype(dtype)

#===============================================================================================================

#activationを減らしている状態
class GroupedQueryAttention:
    def __init__(self, hidden_size, time_size, q_head_num, kv_head_num, layer_num, dropout_rate, half_float):
        assert hidden_size % q_head_num == 0, "hidden_size must be divisible by q_head_num"
        assert q_head_num % kv_head_num == 0, "q_head_num must be divisible by kv_head_num"

        self.grads = []
        self.hidden_size = hidden_size
        self.q_head_num = q_head_num
        self.kv_head_num = kv_head_num
        self.time_size = time_size
        self.half_float = half_float
        self.dropout_rate = dropout_rate

        self.head_dim = hidden_size // q_head_num
        self.n_rep = q_head_num // kv_head_num
        self.kv_dim = kv_head_num * self.head_dim
        self.kqv_out_dim = self.kv_dim + hidden_size + self.kv_dim  # K + Q + V

        if dropout_rate != 0:
            self.dropout1 = TimeDropout(dropout_rate, self.half_float)
            self.dropout2 = TimeDropout(dropout_rate, self.half_float)

        dtype_rope = cp.float16 if half_float else cp.float32
        self.scale_value = float(1.0 / (self.head_dim ** 0.5))
        self.scale = cp.asarray(self.scale_value, dtype=dtype_rope)

        # projectionを1つにまとめる: [hidden, kv_dim + hidden + kv_dim]
        # 出力順序は [K, Q, V]
        W_KQV = init_weight([hidden_size, self.kqv_out_dim], (1 / cp.sqrt(hidden_size)))
        W_O = init_weight([hidden_size, hidden_size], (1 / cp.sqrt(hidden_size * 2 * layer_num)))
        self.params = [W_KQV, W_O]

        self.cache = None
        # Trueにすると backward で out_value=attention@V を再計算しない。
        # FlashAttention時も同じ。速度優先ならTrue、VRAM優先ならFalse。
        self.cache_out_value = False
        # Trueにすると dropout_rate=0 のとき既存CuPy attentionでattnを保持する。
        # FlashAttention時はattnを保持しないので、このフラグがTrueなら安全のためfallbackする。
        self.cache_attn = False

        # FlashAttention backend:
        #   "none"                 : 既存の CuPy matmul + softmax
        #   "cupy"                 : CuPy tiled FlashAttention forward + backward
        #   "triton"               : Triton forward + CuPy tiled backward fallback
        #   "triton_fwd_cupy_bwd"  : 同上の別名
        self.flash_backend = "none"
        self.flash_block_m = 64
        self.flash_block_n = 64
        self.flash_bwd_block_m = 32
        self.flash_bwd_block_n = 64

    def _make_pad_bias(self, pad_mask, batch_size, time_size, dtype):
        """
        score shape: [B, kv_h, n_rep, T, T]

        対応するpad_mask:
        - bool [B, T]              : Trueをmask
        - bool [B, 1, 1, T]        : Trueをmask
        - bool [B, 1, 1, 1, T]     : Trueをmask
        - additive [B, T]          : 0 or -inf等
        - additive [B, 1, 1, T]    : 0 or -inf等
        - additive [B, 1, 1, 1, T] : 0 or -inf等
        """
        if pad_mask is None:
            return 0.0

        is_bool = (pad_mask.dtype == cp.bool_)

        if is_bool:
            if pad_mask.ndim == 1:
                seq_ids = cp.arange(time_size)[None, :]
                m = seq_ids >= pad_mask[:, None]
                m = m[:, None, None, None, :]
            elif pad_mask.ndim == 2:
                m = pad_mask[:, None, None, None, :]
            elif pad_mask.ndim == 4:
                m = pad_mask[:, None, :, :, :]
            elif pad_mask.ndim == 5:
                m = pad_mask
            else:
                raise ValueError(f"Unsupported bool pad_mask shape: {pad_mask.shape}")
            return cp.where(m, -float("inf"), 0.0).astype(dtype)

        # valid_lens [B] が直接渡された場合。Triton用にも使いやすい。
        if pad_mask.ndim == 1:
            seq_ids = cp.arange(time_size)[None, :]
            m = seq_ids >= pad_mask[:, None]
            return cp.where(m[:, None, None, None, :], -float("inf"), 0.0).astype(dtype)

        if pad_mask.ndim == 2:
            return pad_mask[:, None, None, None, :].astype(dtype)
        if pad_mask.ndim == 4:
            return pad_mask[:, None, :, :, :].astype(dtype)
        if pad_mask.ndim == 5:
            return pad_mask.astype(dtype)

        raise ValueError(f"Unsupported additive pad_mask shape: {pad_mask.shape}")

    def _valid_lens_from_pad_mask(self, pad_mask, batch_size, time_size):
        """Triton FlashAttention用に additive/boolean mask から valid length [B] を作る。"""
        if pad_mask is None:
            return cp.full((batch_size,), time_size, dtype=cp.int32)

        # valid_lens [B] が直接来た場合
        if pad_mask.ndim == 1:
            return pad_mask.astype(cp.int32, copy=False)

        if pad_mask.dtype == cp.bool_:
            if pad_mask.ndim == 2:
                m = pad_mask
            elif pad_mask.ndim == 4:
                m = pad_mask[:, 0, 0, :]
            elif pad_mask.ndim == 5:
                m = pad_mask[:, 0, 0, 0, :]
            else:
                raise ValueError(f"Unsupported bool pad_mask shape: {pad_mask.shape}")
            return cp.sum(~m, axis=-1).astype(cp.int32)

        if pad_mask.ndim == 2:
            m = pad_mask
        elif pad_mask.ndim == 4:
            m = pad_mask[:, 0, 0, :]
        elif pad_mask.ndim == 5:
            m = pad_mask[:, 0, 0, 0, :]
        else:
            raise ValueError(f"Unsupported additive pad_mask shape: {pad_mask.shape}")

        return cp.sum(cp.isfinite(m), axis=-1).astype(cp.int32)

    def _project_kqv(self, x):
        """
        x: [B, T, hidden]
        return:
            K: [B, kv_h, T, D]
            Q: [B, q_h,  T, D]
            V: [B, kv_h, T, D]
            x_flat: [B*T, hidden]
        """
        W_KQV, _ = self.params
        batch_size, time_size, _ = x.shape

        x_flat = x.reshape(batch_size * time_size, self.hidden_size)
        KQV = cp.matmul(x_flat, W_KQV)

        k_end = self.kv_dim
        q_end = self.kv_dim + self.hidden_size

        K = KQV[:, :k_end]
        Q = KQV[:, k_end:q_end]
        V = KQV[:, q_end:]

        K = K.reshape(batch_size, time_size, self.kv_head_num, self.head_dim).transpose(0, 2, 1, 3)
        Q = Q.reshape(batch_size, time_size, self.q_head_num, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, time_size, self.kv_head_num, self.head_dim).transpose(0, 2, 1, 3)

        return K, Q, V, x_flat

    def _flash_enabled(self, use_dropout, return_attn):
        backend = str(getattr(self, "flash_backend", "none")).lower()
        if backend in ("none", "false", "0", "off"):
            return False
        # dropout対応Flashはまだ入れていない。dropoutありなら既存実装へ戻す。
        if use_dropout and self.dropout_rate != 0:
            return False
        # attnをcacheしたい場合は、FlashAttentionではなく既存実装にする。
        if return_attn:
            return False
        if cupy_flash_gqa_forward is None or cupy_flash_gqa_backward is None:
            return False
        return True

    def _run_flash_forward(self, Q, K, V, pad_mask, batch_size, time_size):
        backend = str(getattr(self, "flash_backend", "none")).lower()
        block_m = int(getattr(self, "flash_block_m", 64))
        block_n = int(getattr(self, "flash_block_n", 64))

        Q = cp.ascontiguousarray(Q)
        K = cp.ascontiguousarray(K)
        V = cp.ascontiguousarray(V)

        if backend in ("triton", "triton_fwd_cupy_bwd"):
            if triton_flash_gqa_forward is not None and triton_is_available is not None and triton_is_available():
                try:
                    valid_lens = self._valid_lens_from_pad_mask(pad_mask, batch_size, time_size)
                    return triton_flash_gqa_forward(
                        Q, K, V, valid_lens, self.scale_value,
                        block_m=block_m,
                        block_n=block_n,
                    )
                except Exception as e:
                    # Triton/CuPy直渡しが環境で通らない場合、本学習を止めないためCuPy Flashへfallback。
                    # 毎回printするとログが荒れるので、最初の1回だけ表示。
                    if not getattr(self, "_triton_fallback_warned", False):
                        print(f"[GQA] Triton FlashAttention failed; fallback to CuPy flash. reason={type(e).__name__}: {e}")
                        self._triton_fallback_warned = True

        return cupy_flash_gqa_forward(
            Q, K, V, pad_mask, self.scale_value,
            block_m=block_m,
            block_n=block_n,
        )

    def _forward_core(self, x, pad_mask, gqa_pack, use_dropout, return_attn=False):
        cos, sin, ba_sin, attn_bias = gqa_pack
        batch_size, time_size, _ = x.shape
        dtype = cp.float16 if self.half_float else cp.float32

        K, Q, V, _ = self._project_kqv(x)

        # RoPEは4次元 [B,H,T,D] の状態で適用する
        apply_rotary_pos_emb_inplace(K, cos, sin)
        apply_rotary_pos_emb_inplace(Q, cos, sin)

        # GQA shapeへ変換
        # Q: [B, q_h, T, D] -> [B, kv_h, n_rep, T, D]
        Q = Q.reshape(batch_size, self.kv_head_num, self.n_rep, time_size, self.head_dim)

        if self._flash_enabled(use_dropout, return_attn):
            # Flash backendは causal mask を内部で処理するので attn_bias は使わない。
            # pad_maskは additive [B,1,1,T] / valid_lens [B] のどちらも対応。
            out_group, lse = self._run_flash_forward(Q, K, V, pad_mask, batch_size, time_size)
            out_value = out_group.transpose(0, 3, 1, 2, 4).reshape(
                batch_size, time_size, self.hidden_size
            )
            flash_ctx = {
                "backend": str(getattr(self, "flash_backend", "none")).lower(),
                "lse": lse,
            }
            return out_value, None, flash_ctx

        K = K[:, :, None, :, :]  # [B, kv_h, 1, T, D]
        V = V[:, :, None, :, :]  # [B, kv_h, 1, T, D]

        # score: [B, kv_h, n_rep, T, T]
        score = cp.matmul(Q * self.scale, K.transpose(0, 1, 2, 4, 3))
        score += attn_bias[:, :, :, :time_size, :time_size]
        score += self._make_pad_bias(pad_mask, batch_size, time_size, dtype)

        attn = cpx.softmax(score, axis=-1)

        if use_dropout and self.dropout_rate != 0:
            attn_used = self.dropout1.forward(attn)
        else:
            attn_used = attn

        out_group = cp.matmul(attn_used, V)
        out_value = out_group.transpose(0, 3, 1, 2, 4).reshape(
            batch_size, time_size, self.hidden_size
        )

        if (use_dropout and self.dropout_rate != 0) or return_attn:
            return out_value, attn_used, None
        else:
            return out_value, None, None

    def forward(self, x, pad_mask, gqa_pack, is_train=True):
        if is_train:
            self.grads = []

        _, W_O = self.params

        out_value, attn_used, flash_ctx = self._forward_core(
            x, pad_mask, gqa_pack, use_dropout=is_train,
            return_attn=(is_train and self.cache_attn and self.dropout_rate == 0),
        )

        final_out = cp.matmul(out_value, W_O)

        if is_train:
            cached_out_value = out_value if self.cache_out_value else None
            if self.dropout_rate != 0:
                final_out = self.dropout2.forward(final_out)
                self.cache = (x, pad_mask, attn_used, cached_out_value, flash_ctx)
            else:
                self.cache = (
                    x,
                    pad_mask,
                    attn_used if self.cache_attn else None,
                    cached_out_value,
                    flash_ctx,
                )

        return final_out

    def backward(self, dout, gqa_pack):
        cos, sin, ba_sin, attn_bias = gqa_pack
        if self.cache is None:
            raise RuntimeError("Backward called but cache is empty. Run forward(is_train=True) first.")

        # 古いcacheとの互換も残す
        if len(self.cache) == 3:
            x, pad_mask, attn_used_cache = self.cache
            out_value_cache = None
            flash_ctx = None
        elif len(self.cache) == 4:
            x, pad_mask, attn_used_cache, out_value_cache = self.cache
            flash_ctx = None
        else:
            x, pad_mask, attn_used_cache, out_value_cache, flash_ctx = self.cache

        batch_size, time_size, _ = x.shape

        K, Q4, V4, x_flat = self._project_kqv(x)
        apply_rotary_pos_emb_inplace(K, cos, sin)
        apply_rotary_pos_emb_inplace(Q4, cos, sin)

        Q = Q4.reshape(batch_size, self.kv_head_num, self.n_rep, time_size, self.head_dim)

        W_KQV, W_O = self.params

        if self.dropout_rate != 0:
            dout = self.dropout2.backward(dout)

        # ==================================================================================
        # FlashAttention backward path
        # ==================================================================================
        if flash_ctx is not None:
            Q_flash = cp.ascontiguousarray(Q)
            K_flash = cp.ascontiguousarray(K)
            V_flash = cp.ascontiguousarray(V4)

            if out_value_cache is None:
                # cache_out_value=False の場合は、W_O backward と attention backward 用に
                # out_group / lse をもう一度Flash forwardで再計算する。
                out_group, lse = self._run_flash_forward(
                    Q_flash, K_flash, V_flash, pad_mask, batch_size, time_size
                )
                out_value = out_group.transpose(0, 3, 1, 2, 4).reshape(
                    batch_size, time_size, self.hidden_size
                )
            else:
                out_value = out_value_cache
                out_group = out_value.reshape(
                    batch_size, time_size, self.kv_head_num, self.n_rep, self.head_dim
                ).transpose(0, 2, 3, 1, 4)
                lse = flash_ctx.get("lse")
                if lse is None:
                    # 念のため。通常ここには来ない。
                    _, lse = self._run_flash_forward(
                        Q_flash, K_flash, V_flash, pad_mask, batch_size, time_size
                    )

            # W_O backward
            dout_flat = dout.reshape(batch_size * time_size, self.hidden_size)
            out_value_flat = out_value.reshape(batch_size * time_size, self.hidden_size)
            dW_O = cp.matmul(out_value_flat.T, dout_flat)

            # out_value backward
            d_out_value = cp.matmul(dout, W_O.T)
            d_out_group = d_out_value.reshape(
                batch_size, time_size, self.kv_head_num, self.n_rep, self.head_dim
            ).transpose(0, 2, 3, 1, 4)  # [B, kv_h, n_rep, T, D]

            dQ_group, dK, dV = cupy_flash_gqa_backward(
                Q_flash,
                K_flash,
                V_flash,
                cp.ascontiguousarray(d_out_group),
                cp.ascontiguousarray(out_group),
                lse,
                pad_mask,
                self.scale_value,
                block_m=int(getattr(self, "flash_bwd_block_m", 32)),
                block_n=int(getattr(self, "flash_bwd_block_n", 64)),
            )

            # RoPE backward
            dQ = dQ_group.reshape(batch_size, self.q_head_num, time_size, self.head_dim)
            apply_rotary_pos_emb_inplace(dQ, cos, ba_sin)
            apply_rotary_pos_emb_inplace(dK, cos, ba_sin)

            dK_flat = dK.transpose(0, 2, 1, 3).reshape(batch_size * time_size, self.kv_dim)
            dQ_flat = dQ.transpose(0, 2, 1, 3).reshape(batch_size * time_size, self.hidden_size)
            dV_flat = dV.transpose(0, 2, 1, 3).reshape(batch_size * time_size, self.kv_dim)

            dKQV_flat = cp.empty((batch_size * time_size, self.kqv_out_dim), dtype=dQ_flat.dtype)
            k_end = self.kv_dim
            q_end = self.kv_dim + self.hidden_size
            dKQV_flat[:, :k_end] = dK_flat
            dKQV_flat[:, k_end:q_end] = dQ_flat
            dKQV_flat[:, q_end:] = dV_flat

            dW_KQV = cp.matmul(x_flat.T, dKQV_flat)
            dout_next = cp.matmul(dKQV_flat, W_KQV.T).reshape(batch_size, time_size, self.hidden_size)

            if self.half_float:
                dW_KQV = dW_KQV.astype(cp.float16)
                dW_O = dW_O.astype(cp.float16)

            self.grads = [dW_KQV, dW_O]
            self.cache = None
            return dout_next

        # ==================================================================================
        # Original CuPy attention backward path
        # ==================================================================================
        K_gqa = K[:, :, None, :, :]   # [B, kv_h, 1, T, D]
        V_gqa = V4[:, :, None, :, :]  # [B, kv_h, 1, T, D]

        # dropout_rate=0 かつ cache_attn=True の場合は、forwardで保存したattnを使い、
        # backward側の score matmul + softmax 再計算を省く。
        if self.dropout_rate == 0 and self.cache_attn and attn_used_cache is not None:
            attn = attn_used_cache
            attn_for_dv = attn
        else:
            dtype = cp.float16 if self.half_float else cp.float32
            score = cp.matmul(Q * self.scale, K_gqa.transpose(0, 1, 2, 4, 3))
            score += attn_bias[:, :, :, :time_size, :time_size]
            score += self._make_pad_bias(pad_mask, batch_size, time_size, dtype)
            attn = cpx.softmax(score, axis=-1)
            del score

            # out_group = dropout(attn) @ V
            # dVにはdropout後のattentionが必要
            if self.dropout_rate != 0:
                attn_for_dv = attn_used_cache
            else:
                attn_for_dv = attn

        if out_value_cache is None:
            out_group = cp.matmul(attn_for_dv, V_gqa)
            out_value = out_group.transpose(0, 3, 1, 2, 4).reshape(
                batch_size, time_size, self.hidden_size
            )
        else:
            out_value = out_value_cache

        # W_O backward
        dout_flat = dout.reshape(batch_size * time_size, self.hidden_size)
        out_value_flat = out_value.reshape(batch_size * time_size, self.hidden_size)
        del out_value
        dW_O = cp.matmul(out_value_flat.T, dout_flat)

        # out_value backward
        d_out_value = cp.matmul(dout, W_O.T)
        d_out_group = d_out_value.reshape(
            batch_size, time_size, self.kv_head_num, self.n_rep, self.head_dim
        ).transpose(0, 2, 3, 1, 4)  # [B, kv_h, n_rep, T, D]

        # dV: Vはn_rep方向で共有されるためsumする
        dV = cp.matmul(attn_for_dv.transpose(0, 1, 2, 4, 3), d_out_group)
        dV = dV.sum(axis=2)  # [B, kv_h, T, D]

        # attention側の勾配
        d_attn_used = cp.matmul(d_out_group, V_gqa.transpose(0, 1, 2, 4, 3))

        if self.dropout_rate != 0:
            d_attn = self.dropout1.backward(d_attn_used)
        else:
            d_attn = d_attn_used

        # softmax backward
        # d_score = y * (dy - sum(y*dy))
        d_score = attn * (d_attn - cp.sum(attn * d_attn, axis=-1, keepdims=True))

        del attn

        # score = (Q * scale) @ K.T
        dQ = cp.matmul(d_score, K_gqa) * self.scale  # [B, kv_h, n_rep, T, D]
        dK = cp.matmul(d_score.transpose(0, 1, 2, 4, 3), Q) * self.scale
        dK = dK.sum(axis=2)  # [B, kv_h, T, D]

        # RoPE backward
        dQ = dQ.reshape(batch_size, self.q_head_num, time_size, self.head_dim)
        apply_rotary_pos_emb_inplace(dQ, cos, ba_sin)
        apply_rotary_pos_emb_inplace(dK, cos, ba_sin)

        # [B,H,T,D] -> [B,T,H,D] -> flatten
        dK_flat = dK.transpose(0, 2, 1, 3).reshape(batch_size * time_size, self.kv_dim)
        dQ_flat = dQ.transpose(0, 2, 1, 3).reshape(batch_size * time_size, self.hidden_size)
        dV_flat = dV.transpose(0, 2, 1, 3).reshape(batch_size * time_size, self.kv_dim)

        # dKQVを1つにまとめる: [K, Q, V]
        dKQV_flat = cp.empty((batch_size * time_size, self.kqv_out_dim), dtype=dQ_flat.dtype)
        k_end = self.kv_dim
        q_end = self.kv_dim + self.hidden_size
        dKQV_flat[:, :k_end] = dK_flat
        dKQV_flat[:, k_end:q_end] = dQ_flat
        dKQV_flat[:, q_end:] = dV_flat

        dW_KQV = cp.matmul(x_flat.T, dKQV_flat)
        dout_next = cp.matmul(dKQV_flat, W_KQV.T).reshape(batch_size, time_size, self.hidden_size)

        if self.half_float:
            dW_KQV = dW_KQV.astype(cp.float16)
            dW_O = dW_O.astype(cp.float16)

        self.grads = [dW_KQV, dW_O]
        self.cache = None
        return dout_next

    @staticmethod
    def convert_old_params_to_fast(W_K, W_Q, W_V, W_O):
        """
        旧GQA実装の params=[W_K,W_Q,W_V,W_O] から
        高速版 params=[W_KQV,W_O] へ変換するための補助関数。

        old shapes:
            W_K: [kv_h, hidden, head_dim]
            W_Q: [q_h,  hidden, head_dim]
            W_V: [kv_h, hidden, head_dim]
        new shape:
            W_KQV: [hidden, kv_h*head_dim + q_h*head_dim + kv_h*head_dim]
        """
        hidden_size = W_Q.shape[1]
        W_K_flat = W_K.transpose(1, 0, 2).reshape(hidden_size, -1)
        W_Q_flat = W_Q.transpose(1, 0, 2).reshape(hidden_size, -1)
        W_V_flat = W_V.transpose(1, 0, 2).reshape(hidden_size, -1)
        W_KQV = cp.concatenate([W_K_flat, W_Q_flat, W_V_flat], axis=1)
        return [W_KQV, W_O]


#======================================================================================================
#======================================================================================================

class SoftMax:
    def __init__(self,half_float):
       self.half_float=half_float

    def forward(self, x, is_train=True):
        if self.half_float:
            x=x.astype(cp.float32)
        exp_x = cp.exp(x - cp.max(x, axis=-1, keepdims=True)) 
        x = exp_x / cp.sum(exp_x, axis=-1, keepdims=True) 
        if is_train:
            self.x=x
        if self.half_float:
            x=x.astype(cp.float16)
        return x

    def backward(self, dy):
        if self.half_float:
            dy=dy.astype(cp.float32)
        dx = self.x * (dy - cp.sum(dy * self.x, axis=-1, keepdims=True))
        del self.x
        if self.half_float:
            dx=dx.astype(cp.float16)
        return dx



class TimeDropout:
    def __init__(self, dropout_rate, half_float):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_rate
        self.packed_mask = None # 圧縮されたマスク
        self.mask_shape = None  # マスクの形状（復元用）
        self.train_flg = True
        self.half_float = half_float
        
        # スケールは配列でなくスカラ値として計算時に掛ける方が効率的
        self.scale = 1.0 / (1.0 - dropout_rate)

    def forward(self, xs):
        if self.train_flg:
            # 1. ランダム生成 (float64を避けるため float32 で生成)
            # xsと同じ形状の乱数を生成
            rand = cp.random.uniform(0.0, 1.0, xs.shape, dtype=cp.float32)
            
            # 2. マスク作成 (bool型: 1byte)
            mask = rand > self.dropout_ratio
            
            # 不要になった乱数配列は即座に解放
            del rand
            
            # 3. マスク適用
            # ここで計算結果を作成 (dtypeはxsに依存)
            out = xs * mask * self.scale
            
            # 4. マスクの圧縮保存 (ここが重要)
            # bool(1byte) -> bit(1/8byte) に圧縮して保存
            # 例: [1, 0, 1, 1, 0, 0, 1, 0] -> 1つのuint8整数に変換
            self.packed_mask = cp.packbits(mask)
            self.mask_shape = mask.shape
            
            return out
        else:
            return xs

    def backward(self, dout):
        # 1. マスクの復元
        # 保存しておいた形状情報を使ってビット列を展開
        # countを指定しないとパディングされた余分な要素が含まれる可能性があるため指定する
        total_elems = 1
        for s in self.mask_shape:
            total_elems *= s
            
        mask = cp.unpackbits(self.packed_mask, count=total_elems)
        mask = mask.reshape(self.mask_shape)
        
        # 2. 勾配計算
        # maskはuint8(0/1)ですが、積をとると自動的にdoutの型にキャストされます
        out = dout * mask * self.scale
        
        # 3. メモリ解放
        self.packed_mask = None
        self.mask_shape = None
        
        return out

#===========================================================================================================================================



#SwiGLU

swiglu_fwd_kernel = cp.ElementwiseKernel(
    'T gate, T value',
    'T out',
    '''
    T sig = 1.0 / (1.0 + exp(-gate));
    out = (gate * sig) * value;
    ''',
    'swiglu_fwd_kernel'
)

swiglu_bwd_kernel = cp.ElementwiseKernel(
    'T dout, T gate, T value',
    'T d_gate, T d_value',
    '''
    T sig = 1.0 / (1.0 + exp(-gate));
    T swish = gate * sig;

    T d_swish = sig * (1.0 + gate * (1.0 - sig));
    
    d_gate = dout * value * d_swish;
    d_value = dout * swish;
    ''',
    'swiglu_bwd_kernel'
)

#Activationを減らしている状態
class SwiGLU:
    def __init__(self, hidden_size, layer_num, half_float, dropout_rate):
        self.hidden_size = hidden_size
        self.inner_size = int(hidden_size * 8 / 3)
        self.half_float = half_float

        std_dev_1 = (1 / cp.sqrt(hidden_size)).astype(cp.float32)
        W_gate_val = init_weight([hidden_size, self.inner_size * 2], std_dev_1)

        std_dev_2 = (1 / cp.sqrt(2 * layer_num * self.inner_size)).astype(cp.float32)
        W_out = init_weight([self.inner_size, hidden_size], std_dev_2)

        self.params = [W_gate_val, W_out]

        self.dropout_rate = dropout_rate
        if dropout_rate != 0:
            self.dropout = TimeDropout(dropout_rate, half_float)

        self.x = None
        self.gate_val = None
        self.grads = []
        self.cache_gate = False

    def forward(self, x, is_train=True):
        W_gate_val, W_out = self.params

        batch_size, time_size, _ = x.shape

        x_reshaped = x.reshape(-1, self.hidden_size)  # [batch*time, hidden]
        gate_val = cp.matmul(x_reshaped, W_gate_val)  # [batch*time, inner_size*2]
        gate_val = gate_val.reshape(batch_size, time_size, self.inner_size * 2)

        # forward計算用の一時バッファ。
        # backward用には保存しない。
        act_out = cp.empty((batch_size, time_size, self.inner_size), dtype=gate_val.dtype)
        swiglu_fwd_kernel(
            gate_val[:, :, :self.inner_size],
            gate_val[:, :, self.inner_size:],
            act_out,
        )

        out = cp.matmul(act_out.reshape(-1, self.inner_size), W_out)
        out = out.reshape(batch_size, time_size, self.hidden_size)

        if is_train:
            if self.dropout_rate != 0:
                out = self.dropout.forward(out)

            self.x = x
            # 速度優先時はgate/valueを保存し、backward側の x @ W_gate_val 再計算を省く。
            # メモリが大きく増えるためデフォルトはFalse。
            self.gate_val = gate_val if self.cache_gate else None

        return out

    def backward(self, dout):
        if self.x is None:
            raise RuntimeError("Backward called before forward or cache has already been cleared.")

        W_gate_val, W_out = self.params

        batch_size, time_size, _ = dout.shape

        if self.dropout_rate != 0:
            dout = self.dropout.backward(dout)

        dout_reshaped = dout.reshape(-1, self.hidden_size)

        # ---------------------------------------------------------
        # dW_out 用に act_out を用意する。
        # cache_gate=Trueなら forward の gate/value を再利用し、巨大 matmul を1回省く。
        # Falseなら従来どおり再計算してVRAMを節約する。
        # ---------------------------------------------------------
        if self.gate_val is None:
            gate_val = cp.matmul(self.x.reshape(-1, self.hidden_size), W_gate_val)
            gate_val = gate_val.reshape(batch_size, time_size, self.inner_size * 2)
        else:
            gate_val = self.gate_val
        act_out = cp.empty((batch_size, time_size, self.inner_size), dtype=gate_val.dtype)

        swiglu_fwd_kernel(
            gate_val[:, :, :self.inner_size],
            gate_val[:, :, self.inner_size:],
            act_out,
        )
        act_out_reshaped = act_out.reshape(-1, self.inner_size)

        dW_out = cp.matmul(act_out_reshaped.T, dout_reshaped)

        # W_out側からSwiGLU中間への勾配
        dout_act = cp.matmul(dout_reshaped, W_out.T)
        dout_act = dout_act.reshape(batch_size, time_size, self.inner_size)

        # gate/valueへの勾配
        d_gate_val = cp.empty_like(gate_val)
        swiglu_bwd_kernel(
            dout_act,
            gate_val[:, :, :self.inner_size],
            gate_val[:, :, self.inner_size:],
            d_gate_val[:, :, :self.inner_size],
            d_gate_val[:, :, self.inner_size:],
        )

        d_gate_val_reshaped = d_gate_val.reshape(-1, self.inner_size * 2)

        x_reshaped = self.x.reshape(-1, self.hidden_size)
        dW_gate_val = cp.matmul(x_reshaped.T, d_gate_val_reshaped)

        dout_final = cp.matmul(d_gate_val_reshaped, W_gate_val.T)
        dout_final = dout_final.reshape(batch_size, time_size, self.hidden_size)

        self.grads = [dW_gate_val, dW_out]

        # cache解放。act_out系はそもそもselfに保存していない。
        self.x = None
        self.gate_val = None

        return dout_final
    



    

# ======================================================================================
# Fused RMSNorm RawKernels
# ======================================================================================
_rmsnorm_fwd_f16_kernel = cp.RawKernel(r'''
#include <cuda_fp16.h>
extern "C" __global__
void rmsnorm_fwd_f16_kernel(
    const half* __restrict__ x,
    const half* __restrict__ gamma,
    half* __restrict__ out,
    float* __restrict__ rsqrt_out,
    const int N,
    const int H,
    const float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float smem[];

    if (row >= N) return;

    float sum_sq = 0.0f;
    long long base = (long long)row * (long long)H;
    for (int h = tid; h < H; h += blockDim.x) {
        float xv = __half2float(x[base + h]);
        sum_sq += xv * xv;
    }

    smem[tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    float r = rsqrtf(smem[0] / (float)H + eps);
    if (tid == 0) rsqrt_out[row] = r;

    for (int h = tid; h < H; h += blockDim.x) {
        float xv = __half2float(x[base + h]);
        float gv = __half2float(gamma[h]);
        out[base + h] = __float2half_rn(xv * r * gv);
    }
}
''', 'rmsnorm_fwd_f16_kernel')

_rmsnorm_fwd_f32_kernel = cp.RawKernel(r'''
extern "C" __global__
void rmsnorm_fwd_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ out,
    float* __restrict__ rsqrt_out,
    const int N,
    const int H,
    const float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float smem[];

    if (row >= N) return;

    float sum_sq = 0.0f;
    long long base = (long long)row * (long long)H;
    for (int h = tid; h < H; h += blockDim.x) {
        float xv = x[base + h];
        sum_sq += xv * xv;
    }

    smem[tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    float r = rsqrtf(smem[0] / (float)H + eps);
    if (tid == 0) rsqrt_out[row] = r;

    for (int h = tid; h < H; h += blockDim.x) {
        out[base + h] = x[base + h] * r * gamma[h];
    }
}
''', 'rmsnorm_fwd_f32_kernel')

_rmsnorm_dgamma_f16_kernel = cp.RawKernel(r'''
#include <cuda_fp16.h>
extern "C" __global__
void rmsnorm_dgamma_f16_kernel(
    const half* __restrict__ dout,
    const half* __restrict__ y,
    const half* __restrict__ gamma,
    half* __restrict__ dgamma,
    const int N,
    const int H,
    const float gamma_eps
) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float smem[];

    if (h >= H) return;

    float sum = 0.0f;
    float gv = __half2float(gamma[h]) + gamma_eps;
    for (int row = tid; row < N; row += blockDim.x) {
        long long idx = (long long)row * (long long)H + h;
        float dy = __half2float(dout[idx]);
        float yy = __half2float(y[idx]);
        float xnorm = yy / gv;
        sum += dy * xnorm;
    }

    smem[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    if (tid == 0) dgamma[h] = __float2half_rn(smem[0]);
}
''', 'rmsnorm_dgamma_f16_kernel')

_rmsnorm_dgamma_f32_kernel = cp.RawKernel(r'''
extern "C" __global__
void rmsnorm_dgamma_f32_kernel(
    const float* __restrict__ dout,
    const float* __restrict__ y,
    const float* __restrict__ gamma,
    float* __restrict__ dgamma,
    const int N,
    const int H,
    const float gamma_eps
) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float smem[];

    if (h >= H) return;

    float sum = 0.0f;
    float gv = gamma[h] + gamma_eps;
    for (int row = tid; row < N; row += blockDim.x) {
        long long idx = (long long)row * (long long)H + h;
        float xnorm = y[idx] / gv;
        sum += dout[idx] * xnorm;
    }

    smem[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    if (tid == 0) dgamma[h] = smem[0];
}
''', 'rmsnorm_dgamma_f32_kernel')

_rmsnorm_bwd_dx_f16_kernel = cp.RawKernel(r'''
#include <cuda_fp16.h>
extern "C" __global__
void rmsnorm_bwd_dx_f16_kernel(
    const half* __restrict__ dout,
    const half* __restrict__ y,
    const half* __restrict__ gamma,
    const float* __restrict__ rsqrt_in,
    half* __restrict__ dx,
    const int N,
    const int H,
    const float gamma_eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float smem[];

    if (row >= N) return;

    long long base = (long long)row * (long long)H;
    float dot = 0.0f;
    for (int h = tid; h < H; h += blockDim.x) {
        float dy = __half2float(dout[base + h]);
        float gv = __half2float(gamma[h]);
        float xnorm = __half2float(y[base + h]) / (gv + gamma_eps);
        float dnorm = dy * gv;
        dot += dnorm * xnorm;
    }

    smem[tid] = dot;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    float mean_dot = smem[0] / (float)H;
    float r = rsqrt_in[row];

    for (int h = tid; h < H; h += blockDim.x) {
        float dy = __half2float(dout[base + h]);
        float gv = __half2float(gamma[h]);
        float xnorm = __half2float(y[base + h]) / (gv + gamma_eps);
        float dnorm = dy * gv;
        float v = r * (dnorm - xnorm * mean_dot);
        dx[base + h] = __float2half_rn(v);
    }
}
''', 'rmsnorm_bwd_dx_f16_kernel')

_rmsnorm_bwd_dx_f32_kernel = cp.RawKernel(r'''
extern "C" __global__
void rmsnorm_bwd_dx_f32_kernel(
    const float* __restrict__ dout,
    const float* __restrict__ y,
    const float* __restrict__ gamma,
    const float* __restrict__ rsqrt_in,
    float* __restrict__ dx,
    const int N,
    const int H,
    const float gamma_eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float smem[];

    if (row >= N) return;

    long long base = (long long)row * (long long)H;
    float dot = 0.0f;
    for (int h = tid; h < H; h += blockDim.x) {
        float gv = gamma[h];
        float xnorm = y[base + h] / (gv + gamma_eps);
        float dnorm = dout[base + h] * gv;
        dot += dnorm * xnorm;
    }

    smem[tid] = dot;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    float mean_dot = smem[0] / (float)H;
    float r = rsqrt_in[row];

    for (int h = tid; h < H; h += blockDim.x) {
        float gv = gamma[h];
        float xnorm = y[base + h] / (gv + gamma_eps);
        float dnorm = dout[base + h] * gv;
        dx[base + h] = r * (dnorm - xnorm * mean_dot);
    }
}
''', 'rmsnorm_bwd_dx_f32_kernel')


class RMSNorm:
    def __init__(self, hidden_size, half_float, eps=1e-6):
        self.hidden_size = hidden_size
        self.eps = eps
        self.half_float = half_float

        dtype = cp.float16 if half_float else cp.float32
        self.params = [cp.ones(hidden_size, dtype=dtype)]
        self.cache = None
        self.grads = []
        self.use_raw_kernel = True
        self._raw_kernel_disabled = False
        self._last_input_shape = None

    def _forward_fallback(self, x, is_train=True):
        ga = self.params[0]

        if self.half_float:
            x_f32 = x.astype(cp.float32)
            ga_f = ga.astype(cp.float32)
        else:
            x_f32 = x
            ga_f = ga

        ms = cp.mean(cp.square(x_f32), axis=-1, keepdims=True)
        rsqrt = 1.0 / cp.sqrt(ms + self.eps)
        x_norm = x_f32 * rsqrt
        out = x_norm * ga_f

        if is_train:
            self.cache = rsqrt.reshape(-1).astype(cp.float32, copy=False)
            self._last_input_shape = x.shape

        if self.half_float:
            out = out.astype(cp.float16)
        return out

    def forward(self, x, is_train=True):
        if (not self.use_raw_kernel) or self._raw_kernel_disabled:
            return self._forward_fallback(x, is_train=is_train)

        try:
            ga = self.params[0]
            H = int(self.hidden_size)
            N = int(x.size // H)
            original_shape = x.shape

            x_flat = x.reshape(N, H)
            if not x_flat.flags.c_contiguous:
                x_flat = cp.ascontiguousarray(x_flat)

            out = cp.empty_like(x)
            out_flat = out.reshape(N, H)
            rsqrt = cp.empty((N,), dtype=cp.float32)

            threads = 256
            shared_mem = threads * 4
            if self.half_float:
                if x_flat.dtype != cp.float16 or ga.dtype != cp.float16:
                    return self._forward_fallback(x, is_train=is_train)
                _rmsnorm_fwd_f16_kernel(
                    (N,), (threads,),
                    (x_flat, ga, out_flat, rsqrt, N, H, cp.float32(self.eps)),
                    shared_mem=shared_mem,
                )
            else:
                if x_flat.dtype != cp.float32 or ga.dtype != cp.float32:
                    return self._forward_fallback(x, is_train=is_train)
                _rmsnorm_fwd_f32_kernel(
                    (N,), (threads,),
                    (x_flat, ga, out_flat, rsqrt, N, H, cp.float32(self.eps)),
                    shared_mem=shared_mem,
                )

            if is_train:
                self.cache = rsqrt
                self._last_input_shape = original_shape
            return out
        except Exception:
            self._raw_kernel_disabled = True
            return self._forward_fallback(x, is_train=is_train)

    def _backward_fallback(self, dout, y):
        if self.half_float:
            dout = dout.astype(cp.float32)
            y = y.astype(cp.float32)

        ga = self.params[0]
        if self.half_float:
            ga = ga.astype(cp.float32)

        rsqrt = self.cache
        if rsqrt is not None and rsqrt.ndim == 1:
            rsqrt = rsqrt.reshape(y.shape[0], y.shape[1], 1)

        inv_ga = 1.0 / (ga + cp.float32(1e-6))
        x_norm = y * inv_ga

        d_ga = cp.sum(dout * x_norm, axis=(0, 1))
        d_norm = dout * ga

        term1 = d_norm
        term2 = x_norm * cp.mean(d_norm * x_norm, axis=-1, keepdims=True)
        dx = rsqrt * (term1 - term2)

        if self.half_float:
            dx = dx.astype(cp.float16)
            d_ga = d_ga.astype(cp.float16)

        self.grads = [d_ga]
        self.cache = None
        self._last_input_shape = None
        return dx

    def backward(self, dout, y):
        """
        y = RMSNorm forwardの出力。
        つまり y = x_norm * gamma。
        次層がcacheしているRMSNorm出力を渡して使う。
        """
        if self.cache is None:
            raise RuntimeError("RMSNorm backward called before forward or cache has already been cleared.")

        if (not self.use_raw_kernel) or self._raw_kernel_disabled:
            return self._backward_fallback(dout, y)

        try:
            ga = self.params[0]
            H = int(self.hidden_size)
            N = int(dout.size // H)
            out_shape = dout.shape

            dout_flat = dout.reshape(N, H)
            y_flat = y.reshape(N, H)
            if not dout_flat.flags.c_contiguous:
                dout_flat = cp.ascontiguousarray(dout_flat)
            if not y_flat.flags.c_contiguous:
                y_flat = cp.ascontiguousarray(y_flat)

            rsqrt = self.cache.reshape(N)
            dx = cp.empty_like(dout)
            dx_flat = dx.reshape(N, H)
            d_ga = cp.empty_like(ga)

            threads = 256
            shared_mem = threads * 4
            gamma_eps = cp.float32(1e-6)

            if self.half_float:
                if dout_flat.dtype != cp.float16 or y_flat.dtype != cp.float16 or ga.dtype != cp.float16:
                    return self._backward_fallback(dout, y)
                _rmsnorm_dgamma_f16_kernel(
                    (H,), (threads,),
                    (dout_flat, y_flat, ga, d_ga, N, H, gamma_eps),
                    shared_mem=shared_mem,
                )
                _rmsnorm_bwd_dx_f16_kernel(
                    (N,), (threads,),
                    (dout_flat, y_flat, ga, rsqrt, dx_flat, N, H, gamma_eps),
                    shared_mem=shared_mem,
                )
            else:
                if dout_flat.dtype != cp.float32 or y_flat.dtype != cp.float32 or ga.dtype != cp.float32:
                    return self._backward_fallback(dout, y)
                _rmsnorm_dgamma_f32_kernel(
                    (H,), (threads,),
                    (dout_flat, y_flat, ga, d_ga, N, H, gamma_eps),
                    shared_mem=shared_mem,
                )
                _rmsnorm_bwd_dx_f32_kernel(
                    (N,), (threads,),
                    (dout_flat, y_flat, ga, rsqrt, dx_flat, N, H, gamma_eps),
                    shared_mem=shared_mem,
                )

            self.grads = [d_ga]
            self.cache = None
            self._last_input_shape = None
            return dx.reshape(out_shape)
        except Exception:
            self._raw_kernel_disabled = True
            return self._backward_fallback(dout, y)
    

    
class ChunkedCrossEntropy_Share:
    
    def __init__(self, padding_id=None, chunk_size=1024, half_float=True):
        self.padding_id = padding_id
        self.chunk_size = chunk_size
        self.half_float = half_float
        self.params = []
        self.grad = []

    def forward(self, x, t, W, is_train=True):
        """
        x: (batch, seq_len, hidden_size)
        W: (vocab_size, hidden_size)

        dtype方針:
        - affine 部分は LastAF と同じく、x と W の dtype に任せる
          half_float=True なら通常 x/W は float16 の想定
        - CrossEntropy/logsumexp/softmax 部分だけ、TimeSoftmaxWithLoss と同じく float32 に上げる
        """
        B, T, H = x.shape
        V = W.shape[0]
        N = B * T

        # LastAF と同じく、3D -> 2D は view 変換のみ
        x2 = x.reshape(N, H)
        t1 = t.reshape(N).astype(cp.int32)

        if self.padding_id is None:
            valid = cp.ones((N,), dtype=cp.bool_)
        else:
            valid = (t1 != self.padding_id)

        # CE 側の統計量は float32
        valid_count = cp.sum(valid).astype(cp.float32)
        valid_count = cp.maximum(valid_count, cp.array(1.0, dtype=cp.float32))

        # --------------------------------------------------
        # pass 1: 各token位置ごとの max logit を求める
        # affine は x/W の dtype、CE 部分だけ float32
        # --------------------------------------------------
        max_logits = cp.full((N,), -cp.inf, dtype=cp.float32)

        for s in range(0, V, self.chunk_size):
            e = min(s + self.chunk_size, V)

            Wc = W[s:e]

            # LastAF.forward 相当: affine は dtype を切り替えない
            logits = cp.matmul(x2, Wc.T)

            # TimeSoftmaxWithLoss.forward 相当: CE に入る直前で float32 化
            if self.half_float:
                logits = logits.astype(cp.float32)

            max_logits = cp.maximum(max_logits, cp.max(logits, axis=1))

            del logits

        # --------------------------------------------------
        # pass 2: logsumexp と target logit を求める
        # affine は x/W の dtype、CE 部分だけ float32
        # --------------------------------------------------
        sum_exp = cp.zeros((N,), dtype=cp.float32)
        target_logits = cp.zeros((N,), dtype=cp.float32)

        for s in range(0, V, self.chunk_size):
            e = min(s + self.chunk_size, V)

            Wc = W[s:e]

            # LastAF.forward 相当
            logits = cp.matmul(x2, Wc.T)

            # TimeSoftmaxWithLoss.forward 相当
            if self.half_float:
                logits = logits.astype(cp.float32)

            exp_logits = cp.exp(logits - max_logits[:, None])
            sum_exp += cp.sum(exp_logits, axis=1)

            # 正解tokenがこのchunk内にある位置だけ取り出す
            in_chunk = (t1 >= s) & (t1 < e) & valid

            rows = cp.where(in_chunk)[0]
            cols = t1[rows] - s
            target_logits[rows] = logits[rows, cols]

            del logits, exp_logits

        log_z = cp.log(sum_exp + 1e-9) + max_logits
        losses = log_z - target_logits
        losses = losses * valid.astype(cp.float32)

        loss = cp.sum(losses) / valid_count

        if is_train:
            self.x = x
            self.t = t
            self.max_logits = max_logits
            self.sum_exp = sum_exp
            self.valid = valid
            self.valid_count = valid_count

        return loss

    def backward(self, W, GradScale, dout=1.0, grad_writer=None):
        """
        dtype方針:
        - CE backward は TimeSoftmaxWithLoss.backward と同じく float32 で計算
        - half_float=True のとき GradScale を掛ける
        - affine backward に渡す直前で dlogits を float16 に戻す
        - dW/dx の matmul は LastAF.backward と同じく affine 側 dtype に任せる
        """

        x = self.x
        t = self.t

        B, T, H = x.shape
        V = W.shape[0]
        N = B * T

        # LastAF.backward と同じく view 変換のみ
        x2 = x.reshape(N, H)

        if t.dtype == cp.int32:
            t1 = t.reshape(N)
        else:
            t1 = t.reshape(N).astype(cp.int32)

        max_logits = self.max_logits
        sum_exp = self.sum_exp
        valid = self.valid
        valid_count = self.valid_count

        # CE 側の scale は float32
        scale = cp.asarray(dout, dtype=cp.float32) / valid_count

        valid_f = valid.astype(cp.float32)

        # dW は W と同じ dtype。W が float16 なら dW も float16。
        if grad_writer is None:
            dW = cp.zeros_like(W)
        else:
            dW = None

        # dx は最終的に x.dtype で前段へ返す。
        # half_float=True なら通常 x.dtype は float16 の想定。
        dx2 = cp.zeros((N, H), dtype=x.dtype)

        for s in range(0, V, self.chunk_size):
            e = min(s + self.chunk_size, V)

            Wc = W[s:e]

            # LastAF.forward と同じ affine 再計算。ここでは dtype を上げない。
            logits = cp.matmul(x2, Wc.T)

            # CE backward に入る直前だけ float32 化。
            if self.half_float:
                logits = logits.astype(cp.float32)

            # logits を softmax確率 / dlogits 用バッファとして再利用する
            logits -= max_logits[:, None]
            cp.exp(logits, out=logits)
            logits /= (sum_exp[:, None] + 1e-9)

            # padding位置は勾配ゼロ
            logits *= valid_f[:, None]

            in_chunk = (t1 >= s) & (t1 < e) & valid

            rows = cp.where(in_chunk)[0]
            cols = t1[rows] - s
            logits[rows, cols] -= 1.0

            logits *= scale

            # TimeSoftmaxWithLoss.backward と同じく、half時は GradScale を掛けてから float16 に戻す
            if self.half_float:
                logits *= cp.asarray(GradScale, dtype=cp.float32)
                logits = logits.astype(cp.float16)

            # LastAF.backward 相当
            # W shape が (V, H) なので dW_chunk は (chunk, H)
            dW_chunk = cp.matmul(x2.T, logits).T.astype(W.dtype)

            if grad_writer is None:
                dW[s:e] = dW_chunk
            else:
                grad_writer(s, e, dW_chunk)

            dx2 += cp.matmul(logits, Wc)

            del logits, dW_chunk

        dx = dx2.reshape(B, T, H).astype(x.dtype)

        self.x = None
        self.t = None
        self.W = None
        self.max_logits = None
        self.sum_exp = None
        self.valid = None
        self.valid_count = None

        if grad_writer is None:
            return dx, [dW]
        else:
            return dx, []
        


        

class TimeEmbedding_shard:
    def __init__(self,padding_id,half_float):
        self.half_float=half_float
        self.idx = None
        self.padding_id=padding_id

    def forward(self, idx, w,is_train=True):
        if self.padding_id is not None:
            w[0][self.padding_id][...]=0
        if is_train:
            self.idx = idx.astype(int)
        out = w[0][idx]
        return out

    def backward(self, dout,dW):
        for i in range(dout.shape[0]):
            cp.add.at(dW[0],self.idx[i],dout[i])
        if self.padding_id is not None:
            dW[0][self.padding_id] = 0
        return dW