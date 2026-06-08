import cupy as cp
import wandb
import gc
import numpy as np 

def _pinned_empty(shape, dtype):
    """NumPy array backed by CUDA pinned host memory."""
    dtype = np.dtype(dtype)
    size = int(np.prod(shape))
    mem = cp.cuda.alloc_pinned_memory(size * dtype.itemsize)
    arr = np.frombuffer(mem, dtype=dtype, count=size).reshape(shape)
    return mem, arr


class ZeRO_Adam_Offload:
    """
    Fast CPU-offload AdamW.

    Keeps master weight / m / v on CPU, but avoids the biggest slowdowns in the
    original implementation:
      - no per-step m_hat/v_hat/update_step temporary arrays
      - pinned host buffers for grad D2H and param H2D
      - optional direct write into the existing GPU shard
    """
    def __init__(self, initial_shard_f16, lr_max, total_steps, lr_mini,
                 weight_decay, warm_up, wand,
                 beta1=0.9, beta2=0.99, epsilon=1e-6):
        shape = initial_shard_f16.shape

        self._grad_pin_mem, self.grad_cpu_f16 = _pinned_empty(shape, np.float16)
        self._out_pin_mem, self.out_cpu_f16 = _pinned_empty(shape, np.float16)

        # CPU optimizer state
        self.master_weight_shard = initial_shard_f16.get().astype(np.float32, copy=False)
        self.m = np.zeros_like(self.master_weight_shard, dtype=np.float32)
        self.v = np.zeros_like(self.master_weight_shard, dtype=np.float32)

        # Reusable CPU work buffers. These are the key to avoiding allocator churn.
        self.g = np.empty_like(self.master_weight_shard, dtype=np.float32)
        self.tmp = np.empty_like(self.master_weight_shard, dtype=np.float32)

        self.lr_max = lr_max
        self.lr_mini = lr_mini
        self.total_steps = total_steps
        self.warm_up = warm_up
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.wand = wand

        self.t = 0
        self.current_lr = 0.0

    def _update_learning_rate(self):
        self.t += 1
        if self.t < self.warm_up:
            self.current_lr = self.lr_max * self.t / self.warm_up
        else:
            progress = (self.t - self.warm_up) / max(1, self.total_steps - self.warm_up)
            self.current_lr = self.lr_mini + 0.5 * (self.lr_max - self.lr_mini) * (1 + np.cos(np.pi * progress))
        return self.current_lr

    def update(self, grad_shard_f16, scale, is_main_device,
               out_shard_f16=None, stream=None):
        lr = self._update_learning_rate()

        # GPU -> pinned CPU. This avoids allocating a new CPU array every step.
        grad_shard_f16.get(out=self.grad_cpu_f16, stream=stream)
        if stream is not None:
            stream.synchronize()

        # g = grad * scale, in fp32
        np.multiply(self.grad_cpu_f16, np.float32(scale), out=self.g, casting='unsafe')

        # m = beta1*m + (1-beta1)*g
        self.m *= self.beta1
        self.m += (1.0 - self.beta1) * self.g

        # v = beta2*v + (1-beta2)*g*g
        np.multiply(self.g, self.g, out=self.tmp)
        self.v *= self.beta2
        self.v += (1.0 - self.beta2) * self.tmp

        # AdamW weight decay: w *= (1 - lr*wd)
        if self.weight_decay > 0:
            self.master_weight_shard *= (1.0 - lr * self.weight_decay)

        # tmp = sqrt(v / (1-beta2^t)) + eps
        v_corr = 1.0 / (1.0 - self.beta2 ** self.t)
        np.multiply(self.v, np.float32(v_corr), out=self.tmp)
        np.sqrt(self.tmp, out=self.tmp)
        self.tmp += self.epsilon

        # g = lr * (m / (1-beta1^t)) / tmp
        m_corr = 1.0 / (1.0 - self.beta1 ** self.t)
        np.multiply(self.m, np.float32(lr * m_corr), out=self.g)
        np.divide(self.g, self.tmp, out=self.g)
        self.master_weight_shard -= self.g

        if is_main_device and self.wand:
            wandb.log({"learning_rate": float(lr)})

        # CPU fp32 -> CPU fp16 pinned -> existing GPU shard
        np.copyto(self.out_cpu_f16, self.master_weight_shard, casting='unsafe')
        if out_shard_f16 is not None:
            out_shard_f16.set(self.out_cpu_f16, stream=stream)
            return out_shard_f16
        return cp.asarray(self.out_cpu_f16)


_adam_update_kernel = cp.RawKernel(r'''
#include <cuda_fp16.h>
extern "C" __global__
void adam_update_kernel(
    const half* __restrict__ grad,
    half* __restrict__ param_f16,
    float* __restrict__ master,
    float* __restrict__ m,
    float* __restrict__ v,
    const float lr,
    const float scale,
    const float weight_decay,
    const float beta1,
    const float beta2,
    const float eps,
    const float m_corr,
    const float v_corr,
    const long long n
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = __half2float(grad[i]) * scale;

    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;

    float w = master[i];
    if (weight_decay > 0.0f) {
        w *= (1.0f - lr * weight_decay);
    }

    w -= lr * (mi * m_corr) / (sqrtf(vi * v_corr) + eps);

    m[i] = mi;
    v[i] = vi;
    master[i] = w;
    param_f16[i] = __float2half_rn(w);
}
''', 'adam_update_kernel')


class ZeRO_Adam_ShardedGPU:
    """
    ZeRO-3 compatible GPU-sharded AdamW.

    Only this rank's shard has optimizer state on GPU, so ZeRO-3 sharding is kept.
    Persistent extra VRAM per rank is approximately:
        master fp32 + m fp32 + v fp32 = 12 bytes * shard_numel
    """
    def __init__(self, initial_shard_f16, lr_max, total_steps, lr_mini,
                 weight_decay, warm_up, wand,
                 beta1=0.9, beta2=0.99, epsilon=1e-6):
        self.param_shard_f16 = initial_shard_f16
        self.master_weight_shard = initial_shard_f16.astype(cp.float32)
        self.m = cp.zeros_like(self.master_weight_shard, dtype=cp.float32)
        self.v = cp.zeros_like(self.master_weight_shard, dtype=cp.float32)

        self.lr_max = lr_max
        self.lr_mini = lr_mini
        self.total_steps = total_steps
        self.warm_up = warm_up
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.wand = wand

        self.t = 0
        self.current_lr = 0.0

    def _update_learning_rate(self):
        self.t += 1
        if self.t < self.warm_up:
            self.current_lr = self.lr_max * self.t / self.warm_up
        else:
            progress = (self.t - self.warm_up) / max(1, self.total_steps - self.warm_up)
            self.current_lr = self.lr_mini + 0.5 * (self.lr_max - self.lr_mini) * (1 + np.cos(np.pi * progress))
        return self.current_lr

    def update(self, grad_shard_f16, scale, is_main_device,
               out_shard_f16=None, stream=None):
        lr = self._update_learning_rate()
        out = self.param_shard_f16 if out_shard_f16 is None else out_shard_f16

        n = int(out.size)
        threads = 256
        blocks = (n + threads - 1) // threads

        m_corr = np.float32(1.0 / (1.0 - self.beta1 ** self.t))
        v_corr = np.float32(1.0 / (1.0 - self.beta2 ** self.t))

        _adam_update_kernel(
            (blocks,), (threads,),
            (
                grad_shard_f16,
                out,
                self.master_weight_shard,
                self.m,
                self.v,
                np.float32(lr),
                np.float32(scale),
                np.float32(self.weight_decay),
                np.float32(self.beta1),
                np.float32(self.beta2),
                np.float32(self.epsilon),
                m_corr,
                v_corr,
                np.int64(n),
            ),
            stream=stream,
        )

        if is_main_device and self.wand:
            wandb.log({"learning_rate": float(lr)})

        return out