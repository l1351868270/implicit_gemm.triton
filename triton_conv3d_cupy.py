import cupy as cp
import numpy as np

import triton
import triton.language as tl


class LSLndarray(object):
    def __init__(self, cupy_data):
        self.cupy_data= cupy_data
        self.data = cupy_data.data
        self.dtype = cupy_data.dtype
        self.shape = cupy_data.shape
        self.strides = cupy_data.strides

    def data_ptr(self):
        return self.data.ptr
    
    def numpy(self):
        return cp.asnumpy(self.cupy_data)
    

def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=['N', 'C', 'D', 'H', 'W', 'K', 'D_out', 'H_out', 'W_out', 'T', 'R', 'S', 'stride_d', 'stride_h', 'stride_w', 'pad_d', 'pad_h', 'pad_w', 'dila_d', 'dila_h', 'dila_w']
)
@triton.jit
def conv3d_kernel(x_ptr, w_ptr, y_ptr, N, C, D, H, W, K, D_out, H_out, W_out, T, R, S, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dila_d, dila_h, dila_w, 
                  GEMM_M, GEMM_N, GEMM_K,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    gemm_i = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % GEMM_M
    gemm_j = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % GEMM_N

    n = gemm_i // (D_out * H_out * W_out)
    ndhw_residual = gemm_i % (D_out * H_out * W_out)
    d_out = ndhw_residual // (H_out * W_out)
    dhw_residual = ndhw_residual % (H_out * W_out)
    h_out = dhw_residual // W_out
    w_out = dhw_residual % W_out
    k = gemm_j

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for idx_k in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):
        gemm_k = (idx_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) # % GEMM_K
        t = gemm_k // (R * S * C)
        trsc_residual = gemm_k % (R * S * C)
        r = trsc_residual // (S * C)
        rsc_residual = gemm_k % (S * C)
        s = rsc_residual // C
        c = rsc_residual % C
        d = d_out[:, None] * stride_d + t[None, :] * dila_d - pad_d
        h = h_out[:, None] * stride_h + r[None, :] * dila_h - pad_h
        w = w_out[:, None] * stride_w + s[None, :] * dila_w - pad_w
        mask_x = (d >= 0) & (d < D) & (h >= 0) & (h < H) & (w >= 0) & (w < W)
        mask_w = (t < T) & (r < R) & (s < S) & (c < C)
        offs_x = n[:, None] * D * H * W * C + d * H * W * C + h * W * C + w * C + c
        offs_w = k[None, :] * T * R * S * C + t[:, None] * R * S * C + r[:, None] * S * C + s[:, None] * C + c[:, None]

        x_ptrs = x_ptr + offs_x
        w_ptrs = w_ptr + offs_w

        x_data = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w_data = tl.load(w_ptrs, mask=mask_w[:, None], other=0.0)
        accumulator = tl.dot(x_data, w_data, accumulator)
    c_data = accumulator.to(tl.float16)

    offs_y = gemm_i[:, None] * GEMM_N + gemm_j[None, :]
    mask_y = (gemm_i[:, None] < GEMM_M) & (gemm_j[None, :] < GEMM_N)
    y_ptrs = y_ptr + offs_y
    tl.store(y_ptrs, c_data, mask=mask_y)


def triton_conv3d(x: LSLndarray, w: LSLndarray, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    N, D, H, W, C = x.shape
    K, T, R, S, C = w.shape
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    dila_d, dila_h, dila_w = dilation
    D_out = (D + 2 * pad_d - dila_d * (T - 1) - 1) // stride_d + 1
    H_out = (H + 2 * pad_h - dila_h * (R - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dila_w * (S - 1) - 1) // stride_w + 1
    y = LSLndarray(cp.empty((N, D_out, H_out, W_out, K), dtype=cp.float16))
    GEMM_M = N * D_out * H_out * W_out
    GEMM_N = K
    GEMM_K = T * R * S * C

    grid = lambda META: (triton.cdiv(GEMM_M, META['BLOCK_SIZE_M']) * triton.cdiv(GEMM_N, META['BLOCK_SIZE_N']), )
    conv3d_kernel[grid](x, w, y, N, C, D, H, W, K, D_out, H_out, W_out, T, R, S, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dila_d, dila_h, dila_w, GEMM_M, GEMM_N, GEMM_K)
    return y


def h_conv3d(x_ptr, w_ptr, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1)):
    N, D, H, W, C = x_ptr.shape
    K, T, R, S, C = w_ptr.shape
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    dila_d, dila_h, dila_w = dilation
    D_out = (D + 2 * pad_d - dila_d * (T - 1) - 1) // stride_d + 1
    H_out = (H + 2 * pad_h - dila_h * (R - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dila_w * (S - 1) - 1) // stride_w + 1

    GEMM_M = N * D_out * H_out * W_out
    GEMM_N = K
    GEMM_K = T * R * S * C

    y = np.empty((N, D_out, H_out, W_out, K), dtype=np.float16)

    for gemm_i in range(GEMM_M):
        for gemm_j in range(GEMM_N):
            n = gemm_i // (D_out * H_out * W_out)
            ndhw_residual = gemm_i % (D_out * H_out * W_out)
            d_out = ndhw_residual // (H_out * W_out)
            dhw_residual = ndhw_residual % (H_out * W_out)
            h_out = dhw_residual // W_out
            w_out = dhw_residual % W_out
            k = gemm_j

            accum = np.zeros((), dtype=np.float32)
            for gemm_k in range(GEMM_K):
                t = gemm_k // (R * S * C)
                trsc_residual = gemm_k % (R * S * C)
                r = trsc_residual // (S * C)
                rsc_residual = trsc_residual % (S * C)
                s = rsc_residual // C
                c = rsc_residual % C
                d = d_out * stride_d + t * dila_d - pad_d
                h = h_out * stride_h + r * dila_h - pad_h
                w = w_out * stride_w + s * dila_w - pad_w

                if d >= 0 and d < D and h >= 0 and h < H and w >= 0 and w < W:
                    x_data = x_ptr[n, d, h, w, c]
                else:
                    x_data = 0.0
                w_data = w_ptr[k, t, r, s, c]
                accum += x_data * w_data
            y[n, d_out, h_out, w_out, k] = accum.astype(np.float16)

    return y


if __name__ == '__main__':
    np.random.seed(0)
    # N = 1; C = 64; D = 56; H = 56; W = 56; K = 64; T = 1; R = 1; S = 1; pad_d = 0; pad_h = 0; pad_w = 0; stride_d = 1; stride_h = 1; stride_w = 1; dila_d = 1; dila_h = 1; dila_w = 1
    N = 1; C = 16; D = 6; H = 6; W = 6; K = 16; T = 1; R = 1; S = 1; pad_d = 0; pad_h = 0; pad_w = 0; stride_d = 1; stride_h = 1; stride_w = 1; dila_d = 1; dila_h = 1; dila_w = 1

    x = np.random.randn(N, C, D, H, W).astype(np.float16)
    w = np.random.randn(K, C, T, R, S).astype(np.float16)

    x_channel_last = np.ascontiguousarray(x.transpose(0, 2, 3, 4, 1))
    w_channel_last = np.ascontiguousarray(w.transpose(0, 2, 3, 4, 1))

    x_channel_last = LSLndarray(cp.array(x_channel_last))
    w_channel_last = LSLndarray(cp.array(w_channel_last))

    y1 = triton_conv3d(x_channel_last, w_channel_last, stride=(stride_d, stride_h, stride_w), padding=(pad_d, pad_h, pad_w), dilation=(dila_d, dila_h, dila_w))

    h_w = w_channel_last.numpy()
    h_x = x_channel_last.numpy()
    h_y1 = y1.numpy()

    h_y = h_conv3d(h_x, h_w, stride=(stride_d, stride_h, stride_w), padding=(pad_d, pad_h, pad_w), dilation=(dila_d, dila_h, dila_w))
    if np.allclose(h_y1, h_y, atol=1e-2, rtol=1e-2):
        print("Triton and numpy match")
    else:
        print("Triton and numpy differ")
        print(f'Triton: shape:{h_y1.shape}, {h_y1}')
        print(f'numpy: shape:{h_y.shape}, {h_y}')
    