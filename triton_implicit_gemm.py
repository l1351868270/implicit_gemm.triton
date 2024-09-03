import torch
import triton
import triton.language as tl

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
    key=['N', 'C', 'H', 'W', 'K', 'P', 'Q', 'R', 'S', 'U', 'V', 'pad_h', 'pad_w', 'dila_h', 'dila_w']
)
@triton.jit
def conv2d_kernel(x_ptr, w_ptr, y_ptr, N, C, H, W, K, P, Q, R, S, U, V, pad_h, pad_w, dila_h, dila_w,
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

    n = gemm_i // (P * Q)
    npq_residual = gemm_i % (P * Q)
    p = npq_residual // Q
    q = npq_residual % Q
    k = gemm_j

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for idx_k in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):
        gemm_k = (idx_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) # % GEMM_K
        r = gemm_k // (S * C)
        rsc_residual = gemm_k % (S * C)
        s = rsc_residual // C
        c = rsc_residual % C
        h = p[:, None] * U + r[None, :] * dila_h - pad_h
        w = q[:, None] * V + s[None, :] * dila_w - pad_w
        mask_x = (h >= 0) & (h < H) & (w >= 0) & (w < W)
        mask_w = (r < R) & (s < S) & (c < C)
        offs_x = n[:, None] * H * W * C + h * W * C + w * C + c
        offs_w = k[None, :] * R * S * C + r[:, None] * S * C + s[:, None] * C + c[:, None]

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


def triton_implicit_gemm(x: torch.Tensor, w: torch.Tensor, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    N, C, H, W = x.shape
    K, C, R, S = w.shape
    U, V = stride
    pad_h, pad_w = padding
    dila_h, dila_w = dilation
    P = (H + 2 * pad_h - dila_h * (R - 1) - 1) // U + 1
    Q = (W + 2 * pad_w - dila_w * (S - 1) - 1) // V + 1
    y = torch.empty((N, K, P, Q), device=x.device, dtype=torch.float16).to(memory_format=torch.channels_last)
    GEMM_M = N * P * Q
    GEMM_N = K
    GEMM_K = C * R * S
    grid = lambda META: (triton.cdiv(GEMM_M, META['BLOCK_SIZE_M']) * triton.cdiv(GEMM_N, META['BLOCK_SIZE_N']), )
    conv2d_kernel[grid](x, w, y, N, C, H, W, K, P, Q, R, S, U, V, pad_h, pad_w, dila_h, dila_w, GEMM_M, GEMM_N, GEMM_K)
    return y

