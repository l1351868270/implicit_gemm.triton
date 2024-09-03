import torch
import triton
import triton.language as tl
import triton.ops

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
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=1),
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
    key=['GEMM_M', 'GEMM_N', 'GEMM_K'],
)
@triton.jit
def conv2d_kernel_1x1_1x1_0x0_1x1(
        x_ptr, w_ptr, y_ptr,
        N, C, H, W, K, P, Q, R, S, U, V, pad_h, pad_w, dila_h, dila_w, 
        GEMM_M, GEMM_N, GEMM_K,
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
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

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % GEMM_M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % GEMM_N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = x_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = w_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

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

        a = tl.load(x_ptrs, mask=mask_x, other=0.0)
        b = tl.load(w_ptrs, mask=mask_w[:, None], other=0.0)
        # a = tl.load(a_ptrs, mask=offs_k[None, :] < GEMM_K - idx_k * BLOCK_SIZE_K, other=0.0)
        # b = tl.load(b_ptrs, mask=offs_k[:, None] < GEMM_K - idx_k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = y_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < GEMM_M) & (offs_cn[None, :] < GEMM_N)
    tl.store(c_ptrs, c, mask=c_mask)

def triton_implicit_gemm_1x1_0x0_1x1(x: torch.Tensor, w: torch.Tensor, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    # Check constraints.
    assert x.shape[1] == w.shape[1], "Incompatible dimensions"

    N, C, H, W = x.shape
    K, C, R, S = w.shape
    U, V = stride
    pad_h, pad_w = padding
    dila_h, dila_w = dilation
    P = (H + 2 * pad_h - dila_h * (R - 1) - 1) // U + 1
    Q = (W + 2 * pad_w - dila_w * (S - 1) - 1) // V + 1
    # y = torch.empty(N, K, P, Q, device=x.device, dtype=torch.float16).to(memory_format=torch.channels_last)
    
    GEMM_M = N * P * Q
    GEMM_N = K
    GEMM_K = C * R * S
    y = torch.empty((N, K, P, Q), device=x.device, dtype=torch.float16, memory_format=torch.channels_last)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(GEMM_M, META['BLOCK_SIZE_M']) * triton.cdiv(GEMM_N, META['BLOCK_SIZE_N']), )
    conv2d_kernel_1x1_1x1_0x0_1x1[grid](
        x, w, y,  #
        N, C, H, W, K, P, Q, R, S, U, V, pad_h, pad_w, dila_h, dila_w, 
        GEMM_M, GEMM_N, GEMM_K,  #
        GEMM_K, 1,  #
        1, GEMM_K,  #
        GEMM_N, 1,  #
    )
    return y


if __name__ == '__main__':
    torch.manual_seed(0)
    N = 1; C = 128; H = 56; W = 56; K = 64; R = 1; S = 1; pad_h = 0; pad_w = 0; U = 1; V = 1; dila_h = 1; dila_w = 1
    # N = 1; C = 1; H = 3; W = 3; K = 1; R = 3; S = 3; pad_h = 0; pad_w = 0; U = 1; V = 1; dila_h = 1; dila_w = 1

    x = torch.randn(N, C, H, W).cuda().half()
    w = torch.randn(K, C, R, S).cuda().half()
    conv2d = torch.nn.Conv2d(C, K, (R, S), stride=(U, V), padding=(pad_h, pad_w), dilation=(dila_h, dila_w), bias=False).cuda().half()
    conv2d.weight.data = w
    y1 = conv2d(x)

    w = w.to(memory_format=torch.channels_last)
    x = x.to(memory_format=torch.channels_last)
    conv2d = conv2d.to(memory_format=torch.channels_last)
    conv2d.weight.data = w
    y2 = conv2d(x)

    y3 = triton_implicit_gemm_1x1_0x0_1x1(x, w, stride=(U, V), padding=(pad_h, pad_w), dilation=(dila_h, dila_w))
    
    if torch.allclose(y1, y3, atol=1e-2, rtol=1e-2):
        print("Torch and triton_implicit_gemm match")
    else:
        print("Torch and triton_implicit_gemm differ")
        print(f'torch: shape:{y1.shape}, stride:{y1.stride()}, {y1}')
        print(f'triton_implicit_gemm: shape:{y3.shape}, stride:{y3.stride()}, {y3}')
    

