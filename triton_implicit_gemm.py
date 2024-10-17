import torch
import triton
import triton.language as tl
import numpy as np

def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
        #               num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
        #               num_warps=2),
        # # Good config for fp8 inputs.
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
        #               num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
        #               num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4)
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
    print(f'x: shape:{x.shape}, stride:{x.stride()}')
    print(f'w: shape:{w.shape}, stride:{w.stride()}')
    grid = lambda META: (triton.cdiv(GEMM_M, META['BLOCK_SIZE_M']) * triton.cdiv(GEMM_N, META['BLOCK_SIZE_N']), )
    conv2d_kernel[grid](x, w, y, N, C, H, W, K, P, Q, R, S, U, V, pad_h, pad_w, dila_h, dila_w, GEMM_M, GEMM_N, GEMM_K)
    return y


class LSLTensor(object):
    def __init__(self, pycuda_data, shape, dtype, strides):
        self.pycuda_data = pycuda_data
        self.shape = shape
        self.dtype = dtype
        self.strides = strides

    def data_ptr(self):
        # print(f'data_ptr: {int(self.data_ptr_)}')
        return int(self.pycuda_data)


def h_conv2d(x_ptr, w_ptr, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    N, H, W, C = x_ptr.shape
    K, R, S, C = w_ptr.shape
    U, V = stride
    pad_h, pad_w = padding
    dila_h, dila_w = dilation
    P = (H + 2 * pad_h - dila_h * (R - 1) - 1) // U + 1
    Q = (W + 2 * pad_w - dila_w * (S - 1) - 1) // V + 1

    GEMM_M = N * P * Q
    GEMM_N = K
    GEMM_K = C * R * S
    y = np.empty((N, P, Q, K), dtype=np.float16)
    for gemm_i in range(GEMM_M):
        for gemm_j in range(GEMM_N):
            n = gemm_i // (P * Q)
            npq_residual = gemm_i % (P * Q)
            p = npq_residual // Q
            q = npq_residual % Q
            k = gemm_j

            accum = np.zeros((), dtype=np.float32)
            for gemm_k in range(GEMM_K):
                r = gemm_k // (S * C)
                rsc_residual = gemm_k % (S * C)
                s = rsc_residual // C
                c = rsc_residual % C
                h = p * U + r * dila_h - pad_h
                w = q * V + s * dila_w - pad_w

                if h >= 0 and h < H and w >= 0 and w < W:
                    x_data = x_ptr[n, h, w, c]
                else:
                    x_data = 0.0
                w_data = w_ptr[k, r, s, c]
                accum += x_data * w_data
            y[n, p, q, k] = accum.astype(np.float16)

    return y


if __name__ == '__main__':
    torch.manual_seed(0)
    N = 1; C = 64; H = 56; W = 56; K = 64; R = 1; S = 1; pad_h = 0; pad_w = 0; U = 1; V = 1; dila_h = 1; dila_w = 1
    # N = 1; C = 1; H = 3; W = 3; K = 1; R = 3; S = 3; pad_h = 0; pad_w = 0; U = 1; V = 1; dila_h = 1; dila_w = 1

    x = torch.randn(N, C, H, W).cuda().half()
    w = torch.randn(K, C, R, S).cuda().half()
    conv2d = torch.nn.Conv2d(C, K, (R, S), stride=(U, V), padding=(pad_h, pad_w), dilation=(dila_h, dila_w), bias=False).cuda().half()
    conv2d.weight.data = w
    y1 = conv2d(x)
    print(f'x: shape:{x.shape}, stride:{x.stride()}')
    w = w.to(memory_format=torch.channels_last)
    x = x.to(memory_format=torch.channels_last)
    print(f'x: shape:{x.shape}, stride:{x.stride()}')
    conv2d = conv2d.to(memory_format=torch.channels_last)
    conv2d.weight.data = w
    y2 = conv2d(x)

    y3 = triton_implicit_gemm(x, w, stride=(U, V), padding=(pad_h, pad_w), dilation=(dila_h, dila_w))
    print(f'y3: shape:{y3.shape}, stride:{y3.stride()}')
    if torch.allclose(y1, y3, atol=1e-2, rtol=1e-2):
        print("Torch and triton_implicit_gemm match")
    else:
        print("Torch and triton_implicit_gemm differ")
        print(f'torch: shape:{y1.shape}, stride:{y1.stride()}, {y1}')
        print(f'triton_implicit_gemm: shape:{y3.shape}, stride:{y3.stride()}, {y3}')

    # h_w = w.cpu().numpy().transpose(0, 2, 3, 1)
    # h_x = x.cpu().numpy().transpose(0, 2, 3, 1)
    # h_y1 = y1.detach().cpu().numpy().transpose(0, 2, 3, 1)
    # print(f'h_w: shape:{h_y1.shape}, stride:{h_y1.strides}')
    # h_y = h_conv2d(h_x, h_w, stride=(U, V), padding=(pad_h, pad_w), dilation=(dila_h, dila_w))
    # if np.allclose(h_y1, h_y, atol=1e-2, rtol=1e-2):
    #     print("Torch and numpy match")
    # else:
    #     print("Torch and numpy differ")
    #     print(f'torch: shape:{h_y1.shape}, {h_y1}')
    #     print(f'numpy: shape:{h_y.shape}, {h_y}')
    
