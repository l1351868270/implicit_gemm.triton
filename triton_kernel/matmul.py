
import torch
import triton
import triton.language as tl
import pytest


class _naive_matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor):
        ctx.save_for_backward(A, B)
        C = torch.matmul(A, B)
        return C
    
    @staticmethod
    def backward(ctx, dC: torch.Tensor):
        # https://github.com/l1351868270/implicit_gemm.triton/blob/main/triton_kernel/matmul.md
        A, B = ctx.saved_tensors
        dA = torch.matmul(dC, B.t())
        dB = torch.matmul(A.t(), dC)
        return dA, dB
    

naive_matmul = _naive_matmul.apply


autotune_config = [
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
    configs=autotune_config,
    key=['M', 'N', 'K'],
)
@triton.jit
def _tt_matmul_kernel(
        A_ptr, B_ptr, C_ptr, 
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn, 
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  
        ACTIVATION: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


class _tt_matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor, dtype=torch.float16):
        assert A.shape[1] == B.shape[0]
        assert A.is_contiguous()
        assert A.dtype == dtype
        assert B.dtype == dtype
        M, K = A.shape
        K, N = B.shape
        C = torch.empty(M, N, device=A.device, dtype=dtype)
        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
        _tt_matmul_kernel[grid](
            A, B, C,
            M, N, K, 
            A.stride(0), A.stride(1), 
            B.stride(0), B.stride(1), 
            C.stride(0), C.stride(1), 
            ACTIVATION=""
        )
        ctx.save_for_backward(A, B)
        return C
    
    @staticmethod
    def backward(ctx, dC: torch.Tensor):
        # https://github.com/l1351868270/implicit_gemm.triton/blob/main/triton_kernel/matmul.md
        A, B = ctx.saved_tensors
        M, K = A.shape
        K, N = B.shape

        dA, dB = None, None
        if A.requires_grad:
            B_t = B.t()
            dA = torch.zeros_like(A)
            grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(K, META["BLOCK_SIZE_N"]),)
            _tt_matmul_kernel[grid](
                dC, B_t, dA,
                M, K, N, 
                dC.stride(0), dC.stride(1), 
                B_t.stride(0), B_t.stride(1), 
                dA.stride(0), dA.stride(1), 
                ACTIVATION=""
            )

        if B.requires_grad:
            A_t = A.t()
            dB = torch.zeros_like(B)
            grid = lambda META: (triton.cdiv(K, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
            _tt_matmul_kernel[grid](
                A_t, dC, dB,
                K, N, M, 
                A_t.stride(0), A_t.stride(1), 
                dC.stride(0), dC.stride(1), 
                dB.stride(0), dB.stride(1), 
                ACTIVATION=""
            )

        return dA, dB
    

tt_matmul = _tt_matmul.apply


@pytest.mark.parametrize('M, N, K', [(512, 512 + 1, 512 + 2)])
def test_matmul(M, N, K, dtype=torch.float16, device='cuda'):
    A = torch.randn(M, K, device=device, dtype=dtype, requires_grad=True)
    B = torch.randn(K, N, device=device, dtype=dtype, requires_grad=True)
    dC = torch.randn(M, N, device=device, dtype=dtype, requires_grad=True)
    C = naive_matmul(A, B)
    C.backward(dC)
    dA, A.grad = A.grad.clone(), None
    dB, B.grad = B.grad.clone(), None

    naive_C = naive_matmul(A, B)
    naive_C.backward(dC)
    naive_dA, A.grad = A.grad.clone(), None
    naive_dB, B.grad = B.grad.clone(), None

    tt_C = tt_matmul(A, B)
    tt_C.backward(dC)
    tt_dA, A.grad = A.grad.clone(), None
    tt_dB, B.grad = B.grad.clone(), None
    
    assert torch.allclose(C, naive_C, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dA, naive_dA, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dB, naive_dB, atol=1e-3, rtol=1e-3)

    assert torch.allclose(C, tt_C, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dA, tt_dA, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dB, tt_dB, atol=1e-3, rtol=1e-3)

