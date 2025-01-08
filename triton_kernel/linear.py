import torch
import triton
import triton.language as tl
import pytest


class _naive_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output += bias
        ctx.save_for_backward(input, weight, bias)
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight, bias, = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = None, None, None
        if input.requires_grad:
            grad_input = torch.matmul(grad_output, weight)
        if weight.requires_grad:
            grad_weight = torch.matmul(grad_output.t(), input)
        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.sum(0, keepdim=False)
        print(f'grad_bias: {grad_bias.shape}')
        return grad_input, grad_weight, grad_bias
        

naive_linear = _naive_linear.apply


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
def _ld_linear_kernel(
        A_ptr, B_ptr, C_ptr, bias_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn, 
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  
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

    if bias_ptr is not None:
        offs_bias = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N
        bias_ptrs = bias_ptr + offs_bias
        bias = tl.load(bias_ptrs, mask=offs_bias < N, other=0.0)
        accumulator += bias

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def _ld_bias_kernel(
        output_ptr, input_ptr,
        n_rows, n_cols,
        input_row_stride,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_rows
    output_ptrs = output_ptr + pid
    input_ptrs = input_ptr + (pid + col_offsets * input_row_stride)

    input = tl.load(input_ptrs, mask=mask, other=0.0)
    output = tl.sum(input, axis=0)
    tl.store(output_ptrs, output)


class _ld_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        M, K = input.shape
        N, K = weight.shape
        output = torch.empty((M, N), device = input.device, dtype=input.dtype)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
        _ld_linear_kernel[grid](
            input, weight, output, bias,
            M, N, K, 
            input.stride(0), input.stride(1), 
            weight.stride(1), weight.stride(0), 
            output.stride(0), output.stride(1),
        )
        ctx.save_for_backward(input, weight, bias)
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = None, None, None
        M, N = grad_output.shape
        if input.requires_grad:
            _, K = weight.shape
            grad_input = torch.empty((M, K), device=input.device, dtype=input.dtype)
            grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_N']),)
            _ld_linear_kernel[grid](
                grad_output, weight, grad_input, None,
                M, K, N, 
                grad_output.stride(0), grad_output.stride(1), 
                weight.stride(0), weight.stride(1), 
                grad_input.stride(0), grad_input.stride(1),
            )
        if weight.requires_grad:
            _, K = input.shape
            grad_weight = torch.empty((N, K), device=weight.device, dtype=weight.dtype)
            grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_N']),)
            _ld_linear_kernel[grid](
                grad_output, input, grad_weight, None,
                N, K, M, 
                grad_output.stride(1), grad_output.stride(0), 
                input.stride(0), input.stride(1), 
                grad_weight.stride(0), grad_weight.stride(1),
            )

        if bias.requires_grad:
            shape = grad_output.shape
            n_rows, n_cols = grad_output.shape
            grad_bias = torch.empty_like(bias)
            BLOCK_SIZE = triton.next_power_of_2(n_rows)
            _ld_bias_kernel[(n_cols, )](
                grad_bias, grad_output,
                n_rows, n_cols,
                grad_output.stride(0),
                BLOCK_SIZE
            )

            grad_output = grad_output.view(*shape)
        return grad_input, grad_weight, grad_bias


ld_linear = _ld_linear.apply


# python -m pytest -v -rsx -s -W ignore::UserWarning linear.py -k test_linear
@pytest.mark.parametrize('M, in_features, out_features', [(128, 256, 512)])
def test_linear(M, in_features, out_features):
    factory_kwargs = {'device': 'cuda', 'dtype': torch.float}
    input = torch.randn(M, in_features, requires_grad=True, **factory_kwargs)
    weight = torch.randn(out_features, in_features, requires_grad=True, **factory_kwargs)
    bias = torch.randn(out_features, requires_grad=True, **factory_kwargs)
    # bias = None

    output = torch.functional.F.linear(input, weight, bias)
    doutput = torch.rand_like(output)
    output.backward(doutput, retain_graph=True)
    dinput, input.grad = input.grad.clone(), None
    dweight, weight.grad = weight.grad.clone(), None
    # dbias, bias.grad = bias.grad.clone(), None

    print(f'output: {output.shape}')

    naive_output = naive_linear(input, weight, bias)
    naive_output.backward(doutput, retain_graph=True)
    naive_dinput, input.grad = input.grad.clone(), None
    naive_dweight, weight.grad = weight.grad.clone(), None
    naive_dbias, bias.grad = bias.grad.clone(), None
    print(f'naive_dbias: {naive_dbias}')

    ld_output = ld_linear(input, weight, bias)
    ld_output.backward(doutput)
    ld_dinput, input.grad = input.grad.clone(), None
    ld_dweight, weight.grad = weight.grad.clone(), None
    ld_dbias, bias.grad = bias.grad.clone(), None
    print(f'ld_dbias: {ld_dbias}')


    assert torch.allclose(output, naive_output, atol=1e-3, rtol=1e-3)
    assert torch.allclose(output, ld_output, atol=1e-1, rtol=1e-1)
    assert torch.allclose(dinput, naive_dinput, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dweight, naive_dweight, atol=1e-3, rtol=1e-3)
    # assert torch.allclose(dbias, naive_dbias, atol=1e-3, rtol=1e-3)

    