
# Adapted from https://github.com/triton-lang/triton/blob/main/python/tutorials/02-fused-softmax.py
import torch
import triton
import triton.language as tl
from triton.runtime import driver
import pytest
from typing import Optional, Tuple

class _naive_softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        x_max = x.max(dim=-1, keepdim=True).values
        x = x - x_max
        numerator = torch.exp(x)
        denominator = numerator.sum(dim=-1, keepdim=True)
        ret = numerator / denominator
        ctx.save_for_backward(ret)
        return ret

    @staticmethod
    def backward(ctx, dp: torch.Tensor):
        p, = ctx.saved_tensors
        ds = torch.zeros_like(p)
        for i in range(p.shape[0]):
            ds[i] =  p[i]*(dp[i] - (p[i] * dp[i]).sum(dim=-1, keepdim=True))
        return ds


naive_softmax = _naive_softmax.apply


@triton.jit
def _ld_softmax_fwd_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols,
                           BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


@triton.jit
def _ld_softmax_bwd_kernel(ds_ptr, p_ptr, dp_ptr, ds_row_stride, p_row_stride, dp_row_stride, n_rows, n_cols,
                           BLOCK_SIZE: tl.constexpr):
    # https://github.com/l1351868270/implicit_gemm.triton/blob/main/triton_kernel/softmax.md
    row_idx = tl.program_id(0)
    p_start_ptr = p_ptr + row_idx * p_row_stride
    dp_start_ptr = dp_ptr + row_idx * dp_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    p_ptrs = p_start_ptr + col_offsets
    dp_ptrs = dp_start_ptr + col_offsets
    mask = col_offsets < n_cols
    p_row = tl.load(p_ptrs, mask=mask, other=0)
    dp_row = tl.load(dp_ptrs, mask=mask, other=0)
    ds_row = p_row * (dp_row - tl.sum(p_row * dp_row, axis=0))

    ds_start_ptr = ds_ptr + row_idx * ds_row_stride
    ds_ptrs = ds_start_ptr + col_offsets
    tl.store(ds_ptrs, ds_row, mask=mask)


class _ld_softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        shape = x.shape
        x = x.view(-1, x.shape[-1])
        n_rows, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        p = torch.empty_like(x)
        _ld_softmax_fwd_kernel[(n_rows,)](
            p,
            x,
            x.stride(0),
            p.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE,
        )
        p = p.view(*shape)
        x = x.view(*shape)
        ctx.save_for_backward(p)
        return p
    
    @staticmethod
    def backward(ctx, dp: torch.Tensor):
        p, = ctx.saved_tensors
        shape = p.shape
        p = p.view(-1, p.shape[-1])
        dp = dp.view(-1, dp.shape[-1])
        n_rows, n_cols = p.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        ds = torch.empty_like(p)
        _ld_softmax_bwd_kernel[(n_rows,)](
            ds,
            p,
            dp,
            ds.stride(0),
            p.stride(0),
            dp.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE,
        )
        p = p.view(*shape)
        dp = dp.view(*shape)
        return ds.view(*shape)
        

ld_softmax = _ld_softmax.apply


# Adapted from https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py
# only replace forward
class LDSoftmax(torch.nn.Module):
    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.dim is not None and self.dim != -1 and self.dim != input.dim() - 1:
            raise NotImplementedError("Only last dimension ld softmax is supported")
        return ld_softmax(input)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


# python -m pytest -s triton_kernel/softmax.py
@pytest.mark.parametrize('M, N', [(1823, 781)])
def test_2d_softmax(M, N):
    x = torch.randn(M, N, requires_grad=True, device='cuda')
    y = torch.softmax(x, dim=-1)
    dp = torch.randn_like(x)
    y.backward(dp)
    dx, x.grad = x.grad.clone(), None
    naive_y = naive_softmax(x)
    naive_y.backward(dp)
    naive_dx, x.grad = x.grad.clone(), None
    assert torch.allclose(dx, naive_dx, rtol=1e-3, atol=1e-3)
    assert torch.allclose(y, naive_y, rtol=1e-3, atol=1e-3)
    tt_y = ld_softmax(x)
    tt_y.backward(dp)
    tt_dx, x.grad = x.grad.clone(), None
    assert torch.allclose(y, tt_y, rtol=1e-3, atol=1e-3)
    assert torch.allclose(dx, tt_dx, rtol=1e-3, atol=1e-3)
    

@pytest.mark.parametrize('B, N, D', [(3, 781, 129)])
def test_3d_softmax(B, N, D):
    x = torch.randn(B, N, D, requires_grad=True, device='cuda')
    y = torch.softmax(x, dim=-1)
    dp = torch.randn_like(x)
    y.backward(dp)
    dx, x.grad = x.grad.clone(), None
    naive_y = naive_softmax(x)
    naive_y.backward(dp)
    naive_dx, x.grad = x.grad.clone(), None
    assert torch.allclose(dx, naive_dx, rtol=1e-3, atol=1e-3)
    assert torch.allclose(y, naive_y, rtol=1e-3, atol=1e-3)
    tt_y = ld_softmax(x)
    tt_y.backward(dp)
    tt_dx, x.grad = x.grad.clone(), None
    assert torch.allclose(y, tt_y, rtol=1e-3, atol=1e-3)
    assert torch.allclose(dx, tt_dx, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('B, N, H, D', [(3, 781, 8, 129)])
def test_4d_softmax(B, N, H, D):
    x = torch.randn(B, N, H, D, requires_grad=True, device='cuda')
    y = torch.softmax(x, dim=-1)
    dp = torch.randn_like(x)
    y.backward(dp)
    dx, x.grad = x.grad.clone(), None
    naive_y = naive_softmax(x)
    naive_y.backward(dp)
    naive_dx, x.grad = x.grad.clone(), None
    assert torch.allclose(dx, naive_dx, rtol=1e-3, atol=1e-3)
    assert torch.allclose(y, naive_y, rtol=1e-3, atol=1e-3)
    tt_y = ld_softmax(x)
    tt_y.backward(dp)
    tt_dx, x.grad = x.grad.clone(), None
    assert torch.allclose(y, tt_y, rtol=1e-3, atol=1e-3)
    assert torch.allclose(dx, tt_dx, rtol=1e-3, atol=1e-3)


# python -m pytest -s triton_kernel/softmax.py -k test_Softmax
@pytest.mark.parametrize('M, N', [(1823, 781)])
def test_Softmax(M, N):

    m = torch.nn.Softmax(dim=1)
    x = torch.randn(M, N, requires_grad=True, device='cuda')
    y = m(x)
    target = torch.randn_like(x)
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(y, target)
    loss.backward()
    dx, x.grad = x.grad.clone(), None

    ld_m = LDSoftmax(dim=-1)
    ld_y = ld_m(x)
    loss = loss_fn(ld_y, target)
    loss.backward()
    ld_dx, x.grad = x.grad.clone(), None

    assert torch.allclose(y, ld_y, rtol=1e-3, atol=1e-3)
    assert torch.allclose(dx, ld_dx, rtol=1e-3, atol=1e-3)


perf_configs = []


@triton.testing.perf_report(perf_configs)
def benchmark(M, N, mode, provider, device='cuda'):
    x = torch.randn(M, N, device=device, requires_grad=True)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == 'torch':
        fn = lambda: torch.softmax(x, dim=-1)
        if mode == 'bwd':
            o =fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    if provider == 'triton':
        fn = lambda: ld_softmax(x)
        if mode == 'bwd':
            o =fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    if provider == 'naive':
        fn = lambda: naive_softmax(x)
        if mode == 'bwd':
            o =fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    if mode == 'fwd':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e3)
    if mode == 'bwd':
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e3)
    return gbps(ms)


if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--test', action='store_true')
    args.add_argument('--bench', action='store_true')
    args.add_argument('--plot', action='store_true')
    args.add_argument('--mode', choices=('fwd', 'bwd'), default='fwd')
    args = args.parse_args()
    test = args.test
    bench = args.bench
    plot = args.plot
    print_data = bench
    mode = args.mode
    # for mode in ['fwd', 'bwd']:
    if mode == 'fwd':
        perf_configs.append(
            triton.testing.Benchmark(
                x_names=['N'],
                x_vals=[128 * i for i in range(2, 10)],
                line_arg='provider',
                line_vals=['triton', 'torch', 'naive'],
                line_names=['Triton', 'Torch', 'naive'],
                styles=[('blue', '-'), ('green', '-'), ('red', '-')],
                ylabel='GB/s',
                plot_name="softmax-performance",
                args={
                    'M': 4096,
                    'mode': 'fwd',
                },
            )
        )
    if mode == 'bwd':
        perf_configs.append(
            triton.testing.Benchmark(
                x_names=['N'],
                x_vals=[128 * i for i in range(2, 10)],
                line_arg='provider',
                line_vals=['triton', 'torch', 'naive'],
                line_names=['Triton', 'Torch', 'naive'],
                styles=[('blue', '-'), ('green', '-'), ('red', '-')],
                ylabel='GB/s',
                plot_name="softmax-performance",
                args={
                    'M': 4096,
                    'mode': 'bwd',
                },
            )
        )

    if bench:
        benchmark.run(show_plots=plot, print_data=print_data)

    if test:
        M = 4096
        N = 129
        x = torch.randn(M, N, requires_grad=True, device='cuda')
        y = torch.softmax(x, dim=-1)
        dp = torch.randn_like(x)
        y.backward(dp)
        dx, x.grad = x.grad.clone(), None
        naive_y = naive_softmax(x)
        naive_y.backward(dp)
        naive_dx, x.grad = x.grad.clone(), None
        assert torch.allclose(dx, naive_dx, rtol=1e-3, atol=1e-3)
        assert torch.allclose(y, naive_y, rtol=1e-3, atol=1e-3)
        tt_y = ld_softmax(x)
        tt_y.backward(dp)
        tt_dx, x.grad = x.grad.clone(), None
        assert torch.allclose(y, tt_y, rtol=1e-3, atol=1e-3)
        assert torch.allclose(dx, tt_dx, rtol=1e-3, atol=1e-3)



    
