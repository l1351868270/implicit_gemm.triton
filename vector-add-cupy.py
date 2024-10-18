# Adapted from https://github.com/triton-lang/triton/blob/main/python/tutorials/01-vector-add.py

import cupy as cp

import triton
import triton.language as tl

import numpy as np

import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({'BLOCK_SIZE': 1024}), triton.Config({'BLOCK_SIZE': 2048})],
    key=['n_elements'],
)
@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


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


def add(x: LSLndarray, y: LSLndarray, output: LSLndarray, n_elements):
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements,)
    return output

if __name__ == '__main__':
    np.random.seed(0)
    M, N, K = 128, 128, 128
    size = 98432

    h_a = np.random.randn(size).astype(np.float32)
    h_b = np.random.randn(size).astype(np.float32)
    h_c = np.empty(size).astype(np.float32)

    cp.random.seed(0)
    
    d_a = cp.asarray(h_a)
    d_b = cp.asarray(h_b)
    d_c = cp.asarray(h_c)
    tl_a = LSLndarray(d_a)
    tl_b = LSLndarray(d_b)
    tl_c = LSLndarray(d_c)

    add(tl_a, tl_b, tl_c, size)
    print(tl_c.cupy_data)

    h_d = h_a + h_b
    if np.allclose(tl_c.numpy(), h_d, rtol=1e-6, atol=1e-6):
        print(f'Triton and numpy match!')
    else:
        print(f'Triton and numpy mismatch!\n'
              f'triton: {tl_c.cupy_data}\n'
              f'numpy: {h_d}')
