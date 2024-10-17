# Adapted from https://github.com/triton-lang/triton/blob/main/python/tutorials/01-vector-add.py
# and https://github.com/inducer/pycuda/blob/main/pycuda/gpuarray.py
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import triton
import triton.language as tl

import numpy as np

import triton
import triton.language as tl

# @triton.autotune(
#     configs=[triton.Config({'BLOCK_SIZE': 1024}), triton.Config({'BLOCK_SIZE': 2048})],
#     key=['n_elements'],
# )
@triton.autotune(
    configs=[triton.Config({'BLOCK_SIZE': 1024})],
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


# %%
# Let's also declare a helper function to (1) allocate the `z` tensor
# and (2) enqueue the above kernel with appropriate grid/block sizes:

class LSLGPUArray(gpuarray.GPUArray):
    def __init__(self, *args, **kwargs):
        super(LSLGPUArray, self).__init__(*args, **kwargs)

    def data_ptr(self):
        return self.__cuda_array_interface__['data'][0]

def to_gpu(ary, allocator=cuda.mem_alloc):
    """converts a numpy array to a GPUArray"""
    result = LSLGPUArray(ary.shape, ary.dtype, allocator, strides=gpuarray._compact_strides(ary))
    result.set(ary)
    return result

def add(x, y, output, n_elements):
    # We need to preallocate the output.
    # output = torch.empty_like(x)
    # assert x.is_cuda and y.is_cuda and output.is_cuda
    # n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output

if __name__ == '__main__':
    M, N, K = 128, 128, 128
    # torch.manual_seed(0)
    size = 98432

    h_a = np.random.randn(size).astype(np.float32)
    h_b = np.random.randn(size).astype(np.float32)
    h_c = np.empty(size).astype(np.float32)

    tl_a = to_gpu(h_a)
    tl_b = to_gpu(h_b)
    tl_c = to_gpu(h_c)
    output_triton = add(tl_a, tl_b, tl_c, size)

    h_d = h_a + h_b
    if np.allclose(tl_c.get(), h_d, rtol=1e-6, atol=1e-6):
        print(f'Triton and numpy match!')
    else:
        print(f'Triton and numpy mismatch!\n'
              f'triton: {tl_c.get()}\n'
              f'numpy: {h_d}')
