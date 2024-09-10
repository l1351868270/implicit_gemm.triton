import os
import sys
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load

def build_conv2d():
    build_directory = "./build/"
    if not os.path.exists(build_directory):
        os.makedirs(build_directory)
    conda_packages_include = f'{os.path.dirname(sys.executable)}/../lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/include'
    conda_env_include = f'{os.path.dirname(sys.executable)}/../include'

    third_party_include = [f'{os.path.dirname(os.path.abspath(__file__))}/../cutlass/tools/util/include',
                           f'{os.path.dirname(os.path.abspath(__file__))}/../cutlass/include']
    manual_conv2d = load(name='manual_conv2d', 
                         sources=['./cute_implicit_gemm_api.cu'], 
                         build_directory=build_directory,
                         verbose=False,
                         extra_include_paths=[conda_packages_include, conda_env_include] + third_party_include,

                         extra_cuda_cflags=['-O3', '-arch=sm_86',  '-lcublas', '-std=c++20',
                                            "-U__CUDA_NO_HALF_OPERATORS__",
                                            "-U__CUDA_NO_HALF_CONVERSIONS__",
                                            "-U__CUDA_NO_HALF2_OPERATORS__",
                                            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                                            ],
                         with_cuda=True,
                    )

    return manual_conv2d

ref_lib = 'cuBLAS'
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=['x_name', 'N', 'C', 'H', 'W', 'K', 'R', 'S', 'U', 'V', 'pad_h', 'pad_w', 'dila_h', 'dila_w'],
        x_vals=[
            # ('conv1', 1, 3, 224, 224, 64, 7, 7, 2, 2, 3, 3, 1, 1), 
            ('layer1.0.conv1', 1, 64, 56, 56, 64, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer1.0.conv2', 1, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer1.0.conv3', 1, 64, 56, 56, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer1.0.downsample', 1, 64, 56, 56, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer1.1.conv1', 1, 256, 56, 56, 64, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer1.1.conv2', 1, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer1.1.conv3', 1, 64, 56, 56, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer1.2.conv1', 1, 256, 56, 56, 64, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer1.2.conv2', 1, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer1.2.conv3', 1, 64, 56, 56, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer2.0.conv1', 1, 256, 56, 56, 128, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer2.0.conv2', 1, 128, 28, 28, 128, 3, 3, 2, 2, 1, 1, 1, 1),
            ('layer2.0.conv3', 1, 128, 28, 28, 512, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer2.0.downsample', 1, 256, 28, 28, 512, 1, 1, 2, 2, 0, 0, 1, 1),
            ('layer2.1.conv1', 1, 512, 28, 28, 128, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer2.1.conv2', 1, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer2.1.conv3', 1, 128, 28, 28, 512, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer2.2.conv1', 1, 512, 28, 28, 128, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer2.2.conv2', 1, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer2.2.conv3', 1, 128, 28, 28, 512, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer2.3.conv1', 1, 512, 28, 28, 128, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer2.3.conv2', 1, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer2.3.conv3', 1, 128, 28, 28, 512, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.0.conv1', 1, 512, 28, 28, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.0.conv2', 1, 256, 14, 14, 256, 3, 3, 2, 2, 1, 1, 1, 1),
            ('layer3.0.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.0.downsample', 1, 512, 14, 14, 1024, 1, 1, 2, 2, 0, 0, 1, 1),
            ('layer3.1.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.1.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.1.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.2.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.2.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.2.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.3.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.3.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.3.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.4.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.4.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.4.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.5.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.5.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.5.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.6.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.6.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.6.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.7.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.7.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.7.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.8.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.8.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.8.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.9.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.9.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.9.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.10.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.10.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.10.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.11.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.11.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.11.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.12.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.12.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.12.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.13.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.13.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.13.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.14.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.14.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.14.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.15.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.15.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.15.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.15.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.15.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.15.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.16.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.16.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.16.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.17.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.17.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.17.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.18.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.18.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.18.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.19.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.19.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.19.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.20.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.20.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.20.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.21.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.21.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.21.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.22.conv1', 1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer3.22.conv2', 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer3.22.conv3', 1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer4.0.conv1', 1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer4.0.conv2', 1, 512, 7, 7, 512, 3, 3, 2, 2, 1, 1, 1, 1),
            ('layer4.0.conv3', 1, 512, 7, 7, 2048, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer4.0.downsample', 1, 1024, 7, 7, 2048, 1, 1, 2, 2, 0, 0, 1, 1),
            ('layer4.1.conv1', 1, 2048, 7, 7, 512, 1, 1, 1, 1, 0, 0, 1, 1),
            ('layer4.1.conv2', 1, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, 1, 1),
            ('layer4.1.conv3', 1, 512, 7, 7, 2048, 1, 1, 1, 1, 0, 0, 1, 1),
            ],
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=[ref_lib.lower(), 'cutlass_implicit_gemm', 'torch_gemm', 'cute_gemm'],  # Label name for the lines
        line_names=[ref_lib, 'Cutlass_implicit_gemm', 'Torch_gemm', 'Cute_gemm'],  # Line styles
        styles=[("green", "-"), ("blue", "-"), ("red", "-"), ("red", "-")],  # Line color and style
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance-fp16",
        args={},
    ))
@triton.testing.perf_report(configs)
def benchmark(x_name, N, C, H, W, K, R, S, U, V, pad_h, pad_w, dila_h, dila_w, provider):
    P = (H + 2 * pad_h - dila_h * (R - 1) - 1) // U + 1
    Q = (W + 2 * pad_w - dila_w * (S - 1) - 1) // V + 1
    v_M = N * P * Q
    v_N = K
    v_K = C * R * S
    gflops = (2.0*v_M*v_N*v_K) * 1e-9

    x = torch.randn(N, C, H, W).cuda().half()
    w = torch.randn(K, C, R, S).cuda().half()
    y = torch.zeros(N, K, P, Q).cuda().half()
    
    conv2d = torch.nn.Conv2d(C, K, (R, S), stride=(U, V), padding=(pad_h, pad_w), dilation=(dilation_h, dilation_w), bias=False).cuda().half()
    conv2d.weight.data = w
    conv2d = conv2d.to(memory_format=torch.channels_last)
    x = x.to(memory_format=torch.channels_last)
    w = w.to(memory_format=torch.channels_last)
    y = y.to(memory_format=torch.channels_last)

    if provider == ref_lib.lower():
        ms = triton.testing.do_bench(lambda: conv2d(x))

    if provider == 'cutlass_implicit_gemm':
        # y_torch = conv2d(x)
        # manual_conv2d.cutlass_implicit_gemm(y, x, w, pad_h, pad_w, U, V, dilation_h, dilation_w)
        # if not torch.allclose(y_torch, y, atol=1e-2, rtol=1e-2):
        #     print("Torch and triton_implicit_gemm differ")
        #     print(f'torch:{y_torch}, triton:{y}')
        # ms = triton.testing.do_bench(lambda: manual_conv2d.cutlass_implicit_gemm(y, x, w, pad_h, pad_w, U, V, dilation_h, dilation_w))
        ms = float('inf')

    if provider == 'torch_gemm':
        gemm_a = torch.randn(v_M, v_K).cuda().half()
        gemm_b = torch.randn(v_N, v_K).cuda().half().T
        ms = triton.testing.do_bench(lambda: torch.matmul(gemm_a, gemm_b)) 

    if provider == 'cute_gemm':
        gemm_a = torch.randn(v_M, v_K).cuda().half()
        gemm_b = torch.randn(v_N, v_K).cuda().half().T
        gemm_c = torch.zeros(v_M, v_N).cuda().half()
        c_torch = torch.matmul(gemm_a, gemm_b)
        if C == 64 and K == 64 and R == 1 and S == 1 and pad_h == 0 and pad_w == 0 and U == 1 and V == 1 and dila_h == 1 and dila_w == 1:
            manual_conv2d.cute_gemm_f16_s16816fprop_optimized_f16_64x64_32x10_nhwc_align8(gemm_a, gemm_b, gemm_c)
            if not torch.allclose(c_torch, gemm_c, atol=0.5, rtol=0.5):
                print("Torch and cute_gemm differ")
                print(f'torch:{c_torch}, cutlass:{gemm_c}')
            ms = triton.testing.do_bench(lambda: manual_conv2d.cute_gemm_f16_s16816fprop_optimized_f16_64x64_32x10_nhwc_align8(gemm_a, gemm_b, gemm_c)) 
        else:
            manual_conv2d.cute_gemm(gemm_a, gemm_b, gemm_c)
            if not torch.allclose(c_torch, gemm_c, atol=0.5, rtol=0.5):
                print("Torch and cute_gemm differ")
                print(f'torch:{c_torch}, cutlass:{gemm_c}')
            ms = triton.testing.do_bench(lambda: manual_conv2d.cute_gemm(gemm_a, gemm_b, gemm_c))    

    perf = lambda ms: gflops / ms
    return perf(ms)

manual_conv2d = build_conv2d()

if __name__ == '__main__':
    torch.manual_seed(0)
    N = 1; C = 64; H = 56; W = 56; K = 64; R = 1; S = 1; pad_h = 0; pad_w = 0; U = 1; V = 1; dilation_h = 1; dilation_w = 1
    # N = 1; C = 1; H = 3; W = 3; K = 1; R = 3; S = 3; pad_h = 0; pad_w = 0; U = 1; V = 1; dilation_h = 1; dilation_w = 1

    P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) // U + 1
    Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) // V + 1

    x = torch.randn(N, C, H, W).cuda().half()
    w = torch.randn(K, C, R, S).cuda().half()
    y = torch.zeros(N, K, P, Q).cuda().half()
    conv2d = torch.nn.Conv2d(C, K, (R, S), stride=(U, V), padding=(pad_h, pad_w), dilation=(dilation_h, dilation_w), bias=False).cuda().half()
    conv2d.weight.data = w
    y1 = conv2d(x)

    w = w.to(memory_format=torch.channels_last)
    x = x.to(memory_format=torch.channels_last)
    y = y.to(memory_format=torch.channels_last)
    conv2d = conv2d.to(memory_format=torch.channels_last)
    conv2d.weight.data = w
    y2 = conv2d(x)

    GEMM_M = N * P * Q
    GEMM_N = K
    GEMM_K = C * R * S

    gemm_a = torch.randn(GEMM_M, GEMM_K).cuda().half()
    gemm_b = torch.randn(GEMM_N, GEMM_K).cuda().half().T
    gemm_c = torch.zeros(GEMM_M, GEMM_N).cuda().half()
    manual_conv2d.cute_gemm(gemm_a, gemm_b, gemm_c)
    c_torch = torch.matmul(gemm_a, gemm_b)
    if torch.allclose(c_torch, gemm_c, atol=0.5, rtol=0.5):
        print("Torch and cute_gemm match")
    else:
        print("Torch and cute_gemm differ")
        print(f'torch:{c_torch}, cutlass:{gemm_c}')
    gemm_c.fill_(0.0)
    manual_conv2d.cute_gemm_f16_s16816fprop_optimized_f16_64x64_32x10_nhwc_align8(gemm_a, gemm_b, gemm_c)
    if torch.allclose(c_torch, gemm_c, atol=0.5, rtol=0.5):
        print("Torch and cute_gemm_f16_s16816fprop_optimized_f16_64x64_32x10_nhwc_align8 match")
    else:
        print("Torch and cute_gemm_f16_s16816fprop_optimized_f16_64x64_32x10_nhwc_align8 differ")
        print(f'torch:{c_torch}, cutlass:{gemm_c}')

    

    # manual_conv2d.cutlass_implicit_gemm(y, x, w, pad_h, pad_w, U, V, dilation_h, dilation_w)
    # if torch.allclose(y1, y, atol=1e-2, rtol=1e-2):
    #     print("Torch and cutlass_implicit_gemm match")
    # else:
    #     print("Torch and cutlass_implicit_gemm differ")
    #     print(f'torch: shape:{y1.shape}, stride:{y1.stride()}, {y1}')
    #     print(f'cutlass_implicit_gemm: shape:{y.shape}, stride:{y.stride()}, {y}')

    benchmark.run(show_plots=False, print_data=True) 