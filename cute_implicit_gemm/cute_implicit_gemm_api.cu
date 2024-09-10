#include <torch/extension.h>
#include "cute_gemm.h"
#include "cute_gemm_f16_s16816fprop_optimized_f16_64x64_32x10_nhwc_align8.h"
// double py_cutlass_implicit_gemm(torch::Tensor y, torch::Tensor x, torch::Tensor w, int pad_h, int pad_w, int U, int V, int dila_h, int dila_w) {
//     c10::Half *y_ptr = y.data_ptr<c10::Half>();
//     c10::Half *x_ptr = x.data_ptr<c10::Half>();
//     c10::Half *w_ptr = w.data_ptr<c10::Half>();

//     half* y_hf = reinterpret_cast<half*>(y_ptr);
//     half* x_hf = reinterpret_cast<half*>(x_ptr);
//     half* w_hf = reinterpret_cast<half*>(w_ptr);

//     auto N = x.size(0);
//     auto C = x.size(1);
//     auto H = x.size(2);
//     auto W = x.size(3);
    
//     auto K = w.size(0);
//     auto R = w.size(2);
//     auto S = w.size(3);

//     int P = floor((H + 2 * pad_h - dila_h * (R - 1) - 1) / U + 1);
//     int Q = floor((W + 2 * pad_w - dila_w * (S - 1) - 1) / V + 1);

//     if (R == 224 && S == 224) {
//         return bench::cutlass_implicit_gemm_224x224::cutlass_implicit_gemm<half>(y_hf, x_hf, w_hf, N, C, H, W, K, P, Q, R, S, U, V, pad_h, pad_w, dila_h, dila_w, "zeros");
//     } else if (R == 56 && S == 56) {
//         return bench::cutlass_implicit_gemm_56x56::cutlass_implicit_gemm<half>(y_hf, x_hf, w_hf, N, C, H, W, K, P, Q, R, S, U, V, pad_h, pad_w, dila_h, dila_w, "zeros");
//     } else {
//         return bench::cutlass_implicit_gemm_28x28::cutlass_implicit_gemm<half>(y_hf, x_hf, w_hf, N, C, H, W, K, P, Q, R, S, U, V, pad_h, pad_w, dila_h, dila_w, "zeros");

//     }
    
// }


void py_cute_gemm(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    c10::Half *a_ptr = a.data_ptr<c10::Half>();
    c10::Half *b_ptr = b.data_ptr<c10::Half>();
    c10::Half *c_ptr = c.data_ptr<c10::Half>();

    half* a_hf = reinterpret_cast<half*>(a_ptr);
    half* b_hf = reinterpret_cast<half*>(b_ptr);
    half* c_hf = reinterpret_cast<half*>(c_ptr);
    
    int M = a.size(0);
    int N = b.size(1);
    int K = a.size(1);
    
    int stride_am = K;
    int stride_ak = 1;
    int stride_bk = 1;
    int stride_bn = K;
    int stride_cm = N;
    int stride_cn= 1;
    bench::cute_gemm::gemm_tn<half, half, half>(a_hf, b_hf, c_hf, 
                                                M, N, K, 
                                                stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn);
}

void py_cute_gemm_f16_s16816fprop_optimized_f16_64x64_32x10_nhwc_align8(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    c10::Half *a_ptr = a.data_ptr<c10::Half>();
    c10::Half *b_ptr = b.data_ptr<c10::Half>();
    c10::Half *c_ptr = c.data_ptr<c10::Half>();

    half* a_hf = reinterpret_cast<half*>(a_ptr);
    half* b_hf = reinterpret_cast<half*>(b_ptr);
    half* c_hf = reinterpret_cast<half*>(c_ptr);
    
    int M = a.size(0);
    int N = b.size(1);
    int K = a.size(1);
    
    int stride_am = K;
    int stride_ak = 1;
    int stride_bk = 1;
    int stride_bn = K;
    int stride_cm = N;
    int stride_cn= 1;
    bench::cute_gemm_f16_s16816fprop_optimized_f16_64x64_32x10_nhwc_align8::gemm_tn<half, half, half>(a_hf, b_hf, c_hf, 
                                                M, N, K, 
                                                stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("cute_gemm", torch::wrap_pybind_function(py_cute_gemm), "cute_gemm");
m.def("cute_gemm_f16_s16816fprop_optimized_f16_64x64_32x10_nhwc_align8", torch::wrap_pybind_function(py_cute_gemm_f16_s16816fprop_optimized_f16_64x64_32x10_nhwc_align8), "cute_gemm_f16_s16816fprop_optimized_f16_64x64_32x10_nhwc_align8");
}