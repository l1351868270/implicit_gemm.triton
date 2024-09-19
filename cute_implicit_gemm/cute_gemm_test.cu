// ncu -f --set full --call-stack -o build/cute_gemm_test build/cute_gemm_test
// ncu --csv --log-file build/a.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum build/cute_gemm_test
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cute_gemm.h"
#include "cute_gemm_f16_s16816fprop_optimized_f16_64x64_32x10_nhwc_align8.h"

float frand() {
    return (float)rand() / (float)RAND_MAX;
}

void generate_tensor(float * tensor, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        tensor[i] = frand();
    }
}

void generate_range_tensor(float * tensor, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        tensor[i] = i;
    }
}

void print_tensor(half * tensor, int M, int N) {
    printf("[");
    for (int i = 0; i < M; i++) {
        printf("[");
        for (int j = 0; j < N; j++) {
            int offset = i * N + j;
            printf("%.5f, ", __half2float(tensor[offset]));

        }
        printf("],\n");
    }
    printf("]\n");
}

bool check_value(float abs_tol, float rel_tol, half *h_d_c, half *h_c, int m, int n) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float gpu_value = (float)h_d_c[i * n + j];
            float cpu_value = (float)h_c[i * n + j];
            float diff = abs(gpu_value - cpu_value);
            if (diff > max(abs_tol, cpu_value * rel_tol)) {
                std::cout << "blas[" << i << ", " << j << "] = " << gpu_value 
                << ", manual[" << i << ", " << j << "] = " << cpu_value
                << " Abs Diff: " << diff << std::endl;
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    srand(0);
    
    int M = 3316;
    if (argc >= 2) {
        sscanf(argv[1], "%d", &M);
    }
    int N = 64;
    if (argc >= 3) {
        sscanf(argv[2], "%d", &N);
    }
    int K = 64;
    if (argc >= 4) {
        sscanf(argv[3], "%d", &K);
    }

    std::cout << "M: " << M << ", N: " << N << ", K: " << K << std::endl;

    thrust::host_vector<half> h_A(M * K);
    thrust::host_vector<half> h_B(N * K);
    thrust::host_vector<half> h_C(M * N);
    thrust::host_vector<half> h_C1(M * N);

    for (int j = 0; j < M * K; ++j) h_A[j] = static_cast<half>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < N * K; ++j) h_B[j] = static_cast<half>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < M * N; ++j) h_C[j] = static_cast<half>(-1);
    for (int j = 0; j < M * N; ++j) h_C1[j] = static_cast<half>(-1);

    thrust::device_vector<half> d_A = h_A;
    thrust::device_vector<half> d_B = h_B;
    thrust::device_vector<half> d_C = h_C;
    double gflops = (2.0*M*N*K) * 1e-9;

    int stride_am = K;
    int stride_ak = 1;
    int stride_bk = 1;
    int stride_bn = K;
    int stride_cm = N;
    int stride_cn= 1;
    bench::cute_gemm::gemm_tn<half, half, half>(d_A.data().get(), d_B.data().get(), d_C.data().get(), 
                                             M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn);
    thrust::copy(d_C.begin(), d_C.end(), h_C1.begin());
    // print_tensor(h_C1.data(), M, N);

    bench::cute_gemm_f16_s16816fprop_optimized_f16_64x64_32x10_nhwc_align8::gemm_tn<half, half, half>(d_A.data().get(), d_B.data().get(), d_C.data().get(), 
                                             M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn);
    thrust::copy(d_C.begin(), d_C.end(), h_C1.begin());
    // print_tensor(h_C1.data(), M, N);

    return 0;
}