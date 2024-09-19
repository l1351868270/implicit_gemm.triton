// ./build/cudnn_conv2d_test 4 232 400 256 256 3 3 0 0 1 1 1 1
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cute_implicit_gemm.h"

void print_tensor(half * tensor, int N, int H, int W, int C) {
    printf("[");
    for (int n = 0; n < N; n++) {
        printf("[");
        for (int h = 0; h < H; h++) {
            printf("[");
            for (int w = 0; w < W; w++) {
                printf("[");
                for (int c = 0; c < C; c++) {
                    int offset = n * H * W * C + h * W * C + w * C + c;
                    printf("%.3f, ", __half2float(tensor[offset]));
                }
                printf("],\n");
            }
            printf("],\n");
        }
        printf("],\n");
    }
    printf("]\n");
}

int main(int argc, char ** argv) {
    srand(0);

    int N = 4;
    if (argc >= 2) {
        sscanf(argv[1], "%d", &N);
    }
    int H = 56;
    if (argc >= 3) {
        sscanf(argv[2], "%d", &H);
    }
    int W = 56;
    if (argc >= 4) {
        sscanf(argv[3], "%d", &W);
    }
    int C = 64;
    if (argc >= 5) {
        sscanf(argv[4], "%d", &C);
    }
    int K = 64;
    if (argc >= 6) {
        sscanf(argv[5], "%d", &K);
    }
    int R = 3;
    if (argc >= 7) {
        sscanf(argv[6], "%d", &R);
    }
    int S = 3;
    if (argc >= 8) {
        sscanf(argv[7], "%d", &S);
    }
    int pad_h = 1;
    if (argc >= 9) {
        sscanf(argv[8], "%d", &pad_h);
    }
    int pad_w = 1;
    if (argc >= 10) {
        sscanf(argv[9], "%d", &pad_w);
    }
    int U = 1;
    if (argc >= 11) {
        sscanf(argv[10], "%d", &U);
    }
    int V = 1;
    if (argc >= 12) {
        sscanf(argv[11], "%d", &V);
    }

    int dilation_h = 1;
    if (argc >= 13) {
        sscanf(argv[12], "%d", &dilation_h);
    }

    int dilation_w = 1;
    if (argc >= 14) {
        sscanf(argv[13], "%d", &dilation_w);
    }

    printf("N: %d, H: %d, W: %d, C: %d, K: %d, R: %d, S: %d, pad_h: %d, pad_w: %d, U: %d, V: %d, dilation_h: %d, dilation_w: %d\n", 
            N, H, W, C, K, R, S, pad_h, pad_w, U, V, dilation_h, dilation_w);

    int P = floor((H + 2 * pad_h - dilation_h * (R - 1) - 1) / U + 1);
    int Q = floor((W + 2 * pad_w - dilation_w * (S - 1) - 1) / V + 1);
    // printf("P: %d, Q: %d\n", P, Q);

    thrust::host_vector<half> h_x(N * H * W * C);
    thrust::host_vector<half> h_w(K * R * S * C);
    thrust::host_vector<half> h_y(N * P * Q * K);
    thrust::host_vector<half> h_y1(N * P * Q * K);

    for (int j = 0; j < N * H * W * C; ++j) h_x[j] = static_cast<half>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < K * R * S * C; ++j) h_w[j] = static_cast<half>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < N * P * Q * K; ++j) h_y[j] = static_cast<half>(-1);
    for (int j = 0; j < N * P * Q * K; ++j) h_y1[j] = static_cast<half>(-1);

    thrust::device_vector<half> d_x = h_x;
    thrust::device_vector<half> d_w = h_w;
    thrust::device_vector<half> d_y = h_y;

    int v_M = N * P * Q;
    int v_N = K;
    int v_K = C * R * S;
    printf("GEMM_M: %d, GEMM_N: %d, GEMM_K: %d\n", v_M, v_N, v_K);

    double gflops = (2.0*v_M*v_N*v_K) * 1e-9;
    int bytes = 2 * (N * C * H * W + K * C * R * S + N * K * P * Q);
    double arithmetic_intensity = 2.0*v_M*v_N*v_K / bytes;
    double used_time = 0.0;
    int repeat = 10;

    thrust::fill(d_y.begin(), d_y.end(), 0.0);
    bench::cute_implicit_gemm::cute_implicit_gemm<half>(d_y.data().get(), d_x.data().get(), d_w.data().get(), N, H, W, C, 
                       K, R, S, pad_h, pad_w, U, V, dilation_h, dilation_w);
    thrust::copy(d_y.begin(), d_y.end(), h_y.begin());
    print_tensor(h_y.data(), N, H, W, C);

}