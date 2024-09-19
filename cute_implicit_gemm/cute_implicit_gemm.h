#pragma once

#include <cute/tensor.hpp>

namespace bench {
namespace cute_implicit_gemm {

template <class T, class CtaTiler, 
          class GEMMYGLayout, class GEMMXGLayout, class GEMMWGLayout,
          class YGLayout, class XGLayout, class WGLayout, 
          class YSLayout, class XSLayout, class WSLayout,
          class TiledCopyY, class TiledCopyX, class TiledCopyW, class TiledMma>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
cute_implicit_gemm_device(T * y_ptr, T * x_ptr, T * w_ptr, 
                          CtaTiler cta_tiler, 
                          GEMMYGLayout GEMM_y_glayout, GEMMXGLayout GEMM_x_glayout, GEMMWGLayout GEMM_w_glayout,
                          YGLayout y_glayout, XGLayout x_glayout, WGLayout w_glayout,
                          YSLayout y_slayout, XSLayout x_slayout, WSLayout w_slayout,
                          TiledCopyY copy_y, TiledCopyX copy_x, TiledCopyW copy_w, TiledMma mma,
                          int GEMM_M, int GEMM_N, int GEMM_K, int P, int Q,
                          int N, int H, int W, int C, 
                          int K, int R, int S, int pad_h, int pad_w, int U, int V, int dila_h, int dila_w) {
    using namespace cute;

    CUTE_STATIC_ASSERT_V(size(copy_x) == size(mma));
    CUTE_STATIC_ASSERT_V(size(copy_w) == size(mma));

    CUTE_STATIC_ASSERT_V(rank(y_glayout) == Int<2>{}); // (N * P * Q, K)         <-> (GEMM_M, GEMM_N)
    CUTE_STATIC_ASSERT_V(rank(x_glayout) == Int<2>{}); // (N * P * Q, C * R * S) <-> (GEMM_M, GEMM_K)
    CUTE_STATIC_ASSERT_V(rank(w_glayout) == Int<2>{}); // (K        , C * R * S) <-> (GEMM_N, GEMM_K)
    CUTE_STATIC_ASSERT_V(rank(x_slayout) == Int<2>{});
    CUTE_STATIC_ASSERT_V(rank(w_slayout) == Int<2>{});

    // CUTE_STATIC_ASSERT_V(size<0>(y_glayout) == size<0>(x_glayout)); // N * P * Q <-> GEMM_M
    // CUTE_STATIC_ASSERT_V(size<1>(y_glayout) == size<0>(w_glayout)); // K         <-> GEMM_N
    // CUTE_STATIC_ASSERT_V(size<1>(x_glayout) == size<1>(w_glayout)); // C * R * S <-> GEMM_K

    CUTE_STATIC_ASSERT_V(size<0>(XSLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<1>(YSLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(WSLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(YSLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(XSLayout{}) == size<2>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(WSLayout{}) == size<2>(cta_tiler));  // BLK_K

    Tensor GEMM_mY = make_tensor(make_gmem_ptr(y_ptr), GEMM_y_glayout);  // ((N, P, Q), K)
    Tensor GEMM_mX = make_tensor(make_gmem_ptr(x_ptr), GEMM_x_glayout);  // ((N, P, Q), (R, S, C))
    Tensor GEMM_mW = make_tensor(make_gmem_ptr(w_ptr), GEMM_w_glayout);  // (K        , (R, S, C))

    Tensor mY = make_tensor(make_gmem_ptr(y_ptr), y_glayout);  // ((N, P, Q), K)
    Tensor mX = make_tensor(make_gmem_ptr(x_ptr), x_glayout);  // ((N, P, Q), (R, S, C))
    Tensor mW = make_tensor(make_gmem_ptr(w_ptr), w_glayout);  // (K        , (R, S, C))

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
    // Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
    Tensor GEMM_gX = local_tile(GEMM_mX, select<0,2>(cta_tiler), make_coord(blockIdx.x, _));  // (BLK_M,BLK_K,k)
    Tensor GEMM_gW = local_tile(GEMM_mW, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
    Tensor GEMM_gY = local_tile(GEMM_mY, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

    Tensor gX = local_tile(mX, select<0,2>(cta_tiler), make_coord(blockIdx.x, _));  // (BLK_M,BLK_K,k)
    Tensor gW = local_tile(mW, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
    Tensor gY = local_tile(mY, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

    // int GEMM_M = size<0>(y_glayout);
    // int GEMM_N = size<1>(y_glayout);
    // int GEMM_K = size<1>(x_glayout);
    int BLOCK_SIZE_M = size<0>(cta_tiler);
    int BLOCK_SIZE_N = size<1>(cta_tiler);
    int BLOCK_SIZE_K = size<2>(cta_tiler);


    int tid = threadIdx.x;
    int bid_m = blockIdx.x;
    int bid_n = blockIdx.y;

    int lid = tid % 32;
    int wid = tid / 32;

    // global memory -> shared memory
    // Shared memory buffers
    __shared__ T smemX[cosize_v<XSLayout>];
    __shared__ T smemW[cosize_v<WSLayout>];
    __shared__ T smemY[cosize_v<YSLayout>];
    Tensor sX = make_tensor(make_smem_ptr(smemX), x_slayout);            // (BLK_M,BLK_K,PIPE)
    Tensor sW = make_tensor(make_smem_ptr(smemW), w_slayout);            // (BLK_N,BLK_K,PIPE)
    Tensor sY = make_tensor(make_smem_ptr(smemY), y_slayout);            // (BLK_N,BLK_K,PIPE)

    ThrCopy thr_copy_x = copy_x.get_slice(threadIdx.x);
    Tensor GEMM_tAgX = thr_copy_x.partition_S(GEMM_gX);                  // (CPY,CPY_M,CPY_K,k)
    Tensor tAgX = thr_copy_x.partition_S(gX);                            // (CPY,CPY_M,CPY_K,k)
    Tensor tAsX = thr_copy_x.partition_D(sX);                            // (CPY,CPY_M,CPY_K,PIPE)

    ThrCopy thr_copy_w = copy_w.get_slice(threadIdx.x);
    Tensor GEMM_tBgW = thr_copy_w.partition_S(GEMM_gW); 
    Tensor tBgW = thr_copy_w.partition_S(gW);                            // (CPY,CPY_N,CPY_K,k)
    Tensor tBsW = thr_copy_w.partition_D(sW);                            // (CPY,CPY_N,CPY_K,PIPE)

    // CUTE_STATIC_ASSERT_V(size<1>(tAgX) == size<1>(tAsX));                // CPY_M
    // CUTE_STATIC_ASSERT_V(size<2>(tAgX) == size<2>(tAsX));                // CPY_K
    // CUTE_STATIC_ASSERT_V(size<1>(tBgW) == size<1>(tBsW));                // CPY_N
    // CUTE_STATIC_ASSERT_V(size<2>(tBgW) == size<2>(tBsW));   

    //
    // Define A/B partitioning and C accumulators
    //

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsX = thr_mma.partition_A(sX);                               // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsW = thr_mma.partition_B(sW);                               // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCsY = thr_mma.partition_C(sY);                               // (MMA,MMA_M,MMA_N)
    Tensor tCgY = thr_mma.partition_C(gY);                               // (MMA,MMA_M,MMA_N)
    Tensor GEMM_tCgY = thr_mma.partition_C(GEMM_gY);                               // (MMA,MMA_M,MMA_N)

    // Allocate registers for pipelining
    Tensor tCrX = thr_mma.make_fragment_A(tCsX(_,_,_));                // (MMA,MMA_M,MMA_K)
    Tensor tCrW = thr_mma.make_fragment_B(tCsW(_,_,_));                // (MMA,MMA_N,MMA_K)
    Tensor tCrY = thr_mma.make_fragment_C(tCsY(_,_,_));                // (MMA,MMA_N,MMA_K)
    // Allocate the accumulators -- same size as the projected data
    // Tensor tCrY = thr_mma.make_fragment_C(tCgY);                         // (MMA,MMA_M,MMA_N)

    Tensor GEMM_tCrX = thr_mma.make_fragment_A(tCsX(_,_,_));                // (MMA,MMA_M,MMA_K)
    Tensor GEMM_tCrW = thr_mma.make_fragment_B(tCsW(_,_,_));                // (MMA,MMA_N,MMA_K)
    Tensor GEMM_tCrY = thr_mma.make_fragment_C(GEMM_tCgY);                         // (MMA,MMA_M,MMA_N)

    clear(GEMM_tCrY);
    clear(tCrY);

#if 0
    if (thread(127)) {
        print("  mX : "); print(  mX); print("\n");
        print("  gX : "); print(  gX); print("\n");
        print("  GEMM_mX : "); print(  GEMM_mX); print("\n");
        print("  GEMM_gX : "); print(  GEMM_gX); print("\n");

        print("  gX : "); print(  gX(_,_,0)); print("\n");
        print("  rank(gX) : "); print(  rank(gX(_,_,0))); print("\n");
        // print_latex(layout(gX(_,_,0)));
    }
#endif

#if 0
    if (thread(127)) {
        print("  GEMM_mX : "); print(  GEMM_mX); print("\n");
        print("  GEMM_gX : "); print(  GEMM_gX); print("\n");
        print("  mX : "); print(  mX); print("\n");
        print("  gX : "); print(  gX); print("\n");
        print("  sX : "); print(  sX); print("\n");
        print("GEMM_tAgX : "); print(GEMM_tAgX); print("\n");
        print("tAgX : "); print(tAgX); print("\n");
        print("tAsX : "); print(tAsX); print("\n");
        
        print("  GEMM_mW : "); print(  GEMM_mW); print("\n");
        print("  GEMM_gW : "); print(  GEMM_gW); print("\n");
        print("  mW : "); print(  mW); print("\n");
        print("  gW : "); print(  gW); print("\n");
        print("  sW : "); print(  sW); print("\n");
        print("GEMM_tBgW : "); print(GEMM_tBgW); print("\n");
        print("tBgW : "); print(tBgW); print("\n");

        print("  GEMM_mY : "); print(  GEMM_mY); print("\n");
        print("  GEMM_gY : "); print(  GEMM_gY); print("\n");
        print("  mY : "); print(  mY); print("\n");
        print("  gY : "); print(  gY); print("\n");
        print("  sY : "); print(  sY); print("\n");
        print("tCsX : "); print(tCsX); print("\n");
        print("tCsW : "); print(tCsW); print("\n");
        print("tCsY : "); print(tCsY); print("\n");
        print("tCgY : "); print(tCgY); print("\n");
        print("GEMM_tCgY : "); print(GEMM_tCgY); print("\n");

        print("tCrX : "); print(tCrX); print("\n");
        print("tCrW : "); print(tCrW); print("\n");
        print("tCrY : "); print(tCrY); print("\n");
        print("GEMM_tCrX : "); print(GEMM_tCrX); print("\n");
        print("GEMM_tCrW : "); print(GEMM_tCrW); print("\n");
        print("GEMM_tCrY : "); print(GEMM_tCrY); print("\n");

        print("K_BLOCK_MAX : "); print(size<2>(tCrX));print("\n");
        print("k_tile_count : "); print(size<3>(tAgX));print("\n");

        print("K_BLOCK_MAX : "); print(size<2>(GEMM_tCrX));print("\n");
        print("k_tile_count : "); print(size<3>(GEMM_tAgX));print("\n");
    }
#endif

    // Total count of tiles
    int k_tile_count = size<3>(GEMM_tAgX);
    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(GEMM_tCrX);

    int gemm_i = (bid_m * BLOCK_SIZE_M) % GEMM_M;

    auto GEMM_mX_layout = GEMM_mX.layout();
    auto GEMM_gX_layout = GEMM_gX.layout();
    // https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/copy_atom.hpp#L351C3-L351C14
    auto GEMM_tAgX_layout = copy_x.tidfrg_S(GEMM_gX_layout)(_, _, repeat<rank_v<decltype(GEMM_gX)>>(_));
    gemm_i += get<1>(idx2crd(tid, get<0>(shape(GEMM_tAgX_layout))));
    auto x_npq_crd = idx2crd(gemm_i, get<0>(shape(x_glayout)), make_stride(P * Q, Q, 1));

#if 0
    if (thread(127)) {
        printf("GEMM_mX_layout:"); print(GEMM_mX_layout); printf("\n");
        printf("GEMM_gX_layout:"); print(GEMM_gX_layout); printf("\n");
        printf("GEMM_tAgX_layout:"); print(GEMM_tAgX_layout); printf("\n");
    }
#endif

    int gemm_j = (bid_n * BLOCK_SIZE_N) % GEMM_N;
    auto GEMM_mW_layout = GEMM_mW.layout();
    auto GEMM_gW_layout = GEMM_gW.layout();
    auto GEMM_tBgW_layout = copy_w.tidfrg_S(GEMM_gW_layout)(_, _, repeat<rank_v<decltype(GEMM_gW)>>(_));
    gemm_j += get<1>(idx2crd(tid, get<0>(shape(GEMM_tBgW_layout))));
    int k = gemm_j;
#if 0
    if (thread(127)) {
        printf("GEMM_mW_layout:"); print(GEMM_mW_layout); printf("\n");
        printf("GEMM_gW_layout:"); print(GEMM_gW_layout); printf("\n");
        printf("GEMM_tBgW_layout:"); print(GEMM_tBgW_layout); printf("\n");
        printf("idx2crd(tid, get<0>(shape(GEMM_tBgW_layout))):"); print(idx2crd(tid, get<0>(shape(GEMM_tBgW_layout)))); printf("\n");
    }
#endif

#if 1
    int n = gemm_i / (P * Q);
    int npq_residual = gemm_i % (P * Q);
    int p = npq_residual / Q;
    int q = npq_residual % Q;

    // assert(n == get<0>(x_npq_crd));
    // assert(p == get<1>(x_npq_crd));
    // assert(q == get<2>(x_npq_crd));
#endif

    CUTE_UNROLL
    for (int k_tile = 0; k_tile < k_tile_count; k_tile++) {
        // Copy gmem to smem before computing gemm on each k-pipe
        int gemm_k = (k_tile * BLOCK_SIZE_K) % GEMM_K;
        gemm_k += get<0, 0>(GEMM_tAgX_layout)(get<0>(idx2crd(tid, get<0>(shape(GEMM_tAgX_layout)))));

        auto x_crs_crd = idx2crd(gemm_k, get<1>(shape(x_glayout)), make_stride(S * C, C, 1));
        auto x_gcrd = make_coord(x_npq_crd, x_crs_crd);
        int r = gemm_k / (S * C);
        int crs_residual = gemm_k % (S * C);
        int s = crs_residual / C;
        int c = crs_residual % C;

        int h = get<1>(x_npq_crd) * U + get<0>(x_crs_crd) * dila_h - pad_h;
        int w = get<2>(x_npq_crd) * V + get<1>(x_crs_crd) * dila_w - pad_w;
        // auto goffs_x = x_glayout(x_gcrd);
        // goffs_x = goffs_x - pad_h * W * C - pad_w * C;
        int goffs_x = n * H * W * C + h * W * C + w * C + c;

        auto w_gcrd = make_coord(k, x_crs_crd);
        auto goffs_w = w_glayout(w_gcrd);

        int soffs_x = (gemm_i % BLOCK_SIZE_M) * BLOCK_SIZE_K + (gemm_k % BLOCK_SIZE_K); // shared memory offset
        int soffs_w = (gemm_j % BLOCK_SIZE_N) * BLOCK_SIZE_K + (gemm_k % BLOCK_SIZE_K); // shared memory offset
        // mX({{n, p, q}, {c, r, s}});
        if (h >= 0 && h < H && w >= 0 && w < W) {
            *reinterpret_cast<uint4*>(smemX + soffs_x) = *reinterpret_cast<uint4*>(x_ptr + goffs_x);
        } else {
            *reinterpret_cast<uint4*>(smemX + soffs_x) = make_uint4(0, 0, 0, 0);
        }
        *reinterpret_cast<uint4*>(smemW + soffs_w) = *reinterpret_cast<uint4*>(w_ptr + goffs_w);

        __syncthreads();

#if 0
    if (thread(127)) {
        printf("x_glayout: "); print(x_glayout); printf("\n");
    }
#endif
        #if 1

            
            assert(r == get<0>(x_crs_crd));
            assert(s == get<1>(x_crs_crd));
            assert(c == get<2>(x_crs_crd));

            if (c != get<2>(x_crs_crd) || r != get<0>(x_crs_crd) || s != get<1>(x_crs_crd)) {
                printf("tid:%d, gemm_k:%d, r:%d, s:%d, c:%d, x_crs_crd:(%d, %d, %d)\n", tid, gemm_k, r, s, c, get<0>(x_crs_crd), get<1>(x_crs_crd), get<2>(x_crs_crd));
            }

            // int h = p * U + r * dila_h - pad_h;
            // int w = q * V + s * dila_w - pad_w;
            int goffs_x_ = n * H * W * C + h * W * C + w * C + c; // global memory offset
            int goffs_w_ = k * R * S * C + r * S * C + s * C + c; // global memory offset

            // assert(goffs_x == goffs_x_);
            // assert(goffs_w == goffs_w_);
            if (goffs_x != goffs_x_) {
                printf("goffs_x:%d, goffs_x_:%d\n", goffs_x, goffs_x_);
            }
            if (goffs_w != goffs_w_) {
                printf("goffs_w:%d, goffs_w_:%d\n", goffs_w, goffs_w_);
            }
        #endif

#if 0
    if (thread(127)) {
        __syncthreads();
        cute::print_tensor(sW);
    }
#endif
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            // Load A, B shmem->regs for k_block+1
            copy(tCsX(_,_,k_block), tCrX(_,_,k_block));
            copy(tCsW(_,_,k_block), tCrW(_,_,k_block));
            __syncthreads();
            // Thread-level register gemm for k_block
            gemm(mma, tCrX(_,_,k_block), tCrW(_,_,k_block), tCrY);
        }
    }
    __syncthreads();
    copy(tCrY, tCsY);
    __syncthreads();
    auto GEMM_mY_layout = GEMM_mY.layout();
    auto GEMM_gY_layout = GEMM_gY.layout();
    auto GEMM_tYgY_layout = copy_y.tidfrg_S(GEMM_gY_layout)(_, _, repeat<rank_v<decltype(GEMM_gY)>>(_));

#if 0
    if (thread(127)) {
        printf("GEMM_mY_layout:"); print(GEMM_mY_layout); printf("\n");
        printf("GEMM_gY_layout:"); print(GEMM_gY_layout); printf("\n");
        printf("GEMM_tYgY_layout:"); print(GEMM_tYgY_layout); printf("\n");
    }
#endif

    int s2g_gemm_i = gemm_i;
    int s2g_gemm_j = (bid_n * BLOCK_SIZE_N) % GEMM_N;
    s2g_gemm_j += get<0>(idx2crd(tid, get<0>(shape(GEMM_tYgY_layout)))) * 8;

    int soffs_y = (s2g_gemm_i % BLOCK_SIZE_M) * BLOCK_SIZE_N + (s2g_gemm_j % BLOCK_SIZE_N); // shared memory offset
    int goffs_y = s2g_gemm_i * GEMM_N + s2g_gemm_j; // global memory offset
    // printf("tid:%d,s2g_gemm_i:%d, s2g_gemm_j:%d, soffs_y:%d, goffs_y:%d\n", tid, s2g_gemm_i, s2g_gemm_j, soffs_y, goffs_y);
    __syncthreads();
    *reinterpret_cast<uint4*>(y_ptr + goffs_y) = *reinterpret_cast<uint4*>(smemY + soffs_y);

#if 0
    if (thread(127)) {
        cute::print_tensor(sY);
    }
#endif
    
}

template <typename T>
void cute_implicit_gemm(T * y, T * x, T * w, int N, int H, int W, int C, 
                            int K, int R, int S, int pad_h, int pad_w, int U, int V, int dilation_h, int dilation_w)  {
    using namespace cute;

    int P = floor((H + 2 * pad_h - dilation_h * (R - 1) - 1) / U + 1);
    int Q = floor((W + 2 * pad_w - dilation_w * (S - 1) - 1) / V + 1);

    // auto w_layout = make_layout(make_shape(K, make_shape(C, S, R)),
    //                             make_stride(C * R * S, make_stride(Int<1>{}, C , C * S)));
    // auto x_layout = make_layout(make_shape(make_shape(Q, P, N), make_shape(C, S, R)),
    //                             make_stride(make_stride(C, C * S, C * S * R), make_stride(Int<1>{}, C, C * S)));
    // auto y_layout = make_layout(make_shape(K, make_shape(Q, P, N)),
    //                             make_stride(Int<1>{}, make_stride(K, K * Q, K * Q * P)));
    auto x_glayout = make_layout(make_shape(make_shape(N, P, Q),               make_shape(R, S, C)),
                                 make_stride(make_stride(C * H * W, U * C * W, V * C), make_stride(U * C * W, V * C, Int<1>{})));
    auto w_glayout = make_layout(make_shape( K,         make_shape(R, S, C)),
                                 make_stride(C * R * S, make_stride(C * S, C, Int<1>{})));
    auto y_glayout = make_layout(make_shape(make_shape(N, P, Q), K),
                                 make_stride(make_stride(P * Q * K,  Q * K, K), Int<1>{}));

    int GEMM_M = N * P * Q;
    int GEMM_N = K;
    int GEMM_K = C * R * S;

    auto GEMM_y_glayout = make_layout(make_shape(GEMM_M, GEMM_N), make_stride(GEMM_N, Int<1>{}));
    auto GEMM_x_glayout = make_layout(make_shape(GEMM_M, GEMM_K), make_stride(GEMM_K, Int<1>{}));
    auto GEMM_w_glayout = make_layout(make_shape(GEMM_K, GEMM_N), make_stride(Int<1>{}, GEMM_K));
    

    auto bM = Int<32>{};
    auto bN = Int<32>{};
    auto bK = Int<32>{};

    auto cta_tiler = make_shape(bM, bN, bK);

    auto y_slayout = make_layout(make_shape(bM, bN), make_stride(bN, Int<1>{})); 

    using SmemLayoutAtom = decltype(
            make_layout(make_shape(Int<8>{}, Int<bK>{}),
                        make_stride(Int<bK>{}, Int<1>{})));
    using SmemLayoutX = decltype(tile_to_shape(SmemLayoutAtom{},
                                                make_shape(Int<bM>{}, Int<bK>{})));
    using SmemLayoutW = decltype(tile_to_shape(SmemLayoutAtom{},
                                                make_shape(Int<bN>{}, Int<bK>{})));
    using SmemLayoutY = decltype(tile_to_shape(SmemLayoutAtom{},
                                                make_shape(Int<bM>{}, Int<bN>{})));

    SmemLayoutX x_slayout;
    SmemLayoutW w_slayout;

    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
    using G2SCopyA =
        decltype(make_tiled_copy(g2s_copy_atom{},
                                    make_layout(make_shape(Int<32>{}, Int<4>{}),
                                                make_stride(Int<4>{}, Int<1>{})),
                                    make_layout(make_shape(Int<1>{}, Int<8>{}))));
    G2SCopyA copyX;
    G2SCopyA copyW;
    G2SCopyA copyY;

    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;
    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 32;
    static constexpr int kMmaPN = 32;
    static constexpr int kMmaPK = 32;
    using thr_layout = decltype(make_layout(make_shape(Int<kMmaEURepeatM>{}, 
                                                        Int<kMmaEURepeatN>{}, 
                                                        Int<kMmaEURepeatK>{})));
    using permutations = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using MMA = decltype(make_tiled_mma(mma_atom{}, thr_layout{}, permutations{}));
    MMA mmaC;

#if 0
    printf("y_glayout: "); print(y_glayout);printf("\n");
    printf("x_glayout: "); print(x_glayout);printf("\n");
    printf("w_glayout: "); print(w_glayout);printf("\n");
    printf("x_slayout: "); print(x_slayout);printf("\n");
    printf("w_slayout: "); print(w_slayout);printf("\n");
#endif

    dim3 dimBlock(size(mmaC));
    dim3 dimGrid(size(ceil_div(GEMM_M, bM)),
                 size(ceil_div(GEMM_N, bN)));
    int smem_size = cosize_v<SmemLayoutAtom> + cosize_v<SmemLayoutAtom> + cosize_v<decltype(y_slayout)>;

    cudaFuncSetAttribute(cute_implicit_gemm_device<T, decltype(cta_tiler), 
                                                  decltype(GEMM_y_glayout), decltype(GEMM_x_glayout), decltype(GEMM_w_glayout), 
                                                  decltype(y_glayout), decltype(x_glayout), decltype(w_glayout), 
                                                  decltype(y_slayout), decltype(x_slayout), decltype(w_slayout), 
                                                  decltype(copyY), decltype(copyX), decltype(copyW), decltype(mmaC)>, 
                      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cute_implicit_gemm_device<<<dimGrid, dimBlock, smem_size>>>
                                (y, x, w,  
                                 cta_tiler, 
                                 GEMM_y_glayout, GEMM_x_glayout, GEMM_w_glayout,
                                 y_glayout, x_glayout, w_glayout, 
                                 y_slayout, x_slayout, w_slayout,
                                 copyY, copyX, copyW, mmaC,
                                 GEMM_M, GEMM_N, GEMM_K, P, Q,
                                 N, H, W, C, K, R, S, pad_h, pad_w, U, V, dilation_h, dilation_w);
}
} // namespace cute_implicit_gemm
} // namespace bench