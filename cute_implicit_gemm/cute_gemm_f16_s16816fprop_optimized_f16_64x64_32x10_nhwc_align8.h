
#pragma once

#include <cute/tensor.hpp>

namespace bench {
namespace cute_gemm_f16_s16816fprop_optimized_f16_64x64_32x10_nhwc_align8 {

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
            TC      * C, CStride dC, CSmemLayout          , TiledMma mma)
{
    using namespace cute;
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

    CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));
    CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));
    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CSmemLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
    // Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
    Tensor gA = local_tile(mA, select<0,2>(cta_tiler), make_coord(blockIdx.x, _));  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

    // Shared memory buffers
    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K,PIPE)

    //
    // Partition the copying of A and B tiles across the threads
    //

    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K,PIPE)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));   

    //
    // PREFETCH
    //

    auto K_PIPE_MAX = size<3>(tAsA);

    // Total count of tiles
    int k_tile_count = size<3>(tAgA);
    // Current tile index in gmem to read from
    int k_tile_next = 0;

    // Start async loads for all pipes but the last
    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
        copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
    }

    //
    // Define A/B partitioning and C accumulators
    //

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

    // Allocate registers for pipelining
    Tensor tCrA = thr_mma.make_fragment_A(tCsA(_,_,_,0));                // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB(_,_,_,0));                // (MMA,MMA_N,MMA_K)
    // Allocate the accumulators -- same size as the projected data
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

    CUTE_STATIC_ASSERT_V((  shape(tCrA) == take<0,3>(shape(tCsA))));     // (MMA,MMA_M,MMA_K)
    CUTE_STATIC_ASSERT_V((  shape(tCrB) == take<0,3>(shape(tCsB))));     // (MMA,MMA_N,MMA_K)
    CUTE_STATIC_ASSERT_V((  shape(tCrC) == take<0,3>(shape(tCgC))));     // (MMA,MMA_M,MMA_N)
    CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCsA)));              // MMA_M
    CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCsB)));              // MMA_N
    CUTE_STATIC_ASSERT_V((size<2>(tCsA) == size<2>(tCsB)));              // MMA_K

    // Clear the accumulators
    clear(tCrC);

#if 0
    if(thread0()) {
        print("  mA : "); print(  mA); print("\n");
        print("  gA : "); print(  gA); print("\n");
        print("  sA : "); print(  sA); print("\n");
        print("tAgA : "); print(tAgA); print("\n");
        print("tAsA : "); print(tAsA); print("\n");
        // print("rank(tAgA): "); print(rank(tAgA)); print("\n");
        // print("depth(tAgA): "); print(depth(tAgA)); print("\n");
        // print("shape(tAgA): "); print(shape(tAgA)); print("\n");
        // print("stride(tAgA): "); print(stride(tAgA)); print("\n");
        // print("size(tAgA): "); print(size(tAgA)); print("\n");
    }
#endif

#if 0
    if(thread0()) {
        print("  mB : "); print(  mB); print("\n");
        print("  gB : "); print(  gB); print("\n");
        print("  sB : "); print(  sB); print("\n");
        print("tBgB : "); print(tBgB); print("\n");
        print("tBsB : "); print(tBsB); print("\n");
    }
#endif

#if 0
    if(thread0()) {
        print("  mC : "); print(  mC); print("\n");
        print("  gC : "); print(  gC); print("\n");
        print("tCsA : "); print(tCsA); print("\n");
        print("tCsB : "); print(tCsB); print("\n");
        print("tCgC : "); print(tCgC); print("\n");
        print("tCrA : "); print(tCrA); print("\n");
        print("tCrB : "); print(tCrB); print("\n");
        print("tCrC : "); print(tCrC); print("\n");

        print("K_BLOCK_MAX : "); print(size<2>(tCrA));print("\n");
        print("k_tile_count : "); print(size<3>(tAgA));print("\n");
    }
#endif

    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = K_PIPE_MAX-1;

    // Pipe slice
    Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA);

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
        // Wait until our first prefetched tile is loaded in
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();

        // Prefetch the first rmem from the first k-tile
        copy(tCsA_p(_,_,Int<0>{}), tCrA(_,_,Int<0>{}));
        copy(tCsB_p(_,_,Int<0>{}), tCrB(_,_,Int<0>{}));
    }

    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX-1))
    {
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
        {
        if (k_block == K_BLOCK_MAX - 1)
        {
            // Slice the smem_pipe_read smem
            tCsA_p = tCsA(_,_,_,smem_pipe_read);
            tCsB_p = tCsB(_,_,_,smem_pipe_read);

            // Commit the smem for smem_pipe_read
            cp_async_wait<K_PIPE_MAX-2>();
            __syncthreads();
        }

        // Load A, B shmem->regs for k_block+1
        auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;      // static
        copy(tCsA_p(_,_,k_block_next), tCrA(_,_,k_block_next));
        copy(tCsB_p(_,_,k_block_next), tCrB(_,_,k_block_next));
        // Copy gmem to smem before computing gemm on each k-pipe
        if (k_block == 0)
        {
            copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
            copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
            cp_async_fence();

            // Advance the gmem tile
            --k_tile_count;
            if (k_tile_count > 0) { ++k_tile_next; }

            // Advance the smem pipe
            smem_pipe_write = smem_pipe_read;
            ++smem_pipe_read;
            smem_pipe_read = (smem_pipe_read == K_PIPE_MAX) ? 0 : smem_pipe_read;
        }
        // Thread-level register gemm for k_block
        gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
        }

    }
    //
    // Epilogue
    //

    copy(tCrC, tCgC);

}

template <class TA, class TB, class TC>
void
gemm_tn(TA const * A_ptr, TB const * B_ptr, TC * C_ptr, 
    int M, int N, int K,
    int stride_am, int stride_ak,
    int stride_bk, int stride_bn,
    int stride_cm, int stride_cn){

    using namespace cute;
    auto prob_shape = make_shape(M, N, K);
    
    auto dA = make_stride(stride_am, Int<1>{});
    auto dB = make_stride(stride_bn, Int<1>{});
    // auto dB = make_stride(stride_bk, stride_bn);
    auto dC = make_stride(stride_cm, Int<1>{});

    auto bM = Int<64>{};
    auto bN = Int<64>{};
    auto bK = Int<32>{};

    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<10>{};  // Pipeline

    auto sC = make_layout(make_shape(bM, bN)); 

    using SmemLayoutAtom = decltype(composition(
            Swizzle<3, 3, 3>{},
            make_layout(make_shape(Int<8>{}, Int<bK>{}),
                        make_stride(Int<bK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{},
                                                make_shape(Int<bM>{}, Int<bK>{}, bP)));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{},
                                                make_shape(Int<bN>{}, Int<bK>{}, bP)));
    SmemLayoutA sA;
    SmemLayoutB sB;

    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, TA>;
    using G2SCopyA =
        decltype(make_tiled_copy(g2s_copy_atom{},
                                    make_layout(make_shape(Int<32>{}, Int<4>{}),
                                                make_stride(Int<4>{}, Int<1>{})),
                                    make_layout(make_shape(Int<1>{}, Int<8>{}))));
    G2SCopyA copyA;
    G2SCopyA copyB;

    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;
    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 2 * kMmaEURepeatK * get<2>(mma_atom_shape{});
    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));
    MMA mmaC;

#if 0
    print("  dA : "); print(  dA); print("\n");
    print("  dB : "); print(  dB); print("\n");
    print("  dC : "); print(  dC); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("  sC : "); print(  sC); print("\n");
    printf("copyA: \n");
    print(copyA);print("\n");
    printf("copyB: \n");
    print(copyB);print("\n");
    printf("mmaC: \n");
    print(mmaC);print("\n");
    printf("size(mmaC): ");print(size(mmaC));printf("\n");
    printf("size(copyA): ");print(size(copyA));printf("\n");
    printf("cta_tiler: "); print(cta_tiler); printf("\n");
    printf("cta_tiler: "); print(select<0,1>(cta_tiler)); printf("\n");
    printf("smem_size: %d", cosize_v<SmemLayoutAtom> + cosize_v<SmemLayoutAtom>);printf("\n");

    // Layout s2xh15 = make_layout(make_shape(2,make_shape(3,5)),
    //                        make_stride(15,make_stride(5,1)));
    // print_layout(s2xh15); 

    // Layout s6xh5 = make_layout(make_shape(make_shape(2,3),5),
    //                        make_stride(make_stride(15, 5), 1));
    // print_layout(s6xh5); 

    // Layout s15xh2 = make_layout(make_shape(make_shape(3,5), 2),
    //                            make_stride(make_stride(5,1), 15));
    // print_layout(s15xh2); 

    // Layout s4xh8 = make_layout(make_shape(4,make_shape(2,4)),
    //                        make_stride(2,make_stride(1,8)));
    // print_layout(s4xh8);                          
#endif
    dim3 dimBlock(size(mmaC));
    dim3 dimGrid(size(ceil_div(M, bM)),
                 size(ceil_div(N, bN)));
    int smem_size = cosize_v<SmemLayoutAtom> + cosize_v<SmemLayoutAtom>;
    cudaFuncSetAttribute(gemm_device<decltype(prob_shape), decltype(cta_tiler),
                                                  TA, decltype(dA), decltype(sA), decltype(copyA),
                                                  TB, decltype(dB), decltype(sB), decltype(copyB),
                                                  TC, decltype(dC), decltype(sC), decltype(mmaC)>, 
                      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    gemm_device<<<dimGrid, dimBlock, smem_size>>>
        (prob_shape, cta_tiler,
        A_ptr, dA, sA, copyA,
        B_ptr, dB, sB, copyB,
        C_ptr, dC, sC, mmaC);
}

} // namespace cute_gemm_f16_s16816fprop_optimized_f16_64x64_32x10_nhwc_align8
} // namespace bench