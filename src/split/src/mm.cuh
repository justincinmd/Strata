// FNX_MM: the int8 matrix instruction of gfx12 (v_wmma_i32_16x16x16_iu8, wave32) for the fixed-weight kernels (dense.cuh q8s_mm_block,
// hc_mm.cuh), on the engine's int8 weights with a scale per 32 columns.
// Fragment layout (checked bit for bit on gfx1201 with a probe kernel): A (M x K) lane l holds row l & 15, k 8 (l >> 4) + 0..7 (the 8 bytes of
// its two dwords); B (K x N) lane l holds column l & 15, the same k; the accumulator lane l holds column l & 15, rows 8 (l >> 4) + v in register v.
// Here A is the activations (M = token), B the weights (N = weight row): a lane ends with ONE weight row (one weight scale to load) and 8 tokens.
#pragma once

#if FNX_MM && FNX_HIP
namespace fnx_mm {
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
__device__ __forceinline__ v8i wmma(const v2i a, const v2i b, const v8i c) {
#if defined(__GFX12__) || defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
#else
    (void) a;
    (void) b;
    __builtin_trap();   // not a gfx12 device pass: the launchers never select these kernels there (kernels.cu mm_gfx12)
    return c;
#endif
}
// One unit: 16 weight rows x 32 columns (512 bytes of weights) against the same 32 columns of up to 16 tokens' int8 activations, two instructions.
// wv: lane l holds weight row l & 15's 16 bytes at column 16 (l >> 4) of the block; av: lane l holds token l & 15's 16 bytes at the same columns (a lane
// of a token past the window: any bytes, its sums are not read). The first instruction takes each lane's low 8 bytes (columns 0..7 and 16..23), the second
// the high 8 (8..15 and 24..31): every column once, so register v of lane l is the exact dot of row l & 15 and token 8 (l >> 4) + v over the 32 columns
// (at most 32 * 127 * 128, far inside an int).
__device__ __forceinline__ v8i unit(const uint4 wv, const uint4 av) {
    v8i c = {0, 0, 0, 0, 0, 0, 0, 0};
    c = wmma(v2i{(int) av.x, (int) av.y}, v2i{(int) wv.x, (int) wv.y}, c);
    return wmma(v2i{(int) av.z, (int) av.w}, v2i{(int) wv.z, (int) wv.w}, c);
}
// Between two units of a wave: nothing is scheduled across. Without it the compiler runs the matrix instructions of every unit of a round side by side
// and holds their sums (8 registers a unit) and operands at once: k_hc_mm took 174-256 registers a wave (and stack slots at 16 rows), and a wave's
// register count sets how many blocks are resident on an R9700.
__device__ __forceinline__ void seq() {
#if defined(__GFX12__) || defined(__gfx1200__) || defined(__gfx1201__)
    __builtin_amdgcn_sched_barrier(0);
#endif
}
}  // namespace fnx_mm
#endif
