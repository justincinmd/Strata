// The prompt path's routed experts on the tensor cores (moe_mma.cuh), in an object of their own so a change to them
// recompiles in seconds.
#include "kernels.hpp"
#include "env.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "qdot.cuh"   // block layouts and codebook tables (strata-quant-common.h)

namespace fnx {

namespace {

void ck_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
__device__ __forceinline__ float silu_(float x) { return x / (1.0f + __expf(-x)); }

#include "moe_w.cuh"
#include "moe_mma.cuh"
#include "moe_mma_iq.cuh"

template <int TYPE, typename OutT>
void mma_launch(const MoeW& w, const BigJob* jobs, const int* njobs, int njobs_cap, const __half* x, OutT* out, cudaStream_t s) {
    constexpr size_t sm = mm_smem<MM_BM>();
    static_assert(sm <= 48 * 1024, "moe_mma: shared memory past the default limit");
    // FNX_PF_MM_DIRECT=1: outputs straight from the accumulators (moe_mma.cuh, DIRECT; same values); needs 8-row groups
    static const bool direct = env_int("FNX_PF_MM_DIRECT", FAST) != 0;   // on in a FAST build: bench down 1.256 -> 0.982 ms, gate|up 1.165 -> 1.117 (same hashes)
    const dim3 grid((w.nrows + MM_BM - 1) / MM_BM, njobs_cap);
    if (direct && w.nrows % 8 == 0) k_moe_mma<TYPE, MM_BM, OutT, true><<<grid, MM_THREADS, sm, s>>>(w, jobs, njobs, x, w.ncols, out);
    else k_moe_mma<TYPE, MM_BM, OutT><<<grid, MM_THREADS, sm, s>>>(w, jobs, njobs, x, w.ncols, out);
}

template <typename OutT>
void moe_mma(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const __half* x, OutT* out, void* st) {
    const MoeW w{(const char*) A.p, sa, A.row_bytes, A.n_out, A.n_in, A.ep};
    if (w.ncols % 32) throw std::runtime_error("moe_mma: columns not in units of 32");
    cudaStream_t s = (cudaStream_t) st;
    switch (A.type) {
        case T_Q4_K: mma_launch<T_Q4_K, OutT>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_Q5_K: mma_launch<T_Q5_K, OutT>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_Q5_1: mma_launch<T_Q5_1, OutT>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_Q8_0: mma_launch<T_Q8_0, OutT>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_Q4_0: mma_launch<T_Q4_0, OutT>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_IQ4_NL: mma_launch<T_IQ4_NL, OutT>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_IQ4_XS: mma_launch<T_IQ4_XS, OutT>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_IQ3_S: mma_launch<T_IQ3_S, OutT>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_IQ3_XXS: mma_launch<T_IQ3_XXS, OutT>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_IQ2_S: mma_launch<T_IQ2_S, OutT>(w, jobs, njobs, njobs_cap, x, out, s); break;
        default: throw std::runtime_error("moe_mma: expert type " + std::to_string(A.type) + " not handled");
    }
    ck_launch("moe_mma");
}
}  // namespace
void moe_gate_up_mma(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const uint16_t* xh, float* gu, void* st) {
    moe_mma(A, sa, jobs, njobs, njobs_cap, reinterpret_cast<const __half*>(xh), gu, st);
}
void swiglu_h16(const float* gu, int ffs, int P, uint16_t* h16, void* st) {
    k_swiglu_h16<<<(unsigned) std::min<size_t>(4096, ((size_t) P * ffs + 255) / 256), 256, 0, (cudaStream_t) st>>>(gu, ffs, P, reinterpret_cast<__half*>(h16));
    ck_launch("swiglu_h16");
}
void moe_down_mma(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const uint16_t* h16, float* part, void* st) {
    moe_mma(A, sa, jobs, njobs, njobs_cap, reinterpret_cast<const __half*>(h16), part, st);
}
void moe_down_mma16(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const uint16_t* h16, uint16_t* part16, void* st) {
    moe_mma(A, sa, jobs, njobs, njobs_cap, reinterpret_cast<const __half*>(h16), reinterpret_cast<__half*>(part16), st);
}

}  // namespace fnx
