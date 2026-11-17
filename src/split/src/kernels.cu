// fnx kernels. Quantized dot products are ggml's (vecdotq.cuh) arithmetic in qdot.cuh, the rest follows llama.cpp's qwen4exp graph
// (src/models/qwen4exp.cpp) operation by operation.
#include <mutex>
#include "kernels.hpp"
#include "env.hpp"
#include "kprof.hpp"
#include "tk.hpp"   // generator: a launch site's kernel from a code object (FNX_TK_DIR, FNX_TK_<FAMILY>)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>   // HIP: src/hip/mma.h (gfx12 WMMA)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "qdot.cuh"        // the quantized decode dots (ggml vecdotq.cuh's arithmetic, MIT; see the file)
#include "iqdot_hip.cuh"   // HIP: the i-quant dots with a cheaper sign expansion (same integers as ggml's)
#include "nt_load.cuh"     // FNX_WLD1 / FNX_WLD2, the weight streams' loads (non-temporal by FNX_MALL_NT)
#include "mm.cuh"          // the int8 matrix instruction (FNX_MM)
#include "diag.cuh"        // the time-out branches' record (ErrDiag)

// compiled in parts (Makefile: build/kernels_pN.o with -DFNX_KPART=N), each with its share of the launchers below and
// so only the template kernels they instantiate: one file's ~700 kernels took ~7 minutes on one core
#ifndef FNX_KPART
#error "kernels.cu: compile with -DFNX_KPART=0..8 (one object per part)"
#endif

namespace fnx {

namespace {

void ck_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
    if (g_kprof_dev >= 0) kprof_mark(what);
}

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float silu_(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float bf(uint16_t h) { return __uint_as_float((unsigned) h << 16); }
__device__ __forceinline__ float f16(uint16_t h) { return __half2float(__ushort_as_half(h)); }

// block-wide sum for blockDim.x <= 1024; every thread gets the result
__device__ float block_sum(float v, float* sh) {
    v = wsum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = (blockDim.x + 31) >> 5;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    float s = 0.f;
    for (int i = 0; i < nw; ++i) s += sh[i];
    return s;
}
__device__ float block_max(float v, float* sh) {
    v = wmax(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = (blockDim.x + 31) >> 5;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    float s = -INFINITY;
    for (int i = 0; i < nw; ++i) s = fmaxf(s, sh[i]);
    return s;
}

// ---------------------------------------------------------------- q8_1 quantization (ggml's rounding)
}  // namespace
// the split twins of q8_1 activation buffers: one registry for the engine (part 0 holds it)
struct Q8Twin {
    const char* q8;
    size_t bytes;
    int8_t* tq;
    float* td;
};
std::vector<Q8Twin>& q8_twins();
namespace {
// the twin of the q8_1 rows starting at y (rows of n columns), or nulls
void twin_of(const void* y, int n, int8_t*& tq, float*& td) {
    tq = nullptr;
    td = nullptr;
    for (const Q8Twin& w : q8_twins())
        if ((const char*) y >= w.q8 && (const char*) y < w.q8 + w.bytes) {
            const size_t off = (size_t) ((const char*) y - w.q8), rb = (size_t) (n / 32) * 36;
            if (off % rb) throw std::runtime_error("q8 twin: a q8_1 pointer between rows");
            const size_t r = off / rb;
            tq = w.tq + r * n;
            td = w.td + r * (n / 32);
            return;
        }
}

__global__ void k_quantize(const float* x, block_q8_1* q, int n, int8_t* tq, float* td) {
    const int row = blockIdx.y, b = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b >= n / 32) return;
    const float xi = x[(size_t) row * n + b * 32 + lane];
    const float amax = wmax(fabsf(xi)), sum = wsum(xi);
    const float d = amax / 127.0f;
    const int8_t qv = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
    block_q8_1& y = q[(size_t) row * (n / 32) + b];
    y.qs[lane] = qv;
    if (lane == 0) y.ds = make_half2(__float2half(d), __float2half(sum));
    if (tq) {
        tq[(size_t) row * n + b * 32 + lane] = qv;
        if (lane == 0) td[(size_t) row * (n / 32) + b] = __half2float(__float2half(d));
    }
}

// ---------------------------------------------------------------- jobs of quantized matrix x vectors
template <GgmlType type> __device__ __forceinline__ float vdot(const void* v, const block_q8_1* y, int kbx, int iqs) {
    if constexpr (type == T_Q8_0) return vec_dot_q8_0_q8_1(v, y, kbx, iqs);
    else if constexpr (type == T_Q4_K) return vec_dot_q4_K_q8_1(v, y, kbx, iqs);
    else if constexpr (type == T_Q5_K) return vec_dot_q5_K_q8_1(v, y, kbx, iqs);
    else if constexpr (type == T_Q5_1) return vec_dot_q5_1_q8_1(v, y, kbx, iqs);
    else if constexpr (type == T_Q4_0) return vec_dot_q4_0_q8_1(v, y, kbx, iqs);
    else if constexpr (type == T_Q6_K) return vec_dot_q6_K_q8_1(v, y, kbx, iqs);
    else if constexpr (type == T_IQ4_NL) return vec_dot_iq4_nl_q8_1(v, y, kbx, iqs);
    else if constexpr (type == T_IQ4_XS) return vec_dot_iq4_xs_q8_1(v, y, kbx, iqs);
#if FNX_HIP
    else if constexpr (type == T_IQ3_S) return fnx_vec_dot_iq3_s_q8_1(v, y, kbx, iqs);
#else
    else if constexpr (type == T_IQ3_S) return vec_dot_iq3_s_q8_1(v, y, kbx, iqs);
#endif
#if FNX_HIP
    else if constexpr (type == T_IQ3_XXS) return fnx_vec_dot_iq3_xxs_q8_1(v, y, kbx, iqs);
#else
    else if constexpr (type == T_IQ3_XXS) return vec_dot_iq3_xxs_q8_1(v, y, kbx, iqs);
#endif
#if FNX_HIP
    else if constexpr (type == T_IQ2_S) return fnx_vec_dot_iq2_s_q8_1(v, y, kbx, iqs);
#else
    else if constexpr (type == T_IQ2_S) return vec_dot_iq2_s_q8_1(v, y, kbx, iqs);
#endif
    else return 0.0f;
}
template <GgmlType type> constexpr int vdr_of() {
    if constexpr (type == T_Q8_0) return VDR_Q8_0;
    else if constexpr (type == T_Q4_K) return VDR_Q4_K;
    else if constexpr (type == T_Q5_K) return VDR_Q5_K;
    else if constexpr (type == T_Q5_1) return VDR_Q5_1;
    else if constexpr (type == T_Q4_0) return VDR_Q4_0;
    else if constexpr (type == T_Q6_K) return VDR_Q6_K;
    else if constexpr (type == T_IQ4_NL) return VDR_IQ4_NL;
    else if constexpr (type == T_IQ4_XS) return VDR_IQ4_XS;
    else if constexpr (type == T_IQ3_S) return VDR_IQ3_S;
    else if constexpr (type == T_IQ3_XXS) return VDR_IQ3_XXS;
    else return VDR_IQ2_S;
}

constexpr int MMVQ_THREADS = 128;

// G lanes per weight row; a block covers MMVQ_THREADS / G rows of one job
template <GgmlType type, int G>
__global__ void __launch_bounds__(MMVQ_THREADS) k_mmvq(const char* __restrict__ wbase, size_t w_stride, size_t row_bytes,
                                                     int nrows, int ncols, const Job* __restrict__ jobs,
                                                     const int* __restrict__ njobs, int njobs_max,
                                                     const block_q8_1* __restrict__ y, int y_stride,
                                                     float* __restrict__ out, int out_stride) {
    const int ji = blockIdx.y;
    if (ji >= (njobs ? *njobs : njobs_max)) return;
    __shared__ Job jb;
    if (threadIdx.x == 0) jb = jobs[ji];
    __syncthreads();
    constexpr int qk = qtraits<type>::qk, qi = qtraits<type>::qi, vdr = vdr_of<type>();
    constexpr int per_block = qi / vdr;                  // lanes that share one weight block
    static_assert(G >= per_block && G % per_block == 0, "G must cover a block");
    constexpr int bpi = G / per_block;                   // weight blocks a group advances per step
    const int gl = threadIdx.x % G, row = blockIdx.x * (MMVQ_THREADS / G) + threadIdx.x / G;
    const int bpr = ncols / qk, n = jb.n;
    const bool live = row < nrows;
    const char* wr = wbase + (size_t) jb.w * w_stride + (size_t) (live ? row : 0) * row_bytes;
    float tmp[MAXT] = {};
    if (live)
        for (int kbx = gl / per_block; kbx < bpr; kbx += bpi) {
            const int kby = kbx * (qk / QK8_1), kqs = vdr * (gl % per_block);
#pragma unroll
            for (int j = 0; j < MAXT; ++j) {
                if (j >= n) break;
                tmp[j] += vdot<type>(wr, y + (size_t) jb.y[j] * y_stride + kby, kbx, kqs);
            }
        }
#pragma unroll
    for (int j = 0; j < MAXT; ++j) {
        if (j >= n) break;
        float v = tmp[j];
#pragma unroll
        for (int o = G / 2; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (live && gl == 0) out[(size_t) jb.out[j] * out_stride + row] = v;
    }
}

#include "dense.cuh"
#include "moe.cuh"
#include "moe_q4k.cuh"

// ---------------------------------------------------------------- f32 matrix x vectors (router, alpha / beta, gates)
// y[t][row] = W[row] . x[t]: a warp per row (WPR = 1) or a block per row (WPR = 4 warps, the columns split), n = N
template <int T, int WPR>
__global__ void __launch_bounds__(256) k_gemv_f32(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y,
                                                 int rows) {
    constexpr int NC = N / 4 / (32 * WPR);   // float4 chunks per lane
    static_assert(N % (4 * 32 * WPR) == 0, "chunks per lane");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = WPR == 1 ? blockIdx.x * 8 + warp : blockIdx.x;
    const int part = WPR == 1 ? 0 : warp;     // this warp's share of the columns
    if (row >= rows) return;
    const float4* w4 = reinterpret_cast<const float4*>(W + (size_t) row * N) + part * NC * 32;
    float4 wv[NC];
#pragma unroll
    for (int i = 0; i < NC; ++i) wv[i] = FNX_WLD2(w4 + i * 32 + lane);
    float s[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float4* x4 = reinterpret_cast<const float4*>(x + (size_t) t * N) + part * NC * 32;
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < NC; ++i) {
            const float4 b = x4[i * 32 + lane];
            a += wv[i].x * b.x + wv[i].y * b.y + wv[i].z * b.z + wv[i].w * b.w;
        }
        s[t] = wsum(a);
    }
    if (WPR == 1) {
        if (lane < T)
#pragma unroll
            for (int t = 0; t < T; ++t)
                if (t == lane) y[(size_t) t * rows + row] = s[t];
        return;
    }
    __shared__ float red[T][WPR];
    if (lane == 0)
#pragma unroll
        for (int t = 0; t < T; ++t) red[t][warp] = s[t];
    __syncthreads();
    if (threadIdx.x < T) {
        float v = 0.f;
#pragma unroll
        for (int k = 0; k < WPR; ++k) v += red[threadIdx.x][k];
        y[(size_t) threadIdx.x * rows + row] = v;
    }
}

// k_gemv_f32<T, 4>'s block for `row`, in a block of Q8S_THREADS (warps past the four take no columns), the same sums
template <int T>
__device__ __forceinline__ void gemv4_block(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int rows,
                                            int row) {
    constexpr int WPR = 4, NC = N / 4 / (32 * WPR);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __shared__ float red[T][WPR];
    if (warp < WPR) {
        const float4* w4 = reinterpret_cast<const float4*>(W + (size_t) row * N) + warp * NC * 32;
        float4 wv[NC];
#pragma unroll
        for (int i = 0; i < NC; ++i) wv[i] = FNX_WLD2(w4 + i * 32 + lane);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4* x4 = reinterpret_cast<const float4*>(x + (size_t) t * N) + warp * NC * 32;
            float a = 0.f;
#pragma unroll
            for (int i = 0; i < NC; ++i) {
                const float4 b = x4[i * 32 + lane];
                a += wv[i].x * b.x + wv[i].y * b.y + wv[i].z * b.z + wv[i].w * b.w;
            }
            a = wsum(a);
            if (lane == 0) red[t][warp] = a;
        }
    }
    __syncthreads();
    if (threadIdx.x < T) {
        float v = 0.f;
#pragma unroll
        for (int k = 0; k < WPR; ++k) v += red[threadIdx.x][k];
        y[(size_t) threadIdx.x * rows + row] = v;
    }
}

// a Q8_0 matrix (q8s) and a small f32 one (gemv_f32's block-per-row case) in one launch instead of two streams: the
// f32 rows' blocks first (short; they start at once), then the q8s blocks
template <int T, int R, bool SPLIT>
__global__ void __launch_bounds__(Q8S_THREADS) k_q8s_ab(const Q8Segs g, int rows, int cols, const block_q8_1* __restrict__ y,
                                                       const int8_t* __restrict__ tq, const float* __restrict__ td, const float* __restrict__ W,
                                                       const float* __restrict__ x, float* __restrict__ yf, int rf) {
    if ((int) blockIdx.x < rf) gemv4_block<T>(W, x, yf, rf, blockIdx.x);
    else q8s_block<T, R, SPLIT, false>(g, rows, cols, y, tq, td, blockIdx.x - rf);
}
#if FNX_MM && FNX_HIP
// the same launch with the matrix kernel's blocks (16 rows each) after the f32 rows'
template <int T, bool SPLIT>
__global__ void __launch_bounds__(Q8S_THREADS) k_q8s_ab_mm(const Q8Segs g, int rows, int cols, const block_q8_1* __restrict__ y,
                                                          const int8_t* __restrict__ tq, const float* __restrict__ td, const float* __restrict__ W,
                                                          const float* __restrict__ x, float* __restrict__ yf, int rf) {
    if ((int) blockIdx.x < rf) gemv4_block<T>(W, x, yf, rf, blockIdx.x);
    else q8s_mm_block<T, SPLIT, false>(g, rows, cols, y, tq, td, blockIdx.x - rf);
}
#endif

// ---------------------------------------------------------------- hyper-connection read
#include "hc.cuh"
#include "hc_mm.cuh"   // the read as a ticket queue (FNX_MM_HC)

// ---------------------------------------------------------------- MoE routing, swiglu, combine
__global__ void k_topk(const float* logits, int* topi, float* topw) {
    __shared__ float p[NEXP];
    __shared__ float sh[32];
    __shared__ int si[32];
    const int t = blockIdx.x, e = threadIdx.x;
    const float l = logits[(size_t) t * NEXP + e];
    const float m = block_max(l, sh);
    const float ex = __expf(l - m);
    const float z = block_sum(ex, sh);
    p[e] = ex / z;
    __syncthreads();
    float sel_sum = 0.f;
    for (int k = 0; k < TOPK; ++k) {
        // argmax of p over the block (ties: the lower index)
        float v = p[e];
        int idx = e;
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        __syncthreads();
        if ((e & 31) == 0) { sh[e >> 5] = v; si[e >> 5] = idx; }
        __syncthreads();
        if (e == 0) {
            float bv = sh[0];
            int bi = si[0];
            for (int w = 1; w < NEXP / 32; ++w)
                if (sh[w] > bv || (sh[w] == bv && si[w] < bi)) { bv = sh[w]; bi = si[w]; }
            topi[t * TOPK + k] = bi;
            topw[t * TOPK + k] = bv;
            p[bi] = -1.0f;
        }
        __syncthreads();
    }
    if (e == 0) {
        for (int k = 0; k < TOPK; ++k) sel_sum += topw[t * TOPK + k];
        for (int k = 0; k < TOPK; ++k) topw[t * TOPK + k] /= sel_sum;
    }
}

// one thread: distinct experts in first-seen order, each with the tokens (and pairs) that picked it
__global__ void k_jobs(const int* topi, int T, Job* gu, Job* dn, int* njobs) {
    int nj = 0;
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < TOPK; ++k) {
            const int e = topi[t * TOPK + k], pair = t * TOPK + k;
            int j = 0;
            while (j < nj && gu[j].w != e) ++j;
            if (j == nj) { gu[j].w = e; gu[j].n = 0; dn[j].w = e; dn[j].n = 0; ++nj; }
            gu[j].y[gu[j].n] = t;
            gu[j].out[gu[j].n] = pair;
            ++gu[j].n;
            dn[j].y[dn[j].n] = pair;
            dn[j].out[dn[j].n] = pair;
            ++dn[j].n;
        }
    njobs[0] = nj;
    njobs[1] = nj;
}

__global__ void k_swiglu_q8(const float* g, const float* u, block_q8_1* q, int n) {
    const int row = blockIdx.y, b = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b >= n / 32) return;
    const size_t i = (size_t) row * n + b * 32 + lane;
    const float xi = silu_(g[i]) * u[i];
    const float amax = wmax(fabsf(xi)), sum = wsum(xi);
    const float d = amax / 127.0f;
    block_q8_1& y = q[(size_t) row * (n / 32) + b];
    y.qs[lane] = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
    if (lane == 0) y.ds = make_half2(__float2half(d), __float2half(sum));
}

__global__ void k_combine(const float* part, const float* topw, const float* sh, const float* sh_logit, float* out) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) s = fmaf(topw[t * TOPK + k], part[((size_t) t * TOPK + k) * N + i], s);
    out[(size_t) t * N + i] = s + sigm(sh_logit[t]) * sh[(size_t) t * N + i];
}

// ---------------------------------------------------------------- recurrent layer
__global__ void k_gdn_conv(const float* x, float* hist, const float* w, float* out, int ch, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float win[3 + MAXT];
    for (int i = 0; i < 3; ++i) win[i] = hist[i * ch + c];
    for (int t = 0; t < T; ++t) win[3 + t] = x[(size_t) t * ch + c];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = 0; t < T; ++t)
        out[(size_t) t * ch + c] = silu_(w0 * win[t] + w1 * win[t + 1] + w2 * win[t + 2] + w3 * win[t + 3]);
    for (int i = 0; i < 3; ++i) hist[i * ch + c] = win[T + i];
}

// L2 norm of every q and k head: x / sqrt(sum x^2 + eps)
__global__ void k_gdn_qk_norm(float* conv, int nk, int ch, float eps) {
    const int t = blockIdx.y, h = blockIdx.x, i = threadIdx.x;   // 2 * nk heads of HS
    float* x = conv + (size_t) t * ch + h * HS;
    __shared__ float sh[32];
    const float v = x[i];
    const float ss = block_sum(v * v, sh);
    x[i] = v * rsqrtf(ss + eps);
}

__global__ void k_gdn_gates(const float* alpha, const float* beta_raw, const float* dt, const float* a, float* g, float* beta,
                            int nv, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nv * T) return;
    const int h = i % nv;
    const float x = alpha[i] + dt[h];
    const float sp = x > 20.0f ? x : log1pf(__expf(x));
    g[i] = sp * a[h];
    beta[i] = sigm(beta_raw[i]);
}

// one block per value head; thread (col, part) owns rows [part*32, part*32+32) of state column col.
// S is stored transposed: S_T[col][i] = S[i][col].
__global__ void __launch_bounds__(HS * 4) k_gdn_step(float* S, const float* conv, const float* g, const float* beta, float* o,
                                                    int nk, int nv, int ch, int T) {
    const int lv = blockIdx.x, kh = lv % nk, col = threadIdx.x >> 2, part = threadIdx.x & 3;
    float* st = S + ((size_t) lv * HS + col) * HS + part * 32;
    float s[32];
#pragma unroll
    for (int r = 0; r < 32; ++r) s[r] = st[r];
    __shared__ float qs[HS], ks[HS];
    const float scale = rsqrtf((float) HS);
    for (int t = 0; t < T; ++t) {
        const float* row = conv + (size_t) t * ch;
        __syncthreads();
        if (threadIdx.x < HS) qs[threadIdx.x] = row[kh * HS + threadIdx.x];
        else if (threadIdx.x < 2 * HS) ks[threadIdx.x - HS] = row[nk * HS + kh * HS + threadIdx.x - HS];
        __syncthreads();
        const float gv = __expf(g[t * nv + lv]), bv = beta[t * nv + lv];
        const float vv = row[2 * nk * HS + lv * HS + col];
        float kv = 0.f;
#pragma unroll
        for (int r = 0; r < 32; ++r) kv += s[r] * ks[part * 32 + r];
        kv += __shfl_xor_sync(0xffffffffu, kv, 1);
        kv += __shfl_xor_sync(0xffffffffu, kv, 2);
        const float delta = (vv - gv * kv) * bv;
        float at = 0.f;
#pragma unroll
        for (int r = 0; r < 32; ++r) {
            s[r] = gv * s[r] + ks[part * 32 + r] * delta;
            at += s[r] * qs[part * 32 + r];
        }
        at += __shfl_xor_sync(0xffffffffu, at, 1);
        at += __shfl_xor_sync(0xffffffffu, at, 2);
        if (part == 0) o[(size_t) t * nv * HS + lv * HS + col] = at * scale;
    }
#pragma unroll
    for (int r = 0; r < 32; ++r) st[r] = s[r];
}

// rms_norm(o) * w * sigmoid(z), per value head
__global__ void k_gdn_out_norm(const float* o, const float* z, const float* w, float* y, int nv, float eps) {
    const int t = blockIdx.y, h = blockIdx.x, i = threadIdx.x;
    const size_t at = (size_t) t * nv * HS + h * HS + i;
    __shared__ float sh[32];
    const float v = o[at];
    const float ms = block_sum(v * v, sh) / (float) HS;
#if FNX_27B   // the 27B gates with silu(z), Flash-Next with sigmoid(z)
    y[at] = v * rsqrtf(ms + eps) * w[i] * z[at] * sigm(z[at]);
#else
    y[at] = v * rsqrtf(ms + eps) * w[i] * sigm(z[at]);
#endif
}

#include "gdn_core.cuh"
#include "gdn_prompt.cuh"
#include "gdn_tc_prompt.cuh"

// ---------------------------------------------------------------- attention
// per (t, head): RMS norm with the weight, then NEOX rotary on dims [0, NROT): pairs (i, i + NROT/2)
__device__ void norm_rope(float* x, const float* w, int pos, float eps, const Rope& base, float* sh) {
    const int i = threadIdx.x;   // HD threads
    const float v = x[i];
    const float ms = block_sum(v * v, sh) / (float) HD;
    const float y = v * rsqrtf(ms + eps) * w[i];
    __syncthreads();
    x[i] = y;
    __syncthreads();
    if (i < NROT / 2) {
        const float theta = (float) rope_pos(base, pos, i) * rope_unit_freq(base, i, NROT);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        sn *= base.mscale; cs *= base.mscale;
        const float x0 = x[i], x1 = x[i + NROT / 2];
        x[i] = x0 * cs - x1 * sn;
        x[i + NROT / 2] = x0 * sn + x1 * cs;
    }
}

__global__ void k_attn_prep(float* qg, float* k, const float* qn, const float* kn, int nq, int nkv, const int* pos0p, float eps, Rope base,
                            LaneRows lr) {
    const int t = blockIdx.y, h = blockIdx.x, p = row_pos(lr, pos0p, t);
    __shared__ float sh[32];
    base = row_rope(base, lr, t);
    if (h < nq) norm_rope(qg + ((size_t) t * nq + h) * 2 * HD, qn, p, eps, base, sh);
    else norm_rope(k + ((size_t) t * nkv + (h - nq)) * HD, kn, p, eps, base, sh);
}

#include "qsa.cuh"

// split-K attention: block (split s, row t, local KV head kvl) takes positions [s * len, (s + 1) * len) of
// [0, pos0 + t] (len: the context over ATT_SPLITS, rounded up to ATT_CH) in chunks of ATT_CH, for the local q heads
// of that KV head, with an online softmax; it writes each head's running max, sum and HD accumulators. The combine
// (block per local head and row) weighs the splits by exp(max_s - max), divides, and applies sigmoid(gate).
// FNX_ATT_HEADS: a card attends 4x the positions for half the heads, so twice the splits keep the blocks per row the same
#ifndef FNX_ATT_SPLITS_H
#define FNX_ATT_SPLITS_H 32
#endif
constexpr int ATT_SPLITS = ATT_HEADS ? FNX_ATT_SPLITS_H : 16, ATT_CH = 32, ATT_QPK = NQH / NKVH, ATT_MAXH = ATT_QPK, ATT_OWN = NQH / NCARD;
static_assert(!ATT_HEADS || ATT_OWN == NQH_LOC, "by heads, a card's own heads are the heads it attends");
constexpr int ATT_ROWS = 64;   // rows the attention's scratch and merge ring hold (a prompt group; windows use MAXT)
struct AttPart {
    float m, z;
};

// positions <= p that card owns (block j on card kv_card(j)): its blocks before p's block whole, then p's up to p
__device__ __forceinline__ int kv_owned(int p, int card) {
    if (ATT_HEADS) return p + 1;   // by heads: every position
    const int b = p / 4;
    const int blocks = (b + NCARD - 1 - card) / NCARD;   // blocks j < b with j % NCARD == card
    return blocks * 4 + (kv_card(b) == card ? p % 4 + 1 : 0);
}

__global__ void __launch_bounds__(HD) k_attn_part(const float* qg, const __half* kc, const __half* vc, KvPages g, float* acc_out,
                                                  AttPart* mz_out, int nq, int q0, int nkv, int kv0, const int* pos0p,
                                                  const int* __restrict__ cells, const int* __restrict__ n_cells, int card, LaneRows lr) {
    // t: the window's row; its position and pages: its lane's (lr), or the sequence's *pos0p + t
    const int s = blockIdx.x, t = blockIdx.y, kvl = blockIdx.z, i = threadIdx.x, lane = i & 31, warp = i >> 5;
    g = row_pages(g, lr, t);
    // the attended positions (local indices on this card): the selection's list, or every owned one up to the row's
    const int* list = cells && n_cells[t] >= 0 ? cells + (size_t) t * QSA_MAXSEL : nullptr;
    const int ctx = list ? n_cells[t] : kv_owned(row_pos(lr, pos0p, t), card);
    // the local q heads of KV head kv0 + kvl: global heads [(kv0 + kvl) * QPK, + QPK) within [q0, q0 + nq)
    const int g0 = max(q0, (kv0 + kvl) * ATT_QPK), g1 = min(q0 + nq, (kv0 + kvl + 1) * ATT_QPK);
    const int h0 = g0 - q0, nh = g1 - g0;
    const int len = ((ctx + ATT_SPLITS - 1) / ATT_SPLITS + ATT_CH - 1) / ATT_CH * ATT_CH;
    const int p0 = s * len, p1 = min(ctx, p0 + len);
    __shared__ float q[ATT_MAXH][HD];
    __shared__ float kk[ATT_CH][HD + 1];
    __shared__ float e[ATT_MAXH][ATT_CH];
    __shared__ float scale[ATT_MAXH];
    for (int k = i; k < nh * HD; k += HD) q[k / HD][k % HD] = qg[((size_t) t * nq + h0 + k / HD) * 2 * HD + k % HD];
    constexpr int HPW = (ATT_MAXH + HD / 32 - 1) / (HD / 32);   // heads per warp: warp w keeps heads w, w + 8
    float m[HPW], z[HPW];
#pragma unroll
    for (int k = 0; k < HPW; ++k) {
        m[k] = -INFINITY;
        z[k] = 0.f;
    }
    float acc[ATT_MAXH] = {};
    __syncthreads();
#pragma unroll 1
    for (int c0 = p0; c0 < p1; c0 += ATT_CH) {
        const int np = min(ATT_CH, p1 - c0);
        // the chunk's K rows into shared memory and this thread's dimension of its V rows into registers
        float vr[ATT_CH];
#pragma unroll
        for (int pp = 0; pp < ATT_CH; ++pp) {
            const int pos = pp < np ? (list ? list[c0 + pp] : c0 + pp) : 0;
            if (g.q8) {
                const size_t r = kvq_row(g, pos) + (size_t) kvl * HD + i, sc = kvq_scale(g, pos) + ((size_t) kvl * HD + i) / KV8_GROUP;
                kk[pp][i] = pp < np ? reinterpret_cast<const int8_t*>(kc)[r] * __half2float(kc[sc]) : 0.f;
                vr[pp] = pp < np ? reinterpret_cast<const int8_t*>(vc)[r] * __half2float(vc[sc]) : 0.f;
                continue;
            }
            const size_t r = kvp_row(g, pos) + (size_t) kvl * HD + i;   // nkv == NKVH: a row holds both heads
            kk[pp][i] = pp < np ? __half2float(kc[r]) : 0.f;
            vr[pp] = pp < np ? __half2float(vc[r]) : 0.f;
        }
        __syncthreads();
        // head warp + 8 k: lane = position; its score, then the head's online-softmax step
#pragma unroll
        for (int k = 0; k < HPW; ++k) {
            const int hh = warp + k * (HD / 32);
            if (hh >= nh) break;
            float sacc = 0.f;
#pragma unroll 8
            for (int d = 0; d < HD; ++d) sacc = fmaf(q[hh][d], kk[lane][d], sacc);
            const float sv = lane < np ? sacc * rsqrtf((float) HD) : -INFINITY;
            const float mn = fmaxf(m[k], wmax(sv));
            const float w = sv == -INFINITY ? 0.f : __expf(sv - mn);
            const float sc = m[k] == -INFINITY ? 0.f : __expf(m[k] - mn);
            z[k] = z[k] * sc + wsum(w);
            m[k] = mn;
            e[hh][lane] = w;
            if (lane == 0) scale[hh] = sc;
        }
        __syncthreads();
#pragma unroll
        for (int hh = 0; hh < ATT_MAXH; ++hh)
            if (hh < nh) acc[hh] *= scale[hh];
#pragma unroll
        for (int pp = 0; pp < ATT_CH; ++pp)
#pragma unroll
            for (int hh = 0; hh < ATT_MAXH; ++hh)
                if (hh < nh) acc[hh] = fmaf(e[hh][pp], vr[pp], acc[hh]);
    }
#pragma unroll
    for (int hh = 0; hh < ATT_MAXH; ++hh) {
        if (hh >= nh) break;
        const size_t slot = ((size_t) t * nq + h0 + hh) * ATT_SPLITS + s;
        acc_out[slot * HD + i] = acc[hh];
    }
#pragma unroll
    for (int k = 0; k < HPW; ++k) {
        const int hh = warp + k * (HD / 32);
        if (hh < nh && lane == 0) mz_out[((size_t) t * nq + h0 + hh) * ATT_SPLITS + s] = AttPart{m[k], z[k]};
    }
}

// this card's splits of head h, row t, as one partial: part[(t * NQH + h) * (HD + 2)] = acc (HD), m, z
__global__ void __launch_bounds__(HD) k_attn_local(const float* acc_in, const AttPart* mz_in, float* part) {
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const size_t base = ((size_t) t * NQH + h) * ATT_SPLITS;
    float mx = -INFINITY;
#pragma unroll
    for (int s = 0; s < ATT_SPLITS; ++s) mx = fmaxf(mx, mz_in[base + s].m);
    float z = 0.f, acc = 0.f;
#pragma unroll
    for (int s = 0; s < ATT_SPLITS; ++s) {
        const AttPart p = mz_in[base + s];
        if (p.m == -INFINITY) continue;
        const float w = __expf(p.m - mx);
        z = fmaf(w, p.z, z);
        acc = fmaf(w, acc_in[(base + s) * HD + i], acc);
    }
    float* o = part + ((size_t) t * NQH + h) * (HD + 2);
    o[i] = acc;
    if (i == 0) {
        o[HD] = mx;
        o[HD + 1] = z;
    }
}

// (m, z, a) merged with (m2, z2, a2): the softmax of the union of their positions
__device__ __forceinline__ void att_merge(float& m, float& z, float& a, float m2, float z2, float a2) {
    const float mn = fmaxf(m, m2);
    const float w1 = m == -INFINITY ? 0.f : __expf(m - mn), w2 = m2 == -INFINITY ? 0.f : __expf(m2 - mn);
    z = w1 * z + w2 * z2;
    a = w1 * a + w2 * a2;
    m = mn;
}

// the cards' partials reduced onto each head's card (chunk c: heads [ATT_OWN c, ATT_OWN (c + 1))) in NCARD - 1 hops on
// the ring: card g sends its partial of chunk g - 1 to g + 1; each card after merges its own into it and sends it on,
// until the chunk's card, which merges its own last and applies the gate: out[t][h - ATT_OWN g] for its heads. fp32 LL
// words, one receive slot per hop.
__global__ void __launch_bounds__(HD) k_att_merge(AttRing r, const float* qg, const float* part, float* out, int T,
                                                  const unsigned* tagp, int call) {
    const unsigned tag = *tagp * 256u + (unsigned) call + 1u;
    const int g = r.card, i = threadIdx.x, hl = blockIdx.x, t = blockIdx.y;   // head hl of a chunk (0..ATT_OWN-1), row t
    constexpr int W = HD + 2;
    const size_t slot_sz = (size_t) ATT_ROWS * ATT_OWN * W, set_sz = (NCARD - 1) * slot_sz;
    unsigned long long* in = (unsigned long long*) r.in + (tag & 1) * set_sz;
    unsigned long long* nin = (unsigned long long*) r.next_in + (tag & 1) * set_sz;
    auto word = [&](unsigned long long* b, int step, int k) { return b + step * slot_sz + ((size_t) t * ATT_OWN + hl) * W + k; };
    auto mine = [&](int chunk, int k) { return part[((size_t) t * NQH + chunk * ATT_OWN + hl) * W + k]; };
    auto put = [&](unsigned long long* p, float v) {
        *(volatile unsigned long long*) p = (unsigned long long) __float_as_uint(v) | ((unsigned long long) tag << 32);
    };
    auto take = [&](const unsigned long long* p) {
        const long long ts = clock64();
        long long t0 = ts;
        unsigned long long w;
        for (;;) {
            w = *(const volatile unsigned long long*) p;
            if ((unsigned) (w >> 32) == tag) break;
            if (const long long dt = clock64() - t0; __builtin_expect(dt > RING_TIMEOUT_CYCLES, 0)) {
                if (diag_wait_on(r.err, dt, RING_TIMEOUT_CYCLES, ts, t0)) continue;   // the driver had the queues off the cards
                diag_timeout(r.err, DK_ATT_MERGE, tag, (unsigned) (w >> 32), (unsigned) w, DG_NO_SRC, (unsigned) (t * ATT_OWN + hl), dt);
                atomicOr(r.err, 1u);
                break;
            }
        }
        return __uint_as_float((unsigned) w);
    };
    // hop 1: my partial of chunk g - 1
    {
        const int c = (g + NCARD - 1) % NCARD;
        put(word(nin, 0, i), mine(c, i));
        if (i == 0) {
            put(word(nin, 0, HD), mine(c, HD));
            put(word(nin, 0, HD + 1), mine(c, HD + 1));
        }
    }
    // hop s + 1: chunk g - 1 - s from g - 1, merged with mine, on towards its card
#pragma unroll
    for (int s = 1; s < NCARD - 1; ++s) {
        const int c = (g + 2 * NCARD - 1 - s) % NCARD;
        float m = take(word(in, s - 1, HD)), z = take(word(in, s - 1, HD + 1)), a = take(word(in, s - 1, i));
        att_merge(m, z, a, mine(c, HD), mine(c, HD + 1), mine(c, i));
        put(word(nin, s, i), a);
        if (i == 0) {
            put(word(nin, s, HD), m);
            put(word(nin, s, HD + 1), z);
        }
    }
    // my chunk: the others' merged partial, then mine; the gate
    {
        float m = take(word(in, NCARD - 2, HD)), z = take(word(in, NCARD - 2, HD + 1)), a = take(word(in, NCARD - 2, i));
        att_merge(m, z, a, mine(g, HD), mine(g, HD + 1), mine(g, i));
        const int hg = g * ATT_OWN + hl;
        out[((size_t) t * ATT_OWN + hl) * HD + i] = a / z * sigm(qg[((size_t) t * NQH + hg) * 2 * HD + HD + i]);
    }
}

// The same merge in ONE hop (NCARD > 2): card g pushes its partial of chunk c into owner c's receive slot (slot (g - c - 1) mod NCARD, one per
// sender, the ring's NCARD - 1 slots), then merges the chunk it owns: its own partial and the senders' in card order 0 .. NCARD - 1.
// A hop on HIP costs a cross-card flag round trip (13-20 us); the ring's NCARD - 1 of them were most of the merge's 360 us on four cards.
__global__ void __launch_bounds__(HD) k_att_merge_a2a(AttRing r, const float* qg, const float* part, float* out, int T,
                                                      const unsigned* tagp, int call) {
    const unsigned tag = *tagp * 256u + (unsigned) call + 1u;
    const int g = r.card, i = threadIdx.x, hl = blockIdx.x, t = blockIdx.y;
    constexpr int W = HD + 2;
    const size_t slot_sz = (size_t) ATT_ROWS * ATT_OWN * W, set_sz = (NCARD - 1) * slot_sz;
    auto slot_of = [&](int sender, int owner) { return ((sender - owner - 1) % NCARD + NCARD) % NCARD; };   // 0 .. NCARD - 2 for sender != owner
    auto word = [&](unsigned long long* b, int step, int k) { return b + step * slot_sz + ((size_t) t * ATT_OWN + hl) * W + k; };
    auto mine = [&](int chunk, int k) { return part[((size_t) t * NQH + chunk * ATT_OWN + hl) * W + k]; };
    auto put = [&](unsigned long long* p, float v) {
        *(volatile unsigned long long*) p = (unsigned long long) __float_as_uint(v) | ((unsigned long long) tag << 32);
    };
    auto take = [&](const unsigned long long* p) {
        const long long ts = clock64();
        long long t0 = ts;
        unsigned long long w;
        for (;;) {
            w = *(const volatile unsigned long long*) p;
            if ((unsigned) (w >> 32) == tag) break;
            if (const long long dt = clock64() - t0; __builtin_expect(dt > RING_TIMEOUT_CYCLES, 0)) {
                if (diag_wait_on(r.err, dt, RING_TIMEOUT_CYCLES, ts, t0)) continue;   // the driver had the queues off the cards
                diag_timeout(r.err, DK_ATT_MERGE, tag, (unsigned) (w >> 32), (unsigned) w, DG_NO_SRC, (unsigned) (t * ATT_OWN + hl), dt);
                atomicOr(r.err, 1u);
                break;
            }
        }
        return __uint_as_float((unsigned) w);
    };
    for (int c = 0; c < NCARD; ++c) {   // push my partial of every other card's chunk to its owner
        if (c == g) continue;
        unsigned long long* in_c = (unsigned long long*) r.peer_in[c] + (tag & 1) * set_sz;
        const int sl = slot_of(g, c);
        put(word(in_c, sl, i), mine(c, i));
        if (i == 0) {
            put(word(in_c, sl, HD), mine(c, HD));
            put(word(in_c, sl, HD + 1), mine(c, HD + 1));
        }
    }
    unsigned long long* in = (unsigned long long*) r.in + (tag & 1) * set_sz;
    float m = 0.f, z = 0.f, a = 0.f;
    for (int src = 0; src < NCARD; ++src) {   // merge in card order: the same on whichever card owns the chunk
        float m2, z2, a2;
        if (src == g) m2 = mine(g, HD), z2 = mine(g, HD + 1), a2 = mine(g, i);
        else {
            const int sl = slot_of(src, g);
            m2 = take(word(in, sl, HD)), z2 = take(word(in, sl, HD + 1)), a2 = take(word(in, sl, i));
        }
        if (src == 0) m = m2, z = z2, a = a2;
        else att_merge(m, z, a, m2, z2, a2);
    }
    const int hg = g * ATT_OWN + hl;
    out[((size_t) t * ATT_OWN + hl) * HD + i] = a / z * sigm(qg[((size_t) t * NQH + hg) * 2 * HD + HD + i]);
}

__global__ void __launch_bounds__(HD) k_attn_combine(const float* qg, const float* acc_in, const AttPart* mz_in, float* out,
                                                     int nq) {
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const size_t base = ((size_t) t * nq + h) * ATT_SPLITS;
    float mx = -INFINITY;
#pragma unroll
    for (int s = 0; s < ATT_SPLITS; ++s) mx = fmaxf(mx, mz_in[base + s].m);
    float z = 0.f, acc = 0.f;
#pragma unroll
    for (int s = 0; s < ATT_SPLITS; ++s) {
        const AttPart p = mz_in[base + s];
        if (p.m == -INFINITY) continue;
        const float w = __expf(p.m - mx);
        z = fmaf(w, p.z, z);
        acc = fmaf(w, acc_in[(base + s) * HD + i], acc);
    }
    out[((size_t) t * nq + h) * HD + i] = acc / z * sigm(qg[((size_t) t * nq + h) * 2 * HD + HD + i]);
}

#include "attn_mm.cuh"
#include "attn_prompt.cuh"
#include "gdn_mm_prompt.cuh"   // the prompt GDN recurrence on WMMA (FNX_PF_GDN_MM)
#include "attn_prompt_i8.cuh"
#include "topk.cuh"

// ---------------------------------------------------------------- PLE block
// part 1 per (row, stream): grouped norms of key and hidden, the gate, the gated value and its grouped norm
__global__ void __launch_bounds__(1024) k_ple_gate(const float* R, const float* key, const float* value, const float* nk,
                                                   const float* nq, const float* nc, float* gated, float* normd, float eps) {
    const int t = blockIdx.y, c = blockIdx.x;
    __shared__ float sh[32];
    float kk[3], qq[3], vv[3];
    float sk = 0.f, sq = 0.f;
    for (int r = 0; r < 3; ++r) {
        const int i = threadIdx.x + r * 1024;
        kk[r] = i < N ? key[(size_t) t * D + c * N + i] : 0.f;
        qq[r] = i < N ? R[(size_t) t * D + c * N + i] : 0.f;
        vv[r] = i < N ? value[(size_t) t * N + i] : 0.f;
        sk += kk[r] * kk[r];
        sq += qq[r] * qq[r];
    }
    const float rk = rsqrtf(block_sum(sk, sh) / (float) N + eps);
    const float rq = rsqrtf(block_sum(sq, sh) / (float) N + eps);
    float dot = 0.f;
    for (int r = 0; r < 3; ++r) {
        const int i = threadIdx.x + r * 1024;
        if (i < N) dot += (kk[r] * rk * nk[c * N + i]) * (qq[r] * rq * nq[c * N + i]);
    }
    const float s = block_sum(dot, sh) / sqrtf((float) N);
    const float gate = sigm((s > 0.f ? 1.f : (s < 0.f ? -1.f : 0.f)) * sqrtf(fmaxf(fabsf(s), 1e-6f)));
    float sg = 0.f;
    for (int r = 0; r < 3; ++r) {
        vv[r] *= gate;
        sg += vv[r] * vv[r];
    }
    const float rg = rsqrtf(block_sum(sg, sh) / (float) N + eps);
    for (int r = 0; r < 3; ++r) {
        const int i = threadIdx.x + r * 1024;
        if (i < N) {
            gated[(size_t) t * D + c * N + i] = vv[r];
            normd[(size_t) t * D + c * N + i] = vv[r] * rg * nc[c * N + i];
        }
    }
}

// part 2 per channel: the dilated causal conv over the history (9 rows) and this window, silu, residual update
// hsnap [MAXT][9][D]: the history kept when only the first n rows are (variant n - 1, n < T)
__global__ void k_ple_conv(float* R, const float* gated, const float* normd, const float* w, float* hist, float* hsnap, int T) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= D) return;
    constexpr int H = 9, DIL = 3;
    float pad[H + MAXT];
    for (int i = 0; i < H; ++i) pad[i] = hist[i * D + ch];
    for (int t = 0; t < T; ++t) pad[H + t] = normd[(size_t) t * D + ch];
    for (int t = 0; t < T; ++t) {
        float s = 0.f;
        for (int k = 0; k < 4; ++k) s += w[ch * 4 + k] * pad[H + t - (3 - k) * DIL];
        R[(size_t) t * D + ch] += gated[(size_t) t * D + ch] + silu_(s);
    }
    for (int i = 0; i < H; ++i) hist[i * D + ch] = pad[T + i];
    for (int n = 1; n < T; ++n)
        for (int i = 0; i < H; ++i) hsnap[((size_t) (n - 1) * H + i) * D + ch] = pad[n + i];
}

// ---------------------------------------------------------------- ring sum (LL words: two fp16 values | tag << 32)
constexpr int RCH = N / NCARD + 1;     // words per slot row (more than a chunk's pairs)
constexpr int RSTEPS = NCARD - 1;      // receive slots: the reduce hops
constexpr int NP = N / 2;          // pairs per row
// fp32 words: one value | tag per word, pair i at words 2i, 2i + 1
__device__ __forceinline__ void rput32(unsigned long long* p, float2 v, unsigned tag, unsigned* err) {
    if (!(isfinite(v.x) && isfinite(v.y))) atomicOr(err, 2u);
    const unsigned long long t = (unsigned long long) tag << 32;
    *(volatile unsigned long long*) p = (unsigned long long) __float_as_uint(v.x) | t;
    *(volatile unsigned long long*) (p + 1) = (unsigned long long) __float_as_uint(v.y) | t;
}
// timeout in clock64 cycles (Ring::timeout); src (the card whose word this is) and elem only go into the time-out's record
__device__ __forceinline__ float rtake1(const unsigned long long* p, unsigned tag, unsigned* err, long long timeout, unsigned src, unsigned elem) {
    const long long ts = clock64();
    long long t0 = ts;
    unsigned long long w;
    for (;;) {
        w = *(const volatile unsigned long long*) p;
        if ((unsigned) (w >> 32) == tag) break;
        if (const long long dt = clock64() - t0; __builtin_expect(dt > timeout, 0)) {
            if (diag_wait_on(err, dt, timeout, ts, t0)) continue;   // the driver had the queues off the cards
            diag_timeout(err, DK_RING_F32, tag, (unsigned) (w >> 32), (unsigned) w, src, elem, dt);
            atomicOr(err, 1u);
            break;
        }
    }
    return __uint_as_float((unsigned) w);
}
__device__ __forceinline__ void rput(unsigned long long* p, float2 v, unsigned tag, unsigned* err) {
    if (!(fabsf(v.x) <= RING_MAX && fabsf(v.y) <= RING_MAX)) atomicOr(err, 2u);   // out of range (or NaN)
    const __half2 h = __floats2half2_rn(v.x, v.y);
    *(volatile unsigned long long*) p = (unsigned long long) *reinterpret_cast<const unsigned*>(&h) | ((unsigned long long) tag << 32);
}
__device__ __forceinline__ float2 rtake(const unsigned long long* p, unsigned tag, unsigned* err, long long timeout, unsigned src, unsigned elem,
                                        unsigned kind = DK_RING) {
    const long long ts = clock64();
    long long t0 = ts;
    unsigned long long w;
    for (;;) {
        w = *(const volatile unsigned long long*) p;
        if ((unsigned) (w >> 32) == tag) break;
        if (const long long dt = clock64() - t0; __builtin_expect(dt > timeout, 0)) {
            if (diag_wait_on(err, dt, timeout, ts, t0)) continue;   // the driver had the queues off the cards
            diag_timeout(err, kind, tag, (unsigned) (w >> 32), (unsigned) w, src, elem, dt);
            atomicOr(err, 1u);
            break;
        }
    }
    const unsigned lo = (unsigned) w;
    return __half22float2(*reinterpret_cast<const __half2*>(&lo));
}
__device__ __forceinline__ int clo(int k) { return k * NP / NCARD; }   // in pairs
__device__ __forceinline__ int chi(int k) { return (k + 1) * NP / NCARD; }
__device__ __forceinline__ float2 add2(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }

// this card's addend for element (t, i): x, or the routed experts' weighted rows plus the gated shared row
__device__ __forceinline__ float ring_x(const float* x, const RingMoe& m, int T, int t, int i) {
    if (!m.part) return x[(size_t) t * N + i];
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) s = fmaf(m.topw[t * TOPK + k], m.part[((size_t) t * TOPK + k) * N + i], s);
    return fmaf(sigm(m.logits[(size_t) t * (NEXP + 1) + NEXP]), m.part[((size_t) T * TOPK + t) * N + i], s);
}
__device__ __forceinline__ float2 ring_x2(const float* x, const RingMoe& m, int T, int t, int p) {
    return make_float2(ring_x(x, m, T, t, 2 * p), ring_x(x, m, T, t, 2 * p + 1));
}

#if FNX_MM_HC && FNX_HIP
// (FNX_HC_SUM): the window's one-hop sum (k_ring_a2a below: the same steps on every pair, the same words in the same host mailboxes) as the
// first tickets of the hc read that applies the summed output (hc_mm.cuh, hc_mm_run's SUM). x is the read's a.bo.
struct HmRingSum {
    static constexpr bool on = true;
    Ring r;
    float* x;
    const unsigned* tagp;
    int call;
    RingMoe m;
    __device__ __forceinline__ unsigned tag() const { return *tagp * 256u + (unsigned) call + 1u; }
    __device__ __forceinline__ void start() const { diag_prog(r.err, tag()); }
    // how long a sum ticket may take: a wait for each peer's word, one after the other, each with the ring's time-out (saturating)
    __device__ __forceinline__ long long patience() const {
        constexpr long long top = 0x7fffffffffffffffll;
        return r.timeout > top / (NCARD - 1) ? top : r.timeout * (NCARD - 1);
    }
    // token k's pairs c * 256 .. c * 256 + 255: this thread's pair
    template <int T>
    __device__ __forceinline__ void ticket(int k, int c) const {
        static_assert(NP == HM_SUMT * HM_T, "the read's sum tickets cover a token's pairs");
        if (*(volatile const unsigned*) r.err & 1u) return;   // a peer never came: do not wait out every element's timeout
        const unsigned tg = tag();
        const int p = c * HM_T + (int) threadIdx.x;
        const size_t e = (size_t) k * NP + p, set_sz = (size_t) MAXT * NP;
        const float2 v = ring_x2(x, m, T, k, p);
        if (!(fabsf(v.x) <= RING_MAX && fabsf(v.y) <= RING_MAX)) atomicOr(r.err, 2u);   // out of range (or NaN)
        const __half2 h = __floats2half2_rn(v.x, v.y);
        *(volatile unsigned long long*) ((unsigned long long*) r.in + (tg & 1) * set_sz + e) =
            (unsigned long long) *reinterpret_cast<const unsigned*>(&h) | ((unsigned long long) tg << 32);   // volatile: system-coherent, as the ring's words
        float2 sum = make_float2(0.f, 0.f);
#pragma unroll
        for (int src = 0; src < NCARD; ++src) {
            const float2 w = src == r.card ? __half22float2(h)
                                           : rtake((const unsigned long long*) r.peer_in[src] + (tg & 1) * set_sz + e, tg, r.err, r.timeout, (unsigned) src, (unsigned) e);
            sum = src == 0 ? w : add2(sum, w);
        }
        x[2 * e] = sum.x;
        x[2 * e + 1] = sum.y;
    }
};
template <int T>
__global__ void __launch_bounds__(HM_T) k_hc_mm_sum(HcArgs a, unsigned* q, HmRingSum sm) {
    hc_mm_run<T, HmRingSum>(a, q, sm);
}
#endif

template <bool F32>
__global__ void k_ring(Ring r, float* x, int T, const unsigned* tagp, int call, RingMoe m) {
    constexpr int W = F32 ? 2 : 1;   // words per pair
    auto put = [&](unsigned long long* p, float2 v, unsigned tg) {
        if constexpr (F32) rput32(p, v, tg, r.err);
        else rput(p, v, tg, r.err);
    };
    const unsigned prev = (unsigned) ((r.card + NCARD - 1) % NCARD);   // the ring's words come from the card before (the time-out's record names it)
    auto take = [&](const unsigned long long* p, unsigned tg, unsigned el) {   // el: row * NP + pair, for the record only
        if constexpr (F32) return make_float2(rtake1(p, tg, r.err, r.timeout, prev, el), rtake1(p + 1, tg, r.err, r.timeout, prev, el));
        else return rtake(p, tg, r.err, r.timeout, prev, el, DK_RING_CHAIN);
    };
    const unsigned tag = *tagp * 256u + (unsigned) call + 1u;
    diag_prog(r.err, tag);
    const int n = gridDim.x * blockDim.x, tid = blockIdx.x * blockDim.x + threadIdx.x, g = r.card;
    // two buffer sets, alternating by tag: a card can start sum k+1 before a peer has drained sum k, never k+2
    const size_t in_set = (size_t) RSTEPS * r.rows * RCH * W, ag_set = (size_t) r.rows * N * W;
    unsigned long long* in = (unsigned long long*) r.in + (tag & 1) * in_set;
    unsigned long long* nin = (unsigned long long*) r.next_in + (tag & 1) * in_set;
    unsigned long long* ag = (unsigned long long*) r.ag + (tag & 1) * ag_set;
    unsigned long long* nag = (unsigned long long*) r.next_ag + (tag & 1) * ag_set;
    auto slot = [&](unsigned long long* b, int s, int t, int i) { return b + (((size_t) s * r.rows + t) * RCH + i) * W; };
    {   // step 0: my part of chunk g to g+1
        const int lo = clo(g), len = chi(g) - lo;
        for (int e = tid; e < T * len; e += n) {
            const int t = e / len, i = e % len;
            put(slot(nin, 0, t, i), ring_x2(x, m, T, t, lo + i), tag);
        }
    }
#pragma unroll
    for (int s = 1; s < RSTEPS; ++s) {   // step s: chunk g-s arrived with s parts; add mine, on to g+1
        const int c = (g + NCARD - s) % NCARD, lo = clo(c), len = chi(c) - lo;
        for (int e = tid; e < T * len; e += n) {
            const int t = e / len, i = e % len;
            const float2 a = take(slot(in, s - 1, t, i), tag, (unsigned) (t * NP + lo + i));
            put(slot(nin, s, t, i), add2(a, ring_x2(x, m, T, t, lo + i)), tag);
        }
    }
    {   // step NCARD - 1: chunk g+1 with the other cards' parts; add mine: finished. Keep it, send it on
        const int c = (g + 1) % NCARD, lo = clo(c), len = chi(c) - lo;
        for (int e = tid; e < T * len; e += n) {
            const int t = e / len, i = e % len;
            const float2 a = take(slot(in, RSTEPS - 1, t, i), tag, (unsigned) (t * NP + lo + i));
            const float2 v = add2(a, ring_x2(x, m, T, t, lo + i));
            put(ag + ((size_t) t * NP + lo + i) * W, v, tag);
            put(nag + ((size_t) t * NP + lo + i) * W, v, tag);
        }
    }
#pragma unroll
    for (int f = 1; f < RSTEPS; ++f) {   // chunk g-f+1 arrived finished from g-1: forward it (none on two cards)
        const int c = (g + NCARD - f + 1) % NCARD, lo = clo(c), len = chi(c) - lo;
        for (int e = tid; e < T * len; e += n) {
            const int t = e / len, i = e % len;
            put(nag + ((size_t) t * NP + lo + i) * W, take(ag + ((size_t) t * NP + lo + i) * W, tag, (unsigned) (t * NP + lo + i)), tag);
        }
    }
    // everything: written back into x
    for (int e = tid; e < T * NP; e += n) {
        const float2 v = take(ag + (size_t) e * W, tag, (unsigned) e);
        x[2 * e] = v.x;
        x[2 * e + 1] = v.y;
    }
}

// The window sum in ONE hop (NCARD > 2, fp16 words): every card writes its addend of every (row, pair) once, as a tagged word, into its own host
// mailbox ([2 sets][MAXT][NP] words, inside the ring's `in` buffer), then reads the other cards' words and adds all NCARD values in card order 0 ..
// NCARD - 1 in fp32 (its own addend rounded to fp16 first, as the others see it): every card gets the same sum, with one fp16 rounding per addend
// instead of the ring's NCARD - 1. A tagged word is its own valid flag, so there is no fence and no flag write; 2 (NCARD - 1) hops of a cross-card
// round trip (13-20 us each on HIP) become one.
__global__ void k_ring_a2a(Ring r, float* x, int T, const unsigned* tagp, int call, RingMoe m) {
    const unsigned tag = *tagp * 256u + (unsigned) call + 1u;
    const int g = r.card, n = gridDim.x * blockDim.x, tid = blockIdx.x * blockDim.x + threadIdx.x;
    static_assert(NCARD < 3 || NP <= 2 * (NCARD - 1) * RCH, "the one-hop sum's mailbox (2 sets x MAXT x NP words) must fit the ring's receive buffer");
    const size_t set_sz = (size_t) MAXT * NP;
    unsigned long long* mine = (unsigned long long*) r.in + (tag & 1) * set_sz;
    diag_prog(r.err, tag);
    for (int e = tid; e < T * NP; e += n) {
        if (*(volatile const unsigned*) r.err & 1u) return;   // a peer never came: do not wait out every element's timeout
        const int t = e / NP, p = e % NP;
        const float2 v = ring_x2(x, m, T, t, p);
        if (!(fabsf(v.x) <= RING_MAX && fabsf(v.y) <= RING_MAX)) atomicOr(r.err, 2u);   // out of range (or NaN)
        const __half2 h = __floats2half2_rn(v.x, v.y);
        *(volatile unsigned long long*) (mine + e) = (unsigned long long) *reinterpret_cast<const unsigned*>(&h) | ((unsigned long long) tag << 32);   // volatile: system-coherent, as the ring's words
        float2 sum = make_float2(0.f, 0.f);
#pragma unroll
        for (int src = 0; src < NCARD; ++src) {
            const float2 w = src == g ? __half22float2(h)
                                      : rtake((const unsigned long long*) r.peer_in[src] + (tag & 1) * set_sz + e, tag, r.err, r.timeout, (unsigned) src, (unsigned) e);
            sum = src == 0 ? w : add2(sum, w);
        }
        x[2 * e] = sum.x;
        x[2 * e + 1] = sum.y;
    }
}

// The window sum in ONE hop with the words PUSHED into the other cards' device memory (FNX_PEER, AMD): every card stores its tagged word of each (row, pair) into
// slot [source = this card] of every other card's mailbox (a posted remote store, 27 GB/s, no cache to go stale), then reads its own mailbox (local memory) for the
// others' words and adds all NCARD values in card order. A flag round trip over peer memory is 2.1-2.3 us against 13-20 us for stream flags; reads FROM a peer are
// slow across the two pairs of this box, which is why nothing here reads remote memory.
// TWO (FNX_RING_2PHASE, on in a FAST build): a thread pushes ALL of its elements (up to 4 at a time) before it waits for any of the peers' words. As
// published a thread with k elements pushed one, waited for that element's words, pushed the next: k round trips in series on both sides (k = 2 on two cards,
// 4 on four at every row count, from the grid's size). The values and the order of every sum are unchanged.
template <bool TWO>
__global__ void k_ring_a2a_push(Ring r, float* x, int T, const unsigned* tagp, int call, RingMoe m) {
    const unsigned tag = *tagp * 256u + (unsigned) call + 1u;
    const int g = r.card, n = gridDim.x * blockDim.x, tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t slot = (size_t) MAXT * NP, set_sz = (size_t) NCARD * slot;
    const size_t off = (size_t) (tag & 1) * set_sz;
    unsigned long long* mine = (unsigned long long*) r.pk + off;
    diag_prog(r.err, tag);
    if constexpr (TWO) {
        constexpr int G = 4;
        for (int e0 = tid; e0 < T * NP; e0 += G * n) {
            if (*(volatile const unsigned*) r.err & 1u) return;   // a peer never came: do not wait out every element's timeout
            __half2 h[G];
#pragma unroll
            for (int j = 0; j < G; ++j) {
                const int e = e0 + j * n;
                if (e >= T * NP) break;
                const int t = e / NP, p = e % NP;
                const float2 v = ring_x2(x, m, T, t, p);
                if (!(fabsf(v.x) <= RING_MAX && fabsf(v.y) <= RING_MAX)) atomicOr(r.err, 2u);   // out of range (or NaN)
                h[j] = __floats2half2_rn(v.x, v.y);
                const unsigned long long word = (unsigned long long) *reinterpret_cast<const unsigned*>(&h[j]) | ((unsigned long long) tag << 32);
#pragma unroll
                for (int d = 0; d < NCARD; ++d)
                    if (d != g) *(volatile unsigned long long*) ((unsigned long long*) r.peer_pk[d] + off + (size_t) g * slot + e) = word;
            }
#pragma unroll
            for (int j = 0; j < G; ++j) {
                const int e = e0 + j * n;
                if (e >= T * NP) break;
                if (*(volatile const unsigned*) r.err & 1u) return;   // a take timed out: the thread's other elements do not wait their time-outs out too
                float2 sum = make_float2(0.f, 0.f);
#pragma unroll
                for (int src = 0; src < NCARD; ++src) {
                    const float2 w = src == g ? __half22float2(h[j]) : rtake(mine + (size_t) src * slot + e, tag, r.err, r.timeout, (unsigned) src, (unsigned) e);
                    sum = src == 0 ? w : add2(sum, w);
                }
                x[2 * e] = sum.x;
                x[2 * e + 1] = sum.y;
            }
        }
        return;
    }
    for (int e = tid; e < T * NP; e += n) {
        if (*(volatile const unsigned*) r.err & 1u) return;   // a peer never came: do not wait out every element's timeout
        const int t = e / NP, p = e % NP;
        const float2 v = ring_x2(x, m, T, t, p);
        if (!(fabsf(v.x) <= RING_MAX && fabsf(v.y) <= RING_MAX)) atomicOr(r.err, 2u);   // out of range (or NaN)
        const __half2 h = __floats2half2_rn(v.x, v.y);
        const unsigned long long word = (unsigned long long) *reinterpret_cast<const unsigned*>(&h) | ((unsigned long long) tag << 32);
#pragma unroll
        for (int d = 0; d < NCARD; ++d)
            if (d != g) *(volatile unsigned long long*) ((unsigned long long*) r.peer_pk[d] + off + (size_t) g * slot + e) = word;
        float2 sum = make_float2(0.f, 0.f);
#pragma unroll
        for (int src = 0; src < NCARD; ++src) {
            const float2 w = src == g ? __half22float2(h) : rtake(mine + (size_t) src * slot + e, tag, r.err, r.timeout, (unsigned) src, (unsigned) e);
            sum = src == 0 ? w : add2(sum, w);
        }
        x[2 * e] = sum.x;
        x[2 * e + 1] = sum.y;
    }
}

// ---------------------------------------------------------------- argmax
// a float's bits made monotone as unsigned (larger float, larger key)
__device__ __forceinline__ unsigned fkey(float v) {
    const unsigned u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float fkey_inv(unsigned k) { return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k); }

constexpr int AM_BLOCKS = 16, AM_THREADS = 256;
// keys[t] = max over the row of (fkey(value) << 32 | ~index): the largest value, the lower index on a tie. A NaN or
// inf in the row sets err bit 16 (the key order would pick a NaN's token, or skip it, as if it were a logit)
__global__ void __launch_bounds__(AM_THREADS) k_argmax(const float* __restrict__ logits, int rows, unsigned long long* keys, unsigned* err) {
    const int t = blockIdx.y, chunk = (rows + AM_BLOCKS - 1) / AM_BLOCKS, i0 = blockIdx.x * chunk, i1 = min(rows, i0 + chunk);
    const float* r = logits + (size_t) t * rows;
    unsigned long long best = 0;
    bool bad = false;
    for (int i = i0 + threadIdx.x; i < i1; i += AM_THREADS) {
        const float v = r[i];
        bad = bad || !isfinite(v);
        const unsigned long long k = (unsigned long long) fkey(v) << 32 | (unsigned) ~i;
        best = k > best ? k : best;
    }
    if (bad) atomicOr(err, 16u);
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        const unsigned long long ok = __shfl_xor_sync(0xffffffffu, best, o);
        best = ok > best ? ok : best;
    }
    if ((threadIdx.x & 31) == 0) atomicMax(&keys[t], best);
}
// decodes the keys into (value, index + v0) and leaves them at zero for the next argmax
__global__ void k_argmax_fin(unsigned long long* keys, int T, int v0, float* bv, int* bi) {
    const int t = threadIdx.x;
    if (t >= T) return;
    const unsigned long long k = keys[t];
    bv[t] = fkey_inv((unsigned) (k >> 32));
    bi[t] = (int) ~(unsigned) k + v0;
    keys[t] = 0ull;
}

// ---------------------------------------------------------------- the clocks' calibration and the test hooks
// one spin of 2^22 core-clock cycles (about 2 ms), timed on the steady counter too: the card's clock64 rate at load
__global__ void k_diag_clock(unsigned* err) {
    const long long w0 = diag_wall(), c0 = clock64();
    long long c1 = c0;
    while (c1 - c0 < (1LL << 22)) c1 = clock64();
    const long long w1 = diag_wall();
    *(volatile unsigned*) (err + DG_CAL0_CYC) = (unsigned) (c1 - c0);
    *(volatile unsigned*) (err + DG_CAL0_WALL) = (unsigned) (w1 - w0);
    *(volatile unsigned*) (err + DG_NOW_LO) = (unsigned) w1;
    *(volatile unsigned*) (err + DG_NOW_HI) = (unsigned) ((unsigned long long) w1 >> 32);
}
// FNX_DIAG_STALL, mode dev: holds the stream for `ticks` of the steady counter (the host allows 30 s at most; bounded by 2^38 cycles of the core clock,
// over a minute at any clock rate of these cards, so a steady counter that stands still cannot hold the card for ever)
__global__ void k_diag_spin(long long ticks) {
    const long long w0 = diag_wall(), c0 = clock64();
    while (diag_wall() - w0 < ticks && clock64() - c0 < (1LL << 38)) __nanosleep(64);
}
// FNX_DIAG_STALL, mode tag: *p += add in stream order (a card's tag word made wrong for one window)
__global__ void k_diag_poke(unsigned* p, unsigned add) { *p += add; }

}  // namespace

// ================================================================ launch wrappers
#if FNX_KPART == 0
void diag_clock(unsigned* err, void* st) {
    k_diag_clock<<<1, 1, 0, (cudaStream_t) st>>>(err);
    ck_launch("diag_clock");
}
void diag_spin(long long ticks, void* st) {
    k_diag_spin<<<1, 1, 0, (cudaStream_t) st>>>(ticks);
    ck_launch("diag_spin");
}
void diag_poke(unsigned* p, unsigned add, void* st) {
    k_diag_poke<<<1, 1, 0, (cudaStream_t) st>>>(p, add);
    ck_launch("diag_poke");
}
std::vector<Q8Twin>& q8_twins() {
    static std::vector<Q8Twin> v;
    return v;
}
void q8_twin(const void* q8, size_t bytes, int8_t* tq, float* td) { q8_twins().push_back({(const char*) q8, bytes, tq, td}); }
void q8_untwin(const void* q8) {
    auto& v = q8_twins();
    const auto it = std::find_if(v.begin(), v.end(), [&](const Q8Twin& w) { return w.q8 == (const char*) q8; });
    if (it == v.end()) throw std::runtime_error("q8_untwin: a buffer with no twin");
    v.erase(it);
}

void quantize_q8_1(const float* x, void* q, int n, int rows, void* st) {
    int8_t* tq;
    float* td;
    twin_of(q, n, tq, td);
    k_quantize<<<dim3((n / 32 + 7) / 8, rows), 256, 0, (cudaStream_t) st>>>(x, (block_q8_1*) q, n, tq, td);
    ck_launch("quantize_q8_1");
}

#endif
#if FNX_KPART == 1
namespace {
template <GgmlType type>
void mmvq_t(const DW& W, size_t w_stride, const Job* jobs, const int* njobs, int njobs_max, const void* y, float* out,
            int out_stride, cudaStream_t s) {
    constexpr int qk = qtraits<type>::qk, per_block = qtraits<type>::qi / vdr_of<type>();
    const int lanes = (W.n_in / qk) * per_block;   // lanes that keep every block of a row busy
    int G = per_block;
    while (G < lanes && G < 32) G *= 2;
    const int ys = W.n_in / 32;
#define FNX_G(g) k_mmvq<type, g><<<dim3((W.n_out + MMVQ_THREADS / g - 1) / (MMVQ_THREADS / g), njobs_max), MMVQ_THREADS, 0, s>>>( \
        (const char*) W.p, w_stride, W.row_bytes, W.n_out, W.n_in, jobs, njobs, njobs_max, (const block_q8_1*) y, ys, out, out_stride)
    if constexpr (per_block <= 2) { if (G == 2) { FNX_G(2); return; } }
    if constexpr (per_block <= 4) { if (G == 4) { FNX_G(4); return; } }
    if constexpr (per_block <= 8) { if (G == 8) { FNX_G(8); return; } }
    if constexpr (per_block <= 16) { if (G == 16) { FNX_G(16); return; } }
    FNX_G(32);
#undef FNX_G
}
}  // namespace

void mmvq(const DW& W, size_t w_stride, const Job* jobs, const int* njobs, int njobs_max, const void* y, float* out,
          int out_stride, void* st) {
    cudaStream_t s = (cudaStream_t) st;
    switch (W.type) {
        case T_Q8_0: mmvq_t<T_Q8_0>(W, w_stride, jobs, njobs, njobs_max, y, out, out_stride, s); break;
        case T_Q4_K: mmvq_t<T_Q4_K>(W, w_stride, jobs, njobs, njobs_max, y, out, out_stride, s); break;
        case T_Q5_K: mmvq_t<T_Q5_K>(W, w_stride, jobs, njobs, njobs_max, y, out, out_stride, s); break;
        case T_Q5_1: mmvq_t<T_Q5_1>(W, w_stride, jobs, njobs, njobs_max, y, out, out_stride, s); break;
        case T_Q6_K: mmvq_t<T_Q6_K>(W, w_stride, jobs, njobs, njobs_max, y, out, out_stride, s); break;
        default: throw std::runtime_error("mmvq: weight type " + std::to_string(W.type) + " not handled");
    }
    ck_launch("mmvq");
}

namespace {
template <int T>
void gemv_f32_t(const float* W, const float* x, float* y, int rows, cudaStream_t s) {
    // a block per row while that still gives every SM a block or two; a warp per row beyond
    if (rows <= 160) k_gemv_f32<T, 4><<<rows, 128, 0, s>>>(W, x, y, rows);
    else k_gemv_f32<T, 1><<<(rows + 7) / 8, 256, 0, s>>>(W, x, y, rows);
}
}  // namespace

void gemv_f32(const float* W, const float* x, float* y, int rows, int n, int T, void* st) {
    if (n != N) throw std::runtime_error("gemv_f32: rows of N columns only");
    cudaStream_t s = (cudaStream_t) st;
    switch (T) {
        case 1: gemv_f32_t<1>(W, x, y, rows, s); break;
        case 2: gemv_f32_t<2>(W, x, y, rows, s); break;
        case 3: gemv_f32_t<3>(W, x, y, rows, s); break;
        case 4: gemv_f32_t<4>(W, x, y, rows, s); break;
        case 5: gemv_f32_t<5>(W, x, y, rows, s); break;
        case 6: gemv_f32_t<6>(W, x, y, rows, s); break;
        case 7: gemv_f32_t<7>(W, x, y, rows, s); break;
        case 8: gemv_f32_t<8>(W, x, y, rows, s); break;
        case 9: gemv_f32_t<9>(W, x, y, rows, s); break;
        case 10: gemv_f32_t<10>(W, x, y, rows, s); break;
        case 11: gemv_f32_t<11>(W, x, y, rows, s); break;
        case 12: gemv_f32_t<12>(W, x, y, rows, s); break;
        case 13: gemv_f32_t<13>(W, x, y, rows, s); break;
        case 14: gemv_f32_t<14>(W, x, y, rows, s); break;
        case 15: gemv_f32_t<15>(W, x, y, rows, s); break;
        case 16: gemv_f32_t<16>(W, x, y, rows, s); break;
        default: throw std::runtime_error("gemv_f32: T out of range");
    }
    ck_launch("gemv_f32");
}

#endif
#if FNX_KPART == 3
// (the last two floats: the barrier counters, of the card's grid and, of its second grid for wider reads)
size_t hc_scratch_floats(int T) { return (size_t) H_NS * T * H_ROWS + (size_t) T * H_NS + (size_t) T * LR / 4 + (size_t) T * LR / 32 + 2; }

// FNX_HC1=R: windows of up to R rows (at most 8) take k_hc's one-barrier form (default 3: kbench 1-2 us faster a read at
// 1-3 rows, the same at 4-5, slower at 8). Read once; the engine reads it when made, so a bad value fails at start
int hc1_rows() {
    static const int rows = [] {
        // 8 on HIP in a FAST build (FNX_HC1_WIDE): on an R9700 the one-barrier form is 2-3.4 us a read faster at 4-8 rows too (kbench between dense
        // launches, 4 / 5 / 6 / 8 rows: 33.2 / 35.4 / 38.9 / 48.9 us against 36.6 / 38.2 / 41.5 / 50.8); the same bits either way
#if FNX_HIP
        static const long long def = fast_opt("FNX_HC1_WIDE") ? 8 : 3;
#else
        static const long long def = 3;
#endif
        const long long r = env_int("FNX_HC1", def);
        if (r < 0 || r > 8) throw std::runtime_error("FNX_HC1: 0 (off) .. 8 rows, got " + std::to_string(r));
        return (int) r;
    }();
    return rows;
}

int hc_blocks() { return H_BLOCKS; }
#if !(FNX_MM_HC && FNX_HIP)
int hc_mm_grid() { return 0; }
int hc_sum_rows() { return 0; }
#endif
#if FNX_KBENCH
int hc_stub_ = 0;
void hc_set_stub(int bits) { hc_stub_ = bits; }
#endif

// the read's grid on each card (0: hc_prepare not run there)
constexpr int HC_MAXDEV = 64;
int hc_grid_[HC_MAXDEV] = {};
// FNX_HC_GRID2 (HIP, on in a FAST build): a second, larger grid for reads of HC_BIG_ROWS rows and more (0: none). An R9700 reports 32 "SMs" (WGPs) and
// the read's 80 shares took three turns on 32 blocks, half the blocks idle in the third. With the kernel at ~100 registers a wave (hc.cuh, the inject rows)
// two 24-wave blocks fit a WGP, and 48 blocks take the shares in two turns: kbench between dense launches, 32 -> 48 blocks at 4 / 5 / 6 / 8 / 16 rows:
// 33.3 -> 31.9, 35.9 -> 33.1, 39.4 -> 35.2, 49.6 -> 42.9, 85.3 -> 72.6 us a read; slower at 1 row (21.8 -> 24.4) and equal at 3, so the narrow reads
// keep the card's grid. The same sums in the same order whatever the grid, so the same bits. Each grid has its own barrier counter (a counter counts in
// generations of ONE grid). hc_prepare tries the larger grid first (hc_probe): if its blocks are not all resident on this card it is not used.
constexpr int HC_BIG_ROWS = 4;
int hc_grid_big_[HC_MAXDEV] = {};

#if FNX_MM_HC && FNX_HIP
// the ticket queue of each card's reads (hc_mm.cuh), made and zeroed by hc_prepare; and the grid of k_hc_mm, FNX_MM_HC_GRID blocks
// (default 96: any grid gives the same sums; blocks past the tickets leave at once)
unsigned* hc_mm_q_[HC_MAXDEV] = {};
int hc_sum_rows() {
    static const int rows = [] {
        const long long r = env_int("FNX_HC_SUM", APPROX ? 8 : 0);
        if (r < 0 || r > MAXT) throw std::runtime_error("FNX_HC_SUM: 0 (off) .. " + std::to_string(MAXT) + " rows, got " + std::to_string(r));
        return (int) r;
    }();
    return rows;
}
int hc_mm_grid() {
    static const int grid = [] {
        const long long g = env_int("FNX_MM_HC_GRID", 96);
        if (g < 1 || g > 256) throw std::runtime_error("FNX_MM_HC_GRID: 1..256 blocks, got " + std::to_string(g));
        return (int) g;
    }();
    return grid;
}
#endif

int hc_dev() {
    int dev = -1;
    const cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) throw std::runtime_error(std::string("hc: cudaGetDevice: ") + cudaGetErrorString(e));
    if (dev < 0 || dev >= HC_MAXDEV) throw std::runtime_error("hc: card " + std::to_string(dev) + " past the " + std::to_string(HC_MAXDEV) + " it tracks");
    return dev;
}

int hc_grid() {
    const int dev = hc_dev();
    if (!hc_grid_[dev]) throw std::runtime_error("hc_read: hc_prepare was not run on card " + std::to_string(dev));
    return hc_grid_[dev];
}

// are `grid` blocks of the read resident together on the current card? The widest and a narrow read of both forms run once on zeros with a
// short barrier time-out (20 M cycles, ~10 ms): true when no barrier timed out. (A grid that is not resident never passes a barrier: every read would
// time out, so the answer is not a matter of timing.) Uses the scratch's second counter; the caller zeroes the counters after it.
// (the probe's buffers and stream are a struct of their own, HcProbeBuf, so that a caller that holds another kernel resident
// through the probe can allocate before and free after: cudaFree waits for the whole card.)
struct HcProbeBuf {
    std::vector<void*> p;
    cudaStream_t st = nullptr;
    float *R = nullptr, *w_norm = nullptr, *mixed = nullptr, *inject = nullptr;
    Q8S down, up;
    uint16_t* w_inj = nullptr;
    unsigned* err = nullptr;
    void* dz(size_t bytes) {
        void* q = nullptr;
        if (cudaMalloc(&q, bytes) != cudaSuccess) throw std::runtime_error("hc_probe: no memory for the probe");
        p.push_back(q);
        if (cudaMemset(q, 0, bytes) != cudaSuccess) throw std::runtime_error("hc_probe: the probe's buffers could not be zeroed");
        return q;
    }
    HcProbeBuf() {
        try {
            R = (float*) dz((size_t) MAXT * D * 4);
            w_norm = (float*) dz((size_t) D * 4);
            down.q = (int8_t*) dz((size_t) LR * D), down.d = (uint16_t*) dz((size_t) LR * D / 16);
            up.q = (int8_t*) dz((size_t) D * LR), up.d = (uint16_t*) dz((size_t) D * LR / 16);
            w_inj = (uint16_t*) dz((size_t) HC * D * 2);
            mixed = (float*) dz((size_t) MAXT * N * 4);
            inject = (float*) dz((size_t) MAXT * HC * 4);
            err = (unsigned*) dz(ERR_WORDS * sizeof(unsigned));   // a block: a barrier that times out here writes its record (diag.cuh)
            if (cudaStreamCreate(&st) != cudaSuccess) throw std::runtime_error("hc_probe: no stream");
        } catch (...) {
            release();
            throw;
        }
    }
    void release() {
        if (st) cudaStreamDestroy(st);
        st = nullptr;
        for (void* q : p) cudaFree(q);
        p.clear();
    }
    ~HcProbeBuf() { release(); }
    HcProbeBuf(const HcProbeBuf&) = delete;
    HcProbeBuf& operator=(const HcProbeBuf&) = delete;
    // the scratch's two barrier counters to zero, on the probe's stream
    void zero_counters(float* scratch) {
        if (cudaMemsetAsync(scratch + hc_scratch_floats(MAXT) - 2, 0, 2 * sizeof(unsigned), st) != cudaSuccess || cudaStreamSynchronize(st) != cudaSuccess)
            throw std::runtime_error("hc_probe: the barrier counters could not be zeroed");
    }
    // held: another kernel stays resident through the probe (mall_pf_hc_probe): err is zeroed first (a second run) and the result is read on the
    // probe's stream, not with a call on the null stream. Not held: hc_prepare's probe, the calls it always made.
    // (pinned: a pinned host word for the held form's result)
    bool run(float* scratch, int grid, bool held, unsigned* pinned = nullptr) {
        if (held && (!pinned || cudaMemsetAsync(err, 0, ERR_WORDS * sizeof(unsigned), st) != cudaSuccess)) return false;
        const auto args = [&](int T) {
            HcArgs a{R, nullptr, nullptr, w_norm, down, up, w_inj, 1e-6f, mixed, inject, nullptr, nullptr, nullptr, nullptr, nullptr, err, nullptr};
            a.part = scratch;
            a.ssp = a.part + (size_t) H_NS * T * H_ROWS;
            a.loq = (int8_t*) (a.ssp + (size_t) T * H_NS);
            a.lod = a.ssp + (size_t) T * H_NS + (size_t) T * LR / 4;
            a.bar = (unsigned*) (scratch + hc_scratch_floats(MAXT) - 2);
            a.timeout = 20000000LL;
            return a;
        };
        k_hc<MAXT, false, true><<<grid, H_DT, 0, st>>>(args(MAXT));
        k_hc<8, true, true><<<grid, H_DT, 0, st>>>(args(8));
        k_hc<4, false, true><<<grid, H_DT, 0, st>>>(args(4));
        k_hc<4, true, true><<<grid, H_DT, 0, st>>>(args(4));
        unsigned e = 1;
        if (held) {
            *pinned = 1;
            const bool ran = cudaGetLastError() == cudaSuccess && cudaMemcpyAsync(pinned, err, sizeof e, cudaMemcpyDeviceToHost, st) == cudaSuccess &&
                             cudaStreamSynchronize(st) == cudaSuccess;
            return ran && *pinned == 0;
        }
        const bool ran = cudaGetLastError() == cudaSuccess && cudaStreamSynchronize(st) == cudaSuccess &&
                         cudaMemcpy(&e, err, sizeof e, cudaMemcpyDeviceToHost) == cudaSuccess;
        return ran && e == 0;
    }
};
bool hc_probe(float* scratch, int grid) {
    HcProbeBuf buf;
    return buf.run(scratch, grid, false);
}

int hc_prepare(float* scratch, int cap) {
    const int dev = hc_dev();
    int sms = 0;
    const cudaError_t e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (e != cudaSuccess) throw std::runtime_error(std::string("hc_prepare: SM count: ") + cudaGetErrorString(e));
    if (sms <= 0) throw std::runtime_error("hc_prepare: card " + std::to_string(dev) + " reports " + std::to_string(sms) + " SMs");
    if (cap < 0 || cap > H_BLOCKS) throw std::runtime_error("hc_prepare: cap 1.." + std::to_string(H_BLOCKS) + " (0: none), got " + std::to_string(cap));
    hc_grid_[dev] = std::min(sms, cap ? cap : H_BLOCKS);
#if FNX_HIP
    // FNX_HC_GRID=n (screen): n blocks of the read resident at once instead of one per multiprocessor. An R9700 reports 32 (one per WGP of two CUs), so the 80 blocks of work
    // ran in 2-3 turns each; several blocks fit one WGP, and a barrier that waits too long sets err bit 4 rather than hang
    if (const long long g = env_int("FNX_HC_GRID", 0); g > 0) hc_grid_[dev] = (int) std::min<long long>(g, cap ? cap : H_BLOCKS);
#endif
    // a counter left by another grid sits mid-generation for this one: its next barrier would let blocks through early
    const unsigned zero[2] = {0, 0};
    const auto zero_counters = [&] {
        const cudaError_t z = cudaMemcpy(scratch + hc_scratch_floats(MAXT) - 2, zero, sizeof zero, cudaMemcpyHostToDevice);
        if (z != cudaSuccess) throw std::runtime_error(std::string("hc_prepare: the barrier counters: ") + cudaGetErrorString(z));
    };
    zero_counters();
    hc_grid_big_[dev] = 0;
#if FNX_MM_HC && FNX_HIP
    {
        if (!hc_mm_q_[dev] && cudaMalloc((void**) &hc_mm_q_[dev], HM_Q * sizeof(unsigned)) != cudaSuccess) {
            hc_mm_q_[dev] = nullptr;
            throw std::runtime_error("hc_prepare: no memory for the read's ticket queue");
        }
        if (cudaMemset(hc_mm_q_[dev], 0, HM_Q * sizeof(unsigned)) != cudaSuccess) throw std::runtime_error("hc_prepare: the ticket queue could not be zeroed");
    }
#endif
#if FNX_HIP
    if (!cap && env_int("FNX_HC_GRID", 0) <= 0 && fast_opt("FNX_HC_GRID2")) {
        static const long long big = env_int("FNX_HC_GRID2_BLOCKS", 48);   // once per process: a counter counts in generations of one grid
        if (big > hc_grid_[dev] && big < H_BLOCKS) {
            if (hc_probe(scratch, (int) big)) hc_grid_big_[dev] = (int) big;
            else std::fprintf(stderr, "fnx: card %d: a grid of %lld blocks of the hc read was not resident: reads of every width keep %d\n", dev, big, hc_grid_[dev]);
            zero_counters();
        }
    }
#endif
    return hc_grid_[dev];
}

void hc_read(float* R, const float* bo, const float* inj_prev, const float* w_norm, const Q8S& w_down, const Q8S& w_up,
             const uint16_t* w_inj, float eps, float* mixed, float* inject, float* scratch, unsigned* err, int T, void* st,
             void* q8
#if FNX_MALL_PF
             ,
             unsigned* pf_word
#endif
             ,
             const HcSumReq* sum) {
    if (sum) {   // the window's sum of bo first: as this read's first tickets where its kernel takes them, else a launch of its own
        bool fold = false;
#if FNX_MM_HC && FNX_HIP
        static const tk::Family* const tkf0 = tk::family("hc_mm");
        const Ring& sr = *sum->ring;
        fold = mm_hc() && !tkf0 && bo && sum->x == bo && sr.onehop && !sr.peer && !sr.f32 && T >= 1 && T <= MAXT && T <= sr.rows && T <= hc_sum_rows()
               && ((uintptr_t) sum->x & 63) == 0;   // (a sum ticket's 2048 bytes of x are whole cache lines: no block loads a line another block's ticket stores)
#endif
        if (!fold) {
            ring_sum(*sum->ring, sum->x, T, sum->tag, sum->call, st, sum->m);
            sum = nullptr;
        }
    }
    HcArgs a{R, bo, inj_prev, w_norm, w_down, w_up, w_inj, eps, mixed, inject, nullptr, nullptr, nullptr, nullptr, nullptr, err,
             (block_q8_1*) q8};
#if FNX_MALL_PF
    a.pf = pf_word;
#endif
    // FNX_HC_TIMEOUT_CYCLES: the hc read's barrier / gate time-out in clock64 cycles (unset: HcArgs' 2,000,000,000, as before)
    static const long long hc_timeout = env_int("FNX_HC_TIMEOUT_CYCLES", 0);
    if (hc_timeout < 0) throw std::runtime_error("FNX_HC_TIMEOUT_CYCLES: want a number of cycles > 0");
    if (hc_timeout > 0) a.timeout = hc_timeout;
    a.part = scratch;
    a.ssp = a.part + (size_t) H_NS * T * H_ROWS;
    a.loq = (int8_t*) (a.ssp + (size_t) T * H_NS);
    a.lod = a.ssp + (size_t) T * H_NS + (size_t) T * LR / 4;
    // the barrier's counter: the last float of a scratch sized for MAXT rows, so it stays put whatever T is (the one before it: the second grid's)
    const int dev_big = hc_grid_big_[hc_dev()];
    const bool use_big = dev_big > 0 && T >= HC_BIG_ROWS;
    a.bar = (unsigned*) (scratch + hc_scratch_floats(MAXT) - (use_big ? 2 : 1));
#if FNX_KBENCH
    a.stub = hc_stub_;
#endif
    if (q8) twin_of(q8, N, a.tq, a.td);
    cudaStream_t s = (cudaStream_t) st;
#if FNX_MM_HC && FNX_HIP
    if (mm_hc()) {   // the ticket queue (no barrier, no grid to be resident)
        unsigned* const q = hc_mm_q_[hc_dev()];
        if (!q) throw std::runtime_error("hc_read: hc_prepare was not run on this card (the ticket queue)");
        const int mg = hc_mm_grid();
        if (sum) {
            const HmRingSum sm{*sum->ring, sum->x, sum->tag, sum->call, sum->m};
#define FNX_HCMS(t) k_hc_mm_sum<t><<<mg, HM_T, 0, s>>>(a, q, sm)
            switch (T) {
                case 1: FNX_HCMS(1); break;
                case 2: FNX_HCMS(2); break;
                case 3: FNX_HCMS(3); break;
                case 4: FNX_HCMS(4); break;
                case 5: FNX_HCMS(5); break;
                case 6: FNX_HCMS(6); break;
                case 7: FNX_HCMS(7); break;
                case 8: FNX_HCMS(8); break;
                case 9: FNX_HCMS(9); break;
                case 10: FNX_HCMS(10); break;
                case 11: FNX_HCMS(11); break;
                case 12: FNX_HCMS(12); break;
                case 13: FNX_HCMS(13); break;
                case 14: FNX_HCMS(14); break;
                case 15: FNX_HCMS(15); break;
                case 16: FNX_HCMS(16); break;
                default: throw std::runtime_error("hc_read: T out of range");
            }
#undef FNX_HCMS
            ck_launch("hc_read (with the sum)");
            return;
        }
#define FNX_HCM(t) k_hc_mm<t><<<mg, HM_T, 0, s>>>(a, q)
        static const tk::Family* const tkf = tk::family("hc_mm");   // the generator's variant: a kernel per row count
        if (tkf) {
            const tk::Kern& k = tkf->kern(T);
            FNX_TK_LAUNCH(k, k_hc_mm<1>, dim3(mg), dim3(k.threads), 0, s, a, q);
        } else
        switch (T) {
            case 1: FNX_HCM(1); break;
            case 2: FNX_HCM(2); break;
            case 3: FNX_HCM(3); break;
            case 4: FNX_HCM(4); break;
            case 5: FNX_HCM(5); break;
            case 6: FNX_HCM(6); break;
            case 7: FNX_HCM(7); break;
            case 8: FNX_HCM(8); break;
            case 9: FNX_HCM(9); break;
            case 10: FNX_HCM(10); break;
            case 11: FNX_HCM(11); break;
            case 12: FNX_HCM(12); break;
            case 13: FNX_HCM(13); break;
            case 14: FNX_HCM(14); break;
            case 15: FNX_HCM(15); break;
            case 16: FNX_HCM(16); break;
            default: throw std::runtime_error("hc_read: T out of range");
        }
#undef FNX_HCM
        ck_launch("hc_read");
        return;
    }
#endif
    const bool one = T <= hc1_rows();
    const int grid = use_big ? dev_big : hc_grid();
    switch (T) {
        case 1: hc_launch<1>(a, s, one, grid); break;
        case 2: hc_launch<2>(a, s, one, grid); break;
        case 3: hc_launch<3>(a, s, one, grid); break;
        case 4: hc_launch<4>(a, s, one, grid); break;
        case 5: hc_launch<5>(a, s, one, grid); break;
        case 6: hc_launch<6>(a, s, one, grid); break;
        case 7: hc_launch<7>(a, s, one, grid); break;
        case 8: hc_launch<8>(a, s, one, grid); break;
        case 9: hc_launch<9>(a, s, one, grid); break;
        case 10: hc_launch<10>(a, s, one, grid); break;
        case 11: hc_launch<11>(a, s, one, grid); break;
        case 12: hc_launch<12>(a, s, one, grid); break;
        case 13: hc_launch<13>(a, s, one, grid); break;
        case 14: hc_launch<14>(a, s, one, grid); break;
        case 15: hc_launch<15>(a, s, one, grid); break;
        case 16: hc_launch<16>(a, s, one, grid); break;
        default: throw std::runtime_error("hc_read: T out of range");
    }
    ck_launch("hc_read");
}

// the grid a read of T rows takes on the current card
int hc_grid_for(int T) {
    const int big = hc_grid_big_[hc_dev()];
    return big > 0 && T >= HC_BIG_ROWS ? big : hc_grid();
}

// are the hc read's grids in use on the current card resident beside `wgs` resident workgroups of the persistent cache prefetcher?
// hc_prepare's probe ran with the card to itself. Here an idle prefetcher (k_mall_pf_run on `side`: it polls a count that does not move) is held
// resident, each grid in use is probed again (HcProbeBuf::run), and the stop word ends the prefetcher. The card must be idle; `work` is an idle
// stream of the card for the few small kernels and copies (no call here but the last frees waits for the whole card: cudaFree does, so the probe's
// buffers are allocated before the prefetcher starts and freed after it ended). Returns 0: resident; bit 0: the card's grid is not, bit 1: the
// larger grid is not, bit 2: the probe says nothing (the prefetcher's blocks were not all resident in time, or it ended before the probe did).
// Leaves the barrier counters and the words at zero.
// (second form, after the first gave no answer on the card: the words are read and the stop is written by kernels of the card at device scope
// (mall_pf_peek / mall_pf_poke), as the prefetcher reads and writes them, not by host copies; the wait for the blocks is 250 ms of wall time, not a
// number of reads; `info` says which step failed and how long each took. `work` and the probe's own stream must not share the prefetcher's
// hardware queue (mall_pf_stream): a launch on a shared queue waits for the prefetcher's end, that is for its time-out.)
int mall_pf_hc_probe(float* scratch, const PfSet* table, int n, unsigned* words, int wgs, void* side, void* work, PfProbeInfo* info, bool query) {
    const int dev = hc_dev();
    cudaStream_t ws = (cudaStream_t) work, ss = (cudaStream_t) side;
    HcProbeBuf buf;
    unsigned* out = (unsigned*) buf.dz(4 * sizeof(unsigned));   // mall_pf_peek's word
    const auto ok = [](cudaError_t e, const char* what) {
        if (e != cudaSuccess) throw std::runtime_error(std::string("mall_pf_hc_probe: ") + what + ": " + cudaGetErrorString(e));
    };
    struct Pinned {   // two pinned host words: the small copies' and the probes' result
        unsigned* v = nullptr;
        ~Pinned() {
            if (v) cudaFreeHost(v);
        }
    } hw;
    ok(cudaMallocHost((void**) &hw.v, 2 * sizeof(unsigned)), "pinned words");
    using clk = std::chrono::steady_clock;
    const auto ms_since = [](clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); };
    const auto get = [&](int idx) {   // words[idx] by a device-scope load of a kernel on the work stream, then that kernel's word
        hw.v[0] = 0;
        mall_pf_peek(words, idx, out, ws);
        ok(cudaMemcpyAsync(hw.v, out, sizeof(unsigned), cudaMemcpyDeviceToHost, ws), "read");
        ok(cudaStreamSynchronize(ws), "read sync");
        return hw.v[0];
    };
    const auto wipe = [&] {
        ok(cudaMemsetAsync(words, 0, MALL_PF_WORDS * sizeof(unsigned), ws), "words");
        ok(cudaStreamSynchronize(ws), "words sync");
    };
    wipe();
    buf.zero_counters(scratch);
    PfRun idle;
    idle.sets = table, idle.words = words, idle.n = n, idle.base = 0x40000000u, idle.timeout = mall_pf_ticks(4000.0);
    mall_pf_run(idle, wgs, side, query);
    int bad = 0;
    PfProbeInfo pi;
    bool up = false;   // every block of the prefetcher adds one to words[4] when it starts
    const auto t0 = clk::now();
    do {
        pi.started = get(4);
        ++pi.reads;
        up = pi.started >= (unsigned) wgs;
    } while (!up && ms_since(t0) < 250.0);
    pi.ms_up = ms_since(t0);
    const auto t1 = clk::now();
    if (up) {
        if (!buf.run(scratch, hc_grid(), true, hw.v + 1)) bad |= 1;
        buf.zero_counters(scratch);
        if (hc_grid_big_[dev] > 0 && !buf.run(scratch, hc_grid_big_[dev], true, hw.v + 1)) bad |= 2;
    } else
        bad |= 4;
    pi.ms_probe = ms_since(t1);
    const auto t2 = clk::now();
    mall_pf_poke(words, 3, 1u, ws);
    ok(cudaStreamSynchronize(ws), "stop sync");
    ok(cudaStreamSynchronize(ss), "prefetcher sync");
    pi.ms_stop = ms_since(t2);
    pi.status = get(1);
    if (pi.status & 1u) bad |= 4;   // it timed out before the stop: it was not resident through the probe
    wipe();
    buf.zero_counters(scratch);
    if (info) *info = pi;
    return bad;
}

std::string mall_pf_probe_text(int bad, const PfProbeInfo& pi, int wgs) {
    char t[512];
    std::string s = bad == 0 ? " the hc read's grids are resident;" : "";
    if (bad & 1) s += " the hc read's grid is NOT resident;";
    if (bad & 2) s += " the hc read's larger grid is NOT resident;";
    if (bad & 4) {
        if (pi.started < (unsigned) wgs)
            std::snprintf(t, sizeof t, " no answer: %u of %d prefetcher blocks had started after %.0f ms (%d reads);", pi.started, wgs, pi.ms_up, pi.reads);
        else
            std::snprintf(t, sizeof t, " no answer: the idle prefetcher timed out before its stop (status %u);", pi.status);
        s += t;
        if (pi.ms_up > 1000.0 || pi.ms_probe > 1000.0)
            s += " a step took over a second: possibly a launch that waited for the prefetcher's end on a shared hardware queue (mall_pf_stream);";
        else if (pi.ms_stop > 1000.0)
            s += " the stop took over a second: the stop kernel ran late or the prefetcher did not see its word;";
    }
    std::snprintf(t, sizeof t, " (blocks up in %.2f ms and %d reads, probes %.1f ms, stop to end %.2f ms)", pi.ms_up, pi.reads, pi.ms_probe, pi.ms_stop);
    return s + t;
}

void hc_preset_barrier(float* scratch, unsigned barriers_left) {
    const unsigned g = (unsigned) hc_grid(), mod = hc_bar_mod(g);
    if (barriers_left == 0 || barriers_left > mod / g)
        throw std::runtime_error("hc_preset_barrier: barriers_left must be 1.." + std::to_string(mod / g));
    const unsigned v = mod - barriers_left * g;
    const cudaError_t e = cudaMemcpy(scratch + hc_scratch_floats(MAXT) - 1, &v, sizeof v, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) throw std::runtime_error(std::string("hc_preset_barrier: ") + cudaGetErrorString(e));
}

#endif
#if FNX_KPART == 0
void route(const float* logits, int T, int* topi, float* topw, Job* jobs_gu, Job* jobs_down, int* njobs, void* st) {
    k_topk<<<T, NEXP, 0, (cudaStream_t) st>>>(logits, topi, topw);
    k_jobs<<<1, 1, 0, (cudaStream_t) st>>>(topi, T, jobs_gu, jobs_down, njobs);
    ck_launch("route");
}

void swiglu_q8_1(const float* g, const float* u, void* q, int n, int rows, void* st) {
    k_swiglu_q8<<<dim3((n / 32 + 7) / 8, rows), 256, 0, (cudaStream_t) st>>>(g, u, (block_q8_1*) q, n);
    ck_launch("swiglu_q8_1");
}

void moe_combine(const float* part, const float* topw, const float* sh, const float* sh_logit, float* out, int T, void* st) {
    k_combine<<<dim3((N + 255) / 256, T), 256, 0, (cudaStream_t) st>>>(part, topw, sh, sh_logit, out);
    ck_launch("moe_combine");
}

void gdn_conv(const float* qkv, float* hist, const float* w, float* out, int ch, int T, void* st) {
    k_gdn_conv<<<(ch + 255) / 256, 256, 0, (cudaStream_t) st>>>(qkv, hist, w, out, ch, T);
    ck_launch("gdn_conv");
}
void gdn_qk_norm(float* conv, int nk, int ch, int T, void* st) {
    k_gdn_qk_norm<<<dim3(2 * nk, T), HS, 0, (cudaStream_t) st>>>(conv, nk, ch, 1e-6f);
    ck_launch("gdn_qk_norm");
}
void gdn_gates(const float* alpha, const float* beta_raw, const float* dt_bias, const float* ssm_a, float* g, float* beta, int nv,
               int T, void* st) {
    k_gdn_gates<<<(nv * T + 127) / 128, 128, 0, (cudaStream_t) st>>>(alpha, beta_raw, dt_bias, ssm_a, g, beta, nv, T);
    ck_launch("gdn_gates");
}
void gdn_step(float* S, const float* conv, const float* g, const float* beta, float* o, int nk, int nv, int ch, int T, void* st) {
    k_gdn_step<<<nv, HS * 4, 0, (cudaStream_t) st>>>(S, conv, g, beta, o, nk, nv, ch, T);
    ck_launch("gdn_step");
}
void gdn_out_norm(const float* o, const float* z, const float* w, float* y, int nv, int T, float eps, void* st) {
    k_gdn_out_norm<<<dim3(nv, T), HS, 0, (cudaStream_t) st>>>(o, z, w, y, nv, eps);
    ck_launch("gdn_out_norm");
}

#endif
#if FNX_KPART == 5
void attn_prep(float* qg, float* k, const float* q_norm, const float* k_norm, int nq, int nkv, const int* pos0, int T, float eps,
               Rope base, void* st) {
    k_attn_prep<<<dim3(nq + nkv, T), HD, 0, (cudaStream_t) st>>>(qg, k, q_norm, k_norm, nq, nkv, pos0, eps, base, LaneRows{});
    ck_launch("attn_prep");
}
void attn_prep_lanes(float* qg, float* k, const float* q_norm, const float* k_norm, int nq, int nkv, const LaneRows& lr, int T, float eps,
                     Rope base, void* st) {
    if (!lr.pos || !lr.tab) throw std::runtime_error("attn_prep_lanes: no lane rows");
    k_attn_prep<<<dim3(nq + nkv, T), HD, 0, (cudaStream_t) st>>>(qg, k, q_norm, k_norm, nq, nkv, nullptr, eps, base, lr);
    ck_launch("attn_prep_lanes");
}
int qsa_maxsel() { return QSA_MAXSEL; }
// the raw-key ring holds a prompt chunk and the block before it (positions mod its rows): twice the chunk, 2048 at least
namespace {
int qsa_ring = 2048;
}
int qsa_ring_rows() { return qsa_ring; }

#if FNX_ATT_MM
// (attn_mm.cuh): FNX_ATT_MM=0 runs the kernels of a build without the option; kbench switches with att_mm_force
namespace {
int g_att_mm_force = -1;
bool att_mm_on() {
    static const bool env = env_int("FNX_ATT_MM", 1) != 0;
    return g_att_mm_force >= 0 ? g_att_mm_force != 0 : env;
}
// the attention part by k_attn_mm: int8 pages only. FNX_KV8=0 (the engine's f16 pages) keeps k_attn_part; the choice is the process's (kv_q8), so
// that attn_part_lanes and attn_finish agree, and pages of the other kind than kv_q8 says are refused (att_mm_pages)
bool att_mm_attn() { return att_mm_on() && kv_q8(); }
// its splits of a row: FNX_ATT_MM_SPLITS (default 8), the same for every window, so that a row's sums do not depend on the other rows of its
// window: a sequence's greedy text is then the same alone and in a batch (2026-11-14: with the count by the rows, 4..16, the 4-user texts of
// agree.py differed from the 1-user texts; 8 and 16 give equal texts, and 8 costs the least over 1..16 rows). 0: by the rows (about a block a
// compute unit; up to 5 us a layer faster under 5 rows, but batch-dependent sums)
int att_mm_splits(int T) {
    static const long long env = env_int("FNX_ATT_MM_SPLITS", std::min(8, ATT_SPLITS));   // (ATT_SPLITS: the partials' slots a head; a build with fewer than 8 takes them all)
    if (env < 0 || env > ATT_SPLITS) throw std::runtime_error("FNX_ATT_MM_SPLITS: 0 (by the rows) or 1.." + std::to_string(ATT_SPLITS));
    return env ? (int) env : std::min(std::min(16, ATT_SPLITS), std::max(4, 64 / std::max(T, 1)));
}
// FNX_ATT_MM_SCORE_TC=1: one sequence's window scores its blocks with the prompt path's matrix-product kernel (k_idx_score_tc: f16 products, the
// scores differ from k_idx_score's in the last bits, so a near tie can select another block). Default 0: k_idx_score_mm, the same bits
bool att_mm_score_tc() {
    static const bool env = env_int("FNX_ATT_MM_SCORE_TC", 0) != 0;
    return env;
}
void att_mm_pages(const KvPages& g, const char* who) {
    if (!g.q8 || g.rows != AM_PROWS) throw std::runtime_error(std::string(who) + ": FNX_ATT_MM reads int8 pages of " + std::to_string(AM_PROWS) + " rows");
}
}  // namespace
void att_mm_force(int on) { g_att_mm_force = on; }
#if FNX_KBENCH
void kb_idx_score(const float* q, const uint16_t* pooled, const KvPages& g, float* scores, int max_pools, const int* pos0, int T, int mm, void* st) {
    cudaStream_t s = (cudaStream_t) st;
    const __half* pl = reinterpret_cast<const __half*>(pooled);
    if (mm == 2)   // the prompt path's scorer on the window's rows (FNX_ATT_MM_SCORE_TC)
        k_idx_score_tc<<<dim3((max_pools + QSC_KEYS - 1) / QSC_KEYS, (T + QSC_ROWS - 1) / QSC_ROWS), 256, 0, s>>>(q, pl, g, scores, max_pools, pos0, T);
    else if (mm) k_idx_score_mm<8><<<dim3((max_pools + 255) / 256, (T + 7) / 8), 256, 0, s>>>(q, pl, g, scores, max_pools, pos0, LaneRows{}, T);
    else k_idx_score<<<dim3((max_pools + 255) / 256, T), 256, 0, s>>>(q, pl, g, scores, max_pools, pos0, LaneRows{});
    ck_launch("kb_idx_score");
}
void kb_idx_select(const float* scores, int* cells, int* n_cells, int max_pools, const int* pos0, int T, int card, int mm, void* st) {
    cudaStream_t s = (cudaStream_t) st;
    if (mm) k_idx_select_mm<<<T, QS_NT, 0, s>>>(scores, cells, n_cells, max_pools, pos0, LaneRows{});
    else k_idx_select<<<T, QSA_SEL_THREADS, 0, s>>>(scores, cells, n_cells, max_pools, pos0, card, LaneRows{});
    ck_launch("kb_idx_select");
}
#endif
#endif
void set_prompt_rows(int rows) { qsa_ring = std::max(2048, 2 * rows); }

void qsa_index(const float* y, const float* qn, const float* kn, uint16_t* ring, uint16_t* pooled, const KvPages& g, float* q, const int* pos0, int T,
               float eps, Rope base, void* st) {
    cudaStream_t s = (cudaStream_t) st;
    if (T + QSA_POOL - 1 > qsa_ring) throw std::runtime_error("qsa_index: more rows than the raw-key ring holds");
    __half* r = reinterpret_cast<__half*>(ring);
    __half* pl = reinterpret_cast<__half*>(pooled);
    k_idx_prep<<<dim3(T, QSA_HEADS + 1), QSA_DIM, 0, s>>>(y, qn, r, qsa_ring, q, pos0, eps, base, LaneRows{});
    k_idx_pool<<<T, QSA_DIM, 0, s>>>(r, qsa_ring, kn, pl, g, pos0, eps, base, LaneRows{});
    ck_launch("qsa_index");
}

void qsa_index_lanes(const float* y, const float* qn, const float* kn, uint16_t* pooled, const KvPages& g, float* q, const LaneRows& lr, int T,
                     float eps, Rope base, void* st) {
    cudaStream_t s = (cudaStream_t) st;
    if (!lr.pos || !lr.pt || !lr.tab || !lr.ring) throw std::runtime_error("qsa_index_lanes: no lane rows");
    if (T + QSA_POOL - 1 > qsa_ring) throw std::runtime_error("qsa_index_lanes: more rows than the raw-key ring holds");
    __half* pl = reinterpret_cast<__half*>(pooled);
    k_idx_prep<<<dim3(T, QSA_HEADS + 1), QSA_DIM, 0, s>>>(y, qn, nullptr, qsa_ring, q, nullptr, eps, base, lr);
    k_idx_pool<<<T, QSA_DIM, 0, s>>>(nullptr, qsa_ring, kn, pl, g, nullptr, eps, base, lr);
    ck_launch("qsa_index_lanes");
}

void qsa_select(const float* q, const uint16_t* pooled, const KvPages& g, float* scores, int* cells, int* n_cells, int max_pools, const int* pos0, int T,
                int card, void* st) {
    cudaStream_t s = (cudaStream_t) st;
    const __half* pl = reinterpret_cast<const __half*>(pooled);
#if FNX_ATT_MM
    // FNX_PF_SCORE_RT = TOK (0, 32 or 64; default 32: +7.8 % read rate at 127K): k_idx_score_rt<TOK>, the same scores with TOK rows a block over FNX_PF_SCORE_KR keys (default 256)
    static const int rt = (int) env_int("FNX_PF_SCORE_RT", 32), kr = std::max(16, (int) env_int("FNX_PF_SCORE_KR", 256)) / 16 * 16;
    if (rt && att_mm_on()) {
        if (rt != 32 && rt != 64) throw std::runtime_error("FNX_PF_SCORE_RT: 0, 32 or 64");
        if (rt == 32) k_idx_score_rt<32><<<dim3((max_pools + kr - 1) / kr, (T + 31) / 32), 256, 0, s>>>(q, pl, g, scores, max_pools, pos0, T, kr);
        else k_idx_score_rt<64><<<dim3((max_pools + kr - 1) / kr, (T + 63) / 64), 256, 0, s>>>(q, pl, g, scores, max_pools, pos0, T, kr);
        static const int check = (int) env_int("FNX_PF_SCORE_CHECK", 0);   // debug, syncs: the first N calls against k_idx_score_tc
        if (check > 0) {
            // scratch a device (the cards' workers run at once), the count under a lock
            struct Scratch { float* ref = nullptr; unsigned* bad = nullptr; size_t cap = 0; };
            static Scratch sc[16];
            static std::mutex mu;
            static int checked = 0;
            int dev = 0;
            cudaGetDevice(&dev);
            if (dev < 0 || dev >= 16) throw std::runtime_error("FNX_PF_SCORE_CHECK: device");
            bool go;
            {
                std::lock_guard<std::mutex> lk(mu);
                go = checked < check;
            }
            if (go) {
                Scratch& x = sc[dev];
                const size_t need = (size_t) T * max_pools;
                if (need > x.cap) {
                    if (x.ref) cudaFree(x.ref);
                    if (cudaMalloc(&x.ref, need * sizeof(float)) != cudaSuccess) throw std::runtime_error("FNX_PF_SCORE_CHECK: scratch");
                    x.cap = need;
                }
                if (!x.bad && cudaMalloc(&x.bad, sizeof(unsigned)) != cudaSuccess) throw std::runtime_error("FNX_PF_SCORE_CHECK: scratch");
                k_idx_score_tc<<<dim3((max_pools + QSC_KEYS - 1) / QSC_KEYS, (T + QSC_ROWS - 1) / QSC_ROWS), 256, 0, s>>>(q, pl, g, x.ref, max_pools,
                                                                                                                     pos0, T);
                cudaMemsetAsync(x.bad, 0, sizeof(unsigned), s);
                k_score_cmp<<<dim3(32, T), 256, 0, s>>>(scores, x.ref, max_pools, pos0, T, x.bad);
                if (cudaStreamSynchronize(s) != cudaSuccess) throw std::runtime_error("FNX_PF_SCORE_CHECK: sync");
                unsigned nb = 0;
                int p0 = 0;
                if (cudaMemcpy(&nb, x.bad, sizeof(unsigned), cudaMemcpyDeviceToHost) != cudaSuccess ||
                    cudaMemcpy(&p0, pos0, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) throw std::runtime_error("FNX_PF_SCORE_CHECK: copy");
                if ((p0 + T) / QSA_POOL > QSA_TOP) {   // calls with a row past QSA_TOP blocks only (k_score_cmp's rows)
                    std::lock_guard<std::mutex> lk(mu);
                    if (checked < check) {
                        ++checked;
                        fprintf(stderr, "fnx: score check (rt %d, kr %d, card %d, pos0 %d, rows %d): %u scores differ\n", rt, kr, dev, p0, T, nb);
                    }
                }
            }
        }
    } else
#endif
    k_idx_score_tc<<<dim3((max_pools + QSC_KEYS - 1) / QSC_KEYS, (T + QSC_ROWS - 1) / QSC_ROWS), 256, 0, s>>>(q, pl, g, scores, max_pools, pos0, T);
    // FNX_PF_SEL_MM (default 1 when FNX_ATT_MM is on): k_idx_select_mm's sampled threshold, the same list as k_idx_select's bit for bit
    // (by heads every block is this card's); 0: k_idx_select's radix select over every score
#if FNX_ATT_MM
    static const bool sel_mm = env_int("FNX_PF_SEL_MM", 1) != 0;
    if (sel_mm && att_mm_on()) {   // FNX_ATT_MM=0 keeps every original kernel (model.hpp)
        k_idx_select_mm<<<T, QS_NT, 0, s>>>(scores, cells, n_cells, max_pools, pos0, LaneRows{});
        ck_launch("qsa_select");
        return;
    }
#endif
    k_idx_select<<<T, QSA_SEL_THREADS, 0, s>>>(scores, cells, n_cells, max_pools, pos0, card, LaneRows{});
    ck_launch("qsa_select");
}

void qsa_select_lanes(const float* q, const uint16_t* pooled, const KvPages& g, float* scores, int* cells, int* n_cells, int max_pools,
                      const LaneRows& lr, int T, int card, void* st) {
    cudaStream_t s = (cudaStream_t) st;
    if (!lr.pos || !lr.pt) throw std::runtime_error("qsa_select_lanes: no lane rows");
    static const bool dense_only = env_flag("FNX_QSA_DENSE");
    if (dense_only) {
        cudaMemsetAsync(n_cells, 0xff, (size_t) T * sizeof(int), s);   // checked by ck_launch below
        ck_launch("qsa_select_lanes");
        return;
    }
    const __half* pl = reinterpret_cast<const __half*>(pooled);
#if FNX_ATT_MM
    if (att_mm_on()) {   // a page table per row: a block of scores per row
        k_idx_score_mm<8><<<dim3((max_pools + 255) / 256, T), 256, 0, s>>>(q, pl, g, scores, max_pools, nullptr, lr, T);
        k_idx_select_mm<<<T, QS_NT, 0, s>>>(scores, cells, n_cells, max_pools, nullptr, lr);
        ck_launch("qsa_select_lanes");
        return;
    }
#endif
    k_idx_score<<<dim3((max_pools + 255) / 256, T), 256, 0, s>>>(q, pl, g, scores, max_pools, nullptr, lr);
    k_idx_select<<<T, QSA_SEL_THREADS, 0, s>>>(scores, cells, n_cells, max_pools, nullptr, card, lr);
    ck_launch("qsa_select_lanes");
}
int qsa_rows() { return QSA_ROWS; }

void qsa_proj(const uint16_t* wq, const uint16_t* wk, const float* x, float* y, int T, void* st) {
    cudaStream_t s = (cudaStream_t) st;
    const int grid = (QSA_ROWS + 7) / 8;
#define FNX_QP(t) k_idx_proj<t><<<grid, 256, 0, s>>>(wq, wk, x, y)
    switch (T) {
        case 1: FNX_QP(1); break;
        case 2: FNX_QP(2); break;
        case 3: FNX_QP(3); break;
        case 4: FNX_QP(4); break;
        case 5: FNX_QP(5); break;
        case 6: FNX_QP(6); break;
        case 7: FNX_QP(7); break;
        case 8: FNX_QP(8); break;
        case 9: FNX_QP(9); break;
        case 10: FNX_QP(10); break;
        case 11: FNX_QP(11); break;
        case 12: FNX_QP(12); break;
        case 13: FNX_QP(13); break;
        case 14: FNX_QP(14); break;
        case 15: FNX_QP(15); break;
        case 16: FNX_QP(16); break;
        default: throw std::runtime_error("qsa: T out of range");
    }
#undef FNX_QP
    ck_launch("qsa proj");
}

void qsa(const uint16_t* wq, const uint16_t* wk, const float* x, float* y, const float* qn, const float* kn, uint16_t* ring,
         uint16_t* pooled, const KvPages& g, float* q, float* scores, int* cells, int* n_cells, int max_pools, const int* pos0, int T, float eps,
         Rope base, int card, void* st) {
    cudaStream_t s = (cudaStream_t) st;
    if (wq) qsa_proj(wq, wk, x, y, T, st);
    __half* r = reinterpret_cast<__half*>(ring);
    __half* pl = reinterpret_cast<__half*>(pooled);
    if (T + QSA_POOL > qsa_ring) throw std::runtime_error("qsa: the ring is shorter than a window and a block");
    k_idx_prep_pool<<<dim3(T, QSA_HEADS + 2), QSA_DIM, 0, s>>>(y, qn, kn, r, qsa_ring, q, pl, g, pos0, eps, base);
    // FNX_QSA_DENSE=1 (for comparisons): no selection, every row attends every position
    static const bool dense_only = env_flag("FNX_QSA_DENSE");
    if (dense_only) {
        cudaMemsetAsync(n_cells, 0xff, (size_t) T * sizeof(int), s);   // checked by ck_launch below
        ck_launch("qsa");
        return;
    }
#if FNX_ATT_MM
    if (att_mm_on()) {
        if (att_mm_score_tc())
            k_idx_score_tc<<<dim3((max_pools + QSC_KEYS - 1) / QSC_KEYS, (T + QSC_ROWS - 1) / QSC_ROWS), 256, 0, s>>>(q, pl, g, scores, max_pools, pos0, T);
        else k_idx_score_mm<8><<<dim3((max_pools + 255) / 256, (T + 7) / 8), 256, 0, s>>>(q, pl, g, scores, max_pools, pos0, LaneRows{}, T);
        if (g_kprof_dev >= 0) kprof_mark("qsa prep+score");
        k_idx_select_mm<<<T, QS_NT, 0, s>>>(scores, cells, n_cells, max_pools, pos0, LaneRows{});
        ck_launch(g_kprof_dev >= 0 ? "qsa select" : "qsa");
        return;
    }
#endif
    k_idx_score<<<dim3((max_pools + 255) / 256, T), 256, 0, s>>>(q, pl, g, scores, max_pools, pos0, LaneRows{});
    k_idx_select<<<T, QSA_SEL_THREADS, 0, s>>>(scores, cells, n_cells, max_pools, pos0, card, LaneRows{});
    ck_launch("qsa");
}

size_t attn_scratch_floats(int) { return (size_t) ATT_ROWS * NQH_LOC * ATT_SPLITS * (HD + 2) + (size_t) ATT_ROWS * NQH_LOC * (HD + 2); }
size_t att_ring_words() { return (size_t) 2 * (NCARD - 1) * ATT_ROWS * ATT_OWN * (HD + 2); }
int attn_rows() { return ATT_ROWS; }
int kv_rows_per_card(int n_ctx) { return ATT_HEADS ? n_ctx / 4 * 4 + 4 : (n_ctx / 4 + NCARD - 1) / NCARD * 4 + 4; }

// xq (attn, attn_finish): the output also as q8_1 rows, NQH_LOC * HD values a row. The combine kernel writes them itself (FNX_ATT_Q8, on in an APPROX build; 0 or
// another path: quantize_q8_1 after it, as before)
static bool att_q8_fused() {
    static const bool on = env_int("FNX_ATT_Q8", APPROX) != 0;
    return on;
}
void attn(const float* qg, const uint16_t* kc16, const uint16_t* vc16, const KvPages& g, float* out, const AttRing& ar, const unsigned* tag,
          int call, const int* pos0, int T, float* scratch, void* st, const int* cells, const int* n_cells, void* xq) {
    ring_call_check(call, "attn");
    float* acc = scratch;
    if (T > ATT_ROWS) throw std::runtime_error("attn: more rows than its buffers hold");
    AttPart* mz = reinterpret_cast<AttPart*>(scratch + (size_t) ATT_ROWS * NQH_LOC * ATT_SPLITS * HD);
    float* part = scratch + (size_t) ATT_ROWS * NQH_LOC * ATT_SPLITS * (HD + 2);
    const __half* kc = reinterpret_cast<const __half*>(kc16);
    const __half* vc = reinterpret_cast<const __half*>(vc16);
    cudaStream_t s = (cudaStream_t) st;
#if FNX_ATT_MM
    if (att_mm_attn()) {
        att_mm_pages(g, "attn");
        const int S = att_mm_splits(T);
        k_attn_mm<<<dim3(S, T), 256, 0, s>>>(qg, reinterpret_cast<const uint8_t*>(kc16), reinterpret_cast<const uint8_t*>(vc16), g, acc, mz, pos0, cells,
                                             n_cells, LaneRows{}, S);
        if (g_kprof_dev >= 0) kprof_mark("attn part");
        int8_t* tq = nullptr;
        float* td = nullptr;
        const bool fq = xq && att_q8_fused();
        if (fq) twin_of(xq, NQH_LOC * HD, tq, td);
        k_attn_combine_n<<<dim3(NQH_LOC, T), HD, 0, s>>>(qg, acc, mz, out, NQH_LOC, S, fq ? (block_q8_1*) xq : nullptr, tq, td);
        ck_launch(g_kprof_dev >= 0 ? "attn combine" : "attn");
        if (xq && !fq) quantize_q8_1(out, xq, NQH_LOC * HD, T, st);
        (void) part, (void) tag, (void) kc, (void) vc;
        return;
    }
#endif
    k_attn_part<<<dim3(ATT_SPLITS, T, KVH_LOC), HD, 0, s>>>(qg, kc, vc, g, acc, mz, NQH_LOC, 0, KVH_LOC, 0, pos0, cells, n_cells, ar.card,
                                                            LaneRows{});
    if (g_kprof_dev >= 0) kprof_mark("attn part");
    if constexpr (ATT_HEADS) {   // by heads: this card's splits are every position of its heads; combined and gated here
        k_attn_combine<<<dim3(NQH_LOC, T), HD, 0, s>>>(qg, acc, mz, out, NQH_LOC);
        ck_launch(g_kprof_dev >= 0 ? "attn combine" : "attn");
        if (xq) quantize_q8_1(out, xq, NQH_LOC * HD, T, st);
        (void) part, (void) tag;
        return;
    }   // FNX_KPROF: the three kernels one by one
    k_attn_local<<<dim3(NQH, T), HD, 0, s>>>(acc, mz, part);
    if (g_kprof_dev >= 0) kprof_mark("attn local");
    if (ar.onehop) k_att_merge_a2a<<<dim3(ATT_OWN, T), HD, 0, s>>>(ar, qg, part, out, T, tag, call);
    else k_att_merge<<<dim3(ATT_OWN, T), HD, 0, s>>>(ar, qg, part, out, T, tag, call);
    ck_launch(g_kprof_dev >= 0 ? "attn merge" : "attn");
    if (xq) quantize_q8_1(out, xq, NQH_LOC * HD, T, st);
}

void attn_part_lanes(const float* qg, const uint16_t* kc16, const uint16_t* vc16, const KvPages& g, const LaneRows& lr, int T,
                     float* scratch, void* st, const int* cells, const int* n_cells, int card) {
    if (T > ATT_ROWS) throw std::runtime_error("attn: more rows than its buffers hold");
    if (!lr.pos || !lr.pt) throw std::runtime_error("attn_part_lanes: no lane rows");
    AttPart* mz = reinterpret_cast<AttPart*>(scratch + (size_t) ATT_ROWS * NQH_LOC * ATT_SPLITS * HD);
#if FNX_ATT_MM
    if (att_mm_attn()) {   // attn_finish combines the same splits
        att_mm_pages(g, "attn_part_lanes");
        const int S = att_mm_splits(T);
        k_attn_mm<<<dim3(S, T), 256, 0, (cudaStream_t) st>>>(qg, reinterpret_cast<const uint8_t*>(kc16), reinterpret_cast<const uint8_t*>(vc16), g, scratch,
                                                             mz, nullptr, cells, n_cells, lr, S);
        ck_launch("attn lanes");
        (void) card;
        return;
    }
#endif
    k_attn_part<<<dim3(ATT_SPLITS, T, KVH_LOC), HD, 0, (cudaStream_t) st>>>(qg, reinterpret_cast<const __half*>(kc16),
                                                                          reinterpret_cast<const __half*>(vc16), g, scratch, mz, NQH_LOC, 0,
                                                                          KVH_LOC, 0, nullptr, cells, n_cells, card, lr);
    ck_launch("attn lanes");
}

void attn_finish(const float* qg, float* out, const AttRing& ar, const unsigned* tag, int call, int T, float* scratch, void* st, void* xq) {
    ring_call_check(call, "attn_finish");
    AttPart* mz = reinterpret_cast<AttPart*>(scratch + (size_t) ATT_ROWS * NQH_LOC * ATT_SPLITS * HD);
    float* part = scratch + (size_t) ATT_ROWS * NQH_LOC * ATT_SPLITS * (HD + 2);
    cudaStream_t s = (cudaStream_t) st;
#if FNX_ATT_MM
    if (att_mm_attn()) {
        int8_t* tq = nullptr;
        float* td = nullptr;
        const bool fq = xq && att_q8_fused();
        if (fq) twin_of(xq, NQH_LOC * HD, tq, td);
        k_attn_combine_n<<<dim3(NQH_LOC, T), HD, 0, s>>>(qg, scratch, mz, out, NQH_LOC, att_mm_splits(T), fq ? (block_q8_1*) xq : nullptr, tq, td);
        ck_launch("attn finish");
        if (xq && !fq) quantize_q8_1(out, xq, NQH_LOC * HD, T, st);
        (void) part, (void) ar, (void) tag;
        return;
    }
#endif
    if constexpr (ATT_HEADS) {   // by heads: no merge across the cards (attn's)
        k_attn_combine<<<dim3(NQH_LOC, T), HD, 0, s>>>(qg, scratch, mz, out, NQH_LOC);
        ck_launch("attn finish");
        if (xq) quantize_q8_1(out, xq, NQH_LOC * HD, T, st);
        (void) part, (void) ar, (void) tag;
        return;
    }
    k_attn_local<<<dim3(NQH, T), HD, 0, s>>>(scratch, mz, part);
    if (ar.onehop) k_att_merge_a2a<<<dim3(ATT_OWN, T), HD, 0, s>>>(ar, qg, part, out, T, tag, call);
    else k_att_merge<<<dim3(ATT_OWN, T), HD, 0, s>>>(ar, qg, part, out, T, tag, call);
    ck_launch("attn finish");
    if (xq) quantize_q8_1(out, xq, NQH_LOC * HD, T, st);
}

size_t attn_prompt_block_floats(int C) { return (size_t) C * ATT_OWN * AP_W; }
int attn_prompt_union_rows() {
#if FNX_HIP
    if constexpr (ATT_HEADS) {
        static const int r = [] {
            const int v = (int) env_int("FNX_PF_ATT_UNION", 4);
            if (v != 0 && v != 4 && v != 8) throw std::runtime_error("FNX_PF_ATT_UNION: 0, 4 or 8");
            if (v)
                if (const tk::Family* f = tk::family("pf_attn_u")) {   // the generator's variant: its rows a group (the walk's and its lists' builder's)
                    const int r = f->kern(0).meta[0];
                    if (r < 1 || r > 16 || f->kern(1).meta[0] != r) throw std::runtime_error("tk: pf_attn_u " + f->variant + ": the walk and the lists' builder differ in rows a group");
                    return r;
                }
            return v;
        }();
        return r;
    }
#endif
    return 0;
}
size_t attn_prompt_union_ints(int C) {
    const int R = attn_prompt_union_rows();
    if (!R) return 0;
    const size_t groups = (size_t) (C + R - 1) / R;
    return groups * ((size_t) R * (QSA_TOP + 1) + 1);   // the groups' lists, then their counts
}

void attn_prompt(const float* qg, const uint16_t* kc, const uint16_t* vc, const KvPages& g, const int* pos0, const int* cells,
                 const int* n_cells, int card, int C, int r0, int T, float* part, void* st, int* uscr) {
    if (T == 0) return;   // an empty half of a one-row chunk
#if !FNX_HIP && FNX_ARCH >= 80 && !FNX_ATT_HEADS
    // FNX_PFX_ATTN_I8: the INT8 tensor-core kernel (attn_prompt_i8.cuh) on int8 K/V pages, on by default on NVIDIA sm_80+ for the IQ3_S-type files and off for the others (UD-Q4_K_XL lost agent-A8 and A10 with it); FNX_PFX_ATTN_I8=0/1 overrides; HIP keeps the f16 kernel
    static const int i8_env = (int) env_int("FNX_PFX_ATTN_I8", -1);
    const bool i8 = i8_env >= 0 ? i8_env != 0 : attn_i8_default();
    if (i8 && g.q8) {
        static bool attr[64] = {};
        int dev = 0;
        cudaGetDevice(&dev);
        if (dev >= 0 && dev < 64 && !attr[dev]) {
            cudaFuncSetAttribute(k_pf_attn_i8, cudaFuncAttributePreferredSharedMemoryCarveout, 100);   // three blocks an SM
            cudaGetLastError();
            attr[dev] = true;
        }
        k_pf_attn_i8<<<dim3(T, NKVH), AI_THREADS, sizeof(AiSmem), (cudaStream_t) st>>>(qg, reinterpret_cast<const __half*>(kc), reinterpret_cast<const __half*>(vc), g, pos0,
                                                                                       cells, n_cells, card, C, r0, part);
        ck_launch("attn_prompt (int8)");
        return;
    }
#endif
#if FNX_HIP && FNX_ATT_HEADS
    // FNX_PF_ATT_UNION=4 (default) or 8, 0 = off (by heads): groups of rows walk the union of their selected positions (attn_prompt.cuh,
    // k_pf_attn_u). FNX_PF_ATT_UNION_CHECK=N: the first N calls also run k_pf_attn and print how far the two partials differ (syncs)
    if constexpr (ATT_HEADS) {
        const int urows = attn_prompt_union_rows();
        if (urows && cells) {
            if (!uscr) throw std::runtime_error("attn_prompt: the union walk needs its scratch (attn_prompt_union_ints)");
            struct Scratch { float* ref = nullptr; size_t ref_floats = 0; };   // FNX_PF_ATT_UNION_CHECK's own (lazy)
            static Scratch sc[64];
            int dev = 0;
            cudaGetDevice(&dev);
            Scratch& x = sc[dev & 63];
            const size_t groups = (size_t) (T + urows - 1) / urows, ul_n = (size_t) urows * (QSA_TOP + 1);
            if (groups * (ul_n + 1) > attn_prompt_union_ints(C)) throw std::runtime_error("attn_prompt: union scratch too small");
            struct { int* ul; int* uc; } u{uscr, uscr + groups * ul_n};
            cudaStream_t s = (cudaStream_t) st;
            const __half* k16 = reinterpret_cast<const __half*>(kc);
            const __half* v16 = reinterpret_cast<const __half*>(vc);
            static const tk::Family* const tkf = tk::family("pf_attn_u");   // the generator's variant (key 0 the walk, key 1 the lists' builder)
            if (tkf) {
                const tk::Kern &kw = tkf->kern(0), &ku = tkf->kern(1);
                FNX_TK_LAUNCH(ku, k_pf_union<4>, dim3((unsigned) groups), dim3(ku.threads), ku.smem, s, cells, n_cells, pos0, r0, T, u.ul, u.uc);
                FNX_TK_LAUNCH(kw, k_pf_attn_u<4>, dim3((unsigned) groups), dim3(kw.threads), kw.smem, s, qg, k16, v16, g, pos0, u.ul, u.uc, card, C, r0, T, part);
            } else if (urows == 8) {
                k_pf_union<8><<<groups, 256, 0, s>>>(cells, n_cells, pos0, r0, T, u.ul, u.uc);
                k_pf_attn_u<8><<<groups, AP_THREADS, AuShape<8>::SMEM, s>>>(qg, k16, v16, g, pos0, u.ul, u.uc, card, C, r0, T, part);
            } else {
                k_pf_union<4><<<groups, 256, 0, s>>>(cells, n_cells, pos0, r0, T, u.ul, u.uc);
                static const bool ureg = env_int("FNX_PF_ATT_UREG", 1) != 0;   // attn_prompt.cuh AuShape: O in accumulator tiles (default 1: +5.5 % read rate at 127K)
                static const int upc = (int) env_int("FNX_PF_ATT_PC", 32);   // positions a union-walk tile (32 or 64; 64 needs FNX_PF_ATT_UREG=1)
                static const bool ulean = env_int("FNX_PF_ATT_LEAN", 1) != 0;   // attn_prompt.cuh AuShape LEAN: < 32 KB shared memory, 4 blocks a WGP (default 1: +1.3 / +1.6 / +0.9 % read at 127K / 80K / 8K; same sums)
                // FNX_PF_ATT_GW (default 2; the default layout only): the walk's gather. 1: a thread takes one (position, group of 32 values) with two 16-byte loads and
                // one scale load; 2: also, a tile's V gather takes the row places of its K gather; 0: four 8-byte loads of four positions a thread. The same bits
                // (kbench pfattn, 109K: 6.89 / 5.34 / 5.17 ms a launch for 0 / 1 / 2; reads +2.2 % at 80K with 1)
                static const int ugw = (int) env_int("FNX_PF_ATT_GW", 2);
                if (ugw < 0 || ugw > 2) throw std::runtime_error("FNX_PF_ATT_GW: 0, 1 or 2");
                if (ureg && ulean && upc != 64 && ugw == 2) k_pf_attn_u<4, true, APC, true, 0, 2><<<groups, AP_THREADS, AuShape<4, true, APC, true>::SMEM, s>>>(qg, k16, v16, g, pos0, u.ul, u.uc, card, C, r0, T, part);
                else if (ureg && ulean && upc != 64 && ugw == 1) k_pf_attn_u<4, true, APC, true, 0, 1><<<groups, AP_THREADS, AuShape<4, true, APC, true>::SMEM, s>>>(qg, k16, v16, g, pos0, u.ul, u.uc, card, C, r0, T, part);
                else if (ureg && ulean && upc != 64) k_pf_attn_u<4, true, APC, true><<<groups, AP_THREADS, AuShape<4, true, APC, true>::SMEM, s>>>(qg, k16, v16, g, pos0, u.ul, u.uc, card, C, r0, T, part);
                else if (ureg && upc == 64) k_pf_attn_u<4, true, 64><<<groups, AP_THREADS, AuShape<4, true, 64>::SMEM, s>>>(qg, k16, v16, g, pos0, u.ul, u.uc, card, C, r0, T, part);
                else if (ureg) k_pf_attn_u<4, true><<<groups, AP_THREADS, AuShape<4, true>::SMEM, s>>>(qg, k16, v16, g, pos0, u.ul, u.uc, card, C, r0, T, part);
                else k_pf_attn_u<4><<<groups, AP_THREADS, AuShape<4>::SMEM, s>>>(qg, k16, v16, g, pos0, u.ul, u.uc, card, C, r0, T, part);
            }
            ck_launch("attn_prompt (union)");
            static const int checks = (int) env_int("FNX_PF_ATT_UNION_CHECK", 0);
            static std::atomic<int> checked{0};
            static const int check_min = (int) env_int("FNX_PF_ATT_UNION_CHECK_MIN", 0);   // only calls whose first row is at this position or later
            int cp0 = 0;
            if (checks > 0 && checked.load() < checks && check_min > 0) cudaStreamSynchronize(s), cudaMemcpy(&cp0, pos0, sizeof(int), cudaMemcpyDeviceToHost);
            if (checks > 0 && checked.load() < checks && cp0 >= check_min) {
                const size_t blk = (size_t) C * ATT_OWN * AP_W, all = (size_t) NCARD * blk;
                if (all > x.ref_floats) {
                    if (x.ref) cudaFree(x.ref);
                    if (cudaMalloc(&x.ref, all * sizeof(float)) != cudaSuccess) throw std::runtime_error("FNX_PF_ATT_UNION_CHECK: scratch");
                    x.ref_floats = all;
                }
                k_pf_attn<<<dim3(T, KVH_LOC), AP_THREADS, AP_SMEM, s>>>(qg, k16, v16, g, pos0, cells, n_cells, card, C, r0, x.ref);
                ck_launch("attn_prompt (check)");
                std::vector<float> a((size_t) T * ATT_OWN * AP_W), b(a.size());
                const size_t off = (size_t) card * blk + (size_t) r0 * ATT_OWN * AP_W;
                cudaStreamSynchronize(s);
                cudaMemcpy(a.data(), part + off, a.size() * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(b.data(), x.ref + off, b.size() * sizeof(float), cudaMemcpyDeviceToHost);
                int p0 = 0;
                cudaMemcpy(&p0, pos0, sizeof(int), cudaMemcpyDeviceToHost);
                {
                    const cudaError_t le = cudaGetLastError();
                    std::vector<int> uc(std::min<size_t>(groups, 4)), u0(16);
                    cudaMemcpy(uc.data(), u.uc, uc.size() * sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpy(u0.data(), u.ul, u0.size() * sizeof(int), cudaMemcpyDeviceToHost);
                    long bu = 0, bv = 0;
                    for (size_t r = 0; r < (size_t) T * ATT_OWN; ++r) bu += !(a[r * AP_W + HD + 1] > 0.f), bv += !(b[r * AP_W + HD + 1] > 0.f);
                    std::fprintf(stderr, "fnx: attn union diag: last error %d, groups %zu, ucount %d %d %d %d, ul[0..7] %x %x %x %x %x %x %x %x; row 0 head 0: union m %g z %g o0 %g / ref m %g z %g o0 %g; "
                                         "row %d: union m %g z %g / ref m %g z %g; zero-sum rows: union %ld ref %ld\n",
                                 (int) le, groups, uc[0], uc.size() > 1 ? uc[1] : -1, uc.size() > 2 ? uc[2] : -1, uc.size() > 3 ? uc[3] : -1, u0[0], u0[1], u0[2], u0[3], u0[4], u0[5], u0[6], u0[7],
                                 a[HD], a[HD + 1], a[0], b[HD], b[HD + 1], b[0], T - 1, a[((size_t) (T - 1) * ATT_OWN) * AP_W + HD], a[((size_t) (T - 1) * ATT_OWN) * AP_W + HD + 1],
                                 b[((size_t) (T - 1) * ATT_OWN) * AP_W + HD], b[((size_t) (T - 1) * ATT_OWN) * AP_W + HD + 1], bu, bv);
                }
                double worst = 0, lse = 0, sum = 0;
                long bad = 0, n = 0;
                for (size_t r = 0; r < (size_t) T * ATT_OWN; ++r) {
                    const float* u = a.data() + r * AP_W;
                    const float* v = b.data() + r * AP_W;
                    if (!(u[HD + 1] > 0.f) || !(v[HD + 1] > 0.f) || !std::isfinite(u[HD]) || !std::isfinite(v[HD])) { ++bad; continue; }
                    double mx = 0, df = 0;
                    for (int d = 0; d < HD; ++d) {
                        const double ou = u[d] / u[HD + 1], ov = v[d] / v[HD + 1];
                        if (!std::isfinite(ou) || !std::isfinite(ov)) ++bad;
                        mx = std::max(mx, std::fabs(ov)), df = std::max(df, std::fabs(ou - ov));
                    }
                    const double rel = df / std::max(mx, 1e-20);
                    worst = std::max(worst, rel), sum += rel, ++n;
                    lse = std::max(lse, std::fabs((u[HD] + std::log((double) u[HD + 1])) - (v[HD] + std::log((double) v[HD + 1]))));
                }
                std::fprintf(stderr, "fnx: attn union check (R %d, card %d, pos0 %d, rows %d): output rel. diff max %.3g mean %.3g, log-sum diff max %.3g, bad %ld of %ld\n",
                             urows, card, p0, T, worst, n ? sum / n : 0.0, lse, bad, n);
                checked.fetch_add(1);
            }
            return;
        }
    }
#endif
    k_pf_attn<<<dim3(T, KVH_LOC), AP_THREADS, AP_SMEM, (cudaStream_t) st>>>(qg, reinterpret_cast<const __half*>(kc),
                                                                         reinterpret_cast<const __half*>(vc), g, pos0, cells, n_cells,
                                                                         card, C, r0, part);
    ck_launch("attn_prompt");
}

void attn_prompt_merge(const float* own, const float* recv, const float* qg, float* out, int C, int card, void* st) {
    k_pf_attn_merge<<<dim3(C, ATT_OWN), HD, 0, (cudaStream_t) st>>>(own, recv, qg, out, C, card);
    ck_launch("attn_prompt_merge");
}

#endif
#if FNX_KPART == 0
void ple_post(float* R, const float* key, const float* value, const float* nk, const float* nq, const float* nc, const float* convw,
              float* hist, float* hsnap, int T, float eps, void* st, float* scratch) {
    float* gated = scratch;   // gated, then normalized rows
    float* normd = scratch + (size_t) T * D;
    k_ple_gate<<<dim3(HC, T), 1024, 0, (cudaStream_t) st>>>(R, key, value, nk, nq, nc, gated, normd, eps);
    k_ple_conv<<<(D + 255) / 256, 256, 0, (cudaStream_t) st>>>(R, gated, normd, convw, hist, hsnap, T);
    ck_launch("ple_post");
}

// flags: [2 sets][kind: the reduce steps' (in), then the finished chunks' (ag)][block]; on n cards kinds 0..n-2 are in
// steps 0..n-2 and kinds n-1..2n-3 ag steps n-1..2n-3 (three cards: in step 0, in step 1, ag step 2, ag step 3)
constexpr int RB_BLOCKS = 240, RB_KINDS = 2 * NCARD - 2;
size_t ring_flag_words() { return (size_t) 2 * RB_KINDS * RB_BLOCKS; }

// block b's segment of a hop's chunk c: pairs [s0, s1) of the T * len (row, pair) elements, row-major
__device__ __forceinline__ void rb_seg(int T, int c, int& lo, int& len, int& s0, int& s1) {
    lo = clo(c);
    len = chi(c) - lo;
    const int n = T * len, seg = (n + gridDim.x - 1) / gridDim.x;
    s0 = min(n, (int) blockIdx.x * seg);
    s1 = min(n, s0 + seg);
}
__device__ __forceinline__ __half2 rb_pack(float2 v, unsigned* err) {
    if (!(fabsf(v.x) <= RING_MAX && fabsf(v.y) <= RING_MAX)) atomicOr(err, 2u);   // out of range (or NaN)
    return __floats2half2_rn(v.x, v.y);
}
// the block's writes, then its flag on the next card (one fence after the barrier covers the block's writes)
__device__ __forceinline__ void rb_signal(unsigned* flag, unsigned tag) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        *(volatile unsigned*) flag = tag;
    }
}
__device__ __forceinline__ void rb_wait(const unsigned* flag, unsigned tag, unsigned* err) {
    if (threadIdx.x == 0) {
        const long long ts = clock64();
        long long t0 = ts;
        while (*(const volatile unsigned*) flag != tag)
            if (const long long dt = clock64() - t0; __builtin_expect(dt > RING_TIMEOUT_CYCLES, 0)) {
                if (diag_wait_on(err, dt, RING_TIMEOUT_CYCLES, ts, t0)) continue;   // the driver had the queues off the cards
                diag_timeout(err, DK_CHUNK_RING, tag, *(const volatile unsigned*) flag, 0u, DG_NO_SRC, (unsigned) blockIdx.x, dt);
                atomicOr(err, 1u);
                break;
            }
        __threadfence_system();
    }
    __syncthreads();
}
__device__ __forceinline__ float2 rb_load(const __half2* p) {
    const unsigned w = *reinterpret_cast<const volatile unsigned*>(p);   // FNX_HOSTRING: host memory, no cached read
    return __half22float2(*reinterpret_cast<const __half2*>(&w));
}

__global__ void k_ring_big(Ring r, float* x, int T, const unsigned* tagp, int call, RingMoe m) {
    const unsigned tag = *tagp * 256u + (unsigned) call + 1u;
    const int g = r.card, set = tag & 1, b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const size_t in_set = (size_t) RSTEPS * r.rows * RCH, ag_set = (size_t) r.rows * NP;   // in half2 words
    __half2* in = reinterpret_cast<__half2*>(r.in) + set * 2 * in_set;
    __half2* nin = reinterpret_cast<__half2*>(r.next_in) + set * 2 * in_set;
    __half2* ag = reinterpret_cast<__half2*>(r.ag) + set * ag_set;
    __half2* nag = reinterpret_cast<__half2*>(r.next_ag) + set * ag_set;
    unsigned* fl = r.flags + (size_t) set * RB_KINDS * RB_BLOCKS;
    unsigned* nfl = r.next_flags + (size_t) set * RB_KINDS * RB_BLOCKS;
    auto slot = [&](__half2* base, int s, int t, int i) { return base + ((size_t) s * r.rows + t) * RCH + i; };
    int lo, len, s0, s1;
    {   // step 0: my part of chunk g to g+1
        rb_seg(T, g, lo, len, s0, s1);
        for (int e = s0 + tid; e < s1; e += nt) {
            const int t = e / len, i = e % len;
            *slot(nin, 0, t, i) = rb_pack(ring_x2(x, m, T, t, lo + i), r.err);
        }
        rb_signal(nfl + 0 * RB_BLOCKS + b, tag);
    }
    for (int s = 1; s < RSTEPS; ++s) {   // step s: chunk g-s arrived; add mine, on to g+1
        rb_seg(T, (g + NCARD - s) % NCARD, lo, len, s0, s1);
        rb_wait(fl + (s - 1) * RB_BLOCKS + b, tag, r.err);
        for (int e = s0 + tid; e < s1; e += nt) {
            const int t = e / len, i = e % len;
            *slot(nin, s, t, i) = rb_pack(add2(rb_load(slot(in, s - 1, t, i)), ring_x2(x, m, T, t, lo + i)), r.err);
        }
        rb_signal(nfl + s * RB_BLOCKS + b, tag);
    }
    {   // step NCARD - 1: chunk g+1 with the other cards' parts; add mine: finished. Keep it, send it on
        rb_seg(T, (g + 1) % NCARD, lo, len, s0, s1);
        rb_wait(fl + (RSTEPS - 1) * RB_BLOCKS + b, tag, r.err);
        for (int e = s0 + tid; e < s1; e += nt) {
            const int t = e / len, i = e % len;
            const __half2 v = rb_pack(add2(rb_load(slot(in, RSTEPS - 1, t, i)), ring_x2(x, m, T, t, lo + i)), r.err);
            ag[(size_t) t * NP + lo + i] = v;
            nag[(size_t) t * NP + lo + i] = v;
        }
        rb_signal(nfl + RSTEPS * RB_BLOCKS + b, tag);
    }
    for (int f = 1; f < RSTEPS; ++f) {   // step NCARD - 1 + f: chunk g-f+1 arrived finished from g-1: forward it
        rb_seg(T, (g + NCARD - f + 1) % NCARD, lo, len, s0, s1);
        rb_wait(fl + (RSTEPS - 1 + f) * RB_BLOCKS + b, tag, r.err);
        for (int e = s0 + tid; e < s1; e += nt) {
            const int t = e / len, i = e % len;
            const unsigned w = *reinterpret_cast<const volatile unsigned*>(ag + (size_t) t * NP + lo + i);
            nag[(size_t) t * NP + lo + i] = *reinterpret_cast<const __half2*>(&w);
        }
        rb_signal(nfl + (RSTEPS + f) * RB_BLOCKS + b, tag);
    }
    // the last chunk arrives (finished on two cards, forwarded on three: the previous card's last step); then
    // everything into x. Each block waits for its own segments: every block must see every chunk, so it waits on the
    // flags of the blocks that wrote them
    rb_wait(fl + (RB_KINDS - 1) * RB_BLOCKS + b, tag, r.err);
    __syncthreads();
}

// after k_ring_big on every block: all chunks are in ag (this card's own chunk g+1, chunk g from the finishing step's
// sender and the rest forwarded); a second launch reads them once every block of the first has passed its waits
__global__ void k_ring_big_out(Ring r, float* x, int T, const unsigned* tagp, int call) {
    const unsigned tag = *tagp * 256u + (unsigned) call + 1u;
    const int set = tag & 1;
    const __half2* ag = reinterpret_cast<const __half2*>(r.ag) + set * (size_t) r.rows * NP;
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < T * NP; e += gridDim.x * blockDim.x) {
        const float2 v = rb_load(ag + e);
        x[2 * e] = v.x;
        x[2 * e + 1] = v.y;
    }
}

size_t ring_in_words(int rows) { return (size_t) 2 * RSTEPS * rows * RCH * 2; }
size_t ring_ag_words(int rows) { return (size_t) 2 * rows * N * 2; }

void ring_sum(const Ring& r, float* x, int T, const unsigned* tag, int call, void* st, const RingMoe& m) {
    ring_call_check(call, "ring_sum");
    if (T > r.rows) throw std::runtime_error("ring_sum: more rows than the ring's buffers hold");
    if (r.flags && T > MAXT) {
        if (r.f32) throw std::runtime_error("ring_sum: the chunk ring carries fp16 only");
        const int grid = std::min(RB_BLOCKS, std::max(4, (T * NP / NCARD + 255) / 256));
        k_ring_big<<<grid, 256, 0, (cudaStream_t) st>>>(r, x, T, tag, call, m);
        k_ring_big_out<<<grid, 256, 0, (cudaStream_t) st>>>(r, x, T, tag, call);
        ck_launch("ring_sum (chunk)");
        return;
    }
    int grid = std::max(4, std::min(T > MAXT ? 240 : 40, (T * NP / NCARD + 255) / 256));
#if FNX_HIP
    // FNX_RING_GRID=n: the one-hop window sum on n blocks of 256 threads, at most one block for 256 elements of the whole sum (not of a card's
    // share): few blocks leave each thread several elements to push and then wait for one after the other, a remote round trip each.
    // 40 is the rule on 4 cards with the host mailbox (4 x R9700, Q4, 2026-11-13: 5 -> 20 blocks at 4 rows, 10 -> 40 at 8; one user
    // 185.0 -> 191.2 tok/s, the same answers; 80 and 160 add nothing). 0: the upstream rule. Other card counts, the peer push and the legacy
    // rings were not measured: unchanged unless the switch is given
    if (T <= MAXT) {
        static const int g = (int) env_int("FNX_RING_GRID", -1);
        const int use = g >= 0 ? g : (FAST && NCARD == 4 && r.onehop && !r.peer ? 40 : 0);
        if (use > 0) grid = std::min(use, (T * NP + 255) / 256);
    }
#endif
    if (r.peer && T <= MAXT) {
        static const bool two = fast_opt("FNX_RING_2PHASE");
        if (two) k_ring_a2a_push<true><<<grid, 256, 0, (cudaStream_t) st>>>(r, x, T, tag, call, m);
        else k_ring_a2a_push<false><<<grid, 256, 0, (cudaStream_t) st>>>(r, x, T, tag, call, m);
        ck_launch("ring_sum (one hop, peer push)");
        return;
    }
    if (r.onehop && T <= MAXT) {
        k_ring_a2a<<<grid, 256, 0, (cudaStream_t) st>>>(r, x, T, tag, call, m);
        ck_launch("ring_sum (one hop)");
        return;
    }
    if (r.f32) k_ring<true><<<grid, 256, 0, (cudaStream_t) st>>>(r, x, T, tag, call, m);
    else k_ring<false><<<grid, 256, 0, (cudaStream_t) st>>>(r, x, T, tag, call, m);
    ck_launch("ring_sum");
}

void top_tokens(float* logits, int n, int T, int v0, const int* pen_tok, const int* pen_cnt, const int* pen_n, int pen_stride,
                const float* pen_par, float* out_v, int* out_i, unsigned* err, void* st) {
    if (n < TOPK_N) throw std::runtime_error("top_tokens: a vocabulary slice smaller than the candidates");
    k_topk_rows<<<T, TOPK_THREADS, 0, (cudaStream_t) st>>>(logits, n, v0, pen_tok, pen_cnt, pen_n, pen_stride, pen_par,
                                                         out_v, out_i, err);
    ck_launch("top_tokens");
}

void argmax(const float* logits, int rows, int T, int v0, float* best_v, int* best_i, unsigned long long* keys, unsigned* err, void* st) {
    k_argmax<<<dim3(AM_BLOCKS, T), AM_THREADS, 0, (cudaStream_t) st>>>(logits, rows, keys, err);
    k_argmax_fin<<<1, 32, 0, (cudaStream_t) st>>>(keys, T, v0, best_v, best_i);
    ck_launch("argmax");
}

namespace {
__global__ void k_penalize_row(float* L, int n, const int* pen, const float* par) {
    const int k = pen[0];
    const float repeat = par[0], freq = par[1], present = par[2];
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        const int j = pen[1 + i], cnt = pen[1 + k + i];
        if (j < 0 || j >= n || cnt <= 0) continue;
        float l = L[j];
        l = l <= 0.f ? l * repeat : l / repeat;
        L[j] = l - (float) cnt * freq - present;
    }
}
}  // namespace

void penalize_row(float* row, int n, const int* pen, const float* par, void* st) {
    k_penalize_row<<<1, 256, 0, (cudaStream_t) st>>>(row, n, pen, par);
    ck_launch("penalize_row");
}

// (Engine::draft_lanes' table form): k_penalize_row for lane blockIdx.x of a batched draft call, its row from the card's table.
// sel[2 * i + 1] of lane i: bits 0-15 its last row, bit 16 set: the lane penalizes. The arithmetic is k_penalize_row's, word for word.
namespace {
__global__ void k_penalize_lanes(float* logits, int n, const int* pens, int pen_stride, const float* pars, const int* sel) {
    const int lane = blockIdx.x, w = sel[2 * lane + 1];
    if (!(w & 0x10000)) return;
    float* L = logits + (size_t) (w & 0xffff) * n;
    const int* pen = pens + (size_t) lane * pen_stride;
    const float* par = pars + 3 * lane;
    const int k = pen[0];
    const float repeat = par[0], freq = par[1], present = par[2];
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        const int j = pen[1 + i], cnt = pen[1 + k + i];
        if (j < 0 || j >= n || cnt <= 0) continue;
        float l = L[j];
        l = l <= 0.f ? l * repeat : l / repeat;
        L[j] = l - (float) cnt * freq - present;
    }
}
// each lane's last row (sel as above) copied to row `lane` of out: the candidates' launches then take fixed rows
__global__ void k_gather_lanes(const float* __restrict__ logits, int n, const int* sel, float* __restrict__ out) {
    const int lane = blockIdx.y;
    const float* s = logits + (size_t) (sel[2 * lane + 1] & 0xffff) * n;
    float* d = out + (size_t) lane * n;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) d[j] = s[j];
}
}  // namespace

void penalize_lanes(float* logits, int n, const int* pens, int pen_stride, const float* pars, const int* sel, int lanes, void* st) {
    if (lanes <= 0) return;
    k_penalize_lanes<<<lanes, 256, 0, (cudaStream_t) st>>>(logits, n, pens, pen_stride, pars, sel);
    ck_launch("penalize_lanes");
}
void gather_lanes(const float* logits, int n, const int* sel, float* out, int lanes, void* st) {
    if (lanes <= 0 || n <= 0) return;
    k_gather_lanes<<<dim3(4, lanes), 256, 0, (cudaStream_t) st>>>(logits, n, sel, out);
    ck_launch("gather_lanes");
}

namespace {
__global__ void k_copy_many(const CopyJob* jobs) {
    const CopyJob j = jobs[blockIdx.y];
    float4* d = reinterpret_cast<float4*>(j.dst);
    const float4* s = reinterpret_cast<const float4*>(j.src);
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < j.bytes / 16; i += (size_t) gridDim.x * blockDim.x) d[i] = s[i];
}
}  // namespace

void copy_many(const CopyJob* jobs, int n, void* st) {
    if (n <= 0) return;
    k_copy_many<<<dim3(8, n), 256, 0, (cudaStream_t) st>>>(jobs);
    ck_launch("copy_many");
}
#endif

}  // namespace fnx

#if FNX_KPART == 0
namespace fnx {
namespace {
__global__ void k_hc_write(float* R, const float* bo, const float* inj) {
    const int t = blockIdx.y, e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= D) return;
    const int c = e / N, i = e % N;
    R[(size_t) t * D + e] += bo[(size_t) t * N + i] * 2.0f / (1.0f + __expf(-inj[t * HC + c] / (float) HC));
}
}  // namespace
void hc_write(float* R, const float* bo, const float* inj, int T, void* st) {
    k_hc_write<<<dim3((D + 255) / 256, T), 256, 0, (cudaStream_t) st>>>(R, bo, inj);
    ck_launch("hc_write");
}
}  // namespace fnx

namespace fnx {
namespace {
// row t (block y) of the window, if its position's block is this card's: both KV heads to the local index
__global__ void k_kv_append(const float* k, const float* v, __half* kc, __half* vc, KvPages g, int card, const int* pos0p, LaneRows lr) {
    const int t = blockIdx.y, p = row_pos(lr, pos0p, t), j = p / 4;
    if (!kv_mine(j, card)) return;
    g = row_pages(g, lr, t);
    const int r = kv_local_row(p);
    if (g.q8) {
        // a warp's 32 elements are one group: d = max |x| / 127, q = round(x / d) (llama.cpp's q8_0)
        static_assert(KV8_GROUP == 32, "a group per warp");
        int8_t *kq = reinterpret_cast<int8_t*>(kc), *vq = reinterpret_cast<int8_t*>(vc);
        const size_t at = kvq_row(g, r), sc = kvq_scale(g, r);
        for (int e = threadIdx.x; e < KV_W; e += blockDim.x) {
            const float x[2] = {k[(size_t) t * KV_W + e], v[(size_t) t * KV_W + e]};
#pragma unroll
            for (int w = 0; w < 2; ++w) {
                const float d = wmax(fabsf(x[w])) / 127.f, id = d != 0.f ? 1.f / d : 0.f;
                (w ? vq : kq)[at + e] = (int8_t) roundf(x[w] * id);
                if ((e & 31) == 0) (w ? vc : kc)[sc + e / KV8_GROUP] = __float2half_rn(d);
            }
        }
        return;
    }
    const size_t at = kvp_row(g, r);
    for (int e = threadIdx.x; e < KV_W; e += blockDim.x) {
        kc[at + e] = __float2half_rn(k[(size_t) t * KV_W + e]);
        vc[at + e] = __float2half_rn(v[(size_t) t * KV_W + e]);
    }
}

// k_attn_prep and k_kv_append in one launch: block (h, row t), HD threads; h < NQH: q head h normed and rotated; then
// the NKVH k heads, normed, rotated and appended; then the NKVH v heads, appended (a row whose position is another
// card's appends nothing). The same values as the two kernels
__global__ void __launch_bounds__(HD) k_attn_prep_append(float* qg, float* k, const float* v, const float* qn, const float* kn, __half* kc,
                                                         __half* vc, KvPages g, int card, const int* pos0p, float eps, Rope base) {
    const int t = blockIdx.y, h = blockIdx.x, i = threadIdx.x, p = *pos0p + t, j = p / 4;
    __shared__ float sh[32];
    if (h < NQH_LOC) {
        norm_rope(qg + ((size_t) t * NQH_LOC + h) * 2 * HD, qn, p, eps, base, sh);
        return;
    }
    const bool is_k = h < NQH_LOC + KVH_LOC;
    const int kh = h - NQH_LOC - (is_k ? 0 : KVH_LOC), e = kh * HD + i;
    float x;
    if (is_k) {
        float* kr = k + ((size_t) t * KVH_LOC + kh) * HD;
        norm_rope(kr, kn, p, eps, base, sh);
        __syncthreads();
        x = kr[i];
    } else {
        x = v[(size_t) t * KV_W + e];
    }
    if (!kv_mine(j, card)) return;
    const int r = kv_local_row(p);
    __half* cache = is_k ? kc : vc;
    if (g.q8) {
        static_assert(KV8_GROUP == 32 && HD % 32 == 0, "a group per warp");
        const float d = wmax(fabsf(x)) / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        reinterpret_cast<int8_t*>(cache)[kvq_row(g, r) + e] = (int8_t) roundf(x * id);
        if ((e & 31) == 0) cache[kvq_scale(g, r) + e / KV8_GROUP] = __float2half_rn(d);
        return;
    }
    cache[kvp_row(g, r) + e] = __float2half_rn(x);
}
}  // namespace
namespace {
// FNX_PREFETCH_PROBE: guess[t][k], the top 10 of logits row t (NEXP each; ties: the lower index), a warp per row
__global__ void k_probe_guess(const float* logits, int T, int* guess) {
    const int t = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (t >= T) return;
    float v[NEXP / 32];
    for (int i = 0; i < NEXP / 32; ++i) v[i] = logits[(size_t) t * NEXP + lane + 32 * i];
    unsigned taken = 0;
    for (int k = 0; k < TOPK; ++k) {
        float bv = -INFINITY;
        int bi = NEXP;
        for (int i = 0; i < NEXP / 32; ++i)
            if (!(taken >> i & 1u) && v[i] > bv) bv = v[i], bi = lane + 32 * i;
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
        }
        if ((bi & 31) == lane) taken |= 1u << (bi >> 5);
        if (lane == 0) guess[t * TOPK + k] = bi;
    }
}
// the window's distinct experts picked (topi) and guessed: cnt[0] += picked, cnt[1] += guessed, cnt[2] += both,
// cnt[3] += 1 (layer-windows); with the layer's slots (slot_of, before the window's cache plan) cnt[4] += picked and
// not in a slot, cnt[5] += those guessed, cnt[6] += guessed and neither picked nor in a slot (copies a prefetch wastes)
__global__ void k_probe_score(const int* topi, const int* guess, int T, const int* slot_of, unsigned long long* cnt) {
    __shared__ unsigned char a[NEXP], g[NEXP];
    for (int e = threadIdx.x; e < NEXP; e += blockDim.x) a[e] = g[e] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < T * TOPK; i += blockDim.x) {
        a[topi[i]] = 1;
        g[guess[i]] = 1;
    }
    __syncthreads();
    unsigned na = 0, ng = 0, nb = 0, nm = 0, nmg = 0, nw = 0;
    for (int e = threadIdx.x; e < NEXP; e += blockDim.x) {
        na += a[e];
        ng += g[e];
        nb += a[e] & g[e];
        if (slot_of && slot_of[e] < 0) {
            nm += a[e];
            nmg += a[e] & g[e];
            nw += g[e] & !a[e];
        }
    }
    atomicAdd(&cnt[0], (unsigned long long) na);
    atomicAdd(&cnt[1], (unsigned long long) ng);
    atomicAdd(&cnt[2], (unsigned long long) nb);
    if (threadIdx.x == 0) atomicAdd(&cnt[3], 1ull);
    atomicAdd(&cnt[4], (unsigned long long) nm);
    atomicAdd(&cnt[5], (unsigned long long) nmg);
    atomicAdd(&cnt[6], (unsigned long long) nw);
}
__global__ void k_route_count(const int* topi, int n, unsigned* cnt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&cnt[topi[i]], 1u);
}
__global__ void k_route_trace(const int* topi, int n, int layer, unsigned* trace, unsigned long long* len, unsigned long long cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long at = atomicAdd(len, 1ull);
    if (at < cap) trace[at] = (unsigned) layer << 16 | (unsigned) topi[i];
}
}  // namespace
namespace {
// the plan alone (cache_plan, moe.cuh), one warp
__global__ void __launch_bounds__(32) k_cache_plan(const Job* jobs, const int* njobs, int layer, ExpertCache ec, int* list) {
    cache_plan(jobs, njobs[0], layer, ec, list);
}
// the listed experts' matrices (gu_bytes and dn_bytes each, multiples of 16: the layer's) from their host copies to
// their slots in pool p
__global__ void __launch_bounds__(256) k_cache_copy(ExpertCache ec, int p, const int* list, size_t gu_bytes, size_t dn_bytes) {
    const int n = list[0];
    const size_t ug = gu_bytes / 16, per = ug + dn_bytes / 16;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < (size_t) n * per; i += (size_t) gridDim.x * blockDim.x) {
        const int k = (int) (i / per);
        const size_t o = i % per;
        const int g = list[1 + k];
        char* at = ec.pool[p] + (size_t) (ec.slot_of[g] - ec.s0[p]) * ec.slot_bytes[p];
        if (o < ug) reinterpret_cast<uint4*>(at)[o] = reinterpret_cast<const uint4*>(ec.host_gu[g])[o];
        else reinterpret_cast<uint4*>(at + ec.dn_off[p])[o - ug] = reinterpret_cast<const uint4*>(ec.host_dn[g])[o - ug];
    }
}
}  // namespace
void cache_copy(const ExpertCache& ec, int p, const int* list, void* st) {
    k_cache_copy<<<640, 256, 0, (cudaStream_t) st>>>(ec, p, list, ec.dn_off[p], ec.slot_bytes[p] - ec.dn_off[p]);
    ck_launch("cache_copy");
}
void moe_cache(int layer, size_t gu_stride, size_t dn_stride, const ExpertCache& ec, int* list, void* st, const Job* jobs, const int* njobs) {
    const int p = ec.pool_of[layer];
    if (gu_stride != ec.dn_off[p] || ec.dn_off[p] + dn_stride != ec.slot_bytes[p])
        throw std::runtime_error("moe_cache: the layer's experts are not its pool's size");
    if (jobs) k_cache_plan<<<1, 32, 0, (cudaStream_t) st>>>(jobs, njobs, layer, ec, list);
    k_cache_copy<<<160, 256, 0, (cudaStream_t) st>>>(ec, p, list, gu_stride, dn_stride);
    ck_launch("moe_cache");
}
namespace {
// the arena pool's slots [from, to) lent (their experts to RAM, the slots empty and ref 3), or [to, from) given back
// (empty, ref 0: the first misses take them); then lent[p] = to and the hand past the lent ones
__global__ void k_arena_slots(ExpertCache ec, int from, int to) {
    const int p = ec.arena_pool, s0 = ec.s0[p];
    const bool lend = to > from;
    for (int i = (lend ? from : to) + blockIdx.x * blockDim.x + threadIdx.x; i < (lend ? to : from); i += gridDim.x * blockDim.x) {
        const int s = s0 + i;
        if (const int o = ec.owner[s]; o >= 0) {
            ec.slot_of[o] = -1;
            ec.ep_gu[o] = ec.host_gu[o];
            ec.ep_dn[o] = ec.host_dn[o];
        }
        ec.owner[s] = -1;
        ec.ref[s] = lend ? 3 : 0;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        ec.lent[p] = to;
        if (ec.hand[p] < s0 + to) ec.hand[p] = s0 + to;
    }
}
}  // namespace
void arena_slots(const ExpertCache& ec, int from, int to, void* st) {
    if (ec.arena_pool < 0 || from < 0 || to < 0 || from > ec.n[ec.arena_pool] || to > ec.n[ec.arena_pool])
        throw std::runtime_error("arena_slots: slots out of range");
    if (from == to) return;
    k_arena_slots<<<64, 256, 0, (cudaStream_t) st>>>(ec, from, to);
    ck_launch("arena_slots");
}
void route_trace(const int* topi, int n, int layer, unsigned* trace, unsigned long long* len, unsigned long long cap, void* st) {
    k_route_trace<<<(n + 255) / 256, 256, 0, (cudaStream_t) st>>>(topi, n, layer, trace, len, cap);
    ck_launch("route_trace");
}
void route_count(const int* topi, int n, unsigned* cnt, void* st) {
    k_route_count<<<(n + 255) / 256, 256, 0, (cudaStream_t) st>>>(topi, n, cnt);
    ck_launch("route_count");
}
void probe_guess(const float* logits, int T, int* guess, void* st) {
    k_probe_guess<<<1, 32 * MAXT, 0, (cudaStream_t) st>>>(logits, T, guess);
    ck_launch("probe_guess");
}
void probe_score(const int* topi, const int* guess, int T, const int* slot_of, unsigned long long* cnt, void* st) {
    k_probe_score<<<1, 128, 0, (cudaStream_t) st>>>(topi, guess, T, slot_of, cnt);
    ck_launch("probe_score");
}
// int8 K/V is prod's and the default; FNX_KV8=0: f16
bool kv_q8() {
    static const bool q8 = [] {
        const char* s = std::getenv("FNX_KV8");
        if (s && std::strcmp(s, "0") && std::strcmp(s, "1")) throw std::runtime_error(std::string("FNX_KV8: 0 or 1, not '") + s + "'");
        return !s || !std::strcmp(s, "1");
    }();
    return q8;
}
void kv_append(const float* k, const float* v, uint16_t* kc, uint16_t* vc, const KvPages& g, int card, int T, const int* pos0, void* st) {
    k_kv_append<<<dim3(1, T), 256, 0, (cudaStream_t) st>>>(k, v, reinterpret_cast<__half*>(kc), reinterpret_cast<__half*>(vc), g, card,
                                                           pos0, LaneRows{});
    ck_launch("kv_append");
}
void kv_append_lanes(const float* k, const float* v, uint16_t* kc, uint16_t* vc, const KvPages& g, int card, int T, const LaneRows& lr, void* st) {
    if (!lr.pos || !lr.pt) throw std::runtime_error("kv_append_lanes: no lane rows");
    k_kv_append<<<dim3(1, T), 256, 0, (cudaStream_t) st>>>(k, v, reinterpret_cast<__half*>(kc), reinterpret_cast<__half*>(vc), g, card,
                                                           nullptr, lr);
    ck_launch("kv_append_lanes");
}
void attn_prep_append(float* qg, float* k, const float* v, const float* q_norm, const float* k_norm, uint16_t* kc, uint16_t* vc,
                      const KvPages& g, int card, int T, const int* pos0, float eps, Rope base, void* st) {
    k_attn_prep_append<<<dim3(NQH_LOC + 2 * KVH_LOC, T), HD, 0, (cudaStream_t) st>>>(qg, k, v, q_norm, k_norm, reinterpret_cast<__half*>(kc),
                                                                          reinterpret_cast<__half*>(vc), g, card, pos0, eps, base);
    ck_launch("attn_prep_append");
}
}  // namespace fnx
#endif

#if FNX_KPART == 4

namespace fnx {
static std::atomic<bool> g_gdn_tc_default{true};
void gdn_tc_set_default(bool on) { g_gdn_tc_default = on; }
static std::atomic<bool> g_attn_i8_default{false};
void attn_i8_set_default(bool on) { g_attn_i8_default = on; }
bool attn_i8_default() { return g_attn_i8_default.load(); }
size_t gdn_scratch_floats(int nv) { return (size_t) MAXT * nv * HS + (size_t) nv * MAXT * GDN_Q + (size_t) MAXT * nv; }

void gdn_core(const float* qkvz, const float* ab, float* hist, const float* convw, const float* dt, const float* a, const float* nw,
              float* S, void* yq, const int* ctl, float* scratch, float* snap, float* hsnap, int nk, int nv, const int* vmap, int T,
              float eps, void* st, float* y32) {
    cudaStream_t s = (cudaStream_t) st;
    block_q8_1* y = (block_q8_1*) yq;
    int8_t* tq;
    float* td;
    twin_of(yq, nv * HS, tq, td);
#define FNX_GDN(t) k_gdn_core<t><<<nv * GDN_Q, GDN_THREADS, 0, s>>>(qkvz, ab, hist, convw, dt, a, nw, S, y, ctl, scratch, snap, hsnap, nk, nv, vmap, eps, tq, td, y32, GdnLanes{})
    switch (T) {
        case 1: FNX_GDN(1); break;
        case 2: FNX_GDN(2); break;
        case 3: FNX_GDN(3); break;
        case 4: FNX_GDN(4); break;
        case 5: FNX_GDN(5); break;
        case 6: FNX_GDN(6); break;
        case 7: FNX_GDN(7); break;
        case 8: FNX_GDN(8); break;
        case 9: FNX_GDN(9); break;
        case 10: FNX_GDN(10); break;
        case 11: FNX_GDN(11); break;
        case 12: FNX_GDN(12); break;
        case 13: FNX_GDN(13); break;
        case 14: FNX_GDN(14); break;
        case 15: FNX_GDN(15); break;
        case 16: FNX_GDN(16); break;
        default: throw std::runtime_error("gdn_core: T out of range");
    }
#undef FNX_GDN
    ck_launch("gdn_core");
}

void gdn_core_lanes(const float* qkvz, const float* ab, const float* convw, const float* dt, const float* a, const float* nw, void* yq,
                    float* scratch, int nk, int nv, const int* vmap, const GdnLanes& gl, int T, float eps, void* st) {
    if (T < 1 || T > MAXT) throw std::runtime_error("gdn_core_lanes: T out of range");
    if (!gl.S || !gl.hist || !gl.ctl) throw std::runtime_error("gdn_core_lanes: no lane rows");
    int8_t* tq;
    float* td;
    twin_of(yq, nv * HS, tq, td);
    k_gdn_core<1><<<dim3(nv * GDN_Q, T), GDN_THREADS, 0, (cudaStream_t) st>>>(qkvz, ab, nullptr, convw, dt, a, nw, nullptr, (block_q8_1*) yq,
                                                                              nullptr, scratch, nullptr, nullptr, nk, nv, vmap, eps, tq, td,
                                                                              nullptr, gl);
    ck_launch("gdn_core_lanes");
}

void gdn_core_segs(const float* qkvz, const float* ab, const float* convw, const float* dt, const float* a, const float* nw, void* yq,
                   float* scratch, int nk, int nv, const int* vmap, const GdnLanes& gl, int nseg, int T, float eps, void* st) {
    if (nseg < 1 || T < 1 || nseg * T > MAXT) throw std::runtime_error("gdn_core_segs: segments out of range");
    if (!gl.S || !gl.hist || !gl.ctl || !gl.row0 || !gl.snap || !gl.hsnap) throw std::runtime_error("gdn_core_segs: no segment rows");
    cudaStream_t s = (cudaStream_t) st;
    int8_t* tq;
    float* td;
    twin_of(yq, nv * HS, tq, td);
#define FNX_GDN(t) k_gdn_core<t><<<dim3(nv * GDN_Q, nseg), GDN_THREADS, 0, s>>>(qkvz, ab, nullptr, convw, dt, a, nw, nullptr, (block_q8_1*) yq, \
                                                                          nullptr, scratch, nullptr, nullptr, nk, nv, vmap, eps, tq, td, nullptr, gl)
    switch (T) {
        case 1: FNX_GDN(1); break;
        case 2: FNX_GDN(2); break;
        case 3: FNX_GDN(3); break;
        case 4: FNX_GDN(4); break;
        case 5: FNX_GDN(5); break;
        case 6: FNX_GDN(6); break;
        case 7: FNX_GDN(7); break;
        case 8: FNX_GDN(8); break;
        case 9: FNX_GDN(9); break;
        case 10: FNX_GDN(10); break;
        case 11: FNX_GDN(11); break;
        case 12: FNX_GDN(12); break;
        case 13: FNX_GDN(13); break;
        case 14: FNX_GDN(14); break;
        case 15: FNX_GDN(15); break;
        case 16: FNX_GDN(16); break;
        default: throw std::runtime_error("gdn_core_segs: T out of range");
    }
#undef FNX_GDN
    ck_launch("gdn_core_segs");
}

static void gdn_prompt_impl(const float* qkvz, const float* ab, float* hist, const float* convw, const float* dt, const float* a, const float* nw,
                       float* S, const int* ctl, int ng, int nk, int nv, const int* vmap, int C, float eps, float* xc, float* sq, float* y32,
                       float* chw, void* st) {
    cudaStream_t s = (cudaStream_t) st;
    const int ch = (2 * nk + nv) * HS;
    k_gdn_pf_conv<<<dim3(C, 2 * nk + nv), HS, 0, s>>>(qkvz, hist, convw, ctl, xc, nk, nv);
    k_gdn_pf_hist<<<(ch + 255) / 256, 256, 0, s>>>(qkvz, hist, ctl, ng, C, nk, nv);
#if FNX_GDN_TC_ON
    // FNX_PFX_GDN_TC: on by default on NVIDIA sm_80+ except for the IQ3_S-type sets (FNX_PFX_GDN_TC=0/1 overrides): Strata's tensor-core chunk scan (gdn_tc_prompt.cuh); false -> this card cannot run it, the sub-chunk kernels below do
    static const int gdn_tc_env = (int) fnx::env_int("FNX_PFX_GDN_TC", -1);   // -1: the file decides (gdn_tc_set_default)
    const bool gdn_tc = gdn_tc_env >= 0 ? gdn_tc_env != 0 : g_gdn_tc_default.load();
    if (gdn_tc && gdn_tc_run(xc, ab, dt, a, S, y32, nk, nv, vmap, C, s)) {
        k_gdn_pf_norm_own<<<dim3(C, nv), HS, 0, s>>>(qkvz, nw, y32, nk, nv, eps);
        ck_launch("gdn_prompt");
        return;
    }
#endif
    static const bool chunked = [] {
        const char* e = std::getenv("FNX_PF_GDN_CHUNK");
        if (e && std::strcmp(e, "0") && std::strcmp(e, "1")) throw std::runtime_error(std::string("FNX_PF_GDN_CHUNK: 0 or 1, not '") + e + "'");
        return !e || !std::strcmp(e, "1");
    }();
    if (chunked) {
        static thread_local bool smem_set[16] = {};
        int dev = 0;
        cudaGetDevice(&dev);
        if (!smem_set[dev]) {
            if (cudaFuncSetAttribute(k_gdn_ch_rec, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) GCH_REC_SMEM) != cudaSuccess)
                ck_launch("gdn_prompt: shared memory");
            smem_set[dev] = true;
        }
        // FNX_PF_GDN_CHECK=1 (debugging): the row-by-row kernel on the same inputs first, then how far the sub-chunk
        // path's outputs and end state are from its (largest difference over the largest value)
        static const bool check = std::getenv("FNX_PF_GDN_CHECK") != nullptr;
        std::vector<float> y_ref, s_ref;
        const size_t ns = (size_t) nv * HS * HS, ny = (size_t) C * nv * HS;
        if (check) {
            float *s0, *yr;
            if (cudaMalloc(&s0, ns * 4) != cudaSuccess || cudaMalloc(&yr, ny * 4) != cudaSuccess) ck_launch("gdn check: alloc");
            cudaMemcpyAsync(s0, S, ns * 4, cudaMemcpyDeviceToDevice, s);
            k_gdn_pf_rec<<<nv * GDN_Q, GDN_PF_THREADS, 0, s>>>(xc, ab, dt, a, S, yr, sq, nk, nv, vmap, C);
            y_ref.resize(ny), s_ref.resize(ns);
            cudaMemcpyAsync(y_ref.data(), yr, ny * 4, cudaMemcpyDeviceToHost, s);
            cudaMemcpyAsync(s_ref.data(), S, ns * 4, cudaMemcpyDeviceToHost, s);
            cudaMemcpyAsync(S, s0, ns * 4, cudaMemcpyDeviceToDevice, s);
            cudaStreamSynchronize(s);
            cudaFree(s0), cudaFree(yr);
        }
#if FNX_ATT_MM && FNX_HIP
        static const bool gdn_mm = env_int("FNX_PF_GDN_MM", APPROX) != 0;   // measured +8.3-9.3 % prompt reads
        if (gdn_mm) k_gdn_ch_prep<true><<<dim3((C + GCH - 1) / GCH, nv), GCH_THREADS, 0, s>>>(xc, ab, dt, a, vmap, nk, nv, C, chw);
        else
#endif
        k_gdn_ch_prep<<<dim3((C + GCH - 1) / GCH, nv), GCH_THREADS, 0, s>>>(xc, ab, dt, a, vmap, nk, nv, C, chw);
#if FNX_ATT_MM && FNX_HIP
        if (gdn_mm) k_gdn_ch_rec_mm<0><<<nv * GCH_REC_BLOCKS_PER_HEAD, GMM_THREADS, 0, s>>>(xc, chw, S, y32, nk, nv, vmap, C);
        else
#endif
        k_gdn_ch_rec<<<nv * GCH_REC_BLOCKS_PER_HEAD, GCR_THREADS, GCH_REC_SMEM, s>>>(xc, chw, S, y32, nk, nv, vmap, C);
        if (check) {
            std::vector<float> y(ny), sn(ns);
            cudaMemcpyAsync(y.data(), y32, ny * 4, cudaMemcpyDeviceToHost, s);
            cudaMemcpyAsync(sn.data(), S, ns * 4, cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);
            auto rel = [](const std::vector<float>& r, const std::vector<float>& x) {
                double md = 0, mv = 0;
                for (size_t i = 0; i < r.size(); ++i) md = std::max(md, (double) std::fabs(r[i] - x[i])), mv = std::max(mv, (double) std::fabs(r[i]));
                return md / std::max(mv, 1e-30);
            };
            std::fprintf(stderr, "gdn check: C %d, outputs %.3g, end state %.3g (largest difference over largest value)\n", C,
                         rel(y_ref, y), rel(s_ref, sn));
        }
        k_gdn_pf_norm_own<<<dim3(C, nv), HS, 0, s>>>(qkvz, nw, y32, nk, nv, eps);
    } else {
        k_gdn_pf_rec<<<nv * GDN_Q, GDN_PF_THREADS, 0, s>>>(xc, ab, dt, a, S, y32, sq, nk, nv, vmap, C);
        k_gdn_pf_norm<<<dim3(C, nv), HS, 0, s>>>(qkvz, nw, sq, y32, nk, nv, eps);
    }
    ck_launch("gdn_prompt");
}
// FNX_GDN_TIME=1 (measuring): CUDA events around every prompt GDN call of a card (conv, history, scan, norm), their GPU time summed per
// card and printed to stderr each time 36 x 32768 rows (a 32K prompt through the 36 recurrent layers) have gone through it (the events are read a call late, nothing waits on the GPU)
void gdn_prompt(const float* qkvz, const float* ab, float* hist, const float* convw, const float* dt, const float* a, const float* nw,
                float* S, const int* ctl, int ng, int nk, int nv, const int* vmap, int C, float eps, float* xc, float* sq, float* y32,
                float* chw, void* st) {
    static const bool timed = fnx::env_int("FNX_GDN_TIME", 0) != 0;
    if (!timed) {
        gdn_prompt_impl(qkvz, ab, hist, convw, dt, a, nw, S, ctl, ng, nk, nv, vmap, C, eps, xc, sq, y32, chw, st);
        return;
    }
    struct Rec { cudaEvent_t e0, e1; int rows; };
    struct Dev { std::vector<Rec> pend; double ms = 0; long rows = 0, calls = 0, since = 0; };
    static thread_local Dev devs[16];
    int dev = 0;
    cudaGetDevice(&dev);
    Dev& d = devs[dev & 15];
    cudaStream_t s = (cudaStream_t) st;
    Rec r{nullptr, nullptr, C};
    cudaEventCreate(&r.e0), cudaEventCreate(&r.e1);
    cudaEventRecord(r.e0, s);
    gdn_prompt_impl(qkvz, ab, hist, convw, dt, a, nw, S, ctl, ng, nk, nv, vmap, C, eps, xc, sq, y32, chw, st);
    cudaEventRecord(r.e1, s);
    d.pend.push_back(r);
    while (!d.pend.empty() && cudaEventQuery(d.pend.front().e1) == cudaSuccess) {
        Rec q = d.pend.front();
        d.pend.erase(d.pend.begin());
        float ms = 0;
        cudaEventElapsedTime(&ms, q.e0, q.e1);
        d.ms += ms, d.rows += q.rows, d.since += q.rows, ++d.calls;
        cudaEventDestroy(q.e0), cudaEventDestroy(q.e1);
        if (d.since >= 36L * 32768) {
            std::fprintf(stderr, "gdn time: card %d: %.1f ms GPU over %ld calls, %ld rows (%.4f ms a row)\n", dev, d.ms, d.calls, d.rows, d.ms / (double) d.rows);
            d.ms = 0, d.rows = d.calls = d.since = 0;
        }
    }
}

size_t gdn_prompt_work_floats(int C, int nv) { return (size_t) ((C + GCH - 1) / GCH) * nv * GCH_PREP_FLOATS; }
}  // namespace fnx
#endif

namespace fnx {
#if FNX_KPART == 8
void router_route(const float* W, const float* x, float* logits, int T, int* topi, float* topw, Job* jobs_gu, Job* jobs_dn,
                  int* njobs, void* st, const ExpertCache* ec, int layer, int* list, const Q8S* sh_gu, int sh_rows, const void* sh_y,
                  float* sh_out, const uint16_t* Wbf
#if FNX_MALL_PF
                  ,
                  unsigned* pf_word, int pf_late
#endif
) {
    cudaStream_t s = (cudaStream_t) st;
    ShexpUp su;
    if (sh_gu) {
        if (sh_rows < 1 || sh_rows >= 4096) throw std::runtime_error("router_route: the shared gate|up takes 1..4095 rows");
        for (int k = 0; k < Q8S_SEGS; ++k) su.w.q[k] = sh_gu->q, su.w.d[k] = sh_gu->d, su.w.out[k] = sh_out, su.w.end[k] = sh_rows, su.w.stride[k] = sh_rows;
        su.rows = sh_rows, su.cols = N, su.blocks = (sh_rows + Q8S_THREADS / 32 - 1) / (Q8S_THREADS / 32);
        su.y = (const block_q8_1*) sh_y;
        int8_t* tq;
        float* td;
        twin_of(sh_y, N, tq, td);
        su.tq = tq, su.td = td;
    }
    unsigned* counter = (unsigned*) (njobs + 1);
    const ExpertCache none{};
    if (ec && (layer < 0 || layer >= NL || !list)) throw std::runtime_error("router_route: a cache plan needs its layer and list");
// with the prefetch build the kernel has two more arguments, the persistent prefetcher's count (null: none) and where it adds one
#if FNX_MALL_PF
#define FNX_RR_PF , pf_word, pf_late
#else
#define FNX_RR_PF
#endif
#if FNX_ROUTER_BF16
    // the bf16 table when the layer has one (model.cpp: every value exact) and the switch is on; else the f32 table and the kernel as it was
    const int bf = Wbf ? router_bf16_mode() : 0;
    const int grid = (NEXP + 1 + rr_rows(bf) - 1) / rr_rows(bf) + su.blocks;
    const float* Wk = bf ? reinterpret_cast<const float*>(Wbf) : W;
#define FNX_RR(t)                                                                                                                                   \
    do {                                                                                                                                            \
        if (bf == 2) k_router_route<t, 2><<<grid, RR_THREADS, 0, s>>>(Wk, x, logits, topi, topw, jobs_gu, jobs_dn, njobs, counter, ec ? *ec : none, layer, list, su FNX_RR_PF); \
        else if (bf == 1) k_router_route<t, 1><<<grid, RR_THREADS, 0, s>>>(Wk, x, logits, topi, topw, jobs_gu, jobs_dn, njobs, counter, ec ? *ec : none, layer, list, su FNX_RR_PF); \
        else k_router_route<t, 0><<<grid, RR_THREADS, 0, s>>>(Wk, x, logits, topi, topw, jobs_gu, jobs_dn, njobs, counter, ec ? *ec : none, layer, list, su FNX_RR_PF); \
    } while (0)
#else
    (void) Wbf;
    const int grid = (NEXP + 1 + RR_ROWS - 1) / RR_ROWS + su.blocks;
#define FNX_RR(t) k_router_route<t><<<grid, RR_THREADS, 0, s>>>(W, x, logits, topi, topw, jobs_gu, jobs_dn, njobs, counter, ec ? *ec : none, layer, list, su FNX_RR_PF)
#endif
    switch (T) {
        case 1: FNX_RR(1); break;
        case 2: FNX_RR(2); break;
        case 3: FNX_RR(3); break;
        case 4: FNX_RR(4); break;
        case 5: FNX_RR(5); break;
        case 6: FNX_RR(6); break;
        case 7: FNX_RR(7); break;
        case 8: FNX_RR(8); break;
        case 9: FNX_RR(9); break;
        case 10: FNX_RR(10); break;
        case 11: FNX_RR(11); break;
        case 12: FNX_RR(12); break;
        case 13: FNX_RR(13); break;
        case 14: FNX_RR(14); break;
        case 15: FNX_RR(15); break;
        case 16: FNX_RR(16); break;
        default: throw std::runtime_error("router_route: T out of range");
    }
#undef FNX_RR
#undef FNX_RR_PF
    ck_launch("router_route");
}

// FNX_ROUTER_BF16 (kernels.hpp): the environment is read once; router_bf16_force is kbench's override
namespace {
int router_bf16_force_ = -1;
}
void router_bf16_force(int mode) { router_bf16_force_ = mode; }
int router_bf16_mode() {
#if FNX_ROUTER_BF16
    static const int env = [] {
        const long long m = env_int("FNX_ROUTER_BF16", 1);
        if (m < 0 || m > 2) throw std::runtime_error("FNX_ROUTER_BF16: 0 (the f32 table), 1 (bf16) or 2 (bf16, one warp a row), got " + std::to_string(m));
        return (int) m;
    }();
    return router_bf16_force_ >= 0 ? (router_bf16_force_ > 2 ? 2 : router_bf16_force_) : env;
#else
    return 0;
#endif
}

#endif
#if FNX_KPART == 2
namespace {
void moe_mm(bool down, const DW& A, size_t sa, const Job* jobs, const int* njobs, int njobs_max, const void* in, void* out,
            void* st, int tt = 0) {
    const int T = tt ? tt : njobs_max / TOPK;   // the most rows a job has
    const MoeW w{(const char*) A.p, sa, A.row_bytes, A.n_out, A.n_in, A.ep};
    cudaStream_t s = (cudaStream_t) st;
    switch (A.type) {
        case T_Q4_K:
            if (down) moe_launch<T_Q4_K>(down, w, jobs, njobs, njobs_max, in, (float*) out, s, T);
            else gu_q4k_launch(T, w, jobs, njobs, njobs_max, (const block_q8_1*) in, (float*) out, s);
            break;
        case T_Q5_K: moe_launch<T_Q5_K>(down, w, jobs, njobs, njobs_max, in, (float*) out, s, T); break;
        case T_Q5_1: moe_launch<T_Q5_1>(down, w, jobs, njobs, njobs_max, in, (float*) out, s, T); break;
        case T_Q4_0: moe_launch<T_Q4_0>(down, w, jobs, njobs, njobs_max, in, (float*) out, s, T); break;
        case T_Q8_0: moe_launch<T_Q8_0>(down, w, jobs, njobs, njobs_max, in, (float*) out, s, T); break;
        case T_IQ4_NL: moe_launch<T_IQ4_NL>(down, w, jobs, njobs, njobs_max, in, (float*) out, s, T); break;
        case T_IQ4_XS: moe_launch<T_IQ4_XS>(down, w, jobs, njobs, njobs_max, in, (float*) out, s, T); break;
        case T_IQ3_S: moe_launch<T_IQ3_S>(down, w, jobs, njobs, njobs_max, in, (float*) out, s, T); break;
        case T_IQ3_XXS: moe_launch<T_IQ3_XXS>(down, w, jobs, njobs, njobs_max, in, (float*) out, s, T); break;
        case T_IQ2_S: moe_launch<T_IQ2_S>(down, w, jobs, njobs, njobs_max, in, (float*) out, s, T); break;
        default: throw std::runtime_error("moe: expert type " + std::to_string(A.type) + " not handled");
    }
    ck_launch(down ? "moe_down" : "moe_gate_up");
}

__global__ void k_swiglu_stacked(const float* gu, block_q8_1* hq, int ffs, int8_t* tq, float* td) {
    const int t = blockIdx.y, b = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b >= ffs / 32) return;
    const float* g = gu + (size_t) t * 2 * ffs;
    const float xi = silu_(g[b * 32 + lane]) * g[ffs + b * 32 + lane];
    const float amax = wmax(fabsf(xi)), sum = wsum(xi);
    const float d = amax / 127.0f;
    block_q8_1& y = hq[(size_t) t * (ffs / 32) + b];
    const int8_t qv = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
    y.qs[lane] = qv;
    if (lane == 0) y.ds = make_half2(__float2half(d), __float2half(sum));
    if (tq) {
        tq[(size_t) t * ffs + b * 32 + lane] = qv;
        if (lane == 0) td[(size_t) t * (ffs / 32) + b] = __half2float(__float2half(d));
    }
}
}  // namespace

void moe_gate_up(const DW& A, size_t sa, const Job* jobs, const int* njobs, int njobs_max, const void* xq, float* gu, void* st,
                 const Q8S* sh_down, int sh_ffs, const float* sh_gu, float* sh_out) {
    if (!sh_down) {
        moe_mm(false, A, sa, jobs, njobs, njobs_max, xq, gu, st);
        return;
    }
    if (sh_ffs < 32 || sh_ffs > SH_MAXF || sh_ffs % 32) throw std::runtime_error("moe_gate_up: the shared expert's FF slice");
    ShexpDown sd;
    for (int k = 0; k < Q8S_SEGS; ++k) sd.w.q[k] = sh_down->q, sd.w.d[k] = sh_down->d, sd.w.out[k] = sh_out, sd.w.end[k] = N, sd.w.stride[k] = N;
    sd.rows = N, sd.ffs = sh_ffs, sd.gu = sh_gu;
    const MoeW w{(const char*) A.p, sa, A.row_bytes, A.n_out, A.n_in, A.ep};
    const int T = njobs_max / TOPK;
    cudaStream_t s = (cudaStream_t) st;
    const block_q8_1* y = (const block_q8_1*) xq;
    switch (A.type) {
        case T_Q4_K: gu_q4k_launch(T, w, jobs, njobs, njobs_max, y, gu, s, sd); break;
        case T_Q5_K: gu_sd_launch<T_Q5_K>(T, w, jobs, njobs, njobs_max, y, gu, s, sd); break;
        case T_Q5_1: gu_sd_launch<T_Q5_1>(T, w, jobs, njobs, njobs_max, y, gu, s, sd); break;
        case T_Q4_0: gu_sd_launch<T_Q4_0>(T, w, jobs, njobs, njobs_max, y, gu, s, sd); break;
        case T_Q8_0: gu_sd_launch<T_Q8_0>(T, w, jobs, njobs, njobs_max, y, gu, s, sd); break;
        case T_IQ4_NL: gu_sd_launch<T_IQ4_NL>(T, w, jobs, njobs, njobs_max, y, gu, s, sd); break;
        case T_IQ4_XS: gu_sd_launch<T_IQ4_XS>(T, w, jobs, njobs, njobs_max, y, gu, s, sd); break;
        case T_IQ3_S: gu_sd_launch<T_IQ3_S>(T, w, jobs, njobs, njobs_max, y, gu, s, sd); break;
        case T_IQ3_XXS: gu_sd_launch<T_IQ3_XXS>(T, w, jobs, njobs, njobs_max, y, gu, s, sd); break;
        case T_IQ2_S: gu_sd_launch<T_IQ2_S>(T, w, jobs, njobs, njobs_max, y, gu, s, sd); break;
        default: throw std::runtime_error("moe_gate_up: the shared expert rides in the launch of expert type " + std::to_string(A.type) + " only for the types moe_mm handles");
    }
    ck_launch("moe_gate_up (with the shared expert)");
}
namespace {
void moe_mm_big(bool down, const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const void* in, void* out,
                void* st) {
    const MoeW w{(const char*) A.p, sa, A.row_bytes, A.n_out, A.n_in, A.ep};
    cudaStream_t s = (cudaStream_t) st;
    switch (A.type) {
        case T_Q4_K:
            if (down) moe_launch_big<T_Q4_K>(down, w, jobs, njobs, njobs_cap, in, (float*) out, s);
            else {
                if (w.ncols != N || w.row_a != (size_t) N / 256 * 144) throw std::runtime_error("moe gate|up q4_K: row layout");
                k_moe_gu_q4k_big<<<dim3((w.nrows + Q4K_ROWS - 1) / Q4K_ROWS, njobs_cap), Q4K_THREADS, 0, s>>>(w, jobs, njobs,
                                                                                                         (const block_q8_1*) in, (float*) out);
            }
            break;
        case T_Q5_K: moe_launch_big<T_Q5_K>(down, w, jobs, njobs, njobs_cap, in, (float*) out, s); break;
        case T_Q5_1: moe_launch_big<T_Q5_1>(down, w, jobs, njobs, njobs_cap, in, (float*) out, s); break;
        case T_Q4_0: moe_launch_big<T_Q4_0>(down, w, jobs, njobs, njobs_cap, in, (float*) out, s); break;
        case T_Q8_0: moe_launch_big<T_Q8_0>(down, w, jobs, njobs, njobs_cap, in, (float*) out, s); break;
        case T_IQ4_NL: moe_launch_big<T_IQ4_NL>(down, w, jobs, njobs, njobs_cap, in, (float*) out, s); break;
        case T_IQ4_XS: moe_launch_big<T_IQ4_XS>(down, w, jobs, njobs, njobs_cap, in, (float*) out, s); break;
        case T_IQ3_S: moe_launch_big<T_IQ3_S>(down, w, jobs, njobs, njobs_cap, in, (float*) out, s); break;
        case T_IQ3_XXS: moe_launch_big<T_IQ3_XXS>(down, w, jobs, njobs, njobs_cap, in, (float*) out, s); break;
        case T_IQ2_S: moe_launch_big<T_IQ2_S>(down, w, jobs, njobs, njobs_cap, in, (float*) out, s); break;
        default: throw std::runtime_error("moe: expert type " + std::to_string(A.type) + " not handled");
    }
    ck_launch(down ? "moe_down (big jobs)" : "moe_gate_up (big jobs)");
}
}  // namespace

void moe_gate_up_big(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const void* xq, float* gu, void* st) {
    moe_mm_big(false, A, sa, jobs, njobs, njobs_cap, xq, gu, st);
}
void moe_down_big(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const float* gu, float* part, void* st) {
    moe_mm_big(true, A, sa, jobs, njobs, njobs_cap, gu, part, st);
}

void moe_gate_up_jobs(const DW& A, size_t sa, const Job* jobs, const int* njobs, int njobs_cap, const void* xq, float* gu, void* st) {
    moe_mm(false, A, sa, jobs, njobs, njobs_cap, xq, gu, st, MAXT);
}
void moe_down(const DW& A, size_t sa, const Job* jobs, const int* njobs, int njobs_max, const float* gu, float* part, void* st) {
    moe_mm(true, A, sa, jobs, njobs, njobs_max, gu, part, st);
}
void swiglu_stacked_q8_1(const float* gu, void* hq, int ffs, int T, void* st) {
    int8_t* tq;
    float* td;
    twin_of(hq, ffs, tq, td);
    k_swiglu_stacked<<<dim3((ffs / 32 + 7) / 8, T), 256, 0, (cudaStream_t) st>>>(gu, (block_q8_1*) hq, ffs, tq, td);
    ck_launch("swiglu_stacked_q8_1");
}
#endif
}  // namespace fnx

namespace fnx {
#if FNX_KPART == 6
namespace {
template <int R, bool SPLIT, bool SEG>
void q8s_launch(const Q8Segs& g, int rows, int cols, const block_q8_1* y, const int8_t* tq, const float* td, int T, cudaStream_t s) {
    const dim3 grid((rows + Q8S_THREADS / 32 * R - 1) / (Q8S_THREADS / 32 * R));
#define FNX_Q8S(t) k_q8s<t, R, SPLIT, SEG><<<grid, Q8S_THREADS, 0, s>>>(g, rows, cols, y, tq, td)
    switch (T) {
        case 1: FNX_Q8S(1); break;
        case 2: FNX_Q8S(2); break;
        case 3: FNX_Q8S(3); break;
        case 4: FNX_Q8S(4); break;
        case 5: FNX_Q8S(5); break;
        case 6: FNX_Q8S(6); break;
        case 7: FNX_Q8S(7); break;
        case 8: FNX_Q8S(8); break;
        case 9: FNX_Q8S(9); break;
        case 10: FNX_Q8S(10); break;
        case 11: FNX_Q8S(11); break;
        case 12: FNX_Q8S(12); break;
        case 13: FNX_Q8S(13); break;
        case 14: FNX_Q8S(14); break;
        case 15: FNX_Q8S(15); break;
        case 16: FNX_Q8S(16); break;
        default: throw std::runtime_error("q8s_gemv: T out of range");
    }
#undef FNX_Q8S
}
template <int R, bool SEG>
void q8s_launch(const Q8Segs& g, int rows, int cols, const block_q8_1* y, int T, cudaStream_t s) {
    int8_t* tq;
    float* td;
    twin_of(y, cols, tq, td);
    if (tq) q8s_launch<R, true, SEG>(g, rows, cols, y, tq, td, T, s);
    else q8s_launch<R, false, SEG>(g, rows, cols, y, nullptr, nullptr, T, s);
}
#if FNX_MM && FNX_HIP
// the matrix kernel (dense.cuh q8s_mm_block): a block per 16 rows
template <bool SPLIT, bool SEG>
void q8s_mm_launch(const Q8Segs& g, int rows, int cols, const block_q8_1* y, const int8_t* tq, const float* td, int T, cudaStream_t s) {
    const dim3 grid((rows + 15) / 16);
#define FNX_Q8M(t) k_q8s_mm<t, SPLIT, SEG><<<grid, Q8S_THREADS, 0, s>>>(g, rows, cols, y, tq, td)
    switch (T) {
        case 1: FNX_Q8M(1); break;
        case 2: FNX_Q8M(2); break;
        case 3: FNX_Q8M(3); break;
        case 4: FNX_Q8M(4); break;
        case 5: FNX_Q8M(5); break;
        case 6: FNX_Q8M(6); break;
        case 7: FNX_Q8M(7); break;
        case 8: FNX_Q8M(8); break;
        case 9: FNX_Q8M(9); break;
        case 10: FNX_Q8M(10); break;
        case 11: FNX_Q8M(11); break;
        case 12: FNX_Q8M(12); break;
        case 13: FNX_Q8M(13); break;
        case 14: FNX_Q8M(14); break;
        case 15: FNX_Q8M(15); break;
        case 16: FNX_Q8M(16); break;
        default: throw std::runtime_error("q8s_gemv: T out of range");
    }
#undef FNX_Q8M
}
#endif
template <bool SEG>
void q8s_run(const Q8Segs& g, int rows, int cols, const void* y, int T, void* st) {
    if (cols % 16) throw std::runtime_error("q8s_gemv: columns must be a multiple of 16");
    const block_q8_1* yq = (const block_q8_1*) y;
    cudaStream_t s = (cudaStream_t) st;
#if FNX_MM && FNX_HIP
    if (cols % 32 == 0 && mm_dense(T)) {
        int8_t* tq;
        float* td;
        twin_of(yq, cols, tq, td);
        if (tq) q8s_mm_launch<true, SEG>(g, rows, cols, yq, tq, td, T, s);
        else q8s_mm_launch<false, SEG>(g, rows, cols, yq, nullptr, nullptr, T, s);
        ck_launch("q8s_gemv");
        return;
    }
#endif
    // rows per warp: four for big matrices (activation loads shared by more rows), one when that leaves few blocks. The
    // segmented variant only for its one use, q/k/v (8192 rows and more): fewer kernels to compile
    if constexpr (SEG) {
#if FNX_ATT_HEADS
        // by heads q, k and v are 3584 rows a card: the rows per warp q8s_gemv takes for that many (each matrix's output as it alone gives)
        if (rows >= 8192) q8s_launch<4, SEG>(g, rows, cols, yq, T, s);
        else if (rows >= 4096) q8s_launch<2, SEG>(g, rows, cols, yq, T, s);
        else q8s_launch<1, SEG>(g, rows, cols, yq, T, s);
#else
        if (rows < 8192) throw std::runtime_error("q8s_gemv_segs: compiled for 8192 rows or more (q, k and v)");
        q8s_launch<4, SEG>(g, rows, cols, yq, T, s);
#endif
    } else {
#if FNX_27B
        // up to 4 rows by the window (the 27B's windows, bin/s27_run S27_BENCH=1: 1 row 19.4 -> 18.3 ms, 4 rows
        // 24.5 -> 22.3), past that by the matrix (5 to 16 rows: choosing by the window was no faster)
        if (T <= 2) q8s_launch<2, SEG>(g, rows, cols, yq, T, s);
        else if (T <= 4) q8s_launch<1, SEG>(g, rows, cols, yq, T, s);
        else if (rows >= 8192) q8s_launch<4, SEG>(g, rows, cols, yq, T, s);
#else
        if (rows >= 8192) q8s_launch<4, SEG>(g, rows, cols, yq, T, s);
#endif
        else if (rows >= 4096) q8s_launch<2, SEG>(g, rows, cols, yq, T, s);
        else q8s_launch<1, SEG>(g, rows, cols, yq, T, s);
    }
    ck_launch("q8s_gemv");
}
}  // namespace

void q8s_gemv(const Q8S& w, int rows, int cols, const void* y, float* out, int out_stride, int T, void* st) {
    Q8Segs g{};
    for (int s = 0; s < Q8S_SEGS; ++s) g.q[s] = w.q, g.d[s] = w.d, g.out[s] = out, g.end[s] = rows, g.stride[s] = out_stride;
    q8s_run<false>(g, rows, cols, y, T, st);
}

void q8s_gemv_segs(const Q8S* w, const int* rows, float* const* out, const int* out_stride, int n, int cols, const void* y, int T,
                   void* st) {
    if (n < 1 || n > Q8S_SEGS) throw std::runtime_error("q8s_gemv_segs: 1.." + std::to_string(Q8S_SEGS) + " matrices");
    Q8Segs g{};
    int end = 0;
    for (int s = 0; s < Q8S_SEGS; ++s) {
        const int k = std::min(s, n - 1);   // past n: the last segment again (no row reaches it)
        if (s < n) end += rows[s];
        g.q[s] = w[k].q, g.d[s] = w[k].d, g.out[s] = out[k], g.end[s] = end, g.stride[s] = out_stride[k];
    }
    q8s_run<true>(g, end, cols, y, T, st);
}

#endif
#if FNX_KPART == 7
namespace {
template <int R, bool SPLIT>
void q8s_ab_launch(const Q8Segs& g, int rows, int cols, const block_q8_1* y, const int8_t* tq, const float* td, const float* W,
                   const float* x, float* yf, int rf, int T, cudaStream_t s) {
    const dim3 grid(rf + (rows + Q8S_THREADS / 32 * R - 1) / (Q8S_THREADS / 32 * R));
#define FNX_QA(t) k_q8s_ab<t, R, SPLIT><<<grid, Q8S_THREADS, 0, s>>>(g, rows, cols, y, tq, td, W, x, yf, rf)
    switch (T) {
        case 1: FNX_QA(1); break;
        case 2: FNX_QA(2); break;
        case 3: FNX_QA(3); break;
        case 4: FNX_QA(4); break;
        case 5: FNX_QA(5); break;
        case 6: FNX_QA(6); break;
        case 7: FNX_QA(7); break;
        case 8: FNX_QA(8); break;
        case 9: FNX_QA(9); break;
        case 10: FNX_QA(10); break;
        case 11: FNX_QA(11); break;
        case 12: FNX_QA(12); break;
        case 13: FNX_QA(13); break;
        case 14: FNX_QA(14); break;
        case 15: FNX_QA(15); break;
        case 16: FNX_QA(16); break;
        default: throw std::runtime_error("q8s_gemv_f32: T out of range");
    }
#undef FNX_QA
}
template <int R>
void q8s_ab_launch(const Q8Segs& g, int rows, int cols, const block_q8_1* y, const float* W, const float* x, float* yf, int rf, int T,
                   cudaStream_t s) {
    int8_t* tq;
    float* td;
    twin_of(y, cols, tq, td);
    if (tq) q8s_ab_launch<R, true>(g, rows, cols, y, tq, td, W, x, yf, rf, T, s);
    else q8s_ab_launch<R, false>(g, rows, cols, y, nullptr, nullptr, W, x, yf, rf, T, s);
}
}  // namespace

void q8s_gemv_f32(const Q8S& w, int rows, int cols, const void* y, float* out, const float* W, const float* x, float* yf, int rf, int T,
                  void* st) {
    if (cols % 16) throw std::runtime_error("q8s_gemv_f32: columns must be a multiple of 16");
    if (rf < 1 || rf > 160) throw std::runtime_error("q8s_gemv_f32: the f32 matrix takes 1..160 rows (a block a row)");
    Q8Segs g{};
    for (int s = 0; s < Q8S_SEGS; ++s) g.q[s] = w.q, g.d[s] = w.d, g.out[s] = out, g.end[s] = rows, g.stride[s] = rows;
    const block_q8_1* yq = (const block_q8_1*) y;
    cudaStream_t s = (cudaStream_t) st;
#if FNX_MM && FNX_HIP
    if (cols % 32 == 0 && mm_dense(T)) {   // the matrix kernel's blocks after the f32 rows' (any row count)
        int8_t* tq;
        float* td;
        twin_of(yq, cols, tq, td);
        const dim3 grid(rf + (rows + 15) / 16);
#define FNX_QAM(t)                                                                                                    \
    if (tq) k_q8s_ab_mm<t, true><<<grid, Q8S_THREADS, 0, s>>>(g, rows, cols, yq, tq, td, W, x, yf, rf);                \
    else k_q8s_ab_mm<t, false><<<grid, Q8S_THREADS, 0, s>>>(g, rows, cols, yq, nullptr, nullptr, W, x, yf, rf)
        switch (T) {
            case 1: FNX_QAM(1); break;
            case 2: FNX_QAM(2); break;
            case 3: FNX_QAM(3); break;
            case 4: FNX_QAM(4); break;
            case 5: FNX_QAM(5); break;
            case 6: FNX_QAM(6); break;
            case 7: FNX_QAM(7); break;
            case 8: FNX_QAM(8); break;
            case 9: FNX_QAM(9); break;
            case 10: FNX_QAM(10); break;
            case 11: FNX_QAM(11); break;
            case 12: FNX_QAM(12); break;
            case 13: FNX_QAM(13); break;
            case 14: FNX_QAM(14); break;
            case 15: FNX_QAM(15); break;
            case 16: FNX_QAM(16); break;
            default: throw std::runtime_error("q8s_gemv_f32: T out of range");
        }
#undef FNX_QAM
        ck_launch("q8s_gemv_f32");
        return;
    }
#endif
    // compiled for its one use, qkvz, as q8s_gemv alone would run them: 4096 to 8191 rows two a warp; the 27B's 8192
    // and more (on two cards) four
#if FNX_27B
    if (rows < 4096) throw std::runtime_error("q8s_gemv_f32: compiled for 4096 rows or more (qkvz)");
    if (rows >= 8192) q8s_ab_launch<4>(g, rows, cols, yq, W, x, yf, rf, T, s);
    else q8s_ab_launch<2>(g, rows, cols, yq, W, x, yf, rf, T, s);
#else
    if (rows < 4096 || rows >= 8192) throw std::runtime_error("q8s_gemv_f32: compiled for 4096..8191 rows (qkvz)");
    q8s_ab_launch<2>(g, rows, cols, yq, W, x, yf, rf, T, s);
#endif
    ck_launch("q8s_gemv_f32");
}
#endif
}  // namespace fnx

#if FNX_KPART == 0
namespace fnx {
namespace {
// one load per 32-byte sector (L2 fills sectors, not whole lines)
__global__ void k_l2_prefetch(const char* p, size_t sectors, unsigned* sink) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < sectors; i += (size_t) gridDim.x * blockDim.x) {
        unsigned v;
#if FNX_HIP
        v = __ldcg(reinterpret_cast<const unsigned*>(p + i * 32));
#else
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p + i * 32));
#endif
        acc ^= v;
    }
    if (acc == 0x9e3779b9u) *sink = acc;   // keeps the loads; never true in practice
}
}  // namespace
void l2_prefetch(const void* p, size_t bytes, void* st) {
    const size_t sectors = (bytes + 31) / 32;
    static unsigned* sink = nullptr;
    if (!sink && cudaMalloc(&sink, 4) != cudaSuccess) throw std::runtime_error("l2_prefetch: sink");
    k_l2_prefetch<<<(unsigned) std::min<size_t>(160, (sectors + 255) / 256), 256, 0, (cudaStream_t) st>>>((const char*) p, sectors, sink);
    ck_launch("l2_prefetch");
}

// which kernels the fixed-weight products take (kernels.hpp). The environment is read once; mm_force is kbench's override
namespace {
int mm_force_dense_ = -1, mm_force_hc_ = -1;
#if FNX_MM && FNX_HIP
// the matrix kernels are gfx12 code (mm.cuh: on another family the instruction is a trap): is the current card one? A card of another family keeps the
// dp4a kernels and k_hc whatever the switches and mm_force say (a build may hold gfx11 code objects too). Asked once a card.
bool mm_gfx12() {
    static std::atomic<int> known[64] = {};   // 0 not asked, 1 gfx12, -1 not
    int dev = -1;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) return false;
    int k = known[dev].load(std::memory_order_relaxed);
    if (!k) {
        hipDeviceProp_t p{};
        k = hipGetDeviceProperties(&p, dev) == hipSuccess && std::strncmp(p.gcnArchName, "gfx12", 5) == 0 ? 1 : -1;
        known[dev].store(k, std::memory_order_relaxed);
    }
    return k > 0;
}
#endif
}  // namespace
void mm_force(int dense, int hc) { mm_force_dense_ = dense, mm_force_hc_ = hc; }
bool mm_dense(int T) {
#if FNX_MM && FNX_HIP
    if (!mm_gfx12()) return false;
    static const long long min_rows = [] {
        const long long r = env_int("FNX_MM_DENSE", 0);
        if (r < 0 || r > MAXT) throw std::runtime_error("FNX_MM_DENSE: 0 (off) or the window's rows from which the matrix kernel runs, 1.." +
                                                        std::to_string(MAXT) + ", got " + std::to_string(r));
        return r;
    }();
    if (mm_force_dense_ >= 0) return mm_force_dense_ != 0;
    return min_rows > 0 && T >= min_rows;
#else
    (void) T;
    return false;
#endif
}
bool mm_hc() {
#if FNX_MM_HC && FNX_HIP
    if (!mm_gfx12()) return false;
    static const bool on = env_int("FNX_MM_HC", APPROX) != 0;
    return mm_force_hc_ >= 0 ? mm_force_hc_ != 0 : on;
#else
    return false;
#endif
}
// generator: the engine's start. Reads the run's kernel variants (src/tk.hpp: FNX_TK_DIR, FNX_TK_<FAMILY>) and checks that each selected
// one's launch path is on in this run: an arm must not run the built-in kernel under a variant's name. (moe_gu_q4k has no such check: it takes the
// layers whose experts' gate|up weights are Q4_K, and a file may hold other types too.)
void tk_start() {
    tk::init({"pf_attn_u", "hc_mm", "moe_gu_q4k", "gemm8"});   // (the families with a launch site in this file, in moe_q4k.cuh and in prefill.cu)
    if (!mm_hc()) tk::refuse("hc_mm", "the ticket-queue hc read is off in this run (FNX_MM_HC=1 turns it on)");
    if (attn_prompt_union_rows() == 0) tk::refuse("pf_attn_u", "the prompt attention's union walk is off in this run (FNX_PF_ATT_UNION=0, or a build without it)");
}
}  // namespace fnx

// the Infinity Cache prefetch (kernels.hpp PfSet)
#ifndef FNX_MALL_PF
#define FNX_MALL_PF 0
#endif
namespace fnx {
#if FNX_MALL_PF && FNX_HIP
namespace {
typedef unsigned pf_v4 __attribute__((ext_vector_type(4)));
typedef pf_v4 pf_v4_u __attribute__((aligned(1)));
typedef unsigned pf_u32_u __attribute__((aligned(1)));
// lane i0 of T lanes touches its share of `bytes` bytes at p (grid-striding); the xor of what it loaded
__device__ __forceinline__ unsigned pf_touch_range(const char* p, size_t bytes, int line_bytes, int full, size_t T, size_t i0) {
    unsigned acc = 0;
    if (full) {   // every byte, 8 loads of 16 bytes in flight a lane
        const pf_v4_u* q = reinterpret_cast<const pf_v4_u*>(p);
        const size_t n16 = bytes / 16;
        size_t i = i0;
#pragma unroll 1
        for (; i + 7 * T < n16; i += 8 * T) {
            const pf_v4 a0 = q[i], a1 = q[i + T], a2 = q[i + 2 * T], a3 = q[i + 3 * T], a4 = q[i + 4 * T], a5 = q[i + 5 * T], a6 = q[i + 6 * T],
                        a7 = q[i + 7 * T];
            const pf_v4 x = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
            acc ^= x.x ^ x.y ^ x.z ^ x.w;
        }
#pragma unroll 1
        for (; i < n16; i += T) {
            const pf_v4 x = q[i];
            acc ^= x.x ^ x.y ^ x.z ^ x.w;
        }
    } else {        // one word a line: the load brings the whole line into the cache; 4 loads in flight a lane
        const size_t line = (size_t) line_bytes, lines = (bytes + line - 1) / line, last = bytes - 4;
        const auto at = [&](size_t l) { const size_t o = l * line; return *reinterpret_cast<const pf_u32_u*>(p + (o < last ? o : last)); };
        size_t l = i0;
#pragma unroll 1
        for (; l + 3 * T < lines; l += 4 * T) {
            const unsigned a0 = at(l), a1 = at(l + T), a2 = at(l + 2 * T), a3 = at(l + 3 * T);
            acc ^= a0 ^ a1 ^ a2 ^ a3;
        }
#pragma unroll 1
        for (; l < lines; l += T) acc ^= at(l);
    }
    return acc;
}
// the same over a set's ranges, in order
__device__ __forceinline__ unsigned pf_touch(const PfSet& s, size_t T, size_t i0) {
    unsigned acc = 0;
#pragma unroll 1
    for (int k = 0; k < s.n; ++k) acc ^= pf_touch_range(reinterpret_cast<const char*>(s.r[k].base), (size_t) s.r[k].bytes, s.line, s.full, T, i0);
    return acc;
}
__global__ void __launch_bounds__(256) k_mall_pf(PfSet s) {
    const unsigned acc = pf_touch(s, (size_t) gridDim.x * blockDim.x, blockIdx.x * (size_t) blockDim.x + threadIdx.x);
    // keeps the loads: a store through a pointer the compiler cannot see the end of (a store to a file-local global was dropped, and every load with
    // it). True for one lane in 2^32: a write of that one word to the sink
    if (acc == 0x9e3779b9u) *s.sink = acc;
}
// The persistent form (FNX_MALL_PF_MODE=run): launched on a side stream before a window's graph, outside the graph (a side-stream node in a captured
// graph cost ~90 us a layer on ROCm 7.2.4). words[0] counts the fork points run so far: the router's block 0 adds one a trunk layer (k_router_route's
// pf argument). After p = words[0] - base of this window's fork points the block stages layer p's set (the layer after the one whose router ran),
// p = n: layer 0's when wrap, and ends. Each block's thread 0 polls (s_sleep between loads), the rest wait at the block's barrier; a block that
// falls behind goes straight to the newest set. It ends by itself: at p = n; at p > n (the host's count is not this window's: status bit 1); when
// words[3] is set (the host's stop); or after `timeout` wall-clock ticks (status bit 0), so that a window that failed cannot leave it resident.
// The events and what each stages are the host's table: an event is any kernel of the window that adds one to words[0] (the router, an hc read), and
// set p is what the host wants staged after the window's p-th event (it can be empty).
// a.gate (bytes, 0: none): the block stages a set in pieces of so many bytes and its thread 0 looks at the count before each piece. When a later
// event has run it leaves the set (words[8]) and takes the newest one. With an empty set for the event before the expert kernels the block loads
// nothing while they stream (they need the whole bus), and what it had not staged by then stays cold.
__global__ void __launch_bounds__(256) k_mall_pf_run(PfRun a) {
    __shared__ int next;
    __shared__ int moved[2];
    const size_t T = (size_t) gridDim.x * blockDim.x, i0 = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    const long long t0 = threadIdx.x == 0 ? wall_clock64() : 0;
    if (threadIdx.x == 0) atomicAdd(a.words + 4, 1u);   // this block is resident (mall_pf_hc_probe waits for every block)
    int done = 0;
    unsigned acc = 0;
    // the side stream's time account, by block 0's thread 0 (words[12..14], units of 64 ticks: a report's 4096 windows of the longest time-out stay
    // under 2^32; t_rem keeps what is left under a unit): what the wait for the next event followed
    const bool tm = blockIdx.x == 0 && threadIdx.x == 0;
    long long t_mark = t0, t_rem[3] = {0, 0, 0};
    int after = 0;   // 1: a set staged to its end (the wait is spare time), 2: an empty set (the wait is the pause)
    const auto t_add = [&](int k, long long d) {
        t_rem[k] += d;
        atomicAdd(a.words + 12 + k, (unsigned) (t_rem[k] >> 6));
        t_rem[k] &= 63;
    };
#pragma unroll 1
    for (;;) {
        if (threadIdx.x == 0) {
            int p;
            unsigned polls = 0;
#pragma unroll 1
            for (;;) {
                // the count: a device-scope load (the routers' atomicAdd is device scope too; a volatile load would be system scope)
                p = (int) (__hip_atomic_load(a.words, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) - a.base);
                if (p > done) break;                       // a fork point ran (p > n: not this window's count)
                // the stop word, every 64th poll: device scope as well (mall_pf_poke writes it with a kernel of the card; a system-scope load goes
                // past the card's caches to memory, where a word a kernel wrote need not be yet)
                if ((++polls & 63u) == 0 && __hip_atomic_load(a.words + 3, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) { p = -1; break; }
                if (wall_clock64() - t0 > a.timeout) { p = -2; break; }
                __builtin_amdgcn_s_sleep(8);
            }
            next = p;
            if (tm) {
                const long long now = wall_clock64();
                if (after) t_add(after, now - t_mark);
                t_mark = now;
            }
        }
        __syncthreads();
        const int p = next;
        __syncthreads();
        if (threadIdx.x == 0) {   // what this block did, for the host's report: [5] + [7] + [8] is every set a block has to stage (n - 1 + wrap a window)
            const bool over = p < 0 || p > a.n;   // stopped, timed out or past its window: the rest of the window's sets are left out
            const int miss = over ? a.n - 1 - done + a.wrap : p - done - 1;
            if (miss > 0) atomicAdd(a.words + 7, (unsigned) miss);
            if (done == 0 && p >= a.n) atomicAdd(a.words + 6, 1u);   // its first sight of the count was the window's end or later
        }
        if (p < 0 || p > a.n) {
            if (threadIdx.x == 0 && p != -1) atomicOr(a.words + 1, p == -2 ? 1u : 2u);
            break;
        }
        const int l = p < a.n ? p : a.wrap ? 0 : -1;
        if (l >= 0) {
            const PfSet& s = a.sets[l];
            bool left = false;
            size_t staged = 0;   // this block's way into the set, for the report of a set it leaves (every block takes the same pieces)
            if (!a.gate) {
                acc ^= pf_touch(s, T, i0);
            } else {
                const size_t piece = (size_t) a.gate;
                unsigned c = 0;
#pragma unroll 1
                for (int k = 0; k < s.n && !left; ++k) {
                    const char* base = reinterpret_cast<const char*>(s.r[k].base);
                    const size_t bytes = (size_t) s.r[k].bytes;
#pragma unroll 1
                    for (size_t o = 0; o < bytes; ++c) {
                        // has a later event run? thread 0's look, for the block (two slots in turn: the next look's store cannot meet this
                        // one's loads, as thread 0 passes the next barrier only after every thread has)
                        if (threadIdx.x == 0) moved[c & 1u] = (int) (__hip_atomic_load(a.words, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) - a.base) != p;
                        __syncthreads();
                        if (moved[c & 1u]) {
                            left = true;
                            break;
                        }
                        size_t len = bytes - o;   // the last piece of a range takes the rest (under two pieces: no piece of a few bytes)
                        if (len >= 2 * piece) len = piece;
                        acc ^= pf_touch_range(base + o, len, s.line, s.full, T, i0);
                        o += len, staged += len;
                    }
                }
            }
            if (tm) {   // [12] the time in sets with ranges, [15] block 0's position in them when it left, in units of 64 KiB
                const long long now = wall_clock64();
                if (s.n) {
                    size_t all = staged;
                    if (!a.gate)
                        for (int k = 0; k < s.n; ++k) all += (size_t) s.r[k].bytes;
                    t_add(0, now - t_mark);
                    atomicAdd(a.words + 15, (unsigned) (all >> 16));
                }
                after = s.n == 0 ? 2 : left ? 0 : 1;
                t_mark = now;
            }
            if (threadIdx.x == 0) {
                atomicAdd(a.words + (left ? 8 : 5), 1u);
                if (!left && s.n == 0) atomicAdd(a.words + 9, 1u);
                if (left) {   // how far the set was: [10] its staged share in 1/256, [11] what was left cold in units of 256 KiB
                    size_t all = 0;
                    for (int k = 0; k < s.n; ++k) all += (size_t) s.r[k].bytes;
                    atomicAdd(a.words + 10, (unsigned) (all ? staged * 256 / all : 256));
                    atomicAdd(a.words + 11, (unsigned) ((all - staged) >> 18));
                }
            }
        }
        if (p == a.n) break;
        done = p;
    }
    if (acc == 0x9e3779b9u) a.words[2] = acc;   // keeps the loads (see k_mall_pf)
}
// one of the prefetcher's words for the host, at device scope: out: the word's value to *out; no out: val into the word
__global__ void k_mall_pf_word(unsigned* w, unsigned val, unsigned* out) {
    if (out)
        *out = __hip_atomic_load(w, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    else
        atomicExch(w, val);
}
}  // namespace
bool mall_pf_built() { return true; }
namespace {
void pf_check(const PfSet& set, bool need_sink) {
    if (set.n < 0 || set.n > MALL_PF_MAXR || set.line < 4 || (need_sink && !set.sink)) throw std::runtime_error("mall_pf: a bad set");
    for (int k = 0; k < set.n; ++k)
        if (!set.r[k].base || set.r[k].bytes < 16) throw std::runtime_error("mall_pf: a range under 16 bytes");
}
}  // namespace
void mall_pf(const PfSet& set, int wgs, void* st) {
    if (wgs < 1 || wgs > 1024) throw std::runtime_error("mall_pf: 1..1024 workgroups");   // (the engine's own limit is 64; kbench stages with up to a full grid)
    pf_check(set, true);
    k_mall_pf<<<(unsigned) wgs, 256, 0, (cudaStream_t) st>>>(set);
    ck_launch("mall_pf");
}
void mall_pf_table(PfSet* dev_table, const PfSet* sets, int n) {
    if (!dev_table || !sets || n < 1) throw std::runtime_error("mall_pf_table: no table");
    for (int i = 0; i < n; ++i) pf_check(sets[i], false);
    const cudaError_t e = cudaMemcpy(dev_table, sets, (size_t) n * sizeof(PfSet), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) throw std::runtime_error(std::string("mall_pf_table: ") + cudaGetErrorString(e));
}
void mall_pf_run(const PfRun& r, int wgs, void* st, bool query) {
    if (!r.sets || !r.words || r.n < 1 || wgs < 1 || wgs > 64 || r.timeout <= 0) throw std::runtime_error("mall_pf_run: a bad table, grid or time-out");
    if (r.gate && (r.gate < 65536 || r.gate % 4096)) throw std::runtime_error("mall_pf_run: the gate's piece is a multiple of 4096 bytes, 65536 or more");
    k_mall_pf_run<<<(unsigned) wgs, 256, 0, (cudaStream_t) st>>>(r);
    ck_launch("mall_pf_run");
    if (query) {   // (ROCm 7.2.4 clr: Command::enqueue writes the AQL packet and the doorbell in the launch call: nothing is left to flush)
        const cudaError_t q = cudaStreamQuery((cudaStream_t) st);   // "not ready" is the expected answer (it does not set the last error)
        if (q != cudaSuccess && q != cudaErrorNotReady) throw std::runtime_error(std::string("mall_pf_run: stream query: ") + cudaGetErrorString(q));
    }
}
void* mall_pf_stream(int prio, bool* own_queue) {
    hipStream_t s = nullptr;
    bool own = false;
    if (prio) {
        int least = 0, greatest = 0;
        if (hipDeviceGetStreamPriorityRange(&least, &greatest) == hipSuccess && least != greatest)
            own = hipStreamCreateWithPriority(&s, hipStreamNonBlocking, prio > 0 ? greatest : least) == hipSuccess && s != nullptr;
        if (!own) {
            s = nullptr;
            (void) hipGetLastError();   // the failed call's error only
        }
    }
    if (!s && hipStreamCreateWithFlags(&s, hipStreamNonBlocking) != hipSuccess) throw std::runtime_error("mall_pf_stream: no stream");
    if (own_queue) *own_queue = own;
    return s;
}
void mall_pf_poke(unsigned* words, int idx, unsigned val, void* st) {
    if (!words || idx < 0 || idx >= MALL_PF_WORDS) throw std::runtime_error("mall_pf_poke: a bad word");
    k_mall_pf_word<<<1, 1, 0, (cudaStream_t) st>>>(words + idx, val, nullptr);
    ck_launch("mall_pf_poke");
}
void mall_pf_peek(const unsigned* words, int idx, unsigned* dev_out, void* st) {
    if (!words || !dev_out || idx < 0 || idx >= MALL_PF_WORDS) throw std::runtime_error("mall_pf_peek: a bad word");
    k_mall_pf_word<<<1, 1, 0, (cudaStream_t) st>>>(const_cast<unsigned*>(words) + idx, 0u, dev_out);
    ck_launch("mall_pf_peek");
}
long long mall_pf_ticks(double ms) {
    int dev = 0, khz = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev) != hipSuccess || khz <= 0) {
        cudaGetLastError();
        khz = 100000;   // gfx11 and later: 100 MHz
    }
    return (long long) (ms * (double) khz);
}
#else
bool mall_pf_built() { return false; }
void mall_pf(const PfSet&, int, void*) { throw std::runtime_error("mall_pf: built without STRATA_SPLIT_FAST"); }
void mall_pf_run(const PfRun&, int, void*, bool) { throw std::runtime_error("mall_pf_run: built without STRATA_SPLIT_FAST"); }
void* mall_pf_stream(int, bool*) { throw std::runtime_error("mall_pf_stream: built without STRATA_SPLIT_FAST"); }
void mall_pf_poke(unsigned*, int, unsigned, void*) { throw std::runtime_error("mall_pf_poke: built without STRATA_SPLIT_FAST"); }
void mall_pf_peek(const unsigned*, int, unsigned*, void*) { throw std::runtime_error("mall_pf_peek: built without STRATA_SPLIT_FAST"); }
void mall_pf_table(PfSet*, const PfSet*, int) { throw std::runtime_error("mall_pf_table: built without STRATA_SPLIT_FAST"); }
long long mall_pf_ticks(double) { return 0; }
#endif
}  // namespace fnx

namespace fnx {
namespace {
// y[r] = x[r] / sqrt(mean(x[r]^2) + eps) * w, a block per row
__global__ void k_rms_norm_rows(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[32];
    const float* xr = x + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    const float r = rsqrtf(block_sum(s, sh) / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[(size_t) blockIdx.x * n + i] = xr[i] * r * w[i];
}
// R[t][c][i] = h[t][c][i] + e[t][i]
__global__ void k_add_streams(const float* h, const float* e, float* R) {
    const int t = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;
    R[(size_t) t * D + j] = h[(size_t) t * D + j] + e[(size_t) t * N + j % N];
}
}  // namespace

void rms_norm_rows(const float* x, const float* w, float* y, int rows, int n, float eps, void* st) {
    k_rms_norm_rows<<<rows, 1024, 0, (cudaStream_t) st>>>(x, w, y, n, eps);
    ck_launch("rms_norm_rows");
}
void add_streams(const float* h, const float* e, float* R, int T, void* st) {
    k_add_streams<<<dim3((D + 255) / 256, T), 256, 0, (cudaStream_t) st>>>(h, e, R);
    ck_launch("add_streams");
}
}  // namespace fnx
#endif
