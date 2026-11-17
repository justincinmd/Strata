// tools/emu/cuda_runtime.h: a host stand-in for the device runtime, for tools/emu/attn_mm_emu.cpp only. A kernel's block runs as one
// fiber per thread on one OS thread: __syncthreads() waits for every live thread of the block, and a wave-level exchange (shuffles, ballots, a WMMA) waits
// for the live lanes of the wave, so the kernels' control flow, shared-memory layout and lane arithmetic run as written. It checks logic, not speed, and not
// the card's floating-point contraction. The WMMA is the gfx12 16x16x16 f16 product by the layout attn_mm.cuh states.
#pragma once
#ifndef __HIPCC__
#define __HIPCC__ 1
#endif
#include <ucontext.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#define __host__
#define __device__ inline
#define __forceinline__
#define __global__ static
#define __launch_bounds__(...)
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))

struct uint4 { unsigned x, y, z, w; };
struct float4 { float x, y, z, w; };
struct float2 { float x, y; };
struct dim3 {
    unsigned x, y, z;
    dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {}
};
inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) { return uint4{x, y, z, w}; }
inline float4 make_float4(float x, float y, float z, float w) { return float4{x, y, z, w}; }
typedef _Float16 __half;
struct __half2 { __half x, y; };
inline float __half2float(__half h) { return (float) h; }
inline __half __float2half_rn(float f) { return (__half) f; }
inline __half __float2half(float f) { return (__half) f; }
inline __half2 make_half2(__half x, __half y) { return __half2{x, y}; }
inline float2 __half22float2(__half2 h) { return float2{(float) h.x, (float) h.y}; }
inline float __expf(float x) { return expf(x); }
inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
inline unsigned __float_as_uint(float f) { unsigned u; memcpy(&u, &f, 4); return u; }
inline float __uint_as_float(unsigned u) { float f; memcpy(&f, &u, 4); return f; }
inline int __clz(int x) { return x ? __builtin_clz((unsigned) x) : 32; }
inline int __ffs(unsigned x) { return __builtin_ffs((int) x); }
inline int __popc(unsigned x) { return __builtin_popcount(x); }
inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
inline unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
inline unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }
inline unsigned atomicAdd(unsigned* p, unsigned v) { const unsigned o = *p; *p = o + v; return o; }

namespace emu {
enum { RUN, BARRIER, WAVE, DONE };
struct Fiber {
    ucontext_t ctx;
    std::vector<char> stack;
    int state = RUN;
};
struct State {
    ucontext_t main;
    std::vector<Fiber> f;
    int cur = 0;
    std::function<void()> fn;
    unsigned char wbuf[64][32][64], wsnap[64][32][64];
    unsigned wpart[64];   // lanes that took part in the wave's last exchange
    long xchg = 0, barriers = 0;
};
inline State& S() { static State* s = new State; return *s; }
}  // namespace emu
inline dim3 threadIdx, blockIdx, blockDim, gridDim;
namespace emu {
inline void yield(int state) {
    State& s = S();
    Fiber& f = s.f[(size_t) s.cur];
    f.state = state;
    const int me = s.cur;
    swapcontext(&f.ctx, &s.main);
    threadIdx.x = (unsigned) me;
}
inline void entry() {
    S().fn();
    yield(DONE);
}
// one block: nt threads running fn (which reads threadIdx / blockIdx)
inline void run_block(int nt, const std::function<void()>& fn) {
    State& s = S();
    if (nt > 64 * 32) { fprintf(stderr, "emu: block of %d threads\n", nt); exit(3); }
    s.fn = fn;
    s.f.assign((size_t) nt, Fiber{});
    for (int i = 0; i < nt; ++i) {
        Fiber& f = s.f[(size_t) i];
        f.stack.resize(256 << 10);
        getcontext(&f.ctx);
        f.ctx.uc_stack.ss_sp = f.stack.data();
        f.ctx.uc_stack.ss_size = f.stack.size();
        f.ctx.uc_link = &s.main;
        makecontext(&f.ctx, entry, 0);
    }
    for (;;) {
        for (int i = 0; i < nt; ++i)
            if (s.f[(size_t) i].state == RUN) {
                s.cur = i;
                threadIdx.x = (unsigned) i;
                swapcontext(&s.main, &s.f[(size_t) i].ctx);
            }
        bool released = false;
        int live = 0, at_barrier = 0;
        for (int w = 0; w * 32 < nt; ++w) {
            int wl = 0, ww = 0;
            unsigned part = 0;
            for (int l = 0; l < 32 && w * 32 + l < nt; ++l) {
                const int st = s.f[(size_t) (w * 32 + l)].state;
                wl += st != DONE;
                at_barrier += st == BARRIER;
                if (st == WAVE) ++ww, part |= 1u << l;
            }
            live += wl;
            if (wl && ww == wl) {
                memcpy(s.wsnap[w], s.wbuf[w], sizeof s.wbuf[w]);
                s.wpart[w] = part;
                for (int l = 0; l < 32 && w * 32 + l < nt; ++l)
                    if (s.f[(size_t) (w * 32 + l)].state == WAVE) s.f[(size_t) (w * 32 + l)].state = RUN;
                released = true;
                ++s.xchg;
            } else if (ww) {
                bool barrier_mix = false;
                for (int l = 0; l < 32 && w * 32 + l < nt; ++l) barrier_mix |= s.f[(size_t) (w * 32 + l)].state == BARRIER;
                if (barrier_mix) { fprintf(stderr, "emu: wave %d: some lanes at a wave exchange, others at the block barrier\n", w); exit(3); }
            }
        }
        if (released) continue;
        if (!live) break;
        if (at_barrier != live) { fprintf(stderr, "emu: stuck (%d live, %d at the barrier)\n", live, at_barrier); exit(3); }
        for (int i = 0; i < nt; ++i)
            if (s.f[(size_t) i].state == BARRIER) s.f[(size_t) i].state = RUN;
        ++s.barriers;
    }
}
// publish n bytes for this lane, wait for the wave; the wave's values come back as [32][64] bytes
inline const unsigned char (*xchg(const void* v, size_t n))[64] {
    State& s = S();
    const int w = s.cur >> 5, l = s.cur & 31;
    memcpy(s.wbuf[w][l], v, n);
    yield(WAVE);
    return s.wsnap[w];
}
inline unsigned part() { return S().wpart[S().cur >> 5]; }
template <typename T> inline T shfl(T v, int src) {
    const int l = S().cur & 31;
    const unsigned char (*b)[64] = xchg(&v, sizeof(T));
    if (src < 0 || src > 31 || !(part() >> src & 1)) return v;
    (void) l;
    T r;
    memcpy(&r, b[src], sizeof(T));
    return r;
}
template <typename F> inline void launch(dim3 grid, int nt, F kernel) {
    gridDim = grid;
    blockDim = dim3((unsigned) nt);
    for (unsigned z = 0; z < grid.z; ++z)
        for (unsigned y = 0; y < grid.y; ++y)
            for (unsigned x = 0; x < grid.x; ++x) {
                blockIdx = dim3(x, y, z);
                run_block(nt, [&] { kernel(); });
            }
}
}  // namespace emu
inline void __syncthreads() {
    const dim3 b = blockIdx;
    emu::yield(emu::BARRIER);
    blockIdx = b;
}
template <typename T> inline T __shfl_xor_sync(unsigned, T v, int o) { return emu::shfl(v, (emu::S().cur & 31) ^ o); }
template <typename T> inline T __shfl_up_sync(unsigned, T v, int o) { return emu::shfl(v, (emu::S().cur & 31) - o); }
inline unsigned __ballot_sync(unsigned, bool p) {
    const unsigned char v = p;
    const unsigned char (*b)[64] = emu::xchg(&v, 1);
    unsigned r = 0;
    for (int l = 0; l < 32; ++l)
        if (emu::part() >> l & 1) r |= (unsigned) (b[l][0] != 0) << l;
    return r;
}
inline unsigned __match_any_sync(unsigned long long, unsigned v) {
    const unsigned char (*b)[64] = emu::xchg(&v, 4);
    unsigned r = 0;
    for (int l = 0; l < 32; ++l) {
        unsigned o;
        memcpy(&o, b[l], 4);
        if ((emu::part() >> l & 1) && o == v) r |= 1u << l;
    }
    return r;
}
// the gfx12 f16 WMMA: A lane l = row l & 15, k 8 (l >> 4) + 0..7; B lane l = column l & 15, the same k; c element r of lane l = row r + 8 (l >> 4), column l & 15
typedef float emu_f8 __attribute__((ext_vector_type(8)));
inline emu_f8 emu_wmma(uint4 a, uint4 b, emu_f8 c) {
    struct { uint4 a, b; } ab{a, b};
    const int l = emu::S().cur & 31, col = l & 15, hi = l >> 4;
    const unsigned char (*w)[64] = emu::xchg(&ab, 32);
    if (emu::part() != 0xffffffffu) { fprintf(stderr, "emu: a WMMA with idle lanes\n"); exit(3); }
    for (int r = 0; r < 8; ++r) {
        const int row = r + 8 * hi;
        float sum = c[r];
        for (int k = 0; k < 16; ++k) {
            _Float16 x, y;
            memcpy(&x, w[row + 16 * (k / 8)] + 2 * (k % 8), 2);        // A[row][k]
            memcpy(&y, w[col + 16 * (k / 8)] + 16 + 2 * (k % 8), 2);   // B[k][col]
            sum += (float) x * (float) y;
        }
        c[r] = sum;
    }
    return c;
}
#define FNX_EMU_WMMA emu_wmma
