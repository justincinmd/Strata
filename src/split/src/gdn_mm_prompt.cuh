// src/split/src/gdn_mm_prompt.cuh - the prompt GDN's sub-chunk recurrence on the matrix instruction (FNX_PF_GDN_MM=1, gfx12)
//
// k_gdn_ch_rec's algebra (gdn_prompt.cuh: k_gdn_ch_prep's T, P, G, decays and b, sub-chunks of GCH = 32 rows, a block a (value head,
// GCB = 16 state columns)), with its five products a sub-chunk on v_wmma_f32_16x16x16_f16: W = K S and Y = Q S, U = T R, the
// output P U, and the state update K^T (dec U). Every operand is split into f16 hi + lo (lo = the rest x 2^11), three products
// (hi hi + hi lo + lo hi) in two accumulators: ~22-bit operands, fp32 class.
// The B operands (S, R, U, dec U) are scaled per column by a power of two so that the largest is under 2^15 (no f16 overflow);
// T by one power of two a sub-chunk; K, Q (unit rows) and P (|P| <= 1) as they are. The sums run in another order than
// k_gdn_ch_rec's: results differ in the last bits (texts change).
//
// Layout (gfx12 wave32 WMMA): A lane l = row l & 15, k 8 (l >> 4) + 0..7; B lane l = column l & 15, k 8 (l >> 4) + 0..7; the
// accumulator lane l = column l & 15, rows 8 (l >> 4) + 0..7. So an accumulator tile is the B operand of the next product as it is.
// Wave w holds the state rows 32 w .. 32 w + 31 (two accumulator tiles) of the block's columns: W and Y are summed over the four
// waves' row ranges through shared memory (one barrier), then every wave makes R, U (the same values in every wave) and its rows'
// update; waves 0 and 1 make the output's token tiles 0 and 1. The next sub-chunk's K, Q are fetched while this one's U and update
// run, its v, T, P after them; its scalars (G, decay to end, b, G_last) go through shared memory.
#pragma once
#if FNX_ATT_MM && FNX_HIP
#include <type_traits>

constexpr int GMM_THREADS = 128;    // 4 waves
constexpr int GMM_WP = GCH + 4;     // a partial's row: [column][token], 16-B aligned, rows on distinct banks
constexpr int GMM_KP = 32 + 1;      // a wave's K slice: [token][its 32 state rows]
constexpr int GMM_SC = 3 * GCH + 1; // G, decay to end, b, G_last
static_assert(GCB == 16 && GCH == 32 && HS == 128, "k_gdn_ch_rec_mm: 16 columns, 32-row sub-chunks, 4 waves of 32 state rows");

// the power of two that brings m (>= 0) under 2^15; 0 for 0 or a non-finite m (a NaN goes through as it is)
__device__ __forceinline__ int gmm_exp(float m) {
    if (!(m > 0.f) || !(m < INFINITY)) return 0;
    return min(max(ilogbf(m) - 14, -120), 120);
}
// x * 2^-e as f16 hi + lo (lo = (x - hi) x 2^11)
__device__ __forceinline__ void gmm_split(const float (&x)[8], float sc, uint4& hi, uint4& lo) {
    am_h8 h, l;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float v = x[i] * sc;
        h[i] = (_Float16) v;
        l[i] = (_Float16) ((v - (float) h[i]) * 2048.f);
    }
    hi = __builtin_bit_cast(uint4, h), lo = __builtin_bit_cast(uint4, l);
}
__device__ __forceinline__ float gmm_wmax(float m) {   // the largest over the wave
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
    return m;
}
// a block barrier for shared memory only. __syncthreads() is a workgroup fence on gfx12 and so waits for every global load in
// flight first (the next sub-chunk's prefetch, measured 1.8 against 0.85 ms at C 4096); this waits for this wave's LDS operations,
// and the clobbers keep the compiler from moving memory operations across it
__device__ __forceinline__ void gmm_bar() {
    asm volatile("s_wait_dscnt 0x0" ::: "memory");
    __builtin_amdgcn_s_barrier();
    asm volatile("" ::: "memory");
}
// hi x hi into h, the cross terms into x
__device__ __forceinline__ void gmm_prod(uint4 ah, uint4 al, uint4 bh, uint4 bl, am_f8& h, am_f8& x) {
    h = am_wmma(ah, bh, h);
    x = am_wmma(ah, bl, x);
    x = am_wmma(al, bh, x);
}

// SKIP (timing only, tools/gdn_mm_test.cu): 1 = no next v / T / P loads, 2 = no next K / Q loads, 4 = no output stores
template <int SKIP = 0>
__global__ void __launch_bounds__(GMM_THREADS) k_gdn_ch_rec_mm(const float* __restrict__ xc, const float* __restrict__ prep,
                                                              float* __restrict__ S, float* __restrict__ y32, int nk, int nv,
                                                              const int* __restrict__ vmap, int C) {
    const int lv = blockIdx.x / GCH_REC_BLOCKS_PER_HEAD, c0 = (blockIdx.x % GCH_REC_BLOCKS_PER_HEAD) * GCB;
    const int kh = vmap[lv] & 255, tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, h8 = 8 * (lane >> 4);
    const int ch = (2 * nk + nv) * HS;
    __shared__ __attribute__((aligned(16))) float wp[4][2][GCB][GMM_WP];   // the waves' partial W, Y
    __shared__ float kt[4][GCH][GMM_KP];                                    // each wave's K slice, for K^T
    __shared__ __attribute__((aligned(16))) float prs[GMM_PREP_FLOATS];   // the sub-chunk's T, P tiles and scalars (k_gdn_ch_prep<true>)
    float* sst = S + (size_t) lv * HS * HS;
    am_f8 st[2];   // the state: rows 32 w + 16 m + h8 + i, column c0 + l16
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int i = 0; i < 8; ++i) st[m][i] = sst[(size_t) (32 * w + 16 * m + h8 + i) * HS + c0 + l16];
    const float scale = rsqrtf((float) HS);
    const int nsub = (C + GCH - 1) / GCH;
    // the prefetch registers: K, Q as A operands (token 16 mt + l16, rows 32 w + 16 j + h8 + 0..7); v as a B operand (column
    // c0 + l16, tokens 16 js + h8 + 0..7); the prep record (T, P tiles and scalars) in 16-B pieces tid + 128 q, staged in shared
    // memory at the top of the sub-chunk
    constexpr int GMM_PR4 = (GMM_PREP_FLOATS / 4 + GMM_THREADS - 1) / GMM_THREADS;
    float kf[2][2][8], qf[2][2][8], vf[2][8];
    float4 pr4[GMM_PR4];
    auto fetch_kq = [&](int sc) {
        const int t0 = sc * GCH, n = min(GCH, C - t0);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const int t = 16 * mt + l16;
            const float* row = xc + (size_t) (t0 + t) * ch;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int r = 32 * w + 16 * j + h8;
                float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = a, c = a, d = a;
                if (t < n) {
                    a = *reinterpret_cast<const float4*>(row + nk * HS + kh * HS + r), b = *reinterpret_cast<const float4*>(row + nk * HS + kh * HS + r + 4);
                    c = *reinterpret_cast<const float4*>(row + kh * HS + r), d = *reinterpret_cast<const float4*>(row + kh * HS + r + 4);
                }
                kf[mt][j][0] = a.x, kf[mt][j][1] = a.y, kf[mt][j][2] = a.z, kf[mt][j][3] = a.w;
                kf[mt][j][4] = b.x, kf[mt][j][5] = b.y, kf[mt][j][6] = b.z, kf[mt][j][7] = b.w;
                qf[mt][j][0] = c.x, qf[mt][j][1] = c.y, qf[mt][j][2] = c.z, qf[mt][j][3] = c.w;
                qf[mt][j][4] = d.x, qf[mt][j][5] = d.y, qf[mt][j][6] = d.z, qf[mt][j][7] = d.w;
            }
        }
    };
    auto fetch_rest = [&](int sc) {
        const int t0 = sc * GCH, n = min(GCH, C - t0);
#pragma unroll
        for (int js = 0; js < 2; ++js)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int t = 16 * js + h8 + i;
                vf[js][i] = t < n ? xc[(size_t) (t0 + t) * ch + 2 * nk * HS + lv * HS + c0 + l16] : 0.f;
            }
        const float4* pg = reinterpret_cast<const float4*>(prep + ((size_t) sc * nv + lv) * GMM_PREP_FLOATS);   // k_gdn_ch_prep<true>'s layout
#pragma unroll
        for (int q = 0; q < GMM_PR4; ++q) {
            const int e = tid + q * GMM_THREADS;
            pr4[q] = e < GMM_PREP_FLOATS / 4 ? pg[e] : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    };
    fetch_kq(0);
    fetch_rest(0);
    for (int sc = 0; sc < nsub; ++sc) {
        const int t0 = sc * GCH, n = min(GCH, C - t0);
        gmm_bar();   // the last sub-chunk's reads of wp and scl are done
#pragma unroll
        for (int q = 0; q < GMM_PR4; ++q) {
            const int e = tid + q * GMM_THREADS;
            if (e < GMM_PREP_FLOATS / 4) reinterpret_cast<float4*>(prs)[e] = pr4[q];
        }
        // W, Y over this wave's 32 state rows: A = K, Q (unit rows), B = S scaled per column
        {
            float m = 0.f;
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int i = 0; i < 8; ++i) m = fmaxf(m, fabsf(st[j][i]));
            m = fmaxf(m, __shfl_xor(m, 16, 32));
            const int e = gmm_exp(m);
            const float sc_in = ldexpf(1.f, -e), sc_out = ldexpf(1.f, e);
            uint4 sh[2], sl[2];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                float x[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) x[i] = st[j][i];
                gmm_split(x, sc_in, sh[j], sl[j]);
            }
#pragma unroll 1
            for (int mt = 0; mt < 2; ++mt) {
                am_f8 wh = {}, wx = {}, yh = {}, yx = {};
                float k_[2][8], q_[2][8];   // kf[mt], qf[mt] by selects (a run-time index would put the arrays in scratch)
#pragma unroll
                for (int j = 0; j < 2; ++j)
#pragma unroll
                    for (int i = 0; i < 8; ++i) k_[j][i] = mt ? kf[1][j][i] : kf[0][j][i], q_[j][i] = mt ? qf[1][j][i] : qf[0][j][i];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    uint4 ah, al;
                    gmm_split(k_[j], 1.f, ah, al);
                    gmm_prod(ah, al, sh[j], sl[j], wh, wx);
                    gmm_split(q_[j], 1.f, ah, al);
                    gmm_prod(ah, al, sh[j], sl[j], yh, yx);
#pragma unroll
                    for (int i = 0; i < 8; ++i) kt[w][16 * mt + l16][16 * j + h8 + i] = k_[j][i];
                }
                float* dw = &wp[w][0][l16][16 * mt + h8];
                float* dy = &wp[w][1][l16][16 * mt + h8];
#pragma unroll
                for (int i = 0; i < 8; i += 4) {
                    *reinterpret_cast<float4*>(dw + i) = make_float4((wh[i] + wx[i] * (1.f / 2048.f)) * sc_out, (wh[i + 1] + wx[i + 1] * (1.f / 2048.f)) * sc_out,
                                                                     (wh[i + 2] + wx[i + 2] * (1.f / 2048.f)) * sc_out, (wh[i + 3] + wx[i + 3] * (1.f / 2048.f)) * sc_out);
                    *reinterpret_cast<float4*>(dy + i) = make_float4((yh[i] + yx[i] * (1.f / 2048.f)) * sc_out, (yh[i + 1] + yx[i + 1] * (1.f / 2048.f)) * sc_out,
                                                                     (yh[i + 2] + yx[i + 2] * (1.f / 2048.f)) * sc_out, (yh[i + 3] + yx[i + 3] * (1.f / 2048.f)) * sc_out);
                }
            }
        }
        if (!(SKIP & 2) && sc + 1 < nsub) fetch_kq(sc + 1);   // in flight while this sub-chunk's U and update run
        gmm_bar();   // the partials, K slices and scalars are written
        const float* scl = prs + 2 * 3 * 256;
        const float* G = scl;
        const float* dec = scl + GCH;
        const float* B = scl + 2 * GCH;
        const float glast = scl[3 * GCH];
        // R = b (v - G W) as a B operand (column l16, tokens 16 js + h8 + i), W the four waves' partials in order; Y likewise
        float rr[2][8], yv[2][8];
#pragma unroll
        for (int js = 0; js < 2; ++js) {
            float wsum[8], ysum[8];
#pragma unroll
            for (int i = 0; i < 8; i += 4) {
                float4 a = *reinterpret_cast<const float4*>(&wp[0][0][l16][16 * js + h8 + i]);
                float4 b = *reinterpret_cast<const float4*>(&wp[0][1][l16][16 * js + h8 + i]);
#pragma unroll
                for (int q = 1; q < 4; ++q) {
                    const float4 c = *reinterpret_cast<const float4*>(&wp[q][0][l16][16 * js + h8 + i]);
                    const float4 d = *reinterpret_cast<const float4*>(&wp[q][1][l16][16 * js + h8 + i]);
                    a.x += c.x, a.y += c.y, a.z += c.z, a.w += c.w;
                    b.x += d.x, b.y += d.y, b.z += d.z, b.w += d.w;
                }
                wsum[i] = a.x, wsum[i + 1] = a.y, wsum[i + 2] = a.z, wsum[i + 3] = a.w;
                ysum[i] = b.x, ysum[i + 1] = b.y, ysum[i + 2] = b.z, ysum[i + 3] = b.w;
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int t = 16 * js + h8 + i;
                rr[js][i] = B[t] * (vf[js][i] - G[t] * wsum[i]);
                yv[js][i] = ysum[i];
            }
        }
        // U = T R: T scaled by one power of two (its largest over the wave), R per column
        am_f8 u[2];
        {
            float tf[3][8], mt_ = 0.f, mr = 0.f;
#pragma unroll
            for (int x = 0; x < 3; ++x)
#pragma unroll
                for (int i = 0; i < 8; i += 4) {
                    const float4 a = *reinterpret_cast<const float4*>(prs + (x * 32 + lane) * 8 + i);
                    tf[x][i] = a.x, tf[x][i + 1] = a.y, tf[x][i + 2] = a.z, tf[x][i + 3] = a.w;
                }
#pragma unroll
            for (int x = 0; x < 3; ++x)
#pragma unroll
                for (int i = 0; i < 8; ++i) mt_ = fmaxf(mt_, fabsf(tf[x][i]));
#pragma unroll
            for (int js = 0; js < 2; ++js)
#pragma unroll
                for (int i = 0; i < 8; ++i) mr = fmaxf(mr, fabsf(rr[js][i]));
            const int et = gmm_exp(gmm_wmax(mt_)), er = gmm_exp(fmaxf(mr, __shfl_xor(mr, 16, 32)));
            uint4 rh[2], rl[2], th[3], tl[3];
#pragma unroll
            for (int js = 0; js < 2; ++js) gmm_split(rr[js], ldexpf(1.f, -er), rh[js], rl[js]);
#pragma unroll
            for (int x = 0; x < 3; ++x) gmm_split(tf[x], ldexpf(1.f, -et), th[x], tl[x]);
            const float so = ldexpf(1.f, et + er);
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                am_f8 h = {}, x = {};
                gmm_prod(th[mt], tl[mt], rh[0], rl[0], h, x);       // tile (mt, 0): x = 0 or 1
                if (mt == 1) gmm_prod(th[2], tl[2], rh[1], rl[1], h, x);   // tile (1, 1); (0, 1) is zero
#pragma unroll
                for (int i = 0; i < 8; ++i) u[mt][i] = (h[i] + x[i] * (1.f / 2048.f)) * so;
            }
        }
        // the output, waves 0 and 1 a token tile each: o = scale (G Y + P U); U per column as B
        auto out_tile = [&](auto MT) {
            constexpr int mt = decltype(MT)::value;
            float m = 0.f;
#pragma unroll
            for (int js = 0; js < 2; ++js)
#pragma unroll
                for (int i = 0; i < 8; ++i) m = fmaxf(m, fabsf(u[js][i]));
            const int eu = gmm_exp(fmaxf(m, __shfl_xor(m, 16, 32)));
            uint4 uh[2], ul[2], ph[3], pl[3];
#pragma unroll
            for (int js = 0; js < 2; ++js) {
                float x[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) x[i] = u[js][i];
                gmm_split(x, ldexpf(1.f, -eu), uh[js], ul[js]);
            }
            float pf[3][8];
#pragma unroll
            for (int x = 0; x < 3; ++x)
#pragma unroll
                for (int i = 0; i < 8; i += 4) {
                    const float4 a = *reinterpret_cast<const float4*>(prs + 3 * 256 + (x * 32 + lane) * 8 + i);
                    pf[x][i] = a.x, pf[x][i + 1] = a.y, pf[x][i + 2] = a.z, pf[x][i + 3] = a.w;
                }
            am_f8 h = {}, x = {};
            if constexpr (mt == 0) {
                gmm_split(pf[0], 1.f, ph[0], pl[0]);
                gmm_prod(ph[0], pl[0], uh[0], ul[0], h, x);
            } else {
                gmm_split(pf[1], 1.f, ph[1], pl[1]);
                gmm_split(pf[2], 1.f, ph[2], pl[2]);
                gmm_prod(ph[1], pl[1], uh[0], ul[0], h, x);
                gmm_prod(ph[2], pl[2], uh[1], ul[1], h, x);
            }
            const float so = ldexpf(1.f, eu);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int t = 16 * mt + h8 + i;
                const float o = G[t] * yv[mt][i] + (h[i] + x[i] * (1.f / 2048.f)) * so;
                if (!(SKIP & 4) && t < n) y32[(size_t) (t0 + t) * nv * HS + lv * HS + c0 + l16] = o * scale;
            }
        };
        if (w == 0) out_tile(std::integral_constant<int, 0>{});
        else if (w == 1) out_tile(std::integral_constant<int, 1>{});
        // S = G_last S + K^T (dec U) for this wave's rows: A = K^T from the wave's slice, B = dec U per column
        {
            float du[2][8], m = 0.f;
#pragma unroll
            for (int js = 0; js < 2; ++js)
#pragma unroll
                for (int i = 0; i < 8; ++i) du[js][i] = dec[16 * js + h8 + i] * u[js][i], m = fmaxf(m, fabsf(du[js][i]));
            const int ed = gmm_exp(fmaxf(m, __shfl_xor(m, 16, 32)));
            uint4 dh[2], dl[2];
#pragma unroll
            for (int js = 0; js < 2; ++js) gmm_split(du[js], ldexpf(1.f, -ed), dh[js], dl[js]);
            const float so = ldexpf(1.f, ed);
#pragma unroll
            for (int mr = 0; mr < 2; ++mr) {
                am_f8 h = {}, x = {};
#pragma unroll
                for (int js = 0; js < 2; ++js) {
                    float a[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) a[i] = kt[w][16 * js + h8 + i][16 * mr + l16];
                    uint4 ah, al;
                    gmm_split(a, 1.f, ah, al);
                    gmm_prod(ah, al, dh[js], dl[js], h, x);
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) st[mr][i] = fmaf(glast, st[mr][i], (h[i] + x[i] * (1.f / 2048.f)) * so);
            }
        }
        if (!(SKIP & 1) && sc + 1 < nsub) fetch_rest(sc + 1);
    }
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int i = 0; i < 8; ++i) sst[(size_t) (32 * w + 16 * m + h8 + i) * HS + c0 + l16] = st[m][i];
}
#endif   // FNX_ATT_MM && FNX_HIP
