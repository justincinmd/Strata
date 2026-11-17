// attn_mm.cuh (FNX_ATT_MM = build option STRATA_SPLIT_APPROX; included by kernels.cu after k_attn_combine).
// The decode window's attention part and block selection for deep context, by heads (FNX_ATT_HEADS) on gfx12:
//
//  k_attn_mm        k_attn_part's work as matrix products. A block is (split, row) with 8 waves; it walks its tiles of 32 attended positions. A tile's int8 K
//                   and V rows come in 16-byte loads (lane = position; the next tile's are fetched into registers while this one computes), go to shared
//                   memory as f16 codes (K by rows, V transposed), and then
//                     scores   q . k = sum over the 8 scale groups of scale[pos][g] * (q_g . codes_g): the inner sums are f16 WMMA products with fp32
//                              accumulation, exact in the codes; q is fp32 split into f16 high and low parts (A rows 0..5 and 8..13 of one product);
//                     softmax  k_attn_part's online softmax per head (a wave per head, lane = position);
//                     values   acc[h][d] += sum_pos (p[h][pos] vscale[pos][g(d)]) * vcode[pos][d]: P scaled by powers of two into f16's range and split
//                              into high and low parts, V codes exact; the tile's product is scaled back in fp32.
//                   It writes k_attn_part's partials (acc per split and head, {m, z}), so the merge of the splits and the gate stay a second launch
//                   (k_attn_combine_n). The math is k_attn_part's; the order of the sums and the split bounds differ. Where it is not fp32 all the way:
//                   q's two f16 parts hold an element to 2^-22 of itself and to 2^-39 of the head's largest element (a head's q is scaled by a power
//                   of two so that its largest element is in [2^14, 2^15): no clipping, no floor for a small q; an element under 2^-39 of the head's
//                   largest is rounded away, and so is a head whose largest element is under 2^-100); a weight times V scale is held to 2^-22 of itself
//                   and to 2^-38 of the tile's largest weight times the tile's largest V scale of the group (a term at or under that is rounded away:
//                   k_attn_part keeps it, which shows only when everything larger cancels or has a zero scale); a tile whose largest weight is under
//                   2^-88 of the split's adds nothing (fp32's exp gives up at 2^-126). q or scores that are not finite are not supported by either
//                   kernel.
//  k_idx_score_mm   k_idx_score's scores, bit for bit (the same fma chain per row and head), with the query read at uniform addresses (scalar loads, no
//                   shared-memory read per fma) and, for one sequence's window, a key read once for up to 8 rows.
//  k_idx_select_mm  k_idx_select's list, bit for bit. A threshold tau from a strided sample of the scores (its rank chosen so that 512 or more scores
//                   reach tau), the scores >= tau collected in index order (at most QS_CAP), an exact radix select of the 512th among them, and an
//                   order-preserving compaction (ties: the lower block). When the sample fails (fewer than 512 or more than QS_CAP scores reach tau) the
//                   radix select runs on every score instead: the same list either way.
#if FNX_ATT_MM
#if !FNX_HIP && !defined(FNX_EMU_WMMA)
#error "FNX_ATT_MM is a gfx12 (HIP) path"
#endif
static_assert(ATT_HEADS && KVH_LOC == 1 && KV_W == HD && HD == 256 && KV8_GROUP == 32 && NQH_LOC <= 6,
              "k_attn_mm: by heads, one KV head a card, 8 scale groups of 32, at most 6 query heads (softmax waves 0..5, A rows h and 8 + h)");
constexpr int AM_TILE = 32, AM_KLD = HD + 8, AM_VLD = AM_TILE + 8, AM_G = HD / KV8_GROUP, AM_ROWBUF = 512, AM_PROWS = KVP_BLOCKS * 4;
static_assert(AM_PROWS < 256 && AM_ROWBUF % 256 == 0 && AM_ROWBUF % AM_TILE == 0, "a row slot packs the row in its page into 8 bits");
typedef _Float16 am_h8 __attribute__((ext_vector_type(8)));
typedef float am_f8 __attribute__((ext_vector_type(8)));
// c += a b, 16 x 16 x 16, f16 products with fp32 accumulation. A lane l: row l & 15, k 8 (l >> 4) + 0..7; B lane l: column l & 15, the same k; c's element
// r of lane l: row r + 8 (l >> 4), column l & 15
__device__ __forceinline__ am_f8 am_wmma(uint4 a, uint4 b, am_f8 c) {
#if defined(FNX_EMU_WMMA)   // tools/emu: the host emulation of this product
    return FNX_EMU_WMMA(a, b, c);
#elif defined(__GFX12__) || defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(__builtin_bit_cast(am_h8, a), __builtin_bit_cast(am_h8, b), c);
#else
    (void) a; (void) b;
    __builtin_trap();   // not a gfx12 device pass: FNX_ATT_MM is built for gfx12 cards only
    return c;
#endif
}
// 16 bytes at p (no alignment asked of p: a row of scores or a page need not sit at a multiple of 16)
__device__ __forceinline__ uint4 am_ld16(const void* p) {
    uint4 v;
    __builtin_memcpy(&v, p, 16);
    return v;
}
__device__ __forceinline__ float4 am_ld16f(const void* p) {
    float4 v;
    __builtin_memcpy(&v, p, 16);
    return v;
}
__device__ __forceinline__ unsigned am_hbits(_Float16 h) { return (unsigned) __builtin_bit_cast(unsigned short, h); }
// the int8 codes in bytes sh / 8 and sh / 8 + 1 of w as two f16 (exact)
__device__ __forceinline__ unsigned am_c2(unsigned w, int sh) {
#if defined(FNX_EMU_WMMA)
    const _Float16 a = (_Float16) (short) (int8_t) (w >> sh), b = (_Float16) (short) (int8_t) (w >> (sh + 8));
    return am_hbits(a) | (am_hbits(b) << 16);
#else   // a bit-field extract and an int -> f32 per code, one packed f32 -> f16 (round to zero: exact here) per pair
    return __builtin_bit_cast(unsigned, __builtin_amdgcn_cvt_pkrtz((float) (int) __builtin_amdgcn_sbfe((int) w, sh, 8),
                                                                   (float) (int) __builtin_amdgcn_sbfe((int) w, sh + 8, 8)));
#endif
}
// 16 int8 codes as 16 f16: lo the first 8, hi the last 8
__device__ __forceinline__ void am_codes(uint4 c, uint4& lo, uint4& hi) {
    lo = make_uint4(am_c2(c.x, 0), am_c2(c.x, 16), am_c2(c.y, 0), am_c2(c.y, 16));
    hi = make_uint4(am_c2(c.z, 0), am_c2(c.z, 16), am_c2(c.w, 0), am_c2(c.w, 16));
}
// x's f16 high part, or (low) what the high part leaves: high + low = x to ~22 bits (|x| < 2^15: the caller scales)
__device__ __forceinline__ unsigned am_part(float x, bool low) {
    const _Float16 a = (_Float16) x;
    return low ? am_hbits((_Float16) (x - (float) a)) : am_hbits(a);
}
__device__ __forceinline__ float am_h2f(unsigned bits) { return (float) __builtin_bit_cast(_Float16, (unsigned short) bits); }
// 2^e as a float (e within the normal range)
__device__ __forceinline__ float am_pow2(int e) { return __uint_as_float((unsigned) (127 + e) << 23); }

// grid (S splits, T rows), 256 threads. Split s of row t takes tiles [s ntile / S, (s + 1) ntile / S) of the row's attended positions (the selection's
// list, or dense every position up to the row's). Outputs as k_attn_part's with nq = NQH_LOC: acc_out[((t NQH_LOC + h) ATT_SPLITS + s) HD + d] and
// mz_out[(t NQH_LOC + h) ATT_SPLITS + s] for s < S (S <= ATT_SPLITS)
__global__ void __launch_bounds__(256) k_attn_mm(const float* __restrict__ qg, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc, KvPages g,
                                                 float* __restrict__ acc_out, AttPart* __restrict__ mz_out, const int* pos0p,
                                                 const int* __restrict__ cells, const int* __restrict__ n_cells, LaneRows lr, int S) {
    constexpr int NH = NQH_LOC;
    __shared__ __align__(16) unsigned short sk[AM_TILE][AM_KLD];      // the tile's K codes as f16, by position
    __shared__ __align__(16) unsigned short svt[HD][AM_VLD];          // its V codes as f16, by dimension (transposed)
    __shared__ __align__(16) unsigned short sp[AM_G][16][AM_TILE];    // P's parts per V scale group: rows h (high) and 8 + h (low)
    __shared__ float ss[4][2][NH][AM_TILE];                           // score partials: [dim quarter][q part][head][position]
    __shared__ __align__(16) float sks[AM_TILE][AM_G];                // K scales
    __shared__ __align__(16) float svs[AM_TILE][AM_G];                // V scales, times 2^kg (group g's largest in [2^7, 2^8))
    __shared__ unsigned srow[AM_ROWBUF];                              // row slots: physical page << 8 | row in the page
    __shared__ float salpha[8], sfh[8], sfg[AM_G];                    // per head: the accumulator's rescale, 2^-kh; per group: 2^-kg
    __shared__ __align__(16) float sqm[8][8];                         // [head][wave]: the largest |q| of the wave's dimensions
    const int s = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, lane = tid & 31, w = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    g = row_pages(g, lr, t);
    const int* list = cells && n_cells[t] >= 0 ? cells + (size_t) t * QSA_MAXSEL : nullptr;
    const int ctx = list ? n_cells[t] : row_pos(lr, pos0p, t) + 1;
    const int ntile = (ctx + AM_TILE - 1) / AM_TILE;
    const int tile0 = (int) ((long long) s * ntile / S), tile1 = (int) ((long long) (s + 1) * ntile / S);
    if (tile0 >= tile1) {   // no tile for this split: an empty partial (the combine skips m = -inf)
        if (lane == 0 && w < NH) mz_out[((size_t) t * NH + w) * ATT_SPLITS + s] = AttPart{-INFINITY, 0.f};
        return;
    }
    const size_t pg2 = g.page * 2;   // bytes a page
    const int jbase = tile0 * AM_TILE, jn = min(ctx, tile1 * AM_TILE) - jbase;   // the split's positions: list indices [jbase, jbase + jn)
    // the row slots of the split's positions [e0, e0 + AM_ROWBUF)
    auto fill = [&](int e0) {
        for (int i = tid; i < AM_ROWBUF; i += 256) {
            const int jl = e0 + i;
            if (jl >= jn) continue;
            const int pos = list ? list[jbase + jl] : jbase + jl;
            srow[i] = ((unsigned) g.pt[pos / AM_PROWS] << 8) | (unsigned) (pos % AM_PROWS);
        }
    };
    fill(0);
    for (int i = tid; i < AM_G * 16 * AM_TILE / 2; i += 256) reinterpret_cast<unsigned*>(&sp[0][0][0])[i] = 0u;   // rows NH..7 and 8 + NH..15 stay zero
    // QK: wave w takes the positions 16 (w & 1) + 0..15 and the scale groups 2 (w >> 1), + 1 (K steps 4 (w >> 1) + 0..3 of 16 dimensions)
    const int kq = w >> 1, ntq = w & 1;
    // q of head l16 & 7, dimensions 16 (4 kq + i) + 8 hi + 0..7; the head's q is scaled by 2^(14 - e) (e: the exponent of its largest |element|), so
    // that its parts are in f16's range whatever its size; the scores are scaled back with rs
    float4 qf[4][2];
    float qmx = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        qf[i][0] = qf[i][1] = make_float4(0.f, 0.f, 0.f, 0.f);
        if ((l16 & 7) < NH) {
            const float* qp = qg + ((size_t) t * NH + (l16 & 7)) * 2 * HD + 16 * (4 * kq + i) + 8 * hi;
            qf[i][0] = am_ld16f(qp);
            qf[i][1] = am_ld16f(qp + 4);
#pragma unroll
            for (int j = 0; j < 2; ++j)
                qmx = fmaxf(fmaxf(qmx, fmaxf(fabsf(qf[i][j].x), fabsf(qf[i][j].y))), fmaxf(fabsf(qf[i][j].z), fabsf(qf[i][j].w)));
        }
    }
    qmx = fmaxf(qmx, __shfl_xor_sync(0xffffffffu, qmx, 8));    // the lanes of a head: l16 & 7, both parts' rows and both hi
    qmx = fmaxf(qmx, __shfl_xor_sync(0xffffffffu, qmx, 16));
    if (lane < 8) sqm[lane][w] = qmx;
    __syncthreads();
    // the exponent e of head h's largest |q| (14: a zero q, scale 1; at least -100), from the 8 waves' maxima
    auto q_exp = [&](int h) {
        const float4 a = *reinterpret_cast<const float4*>(&sqm[h][0]), b = *reinterpret_cast<const float4*>(&sqm[h][4]);
        const float mx = fmaxf(fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w)), fmaxf(fmaxf(b.x, b.y), fmaxf(b.z, b.w)));
        const int ef = (int) ((__float_as_uint(mx) >> 23) & 255u);
        return ef ? max(ef - 127, -100) : 14;
    };
    uint4 qa[4];
    {
        const float qs = am_pow2(14 - q_exp(l16 & 7));
        const bool low = l16 >= 8;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 a = qf[i][0], b = qf[i][1];
            qa[i] = make_uint4(am_part(a.x * qs, low) | (am_part(a.y * qs, low) << 16), am_part(a.z * qs, low) | (am_part(a.w * qs, low) << 16),
                               am_part(b.x * qs, low) | (am_part(b.y * qs, low) << 16), am_part(b.z * qs, low) | (am_part(b.w * qs, low) << 16));
        }
    }
    // a tile's loads of this thread: its position's (lane's) K and V codes of dimensions 32 w + 0..31, and in waves 0 / 1 its 8 K / V scales
    uint4 pk[2], pv[2], psc = make_uint4(0, 0, 0, 0);
    auto fetch = [&](int tile) {
        const int jl = (tile - tile0) * AM_TILE + lane;
        const bool ok = jl < jn;
        const unsigned e = srow[jl % AM_ROWBUF];
        const size_t pb = (size_t) (e >> 8) * pg2, off = pb + (size_t) (e & 255u) * KV_W + 32 * w;
        const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            pk[j] = zero;
            pv[j] = zero;
            if (ok) {
                pk[j] = am_ld16(kc + off + 16 * j);
                pv[j] = am_ld16(vc + off + 16 * j);
            }
        }
        if (w < 2) {
            psc = zero;
            if (ok) psc = am_ld16((w ? vc : kc) + pb + (size_t) AM_PROWS * KV_W + (size_t) (e & 255u) * (2 * AM_G));
        }
    };
    float m = -INFINITY, z = 0.f;
    am_f8 acc[2] = {am_f8{}, am_f8{}};
    const float rs = rsqrtf((float) HD) * am_pow2(q_exp(min(w, 7)) - 14);   // waves 0..NH - 1 are the heads' softmax waves: q's scale back
    fetch(tile0);
#pragma unroll 1
    for (int tile = tile0; tile < tile1; ++tile) {
        const int rel = tile - tile0;
        __syncthreads();   // the last tile's products are done with sk, svt, sp and the scales
        {
            uint4 lo, up;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                am_codes(pk[j], lo, up);
                *reinterpret_cast<uint4*>(&sk[lane][32 * w + 16 * j]) = lo;
                *reinterpret_cast<uint4*>(&sk[lane][32 * w + 16 * j + 8]) = up;
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                am_codes(pv[j], lo, up);
                const unsigned c[8] = {lo.x, lo.y, lo.z, lo.w, up.x, up.y, up.z, up.w};
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    svt[32 * w + 16 * j + 2 * i][lane] = (unsigned short) (c[i] & 0xffffu);
                    svt[32 * w + 16 * j + 2 * i + 1][lane] = (unsigned short) (c[i] >> 16);
                }
            }
        }
        if (w < 2) {
            const unsigned c[4] = {psc.x, psc.y, psc.z, psc.w};
#pragma unroll
            for (int gq = 0; gq < AM_G; ++gq) {
                const float sc = am_h2f(gq & 1 ? c[gq / 2] >> 16 : c[gq / 2] & 0xffffu);
                if (w == 0) {
                    sks[lane][gq] = sc;
                } else {   // the group's V scales times 2^kg, the largest in [2^7, 2^8); 2^-kg for the products
                    const float vmax = wmax(sc);
                    const int ef = (int) ((__float_as_uint(vmax) >> 23) & 255u);   // 0: no scale above zero in the tile
                    const int ev = ef ? ef - 127 : 7;
                    svs[lane][gq] = sc * am_pow2(7 - ev);
                    if (lane == 0) sfg[gq] = am_pow2(ev - 7);
                }
            }
        }
        if ((rel + 1) % (AM_ROWBUF / AM_TILE) == 0 && tile + 1 < tile1) fill((rel + 1) * AM_TILE);   // the next tiles' row slots (this tile's are fetched)
        __syncthreads();
        if (tile + 1 < tile1) fetch(tile + 1);
        {   // scores: per scale group a product of its own, times the position's scale
            float part[NH];
#pragma unroll
            for (int r = 0; r < NH; ++r) part[r] = 0.f;
#pragma unroll
            for (int gi = 0; gi < 2; ++gi) {
                const int gq = 2 * kq + gi;
                am_f8 c = am_f8{};
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 b = *reinterpret_cast<const uint4*>(&sk[16 * ntq + l16][16 * (2 * gq + i) + 8 * hi]);
                    c = am_wmma(qa[2 * gi + i], b, c);
                }
                const float sc = sks[16 * ntq + l16][gq];
#pragma unroll
                for (int r = 0; r < NH; ++r) part[r] = fmaf(c[r], sc, part[r]);
            }
#pragma unroll
            for (int r = 0; r < NH; ++r) ss[kq][hi][r][16 * ntq + l16] = part[r];
        }
        __syncthreads();
        if (w < NH) {   // head w: lane = position; its score, the head's online-softmax step (k_attn_part's), then P's parts
            const bool ok = tile * AM_TILE + lane < ctx;
            float xh = 0.f, xl = 0.f;
#pragma unroll
            for (int q4 = 0; q4 < 4; ++q4) {
                xh += ss[q4][0][w][lane];
                xl += ss[q4][1][w][lane];
            }
            const float sv = ok ? (xh + xl) * rs : -INFINITY;
            const float mn = fmaxf(m, wmax(sv));
            const float wv = sv == -INFINITY ? 0.f : __expf(sv - mn);
            const float sc = m == -INFINITY ? 0.f : __expf(m - mn);
            z = z * sc + wsum(wv);
            m = mn;
            // (wv 2^kh) (vscale 2^kg) < 2^15 with the tile's largest weight times 2^kh in [2^6, 2^7). A tile whose largest weight is under 2^-88 of
            // the split's adds nothing (2^-88: sfh sfg stays a normal number down to the smallest f16 V scale)
            const float pmax = wmax(wv);
            const int eh = (int) ((__float_as_uint(pmax) >> 23) & 255u) - 127;
            const bool live = eh >= -88;
            const float pn = live ? wv * am_pow2(6 - eh) : 0.f;
            const float4 v0 = *reinterpret_cast<const float4*>(&svs[lane][0]), v1 = *reinterpret_cast<const float4*>(&svs[lane][4]);
            const float vs[AM_G] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
            for (int gq = 0; gq < AM_G; ++gq) {
                const float xn = pn * vs[gq];
                const _Float16 a = (_Float16) xn;
                sp[gq][w][lane] = (unsigned short) am_hbits(a);
                sp[gq][8 + w][lane] = (unsigned short) am_hbits((_Float16) (xn - (float) a));
            }
            if (lane == 0) {
                salpha[w] = sc;
                sfh[w] = live ? am_pow2(eh - 6) : 0.f;
            }
        }
        __syncthreads();
        {   // values: wave w takes dimensions 32 w + 0..31 (V scale group w)
            am_f8 c[2] = {am_f8{}, am_f8{}};
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const uint4 a = *reinterpret_cast<const uint4*>(&sp[w][l16][16 * ks + 8 * hi]);
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    const uint4 b = *reinterpret_cast<const uint4*>(&svt[32 * w + 16 * n + l16][16 * ks + 8 * hi]);
                    c[n] = am_wmma(a, b, c[n]);
                }
            }
            const float fg = sfg[w];
#pragma unroll
            for (int r = 0; r < NH; ++r) {
                const float al = salpha[r], f = sfh[r] * fg;
#pragma unroll
                for (int n = 0; n < 2; ++n) acc[n][r] = fmaf(c[n][r], f, acc[n][r] * al);
            }
        }
    }
    if (lane == 0 && w < NH) mz_out[((size_t) t * NH + w) * ATT_SPLITS + s] = AttPart{m, z};
    // lanes l and l + 16 hold the high and the low parts' sums of the same head and dimension
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
        for (int r = 0; r < NH; ++r) {
            const float v = acc[n][r] + __shfl_xor_sync(0xffffffffu, acc[n][r], 16);
            if (hi == 0) acc_out[(((size_t) t * NH + r) * ATT_SPLITS + s) * HD + 32 * w + 16 * n + l16] = v;
        }
}

// k_attn_combine over the first ns splits. (FNX_ATT_Q8, on in an APPROX build): with q it also writes the output as q8_1 rows (and their twin), which
// k_quantize did in a launch of its own on the same grid (a block = a head of a row = 8 groups of 32; a launch less in every attention layer). The
// same bits: the value is read back from `out` (a volatile load, so the product that made it cannot contract with the group's sum), then k_quantize's steps
__global__ void __launch_bounds__(HD) k_attn_combine_n(const float* qg, const float* acc_in, const AttPart* mz_in, float* out, int nq, int ns, block_q8_1* q,
                                                       int8_t* tq, float* td) {
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const size_t base = ((size_t) t * nq + h) * ATT_SPLITS;
    float mx = -INFINITY;
    for (int s = 0; s < ns; ++s) mx = fmaxf(mx, mz_in[base + s].m);
    float z = 0.f, acc = 0.f;
    for (int s = 0; s < ns; ++s) {
        const AttPart p = mz_in[base + s];
        if (p.m == -INFINITY) continue;
        const float w = __expf(p.m - mx);
        z = fmaf(w, p.z, z);
        acc = fmaf(w, acc_in[(base + s) * HD + i], acc);
    }
    out[((size_t) t * nq + h) * HD + i] = acc / z * sigm(qg[((size_t) t * nq + h) * 2 * HD + HD + i]);
    if (!q) return;
    static_assert(HD % 32 == 0, "k_attn_combine_n: a head in whole groups of 32");
    const int n = nq * HD, b = h * (HD / 32) + (i >> 5), lane = i & 31;
    const float xi = *(volatile const float*) (out + (size_t) t * n + b * 32 + lane);
    const float amax = wmax(fabsf(xi)), sum = wsum(xi);
    const float d = amax / 127.0f;
    const int8_t qv = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
    block_q8_1& y = q[(size_t) t * (n / 32) + b];
    y.qs[lane] = qv;
    if (lane == 0) y.ds = make_half2(__float2half(d), __float2half(sum));
    if (tq) {
        tq[(size_t) t * n + b * 32 + lane] = qv;
        if (lane == 0) td[(size_t) t * (n / 32) + b] = __half2float(__float2half(d));
    }
}

// ---------------------------------------------------------------- the block selection
// k_idx_score's scores for up to TR rows a block: grid ((max_pools + 255) / 256, rows' blocks), a thread per pooled key. One sequence's window (no lr.pt):
// block y takes rows TR y .. and reads each key once; a batched window (lr.pt: a page table per row): block y is row y. The fma chain of a row and head
// is k_idx_score's, so the scores are the same bits; q is read at uniform addresses
template <int TR>
__global__ void __launch_bounds__(256) k_idx_score_mm(const float* __restrict__ q, const __half* __restrict__ pooled, KvPages g,
                                                      float* __restrict__ scores, int max_pools, const int* __restrict__ pos0p, LaneRows lr, int T) {
    const int t0 = blockIdx.y * (lr.pt ? 1 : TR), nrow = lr.pt ? 1 : min(TR, T - t0);
    int nv[TR], nv_max = 0;
#pragma unroll
    for (int r = 0; r < TR; ++r) {
        nv[r] = r < nrow ? (row_pos(lr, pos0p, t0 + r) + 1) / QSA_POOL : 0;
        if (nv[r] <= QSA_TOP) nv[r] = 0;   // dense: no scores (k_idx_score's)
        nv_max = max(nv_max, nv[r]);
    }
    const int j = blockIdx.x * 256 + threadIdx.x;
    if (j >= nv_max) return;
    g = row_pages(g, lr, t0);
    const uint4* kr = reinterpret_cast<const uint4*>(pooled + kvp_block(g, j));
    const float* qr = q + (size_t) t0 * QSA_HEADS * QSA_DIM;
    float d[TR][QSA_HEADS];
#pragma unroll
    for (int r = 0; r < TR; ++r)
#pragma unroll
        for (int h = 0; h < QSA_HEADS; ++h) d[r][h] = 0.f;
#pragma unroll 1
    for (int c = 0; c < QSA_DIM / 8; ++c) {
        const uint4 u = kr[c];
        const __half2* h2 = reinterpret_cast<const __half2*>(&u);
        float kx[4], ky[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 kv = __half22float2(h2[k]);
            kx[k] = kv.x;
            ky[k] = kv.y;
        }
#pragma unroll
        for (int r = 0; r < TR; ++r) {
            if (r < nrow) {   // (no break: the rows unroll into registers of their own)
                const float* qt = qr + (size_t) r * QSA_HEADS * QSA_DIM + c * 8;
#pragma unroll
                for (int k = 0; k < 4; ++k)
#pragma unroll
                    for (int h = 0; h < QSA_HEADS; ++h)
                        d[r][h] = fmaf(qt[h * QSA_DIM + 2 * k], kx[k], fmaf(qt[h * QSA_DIM + 2 * k + 1], ky[k], d[r][h]));
            }
        }
    }
    const float w = rsqrtf((float) QSA_DIM);
#pragma unroll
    for (int r = 0; r < TR; ++r) {
        float s = 0.f;
#pragma unroll
        for (int h = 0; h < QSA_HEADS; ++h) s += fmaxf(d[r][h], 0.f) * w;
        if (j < nv[r]) scores[(size_t) (t0 + r) * max_pools + j] = s;   // nv[r] = 0 for r >= nrow
    }
}

constexpr int QS_NT = 1024, QS_CAP = 4096;
static_assert(QS_NT == 32 * 32 && QS_CAP % QS_NT == 0 && QSA_TOP <= QS_CAP, "32 waves: a wave reduces the waves' sums");

// wave 0: the digit at `shift` where the count from the highest bin down reaches *s_need (k_idx_select's search): the prefix with it, and what is left to
// take among the keys of that bin
__device__ __forceinline__ void qs_digit(const unsigned* hist, int shift, unsigned pre, unsigned* s_prefix, unsigned* s_need) {
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x, top = 255 - 8 * lane;   // lane l: bins top .. top - 7
        const unsigned need = *s_need;
        unsigned own = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) own += hist[top - i];
        unsigned inc = own;   // inclusive scan over the lanes (higher bins first)
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const unsigned v = __shfl_up_sync(0xffffffffu, inc, o);
            if (lane >= o) inc += v;
        }
        const unsigned before = inc - own;
        const unsigned hit = __ballot_sync(0xffffffffu, inc >= need);
        const int first = hit ? __ffs(hit) - 1 : 31;
        if (lane == first) {
            unsigned acc = before;
            int b = top - 7;
            for (int i = 0; i < 8; ++i) {
                const int bb = top - i;
                if (bb == 0 || acc + hist[bb] >= need) {
                    b = bb;
                    break;
                }
                acc += hist[bb];
            }
            *s_prefix = pre | ((unsigned) b << shift);
            *s_need = need - acc;
        }
    }
}

// the sum of v over the threads before this one (tmp: 32 ints), and the block's total
__device__ __forceinline__ int qs_scan(int v, int* tmp, int& total) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) tmp[wid] = x;
    __syncthreads();
    if (wid == 0) {
        int ws = tmp[lane];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, ws, o);
            if (lane >= o) ws += y;
        }
        tmp[lane] = ws;
    }
    __syncthreads();
    const int before = (wid > 0 ? tmp[wid - 1] : 0) + x - v;
    total = tmp[31];
    __syncthreads();
    return before;
}

// the K-th largest of the n keys key_at(0 .. n - 1) (K <= n): *s_prefix its value, *s_need how many keys equal to it are among the K largest. A radix
// select from the highest byte in which the keys differ (the bytes above it are the same in every key: a pass over them would send every key's atomic
// to one bin)
template <typename KF>
__device__ __forceinline__ void qs_select(KF key_at, int n, unsigned K, unsigned* hist, unsigned* s_mn, unsigned* s_mx, unsigned* s_prefix,
                                          unsigned* s_need) {
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    unsigned mn = 0xffffffffu, mx = 0u;
    for (int j = tid; j < n; j += QS_NT) {
        const unsigned k = key_at(j);
        mn = min(mn, k);
        mx = max(mx, k);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        mn = min(mn, (unsigned) __shfl_xor_sync(0xffffffffu, mn, o));
        mx = max(mx, (unsigned) __shfl_xor_sync(0xffffffffu, mx, o));
    }
    if (lane == 0) {
        s_mn[wid] = mn;
        s_mx[wid] = mx;
    }
    __syncthreads();
    if (wid == 0) {
        mn = s_mn[lane];
        mx = s_mx[lane];
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            mn = min(mn, (unsigned) __shfl_xor_sync(0xffffffffu, mn, o));
            mx = max(mx, (unsigned) __shfl_xor_sync(0xffffffffu, mx, o));
        }
        if (lane == 0) {
            s_mn[0] = mn;
            s_mx[0] = mx;
        }
    }
    __syncthreads();
    mn = s_mn[0];
    mx = s_mx[0];
    const unsigned diff = mn ^ mx;
    const int topbit = diff ? 31 - __clz((int) diff) : -1;
    const int start = topbit < 0 ? -8 : topbit / 8 * 8;   // the first byte to resolve (-8: every key is the same)
    unsigned msk = start >= 24 ? 0u : start < 0 ? 0xffffffffu : ~((1u << (start + 8)) - 1u);
    if (tid == 0) {
        *s_prefix = mx & msk;
        *s_need = K;
    }
    __syncthreads();
    for (int shift = start; shift >= 0; shift -= 8) {
        if (tid < 256) hist[tid] = 0u;
        __syncthreads();
        const unsigned pre = *s_prefix;
        for (int j = tid; j < n; j += QS_NT) {
            const unsigned k = key_at(j);
            if ((k & msk) == pre) atomicAdd(&hist[(k >> shift) & 255u], 1u);
        }
        __syncthreads();
        qs_digit(hist, shift, pre, s_prefix, s_need);
        msk |= 255u << shift;
        __syncthreads();
    }
}

// the n keys key_at(a) of blocks idx_at(a), in ascending block order: those above thr and the first `need` equal to it, as k_idx_select writes them
template <typename KF, typename IF>
__device__ __forceinline__ void qs_compact(KF key_at, IF idx_at, int n, unsigned thr, int need, int* tmp, int* row) {
    const int tid = threadIdx.x, ch = (n + QS_NT - 1) / QS_NT, a0 = min(n, tid * ch), a1 = min(n, a0 + ch);
    int gt = 0, eq = 0;
    for (int a = a0; a < a1; ++a) {
        const unsigned k = key_at(a);
        gt += k > thr;
        eq += k == thr;
    }
    int total;
    const int eq_before = qs_scan(eq, tmp, total);
    int take_eq = min(max(need - eq_before, 0), eq);   // this chunk's share of the first `need` ties
    int out = qs_scan(gt + take_eq, tmp, total);
    for (int a = a0; a < a1; ++a) {
        const unsigned k = key_at(a);
        bool take = k > thr;
        if (k == thr && take_eq > 0) {
            take = true;
            --take_eq;
        }
        if (take) {
            const int j = idx_at(a);
#pragma unroll
            for (int m = 0; m < QSA_POOL; ++m) row[out * QSA_POOL + m] = kv_local_block(j) * QSA_POOL + m;
            ++out;
        }
    }
}

// grid T, QS_NT threads: k_idx_select's cells[t] and n_cells[t] (by heads every block is this card's)
__global__ void __launch_bounds__(QS_NT) k_idx_select_mm(const float* __restrict__ scores, int* __restrict__ cells, int* __restrict__ n_cells,
                                                         int max_pools, const int* pos0p, LaneRows lr) {
    const int t = blockIdx.x, tid = threadIdx.x, p = row_pos(lr, pos0p, t), nv = (p + 1) / QSA_POOL, n_tail = (p + 1) % QSA_POOL;
    if (nv <= QSA_TOP) {
        if (tid == 0) n_cells[t] = -1;
        return;
    }
    const unsigned* key = reinterpret_cast<const unsigned*>(scores + (size_t) t * max_pools);   // scores >= 0: their bits order as unsigned
    __shared__ unsigned ck[QS_CAP];   // the sample's keys, then the candidates'
    __shared__ unsigned ci[QS_CAP];   // the candidates' blocks
    __shared__ unsigned hist[256];
    __shared__ int tmp[32];
    __shared__ unsigned s_mn[32], s_mx[32];
    __shared__ unsigned s_prefix, s_need;
    int* row = cells + (size_t) t * QSA_MAXSEL;
    // 1. tau: with a stride of 2^sh (the sample: at most QS_CAP keys), the kq-th largest of the sample. About 512 / 2^sh of the 512 largest keys are in
    // the sample; kq is 4 standard deviations above that, so 512 or more keys reach tau but for a rare sample (checked below)
    unsigned tau = 0u;
    if (nv > QS_CAP) {
        int sh = 1;
        while (((nv - 1) >> sh) + 1 > QS_CAP) ++sh;
        const int m = ((nv - 1) >> sh) + 1, base = max(QSA_TOP >> sh, 1);
        int rt = 1;
        while (rt * rt < base) ++rt;
        const int kq = base + 4 * rt + 2;
        for (int a = tid; a < m; a += QS_NT) ck[a] = key[(size_t) a << sh];
        __syncthreads();
        qs_select([&](int a) { return ck[a]; }, m, (unsigned) min(kq, m), hist, s_mn, s_mx, &s_prefix, &s_need);
        tau = s_prefix;
        __syncthreads();
    }
    // 2. the keys >= tau of this thread's chunk of blocks (chunks in block order)
    const int chunk = ((nv + QS_NT - 1) / QS_NT + 3) & ~3, j0 = min(nv, tid * chunk), j1 = min(nv, j0 + chunk);
    auto ld4 = [&](int j, unsigned* k) {   // keys j .. j + 3 (0 past the chunk)
        k[0] = k[1] = k[2] = k[3] = 0u;
        if (j + 4 <= j1) {
            const uint4 v = am_ld16(key + j);
            k[0] = v.x, k[1] = v.y, k[2] = v.z, k[3] = v.w;
        } else {
#pragma unroll
            for (int i = 0; i < 3; ++i)
                if (j + i < j1) k[i] = key[j + i];
        }
    };
    int cnt = 0;
    for (int j = j0; j < j1; j += 4) {
        unsigned k[4];
        ld4(j, k);
#pragma unroll
        for (int i = 0; i < 4; ++i) cnt += j + i < j1 && k[i] >= tau;
    }
    int tot;
    int wr = qs_scan(cnt, tmp, tot);
    if (tot >= QSA_TOP && tot <= QS_CAP) {
        for (int j = j0; j < j1; j += 4) {
            unsigned k[4];
            ld4(j, k);
#pragma unroll
            for (int i = 0; i < 4; ++i)
                if (j + i < j1 && k[i] >= tau) {
                    ck[wr] = k[i];
                    ci[wr] = (unsigned) (j + i);
                    ++wr;
                }
        }
        __syncthreads();
        // 3. the exact threshold among the candidates (every key above the 512th largest is one), 4. the list
        qs_select([&](int a) { return ck[a]; }, tot, (unsigned) QSA_TOP, hist, s_mn, s_mx, &s_prefix, &s_need);
        const unsigned thr = s_prefix;
        const int need = (int) s_need;
        __syncthreads();
        qs_compact([&](int a) { return ck[a]; }, [&](int a) { return (int) ci[a]; }, tot, thr, need, tmp, row);
    } else {   // the sample failed (or the ties at tau are many): the same select over every score
        qs_select([&](int a) { return key[a]; }, nv, (unsigned) QSA_TOP, hist, s_mn, s_mx, &s_prefix, &s_need);
        const unsigned thr = s_prefix;
        const int need = (int) s_need;
        __syncthreads();
        qs_compact([&](int a) { return key[a]; }, [&](int a) { return a; }, nv, thr, need, tmp, row);
    }
    if (tid == 0) {   // the tail (k_idx_select's)
        int n = QSA_TOP * QSA_POOL;
        const int jt = p / QSA_POOL;
        for (int m = 0; m < n_tail; ++m) row[n++] = kv_local_block(jt) * QSA_POOL + (p - n_tail + 1 + m) % QSA_POOL;
        n_cells[t] = n;
    }
}

// k_idx_score_rt<TOK> (FNX_PF_SCORE_RT, the prompt path): k_idx_score_tc's scores bit for bit (per 16 query rows x 16 keys the same
// v_wmma products in the same order: d0 ascending, q's high then low half; the heads' relus summed in k_idx_score's order), with TOK
// prompt rows a block instead of 16: a key tile is read once for TOK rows. Wave w holds its MW row tiles' q (f16 high and low halves) in
// registers for the whole walk; the block walks kr keys (a multiple of 16: tiles never cross a page of KVP_BLOCKS) from blockIdx.x * kr
template <int TOK>
__global__ void __launch_bounds__(256) k_idx_score_rt(const float* __restrict__ q, const __half* __restrict__ pooled, KvPages g,
                                                      float* __restrict__ scores, int max_pools, const int* pos0p, int T, int kr) {
    constexpr int MT = TOK * QSA_HEADS / 16, MW = MT / 8, DS = QSA_DIM / 16;
    static_assert(MT % 8 == 0 && QSA_HEADS == 4 && KVP_BLOCKS % 16 == 0, "8 waves, MW row tiles each; a row tile = 4 rows x 4 heads");
    const int t0 = blockIdx.y * TOK, jb = blockIdx.x * kr, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tl = min(T, t0 + TOK) - 1;   // the block's last row sees the most blocks
    const int nv_max = (*pos0p + tl + 1) / QSA_POOL;
    if (nv_max <= QSA_TOP || jb >= nv_max) return;
    const int je = min(nv_max, jb + kr), kq = 8 * (lane >> 4);
    uint4 ah[MW][DS], al[MW][DS];
#pragma unroll
    for (int m = 0; m < MW; ++m) {
        const int row = (warp * MW + m) * 16 + (lane & 15);   // the block's q row: prompt row row / 4, head row % 4
        const bool live = t0 + row / QSA_HEADS < T;
        const float* qr = q + ((size_t) t0 * QSA_HEADS + row) * QSA_DIM + kq;
#pragma unroll
        for (int d = 0; d < DS; ++d) {
            float x[8];
            if (live) {
                const float4 u = *reinterpret_cast<const float4*>(qr + d * 16), v = *reinterpret_cast<const float4*>(qr + d * 16 + 4);
                x[0] = u.x; x[1] = u.y; x[2] = u.z; x[3] = u.w; x[4] = v.x; x[5] = v.y; x[6] = v.z; x[7] = v.w;
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) x[i] = 0.f;
            }
            __half hh[8], hl[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                hh[i] = __float2half_rn(x[i]);
                hl[i] = __float2half_rn(x[i] - __half2float(hh[i]));
            }
            __builtin_memcpy(&ah[m][d], hh, 16);
            __builtin_memcpy(&al[m][d], hl, 16);
        }
    }
    const float w = rsqrtf((float) QSA_DIM);
    for (int jt = jb; jt < je; jt += 16) {
        const __half* kp = pooled + kvp_block(g, jt) + (size_t) (lane & 15) * QSA_DIM + kq;   // 16 blocks never cross a page
        uint4 b[DS];
#pragma unroll
        for (int d = 0; d < DS; ++d) b[d] = *reinterpret_cast<const uint4*>(kp + d * 16);
#pragma unroll
        for (int m = 0; m < MW; ++m) {
            am_f8 c = {};
#pragma unroll
            for (int d = 0; d < DS; ++d) {
                c = am_wmma(ah[m][d], b[d], c);
                c = am_wmma(al[m][d], b[d], c);
            }
            // c[r]: q row 8 (lane >> 4) + r of the tile, key jt + (lane & 15): rows r = 4 a + h are prompt row 2 (lane >> 4) + a, head h
            const int j = jt + (lane & 15);
#pragma unroll
            for (int a = 0; a < 2; ++a) {
                const int t = t0 + (warp * MW + m) * 4 + 2 * (lane >> 4) + a;
                float sc = 0.f;
#pragma unroll
                for (int h = 0; h < QSA_HEADS; ++h) sc += fmaxf(c[4 * a + h], 0.f) * w;
                if (t < T && j < max_pools) scores[(size_t) t * max_pools + j] = sc;
            }
        }
    }
}
// FNX_PF_SCORE_CHECK (debug): scores a and b that differ in any bit, over each row's visible blocks of the rows past QSA_TOP of them
__global__ void k_score_cmp(const float* __restrict__ a, const float* __restrict__ b, int max_pools, const int* pos0p, int T, unsigned* bad) {
    const int t = blockIdx.y, nv = (*pos0p + t + 1) / QSA_POOL;
    if (t >= T || nv <= QSA_TOP) return;
    unsigned n = 0;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < nv; j += gridDim.x * blockDim.x)
        n += __float_as_uint(a[(size_t) t * max_pools + j]) != __float_as_uint(b[(size_t) t * max_pools + j]);
    if (n) atomicAdd(bad, n);
}
#endif   // FNX_ATT_MM
