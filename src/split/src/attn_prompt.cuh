// Attention over a prompt chunk (included by kernels.cu after the decode attention): one block per (row, KV head)
// walks the row's attended positions on this card, APC at a time, for the KV head's ATT_QPK query heads (rows of a
// 16-row tile) on the tensor cores: S = q k^T and O += p v as f16 x f16 with f32 accumulation, q and p rounded to f16
// as llama.cpp's CUDA attention takes them. Online softmax per head.
// The partial (acc, max, sum) of every head goes to part[owner card][row][head of ATT_OWN][HD + 2]; the merge across cards
// (k_pf_attn_merge, after the exchange) applies the gate.
namespace {

constexpr int APC = 32;                  // positions per tile
constexpr int AP_NPT = APC / 16;         // their 16-position tiles
constexpr int AP_THREADS = 256;
constexpr int AP_LD = HD + 8;            // halves per Q / K / V row in shared memory (a multiple of 8, for wmma)
constexpr int AP_LDF = HD + 4;           // floats per row of the f32 staging tile
constexpr int AP_PLD = APC + 8;          // halves per P row
constexpr int AP_W = HD + 2;             // floats per partial: acc, max, sum
constexpr size_t AP_SMEM = sizeof(__half) * (16 * AP_LD + APC * AP_LD + 16 * AP_PLD) + sizeof(float) * (16 * AP_LDF + 3 * 16);
static_assert(ATT_QPK <= 16, "a KV head's query heads in one 16-row tile");
// the query heads a block takes: its KV head's ATT_QPK (by position), or the card's NQH_LOC of its one KV head (FNX_ATT_HEADS)
constexpr int AP_QH = ATT_HEADS ? NQH_LOC : ATT_QPK;
static_assert(AP_THREADS / 32 * 2 * 16 == HD, "eight warps of two 16-column O tiles");
static_assert(AP_THREADS / 32 * APC <= AP_LDF && APC % 16 == 0, "the S partials of a tile side by side in st");

__global__ void __launch_bounds__(AP_THREADS) k_pf_attn(const float* __restrict__ qg, const __half* __restrict__ kc,
                                                        const __half* __restrict__ vc, KvPages g, const int* __restrict__ pos0p,
                                                        const int* __restrict__ cells, const int* __restrict__ n_cells, int card,
                                                        int C, int r0, float* __restrict__ part) {
    using namespace nvcuda;
    extern __shared__ __align__(16) unsigned char ap_smem[];
    __half* qh = reinterpret_cast<__half*>(ap_smem);           // [16][AP_LD]
    __half* kv = qh + 16 * AP_LD;                               // [APC][AP_LD]: K, then V
    __half* ph = kv + APC * AP_LD;                              // [16][AP_PLD]
    float* st = reinterpret_cast<float*>(ph + 16 * AP_PLD);     // [16][AP_LDF]: S partials, then the P V tile
    float* sm = st + 16 * AP_LDF;                               // [16]: running max
    float* sz = sm + 16;                                        // [16]: running sum
    float* ssc = sz + 16;                                       // [16]: this tile's rescale
    const int t = r0 + blockIdx.x, kvh = blockIdx.y, tid = threadIdx.x, warp = tid >> 5;
    const int* list = cells && n_cells[t] >= 0 ? cells + (size_t) t * QSA_MAXSEL : nullptr;
    const int ctx = list ? n_cells[t] : kv_owned(*pos0p + t, card);
    // q rows (heads kvh * QPK + j; by heads the card's j) as f16; rows past AP_QH are zero
    for (int e = tid; e < 16 * HD; e += AP_THREADS) {
        const int r = e / HD, d = e % HD;
        const float x = r < AP_QH ? qg[((size_t) t * NQH_LOC + kvh * AP_QH + r) * 2 * HD + d] : 0.f;
        qh[r * AP_LD + d] = __float2half_rn(x);
    }
    if (tid < 16) {
        sm[tid] = -INFINITY;
        sz[tid] = 0.f;
    }
    // this thread's O elements: row orow, columns ocol + 16 j
    const int orow = tid / 16, ocol = tid % 16;
    float o[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) o[j] = 0.f;
    const float scale = rsqrtf((float) HD);
    // a tile's K or V rows, 8 halves a load, gathered into registers ahead of their use: V during S and the softmax,
    // the next tile's K during P V (each a load of the position's index, then of the row)
    constexpr int AP_GL = APC * HD / 8 / AP_THREADS;
    static_assert(APC * HD / 8 % AP_THREADS == 0, "a tile's loads spread evenly");
    auto gather = [&](const __half* base, int c, int nr, uint4 (&r)[AP_GL]) {
#pragma unroll
        for (int i = 0; i < AP_GL; ++i) {
            const int e = tid + i * AP_THREADS, pp = e / (HD / 8), d8 = e % (HD / 8);
            r[i] = make_uint4(0, 0, 0, 0);
            if (pp < nr) {
                const int pos = list ? list[c + pp] : c + pp;
                if (g.q8) {   // 8 int8 values and their group's scale, as 8 halves
                    const size_t at = kvq_row(g, pos) + (size_t) kvh * HD + d8 * 8;
                    const uint2 q = *reinterpret_cast<const uint2*>(reinterpret_cast<const int8_t*>(base) + at);
                    const float d = __half2float(base[kvq_scale(g, pos) + ((size_t) kvh * HD + d8 * 8) / KV8_GROUP]);
                    const int8_t* qb = reinterpret_cast<const int8_t*>(&q);
                    __align__(16) __half h[8];
#pragma unroll
                    for (int k = 0; k < 8; ++k) h[k] = __float2half_rn(qb[k] * d);
                    r[i] = *reinterpret_cast<const uint4*>(h);
                } else {
                    r[i] = *reinterpret_cast<const uint4*>(base + kvp_row(g, pos) + (size_t) kvh * HD + d8 * 8);
                }
            }
        }
    };
    auto place = [&](const uint4 (&r)[AP_GL]) {
#pragma unroll
        for (int i = 0; i < AP_GL; ++i) {
            const int e = tid + i * AP_THREADS, pp = e / (HD / 8), d8 = e % (HD / 8);
            *reinterpret_cast<uint4*>(kv + pp * AP_LD + d8 * 8) = r[i];
        }
    };
    uint4 kr[AP_GL], vr[AP_GL];
    if (ctx > 0) gather(kc, 0, min(APC, ctx), kr);
    for (int c0 = 0; c0 < ctx; c0 += APC) {
        const int np = min(APC, ctx - c0);
        __syncthreads();   // the last tile's V and P reads are done
        place(kr);
        gather(vc, c0, np, vr);
        __syncthreads();
        // S partials: warp w takes dimensions [32 w, 32 w + 32) (two k-steps) for each 16-position tile
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fs[AP_NPT];
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) wmma::load_matrix_sync(fa[ks], qh + warp * 32 + ks * 16, AP_LD);
#pragma unroll
            for (int pt = 0; pt < AP_NPT; ++pt) {
                wmma::fill_fragment(fs[pt], 0.f);
#pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    wmma::load_matrix_sync(fb, kv + pt * 16 * AP_LD + warp * 32 + ks * 16, AP_LD);
                    wmma::mma_sync(fs[pt], fa[ks], fb, fs[pt]);
                }
                // the eight warps' partials side by side in st: warp w at columns [APC w, APC w + APC)
                wmma::store_matrix_sync(st + warp * APC + pt * 16, fs[pt], AP_LDF, wmma::mem_row_major);
            }
        }
        __syncthreads();
        // the scores: thread (row r, positions c + 16 u) sums the eight partials; the row's online-softmax step (16
        // lanes a row)
        {
            const int r = tid / 16, c = tid % 16;
            float sv[AP_NPT], mx = -INFINITY;
#pragma unroll
            for (int u = 0; u < AP_NPT; ++u) {
                float sacc = 0.f;
#pragma unroll
                for (int w = 0; w < AP_THREADS / 32; ++w) sacc += st[r * AP_LDF + w * APC + u * 16 + c];
                sv[u] = c + 16 * u < np ? sacc * scale : -INFINITY;
                mx = fmaxf(mx, sv[u]);
            }
#pragma unroll
            for (int off = 8; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off, 16));
            const float mo = sm[r], mn = fmaxf(mo, mx);
            float ps = 0.f;
#pragma unroll
            for (int u = 0; u < AP_NPT; ++u) {
                const float p = sv[u] == -INFINITY ? 0.f : __expf(sv[u] - mn);
                ps += p;
                ph[r * AP_PLD + u * 16 + c] = __float2half_rn(p);
            }
#pragma unroll
            for (int off = 8; off; off >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, off, 16);
            __syncwarp();
            if (c == 0) {
                const float sc = mo == -INFINITY ? 0.f : __expf(mo - mn);
                ssc[r] = sc;
                sz[r] = sz[r] * sc + ps;
                sm[r] = mn;
            }
        }
        // V rows of the tile into the K buffer (its S reads are done: the barrier orders them)
        __syncthreads();
        place(vr);
        __syncthreads();
        if (c0 + APC < ctx) gather(kc, c0 + APC, min(APC, ctx - c0 - APC), kr);
        // P V: warp w takes O columns [32 w, 32 w + 32) (two 16-column tiles), the tile's positions in k-steps of 16
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fp[AP_NPT];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fv;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fo;
#pragma unroll
            for (int ks = 0; ks < AP_NPT; ++ks) wmma::load_matrix_sync(fp[ks], ph + ks * 16, AP_PLD);
#pragma unroll
            for (int ct = 0; ct < 2; ++ct) {
                const int d0 = warp * 32 + ct * 16;
                wmma::fill_fragment(fo, 0.f);
#pragma unroll
                for (int ks = 0; ks < AP_NPT; ++ks) {
                    wmma::load_matrix_sync(fv, kv + ks * 16 * AP_LD + d0, AP_LD);
                    wmma::mma_sync(fo, fp[ks], fv, fo);
                }
                wmma::store_matrix_sync(st + d0, fo, AP_LDF, wmma::mem_row_major);
            }
        }
        __syncthreads();
        const float sc = ssc[orow];
#pragma unroll
        for (int j = 0; j < 16; ++j) o[j] = fmaf(o[j], sc, st[orow * AP_LDF + ocol + 16 * j]);
    }
    __syncthreads();
    if (orow < AP_QH) {
        // by heads every head is this card's own: block `card`
        const int h = kvh * AP_QH + orow, k = ATT_HEADS ? card : h / ATT_OWN, hl = h % ATT_OWN;
        float* dst = part + (((size_t) k * C + t) * ATT_OWN + hl) * AP_W;
#pragma unroll
        for (int j = 0; j < 16; ++j) dst[ocol + 16 * j] = o[j];
        if (ocol == 0) {
            dst[HD] = sm[orow];
            dst[HD + 1] = sz[orow];
        }
    }
}

#if FNX_HIP && FNX_ATT_HEADS
// FNX_PF_ATT_UNION=R (by heads, HIP): R consecutive rows walk the union of their selected positions once (K/V loaded once
// for the R rows instead of once a row), each row masked to its own set. Selected positions come in whole QSA_POOL blocks (plus a row's
// incomplete last block, its tail): the union is a list of blocks, each with a mask of the rows that attend it; a row attends position q of a
// listed block iff its bit is set and q <= its own position (that cuts its tail block and the later rows' blocks). Dense rows (n_cells < 0:
// every visible position) take blocks 0 .. p / QSA_POOL. Same set of positions as k_pf_attn; the online softmax runs over other tiles, so the
// last bits differ. A masked position's V still enters P V with weight 0: a nonfinite V there (a diverged cache, which the
// rings' NaN guard fails anyway) would reach rows that do not attend it.
constexpr int AU_UL(int R) { return R * (QSA_TOP + 1); }   // a group's union entries, at most (a row: QSA_TOP blocks + its tail, or a dense row's <= QSA_TOP + 1)
constexpr int au_pow2(int v) { int n = 1; while (n < v) n <<= 1; return n; }
// one block a group: the group's (block << 3 | row) keys sorted, then one entry (block << 8 | row mask) a block, ascending
template <int R>
__global__ void __launch_bounds__(256) k_pf_union(const int* __restrict__ cells, const int* __restrict__ n_cells, const int* __restrict__ pos0p,
                                                  int r0, int T, int* __restrict__ ulist, int* __restrict__ ucount) {
    static_assert(R >= 1 && R <= 8, "a row mask of 8 bits");
    constexpr int NS = au_pow2(AU_UL(R));
    __shared__ int key[NS];
    __shared__ int cnt[256];
    const int grp = blockIdx.x, tid = threadIdx.x, t0 = r0 + grp * R, nr = min(R, r0 + T - t0), pos0 = *pos0p;
    for (int i = tid; i < NS; i += 256) key[i] = 0x7fffffff;
    __syncthreads();
    for (int r = 0; r < nr; ++r) {
        const int t = t0 + r, n = n_cells[t];
        int* dst = key + r * (QSA_TOP + 1);
        if (n < 0) {
            const int nb = (pos0 + t) / QSA_POOL + 1;
            for (int b = tid; b < nb; b += 256) dst[b] = b << 3 | r;
        } else {
            const int* list = cells + (size_t) t * QSA_MAXSEL;
            const int nsel = n / QSA_POOL;
            for (int b = tid; b < nsel; b += 256) dst[b] = (list[b * QSA_POOL] / QSA_POOL) << 3 | r;
            if (n % QSA_POOL && tid == 0) dst[nsel] = (list[nsel * QSA_POOL] / QSA_POOL) << 3 | r;
        }
    }
    __syncthreads();
    for (int k = 2; k <= NS; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < NS; i += 256) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const int a = key[i], b = key[ixj];
                    if ((a > b) == ((i & k) == 0)) key[i] = b, key[ixj] = a;
                }
            }
            __syncthreads();
        }
    // block starts: thread tid takes keys [tid * PER, tid * PER + PER)
    constexpr int PER = NS / 256;
    static_assert(NS % 256 == 0, "keys spread evenly");
    auto start = [&](int i) { return key[i] != 0x7fffffff && (i == 0 || key[i] >> 3 != key[i - 1] >> 3); };
    int mine = 0;
    for (int i = tid * PER; i < tid * PER + PER; ++i) mine += start(i);
    cnt[tid] = mine;
    __syncthreads();
    for (int o = 1; o < 256; o <<= 1) {
        const int a = tid >= o ? cnt[tid - o] : 0;
        __syncthreads();
        cnt[tid] += a;
        __syncthreads();
    }
    int out = cnt[tid] - mine;
    int* ul = ulist + (size_t) grp * AU_UL(R);
    for (int i = tid * PER; i < tid * PER + PER; ++i) {
        if (!start(i)) continue;
        const int blk = key[i] >> 3;
        int m = 0;
        for (int e = i; e < NS && key[e] != 0x7fffffff && key[e] >> 3 == blk; ++e) m |= 1 << (key[e] & 7);
        ul[out++] = blk << 8 | m;
    }
    if (tid == 255) ucount[grp] = cnt[255];
}

// REG (FNX_PF_ATT_UREG=1): O in the warps' accumulator tiles (rescaled before each P V product) instead of a P V tile staged in st and added
// to registers: no st tile for P V (less shared memory), two fewer barriers a query tile. The sums' order changes (last bits)
// PC (FNX_PF_ATT_PC=64, REG only): positions a tile; 64 gives every warp a score tile at R = 4 and half the barriers a position
// LEAN (FNX_PF_ATT_LEAN=1, REG only): less shared memory for 4 blocks a WGP. Q and P keep only their QR live rows (a 16-row fragment load
// of the last tile reads past them into the next buffer: those rows' scores, P and O are garbage and never used, since an MMA row depends
// only on its own A row), and the score tile lives in the K/V buffer (one more barrier a tile: the scores stay in registers until every
// warp has read its K)
template <int R, bool REG = false, int PC = APC, bool LEAN = false>
struct AuShape {
    static_assert(PC == APC || REG, "a wider tile only with O in accumulators");
    static_assert(!LEAN || REG, "the lean layout only with O in accumulators");
    static constexpr int QR = R * AP_QH, QT = (QR + 15) / 16, QM = QT * 16, SLD = PC + 4, PLD = PC + 8, NPT = PC / 16;
    static constexpr int QA = LEAN ? QR : QM;   // Q and P rows in shared memory
    static constexpr int GL = PC * HD / 8 / AP_THREADS;
    static constexpr int STF = LEAN ? 0 : REG || QM * SLD > 16 * AP_LDF ? QM * SLD : 16 * AP_LDF;
    static_assert(!LEAN || (size_t) QM * SLD * sizeof(float) <= (size_t) PC * AP_LD * sizeof(__half), "the score tile fits in the K/V buffer");
    static_assert(!LEAN || (QA * AP_LD) % 8 == 0 && (QA * PLD) % 8 == 0, "16-byte buffer starts");
    static constexpr size_t SMEM = sizeof(__half) * ((size_t) QA * AP_LD + PC * AP_LD + QA * PLD) + sizeof(float) * ((size_t) STF + 3 * QM) +
                                   sizeof(int) * (PC / QSA_POOL);
};

// AT (generator only, a variant somebody selects by name; the results are not valid): what the walk's time is made of. 1: the K/V bytes and scales are loaded but not made into
// halves (the 8 bytes go to shared memory as they are); 2: no K/V loads at all; 4: no score products; 8: no P V products (REG)
// GW (generator): 1: a thread gathers one (position, group of 32 values): two 16-byte loads and one scale load in place of four 8-byte loads
// and four scale loads (of four positions). The same values at the same places: the same bits. 2: also, a tile's V gather takes the row and scale
// places its K gather found (no second read of the union list and the page table)
template <int R, bool REG = false, int PC = APC, bool LEAN = false, int AT = 0, int GW = 0>
__global__ void __launch_bounds__(AP_THREADS) k_pf_attn_u(const float* __restrict__ qg, const __half* __restrict__ kc, const __half* __restrict__ vc,
                                                          KvPages g, const int* __restrict__ pos0p, const int* __restrict__ ulist,
                                                          const int* __restrict__ ucount, int card, int C, int r0, int T, float* __restrict__ part) {
    using namespace nvcuda;
    using S = AuShape<R, REG, PC, LEAN>;
    constexpr int QR = S::QR, QT = S::QT, QM = S::QM, SLD = S::SLD, PLD = S::PLD, NPT = S::NPT, PT = PC / 16;
    static_assert(ATT_HEADS && KVH_LOC == 1, "the union walk: by heads (one KV head a card)");
    static_assert(PC % QSA_POOL == 0 && AP_THREADS / 32 >= PT * QT, "a warp a (query tile, 16 positions) for the scores");
    extern __shared__ __align__(16) unsigned char au_smem[];
    // order: Q, P, K/V when LEAN (a fragment load past Q's or P's last live row stays inside the block's buffers), else Q, K/V, P, st
    __half* qh = reinterpret_cast<__half*>(au_smem);             // [QA][AP_LD]
    __half* kv = LEAN ? qh + S::QA * (AP_LD + PLD) : qh + QM * AP_LD;   // [PC][AP_LD]: K, then V
    __half* ph = LEAN ? qh + S::QA * AP_LD : kv + PC * AP_LD;       // [QA][PLD]
    float* st = LEAN ? reinterpret_cast<float*>(kv) : reinterpret_cast<float*>(ph + QM * PLD);   // scores [QM][SLD] (LEAN: in K/V), then a P V tile [16][AP_LDF]
    float* sm = LEAN ? reinterpret_cast<float*>(kv + PC * AP_LD) : st + S::STF;
    float* sz = sm + QM;
    float* ssc = sz + QM;
    int* um = reinterpret_cast<int*>(ssc + QM);                    // [PC / QSA_POOL]: the tile's union entries
    const int grp = blockIdx.x, tid = threadIdx.x, warp = tid >> 5;
    const int t0 = r0 + grp * R, nr = min(R, r0 + T - t0), pos0 = *pos0p, pmax = pos0 + t0 + nr - 1;
    const int* ul = ulist + (size_t) grp * AU_UL(R);
    const int ctx = ucount[grp] * QSA_POOL;
    for (int e = tid; e < S::QA * HD; e += AP_THREADS) {
        const int qr = e / HD, d = e % HD, rl = qr / AP_QH, h = qr % AP_QH;
        const float x = qr < QR && rl < nr ? qg[((size_t) (t0 + rl) * NQH_LOC + h) * 2 * HD + d] : 0.f;
        qh[qr * AP_LD + d] = __float2half_rn(x);
    }
    for (int r = tid; r < QM; r += AP_THREADS) sm[r] = -INFINITY, sz[r] = 0.f;
    const int orow = tid / 16, ocol = tid % 16;
    float o[QT][16];
#pragma unroll
    for (int q = 0; q < QT; ++q)
#pragma unroll
        for (int j = 0; j < 16; ++j) o[q][j] = 0.f;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[REG ? QT : 1][2];
    if constexpr (REG) {
#pragma unroll
        for (int q = 0; q < QT; ++q) wmma::fill_fragment(acc[q][0], 0.f), wmma::fill_fragment(acc[q][1], 0.f);
    }
    const float scale = rsqrtf((float) HD);
    constexpr int AP_GL = S::GL;
    // a tile's K or V rows: positions past the group's last row are not written yet (zero, so 0 * V stays 0)
    static_assert(GW >= 0 && GW <= 2 && (!GW || (AP_GL == 4 && PC * (HD / KV8_GROUP) == AP_THREADS && KV8_GROUP == 32)), "GW: a thread a (position, group)");
    static_assert(AT >= 0 && AT < 16 && !(AT & 3 && GW) && !(AT & 8 && !REG), "AT: bits 1 and 2 are in the first gather, bit 8 in the REG products");
    // GW = 2: the tile's row place (bytes of an int8 page, halves of an f16 page) and scale place (halves), from its K gather for its V gather.
    // Whole offsets: a physical page past 2830 is over 4 GiB from the layer's pointer
    constexpr size_t AU_NONE = ~(size_t) 0;
    size_t g_at = AU_NONE, g_sc = 0;
    auto gather = [&](const __half* base, int c, uint4 (&r)[AP_GL], bool is_v = false) {
        if constexpr (GW) {
            const int pp = tid / (HD / 32), gq = tid % (HD / 32);
#pragma unroll
            for (int i = 0; i < AP_GL; ++i) r[i] = make_uint4(0, 0, 0, 0);
            if constexpr (GW == 2) {
                if (is_v) {
                    if (g_at == AU_NONE) return;
                    if (g.q8) {
                        const int8_t* src = reinterpret_cast<const int8_t*>(base) + g_at;
                        const uint4 lo = *reinterpret_cast<const uint4*>(src), hi = *reinterpret_cast<const uint4*>(src + 16);
                        const float d = __half2float(base[g_sc]);
                        const int8_t *ql = reinterpret_cast<const int8_t*>(&lo), *qh8 = reinterpret_cast<const int8_t*>(&hi);
#pragma unroll
                        for (int i = 0; i < AP_GL; ++i) {
                            __align__(16) __half h[8];
#pragma unroll
                            for (int k = 0; k < 8; ++k) h[k] = __float2half_rn((i < 2 ? ql : qh8)[(i & 1) * 8 + k] * d);
                            r[i] = *reinterpret_cast<const uint4*>(h);
                        }
                    } else {
#pragma unroll
                        for (int i = 0; i < AP_GL; ++i) r[i] = *reinterpret_cast<const uint4*>(base + g_at + i * 8);
                    }
                    return;
                }
                g_at = AU_NONE;
            }
            if (c + pp >= ctx) return;
            const int pos = (ul[(c + pp) / QSA_POOL] >> 8) * QSA_POOL + (c + pp) % QSA_POOL;
            if (pos > pmax) return;
            if constexpr (GW == 2) {
                g_at = (g.q8 ? kvq_row(g, pos) : kvp_row(g, pos)) + gq * 32;
                g_sc = g.q8 ? kvq_scale(g, pos) + gq : 0;
            }
            if (g.q8) {
                const int8_t* src = reinterpret_cast<const int8_t*>(base) + kvq_row(g, pos) + gq * 32;
                const uint4 lo = *reinterpret_cast<const uint4*>(src), hi = *reinterpret_cast<const uint4*>(src + 16);
                const float d = __half2float(base[kvq_scale(g, pos) + gq]);
                const int8_t *ql = reinterpret_cast<const int8_t*>(&lo), *qh8 = reinterpret_cast<const int8_t*>(&hi);
#pragma unroll
                for (int i = 0; i < AP_GL; ++i) {
                    __align__(16) __half h[8];
#pragma unroll
                    for (int k = 0; k < 8; ++k) h[k] = __float2half_rn((i < 2 ? ql : qh8)[(i & 1) * 8 + k] * d);
                    r[i] = *reinterpret_cast<const uint4*>(h);
                }
            } else {
#pragma unroll
                for (int i = 0; i < AP_GL; ++i) r[i] = *reinterpret_cast<const uint4*>(base + kvp_row(g, pos) + gq * 32 + i * 8);
            }
            return;
        }
#pragma unroll
        for (int i = 0; i < AP_GL; ++i) {
            const int e = tid + i * AP_THREADS, pp = e / (HD / 8), d8 = e % (HD / 8);
            r[i] = make_uint4(0, 0, 0, 0);
            if constexpr (AT & 2) {
                r[i] = make_uint4(0x3c003800u + (unsigned) e, 0x38003c00u, 0x3c003800u, 0x38003c00u);
                continue;
            }
            if (c + pp < ctx) {
                const int pos = (ul[(c + pp) / QSA_POOL] >> 8) * QSA_POOL + (c + pp) % QSA_POOL;
                if (pos <= pmax) {
                    if (g.q8) {   // 8 int8 values and their group's scale, as 8 halves (as k_pf_attn)
                        const size_t at = kvq_row(g, pos) + d8 * 8;
                        const uint2 q = *reinterpret_cast<const uint2*>(reinterpret_cast<const int8_t*>(base) + at);
                        const float d = __half2float(base[kvq_scale(g, pos) + (d8 * 8) / KV8_GROUP]);
                        if constexpr (AT & 1) {
                            r[i] = make_uint4(q.x & 0x3bff3bffu, q.y & 0x3bff3bffu, __float_as_uint(d) & 0x3bff3bffu, 0);   // (finite halves)
                            continue;
                        }
                        const int8_t* qb = reinterpret_cast<const int8_t*>(&q);
                        __align__(16) __half h[8];
#pragma unroll
                        for (int k = 0; k < 8; ++k) h[k] = __float2half_rn(qb[k] * d);
                        r[i] = *reinterpret_cast<const uint4*>(h);
                    } else {
                        r[i] = *reinterpret_cast<const uint4*>(base + kvp_row(g, pos) + d8 * 8);
                    }
                }
            }
        }
    };
    auto place = [&](const uint4 (&r)[AP_GL]) {
        if constexpr (GW) {
            const int pp = tid / (HD / 32), gq = tid % (HD / 32);
#pragma unroll
            for (int i = 0; i < AP_GL; ++i) *reinterpret_cast<uint4*>(kv + pp * AP_LD + gq * 32 + i * 8) = r[i];
            return;
        }
#pragma unroll
        for (int i = 0; i < AP_GL; ++i) {
            const int e = tid + i * AP_THREADS, pp = e / (HD / 8), d8 = e % (HD / 8);
            *reinterpret_cast<uint4*>(kv + pp * AP_LD + d8 * 8) = r[i];
        }
    };
    uint4 kr[AP_GL], vr[AP_GL];
    if (ctx > 0) gather(kc, 0, kr);
    for (int c0 = 0; c0 < ctx; c0 += PC) {
        __syncthreads();   // the last tile's V and P reads are done
        place(kr);
        if (tid < PC / QSA_POOL) um[tid] = c0 / QSA_POOL + tid < ctx / QSA_POOL ? ul[c0 / QSA_POOL + tid] : 0;
        gather(vc, c0, vr, true);
        __syncthreads();
        // scores: warp w < 2 QT takes query tile w / 2 against the tile's positions 16 (w % 2) ..
        {
            const bool sw = warp < PT * QT;   // a score warp (wave-uniform)
            const int qt = warp / PT, pt = warp % PT;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fs;
            if (sw) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::fill_fragment(fs, 0.f);
#pragma unroll
                for (int ks = 0; ks < (AT & 4 ? 0 : HD / 16); ++ks) {
                    wmma::load_matrix_sync(fa, qh + qt * 16 * AP_LD + ks * 16, AP_LD);
                    wmma::load_matrix_sync(fb, kv + pt * 16 * AP_LD + ks * 16, AP_LD);
                    wmma::mma_sync(fs, fa, fb, fs);
                }
            }
            if constexpr (LEAN) __syncthreads();   // every score warp has read K: st is in the K/V buffer
            if (sw) wmma::store_matrix_sync(st + qt * 16 * SLD + pt * 16, fs, SLD, wmma::mem_row_major);
        }
        __syncthreads();
        // the online-softmax step, 16 lanes a row; a row sees a position iff the block's mask has its bit and the position is not after it
#pragma unroll
        for (int rr = 0; rr < QT; ++rr) {
            const int r = rr * 16 + tid / 16, c = tid % 16, rl = r / AP_QH;
            const bool live = r < QR && rl < nr;
            float sv[NPT], mx = -INFINITY;
#pragma unroll
            for (int u = 0; u < NPT; ++u) {
                const int col = u * 16 + c, ent = um[col / QSA_POOL];
                const int q = (ent >> 8) * QSA_POOL + col % QSA_POOL;
                const bool see = live && c0 + col < ctx && (ent >> rl & 1) && q <= pos0 + t0 + rl;
                sv[u] = see ? st[r * SLD + col] * scale : -INFINITY;
                mx = fmaxf(mx, sv[u]);
            }
#pragma unroll
            for (int off = 8; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off, 16));
            const float mo = sm[r], mn = fmaxf(mo, mx);
            float ps = 0.f;
#pragma unroll
            for (int u = 0; u < NPT; ++u) {
                const float pv = sv[u] == -INFINITY ? 0.f : __expf(sv[u] - mn);
                ps += pv;
                if (!LEAN || r < S::QA) ph[r * PLD + u * 16 + c] = __float2half_rn(pv);   // LEAN: rows past QA would be the scores
            }
#pragma unroll
            for (int off = 8; off; off >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, off, 16);
            __syncwarp();
            if (c == 0) {
                const float sc = mo == -INFINITY ? 0.f : __expf(mo - mn);
                ssc[r] = sc;
                sz[r] = sz[r] * sc + ps;
                sm[r] = mn;
            }
        }
        __syncthreads();
        place(vr);
        __syncthreads();
        if (c0 + PC < ctx) gather(kc, c0 + PC, kr);
        if constexpr (REG) {   // lane l of an accumulator tile: column l & 15, rows 8 (l >> 4) + 0..7
            const int lane = tid & 31;
#pragma unroll
            for (int qt = 0; qt < QT; ++qt) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fp[NPT];
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fv;
#pragma unroll
                for (int ks = 0; ks < NPT; ++ks) wmma::load_matrix_sync(fp[ks], ph + qt * 16 * PLD + ks * 16, PLD);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float sc = ssc[qt * 16 + 8 * (lane >> 4) + i];
                    acc[qt][0].v[i] *= sc;
                    acc[qt][1].v[i] *= sc;
                }
#pragma unroll
                for (int ct = 0; ct < 2; ++ct) {
                    const int d0 = warp * 32 + ct * 16;
#pragma unroll
                    for (int ks = 0; ks < (AT & 8 ? 0 : NPT); ++ks) {
                        wmma::load_matrix_sync(fv, kv + ks * 16 * AP_LD + d0, AP_LD);
                        wmma::mma_sync(acc[qt][ct], fp[ks], fv, acc[qt][ct]);
                    }
                }
            }
            continue;
        }
        // P V a query tile at a time: warp w takes O columns [32 w, 32 w + 32)
#pragma unroll
        for (int qt = 0; qt < QT; ++qt) {
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fp[NPT];
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fv;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> fo;
#pragma unroll
                for (int ks = 0; ks < NPT; ++ks) wmma::load_matrix_sync(fp[ks], ph + qt * 16 * PLD + ks * 16, PLD);
#pragma unroll
                for (int ct = 0; ct < 2; ++ct) {
                    const int d0 = warp * 32 + ct * 16;
                    wmma::fill_fragment(fo, 0.f);
#pragma unroll
                    for (int ks = 0; ks < NPT; ++ks) {
                        wmma::load_matrix_sync(fv, kv + ks * 16 * AP_LD + d0, AP_LD);
                        wmma::mma_sync(fo, fp[ks], fv, fo);
                    }
                    wmma::store_matrix_sync(st + d0, fo, AP_LDF, wmma::mem_row_major);
                }
            }
            __syncthreads();
            const float sc = ssc[qt * 16 + orow];
#pragma unroll
            for (int j = 0; j < 16; ++j) o[qt][j] = fmaf(o[qt][j], sc, st[orow * AP_LDF + ocol + 16 * j]);
            __syncthreads();
        }
    }
    if constexpr (REG) {
        const int lane = tid & 31;
#pragma unroll
        for (int qt = 0; qt < QT; ++qt)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int qr = qt * 16 + 8 * (lane >> 4) + i, rl = qr / AP_QH, h = qr % AP_QH;
                if (qr >= QR || rl >= nr) continue;
                float* dst = part + (((size_t) card * C + t0 + rl) * ATT_OWN + h % ATT_OWN) * AP_W;
                dst[warp * 32 + (lane & 15)] = acc[qt][0].v[i];
                dst[warp * 32 + 16 + (lane & 15)] = acc[qt][1].v[i];
            }
        for (int qr = tid; qr < QR; qr += AP_THREADS) {
            const int rl = qr / AP_QH, h = qr % AP_QH;
            if (rl >= nr) continue;
            float* dst = part + (((size_t) card * C + t0 + rl) * ATT_OWN + h % ATT_OWN) * AP_W;
            dst[HD] = sm[qr];
            dst[HD + 1] = sz[qr];
        }
        return;
    }
#pragma unroll
    for (int qt = 0; qt < QT; ++qt) {
        const int qr = qt * 16 + orow, rl = qr / AP_QH, h = qr % AP_QH;
        if (qr >= QR || rl >= nr) continue;
        float* dst = part + (((size_t) card * C + t0 + rl) * ATT_OWN + h % ATT_OWN) * AP_W;
#pragma unroll
        for (int j = 0; j < 16; ++j) dst[ocol + 16 * j] = o[qt][j];
        if (ocol == 0) {
            dst[HD] = sm[qr];
            dst[HD + 1] = sz[qr];
        }
    }
}

#endif  // FNX_HIP && FNX_ATT_HEADS

// this card's heads: its own partial and the others' (from card g + 1 in slot 0, g + 2 in slot 1, ...), merged; the gate
__global__ void __launch_bounds__(HD) k_pf_attn_merge(const float* __restrict__ own, const float* __restrict__ recv,
                                                      const float* __restrict__ qg, float* __restrict__ out, int C, int card) {
    const int t = blockIdx.x, hl = blockIdx.y, i = threadIdx.x;
    const size_t at = ((size_t) t * ATT_OWN + hl) * AP_W;
    float m = own[at + HD], z = own[at + HD + 1], a = own[at + i];
#pragma unroll
    for (int k = 0; k < (ATT_HEADS ? 0 : NCARD - 1); ++k) {   // by heads: nothing to merge (recv is not read)
        const float* src = recv + (size_t) k * C * ATT_OWN * AP_W + at;
        att_merge(m, z, a, src[HD], src[HD + 1], src[i]);
    }
    (void) m;
    const int hg = ATT_HEADS ? hl : card * ATT_OWN + hl;   // qg's head: local by heads
    out[((size_t) t * ATT_OWN + hl) * HD + i] = a / z * sigm(qg[((size_t) t * NQH_LOC + hg) * 2 * HD + HD + i]);
}

}  // namespace
