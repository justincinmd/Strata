// FNX_MM_HC: the hyper-connection read (hc.cuh, build_hc_mix) as a queue of tickets, without grid barriers, on the engine's int8 weights
// with the int8 matrix instruction (mm.cuh). Included by kernels.cu
// after hc.cuh: HcArgs, the read's constants and its helpers are that file's.
//
// The queue q (device memory, zero between launches): q[0] the next ticket, q[1] the down tickets done, q[2] the blocks that left. A block (256 threads,
// 8 waves) takes tickets until none is left: thread 0 takes one with an atomic add, the block runs it. Tickets 0 .. HM_ND - 1 are the down part, the
// next HM_NU the up part; a block that takes an up ticket first issues its up weights' loads, then waits until q[1] says every down ticket is done (one
// lane polls, the block waits at its own barrier), makes lo for itself once (as k_hc's one-barrier form does), and runs up tickets until the end. The
// last block to leave zeroes q for the next launch.
// No block waits for another block to be resident: the down tickets are taken by whatever blocks run, and a block that waits at the gate holds an up
// ticket only, so any grid of 1 block or more ends, and nothing else on the card can stall it (k_hc's barriers need its whole grid resident at once: a
// prefetch kernel beside it holds workgroups and the barrier times out). Every poll has the deadline a.timeout (clock64 cycles): past it the block
// sets bit 4 of err and goes on, so the read fails closed as k_hc's does and still ends, with the queue zeroed.
//
//   down ticket (slice s of 512 columns, group g of 3): R' and xs = R' * w_norm for the slice, as int8 per 32 (as k_hc: the same values); group 0 also the
//       slice's square sums. Wave w: row tile 8 g + w, 16 rows of the down matrix against the slice's 16 blocks, a unit (16 rows x 32 columns, 512
//       bytes) per two matrix instructions, 8 units' loads in flight; in group 2 waves 4..7 take the four inject rows (bf16 on xs itself).
//       part[s][k][row], row r of token k:  p = 0; for b = 0 .. 15 of the slice: p = fma(dw[r][b] * xd[k][b], (float) S_b, p)   (S_b: the exact dot)
//       inject row: each lane 16 columns, p = fma(w, xs, p) in column order, then the wave's 32 lanes added by xor shuffles (16, 8, 4, 2, 1).
//   reduce (each block before its first up ticket): rs and lo as k_hc's one-barrier form (the same expressions), the inject rows by the block of up ticket 0
//   up ticket b (32 columns): wave w: stream w >> 1, 16 of the columns: 16 up rows against lo's 10 blocks:
//       g = 0; for b = 0 .. 9: g = fma(du[row][b] * ld[k][b], (float) S_b, g); then k_hc's last steps unchanged (sigmoid, mix, R <- R', q8_1 output).
// The sums do not depend on the grid, on which block takes which ticket or on timing. They are not k_hc's bits: k_hc converts the down dots every 8
// columns and adds lanes by shuffles; the integers and the scales are the same.
#if FNX_MM_HC && FNX_HIP
namespace {

constexpr int HM_T = 256, HM_W = HM_T / 32;
constexpr int HM_TILES = LR / 16;                          // the down matrix's row tiles
constexpr int HM_NG = (HM_TILES + HC + HM_W - 1) / HM_W;   // row groups of a slice: a wave a tile or an inject row
constexpr int HM_ND = H_NS * HM_NG, HM_NU = N / H_UPC;     // down and up tickets
// generator parameters (gen/families.py hc_mm; the defaults are the kernel in the binary)
#ifndef FNX_TK_HM_UD
#define FNX_TK_HM_UD 8
#endif
#ifndef FNX_TK_HM_POLL
#define FNX_TK_HM_POLL 64
#endif
constexpr int HM_UD = FNX_TK_HM_UD;                        // a down wave's units in flight (16 a ticket)
constexpr int HM_Q = 4;                                    // the queue's words
static_assert(LR % 16 == 0 && H_SL == 2 * HM_T && (H_SL / 32) % HM_UD == 0, "hc_mm: 16-row tiles, two columns a thread, whole rounds of units");
static_assert(HM_UD >= 1 && FNX_TK_HM_POLL >= 1, "FNX_TK_HM_UD, FNX_TK_HM_POLL: at least 1");
static_assert(HM_W * 16 == HC * H_UPC && HM_W % HC == 0, "hc_mm: an up ticket's rows (streams x 32 columns) on 8 waves of 16 rows");
static_assert(HM_NG * HM_W >= HM_TILES + HC && MAXT <= 16, "hc_mm: every tile and inject row has a wave; the window rides in the matrix");

// FNX_HC_SUM: the window's sum across the cards of the output this read applies (a.bo), as the read's FIRST tickets: T * HM_SUMT sum
// tickets, each token's pairs of values 256 at a time, a thread a pair (the SUM type's ticket: HmRingSum in kernels.cu, k_ring_a2a's steps), counted in
// q[3]. A block's first down or up ticket waits at a gate until every sum ticket is done (as the up tickets wait for the down tickets); a block that
// holds a sum ticket waits for no other block, so the rule above stands. The sum's own launch and the gap before the read are gone; its host words still
// go by whole waves. SUM = HmNoSum: no sum tickets, the kernel as it was.
// - The sum tickets come from the queue, first come first served, and must: with 16 rows only 64 of the read's 96 blocks are resident at once (43.5 KB
//   of LDS a block), so a ticket tied to a block number waits for a block that the waiting blocks keep out (tried: gate time-out at the first window).
// - A down ticket loads a.bo with plain loads after the gate. That holds because a sum ticket's part of a.bo is whole cache lines (2048 bytes from a
//   64-byte boundary: hc_read checks the pointer) and no block loads a.bo outside its own sum ticket before the gate, so no compute unit keeps a line
//   that another block's ticket stores later.
// - Measured (4 x R9700, 2026-11-16, sum start to read end on the last card to start, us at 2..7 rows): two launches 22.4 25.0 27.8 31.4 34.4 37.6,
//   this kernel 17.7 20.5 23.7 27.5 30.8 34.8. It saves the launch and the gap (4.7 us at 2 rows) and loses ~0.35 us a row: windows of hc_sum_rows()
//   rows and fewer take it (FNX_HC_SUM: 8 in an APPROX build; the fit of many-user windows breaks even at 9 rows). The row cost is not the queue's atomic adds
//   (tickets by block number were no faster), not the waiting blocks' polls (longer sleeps: the same) and not the grid (64 blocks: the same).
constexpr int HM_SUMT = (N / 2) / HM_T;
static_assert((N / 2) % HM_T == 0, "hc_mm: a token's pairs in whole sum tickets");
struct HmNoSum {
    static constexpr bool on = false;
};
// the sum gate: every sum ticket done (a.bo is the sum from here on). One lane polls with a load (up to 90 blocks wait here, and the queue's line is the
// ticket counter's too); the others wait at the block's barrier. The deadline is a sum ticket's own longest time (`patience` cycles: its waits for the
// peers' words, each inside the ring's time-out) plus the read's: a peer that is late but inside the ring's time-out must not fail the read's gate first. Past it the block sets bit 4 of err and goes on (fails closed).
__device__ __forceinline__ void hm_sum_gate(const HcArgs& a, const unsigned* q, unsigned nsum, long long patience) {
    if (threadIdx.x == 0) {
        constexpr long long top = 0x7fffffffffffffffll;
        const long long ts = clock64(), limit = a.timeout > top - patience ? top : a.timeout + patience;   // (both are 0 or more; no time-out at the top)
        long long t0 = ts;
        while (__ldcg(q + 3) < nsum) {
            if (const long long dt = clock64() - t0; __builtin_expect(dt > limit, 0)) {
                if (diag_wait_on(a.err, dt, limit, ts, t0)) continue;   // the driver had the queues off the cards
                diag_timeout(a.err, DK_HC_GATE, nsum, __ldcg(q + 3), __ldcg(q), DG_NO_SRC, (unsigned) blockIdx.x, dt);
                atomicOr(a.err, 4u);
                break;
            }
            __nanosleep(FNX_TK_HM_POLL);
        }
    }
    __syncthreads();
    __threadfence();   // every wave: what the sum tickets' blocks stored is what it now loads
}

template <int T, typename SUM>
__device__ __forceinline__ void hc_mm_run(const HcArgs& a, unsigned* q, const SUM& sm) {
    static_assert(sizeof(HcUp<T>) <= sizeof(float) * T * H_SL, "the up part's arrays inside xs");
    constexpr unsigned NSUM = SUM::on ? (unsigned) (T * HM_SUMT) : 0u;   // the sum tickets come first
    constexpr int NR = T < 8 ? T : 8;   // token registers in use: token 8 (lane >> 4) + v
    __shared__ __align__(16) float xs[T][H_SL];
    __shared__ __align__(16) int8_t xq[T][H_SL];
    __shared__ float xd[T][H_SL / 32];
    __shared__ float red[T][H_SL / 32];
    __shared__ unsigned tick;
    __shared__ float sink[HM_T];   // where a lane's sums of tokens past the window go (see the up dots)
    HcUp<T>& up = *reinterpret_cast<HcUp<T>*>(&xs[0][0]);   // a block's down tickets all come before its up tickets
    auto& lq = up.lq;
    auto& ld = up.ld;
    auto& rs = up.rs;
    auto& ms = up.ms;
    auto& gs = up.gs;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31, j = lane & 15, h = lane >> 4;
    const int ta = min(j, T - 1);   // the lane's token as an operand
    bool lo_made = false;
#if FNX_MALL_PF   // under the option, so that the build without it has the same kernel code as before (an empty statement here changed it)
    HC_PF_EVENT(a);
#endif
    [[maybe_unused]] bool sums_in = false;
    if constexpr (SUM::on) sm.start();
    for (;;) {
        if (t == 0) tick = atomicAdd(q, 1u);
        __syncthreads();
        const unsigned tk0 = tick;
        if constexpr (SUM::on) {
            if (tk0 < NSUM) {
                // ---------------- sum ticket: 256 pairs of one token
                sm.template ticket<T>((int) (tk0 / HM_SUMT), (int) (tk0 % HM_SUMT));
                __threadfence();   // the ticket's stores are out before it counts as done
                __syncthreads();
                if (t == 0) atomicAdd(q + 3, 1u);
                continue;
            }
        }
        const unsigned tk = tk0 - NSUM;
        if (tk >= (unsigned) (HM_ND + HM_NU)) break;
        if (tk < (unsigned) HM_ND) {
            if constexpr (SUM::on) {
                if (!sums_in) {
                    sums_in = true;
                    hm_sum_gate(a, q, NSUM, sm.patience());
                }
            }
            // ---------------- down ticket
            const int s = (int) tk / HM_NG, grp = (int) tk % HM_NG;
            const int col0 = s * H_SL, c = col0 / N, i0 = col0 - c * N;
            // thread t takes columns t and t + 256 of every token (the loads side by side)
#pragma unroll
            for (int p = 0; p < H_SL / HM_T; ++p) {
                const int col = t + p * HM_T;
                const float wn = a.w_norm[col0 + col];
                float r[T];
#pragma unroll
                for (int k = 0; k < T; ++k) r[k] = rprime(a, k, c, i0 + col);
#pragma unroll
                for (int k = 0; k < T; ++k) xs[k][col] = r[k] * wn;
                if (grp == 0) {
#pragma unroll
                    for (int k = 0; k < T; ++k) {
                        const float v = wsum(r[k] * r[k]);
                        if (lane == 0) red[k][warp + p * HM_W] = v;
                    }
                }
            }
            __syncthreads();
            if (grp == 0 && t < T) {
                float v = 0.f;
                for (int w = 0; w < H_SL / 32; ++w) v += red[t][w];
                a.ssp[(size_t) t * H_NS + s] = v;
            }
            for (int it = warp; it < T * (H_SL / 32); it += HM_W) {
                const int k = it / (H_SL / 32), blk = it % (H_SL / 32);
                q8_block(xs[k][blk * 32 + lane], &xq[k][blk * 32], &xd[k][blk]);
            }
            __syncthreads();
            const int rt = grp * HM_W + warp;   // the wave's row tile, or past the tiles its inject row
            if (rt < HM_TILES) {
                const int row = rt * 16 + j;
                const int8_t* const wq = a.w_down.q + (size_t) row * D + col0 + 16 * h;
                const uint16_t* const wd = a.w_down.d + ((size_t) row * D + col0) / 32;
                float acc[NR];
#pragma unroll
                for (int v = 0; v < NR; ++v) acc[v] = 0.f;
#pragma unroll 1
                for (int b0 = 0; b0 < H_SL / 32; b0 += HM_UD) {
                    uint4 w[HM_UD];
                    float dw[HM_UD];
#pragma unroll
                    for (int u = 0; u < HM_UD; ++u) {
                        w[u] = FNX_WLD2(reinterpret_cast<const uint4*>(wq + (b0 + u) * 32));
                        dw[u] = f16(FNX_WLD2(wd + b0 + u));
                    }
#pragma unroll
                    for (int u = 0; u < HM_UD; ++u) {
                        const int b = b0 + u;
                        const fnx_mm::v8i cs = fnx_mm::unit(w[u], *reinterpret_cast<const uint4*>(&xq[ta][b * 32 + 16 * h]));
#pragma unroll
                        for (int v = 0; v < NR; ++v) acc[v] = fmaf(dw[u] * xd[min(8 * h + v, T - 1)][b], (float) cs[v], acc[v]);
                        fnx_mm::seq();
                    }
                }
#pragma unroll
                for (int v = 0; v < NR; ++v)
                    if (8 * h + v < T) a.part[((size_t) s * T + 8 * h + v) * H_ROWS + row] = acc[v];
            } else if (a.w_inj && rt < HM_TILES + HC) {
                const int row = LR + rt - HM_TILES;
                const uint4* const w4 = reinterpret_cast<const uint4*>(a.w_inj + (size_t) (row - LR) * D + col0 + lane * 16);
                const uint4 u0 = FNX_WLD2(w4), u1 = FNX_WLD2(w4 + 1);
                const uint16_t* const h0 = reinterpret_cast<const uint16_t*>(&u0);
                const uint16_t* const h1 = reinterpret_cast<const uint16_t*>(&u1);
#pragma unroll 1
                for (int k = 0; k < T; ++k) {
                    const float* const x = &xs[k][lane * 16];
                    float acc = 0.f;
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc = fmaf(bf(h0[i]), x[i], acc);
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc = fmaf(bf(h1[i]), x[8 + i], acc);
                    acc = wsum(acc);
                    if (lane == 0) a.part[((size_t) s * T + k) * H_ROWS + row] = acc;
                }
            }
            // the ticket's stores are out before it counts as done
            __threadfence();
            __syncthreads();
            if (t == 0) atomicAdd(q + 1, 1u);
        } else {
            // ---------------- up ticket: 32 columns b, wave w stream w / 2 and 16 of the columns; the weights' loads first (in flight through the gate)
            const int b = (int) tk - HM_ND;
            const int wc = warp / (HM_W / HC), ucol = (warp % (HM_W / HC)) * 16 + j;
            const size_t urow = (size_t) wc * N + b * H_UPC + ucol;
            uint4 wu[H_UB];
            float du[H_UB];
#pragma unroll
            for (int ub = 0; ub < H_UB; ++ub) {
                wu[ub] = FNX_WLD2(reinterpret_cast<const uint4*>(a.w_up.q + urow * LR + ub * 32 + 16 * h));
                du[ub] = f16(FNX_WLD2(a.w_up.d + urow * (LR / 32) + ub));
            }
            if (!lo_made) {
                lo_made = true;
                if constexpr (SUM::on) {
                    if (!sums_in) {   // a block whose first ticket is an up ticket: the sums' wait is not on the down gate's clock
                        sums_in = true;
                        hm_sum_gate(a, q, NSUM, sm.patience());
                    }
                }
                // the gate: every down ticket done (their stores are out: each counts after its fence). One lane polls; the deadline fails closed
                if (t == 0) {
                    const long long ts = clock64();
                    long long t0 = ts;
                    while (atomicAdd(q + 1, 0u) < (unsigned) HM_ND) {
                        if (const long long dt = clock64() - t0; __builtin_expect(dt > a.timeout, 0)) {
                            if (diag_wait_on(a.err, dt, a.timeout, ts, t0)) continue;   // the driver had the queues off the cards
                            diag_timeout(a.err, DK_HC_GATE, (unsigned) HM_ND, atomicAdd(q + 1, 0u), atomicAdd(q, 0u), DG_NO_SRC, (unsigned) blockIdx.x, dt);
                            atomicOr(a.err, 4u);
                            break;
                        }
                        __nanosleep(FNX_TK_HM_POLL);
                    }
                }
                __syncthreads();
                __threadfence();   // every wave: what the down tickets' blocks stored is what it now loads (k_hc's barrier ends with one too)
                // rs, then lo per (token k, lo block) a wave into lq / ld; the block of up ticket 0 also the inject rows
                if (t < T * HC) rs[t / HC][t % HC] = rs_of(a, t / HC, t % HC);
                __syncthreads();
                for (int it = warp; it < T * (LR / 32); it += HM_W) {
                    const int k = it / (LR / 32), r = (it % (LR / 32)) * 32 + lane;
                    float pv[H_NS];
#pragma unroll
                    for (int n = 0; n < H_NS; ++n) pv[n] = __ldcg(a.part + ((size_t) n * T + k) * H_ROWS + r);
                    float sum = 0.f;
#pragma unroll
                    for (int cc = 0; cc < HC; ++cc) {
                        float p = 0.f;
#pragma unroll
                        for (int ss = 0; ss < H_SPS; ++ss) p += pv[cc * H_SPS + ss];
                        sum = fmaf(rs[k][cc], p, sum);
                    }
                    q8_block(silu_(sum / (float) HC), &lq[k][r & ~31], &ld[k][r / 32]);
                }
                if (b == 0 && a.w_inj && lane < HC) {
                    for (int k = warp; k < T; k += HM_W) {
                        const int r = LR + lane;
                        float pv[H_NS];
#pragma unroll
                        for (int n = 0; n < H_NS; ++n) pv[n] = __ldcg(a.part + ((size_t) n * T + k) * H_ROWS + r);
                        float sum = 0.f;
#pragma unroll
                        for (int cc = 0; cc < HC; ++cc) {
                            float p = 0.f;
#pragma unroll
                            for (int ss = 0; ss < H_SPS; ++ss) p += pv[cc * H_SPS + ss];
                            sum = fmaf(rs[k][cc], p, sum);
                        }
                        a.inject[k * HC + lane] = sum;
                    }
                }
                __syncthreads();
            }
            {
                float acc[NR];
#pragma unroll
                for (int v = 0; v < NR; ++v) acc[v] = 0.f;
#pragma unroll
                for (int ub = 0; ub < H_UB; ++ub) {
                    const fnx_mm::v8i cs = fnx_mm::unit(wu[ub], *reinterpret_cast<const uint4*>(&lq[ta][ub * 32 + 16 * h]));
#pragma unroll
                    for (int v = 0; v < NR; ++v) acc[v] = fmaf(du[ub] * ld[min(8 * h + v, T - 1)][ub], (float) cs[v], acc[v]);
                    fnx_mm::seq();
                }
                // every lane stores every register, a token past the window into the sink: with the stores under a condition the compiler moves the
                // whole pass (pure code with one use) into the condition's branch, past the barriers between its units, and runs the units side by side
#pragma unroll
                for (int v = 0; v < NR; ++v) *(8 * h + v < T ? &gs[min(8 * h + v, T - 1)][wc][ucol] : &sink[t]) = acc[v];
            }
            __syncthreads();
            // item (k, c, col): this ticket's 32 columns of stream c, row k (k_hc's steps)
            for (int e = t; e < T * HC * H_UPC; e += HM_T) {
                const int col = e % H_UPC, cc = (e / H_UPC) % HC, k = e / (H_UPC * HC), dc = b * H_UPC + col;
                const float rv = rprime(a, k, cc, dc);
                if (a.bo) a.R[(size_t) k * D + cc * N + dc] = rv;
                gs[k][cc][col] = sigm(gs[k][cc][col]) * (rv * a.w_norm[cc * N + dc] * rs[k][cc]);
            }
            __syncthreads();
            for (int e = t; e < T * H_UPC; e += HM_T) {
                const int col = e % H_UPC, k = e / H_UPC;
                const float sum = (gs[k][0][col] + gs[k][1][col] + gs[k][2][col] + gs[k][3][col]) / (float) HC;
                a.mixed[(size_t) k * N + b * H_UPC + col] = sum;
                ms[k][col] = sum;
            }
            __syncthreads();
            if (a.q8) {
                for (int k = warp; k < T; k += HM_W) {
                    const float xi = ms[k][lane];
                    const float amax = wmax(fabsf(xi)), sm = wsum(xi);
                    const float dq = amax / 127.0f;
                    block_q8_1& yq = a.q8[(size_t) k * (N / 32) + b];
                    const int8_t qv = amax == 0.0f ? 0 : (int8_t) roundf(xi / dq);
                    yq.qs[lane] = qv;
                    if (lane == 0) yq.ds = make_half2(__float2half(dq), __float2half(sm));
                    if (a.tq) {
                        a.tq[(size_t) k * N + b * 32 + lane] = qv;
                        if (lane == 0) a.td[(size_t) k * (N / 32) + b] = __half2float(__float2half(dq));
                    }
                }
            }   // (the next ticket's barrier comes before anything writes ms again)
        }
    }
    // the last block to leave: the queue back to zero for the next launch (every other block is past its last use of q)
    if (t == 0 && atomicAdd(q + 2, 1u) + 1u == gridDim.x) {
        q[0] = 0;
        q[1] = 0;
        q[2] = 0;
        if constexpr (SUM::on) q[3] = 0;
        __threadfence();
    }
}

template <int T>
__global__ void __launch_bounds__(HM_T) k_hc_mm(HcArgs a, unsigned* q) {
    hc_mm_run<T, HmNoSum>(a, q, HmNoSum{});
}

}  // namespace
#endif
