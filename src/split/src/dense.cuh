// Dense Q8_0 matrix x up to MAXT q8_1 rows, from the split layout (int8 rows, f16 scale per 32) (included by
// kernels.cu). Each lane takes 16 columns per step (one 16-byte load per row), each warp Q8S_R rows, so a lane's
// activation values are loaded once per step for all its rows. Fewer rows per warp for small matrices: more blocks.
namespace {

constexpr int Q8S_THREADS = 256, Q8S_SEGS = 3;

// up to Q8S_SEGS matrices over the same activations in one launch, their rows one after another: segment s holds rows
// [end[s - 1], end[s]) of the launch, its own weights and output (row r of the segment at out[s][t * stride[s] + r])
struct Q8Segs {
    const int8_t* q[Q8S_SEGS];
    const uint16_t* d[Q8S_SEGS];
    float* out[Q8S_SEGS];
    int end[Q8S_SEGS];
    int stride[Q8S_SEGS];
};
__device__ __forceinline__ int q8s_seg(const Q8Segs& g, int row) { return row < g.end[0] ? 0 : row < g.end[1] ? 1 : 2; }
// a segment's field by selects: indexing the parameter arrays by a run-time s puts them in local memory (q8s 60% slower)
template <typename F> __device__ __forceinline__ F q8s_pick(int s, F a, F b, F c) { return s == 0 ? a : s == 1 ? b : c; }

// SEG: the rows' segments looked up (q8s_gemv_segs); else one matrix, g's first segment, its row pointers made in the
// loops as before (held per row in registers they cost the 16-row path ~30-80%)
// block bx of the launch (a kernel of its own, or part of a launch shared with other work: k_q8s_ab)
template <int T, int Q8S_R, bool SPLIT, bool SEG>
__device__ __forceinline__ void q8s_block(const Q8Segs& g, int rows, int cols, const block_q8_1* __restrict__ y,
                                          const int8_t* __restrict__ tq, const float* __restrict__ td, int bx) {
    constexpr int Q8S_ROWS = Q8S_THREADS / 32 * Q8S_R;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = bx * Q8S_ROWS + warp * Q8S_R, nb = cols / 32;
    // each of the warp's rows: its segment's weights at the row's place in it
    const int8_t* qr[Q8S_R];
    const uint16_t* dr[Q8S_R];
    if constexpr (SEG) {
#pragma unroll
        for (int r = 0; r < Q8S_R; ++r) {
            const int row = min(row0 + r, rows - 1), s = q8s_seg(g, row), lr = row - q8s_pick(s, 0, g.end[0], g.end[1]);
            qr[r] = q8s_pick(s, g.q[0], g.q[1], g.q[2]) + (size_t) lr * cols;
            dr[r] = q8s_pick(s, g.d[0], g.d[1], g.d[2]) + (size_t) lr * nb;
        }
    }
    auto wq = [&](int r) { return SEG ? qr[r] : g.q[0] + (size_t) min(row0 + r, rows - 1) * cols; };
    auto wd = [&](int r) { return SEG ? dr[r] : g.d[0] + (size_t) min(row0 + r, rows - 1) * nb; };
    float acc[Q8S_R][T];
#pragma unroll
    for (int r = 0; r < Q8S_R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) acc[r][t] = 0.f;
    if constexpr ((T > 8 && Q8S_R >= 2) || (Q8S_R >= 4 && T > 4)) {
        // the step's weights for the warp's rows held, the activations TC rows at a time against them: all T rows'
        // held took up to ~160 registers (16 rows, 4 a warp: one block an SM, 12% occupancy). kbench, 16 rows: the
        // head 722 -> 495 us, qkvz 60 -> 48; at 5-8 rows it pays only with 4 rows a warp, and not at all with 1. Each
        // sum takes the same steps in the same order
        constexpr int TC = 4;
#pragma unroll 1
        for (int c = lane * 16; c < cols; c += 512) {
            const int b = c >> 5, half = (c >> 4) & 1;
            int4 w[Q8S_R];
            float dw[Q8S_R];
#pragma unroll
            for (int r = 0; r < Q8S_R; ++r) {
                w[r] = FNX_WLD2(reinterpret_cast<const int4*>(wq(r) + c));
                dw[r] = __half2float(__ushort_as_half(FNX_WLD2(wd(r) + b)));
            }
#pragma unroll
            for (int t0 = 0; t0 < T; t0 += TC) {
                int a[TC][4];
                float da[TC];
#pragma unroll
                for (int tt = 0; tt < TC; ++tt) {
                    const int t = min(t0 + tt, T - 1);
                    if constexpr (SPLIT) {
                        const int4 v = *reinterpret_cast<const int4*>(tq + (size_t) t * cols + c);
                        a[tt][0] = v.x;
                        a[tt][1] = v.y;
                        a[tt][2] = v.z;
                        a[tt][3] = v.w;
                        da[tt] = td[(size_t) t * nb + b];
                    } else {
                        const block_q8_1* yb = y + (size_t) t * nb + b;
                        const int* qs = reinterpret_cast<const int*>(yb->qs) + half * 4;
#pragma unroll
                        for (int k = 0; k < 4; ++k) a[tt][k] = qs[k];
                        da[tt] = __low2float(yb->ds);
                    }
                }
#pragma unroll
                for (int r = 0; r < Q8S_R; ++r) {
                    if (row0 + r >= rows) break;
#pragma unroll
                    for (int tt = 0; tt < TC; ++tt) {
                        if (t0 + tt >= T) break;
                        int s = __dp4a(w[r].x, a[tt][0], 0);
                        s = __dp4a(w[r].y, a[tt][1], s);
                        s = __dp4a(w[r].z, a[tt][2], s);
                        s = __dp4a(w[r].w, a[tt][3], s);
                        acc[r][t0 + tt] = fmaf(dw[r] * da[tt], (float) s, acc[r][t0 + tt]);
                    }
                }
            }
        }
    } else {
#pragma unroll 2
    for (int c = lane * 16; c < cols; c += 512) {
        const int b = c >> 5, half = (c >> 4) & 1;
        int a[T][4];
        float da[T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            if constexpr (SPLIT) {
                // the twin: 16 values in one load, the block's scale as a float
                const int4 v = *reinterpret_cast<const int4*>(tq + (size_t) t * cols + c);
                a[t][0] = v.x;
                a[t][1] = v.y;
                a[t][2] = v.z;
                a[t][3] = v.w;
                da[t] = td[(size_t) t * nb + b];
            } else {
                const block_q8_1* yb = y + (size_t) t * nb + b;
                const int* qs = reinterpret_cast<const int*>(yb->qs) + half * 4;
#pragma unroll
                for (int k = 0; k < 4; ++k) a[t][k] = qs[k];
                da[t] = __low2float(yb->ds);
            }
        }
#pragma unroll
        for (int r = 0; r < Q8S_R; ++r) {
            if (row0 + r >= rows) break;
            const int4 w = FNX_WLD2(reinterpret_cast<const int4*>(wq(r) + c));
            const float dw = __half2float(__ushort_as_half(FNX_WLD2(wd(r) + b)));
#pragma unroll
            for (int t = 0; t < T; ++t) {
                int s = __dp4a(w.x, a[t][0], 0);
                s = __dp4a(w.y, a[t][1], s);
                s = __dp4a(w.z, a[t][2], s);
                s = __dp4a(w.w, a[t][3], s);
                acc[r][t] = fmaf(dw * da[t], (float) s, acc[r][t]);
            }
        }
    }
    }
#pragma unroll
    for (int r = 0; r < Q8S_R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && row0 + r < rows) {
                const int row = row0 + r, s = SEG ? q8s_seg(g, row) : 0;
                q8s_pick(s, g.out[0], g.out[1], g.out[2])[(size_t) t * q8s_pick(s, g.stride[0], g.stride[1], g.stride[2]) + row -
                                                          q8s_pick(s, 0, g.end[0], g.end[1])] = v;
            }
        }
}

template <int T, int Q8S_R, bool SPLIT, bool SEG>
__global__ void __launch_bounds__(Q8S_THREADS) k_q8s(const Q8Segs g, int rows, int cols, const block_q8_1* __restrict__ y,
                                                    const int8_t* __restrict__ tq, const float* __restrict__ td) {
    q8s_block<T, Q8S_R, SPLIT, SEG>(g, rows, cols, y, tq, td, blockIdx.x);
}

#if FNX_MM && FNX_HIP
// ---------------------------------------------------------------- the same product on the int8 matrix instruction (mm.cuh)
// A block is 16 weight rows (lane l: row l & 15 of them, in both operands' halves) and Q8S_THREADS / 32 = 8 waves, each a share of the columns: wave w
// the 32-column blocks w, w + 8, w + 16, ... of the rows. A unit (16 rows x 32 columns x up to 16 tokens) is two instructions whatever the window's size:
// the window rides in the matrix (tokens 0..7 end in lanes 0..15, tokens 8..15 in lanes 16..31; a window of 8 rows or fewer leaves the upper lanes' sums
// unread). Per unit and token register: one convert, one product of the scales, one fma. The weights' loads of MM_U units are issued together (the
// stream's loads: non-temporal at FNX_MALL_NT 2), then the units run.
// The sum of (row r, token t), in this order whatever the grid or the timing:
//   wave w:  p_w = 0; for b = w, w + 8, ... (ascending): p_w = fma(dw[r][b] * da[t][b], (float) S_b, p_w), S_b the block's exact integer dot
//   out = ((p_0 + p_1) + (p_2 + p_3)) + ((p_4 + p_5) + (p_6 + p_7))
// The dp4a kernels (q8s_block) convert every 16 columns and add in another order: the two differ in the last bits (same integers, same scales).
constexpr int MM_W = Q8S_THREADS / 32, MM_U = 6;

template <int T, bool SPLIT, bool SEG>
__device__ __forceinline__ void q8s_mm_block(const Q8Segs& g, int rows, int cols, const block_q8_1* __restrict__ y, const int8_t* __restrict__ tq,
                                             const float* __restrict__ td, int bx) {
    static_assert(T >= 1 && T <= 16, "the window rides in the matrix: 16 tokens at most");
    constexpr int NR = T < 8 ? T : 8;   // token registers in use: token 8 (lane >> 4) + v
    __shared__ float red[MM_W][T][16];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, j = lane & 15, h = lane >> 4, nb = cols / 32;
    // the lane's weight row (past the last row: the last one again, not stored) and its token as an operand
    const int row = min(bx * 16 + j, rows - 1), s = SEG ? q8s_seg(g, row) : 0, lr = row - (SEG ? q8s_pick(s, 0, g.end[0], g.end[1]) : 0);
    const int8_t* const wq = (SEG ? q8s_pick(s, g.q[0], g.q[1], g.q[2]) : g.q[0]) + (size_t) lr * cols + 16 * h;
    const uint16_t* const wd = (SEG ? q8s_pick(s, g.d[0], g.d[1], g.d[2]) : g.d[0]) + (size_t) lr * nb;
    const int ta = min(j, T - 1);
    float acc[NR];
#pragma unroll
    for (int v = 0; v < NR; ++v) acc[v] = 0.f;
#pragma unroll 1
    for (int b0 = warp; b0 < nb; b0 += MM_W * MM_U) {
        uint4 w[MM_U];
        float dw[MM_U];
#pragma unroll
        for (int u = 0; u < MM_U; ++u) {
            const int b = b0 + u * MM_W;
            if (b < nb) {   // (the same for the whole wave)
                w[u] = FNX_WLD2(reinterpret_cast<const uint4*>(wq + (size_t) b * 32));
                dw[u] = f16(FNX_WLD2(wd + b));
            }
        }
#pragma unroll
        for (int u = 0; u < MM_U; ++u) {
            const int b = b0 + u * MM_W;
            if (b < nb) {
                uint4 av;
                float da[NR];
                if constexpr (SPLIT) {
                    av = *reinterpret_cast<const uint4*>(tq + (size_t) ta * cols + b * 32 + 16 * h);
#pragma unroll
                    for (int v = 0; v < NR; ++v) da[v] = td[(size_t) min(8 * h + v, T - 1) * nb + b];
                } else {
                    const unsigned* qs = reinterpret_cast<const unsigned*>(y[(size_t) ta * nb + b].qs) + 4 * h;
                    av.x = qs[0], av.y = qs[1], av.z = qs[2], av.w = qs[3];
#pragma unroll
                    for (int v = 0; v < NR; ++v) da[v] = __low2float(y[(size_t) min(8 * h + v, T - 1) * nb + b].ds);
                }
                const fnx_mm::v8i c = fnx_mm::unit(w[u], av);
#pragma unroll
                for (int v = 0; v < NR; ++v) acc[v] = fmaf(dw[u] * da[v], (float) c[v], acc[v]);
                fnx_mm::seq();
            }
        }
    }
#pragma unroll
    for (int v = 0; v < NR; ++v)
        if (8 * h + v < T) red[warp][8 * h + v][j] = acc[v];
    __syncthreads();
    // thread e: token e >> 4, row e & 15 of the block (a store per value, 16 rows side by side)
    const int t = threadIdx.x >> 4, orow = bx * 16 + (threadIdx.x & 15);
    if (t < T && orow < rows) {
        const int jj = threadIdx.x & 15;
        const float v = ((red[0][t][jj] + red[1][t][jj]) + (red[2][t][jj] + red[3][t][jj])) + ((red[4][t][jj] + red[5][t][jj]) + (red[6][t][jj] + red[7][t][jj]));
        const int so = SEG ? q8s_seg(g, orow) : 0;
        q8s_pick(so, g.out[0], g.out[1], g.out[2])[(size_t) t * q8s_pick(so, g.stride[0], g.stride[1], g.stride[2]) + orow -
                                                   q8s_pick(so, 0, g.end[0], g.end[1])] = v;
    }
}
static_assert(MM_W == 8 && Q8S_THREADS == 256, "q8s_mm_block: 8 waves' partial sums, a thread per (token, row)");

template <int T, bool SPLIT, bool SEG>
__global__ void __launch_bounds__(Q8S_THREADS) k_q8s_mm(const Q8Segs g, int rows, int cols, const block_q8_1* __restrict__ y,
                                                       const int8_t* __restrict__ tq, const float* __restrict__ td) {
    q8s_mm_block<T, SPLIT, SEG>(g, rows, cols, y, tq, td, blockIdx.x);
}
#endif

// The shared expert inside the routed experts' launches (no second stream): its gate|up as extra blocks of the router's
// launch, its swiglu and down as an extra row of blocks of the routed down's. Both matrices have fewer than 4096 rows:
// q8s with one row a warp, as q8s_gemv runs them alone. Its FF slice is SH_MAXF at most (model.hpp).
struct ShexpUp {               // gate|up: rows 2 * ffs over the hc read's q8_1 rows y (and its split twin tq / td)
    Q8Segs w;
    int rows = 0, cols = 0, blocks = 0;   // blocks: the launch's last `blocks` blocks (0: none)
    const block_q8_1* y = nullptr;
    const int8_t* tq = nullptr;
    const float* td = nullptr;
};
struct ShexpDown {             // swiglu of gu ([T][2 * ffs]: gate, then up), as int8 per 32, then down (rows N) into w's out
    Q8Segs w;
    int rows = 0, ffs = 0;     // rows 0: none
    const float* gu = nullptr;
};
template <int T> __device__ __forceinline__ void shexp_up_block(const ShexpUp& s, int bx) {
    if (s.tq) q8s_block<T, 1, true, false>(s.w, s.rows, s.cols, s.y, s.tq, s.td, bx);
    else q8s_block<T, 1, false, false>(s.w, s.rows, s.cols, s.y, nullptr, nullptr, bx);
}
// this block's share (q8s blocks bx0, bx0 + nbx, ...) of the down rows, after the swiglu made in shared memory as
// k_swiglu_stacked makes its split twin
template <int T> __device__ void shexp_down_blocks(const ShexpDown& s, int bx0, int nbx) {
    __shared__ __align__(16) int8_t tq[T * SH_MAXF];
    __shared__ float td[T * (SH_MAXF / 32)];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nb = s.ffs / 32;
    for (int it = warp; it < T * nb; it += blockDim.x / 32) {
        const int t = it / nb, b = it % nb;
        const float* g = s.gu + (size_t) t * 2 * s.ffs;
        const float xi = silu_(g[b * 32 + lane]) * g[s.ffs + b * 32 + lane];
        const float amax = wmax(fabsf(xi)), d = amax / 127.0f;
        tq[t * s.ffs + b * 32 + lane] = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
        if (lane == 0) td[t * nb + b] = __half2float(__float2half(d));
    }
    __syncthreads();
    const int nqb = (s.rows + Q8S_THREADS / 32 - 1) / (Q8S_THREADS / 32);
    for (int bx = bx0; bx < nqb; bx += nbx) q8s_block<T, 1, true, false>(s.w, s.rows, s.ffs, nullptr, tq, td, bx);
}

}  // namespace
