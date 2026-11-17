// tools/emu/attn_mm_emu.cpp: the FNX_ATT_MM kernels (src/attn_mm.cuh) run on the host, lane by lane (cuda_runtime.h here), against the
// kernels they replace and against a double-precision reference. No card. It checks the logic: the lists of the two selections must be equal, the scores
// equal bits, the attention equal within rounding. Build and run: tools/emu/build.sh /tmp/attn_mm_emu && /tmp/attn_mm_emu   (exit 0: every check passed)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "kernels.hpp"

namespace fnx {
namespace {
// the engine's q8_1 block (ggml's layout, strata-quant-common.h): the block's scale and its sum term as halves, then 32 codes
struct block_q8_1 {
    __half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(block_q8_1) == 36, "a q8_1 block is 9 words");
#include "emu_gen.inc"
#include "attn_mm.cuh"

std::mt19937_64 rng(20261114);
int fails = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    fails += !ok;
    std::fflush(stdout);
}

// one layer's slot of every page: int8 K and V rows with f16 scales, f16 pooled keys; a shuffled page table (as kbench's kv_pool)
struct Pool {
    std::vector<uint16_t> mem;
    std::vector<int> pt;
    KvPages g;
    size_t rows_h = 0;
    const __half* kc() const { return reinterpret_cast<const __half*>(mem.data()); }
    const __half* vc() const { return reinterpret_cast<const __half*>(mem.data() + rows_h); }
    const __half* pooled() const { return reinterpret_cast<const __half*>(mem.data() + 2 * rows_h); }
};
Pool make_pool(int pages, int spare = 3) {
    Pool p;
    const int rows = kvp_rows(0), phys = pages + spare;
    p.rows_h = kvp_rows_halves(0, true);
    const size_t slot_h = kvp_slot(0, true), codes = (size_t) rows * KV_W, scales = (size_t) rows * KV_W / KV8_GROUP;
    p.mem.resize((size_t) phys * slot_h);
    std::uniform_real_distribution<float> us(0.004f, 0.06f), uk(-1.f, 1.f);
    for (int pg = 0; pg < phys; ++pg) {
        uint16_t* s = p.mem.data() + (size_t) pg * slot_h;
        for (int kv = 0; kv < 2; ++kv) {
            int8_t* c = reinterpret_cast<int8_t*>(s + kv * p.rows_h);
            for (size_t i = 0; i < codes; ++i) c[i] = (int8_t) ((int) (rng() % 255) - 127);
            __half* sc = reinterpret_cast<__half*>(s + kv * p.rows_h + codes / 2);
            for (size_t i = 0; i < scales; ++i) sc[i] = (__half) (rng() % 61 == 0 ? 0.f : us(rng));   // a zero scale now and then
        }
        __half* pl = reinterpret_cast<__half*>(s + 2 * p.rows_h);
        for (size_t i = 0; i < (size_t) KVP_BLOCKS * 128; ++i) pl[i] = (__half) uk(rng);
    }
    std::vector<int> all((size_t) phys);
    for (int i = 0; i < phys; ++i) all[(size_t) i] = i;
    std::shuffle(all.begin(), all.end(), rng);
    p.pt.assign(all.begin(), all.begin() + pages);
    p.g.pt = p.pt.data();
    p.g.page = slot_h;
    p.g.rows = rows;
    p.g.q8 = 1;
    return p;
}

// ---- the selection
struct Sel {
    std::vector<int> cells, n;
};
Sel select(bool mm, const std::vector<float>& scores, int max_pools, const int* pos0, int T, LaneRows lr = {}) {
    Sel s;
    s.cells.assign((size_t) T * QSA_MAXSEL, -7);
    s.n.assign((size_t) T, -7);
    if (mm) emu::launch(dim3((unsigned) T), QS_NT, [&] { k_idx_select_mm(scores.data(), s.cells.data(), s.n.data(), max_pools, pos0, lr); });
    else emu::launch(dim3((unsigned) T), QSA_SEL_THREADS, [&] { k_idx_select(scores.data(), s.cells.data(), s.n.data(), max_pools, pos0, 0, lr); });
    return s;
}
// the list by definition: the 512 highest scores, the lower block on a tie, ascending, then the tail
Sel select_ref(const std::vector<float>& scores, int max_pools, int pos0, int T) {
    Sel s;
    s.cells.assign((size_t) T * QSA_MAXSEL, -7);
    s.n.assign((size_t) T, -7);
    for (int t = 0; t < T; ++t) {
        const int p = pos0 + t, nv = (p + 1) / 4, n_tail = (p + 1) % 4;
        if (nv <= QSA_TOP) {
            s.n[(size_t) t] = -1;
            continue;
        }
        std::vector<int> idx((size_t) nv);
        for (int j = 0; j < nv; ++j) idx[(size_t) j] = j;
        const float* sc = scores.data() + (size_t) t * max_pools;
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return __float_as_uint(sc[a]) > __float_as_uint(sc[b]); });
        idx.resize(QSA_TOP);
        std::sort(idx.begin(), idx.end());
        int n = 0;
        int* row = s.cells.data() + (size_t) t * QSA_MAXSEL;
        for (int j : idx)
            for (int m = 0; m < 4; ++m) row[n++] = j * 4 + m;
        for (int m = 0; m < n_tail; ++m) row[n++] = (p / 4) * 4 + m;
        s.n[(size_t) t] = n;
    }
    return s;
}
bool same(const Sel& a, const Sel& b, int T) {
    for (int t = 0; t < T; ++t) {
        if (a.n[(size_t) t] != b.n[(size_t) t]) return false;
        for (int i = 0; i < a.n[(size_t) t]; ++i)
            if (a.cells[(size_t) t * QSA_MAXSEL + i] != b.cells[(size_t) t * QSA_MAXSEL + i]) return false;
    }
    return true;
}
void test_select(bool quick) {
    const char* names[] = {"uniform", "64 values", "all equal", "half zero", "ascending", "descending", "high at 8k", "high at 8k+1", "all zero", "two values"};
    const int T = 5, mp = 65536 + 16;
    std::uniform_real_distribution<float> u(0.f, 1.f);
    for (int nvs : quick ? std::vector<int>{513, 4097, 9000} : std::vector<int>{513, 600, 2047, 4096, 4097, 9000, 27250, 65536}) {
        int pos0 = 4 * nvs - 1;
        for (int dist = 0; dist < 10; ++dist) {
            std::vector<float> h((size_t) T * mp, 0.f);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < nvs + 2; ++j) {
                    float v = 0.f;
                    switch (dist) {
                        case 0: v = 4.f * u(rng); break;
                        case 1: v = 0.125f * (float) (rng() % 64); break;
                        case 2: v = 1.f; break;
                        case 3: v = rng() % 2 ? 0.f : u(rng); break;
                        case 4: v = 1e-3f * (float) j; break;
                        case 5: v = 1e-3f * (float) (nvs + 2 - j); break;
                        case 6: v = j % 8 == 0 ? 10.f + u(rng) : u(rng); break;
                        case 7: v = j % 8 == 1 ? 10.f + u(rng) : u(rng); break;
                        case 8: v = 0.f; break;
                        default: v = rng() % 100 < 2 ? 2.f : 1.f; break;
                    }
                    h[(size_t) t * mp + j] = v;
                }
            const Sel ref = select_ref(h, mp, pos0, T), b = select(true, h, mp, &pos0, T);
            bool ok = same(ref, b, T);
            if (nvs <= 9000 || dist < 2) ok = ok && same(ref, select(false, h, mp, &pos0, T), T);   // the old kernel too (slow to emulate at depth)
            check(ok, "select: " + std::to_string(nvs) + " blocks, " + names[dist]);
        }
    }
    // a batched window: rows of their own positions (one dense)
    {
        const int Tl = 4;
        int posv[Tl] = {4 * 3000 + 1, 100, 4 * 700 + 3, 4 * 20000};
        const int* pp[Tl] = {&posv[0], &posv[1], &posv[2], &posv[3]};
        const int* ptp[Tl] = {nullptr, nullptr, nullptr, nullptr};
        LaneRows lr;
        lr.pos = pp;
        lr.pt = ptp;
        std::vector<float> h((size_t) Tl * mp);
        for (float& v : h) v = 3.f * u(rng);
        const Sel a = select(false, h, mp, nullptr, Tl, lr), b = select(true, h, mp, nullptr, Tl, lr);
        check(same(a, b, Tl) && a.n[1] == -1 && b.n[1] == -1, "select: a batched window's rows");
    }
}

// ---- the scores
void test_score() {
    const Pool pool = make_pool(40);   // 1920 blocks
    const int mp = 2048;
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<float> q((size_t) MAXT * QSA_HEADS * QSA_DIM);
    for (float& v : q) v = u(rng);
    const auto run = [&](bool mm, const int* pos0, int T, LaneRows lr) {
        std::vector<float> sc((size_t) T * mp, -3.f);
        if (mm) emu::launch(dim3((unsigned) (mp + 255) / 256, lr.pt ? (unsigned) T : (unsigned) (T + 7) / 8), 256,
                            [&] { k_idx_score_mm<8>(q.data(), pool.pooled(), pool.g, sc.data(), mp, pos0, lr, T); });
        else emu::launch(dim3((unsigned) (mp + 255) / 256, (unsigned) T), 256, [&] { k_idx_score(q.data(), pool.pooled(), pool.g, sc.data(), mp, pos0, lr); });
        return sc;
    };
    for (const auto& c : std::vector<std::pair<int, int>>{{4 * 700, 5}, {2046, 6}, {4 * 1500 + 2, 11}, {4 * 1900, 16}, {100, 3}, {4 * 600 + 1, 1}}) {
        int pos0 = c.first;
        const std::vector<float> a = run(false, &pos0, c.second, LaneRows{}), b = run(true, &pos0, c.second, LaneRows{});
        check(!memcmp(a.data(), b.data(), a.size() * 4), "score: position " + std::to_string(pos0) + ", " + std::to_string(c.second) + " rows: equal bits");
    }
    {
        const int Tl = 5;
        Pool other = make_pool(40);
        int posv[Tl] = {4 * 900 + 1, 100, 4 * 700 + 3, 4 * 1800, 4 * 513 + 2};
        const int* pp[Tl] = {&posv[0], &posv[1], &posv[2], &posv[3], &posv[4]};
        std::vector<int> pt2 = pool.pt;
        std::reverse(pt2.begin(), pt2.end());
        const int* ptp[Tl] = {pool.pt.data(), pt2.data(), pt2.data(), pool.pt.data(), pt2.data()};
        LaneRows lr;
        lr.pos = pp;
        lr.pt = ptp;
        const std::vector<float> a = run(false, nullptr, Tl, lr), b = run(true, nullptr, Tl, lr);
        check(!memcmp(a.data(), b.data(), a.size() * 4), "score: a batched window's rows: equal bits");
        (void) other;
    }
}

// ---- the attention
struct AttIn {
    const Pool* pool;
    std::vector<float> qg;
    std::vector<int> cells, n;
    int pos0 = 0, T = 0;
    LaneRows lr;
};
// the attention output as q8_1 rows and their twin (int8 codes, a float scale a block of 32)
struct Q8Rows {
    std::vector<block_q8_1> q;
    std::vector<int8_t> tq;
    std::vector<float> td;
    Q8Rows(int T, int n) : q((size_t) T * n / 32), tq((size_t) T * n), td((size_t) T * n / 32) {}
};
// k_quantize's steps for T rows of n values on the host (a block's sum in the warp's order: lane l takes lane l ^ 16, then ^ 8, 4, 2, 1)
Q8Rows quantize_ref(const std::vector<float>& x, int T, int n) {
    Q8Rows r(T, n);
    for (size_t b = 0; b < (size_t) T * n / 32; ++b) {
        const float* v = x.data() + b * 32;
        float amax = 0.f, sum[32];
        for (int l = 0; l < 32; ++l) amax = fmaxf(amax, fabsf(v[l])), sum[l] = v[l];
        for (int o = 16; o; o >>= 1) {
            float nx[32];
            for (int l = 0; l < 32; ++l) nx[l] = sum[l] + sum[l ^ o];
            std::copy(nx, nx + 32, sum);
        }
        const float d = amax / 127.0f;
        for (int l = 0; l < 32; ++l) r.q[b].qs[l] = r.tq[b * 32 + (size_t) l] = amax == 0.0f ? 0 : (int8_t) roundf(v[l] / d);
        r.q[b].ds = make_half2(__float2half(d), __float2half(sum[0]));
        r.td[b] = __half2float(__float2half(d));
    }
    return r;
}
// q8 (mm only): k_attn_combine_n also writes the output's q8_1 rows and twin there (FNX_ATT_Q8)
std::vector<float> attn_run(bool mm, const AttIn& in, int S, Q8Rows* q8 = nullptr) {
    const int T = in.T;
    std::vector<float> acc((size_t) T * NQH_LOC * ATT_SPLITS * HD, 1e30f), out((size_t) T * NQH_LOC * HD, -9.f);
    std::vector<AttPart> mz((size_t) T * NQH_LOC * ATT_SPLITS, AttPart{1e30f, 1e30f});
    const int* pos0 = in.lr.pos ? nullptr : &in.pos0;
    if (mm) {
        emu::launch(dim3((unsigned) S, (unsigned) T), 256, [&] {
            k_attn_mm(in.qg.data(), reinterpret_cast<const uint8_t*>(in.pool->kc()), reinterpret_cast<const uint8_t*>(in.pool->vc()), in.pool->g, acc.data(),
                      mz.data(), pos0, in.cells.data(), in.n.data(), in.lr, S);
        });
        emu::launch(dim3(NQH_LOC, (unsigned) T), HD, [&] {
            k_attn_combine_n(in.qg.data(), acc.data(), mz.data(), out.data(), NQH_LOC, S, q8 ? q8->q.data() : nullptr, q8 ? q8->tq.data() : nullptr,
                             q8 ? q8->td.data() : nullptr);
        });
    } else {
        emu::launch(dim3(ATT_SPLITS, (unsigned) T, KVH_LOC), HD, [&] {
            k_attn_part(in.qg.data(), in.pool->kc(), in.pool->vc(), in.pool->g, acc.data(), mz.data(), NQH_LOC, 0, KVH_LOC, 0, pos0, in.cells.data(),
                        in.n.data(), 0, in.lr);
        });
        emu::launch(dim3(NQH_LOC, (unsigned) T), HD, [&] { k_attn_combine(in.qg.data(), acc.data(), mz.data(), out.data(), NQH_LOC); });
    }
    return out;
}
std::vector<double> attn_ref(const AttIn& in) {
    const int T = in.T;
    std::vector<double> out((size_t) T * NQH_LOC * HD);
    const int8_t* kc = reinterpret_cast<const int8_t*>(in.pool->kc());
    const int8_t* vc = reinterpret_cast<const int8_t*>(in.pool->vc());
    for (int t = 0; t < T; ++t) {
        KvPages g = in.pool->g;
        if (in.lr.pt) g.pt = in.lr.pt[t];
        const int p = in.lr.pos ? *in.lr.pos[t] : in.pos0 + t;
        std::vector<int> pos;
        if (in.n[(size_t) t] >= 0) pos.assign(in.cells.begin() + (size_t) t * QSA_MAXSEL, in.cells.begin() + (size_t) t * QSA_MAXSEL + in.n[(size_t) t]);
        else
            for (int i = 0; i <= p; ++i) pos.push_back(i);
        for (int h = 0; h < NQH_LOC; ++h) {
            const float* q = in.qg.data() + ((size_t) t * NQH_LOC + h) * 2 * HD;
            std::vector<double> s(pos.size());
            double mx = -1e300;
            for (size_t i = 0; i < pos.size(); ++i) {
                const size_t r = kvq_row(g, pos[i]), sc = kvq_scale(g, pos[i]);
                double a = 0;
                for (int d = 0; d < HD; ++d) a += (double) q[d] * ((double) kc[r + d] * (double) (float) in.pool->kc()[sc + d / KV8_GROUP]);
                s[i] = a / 16.0;
                mx = std::max(mx, s[i]);
            }
            double z = 0;
            std::vector<double> acc(HD, 0.0);
            for (size_t i = 0; i < pos.size(); ++i) {
                const double w = std::exp(s[i] - mx);
                z += w;
                const size_t r = kvq_row(g, pos[i]), sc = kvq_scale(g, pos[i]);
                for (int d = 0; d < HD; ++d) acc[(size_t) d] += w * ((double) vc[r + d] * (double) (float) in.pool->vc()[sc + d / KV8_GROUP]);
            }
            for (int d = 0; d < HD; ++d) out[((size_t) t * NQH_LOC + h) * HD + d] = acc[(size_t) d] / z / (1.0 + std::exp(-(double) q[HD + d]));
        }
    }
    return out;
}
double rel_err(const std::vector<float>& a, const std::vector<double>& ref) {
    double d2 = 0, r2 = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d2 += ((double) a[i] - ref[i]) * ((double) a[i] - ref[i]);
        r2 += ref[i] * ref[i];
    }
    return std::sqrt(d2 / r2);
}
void fill_q(AttIn& in, float amp) {
    in.qg.resize((size_t) in.T * QG_W);
    std::uniform_real_distribution<float> u(-amp, amp);
    for (float& v : in.qg) v = u(rng);
}
// a random selection of 512 blocks and the tail per row past 2048 positions (dense under them)
void fill_cells(AttIn& in, const int* posv) {
    in.cells.assign((size_t) in.T * QSA_MAXSEL, -1);
    in.n.assign((size_t) in.T, -1);
    for (int t = 0; t < in.T; ++t) {
        const int p = posv[t], nv = (p + 1) / 4, n_tail = (p + 1) % 4;
        if (nv <= QSA_TOP) continue;
        std::vector<int> b((size_t) nv);
        for (int j = 0; j < nv; ++j) b[(size_t) j] = j;
        std::shuffle(b.begin(), b.end(), rng);
        std::sort(b.begin(), b.begin() + QSA_TOP);
        int n = 0;
        int* row = in.cells.data() + (size_t) t * QSA_MAXSEL;
        for (int j = 0; j < QSA_TOP; ++j)
            for (int m = 0; m < 4; ++m) row[n++] = b[(size_t) j] * 4 + m;
        for (int m = 0; m < n_tail; ++m) row[n++] = (p / 4) * 4 + m;
        in.n[(size_t) t] = n;
    }
}
// head h's q (not its gate) times f[h]
void scale_q(AttIn& in, const float* f) {
    for (int t = 0; t < in.T; ++t)
        for (int h = 0; h < NQH_LOC; ++h)
            for (int d = 0; d < HD; ++d) in.qg[((size_t) t * NQH_LOC + h) * 2 * HD + d] *= f[h];
}
void att_case(const std::string& name, AttIn& in, const std::vector<int>& splits, bool old_too) {
    const std::vector<double> ref = attn_ref(in);
    if (old_too) {
        const double e = rel_err(attn_run(false, in, 0), ref);
        check(e < 2e-5, "attn " + name + ": old kernel against the reference, rms rel " + std::to_string(e * 1e6) + "e-6");
    }
    for (int S : splits) {
        const int n = NQH_LOC * HD;
        Q8Rows got(in.T, n);
        const std::vector<float> out = attn_run(true, in, S, &got);
        const double e = rel_err(out, ref);
        check(e < 2e-5, "attn " + name + ": mm, " + std::to_string(S) + " splits, against the reference, rms rel " + std::to_string(e * 1e6) + "e-6");
        // the q8_1 rows of the combine's launch: the bytes a quantize of its output gives, and the output itself as without them
        const Q8Rows want = quantize_ref(out, in.T, n);
        const bool same = std::memcmp(got.q.data(), want.q.data(), got.q.size() * sizeof(block_q8_1)) == 0 && got.tq == want.tq &&
                          std::memcmp(got.td.data(), want.td.data(), got.td.size() * sizeof(float)) == 0 &&
                          std::memcmp(out.data(), attn_run(true, in, S).data(), out.size() * sizeof(float)) == 0;
        check(same, "attn " + name + ": mm, " + std::to_string(S) + " splits, the q8_1 rows and their twin = the quantize of the output");
    }
}
void test_attn(bool quick) {
    const Pool pool = make_pool(48);   // 9216 positions
    {   // dense rows of a short context (a row with one position too)
        AttIn in;
        in.pool = &pool;
        in.T = 4;
        in.pos0 = 297;
        fill_q(in, 2.f);
        int posv[4] = {297, 298, 299, 300};
        fill_cells(in, posv);
        att_case("dense, positions 297..300", in, {1, 4, 16}, true);
        in.pos0 = 0;
        int p0[4] = {0, 1, 2, 3};
        fill_cells(in, p0);
        att_case("dense, positions 0..3", in, {4, 16}, true);
    }
    {   // heads' q of very different sizes: tiny, small, past f16's range (the kernel scales a head's q by a power of two), and a zero q
        AttIn in;
        in.pool = &pool;
        in.T = 3;
        in.pos0 = 700;
        fill_q(in, 2.f);
        const float f[6] = {1e-6f, 3e-3f, 1.f, 40.f, 7e4f, 0.f};
        scale_q(in, f);
        int posv[3] = {700, 701, 702};
        fill_cells(in, posv);
        att_case("dense, positions 700..702, q times 1e-6 .. 7e4 and 0 by head", in, {1, 5}, true);
        in.pos0 = 5000;
        int pd[3] = {5000, 5001, 5002};
        fill_cells(in, pd);
        att_case("selected, positions 5000..5002, q times 1e-6 .. 7e4 and 0 by head", in, {4}, true);
    }
    {   // dense at the limit (2051 positions): more tiles a split than the row slots hold with 1 or 2 splits
        AttIn in;
        in.pool = &pool;
        in.T = 2;
        in.pos0 = 2049;
        fill_q(in, 1.5f);
        int posv[2] = {2049, 2050};
        fill_cells(in, posv);
        att_case("dense, positions 2049..2050", in, quick ? std::vector<int>{2} : std::vector<int>{1, 2, 9}, !quick);
    }
    {   // selected rows at depth, the window crossing a block; peaked scores (large q)
        AttIn in;
        in.pool = &pool;
        in.T = 3;
        in.pos0 = 9000;
        fill_q(in, 6.f);
        int posv[3] = {9000, 9001, 9002};
        fill_cells(in, posv);
        att_case("selected, positions 9000..9002, peaked", in, quick ? std::vector<int>{8} : std::vector<int>{1, 8, 16, 32}, true);
    }
    {   // a batched window: rows of their own positions and page tables, one dense
        AttIn in;
        in.pool = &pool;
        in.T = 3;
        static int posv[3] = {8000, 150, 5003};
        static const int* pp[3] = {&posv[0], &posv[1], &posv[2]};
        static std::vector<int> pt2 = pool.pt;
        std::rotate(pt2.begin(), pt2.begin() + 5, pt2.end());
        static const int* ptp[3] = {pool.pt.data(), pt2.data(), pt2.data()};
        in.lr.pos = pp;
        in.lr.pt = ptp;
        fill_q(in, 2.f);
        fill_cells(in, posv);
        att_case("a batched window's rows", in, {8}, true);
    }
}
}  // namespace
}  // namespace fnx

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::string(argv[1]) == "quick";
    fnx::test_score();
    fnx::test_attn(quick);
    fnx::test_select(quick);
    std::printf("%d failed (%ld wave exchanges, %ld barriers emulated)\n", fnx::fails, emu::S().xchg, emu::S().barriers);
    return fnx::fails ? 1 : 0;
}
