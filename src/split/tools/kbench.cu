// kbench: the decode window's kernels alone on one card, with random weights of the model's shapes: no model, no engine.
// Each case is a stream-captured graph of REP launches (as the window is one graph of ~600 launches), run RUNS times; it prints the us a launch and an
// FNV-1a hash of the outputs, so two builds (or two settings of a switch) are compared for speed and for equal bits.
//   strata-split-kbench [device=0] [rows="1,4,6,8,16"] [case=all|hc|dense|hchash|hcstub|chain|router|attn|qsasel|pfattn|moegu|tkgemm]      (FNX_* switches as in the engine)
// (builds with STRATA_SPLIT_FAST): the router case times k_router_route alone (with the shared expert's blocks, as the engine
// launches it) on the f32 table and on the bf16 tables and says whether all its outputs are the same bits; KBENCH_ROUTER_BF16=0|1|2 times one table
// only (and is the table the chain's router reads; default there 1).
// (builds with STRATA_SPLIT_APPROX): KBENCH_MM=0|1 (default 1) times the dp4a kernels or the matrix kernels in the dense
// and hc cases (KBENCH_MM_DENSE / KBENCH_MM_HC: one of them alone), and both cases print the largest difference of the matrix kernels' outputs against
// the dp4a kernels' on the same inputs (max abs, and max relative over the values of at least 1e-3 of the largest).
// attn, qsasel (builds with STRATA_SPLIT_APPROX; not part of `all`): the decode attention and the block selection at KBENCH_CTX positions
// (default 109000), the kernels of a build without the option against FNX_ATT_MM's in one process (KBENCH_ATT_MM=0|1: one of them only)
// Built by name: cmake --build <dir> --target strata-split-kbench
#include <cuda_runtime.h>
#if FNX_HIP
#include <hip/hip_ext.h>   // anyord: hipExtLaunchKernelGGL, hipExtAnyOrderLaunch
#endif
#include <cublas_v2.h>   // tkgemm KBENCH_GEMM_REF: the engine's f16 product through the BLAS library

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "../src/kernels.hpp"
#include "../src/tk.hpp"   // tkgemm: a generator family with no kernel in the binary

using namespace fnx;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kbench: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
std::mt19937_64 rng(12345);
template <typename T> T* dev(size_t n) {
    void* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    ck(cudaMemset(p, 0, n * sizeof(T)), "cudaMemset");
    return (T*) p;
}
int8_t* dev_i8(size_t n) {
    std::vector<int8_t> h(n);
    for (auto& v : h) v = (int8_t) ((int) (rng() % 255) - 127);
    int8_t* p = dev<int8_t>(n);
    ck(cudaMemcpy(p, h.data(), n, cudaMemcpyHostToDevice), "copy");
    return p;
}
float* dev_f32(size_t n, float lo, float hi) {
    std::vector<float> h(n);
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto& v : h) v = d(rng);
    float* p = dev<float>(n);
    ck(cudaMemcpy(p, h.data(), n * 4, cudaMemcpyHostToDevice), "copy");
    return p;
}
uint16_t f2h(float f) {   // positive normal floats only (scales)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const int e = (int) ((u >> 23) & 255) - 127 + 15;
    return (uint16_t) ((e << 10) | ((u >> 13) & 1023));
}
uint16_t* dev_scale16(size_t n, float lo, float hi) {
    std::vector<uint16_t> h(n);
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto& v : h) v = f2h(d(rng));
    uint16_t* p = dev<uint16_t>(n);
    ck(cudaMemcpy(p, h.data(), n * 2, cudaMemcpyHostToDevice), "copy");
    return p;
}
uint16_t* dev_bf16(size_t n, float lo, float hi) {
    std::vector<uint16_t> h(n);
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto& v : h) {
        const float f = d(rng);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        v = (uint16_t) (u >> 16);
    }
    uint16_t* p = dev<uint16_t>(n);
    ck(cudaMemcpy(p, h.data(), n * 2, cudaMemcpyHostToDevice), "copy");
    return p;
}
Q8S dev_q8s(int rows, int cols) {
    Q8S s;
    s.q = dev_i8((size_t) rows * cols);
    s.d = dev_scale16((size_t) rows * cols / 32, 0.5e-3f, 2e-3f);
    return s;
}
// n copies of a matrix at their own addresses, as a window reads a different matrix at every launch (the card's 64 MB cache would serve one matrix read
// 96 times; the same values in each, so the hashes do not depend on n)
std::vector<Q8S> copies(const Q8S& a, int rows, int cols, int n) {
    std::vector<Q8S> v(1, a);
    for (int i = 1; i < n; ++i) {
        Q8S s;
        s.q = dev<int8_t>((size_t) rows * cols);
        s.d = dev<uint16_t>((size_t) rows * cols / 32);
        ck(cudaMemcpy(s.q, a.q, (size_t) rows * cols, cudaMemcpyDeviceToDevice), "copy");
        ck(cudaMemcpy(s.d, a.d, (size_t) rows * cols / 16, cudaMemcpyDeviceToDevice), "copy");
        v.push_back(s);
    }
    return v;
}
unsigned long long fnv(const void* devp, size_t bytes, unsigned long long h = 1469598103934665603ull) {
    std::vector<unsigned char> b(bytes);
    ck(cudaMemcpy(b.data(), devp, bytes, cudaMemcpyDeviceToHost), "hash copy");
    for (unsigned char c : b) h = (h ^ c) * 1099511628211ull;
    return h;
}
// a router's table of NEXP + 1 rows as the files hold it (f32 values with the low 16 bits zero), and the same as bf16 in k_router_route's order (null
// without the build option)
struct RouterTab {
    float* f = nullptr;
    uint16_t* b = nullptr;
};
RouterTab dev_router(float lo, float hi) {
    const size_t n = (size_t) (NEXP + 1) * N;
    std::vector<float> h(n);
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto& v : h) {
        const float f = d(rng);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        u &= 0xffff0000u;
        std::memcpy(&v, &u, 4);
    }
    // KBENCH_ROUTER_TIES=1: equal table rows, so equal logits in the top 10's reduce: pairs in other lanes and other registers of a lane (r, r + 133),
    // in one lane (r, r + 32), in neighbour lanes (r, r + 1), and three equal rows. 2: also rows that lead where their logit is positive (about
    // half the tokens): five equal rows in one lane (x 16: the lane gives five picks in a row), three equal rows in three lanes (x 8), experts
    // 31 and 32 equal (x 4: the lower expert is in the higher lane), a row of NaN (never picked), and a row with one -inf (its logit is -inf or
    // +inf by the token: never picked, or first with NaN weights)
    if (const char* tv = std::getenv("KBENCH_ROUTER_TIES"); tv && std::atoi(tv)) {
        const auto same = [&](int a, int b, float f = 1.f) {
            for (int c = 0; c < N; ++c) h[(size_t) b * N + c] = h[(size_t) a * N + c] * f;   // (a power of two: bf16 stays exact)
        };
        for (int r = 0; r < 128; r += 2) same(r, r + 133);
        for (int r = 300; r < 332; ++r) same(r, r + 32);
        for (int r = 400; r < 440; r += 2) same(r, r + 1);
        same(450, 451), same(450, 452);
        if (std::atoi(tv) >= 2) {
            for (int r : {39, 71, 103, 135, 7}) same(7, r, 16.f);
            for (int r : {233, 266, 200}) same(200, r, 8.f);
            same(31, 32, 4.f), same(31, 31, 4.f);
            const uint32_t nan = 0x7fc00000u, ninf = 0xff800000u;
            for (int c = 0; c < N; ++c) std::memcpy(&h[(size_t) 300 * N + c], &nan, 4);
            std::memcpy(&h[(size_t) 301 * N], &ninf, 4);
        }
    }
    RouterTab t;
    t.f = dev<float>(n);
    ck(cudaMemcpy(t.f, h.data(), n * 4, cudaMemcpyHostToDevice), "copy");
#if FNX_ROUTER_BF16
    if (!router_bf16_exact(h.data(), n)) {
        std::fprintf(stderr, "kbench: the router table is not exact bf16\n");
        std::exit(2);
    }
    std::vector<uint16_t> o(n);
    router_bf16_pack(h.data(), NEXP + 1, o.data());
    t.b = dev<uint16_t>(n);
    ck(cudaMemcpy(t.b, o.data(), n * 2, cudaMemcpyHostToDevice), "copy");
#endif
    return t;
}
RouterTab copy_router(const RouterTab& a) {
    const size_t n = (size_t) (NEXP + 1) * N;
    RouterTab t;
    t.f = dev<float>(n);
    ck(cudaMemcpy(t.f, a.f, n * 4, cudaMemcpyDeviceToDevice), "copy");
    if (a.b) {
        t.b = dev<uint16_t>(n);
        ck(cudaMemcpy(t.b, a.b, n * 2, cudaMemcpyDeviceToDevice), "copy");
    }
    return t;
}
// everything a router_route launch of T rows writes, as one hash: the logits, the picks and their weights, the job count and the block counter, the
// jobs (a job's rows sorted: the kernel fills them in any order), the shared expert's rows
unsigned long long router_hash(const float* logits, const int* topi, const float* topw, const Job* gu, const Job* dn, const int* njobs, const float* sh_out,
                               int sh_rows, int T) {
    unsigned long long h = fnv(logits, (size_t) T * (NEXP + 1) * 4);
    h = fnv(topi, (size_t) T * TOPK * 4, h);
    h = fnv(topw, (size_t) T * TOPK * 4, h);
    h = fnv(njobs, 8, h);
    int nj[2];
    ck(cudaMemcpy(nj, njobs, 8, cudaMemcpyDeviceToHost), "njobs");
    if (nj[0] < 0 || nj[0] > T * TOPK) {
        std::fprintf(stderr, "kbench: router_route left %d jobs\n", nj[0]);
        std::exit(2);
    }
    for (const Job* jobs : {gu, dn}) {
        std::vector<Job> j((size_t) nj[0]);
        if (nj[0]) ck(cudaMemcpy(j.data(), jobs, (size_t) nj[0] * sizeof(Job), cudaMemcpyDeviceToHost), "jobs");
        for (Job& a : j) {
            if (a.n < 1 || a.n > MAXT) {
                std::fprintf(stderr, "kbench: a job of %d rows\n", a.n);
                std::exit(2);
            }
            std::vector<std::pair<int, int>> rows;
            for (int i = 0; i < a.n; ++i) rows.emplace_back(a.out[i], a.y[i]);
            std::sort(rows.begin(), rows.end());
            h = (h ^ (unsigned) a.w) * 1099511628211ull;
            h = (h ^ (unsigned) a.n) * 1099511628211ull;
            for (const auto& r : rows) h = ((h ^ (unsigned) r.first) * 1099511628211ull ^ (unsigned) r.second) * 1099511628211ull;
        }
    }
    if (sh_rows) h = fnv(sh_out, (size_t) T * sh_rows * 4, h);
    return h;
}
std::vector<float> fetch(const float* devp, size_t n) {
    std::vector<float> v(n);
    ck(cudaMemcpy(v.data(), devp, n * 4, cudaMemcpyDeviceToHost), "fetch");
    return v;
}
// the largest |a - b|, and the largest |a - b| / |a| over the values of a of at least 1e-3 of its largest (a: the reference)
struct Diff {
    double abs = 0, rel = 0, top = 0;
    void add(const std::vector<float>& a, const std::vector<float>& b) {
        double m = 0;
        for (float x : a) m = std::max(m, (double) std::fabs(x));
        top = std::max(top, m);
        for (size_t i = 0; i < a.size(); ++i) {
            const double d = std::fabs((double) a[i] - (double) b[i]);
            if (!(d == d)) abs = rel = 1e30;   // a NaN on either side
            abs = std::max(abs, d);
            if (std::fabs(a[i]) >= 1e-3 * m && m > 0) rel = std::max(rel, d / std::fabs(a[i]));
        }
    }
};
#if FNX_ATT_MM
// one layer's K/V slot of every page for `pages` pages (int8 rows of random codes with their f16 scales, f16 pooled keys), `layers` times at addresses of
// their own, and a shuffled page table: what the attention and the indexer read, with a page of one slot (the kernels take the page as a stride)
struct KvPool {
    std::vector<uint16_t*> base;   // per layer
    KvPages g;
    size_t rows_h = 0;   // halves of a layer's K (or V) rows with their scales
    const uint16_t* kc(int l) const { return base[(size_t) l]; }
    const uint16_t* vc(int l) const { return base[(size_t) l] + rows_h; }
    const uint16_t* pooled(int l) const { return base[(size_t) l] + 2 * rows_h; }
};
KvPool kv_pool(int pages, int layers) {
    KvPool p;
    const int rows = kvp_rows(0);
    p.rows_h = kvp_rows_halves(0, true);
    const size_t slot_h = kvp_slot(0, true), codes = (size_t) rows * KV_W, scales = (size_t) rows * KV_W / KV8_GROUP;
    std::vector<uint16_t> img((size_t) pages * slot_h);
    unsigned long long x = 0x9e3779b97f4a7c15ull;
    const auto next = [&] { x ^= x << 13, x ^= x >> 7, x ^= x << 17; return x; };
    for (int pg = 0; pg < pages; ++pg) {
        uint16_t* s = img.data() + (size_t) pg * slot_h;
        for (int kv = 0; kv < 2; ++kv) {
            int8_t* c = reinterpret_cast<int8_t*>(s + kv * p.rows_h);
            for (size_t i = 0; i < codes; ++i) c[i] = (int8_t) ((int) (next() % 255) - 127);
            uint16_t* sc = s + kv * p.rows_h + codes / 2;
            for (size_t i = 0; i < scales; ++i) sc[i] = f2h(0.005f + 0.045f * (float) (next() % 4096) / 4096.f);
        }
        uint16_t* pl = s + 2 * p.rows_h;
        for (size_t i = 0; i < (size_t) KVP_BLOCKS * 128; ++i) {
            const unsigned long long r = next();
            pl[i] = (uint16_t) (f2h(0.001f + (float) (r % 4096) / 4096.f) | ((r >> 20 & 1) << 15));
        }
    }
    std::vector<int> pt((size_t) pages);
    for (int i = 0; i < pages; ++i) pt[(size_t) i] = i;
    std::shuffle(pt.begin(), pt.end(), rng);
    int* dpt = dev<int>((size_t) pages);
    ck(cudaMemcpy(dpt, pt.data(), (size_t) pages * 4, cudaMemcpyHostToDevice), "copy");
    for (int l = 0; l < layers; ++l) {
        uint16_t* b = dev<uint16_t>(img.size());
        if (l == 0) ck(cudaMemcpy(b, img.data(), img.size() * 2, cudaMemcpyHostToDevice), "copy");
        else ck(cudaMemcpy(b, p.base[0], img.size() * 2, cudaMemcpyDeviceToDevice), "copy");
        p.base.push_back(b);
    }
    p.g.pt = dpt;
    p.g.page = slot_h;
    p.g.rows = rows;
    p.g.q8 = 1;
    return p;
}
template <typename T> std::vector<T> host(const T* devp, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), devp, n * sizeof(T), cudaMemcpyDeviceToHost), "host copy");
    return h;
}
template <typename T> void put(T* devp, const std::vector<T>& h) { ck(cudaMemcpy(devp, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "copy"); }
#endif

// f() queues the case's REP launches on st; returns the mean us a launch over RUNS graph launches (after 3 warm ones)
// (pre(): called before every launch of the graph, e.g. a launch on another stream that belongs to it)
template <typename F, typename P> double graph_us(cudaStream_t st, int rep, int runs, F f, P pre) {
    cudaGraph_t g;
    cudaGraphExec_t ex;
    ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed), "begin capture");
    f();
    ck(cudaStreamEndCapture(st, &g), "end capture");
    ck(cudaGraphInstantiate(&ex, g, nullptr, nullptr, 0), "instantiate");
    for (int i = 0; i < 3; ++i) {
        pre();
        ck(cudaGraphLaunch(ex, st), "launch");
    }
    ck(cudaStreamSynchronize(st), "sync");
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < runs; ++i) {
        pre();
        ck(cudaGraphLaunch(ex, st), "launch");
    }
    ck(cudaStreamSynchronize(st), "sync");
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    cudaGraphExecDestroy(ex);
    cudaGraphDestroy(g);
    return us / runs / rep;
}
template <typename F> double graph_us(cudaStream_t st, int rep, int runs, F f) {
    return graph_us(st, rep, runs, f, [] {});
}
// chain: the parts of a layer's weights, as bits, in the order the layer reads them
constexpr int KB_HC0 = 1, KB_QKVZ = 2, KB_SOUT = 4, KB_HC1 = 8, KB_ROUTER = 16, KB_EXP = 32, KB_MIXER = 7, KB_MOE = 24, KB_ALL = 31;
const struct { const char* name; int bits; } kb_part_names[] = {{"hc0", KB_HC0}, {"qkvz", KB_QKVZ}, {"sout", KB_SOUT}, {"hc1", KB_HC1},
    {"router", KB_ROUTER}, {"exp", KB_EXP}, {"hc", KB_HC0 | KB_HC1}, {"dense", KB_QKVZ | KB_SOUT}, {"mixer", KB_MIXER}, {"moe", KB_MOE},
    {"all", KB_ALL}, {"none", 0}};
// a part list: a number (the bits) or names joined by '+' (hc0 qkvz sout hc1 router exp; hc dense mixer moe all none); -1: neither
int kb_parts(const std::string& s) {
    if (s.empty()) return -1;
    if (s.find_first_not_of("0123456789") == std::string::npos) {
        const int m = std::atoi(s.c_str());
        return s.size() <= 2 && m <= 63 ? m : -1;
    }
    int m = 0;
    for (size_t i = 0;;) {
        const size_t j = s.find('+', i);
        const std::string t = s.substr(i, j - i);
        int b = -1;
        for (const auto& n : kb_part_names)
            if (t == n.name) b = n.bits;
        if (b < 0) return -1;
        m |= b;
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return m;
}
std::string kb_parts_text(int m) {
    if (!m) return "none";
    std::string t;
    for (int k = 0; k < 6; ++k)
        if (m >> k & 1) t += (t.empty() ? "" : "+") + std::string(kb_part_names[k].name);
    return t;
}
std::vector<std::string> kb_split(const std::string& s, char c) {
    std::vector<std::string> out;
    for (size_t i = 0;;) {
        const size_t j = s.find(c, i);
        out.push_back(s.substr(i, j - i));
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return out;
}
// resid: how many blocks of 256 threads the card runs at once, by registers a wave and shared memory a block. A block keeps K floats a lane live
// through its wait (the kernel then needs K vector registers and a few), counts itself in w[0] and waits until every block of the grid has counted
// or `spins` polls passed; w[1] = the least count a block saw at its wait's end (the blocks that ran from the start: the card's residency; a grid
// that is all resident gives the grid), w[2] = how many blocks waited to the end. lds_floats of dynamic shared memory are touched.
template <int K>
__global__ void __launch_bounds__(256) k_kb_resid(unsigned* w, const float* src, float* sink, int spins, int lds_floats) {
    extern __shared__ float lds[];
    float v[K];
#pragma unroll
    for (int i = 0; i < K; ++i) v[i] = src[threadIdx.x + i * 256];
    if (threadIdx.x == 0) {
        atomicAdd(w, 1u);
        int n = 0;
        while (atomicAdd(w, 0u) < gridDim.x && n < spins) {
            __builtin_amdgcn_s_sleep(8);
            ++n;
        }
        atomicMin(w + 1, atomicAdd(w, 0u));
        if (n >= spins) atomicAdd(w + 2, 1u);
    }
    __syncthreads();
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < K; ++i) s += v[i] * (float) (i + 1);
    lds[threadIdx.x % lds_floats] = s;
    lds[(threadIdx.x * 97 + 13) % lds_floats] = s;
    __syncthreads();
    sink[(size_t) blockIdx.x * 256 + threadIdx.x] = s + lds[threadIdx.x % lds_floats];
}
template <int K> void kb_resid_launch(cudaStream_t st, int grid, int lds_bytes, unsigned* w, const float* src, float* sink, int spins) {
    k_kb_resid<K><<<grid, 256, (size_t) lds_bytes, st>>>(w, src, sink, spins, lds_bytes / 4);
}
#if FNX_MALL_PF
// chain, KBENCH_PF_EVENTS: one more event for the persistent prefetcher's count, as a kernel of the chain's stream before a kernel of the layer (in
// the engine that kernel would carry the count itself, as the router does: no launch of its own)
__global__ void k_kb_tick(unsigned* w) { atomicAdd(w, 1u); }
void kb_tick(unsigned* w, cudaStream_t st) {
    k_kb_tick<<<1, 1, 0, st>>>(w);
    ck(cudaGetLastError(), "tick");
}
#else
void kb_tick(unsigned*, cudaStream_t) {}
#endif
// ktok: ALU work behind a hand-off word: a block waits (thread 0) until cnt[0] == want (every block of the launches before has ended), works,
// then adds 1. A wait of ~2 s gives up and counts in cnt[1]. spin: polls without a sleep
__global__ void k_kb_tok(float* out, int turns, unsigned* cnt, unsigned want, int tok, int spin) {
    if (tok) {
        if (threadIdx.x == 0) {
            unsigned n = 0;
#pragma unroll 1
            while (__hip_atomic_load(cnt, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != want) {
                if (++n > 40000000u) {
                    atomicAdd(cnt + 1, 1u);
                    break;
                }
                if (!spin) __builtin_amdgcn_s_sleep(1);
            }
        }
        __syncthreads();
    }
    float x = 1.0f + (float) (blockIdx.x * blockDim.x + threadIdx.x) * 1e-6f;
#pragma unroll 1
    for (int i = 0; i < turns; ++i) x = __builtin_fmaf(x, 1.0000001f, 1e-7f);
    if (threadIdx.x == 0) out[blockIdx.x] = x;
    if (tok) {
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(cnt, 1u);
    }
}
__global__ void k_kb_set(unsigned* w, unsigned v) { __hip_atomic_store(w, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
// pfhit: every 16-byte word of [p, p + bytes) read once by the grid (thread i takes words i, i + T, ...; 8 loads in flight), with non-temporal
// loads (nt: they fill no cache, as the engine's weight streams) or plain ones
__global__ void k_kb_read(const char* p, size_t bytes, int nt, unsigned* sink) {
    typedef unsigned kv4 __attribute__((ext_vector_type(4)));
    typedef kv4 kv4_u __attribute__((aligned(1)));
    const size_t T = (size_t) gridDim.x * blockDim.x, i0 = (size_t) blockIdx.x * blockDim.x + threadIdx.x, W = bytes / 16;
    unsigned acc = 0;
#pragma unroll 1
    for (size_t w = i0; w < W; w += 8 * T) {
        kv4 v[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const size_t o = w + (size_t) k * T < W ? w + (size_t) k * T : i0;
            const kv4_u* q = reinterpret_cast<const kv4_u*>(p + o * 16);
            v[k] = nt ? __builtin_nontemporal_load(q) : *q;
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) acc ^= v[k].x ^ v[k].w;
    }
    if (acc == 0x9e3779b9u) *sink = acc;   // (keeps the loads; a store only by chance)
}
// pfside: ALU work only (no loads, one store a block): a launch that holds the compute units and leaves the bus alone
__global__ void k_kb_spin(float* out, int turns) {
    float x = 1.0f + (float) (blockIdx.x * blockDim.x + threadIdx.x) * 1e-6f;
#pragma unroll 1
    for (int i = 0; i < turns; ++i) x = __builtin_fmaf(x, 1.0000001f, 1e-7f);
    if (threadIdx.x == 0) out[blockIdx.x] = x;
}
// anyord: ALU work, then a check of the chain's order: the launch before it in its chain left `want` in *w (bad counts the launches that saw another value)
__global__ void k_kb_seq(float* out, int turns, unsigned* w, unsigned want, unsigned* bad) {
    float x = 1.0f + (float) (blockIdx.x * blockDim.x + threadIdx.x) * 1e-6f;
#pragma unroll 1
    for (int i = 0; i < turns; ++i) x = __builtin_fmaf(x, 1.0000001f, 1e-7f);
    if (threadIdx.x == 0) out[blockIdx.x] = x;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (__hip_atomic_load(w, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != want) atomicAdd(bad, 1u);
        __hip_atomic_store(w, want + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
}
}  // namespace

int main(int argc, char** argv) {
    const int device = argc > 1 ? std::atoi(argv[1]) : 0;
    const std::string rows_s = argc > 2 ? argv[2] : "1,4,6,8,16", which = argc > 3 ? argv[3] : "all";
    std::vector<int> Ts;
    for (size_t i = 0; i < rows_s.size();) {
        const size_t j = rows_s.find(',', i);
        Ts.push_back(std::atoi(rows_s.substr(i, j - i).c_str()));
        if (j == std::string::npos) break;
        i = j + 1;
    }
    ck(cudaSetDevice(device), "set device");
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    const auto env_n = [](const char* k, int d) { const char* v = std::getenv(k); return v ? std::atoi(v) : d; };
    const int REP = env_n("KBENCH_REP", 96), RUNS = env_n("KBENCH_RUNS", 40);   // (a short probe of a grid that may never be resident: 2 and 1)
    std::printf("kbench: device %d, %d cards' shapes, graphs of %d launches, %d runs\n", device, NCARD, REP, RUNS);
    // which kernels the timed launches take (without the build options mm_force changes nothing)
    const int kmm = env_n("KBENCH_MM", 1), kmm_dense = env_n("KBENCH_MM_DENSE", kmm), kmm_hc = env_n("KBENCH_MM_HC", kmm);
    mm_force(kmm_dense, kmm_hc);
    // the router's table (the chain; the router case walks the tables itself)
    const int krt = FNX_ROUTER_BF16 ? std::min(2, std::max(0, env_n("KBENCH_ROUTER_BF16", 1))) : 0;
    std::printf("kbench: matrix kernels: dense %s, hc %s\n", !FNX_MM ? "not built" : kmm_dense ? "on" : "off (dp4a)",
                !FNX_MM_HC ? "not built" : kmm_hc ? "on (ticket queue)" : "off (k_hc)");

    // activations: mixed as q8_1 rows with the split twin, as every writer in the engine fills them
    void* xq = dev<char>(q8_1_bytes(N) * MAXT);
    int8_t* tq = dev<int8_t>((size_t) MAXT * N);
    float* td = dev<float>((size_t) MAXT * N / 32);
    q8_twin(xq, q8_1_bytes(N) * MAXT, tq, td);
    float* mixed = dev<float>((size_t) MAXT * N);
    unsigned* err = dev<unsigned>(ERR_WORDS);   // the error word and the time-out record behind it

    if (which == "all" || which == "hc" || which == "hchash" || which == "hcstub") {
        const std::vector<Q8S> down = copies(dev_q8s(LR, D), LR, D, REP), up = copies(dev_q8s(D, LR), D, LR, REP);
        const float* w_norm = dev_f32(D, 0.5f, 1.5f);
        const uint16_t* w_inj = dev_bf16((size_t) HC * D, -0.02f, 0.02f);
        float* R = dev<float>((size_t) MAXT * D);
        const float* R0 = dev_f32((size_t) MAXT * D, -2.f, 2.f);
        const float* bo = dev_f32((size_t) MAXT * N, -0.01f, 0.01f);
        float* inj_a = dev<float>((size_t) MAXT * HC);
        float* inj_b = dev<float>((size_t) MAXT * HC);
        float* hcs = dev<float>(hc_scratch_floats(MAXT));
        const int grid = hc_prepare(hcs);
        // the dense launch a window runs between two reads (qkvz's shape, a matrix of its own per launch): the read's code and data are then as cold as
        // in a window, where back-to-back reads keep them warm
        const int drows = (2 * (NKH / NCARD) + 2 * (NVH / NCARD)) * HS;
        const std::vector<Q8S> dw = copies(dev_q8s(drows, N), drows, N, REP);
        float* dout = dev<float>((size_t) MAXT * drows);
        // the engine's three shapes of the call: a layer's read (the last half's write folded in, inject rows), the first read of a window or of the
        // draft layer (no write), the head's and the draft mixer's (no inject rows)
        struct Shape { const char* name; bool bo, inj; };
        const Shape shapes[] = {{"layer", true, true}, {"first", false, true}, {"head", true, false}};
        const auto reset = [&] {
            ck(cudaMemcpy(R, R0, (size_t) MAXT * D * 4, cudaMemcpyDeviceToDevice), "R");
            ck(cudaMemset(inj_a, 0, MAXT * HC * 4), "inj");
            ck(cudaMemset(inj_b, 0, MAXT * HC * 4), "inj");
        };
        const auto reads = [&](const Shape& sh, int T, bool dense_between) {
            for (int i = 0; i < REP; ++i) {
                hc_read(R, sh.bo ? bo : nullptr, (i & 1) ? inj_b : inj_a, w_norm, down[(size_t) i], up[(size_t) i], sh.inj ? w_inj : nullptr, 1e-6f, mixed,
                        (i & 1) ? inj_a : inj_b, hcs, err, T, st, xq);
                if (dense_between) q8s_gemv(dw[(size_t) i], drows, N, xq, dout, drows, T, st);
            }
        };
        const auto hash = [&](int T) {
            unsigned long long h = fnv(R, (size_t) T * D * 4);
            h = fnv(mixed, (size_t) T * N * 4, h);
            h = fnv(inj_a, (size_t) T * HC * 4, h);
            h = fnv(inj_b, (size_t) T * HC * 4, h);
            h = fnv(xq, q8_1_bytes(N) * T, h);
            h = fnv(tq, (size_t) T * N, h);
            return fnv(td, (size_t) T * N / 32 * 4, h);
        };
        if (which == "hchash") {   // every shape and row count: the reference a changed kernel must reproduce
            for (const Shape& sh : shapes)
                for (int T = 1; T <= MAXT; ++T) {
                    reset();
                    // KBENCH_DIRTY=seed: the scratch (all but its barrier counters) filled with random bytes before each case. A read that takes
                    // anything from the scratch that this launch did not write then gives another hash than the reference (a fresh scratch is zeros)
                    if (const int seed = env_n("KBENCH_DIRTY", 0); seed) {
                        std::vector<unsigned> junk(hc_scratch_floats(MAXT) - 2);
                        unsigned x = (unsigned) seed * 2654435761u + (unsigned) T;
                        for (unsigned& v : junk) x ^= x << 13, x ^= x >> 17, x ^= x << 5, v = x;
                        ck(cudaMemcpy(hcs, junk.data(), junk.size() * 4, cudaMemcpyHostToDevice), "dirty scratch");
                    }
                    graph_us(st, REP, 1, [&] { reads(sh, T, false); });
                    unsigned e = 0;
                    ck(cudaMemcpy(&e, err, 4, cudaMemcpyDeviceToHost), "err");
                    std::printf("hchash %-5s T=%2d err %u %016llx\n", sh.name, T, e, hash(T));
                }
            return 0;
        }
#if FNX_MM_HC
        // one read of each shape from the same state with k_hc, then with the ticket queue: the largest differences of every output.
        // Before each of the two reads every output and the scratch (all but k_hc's barrier counters) are filled with 0xff bytes (NaN floats): a value
        // a kernel does not write, or takes from the other kernel's run, shows as a difference of 1e30
        if (which != "hcstub")
            for (int T : Ts)
                for (const Shape& sh : shapes) {
                    std::vector<float> ref[4], got[4];
                    std::vector<int8_t> qb[2], tb[2];
                    unsigned e2[2] = {0, 0};
                    for (int m = 0; m < 2; ++m) {
                        mm_force(kmm_dense, m);
                        reset();
                        ck(cudaMemset(mixed, 0xff, (size_t) MAXT * N * 4), "mixed");
                        ck(cudaMemset(inj_b, 0xff, (size_t) MAXT * HC * 4), "inj");
                        ck(cudaMemset(xq, 0xff, q8_1_bytes(N) * MAXT), "xq");
                        ck(cudaMemset(tq, 0xff, (size_t) MAXT * N), "tq");
                        ck(cudaMemset(td, 0xff, (size_t) MAXT * N / 32 * 4), "td");
                        ck(cudaMemset(hcs, 0xff, (hc_scratch_floats(MAXT) - 2) * 4), "scratch");
                        ck(cudaMemset(err, 0, ERR_WORDS * 4), "err");
                        hc_read(R, sh.bo ? bo : nullptr, inj_a, w_norm, down[0], up[0], sh.inj ? w_inj : nullptr, 1e-6f, mixed, inj_b, hcs, err, T, st, xq);
                        ck(cudaStreamSynchronize(st), "sync");
                        std::vector<float>* o = m ? got : ref;
                        o[0] = fetch(mixed, (size_t) T * N), o[1] = fetch(R, (size_t) T * D);
                        o[2] = sh.inj ? fetch(inj_b, (size_t) T * HC) : std::vector<float>();
                        o[3] = fetch(td, (size_t) T * N / 32);
                        qb[m].resize(q8_1_bytes(N) * T), tb[m].resize((size_t) T * N);
                        ck(cudaMemcpy(qb[m].data(), xq, qb[m].size(), cudaMemcpyDeviceToHost), "xq");
                        ck(cudaMemcpy(tb[m].data(), tq, tb[m].size(), cudaMemcpyDeviceToHost), "tq");
                        ck(cudaMemcpy(&e2[m], err, 4, cudaMemcpyDeviceToHost), "err");
                    }
                    Diff dm, dr, di, dq;
                    dm.add(ref[0], got[0]), dr.add(ref[1], got[1]), di.add(ref[2], got[2]), dq.add(ref[3], got[3]);
                    // the q8_1 rows byte for byte (scales and sums as halves, the int8 values) and the twin's int8 values: how many differ, by how much
                    size_t nq = 0, nt = 0;
                    int mt = 0;
                    for (size_t i = 0; i < qb[0].size(); ++i) nq += qb[0][i] != qb[1][i];
                    for (size_t i = 0; i < tb[0].size(); ++i)
                        if (tb[0][i] != tb[1][i]) ++nt, mt = std::max(mt, std::abs((int) tb[0][i] - (int) tb[1][i]));
                    std::printf("hc_read diff %-5s T=%2d  mixed abs %.3g rel %.3g (top %.3g)  R abs %.3g  inject abs %.3g rel %.3g  q8 scale rel %.3g  q8_1 bytes "
                                "%zu of %zu  twin int8 %zu of %zu (max %d)  err %u / %u\n",
                                sh.name, T, dm.abs, dm.rel, dm.top, dr.abs, di.abs, di.rel, dq.rel, nq, qb[0].size(), nt, tb[0].size(), mt, e2[0], e2[1]);
                }
        // the stubs are k_hc's (k_hc_mm has none): that case times k_hc
        mm_force(kmm_dense, which == "hcstub" ? 0 : kmm_hc);
        ck(cudaMemset(err, 0, ERR_WORDS * 4), "err");
        ck(cudaMemset(hcs, 0, (hc_scratch_floats(MAXT) - 2) * 4), "scratch");
#endif
        for (int T : Ts) {
#if FNX_KBENCH
            const std::vector<int> stubs = which == "hcstub" ? std::vector<int>{0, 1, 2, 3, 4, 8, 16, 32, 63} : std::vector<int>{0};
#else
            const std::vector<int> stubs{0};
#endif
            for (int stub : stubs) {
#if FNX_KBENCH
                hc_set_stub(stub);
#endif
                // 3 passes, the best of each: alone (back to back), and between dense launches minus the dense launches alone
                double alone = 1e30, mixed_us = 1e30, dense_us = 1e30;
                for (int pass = 0; pass < 3; ++pass) {
                    reset();
                    alone = std::min(alone, graph_us(st, REP, RUNS, [&] { reads(shapes[0], T, false); }));
                    reset();
                    mixed_us = std::min(mixed_us, graph_us(st, REP, RUNS, [&] { reads(shapes[0], T, true); }));
                    dense_us = std::min(dense_us, graph_us(st, REP, RUNS, [&] {
                        for (int i = 0; i < REP; ++i) q8s_gemv(dw[(size_t) i], drows, N, xq, dout, drows, T, st);
                    }));
                }
                unsigned e = 0;
                ck(cudaMemcpy(&e, err, 4, cudaMemcpyDeviceToHost), "err");
                std::printf("hc_read T=%2d stub %2d  alone %6.2f us   in a window %6.2f us (with dense %6.2f - dense %6.2f)   grid %d%s err %u\n", T, stub, alone,
                            mixed_us - dense_us, mixed_us, dense_us, mm_hc() ? hc_mm_grid() : grid, mm_hc() ? " (ticket queue)" : "", e);
            }
        }
    }
    // router: k_router_route alone, as the engine launches it: a table of its own a launch (48 layers' tables do not stay in the card's
    // 64 MB cache; KBENCH_WARM=1: one table), the shared expert's gate|up blocks in the same launch (KBENCH_ROUTER_SH=0: without them), no cache plan.
    // Each table mode in turn (KBENCH_ROUTER_BF16=m: that one only): us a launch and the hash of everything the launch writes.
    if (which == "router") {
        const int nw = env_n("KBENCH_WARM", 0) ? 1 : REP, with_sh = env_n("KBENCH_ROUTER_SH", 1);
        const int card = std::min(std::max(env_n("KBENCH_CARD", NCARD - 1), 0), NCARD - 1);   // whose FF slice the shared expert's blocks have
        const int sh_rows = with_sh ? 2 * (ff_hi(NCARD, card) - ff_lo(NCARD, card)) : 0;
        std::vector<RouterTab> tabs(1, dev_router(-0.05f, 0.05f));
        for (int i = 1; i < nw; ++i) tabs.push_back(copy_router(tabs[0]));
        std::vector<Q8S> sh;
        if (sh_rows) sh = copies(dev_q8s(sh_rows, N), sh_rows, N, nw);
        const float* x = dev_f32((size_t) MAXT * N, -1.f, 1.f);
        quantize_q8_1(x, xq, N, MAXT, st);   // the shared expert's activations: x as q8_1 rows (and their twin)
        ck(cudaStreamSynchronize(st), "sync");
        float* logits = dev<float>((size_t) MAXT * (NEXP + 1));
        int* topi = dev<int>(MAXT * TOPK);
        float* topw = dev<float>(MAXT * TOPK);
        Job* jobs_gu = dev<Job>(MAXT * TOPK + 1);
        Job* jobs_dn = dev<Job>(MAXT * TOPK + 1);
        int* njobs = dev<int>(2);
        float* sh_out = dev<float>((size_t) MAXT * std::max(sh_rows, 1));
        std::vector<int> modes;
        if (std::getenv("KBENCH_ROUTER_BF16") || !FNX_ROUTER_BF16) modes = {krt};
        else modes = {0, 1, 2};
        const char* mname[] = {"f32", "bf16", "bf16, a warp a row"};
        std::printf("router: %d table(s) of %.2f MB f32 / %.2f MB bf16, shared expert %d rows (%.2f MB) in the launch%s\n", nw, (NEXP + 1) * N * 4 / 1e6,
                    (NEXP + 1) * N * 2 / 1e6, sh_rows, sh_rows * N * (1.0 + 2.0 / 32) / 1e6, FNX_ROUTER_BF16 ? "" : " (no bf16 table: STRATA_SPLIT_FAST is off)");
        bool all_equal = true;
        for (int T : Ts) {
            unsigned long long h0 = 0;
            for (size_t k = 0; k < modes.size(); ++k) {
                const int m = modes[k];
                router_bf16_force(m);
                ck(cudaMemset(njobs, 0, 8), "njobs");   // njobs[1] is the block counter: zero before a launch
                const auto poison = [&] {
                    ck(cudaMemset(logits, 0xff, (size_t) MAXT * (NEXP + 1) * 4), "poison");
                    ck(cudaMemset(topi, 0xff, MAXT * TOPK * 4), "poison");
                    ck(cudaMemset(topw, 0xff, MAXT * TOPK * 4), "poison");
                    ck(cudaMemset(jobs_gu, 0xff, (MAXT * TOPK + 1) * sizeof(Job)), "poison");
                    ck(cudaMemset(jobs_dn, 0xff, (MAXT * TOPK + 1) * sizeof(Job)), "poison");
                    ck(cudaMemset(sh_out, 0xff, (size_t) MAXT * std::max(sh_rows, 1) * 4), "poison");
                };
                // one launch alone on poisoned outputs, a table at each end of the list: what a single launch writes (the timed graphs below write
                // over their own outputs, so their hash alone would not show a launch that left something out)
                unsigned long long h1[2];
                for (int c = 0; c < 2; ++c) {
                    const size_t i = c ? (size_t) nw - 1 : 0;
                    poison();
                    router_route(tabs[i].f, x, logits, T, topi, topw, jobs_gu, jobs_dn, njobs, st, nullptr, -1, nullptr, sh_rows ? &sh[i] : nullptr, sh_rows, xq,
                                 sh_out, tabs[i].b);
                    ck(cudaStreamSynchronize(st), "sync");
                    h1[c] = router_hash(logits, topi, topw, jobs_gu, jobs_dn, njobs, sh_out, sh_rows, T);
                    if (c == 0 && std::getenv("KBENCH_ROUTER_DUMP")) {   // the picks of each token, in the picks' order
                        std::vector<int> ti((size_t) T * TOPK);
                        ck(cudaMemcpy(ti.data(), topi, ti.size() * 4, cudaMemcpyDeviceToHost), "picks");
                        for (int t = 0; t < T; ++t) {
                            std::printf("picks T=%2d token %2d:", T, t);
                            for (int q = 0; q < TOPK; ++q) std::printf(" %d", ti[(size_t) t * TOPK + q]);
                            std::printf("\n");
                        }
                    }
                }
                poison();
                double best = 1e30;
                for (int pass = 0; pass < 3; ++pass)
                    best = std::min(best, graph_us(st, REP, RUNS, [&] {
                        for (int i = 0; i < REP; ++i) {
                            const RouterTab& t = tabs[(size_t) (i % nw)];
                            router_route(t.f, x, logits, T, topi, topw, jobs_gu, jobs_dn, njobs, st, nullptr, -1, nullptr, sh_rows ? &sh[(size_t) (i % nw)] : nullptr,
                                         sh_rows, xq, sh_out, t.b);
                        }
                    }));
                const unsigned long long h = router_hash(logits, topi, topw, jobs_gu, jobs_dn, njobs, sh_out, sh_rows, T);
                const bool steady = h1[0] == h && h1[1] == h;   // the single launches and the last of the timed ones wrote the same
                if (k == 0) h0 = h;
                if (h != h0 || !steady) all_equal = false;
                std::printf("router T=%2d  %-18s %6.2f us a launch  hash %016llx%s%s\n", T, mname[m], best, h,
                            modes.size() < 2 ? "" : h == h0 ? (k ? "  = f32's" : "") : "  DIFFERS from f32's", steady ? "" : "  LAUNCHES DIFFER (single / timed)");
            }
        }
        router_bf16_force(-1);
        std::printf("router: outputs %s\n", !all_equal ? "DIFFER" : modes.size() > 1 ? "equal in every mode and launch" : "equal in every launch");
        return all_equal ? 0 : 1;
    }
    // chain: 12 "layers" of one card with a weight set of their own each, as a window walks them: the hc read, a qkvz-size
    // projection, an ssm_out-size projection, the second hc read, then a read-once stream of ~25 MB standing for the routed experts (a Q8 matrix of its
    // own per layer; its loads are the build's FNX_MALL_NT level 2 loads, not the expert kernels'). One captured graph. KBENCH_PF=1: before each layer's
    // expert stream the next layer's fixed weights are staged with mall_pf on a side stream (KBENCH_PF_WG workgroups, default 4; KBENCH_PF_FULL=1 every
    // 16 bytes; KBENCH_PF_LINE), one join at the graph's end. Prints the us a layer and the hash of the outputs (the same with and without).
    if (which == "chain") {
        constexpr int NLAY = 12;
        const int qrows = (2 * (NKH / NCARD) + 2 * (NVH / NCARD)) * HS, ocols = (NVH / NCARD) * HS, erows = 9600;
        struct Lay {
            Q8S down[2], up[2], qkvz, sout, exp;
            RouterTab router;
            float* logits = nullptr;
        };
        // KBENCH_CHAIN_ROUTER (default 1): the layer's router (NEXP + 1 rows; its table by KBENCH_ROUTER_BF16) between the second hc read and the expert
        // stream, as in the engine; no shared-expert blocks and no cache plan. The table it reads is in the staged set, and its launch can carry the
        // persistent prefetcher's count (KBENCH_PF=2). 0: the chain without it
        const bool with_router = env_n("KBENCH_CHAIN_ROUTER", 1) != 0;
        if (with_router) router_bf16_force(krt);
        std::vector<Lay> lay((size_t) NLAY);
        for (Lay& l : lay) {   // every layer its own values (a read of the wrong layer's matrix changes the hash)
            for (int h = 0; h < 2; ++h) l.down[h] = dev_q8s(LR, D), l.up[h] = dev_q8s(D, LR);
            l.qkvz = dev_q8s(qrows, N), l.sout = dev_q8s(N, ocols), l.exp = dev_q8s(erows, N);
        }
        // (the routers after every layer's matrices, as the engine's chain makes them: the same random values, so the same hash)
        if (with_router)
            for (Lay& l : lay) l.router = dev_router(-0.05f, 0.05f), l.logits = dev<float>((size_t) MAXT * (NEXP + 1));
        // KBENCH_CHAIN_WARM=1: the shared-weights run: every layer reads layer 0's fixed weights (hc matrices, both projections, the router); each
        // layer's expert stream stays its own. No prefetcher (KBENCH_PF must be 0), a hash of its own. It is the warm bound (a prefetch that costs
        // nothing) as far as the set stays in the 64 MB cache: with mall_pf in the build the set is staged with plain loads before each pass and,
        // KBENCH_WARM_RESTAGE=1 (default), again on the side stream beside every graph launch (the chain's own loads of it can be non-temporal:
        // a hint, no promise that a line stays); without mall_pf only the chain's own loads keep it (said in the header line)
        const bool warm = env_n("KBENCH_CHAIN_WARM", 0) != 0;
        if (warm)
            for (size_t i = 1; i < lay.size(); ++i) {
                for (int h = 0; h < 2; ++h) lay[i].down[h] = lay[0].down[h], lay[i].up[h] = lay[0].up[h];
                lay[i].qkvz = lay[0].qkvz, lay[i].sout = lay[0].sout;
                if (with_router) lay[i].router = lay[0].router;
            }
        int* topi = dev<int>((size_t) MAXT * TOPK);
        float* topw = dev<float>((size_t) MAXT * TOPK);
        Job* jobs_gu = dev<Job>((size_t) MAXT * TOPK + 1);
        Job* jobs_dn = dev<Job>((size_t) MAXT * TOPK + 1);
        int* njobs = dev<int>(2);
        const float* w_norm = dev_f32(D, 0.5f, 1.5f);
        const uint16_t* w_inj = dev_bf16((size_t) HC * D, -0.02f, 0.02f);
        float* R = dev<float>((size_t) MAXT * D);
        const float* R0 = dev_f32((size_t) MAXT * D, -2.f, 2.f);
        const float* bo = dev_f32((size_t) MAXT * N, -0.01f, 0.01f);
        float* inj_a = dev<float>((size_t) MAXT * HC);
        float* inj_b = dev<float>((size_t) MAXT * HC);
        float* hcs = dev<float>(hc_scratch_floats(MAXT));
        hc_prepare(hcs);
        // every layer keeps its outputs: the hash covers all of them, not the last layer's only
        std::vector<float*> qout((size_t) NLAY), sout((size_t) NLAY), eout((size_t) NLAY);
        for (int i = 0; i < NLAY; ++i) {
            qout[(size_t) i] = dev<float>((size_t) MAXT * qrows);
            sout[(size_t) i] = dev<float>((size_t) MAXT * N);
            eout[(size_t) i] = dev<float>((size_t) MAXT * erows);
        }
        void* yo = dev<char>(q8_1_bytes(ocols) * MAXT);   // ssm_out's input rows (ocols wide)
        {
            int8_t* yq = dev_i8((size_t) MAXT * ocols);
            float* yd = dev_f32((size_t) MAXT * ocols / 32, 0.005f, 0.02f);
            q8_twin(yo, q8_1_bytes(ocols) * MAXT, yq, yd);
        }
        // KBENCH_PF: 0 none; 1 a fork a layer in the graph (k_mall_pf on the side stream, one join at the end); 2 the persistent prefetcher: one
        // k_mall_pf_run on the side stream before every launch of the chain's graph, outside it, that follows the routers' count
        const int pf = env_n("KBENCH_PF", 0), wgs = env_n("KBENCH_PF_WG", 4);
        if (pf && !mall_pf_built()) {
            std::printf("chain: KBENCH_PF=%d but this build has no prefetch (STRATA_SPLIT_FAST)\n", pf);
            return 2;
        }
        if (pf < 0 || pf > 3) {
            std::printf("chain: KBENCH_PF 0, 1, 2 or 3\n");
            return 2;
        }
        // KBENCH_PF=3: the test of "a warm set speeds the layer up": in the graph, on the MAIN stream, layer i's set is staged (k_mall_pf, plain
        // loads: the hint that keeps a line in the cache) right before layer i's kernels, so the set is in the cache when the layer's first kernel
        // starts. The staging costs its own time, so the same graph with the staging launches only is timed too and subtracted: "the layers".
        // KBENCH_PF3_ONLY=1: only that staging graph. KBENCH_PF_WG up to 1024 here (a full grid); KBENCH_PF_FULL / KBENCH_PF_LINE as for the others.
        // KBENCH_PF3_AT / KBENCH_PF3_SET (below): the staging launches at another point of the graph, and of a part of the set only.
        const bool pf3_only = pf == 3 && env_n("KBENCH_PF3_ONLY", 0) != 0;
        if (wgs < 1 || wgs > (pf == 3 ? 1024 : 64)) {
            std::printf("chain: KBENCH_PF_WG 1..64 (KBENCH_PF=3: 1..1024)\n");
            return 2;
        }
        if (warm && pf) {
            std::printf("chain: KBENCH_CHAIN_WARM=1 is the run without a prefetcher: KBENCH_PF must be 0\n");
            return 2;
        }
        // the side stream. KBENCH_PF=2: of a priority class of its own (KBENCH_PF_PRIO: 1 high (default), -1 low; 0 a default stream), so that the
        // persistent prefetcher has a hardware queue no other stream shares (kernels.hpp mall_pf_stream), as the engine's
        // KBENCH_PF_QUERY=1: a stream query after every prefetcher launch (a test switch)
        cudaStream_t st3;
        cudaEvent_t ev_fork, ev_join;
        bool pf_own = false;
        const bool pf_query = env_n("KBENCH_PF_QUERY", 0) != 0;
        if (pf == 2)
            st3 = (cudaStream_t) mall_pf_stream(env_n("KBENCH_PF_PRIO", 1), &pf_own);
        else
            ck(cudaStreamCreateWithFlags(&st3, cudaStreamNonBlocking), "side stream");
        ck(cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming), "event");
        ck(cudaEventCreateWithFlags(&ev_join, cudaEventDisableTiming), "event");
        unsigned* pf_sink = dev<unsigned>(16);
        // The parts of a layer's fixed weights (kb_parts: names and bits), in the order the layer reads them: hc0 (the first hc read's down and up),
        // qkvz, sout, hc1 (the second hc read's), router. mixer = hc0 + qkvz + sout: what the layer's first half reads; moe = hc1 + router: the second
        // half's. exp is the layer's expert stream: not in "all" (it stands for the routed experts, which nobody knows before the router); staged,
        // it stands for a shared expert of that size. The chain has no shared expert of its own.
        const auto ranges_of = [&](int layer, int parts) {
            std::vector<PfSet::R> out;
            const Lay& l = lay[(size_t) (layer % NLAY)];   // (a layer past the last: the caller's choice, mk_set)
            const auto add = [&](const void* p, long long bytes) { out.push_back(PfSet::R{p, bytes}); };
            const auto q8s = [&](const Q8S& q, long long rows, long long cols) { add(q.q, rows * cols), add(q.d, rows * cols / 16); };
            if (parts & KB_HC0) q8s(l.down[0], LR, D), q8s(l.up[0], D, LR);
            if (parts & KB_QKVZ) q8s(l.qkvz, qrows, N);
            if (parts & KB_SOUT) q8s(l.sout, N, ocols);
            if (parts & KB_HC1) q8s(l.down[1], LR, D), q8s(l.up[1], D, LR);
            if ((parts & KB_ROUTER) && with_router) {   // the table the router reads, as the engine's set has it
                if (krt && l.router.b) add(l.router.b, (long long) (NEXP + 1) * N * 2);
                else add(l.router.f, (long long) (NEXP + 1) * N * 4);
            }
            if (parts & KB_EXP) q8s(l.exp, erows, N);
            return out;
        };
        const auto part_mb = [&](int parts) {
            double mb = 0;
            for (const PfSet::R& r : ranges_of(0, parts)) mb += r.bytes / 1e6;
            return mb;
        };
        // a staged set: its items (parts of layer i + off) in order. KBENCH_PF_CAP_MB (KBENCH_PF=2 and 3; MB of 1e6 bytes, 0 none): the engine's
        // rule (FNX_MALL_PF_CAP, in MiB there): a range that would pass the cap is left out, smaller ones after it may still fit.
        // wrap false: an item of a layer past the last one is left out (the engine's FNX_MALL_PF_WRAP=0); true: it is layer 0's, for the next launch
        struct Item { int parts, off; };
        const double cap_mb = std::getenv("KBENCH_PF_CAP_MB") ? std::atof(std::getenv("KBENCH_PF_CAP_MB")) : 0.0;
        const long long cap_b = cap_mb > 0 && pf >= 2 ? (long long) (cap_mb * 1e6) : 0;
        const auto mk_set = [&](int i, const std::vector<Item>& items, bool capped = true, bool wrap = true) {
            PfSet s;
            s.line = env_n("KBENCH_PF_LINE", 128);
            s.full = env_n("KBENCH_PF_FULL", 0);
            s.sink = pf_sink;
            long long used = 0;
            for (const Item& it : items) {
                if (!wrap && i + it.off >= NLAY) continue;
                for (const PfSet::R& r : ranges_of(i + it.off, it.parts)) {
                    if (capped && cap_b && used + r.bytes > cap_b) continue;   // (the cap)
                    if (s.n == MALL_PF_MAXR) {
                        std::printf("chain: a staged set of more than %d ranges\n", MALL_PF_MAXR);
                        std::exit(2);
                    }
                    s.r[s.n++] = r, used += r.bytes;
                }
            }
            return s;
        };
        const auto set_mb_of = [](const PfSet& s) {
            double mb = 0;
            for (int k = 0; k < s.n; ++k) mb += s.r[k].bytes / 1e6;
            return mb;
        };
        // a layer's whole set (KBENCH_PF=1, the shared-weights run)
        std::vector<PfSet> sets((size_t) NLAY);
        for (int i = 0; i < NLAY; ++i) sets[(size_t) i] = mk_set(i, {{KB_ALL, 0}}, false);
        const double set_mb = part_mb(KB_ALL);
        // KBENCH_PF=3: where the staging launches go. plan[layer * 6 + k]: the set staged on the main stream before kernel k of that layer
        // (k: 0 the first hc read, 1 qkvz, 2 sout, 3 the second hc read, 4 the router, 5 the expert stream).
        // KBENCH_PF3_SET: the parts staged (kb_parts; default all). KBENCH_PF3_AT: where layer i's parts are staged:
        //   0 before layer i (default)         1 before layer i-1: a whole layer ahead          2 before layer i-1's second half (its second hc read)
        //   3 before layer i-1's router        4 before layer i-1's expert stream
        //   5 or "split": the mixer parts before layer i-1's second half, the moe parts (and exp) before layer i: each half a layer ahead
        //   6 or "own": every part right before the kernel that reads it (the gain of a part with no question of how long a staged line stays)
        // (layer 0's "layer i-1" is the last layer of the launch before: the launches follow each other)
        static const char* const at_text[] = {"before its layer", "before the layer before (a whole layer ahead)",
            "before the second half (the second hc read) of the layer before", "before the router of the layer before",
            "before the expert stream of the layer before",
            "split: mixer parts before the second half of the layer before, moe parts before their layer",
            "each part right before the kernel that reads it"};
        static const char* const ev_text[] = {"before the first hc read", "before qkvz", "before sout", "before the second hc read", "the router",
            "before the expert stream"};
        std::vector<PfSet> plan((size_t) NLAY * 6);
        int pf3_at = 0, pf3_set = KB_ALL;
        double pf3_mb = 0;
        if (pf == 3) {
            const char* a = std::getenv("KBENCH_PF3_AT");
            const char* p = std::getenv("KBENCH_PF3_SET");
            const std::string as = a ? a : "0";
            pf3_at = as == "split" ? 5 : as == "own" ? 6 : as.size() == 1 && as[0] >= '0' && as[0] <= '6' ? as[0] - '0' : -1;
            pf3_set = kb_parts(p ? p : "all");
            if (pf3_at < 0 || pf3_set < 0) {
                std::printf("chain: KBENCH_PF3_AT 0..6, split or own; KBENCH_PF3_SET a number 0..63 or names joined by '+' (hc0 qkvz sout hc1 router exp, "
                            "hc dense mixer moe all)\n");
                return 2;
            }
            for (int j = 0; j < NLAY; ++j) {
                PfSet* at = &plan[(size_t) j * 6];
                switch (pf3_at) {
                case 0: at[0] = mk_set(j, {{pf3_set, 0}}); break;
                case 1: at[0] = mk_set(j, {{pf3_set, 1}}); break;
                case 2: at[3] = mk_set(j, {{pf3_set, 1}}); break;
                case 3: at[4] = mk_set(j, {{pf3_set, 1}}); break;
                case 4: at[5] = mk_set(j, {{pf3_set, 1}}); break;
                case 5:
                    at[0] = mk_set(j, {{pf3_set & (KB_MOE | KB_EXP), 0}});
                    at[3] = mk_set(j, {{pf3_set & KB_MIXER, 1}});
                    break;
                default:
                    for (int k = 0; k < 6; ++k) at[k] = mk_set(j, {{pf3_set & (1 << k), 0}});
                }
            }
            for (const PfSet& s : plan) pf3_mb += set_mb_of(s) / NLAY;
            if (pf3_mb <= 0) {
                std::printf("chain: KBENCH_PF=3 with nothing to stage (KBENCH_PF3_SET %s at KBENCH_PF3_AT=%d)\n", kb_parts_text(pf3_set).c_str(), pf3_at);
                return 2;
            }
        }
        // KBENCH_PF=2, the persistent prefetcher's schedule. An event is one more of the count the prefetcher follows.
        // KBENCH_PF_EVENTS: the events of a layer, bit k: before kernel k of the layer (k as above). Bits 0 and 3 are the hc reads' own count and
        // bit 4 the router's (default 16: the router only), as the engine's kernels carry them; every other bit is a kernel of one thread on the
        // chain's stream before kernel k (kb_tick).
        // KBENCH_PF_ORDER: what each event of a layer stages, events in the layer's order, separated by '/': a list "parts:off,parts:off" (kb_parts
        // names; off 0: of the event's layer, 1: of the next layer). Default "all:1": the router of layer i stages layer i+1's whole set (as before).
        // "split" = KBENCH_PF_EVENTS=9 with "moe:0/mixer:1": before a layer's first hc read its moe parts, before its second hc read the next
        // layer's mixer parts (each half a layer ahead). "none" stages nothing (events only: what the event kernels cost).
        // KBENCH_PF_CAP_MB caps every entry. KBENCH_PF_WRAP (default 0), as the engine's FNX_MALL_PF_WRAP: 1: a part of a layer past the last one
        // is layer 0's, staged for the next launch; 0: it is left out. (The last event's entry is always staged: the prefetcher's entry 0.)
        // KBENCH_PF_ROUTER_LATE=1: the router's event (bit 4) is added when its logits are written, not at its start (the engine's moe event).
        // KBENCH_PF_GATE=1 (PfRun::gate): a prefetcher block looks at the count before every KBENCH_PF_CHUNK_KB (default 1024) KiB and leaves its
        // entry when a later event has run. With the router's late event staging nothing (KBENCH_PF_EVENTS=17 KBENCH_PF_ROUTER_LATE=1
        // KBENCH_PF_ORDER=moe:0,mixer:1/none) the prefetcher stops its loads before the expert stream, as the engine's FNX_MALL_PF_GATE=1 (a block
        // that looked at the count just before the event still loads its piece: up to two pieces a block).
        int pf_events = 16;
        std::vector<std::vector<Item>> pf_order;
        std::vector<std::string> pf_order_text;
        const char* const pf_ev_s = std::getenv("KBENCH_PF_EVENTS");
        const char* const pf_od_s = std::getenv("KBENCH_PF_ORDER");
        if (pf == 2) {
            std::string order = pf_od_s ? pf_od_s : "";
            if (pf_ev_s) pf_events = std::atoi(pf_ev_s);
            if (order == "split") {
                if (pf_ev_s && pf_events != 9) {
                    std::printf("chain: KBENCH_PF_ORDER=split is KBENCH_PF_EVENTS=9\n");
                    return 2;
                }
                pf_events = 9, order = "moe:0/mixer:1";
            }
            if (order.empty() && pf_events == 16) order = "all:1";
            bool ok = pf_events >= 1 && pf_events <= 63 && !order.empty();
            if (ok) {
                pf_order_text = kb_split(order, '/');
                for (const std::string& ent : pf_order_text) {
                    std::vector<Item> items;
                    for (const std::string& it : kb_split(ent, ',')) {
                        const size_t c = it.find(':');
                        const int parts = kb_parts(it.substr(0, c)), off = c == std::string::npos ? 0 : std::atoi(it.c_str() + c + 1);
                        if (parts < 0 || off < 0 || off >= NLAY) ok = false;
                        if (parts > 0) items.push_back({parts, off});
                    }
                    pf_order.push_back(items);
                }
                if ((int) pf_order.size() != __builtin_popcount((unsigned) pf_events)) ok = false;
            }
            if (!ok) {
                std::printf("chain: KBENCH_PF_EVENTS 1..63 (bit k: an event before kernel k of the layer; 16: the router's), and KBENCH_PF_ORDER with one "
                            "entry an event, '/' between them: parts:off,parts:off (off 0 the event's layer, 1 the next), or split\n");
                return 2;
            }
            if ((pf_events & 16) && !with_router) {
                std::printf("chain: the router's event needs the router (KBENCH_CHAIN_ROUTER=1)\n");
                return 2;
            }
        }
        const int pf_ne = pf == 2 ? __builtin_popcount((unsigned) pf_events) : 1, pf_n = pf_ne * NLAY;   // events a layer, a launch
        // the persistent prefetcher's sets and words on the card, and the count the next launch's first event will find
        PfSet* pf_table = dev<PfSet>((size_t) pf_n);
        unsigned* pf_words = dev<unsigned>(MALL_PF_WORDS);
        unsigned pf_count = 0;
        unsigned long long pf_runs = 0;
        const int pf_wrap = env_n("KBENCH_PF_WRAP", 0), pf_late = env_n("KBENCH_PF_ROUTER_LATE", 0) ? 1 : 0;
        (void) pf_late;
        const int pf_chunk_kb = env_n("KBENCH_PF_CHUNK_KB", 1024), pf_gate = env_n("KBENCH_PF_GATE", 0) ? pf_chunk_kb << 10 : 0;
        if (pf == 2 && (pf_chunk_kb < 64 || pf_chunk_kb > 16384 || pf_chunk_kb % 4)) {
            std::printf("chain: KBENCH_PF_CHUNK_KB 64..16384, a multiple of 4\n");
            return 2;
        }
        const long long pf_ticks = pf == 2 ? mall_pf_ticks((double) env_n("KBENCH_PF_TIMEOUT_MS", 500)) : 0;
        std::vector<PfSet> pf_tab((size_t) pf_n);
        if (pf == 2) {
            // after p events of a launch the prefetcher stages entry p (p = the launch's events: entry 0, when wrap): event e's set is entry e + 1
            for (int i = 0, e = 0; i < NLAY; ++i)
                for (int k = 0; k < pf_ne; ++k, ++e) pf_tab[(size_t) ((e + 1) % pf_n)] = mk_set(i, pf_order[(size_t) k], true, pf_wrap != 0);
            mall_pf_table(pf_table, pf_tab.data(), pf_n);
            // the engine's check at load: the hc read's grids in use, probed again beside a resident idle prefetcher (ended by the stop word)
            // (not with the ticket queue, FNX_MM_HC=1: it has no grid to be resident)
            PfProbeInfo pi;
            const int bad = mm_hc() ? 0 : mall_pf_hc_probe(hcs, pf_table, pf_n, pf_words, wgs, st3, st, &pi, pf_query);
            std::printf("chain: side stream %s; beside %d resident prefetch workgroups:%s\n",
                        pf_own ? "with a hardware queue of its own" : "a default stream (it can share a hardware queue)", wgs,
                        mm_hc() ? " the hc read is the ticket queue: no grid to probe" : mall_pf_probe_text(bad, pi, wgs).c_str());
            // the engine turns the prefetcher off then. Here: stop, as every hc read of the chain would wait for its barrier's time-out
            // (KBENCH_PF_FORCE=1 runs the chain all the same, to see the stall and err)
            if (bad && !env_n("KBENCH_PF_FORCE", 0)) return 3;
        }
        // the shared-weights run's staging: layer 0's set, a word a 128-byte line, whatever KBENCH_PF_LINE / KBENCH_PF_FULL say
        const bool warm_stage = warm && mall_pf_built(), warm_re = warm_stage && env_n("KBENCH_WARM_RESTAGE", 1) != 0;
        PfSet warm_set = sets[0];
        warm_set.line = 128, warm_set.full = 0;
        const auto pf_launch = [&] {
            if (warm_re) mall_pf(warm_set, 4, st3);   // beside the launch, nothing waits for it (4 workgroups, as a prefetcher holds)
            if (pf != 2) return;
            PfRun r;
            r.sets = pf_table, r.words = pf_words, r.n = pf_n, r.wrap = 1, r.base = pf_count, r.gate = pf_gate, r.timeout = pf_ticks;
            mall_pf_run(r, wgs, st3, pf_query);
            pf_count += (unsigned) pf_n;
            ++pf_runs;
        };
        const auto reset = [&] {
            ck(cudaMemcpy(R, R0, (size_t) MAXT * D * 4, cudaMemcpyDeviceToDevice), "R");
            ck(cudaMemset(inj_a, 0, MAXT * HC * 4), "inj");
            ck(cudaMemset(inj_b, 0, MAXT * HC * 4), "inj");
        };
        const double emb = (double) erows * N * (1.0 + 2.0 / 32) / 1e6;
        std::printf("chain: %d layers, fixed weights %.1f MB a layer (staged set%s), expert stream %.1f MB a layer, prefetch %s (%d workgroups), router %s%s\n", NLAY,
                    set_mb, with_router ? ", with the router" : ", no router", emb,
                    pf == 3 ? (pf3_only ? "staging only, on the main stream" : "staged on the main stream")
                    : pf == 2 ? "by one persistent kernel a graph launch" : pf ? "by a fork a layer in the graph" : "off", wgs,
                    !with_router ? "off" : krt == 2 ? "bf16, a warp a row" : krt == 1 ? "bf16" : "f32",
                    !warm ? ""
                    : warm_re ? "; KBENCH_CHAIN_WARM=1: every layer reads layer 0's fixed weights, staged before each pass and beside every launch"
                    : warm_stage ? "; KBENCH_CHAIN_WARM=1: every layer reads layer 0's fixed weights, staged before each pass"
                    : "; KBENCH_CHAIN_WARM=1: every layer reads layer 0's fixed weights, NOT staged (no mall_pf in this build)");
        if (pf >= 2)
            std::printf("chain: a layer's parts, MB: hc0 %.1f, qkvz %.1f, sout %.1f, hc1 %.1f, router %.1f (mixer %.1f, moe %.1f, all %.1f); exp %.1f (the "
                        "expert stream; the chain has no shared expert)\n", part_mb(KB_HC0), part_mb(KB_QKVZ), part_mb(KB_SOUT), part_mb(KB_HC1),
                        part_mb(KB_ROUTER), part_mb(KB_MIXER), part_mb(KB_MOE), part_mb(KB_ALL), part_mb(KB_EXP));
        if (pf == 3) {
            std::printf("chain: staged: %s, %.1f MB a layer, %s", kb_parts_text(pf3_set).c_str(), pf3_mb, at_text[pf3_at]);
            if (cap_b) std::printf("; a cap of %.1f MB a staging launch", cap_b / 1e6);
            std::printf("\n");
            // the staging-only graph has nothing but the staged parts in the cache: when all layers' parts fit, it reads warm lines
            if (pf3_mb * NLAY < 96.0)
                std::printf("chain: note: the staged parts of all %d layers are %.0f MB, under or near the 64 MB cache: the staging-only graph finds them "
                            "warm, so \"staging alone\" is a lower bound and \"the layers\" an upper bound\n", NLAY, pf3_mb * NLAY);
        }
        if (pf == 2) {
            std::printf("chain: prefetcher events a layer:");
            for (int k = 0, e = 0; k < 6; ++k)
                if (pf_events >> k & 1) {
                    std::printf("%s %s -> %s (%.1f MB)", e ? ";" : "", ev_text[k], pf_order_text[(size_t) e].c_str(), set_mb_of(pf_tab[(size_t) ((e + 1) % pf_n)]));
                    ++e;
                }
            if (cap_b) std::printf("; a cap of %.1f MB an entry", cap_b / 1e6);
            std::printf("; wrap %d; gate %s%s\n", pf_wrap, pf_gate ? "on" : "off", (pf_events & 16) && pf_late ? "; the router's event when its logits are written" : "");
        }
        // the chain's graph for T rows: stage: the plan's staging launches on the main stream (KBENCH_PF=3); layers false: those launches only
        const auto chain_us = [&](int T, bool stage, bool layers) {
            return graph_us(st, NLAY, RUNS, [&] {
                bool open = false;
                for (int i = 0; i < NLAY; ++i) {
                    const Lay& l = lay[(size_t) i];
                    // before kernel k of the layer: the plan's staging launch (KBENCH_PF=3), the prefetcher's event (KBENCH_PF=2). The hc reads and
                    // the router carry their event themselves, as in the engine; the other points take a kernel of one thread (kb_tick)
                    const auto before = [&](int k) {
                        if (stage && plan[(size_t) i * 6 + k].n) mall_pf(plan[(size_t) i * 6 + k], wgs, st);
                        if (layers && pf == 2 && k != 0 && k != 3 && k != 4 && (pf_events >> k & 1)) kb_tick(pf_words, st);
                    };
#if FNX_MALL_PF
#define KB_HC_EV(k) , pf == 2 && (pf_events >> (k) & 1) ? pf_words : nullptr
#else
#define KB_HC_EV(k)
#endif
                    before(0);
                    if (layers) hc_read(R, i ? bo : nullptr, inj_b, w_norm, l.down[0], l.up[0], w_inj, 1e-6f, mixed, inj_a, hcs, err, T, st, xq KB_HC_EV(0));
                    before(1);
                    if (layers) q8s_gemv(l.qkvz, qrows, N, xq, qout[(size_t) i], qrows, T, st);
                    before(2);
                    if (layers) q8s_gemv(l.sout, N, ocols, yo, sout[(size_t) i], N, T, st);
                    before(3);
                    if (layers) hc_read(R, bo, inj_a, w_norm, l.down[1], l.up[1], w_inj, 1e-6f, mixed, inj_b, hcs, err, T, st, xq KB_HC_EV(3));
#undef KB_HC_EV
                    if (layers && pf == 1 && i + 1 < NLAY) {
                        ck(cudaEventRecord(ev_fork, st), "fork");
                        ck(cudaStreamWaitEvent(st3, ev_fork, 0), "fork wait");
                        mall_pf(sets[(size_t) i + 1], wgs, st3);
                        open = true;
                    }
                    before(4);
                    if (layers && with_router)
                        router_route(l.router.f, mixed, l.logits, T, topi, topw, jobs_gu, jobs_dn, njobs, st, nullptr, -1, nullptr, nullptr, 0, nullptr, nullptr,
                                     l.router.b
#if FNX_MALL_PF
                                     ,
                                     pf == 2 && (pf_events & 16) ? pf_words : nullptr, pf_late
#endif
                        );
                    before(5);
                    if (layers) q8s_gemv(l.exp, erows, N, xq, eout[(size_t) i], erows, T, st);
                }
                if (open) {
                    ck(cudaEventRecord(ev_join, st3), "join");
                    ck(cudaStreamWaitEvent(st, ev_join, 0), "join wait");
                }
            }, pf_launch);
        };
        for (int T : Ts) {
            double best = 1e30, best_stage = 1e30;
            for (int pass = 0; pass < 3; ++pass) {
                reset();
                if (warm_stage) {   // layer 0's set into the cache with plain loads (the chain's own loads may be non-temporal)
                    mall_pf(warm_set, 64, st);
                    ck(cudaStreamSynchronize(st), "warm");
                }
                if (!pf3_only) best = std::min(best, chain_us(T, pf == 3, true));
                // KBENCH_PF=3: the staging launches alone (they write nothing: the outputs stay the full chain's)
                if (pf == 3) best_stage = std::min(best_stage, chain_us(T, true, false));
                if (pf == 2 || warm_re) ck(cudaStreamSynchronize(st3), "prefetch sync");
            }
            if (pf3_only) {
                std::printf("chain T=%2d  staging only %7.2f us a layer (%.2f ms for %d), %.1f MB a layer, %d workgroups\n", T, best_stage,
                            best_stage * NLAY / 1e3, NLAY, pf3_mb, wgs);
                continue;
            }
            unsigned e = 0;
            ck(cudaMemcpy(&e, err, 4, cudaMemcpyDeviceToHost), "err");
            unsigned long long h = fnv(R, (size_t) T * D * 4);
            h = fnv(mixed, (size_t) T * N * 4, h);
            for (int i = 0; i < NLAY; ++i) {
                h = fnv(qout[(size_t) i], (size_t) T * qrows * 4, h);
                h = fnv(sout[(size_t) i], (size_t) T * N * 4, h);
                h = fnv(eout[(size_t) i], (size_t) T * erows * 4, h);
            }
            if (with_router) {   // every layer's logits, the last layer's picks and weights (the hash of the engine's chain)
                for (int i = 0; i < NLAY; ++i) h = fnv(lay[(size_t) i].logits, (size_t) T * (NEXP + 1) * 4, h);
                h = fnv(topi, (size_t) T * TOPK * 4, h);
                h = fnv(topw, (size_t) T * TOPK * 4, h);
            }
            std::printf("chain T=%2d  %7.2f us a layer (%.2f ms for %d)  hc grid %d  err %u  hash %016llx", T, best, best * NLAY / 1e3, NLAY, hc_grid_for(T), e, h);
            if (pf == 2) {   // the prefetcher's status (1: one timed out, 2: a count past its window) and the routers' count against the launches'
                unsigned w[10] = {};
                ck(cudaMemcpy(w, pf_words, sizeof w, cudaMemcpyDeviceToHost), "words");
                std::printf("  prefetcher status %u, count %u of %u", w[1], w[0], pf_count);
                // what the prefetchers did since the start (a block each). Of the sets with something to stage: staged to their end
                // (words[5] less the empty ones, [9]), left at a later event (the gate, [8]). Of all sets: left out because the block fell
                // behind ([7]). Blocks that started after their window's end ([6])
                int full = 0;
                for (const PfSet& ps : pf_tab) full += ps.n > 0;
                const double some = std::max(1.0, (double) pf_runs * full * wgs), all = std::max(1.0, (double) pf_runs * pf_n * wgs);
                std::printf("; %llu prefetchers: %.1f %% of the sets staged to the end, %.1f %% left at a later event, %.1f %% of all sets left out, %.1f %% "
                            "of the blocks started at the window's end or later", pf_runs, 100.0 * (w[5] - w[9]) / some, 100.0 * w[8] / some,
                            100.0 * w[7] / all, pf_runs ? 100.0 * w[6] / ((double) pf_runs * wgs) : 0.0);
            }
            if (pf == 3)   // the layers' own time beside the plan's staging launches
                std::printf("  staging alone %.2f, the layers %.2f us a layer (%.1f MB a layer staged, %d workgroups)", best_stage, best - best_stage, pf3_mb,
                            wgs);
            std::printf("\n");
        }
        return 0;
    }
    // resid: the card's residency by registers and shared memory (k_kb_resid). KBENCH_RESID_GRID blocks (default 256), KBENCH_RESID_SPINS polls
    // (default 20000). One line a (K, shared memory): the blocks that ran at once, and the time of the launch
    if (which == "resid") {
        const int grid = env_n("KBENCH_RESID_GRID", 256), spins = env_n("KBENCH_RESID_SPINS", 20000);
        unsigned* w = dev<unsigned>(4);
        const float* src = dev_f32((size_t) 256 * 256, -1.f, 1.f);
        float* sink = dev<float>((size_t) 256 * 256);
        const int lds_kb[] = {1, 4, 8, 12, 16, 20, 24, 28, 31, 32, 40, 48, 60};
        std::printf("resid: grid %d blocks of 256 threads; columns: shared memory KB a block; cells: blocks that ran at once\n   K  ", grid);
        for (int kb : lds_kb) std::printf(" %4d", kb);
        std::printf("\n");
        const auto run = [&](int K, int kb) {
            const unsigned init[4] = {0u, 0xffffffffu, 0u, 0u};
            ck(cudaMemcpy(w, init, sizeof init, cudaMemcpyHostToDevice), "resid words");
            const int bytes = kb * 1024 - (kb >= 60 ? 1024 : 0);
            switch (K) {
                case 32: kb_resid_launch<32>(st, grid, bytes, w, src, sink, spins); break;
                case 56: kb_resid_launch<56>(st, grid, bytes, w, src, sink, spins); break;
                case 72: kb_resid_launch<72>(st, grid, bytes, w, src, sink, spins); break;
                case 88: kb_resid_launch<88>(st, grid, bytes, w, src, sink, spins); break;
                case 104: kb_resid_launch<104>(st, grid, bytes, w, src, sink, spins); break;
                case 120: kb_resid_launch<120>(st, grid, bytes, w, src, sink, spins); break;
                case 136: kb_resid_launch<136>(st, grid, bytes, w, src, sink, spins); break;
                case 152: kb_resid_launch<152>(st, grid, bytes, w, src, sink, spins); break;
                case 176: kb_resid_launch<176>(st, grid, bytes, w, src, sink, spins); break;
                case 200: kb_resid_launch<200>(st, grid, bytes, w, src, sink, spins); break;
                case 224: kb_resid_launch<224>(st, grid, bytes, w, src, sink, spins); break;
                case 244: kb_resid_launch<244>(st, grid, bytes, w, src, sink, spins); break;
                default: throw std::runtime_error("resid: no kernel of that K");
            }
            ck(cudaGetLastError(), "resid launch");
            ck(cudaStreamSynchronize(st), "resid sync");
            unsigned h[4];
            ck(cudaMemcpy(h, w, sizeof h, cudaMemcpyDeviceToHost), "resid words");
            return h[1];
        };
        for (int K : {32, 56, 72, 88, 104, 120, 136, 152, 176, 200, 224, 244}) {
            std::printf("%4d  ", K);
            for (int kb : lds_kb) std::printf(" %4u", run(K, kb));
            std::printf("\n");
            std::fflush(stdout);
        }
        return 0;
    }
    if (which == "all" || which == "dense") {
        struct Shape { const char* name; int rows, cols; };
        const int nk = NKH / NCARD, nv = NVH / NCARD;
        const Shape shapes[] = {{"qkvz", (2 * nk + 2 * nv) * HS, N}, {"ssm_out", N, nv * HS}, {"attn wo", N, NQH / NCARD * HD}, {"head slice", 32768, N}};
        for (const Shape& s : shapes) {
            // KBENCH_WARM=1: one copy of the weights, so they stay in the 64 MB MALL (the upper bound of a prefetch); default: distinct copies (cold)
            const int nw = env_n("KBENCH_WARM", 0) ? 1 : (int) std::min<size_t>(REP, ((size_t) 3 << 30) / ((size_t) s.rows * s.cols));
            const std::vector<Q8S> w = copies(dev_q8s(s.rows, s.cols), s.rows, s.cols, nw);
            float* out = dev<float>((size_t) MAXT * s.rows);
            void* y = dev<char>(q8_1_bytes(s.cols) * MAXT);
            int8_t* yq = dev_i8((size_t) MAXT * s.cols);
            float* yd = dev_f32((size_t) MAXT * s.cols / 32, 0.005f, 0.02f);
            q8_twin(y, q8_1_bytes(s.cols) * MAXT, yq, yd);
            for (int T : Ts) {
                const double us = graph_us(st, REP, RUNS, [&] {
                    for (int i = 0; i < REP; ++i) q8s_gemv(w[(size_t) (i % nw)], s.rows, s.cols, y, out, s.rows, T, st);
                });
                const double mb = (double) s.rows * s.cols * (1.0 + 2.0 / 32) / 1e6;
#if FNX_MM
                // the same launch with the dp4a kernel and with the matrix kernel
                const unsigned long long timed_hash = fnv(out, (size_t) T * s.rows * 4);
                Diff dd;
                {
                    std::vector<float> o[2];
                    for (int m = 0; m < 2; ++m) {
                        mm_force(m, kmm_hc);
                        ck(cudaMemset(out, 0xff, (size_t) MAXT * s.rows * 4), "out");   // (NaN: a value not written shows as 1e30)
                        q8s_gemv(w[0], s.rows, s.cols, y, out, s.rows, T, st);
                        ck(cudaStreamSynchronize(st), "sync");
                        o[m] = fetch(out, (size_t) T * s.rows);
                    }
                    dd.add(o[0], o[1]);
                    mm_force(kmm_dense, kmm_hc);
                }
                std::printf("q8s_gemv %-10s %5dx%-5d T=%2d  %7.2f us a launch  (%.1f MB: %.0f GB/s)  hash %016llx  mm-vs-dp4a abs %.3g rel %.3g (top %.3g)\n",
                            s.name, s.rows, s.cols, T, us, mb, mb / us * 1e3, timed_hash, dd.abs, dd.rel, dd.top);
#else
                std::printf("q8s_gemv %-10s %5dx%-5d T=%2d  %7.2f us a launch  (%.1f MB: %.0f GB/s)  hash %016llx\n", s.name, s.rows, s.cols, T, us, mb,
                            mb / us * 1e3, fnv(out, (size_t) T * s.rows * 4));
#endif
            }
        }
    }
    if (which == "moegu") {
        // moegu (generator; not part of `all`): the routed experts' gate|up alone on Q4_K weights (moe_gate_up without the shared expert's
        // rows; the generator's variant with FNX_TK_MOE_GU_Q4K): a card's slice of KBENCH_MOE_EXPERTS experts (NEXP) with random weights, a graph of REP
        // launches of T * TOPK jobs of one row each, every launch its own experts (a window reads other experts at every layer: the weights come over
        // the bus or from the card's cache as in the engine). Prints us a launch and a hash of one launch's outputs.
        const int card = std::min(std::max(env_n("KBENCH_CARD", NCARD - 1), 0), NCARD - 1);
        const int rows = 2 * (ff_hi(NCARD, card) - ff_lo(NCARD, card)), E = std::max(TOPK * MAXT, std::min(NEXP, env_n("KBENCH_MOE_EXPERTS", NEXP)));
        const size_t row_a = (size_t) N / 256 * 144, sa = (size_t) rows * row_a;
        std::vector<unsigned char> hw((size_t) E * sa);
        for (size_t i = 0; i + 8 <= hw.size(); i += 8) {
            const unsigned long long r = rng();
            std::memcpy(hw.data() + i, &r, 8);
        }
        for (size_t b = 0; b < hw.size(); b += 144) {   // a block's d and dmin: f16 scales (random bits would hold infinities)
            const uint16_t d = f2h(1e-3f + (float) (rng() % 1024) * 1e-6f), dmin = f2h(2e-3f + (float) (rng() % 1024) * 1e-6f);
            std::memcpy(hw.data() + b, &d, 2);
            std::memcpy(hw.data() + b + 2, &dmin, 2);
        }
        unsigned char* wts = dev<unsigned char>(hw.size());
        ck(cudaMemcpy(wts, hw.data(), hw.size(), cudaMemcpyHostToDevice), "copy");
        const float* x = dev_f32((size_t) MAXT * N, -1.f, 1.f);
        quantize_q8_1(x, xq, N, MAXT, st);
        ck(cudaStreamSynchronize(st), "sync");
        DW A;
        A.p = wts, A.type = T_Q4_K, A.n_in = N, A.n_out = rows, A.row_bytes = row_a;
        float* gu = dev<float>((size_t) MAXT * TOPK * rows);
        std::printf("moegu: card %d's slice (%d rows of gate|up an expert), %d experts, %.0f MB\n", card, rows, E, (double) hw.size() / 1e6);
        for (int T : Ts) {
            if (T < 1 || T > MAXT) continue;
            const int nj = T * TOPK;
            std::vector<Job> hj((size_t) REP * nj);
            std::vector<int> perm((size_t) E);
            for (int i = 0; i < E; ++i) perm[(size_t) i] = i;
            for (int l = 0; l < REP; ++l) {
                for (int j = 0; j < nj; ++j) std::swap(perm[(size_t) j], perm[(size_t) j + rng() % (size_t) (E - j)]);
                for (int j = 0; j < nj; ++j) {
                    Job& jb = hj[(size_t) l * nj + j];
                    std::memset(&jb, 0, sizeof jb);
                    jb.w = perm[(size_t) j], jb.n = 1, jb.y[0] = j / TOPK, jb.out[0] = j;
                }
            }
            Job* jobs = dev<Job>(hj.size());
            ck(cudaMemcpy(jobs, hj.data(), hj.size() * sizeof(Job), cudaMemcpyHostToDevice), "copy");
            int* njobs = dev<int>(2);
            const int hn[2] = {nj, 0};
            ck(cudaMemcpy(njobs, hn, sizeof hn, cudaMemcpyHostToDevice), "copy");
            // KBENCH_MOE_STAGE=1 (the preload bound of the expert kernel): every launch's experts staged into the card's cache with plain
            // loads right before it (mall_pf on the main stream, KBENCH_PF_WG workgroups, every 16 bytes), as a prefetcher with a perfect guess and
            // a free bus would leave them. Prints the pair's time, the staging alone, and the difference = the launch on warm weights.
            const bool stage = env_n("KBENCH_MOE_STAGE", 0) != 0 && mall_pf_built() && nj <= MALL_PF_MAXR;
            std::vector<PfSet> psets((size_t) (stage ? REP : 0));
            const int swg = std::max(1, env_n("KBENCH_PF_WG", 8));
            unsigned* ssink = stage ? dev<unsigned>(1) : nullptr;
            for (size_t l = 0; l < psets.size(); ++l) {
                PfSet& ps = psets[l];
                ps.n = nj, ps.line = 128, ps.full = 1, ps.sink = ssink;
                for (int j = 0; j < nj; ++j) ps.r[j] = {wts + (size_t) hj[l * (size_t) nj + j].w * sa, (long long) sa};
            }
            const double us = graph_us(st, REP, RUNS, [&] {
                for (int l = 0; l < REP; ++l) {
                    if (stage) mall_pf(psets[(size_t) l], swg, st);
                    moe_gate_up(A, sa, jobs + (size_t) l * nj, njobs, nj, xq, gu, st);
                }
            });
            const double us_st = stage ? graph_us(st, REP, RUNS, [&] {
                for (int l = 0; l < REP; ++l) mall_pf(psets[(size_t) l], swg, st);
            }) : 0;
            ck(cudaMemset(gu, 0, (size_t) MAXT * TOPK * rows * 4), "memset");
            graph_us(st, 1, 1, [&] { moe_gate_up(A, sa, jobs, njobs, nj, xq, gu, st); });
            if (stage)
                std::printf("moegu T=%2d  %8.2f us a launch with its staging  (%d jobs, %.1f MB)  staging alone %.2f, the launch on warm weights %.2f  hash %016llx\n",
                            T, us, nj, (double) nj * sa / 1e6, us_st, us - us_st, fnv(gu, (size_t) nj * rows * 4));
            else
                std::printf("moegu T=%2d  %8.2f us a launch  (%d jobs, %.1f MB)  hash %016llx\n", T, us, nj, (double) nj * sa / 1e6, fnv(gu, (size_t) nj * rows * 4));
            if (ssink) cudaFree(ssink);
            cudaFree(jobs);
            cudaFree(njobs);
        }
        cudaFree(wts);
        cudaFree(gu);
    }
    if (which == "fuse") {
        // fuse (not part of `all`): the fused product launches against the launches they replace: q, k and v (q8s_gemv_segs against three
        // q8s_gemv) and qkvz + ab (q8s_gemv_f32 against gemv_f32 + q8s_gemv). The outputs must be the same bytes.
        // The weights here come from memory at the bus's rate. That misleads for qkvz + ab at 1-4 rows: the fused launch reads 3-8 us slower than
        // the two launches here, while the engine's windows (weights staged by the side stream) are faster with it.
        mm_force(kmm_dense, kmm_hc);
        const int nk = NKH / NCARD, nv = NVH / NCARD, rz = (2 * nk + 2 * nv) * HS;
        // KBENCH_WARM=1: one copy of every matrix (it stays in the card's 64 MB cache); default: a copy a launch set, as a window's layers (cold)
        const int nw = env_n("KBENCH_WARM", 0) ? 1 : (int) std::min<size_t>(REP, ((size_t) 2 << 30) / ((size_t) rz * N));
        const std::vector<Q8S> wq = copies(dev_q8s(QG_W, N), QG_W, N, nw), wk = copies(dev_q8s(KV_W, N), KV_W, N, nw), wv = copies(dev_q8s(KV_W, N), KV_W, N, nw),
                               wz = copies(dev_q8s(rz, N), rz, N, nw);
        const float* wab = dev_f32((size_t) 2 * nv * N, -0.05f, 0.05f);
        const float* x = dev_f32((size_t) MAXT * N, -2.f, 2.f);
        void* y = dev<char>(q8_1_bytes(N) * MAXT);
        int8_t* yq = dev_i8((size_t) MAXT * N);
        float* yd = dev_f32((size_t) MAXT * N / 32, 0.005f, 0.02f);
        q8_twin(y, q8_1_bytes(N) * MAXT, yq, yd);
        float* o[2][3];
        for (int f = 0; f < 2; ++f) o[f][0] = dev<float>((size_t) MAXT * QG_W), o[f][1] = dev<float>((size_t) MAXT * KV_W), o[f][2] = dev<float>((size_t) MAXT * KV_W);
        float* oz[2] = {dev<float>((size_t) MAXT * rz), dev<float>((size_t) MAXT * rz)};
        float* oab[2] = {dev<float>((size_t) MAXT * 2 * nv), dev<float>((size_t) MAXT * 2 * nv)};
        int fails = 0;
        const bool segs_ok = ATT_HEADS || QG_W + 2 * KV_W >= 8192, f32_ok = 2 * nv <= 160 && rz >= 4096 && rz < 8192;
        std::printf("fuse: q %d + k %d + v %d rows of %d (fused form %s); qkvz %d rows + ab %d rows (fused form %s); %d copies of the weights\n", QG_W, KV_W, KV_W, N,
                    segs_ok ? "there" : "not in this build", rz, 2 * nv, f32_ok ? "there" : "not in this build", nw);
        for (int T : Ts) {
            if (segs_ok) {
                const int rows[3] = {QG_W, KV_W, KV_W};
                const auto sep = [&] {
                    for (int i = 0; i < REP; ++i) {
                        const size_t k = (size_t) (i % nw);
                        q8s_gemv(wq[k], QG_W, N, y, o[0][0], QG_W, T, st);
                        q8s_gemv(wk[k], KV_W, N, y, o[0][1], KV_W, T, st);
                        q8s_gemv(wv[k], KV_W, N, y, o[0][2], KV_W, T, st);
                    }
                };
                const auto one = [&] {
                    for (int i = 0; i < REP; ++i) {
                        const size_t k = (size_t) (i % nw);
                        const Q8S ws[3] = {wq[k], wk[k], wv[k]};
                        q8s_gemv_segs(ws, rows, o[1], rows, 3, N, y, T, st);
                    }
                };
                // each form twice, interleaved: the lower reading of each (a card's clock state moves a reading by several us)
                double us3 = graph_us(st, REP, RUNS, sep), us1 = graph_us(st, REP, RUNS, one);
                us3 = std::min(us3, graph_us(st, REP, RUNS, sep)), us1 = std::min(us1, graph_us(st, REP, RUNS, one));
                bool eq = true;
                for (int k = 0; k < 3; ++k) eq = eq && fnv(o[0][k], (size_t) T * rows[k] * 4) == fnv(o[1][k], (size_t) T * rows[k] * 4);
                std::printf("fuse qkv     T=%2d  three launches %7.2f us, one %7.2f us (%+.2f)  outputs %s\n", T, us3, us1, us1 - us3, eq ? "equal" : "DIFFER");
                fails += !eq;
            }
            if (f32_ok) {
                const auto sep = [&] {
                    for (int i = 0; i < REP; ++i) {
                        gemv_f32(wab, x, oab[0], 2 * nv, N, T, st);
                        q8s_gemv(wz[(size_t) (i % nw)], rz, N, y, oz[0], rz, T, st);
                    }
                };
                const auto one = [&] {
                    for (int i = 0; i < REP; ++i) q8s_gemv_f32(wz[(size_t) (i % nw)], rz, N, y, oz[1], wab, x, oab[1], 2 * nv, T, st);
                };
                const auto alone = [&] {
                    for (int i = 0; i < REP; ++i) q8s_gemv(wz[(size_t) (i % nw)], rz, N, y, oz[0], rz, T, st);
                };
                double us2 = graph_us(st, REP, RUNS, sep), usq = graph_us(st, REP, RUNS, alone);
                us2 = std::min(us2, graph_us(st, REP, RUNS, sep)), usq = std::min(usq, graph_us(st, REP, RUNS, alone));
                const unsigned long long hz = fnv(oz[0], (size_t) T * rz * 4), hab = fnv(oab[0], (size_t) T * 2 * nv * 4);
                ck(cudaMemset(oz[1], 0xff, (size_t) MAXT * rz * 4), "out");
                ck(cudaMemset(oab[1], 0xff, (size_t) MAXT * 2 * nv * 4), "out");
                double us1 = graph_us(st, REP, RUNS, one);
                us1 = std::min(us1, graph_us(st, REP, RUNS, one));
                const bool eq = fnv(oz[1], (size_t) T * rz * 4) == hz && fnv(oab[1], (size_t) T * 2 * nv * 4) == hab;
                std::printf("fuse qkvz+ab T=%2d  qkvz alone %7.2f us, two launches %7.2f us, one %7.2f us (%+.2f)  outputs %s\n", T, usq, us2, us1, us1 - us2,
                            eq ? "equal" : "DIFFER");
                fails += !eq;
            }
        }
        if (fails) {
            std::fprintf(stderr, "kbench fuse: %d comparisons failed\n", fails);
            return 1;
        }
    }
#if FNX_HIP && FNX_MALL_PF
    if (which == "anyord") {
        // anyord (not part of `all`): does a launch without the queue's barrier bit run beside the launch before it? Two chains of ALU
        // launches, A (KBENCH_AO_BA blocks of 256 threads, default 16) and B (KBENCH_AO_BB, default 16), KBENCH_AO_TURNS spin turns each (1000 and
        // 4000); a launch depends on the one before it in its own chain only, and checks that (a word a chain). REP pairs A_i, B_i in forms:
        //   graph        one captured graph, a linear chain (the engine's form today)
        //   graph fork   the same graph with its edges as the data says: B_i after B_(i-1) only, A_(i+1) after A_i and B_i. What the runtime makes of
        //                the fork is its business: with DEBUG_HIP_FORCE_ASYNC_QUEUE=1 DEBUG_HIP_FORCE_GRAPH_QUEUES=1 the second child of a fork has no
        //                barrier bit (hip_graph_internal.hpp, GraphKernelNode::CreateCommand)
        //   eager        the launches sent one by one
        //   eager any    the same, B_i with hipExtAnyOrderLaunch (no barrier bit: it may start while A_i runs)
        // us a pair = a run with one host sync, over RUNS runs; "order errors" = launches that did not see the word of the launch before them.
        const int BA = std::min(4096, std::max(1, env_n("KBENCH_AO_BA", 16))), BB = std::min(4096, std::max(1, env_n("KBENCH_AO_BB", 16)));
        std::vector<int> turns_v;
        if (const int t = env_n("KBENCH_AO_TURNS", 0)) turns_v = {std::max(1, t)};
        else turns_v = {1000, 4000};
        const int every = std::max(1, env_n("KBENCH_AO_EVERY", 1));   // graph fork: only every n-th pair is a fork (the others stay in the chain)
        // KBENCH_AO_WIRE: the fork's edges. 0: as the data says (above). 1 and 2 keep B_i after A_i in the graph and only make it A_i's SECOND edge,
        // which is what the runtime's rule looks at (1: the first edge goes to an empty leaf node; 2: to A_(i+1)): the graph stays a chain
        const int wire = std::min(2, std::max(0, env_n("KBENCH_AO_WIRE", 0)));
        float* oa = dev<float>(4096);
        float* ob = dev<float>(4096);
        unsigned* w = dev<unsigned>(4);   // [0] A's word, [1] B's word, [2] order errors
        std::printf("anyord: a fork every %d pairs, wiring %d\n", every, wire);
        std::printf("anyord: %d pairs a chain, %d runs, A %d blocks, B %d blocks; DEBUG_HIP_FORCE_ASYNC_QUEUE=%s DEBUG_HIP_FORCE_GRAPH_QUEUES=%s "
                    "DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=%s\n", REP, RUNS, BA, BB, std::getenv("DEBUG_HIP_FORCE_ASYNC_QUEUE") ? std::getenv("DEBUG_HIP_FORCE_ASYNC_QUEUE") : "-",
                    std::getenv("DEBUG_HIP_FORCE_GRAPH_QUEUES") ? std::getenv("DEBUG_HIP_FORCE_GRAPH_QUEUES") : "-",
                    std::getenv("DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING") ? std::getenv("DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING") : "-");
        for (int turns : turns_v) {
            double res[4] = {}, host[2] = {};
            unsigned bad[4] = {};
            const auto chain = [&](int any) {
                k_kb_set<<<1, 1, 0, st>>>(w, 0u);
                k_kb_set<<<1, 1, 0, st>>>(w + 1, 0u);
                for (int i = 0; i < REP; ++i) {
                    k_kb_seq<<<BA, 256, 0, st>>>(oa, turns, w, (unsigned) i, w + 2);
                    if (any) hipExtLaunchKernelGGL(k_kb_seq, dim3(BB), dim3(256), 0, st, nullptr, nullptr, hipExtAnyOrderLaunch, ob, turns, w + 1, (unsigned) i, w + 2);
                    else k_kb_seq<<<BB, 256, 0, st>>>(ob, turns, w + 1, (unsigned) i, w + 2);
                }
                ck(cudaGetLastError(), "anyord launch");
            };
            for (int form = 0; form < 4; ++form) {
                ck(cudaMemset(w, 0, 16), "memset");
                if (form < 2) {
                    cudaGraph_t g;
                    cudaGraphExec_t ex;
                    ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed), "begin capture");
                    chain(0);
                    ck(cudaStreamEndCapture(st, &g), "end capture");
                    if (form == 1) {
                        // the chain in its order: the root, then each node's one dependent
                        std::vector<hipGraphNode_t> c;
                        hipGraphNode_t n = nullptr;
                        size_t k = 1;
                        ck(hipGraphGetRootNodes(g, &n, &k), "roots");
                        while (n) {
                            c.push_back(n);
                            hipGraphNode_t d[2] = {};
                            size_t nd = 2;
                            ck(hipGraphNodeGetDependentNodes(n, d, &nd), "dependents");
                            n = nd ? d[0] : nullptr;
                        }
                        if ((int) c.size() != 2 + 2 * REP) {
                            std::fprintf(stderr, "kbench anyord: the captured graph has %zu nodes in its chain, not %d\n", c.size(), 2 + 2 * REP);
                            return 2;
                        }
                        for (int i = 0; i < REP; i += every) {
                            const hipGraphNode_t A = c[(size_t) 2 + 2 * i], B = c[(size_t) 3 + 2 * i], prev = c[(size_t) 1 + 2 * i];
                            ck(hipGraphRemoveDependencies(g, &A, &B, 1), "remove");
                            if (wire == 0) {
                                ck(hipGraphAddDependencies(g, &prev, &B, 1), "add");   // B_i after B_(i-1) (after the second set for i = 0)
                                if (i + 1 < REP) {
                                    const hipGraphNode_t An = c[(size_t) 4 + 2 * i];
                                    ck(hipGraphAddDependencies(g, &A, &An, 1), "add");   // A_(i+1) after A_i (it has B_i from the capture)
                                }
                            } else if (wire == 1) {   // B_i stays after A_i in the graph, as A_i's second edge: the first is an empty leaf node
                                hipGraphNode_t e = nullptr;
                                ck(hipGraphAddEmptyNode(&e, g, &A, 1), "empty node");
                                ck(hipGraphAddDependencies(g, &A, &B, 1), "add");
                            } else {   // B_i stays after A_i, as A_i's second edge: the first goes to A_(i+1) (which has B_i from the capture)
                                if (i + 1 < REP) {
                                    const hipGraphNode_t An = c[(size_t) 4 + 2 * i];
                                    ck(hipGraphAddDependencies(g, &A, &An, 1), "add");
                                }
                                ck(hipGraphAddDependencies(g, &A, &B, 1), "add");
                            }
                        }
                    }
                    ck(cudaGraphInstantiate(&ex, g, nullptr, nullptr, 0), "instantiate");
                    double host_us = 0;   // the host's time inside cudaGraphLaunch
                    const auto run = [&] {
                        const auto h0 = std::chrono::steady_clock::now();
                        ck(cudaGraphLaunch(ex, st), "launch");
                        host_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - h0).count();
                        ck(cudaStreamSynchronize(st), "sync");
                    };
                    for (int i = 0; i < 3; ++i) run();
                    host_us = 0;
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < RUNS; ++i) run();
                    res[form] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / RUNS / REP;
                    host[form] = host_us / RUNS / (2 * REP + 2);
                    cudaGraphExecDestroy(ex);
                    cudaGraphDestroy(g);
                } else {
                    const auto run = [&] {
                        chain(form == 3);
                        ck(cudaStreamSynchronize(st), "sync");
                    };
                    for (int i = 0; i < 3; ++i) run();
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < RUNS; ++i) run();
                    res[form] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / RUNS / REP;
                }
                ck(cudaMemcpy(bad + form, w + 2, 4, cudaMemcpyDeviceToHost), "copy");
            }
            std::printf("anyord %6d turns: graph %7.2f us a pair, graph fork %7.2f, eager %7.2f, eager any %7.2f  (order errors %u %u %u %u); the host in "
                        "cudaGraphLaunch: %.2f / %.2f us a node\n", turns, res[0], res[1], res[2], res[3], bad[0], bad[1], bad[2], bad[3], host[0], host[1]);
        }
        cudaFree(oa);
        cudaFree(ob);
        cudaFree(w);
    }
#endif
    if (which == "ktok") {
        // ktok (not part of `all`): can a launch's dispatch gap hide behind the launch before it? A chain of REP dependent launches
        // of ALU work (KBENCH_TOK_BLOCKS blocks of 256 threads: 10 and 64; KBENCH_TOK_TURNS: 1000, 4000 and 12000) in four forms:
        //   plain: one stream, one graph (a launch waits for the one before it by the queue's order)
        //   1 stream + word: the same with the hand-off word (its cost when it never waits)
        //   2 / 3 streams: launch i on stream i % 2 (3), a graph a stream, launched together: launch i is sent while launch i - 1 runs and
        //   waits on the card for its word
        // us a launch = a run of the graphs with one host sync, over RUNS runs. KBENCH_TOK_SPIN=1: the wait polls without a sleep.
        std::vector<int> blocks_v, turns_v;
        if (const int b = env_n("KBENCH_TOK_BLOCKS", 0)) blocks_v = {std::min(4096, std::max(1, b))};
        else blocks_v = {10, 64};
        if (const int t = env_n("KBENCH_TOK_TURNS", 0)) turns_v = {std::max(1, t)};
        else turns_v = {1000, 4000, 12000};
        const int spin = env_n("KBENCH_TOK_SPIN", 0) ? 1 : 0;
        cudaStream_t ss[3] = {st, nullptr, nullptr};
        for (int k = 1; k < 3; ++k) ck(cudaStreamCreateWithFlags(&ss[k], cudaStreamNonBlocking), "stream");
        float* out = dev<float>(4096);
        unsigned* cnt = dev<unsigned>(2);
        ck(cudaMemset(cnt, 0, 8), "memset");
        std::printf("ktok: %d launches a chain, %d runs%s\n", REP, RUNS, spin ? ", waits without a sleep" : "");
        for (int B : blocks_v)
            for (int turns : turns_v) {
                double res[4] = {};
                for (int form = 0; form < 4; ++form) {
                    const int S = form < 2 ? 1 : form, tok = form ? 1 : 0;
                    cudaGraph_t g[3];
                    cudaGraphExec_t ex[3];
                    for (int k = 0; k < S; ++k) {
                        ck(cudaStreamBeginCapture(ss[k], cudaStreamCaptureModeRelaxed), "begin capture");
                        if (tok && k == 0) k_kb_set<<<1, 1, 0, ss[k]>>>(cnt, 0u);
                        for (int i = k; i < REP; i += S) k_kb_tok<<<B, 256, 0, ss[k]>>>(out, turns, cnt, (unsigned) i * (unsigned) B, tok, spin);
                        ck(cudaStreamEndCapture(ss[k], &g[k]), "end capture");
                        ck(cudaGraphInstantiate(&ex[k], g[k], nullptr, nullptr, 0), "instantiate");
                    }
                    const auto run = [&] {
                        for (int k = 0; k < S; ++k) ck(cudaGraphLaunch(ex[k], ss[k]), "launch");
                        for (int k = 0; k < S; ++k) ck(cudaStreamSynchronize(ss[k]), "sync");
                    };
                    for (int i = 0; i < 3; ++i) run();
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < RUNS; ++i) run();
                    res[form] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / RUNS / REP;
                    for (int k = 0; k < S; ++k) {
                        cudaGraphExecDestroy(ex[k]);
                        cudaGraphDestroy(g[k]);
                    }
                }
                unsigned hc[2] = {};
                ck(cudaMemcpy(hc, cnt, 8, cudaMemcpyDeviceToHost), "copy");
                std::printf("ktok %4d blocks %6d turns: plain %6.2f us a launch, 1 stream + word %6.2f, 2 streams %6.2f, 3 streams %6.2f  (waits given up: %u)\n", B,
                            turns, res[0], res[1], res[2], res[3], hc[1]);
            }
        for (int k = 1; k < 3; ++k) cudaStreamDestroy(ss[k]);
        cudaFree(out);
        cudaFree(cnt);
    }
    if (which == "pfhit") {
        // pfhit (not part of `all`): do reads that hit the card's cache share the path of the reads from memory? Main stream: launches
        // that read KBENCH_HIT_MB (16) with non-temporal loads (64 blocks of 256 threads), either the same range every launch, staged once with
        // plain loads ("hits": nothing fills the cache after that), or a new range of a 1 GiB buffer every launch ("memory"). Side stream: 8
        // workgroups read a 256 MB buffer KBENCH_SIDE_N (12) times with non-temporal loads (memory, no cache fill). Prints each one's rate alone
        // and with the other running.
        const size_t hit_b = (size_t) std::max(1, env_n("KBENCH_HIT_MB", 16)) << 20, big_b = (size_t) 1 << 30, side_b = (size_t) 256 << 20;
        const int side_n = std::max(1, env_n("KBENCH_SIDE_N", 12)), swg = std::max(1, env_n("KBENCH_PF_WG", 8)), mb = std::max(1, env_n("KBENCH_HIT_BLOCKS", 64));
        unsigned char* big = dev<unsigned char>(big_b);
        unsigned char* sbuf = dev<unsigned char>(side_b);
        ck(cudaMemset(big, 1, big_b), "memset");
        ck(cudaMemset(sbuf, 2, side_b), "memset");
        unsigned* sink = dev<unsigned>(2);
        cudaStream_t st3 = nullptr;
        ck(cudaStreamCreateWithFlags(&st3, cudaStreamNonBlocking), "stream");
        const auto clk = [] { return std::chrono::steady_clock::now(); };
        const auto usd = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
        const auto busy = [](cudaStream_t s) { return cudaStreamQuery(s) != cudaSuccess; };
        const auto side_run = [&] {
            for (int k = 0; k < side_n; ++k) k_kb_read<<<swg, 256, 0, st3>>>((const char*) sbuf, side_b, 1, sink + 1);
        };
        const double side_mb = (double) side_b * side_n / 1e6, hit_mb = (double) hit_b / 1e6;
        double side_us = 1e30;
        for (int i = 0; i < 4; ++i) {
            const auto t0 = clk();
            side_run();
            ck(cudaStreamSynchronize(st3), "sync");
            if (i) side_us = std::min(side_us, usd(t0, clk()));
        }
        std::printf("pfhit: side stream (%d workgroups, non-temporal) alone: %.0f MB in %.2f ms = %.0f GB/s; main: %d blocks read %.0f MB a launch\n", swg, side_mb,
                    side_us / 1e3, side_mb * 1e3 / side_us, mb, hit_mb);
        const int slots = (int) (big_b / hit_b);
        for (int kind = 0; kind < 3; ++kind) {   // 0 hits (staged once), 1 memory (a new range a launch), 2 the same range, not staged (non-temporal loads fill nothing)
            cudaGraph_t g;
            cudaGraphExec_t ex;
            ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed), "begin capture");
            for (int l = 0; l < REP; ++l) k_kb_read<<<mb, 256, 0, st>>>((const char*) big + (kind == 1 ? (size_t) ((l * 7 + 3) % slots) * hit_b : 0), hit_b, 1, sink);
            ck(cudaStreamEndCapture(st, &g), "end capture");
            ck(cudaGraphInstantiate(&ex, g, nullptr, nullptr, 0), "instantiate");
            const auto stage = [&] {
                if (kind == 0) k_kb_read<<<64, 256, 0, st>>>((const char*) big, hit_b, 0, sink);   // plain loads: into the cache
            };
            const auto run_main = [&](int n) {
                for (int i = 0; i < n; ++i) ck(cudaGraphLaunch(ex, st), "launch");
            };
            if (kind == 2) {   // push the range out of the cache first: 256 MB of plain loads elsewhere
                k_kb_read<<<64, 256, 0, st>>>((const char*) sbuf, side_b, 0, sink);
            }
            stage();
            run_main(3);
            ck(cudaStreamSynchronize(st), "sync");
            auto t0 = clk();
            run_main(RUNS);
            ck(cudaStreamSynchronize(st), "sync");
            const double g_us = usd(t0, clk()) / RUNS;
            const int n_long = (int) std::min(100000.0, std::ceil(3.0 * side_us / g_us)) + 1;
            double beside_us = 1e30, main_us = 1e30;
            bool whole = true, whole2 = true;
            for (int i = 0; i < 2; ++i) {
                run_main(n_long);
                t0 = clk();
                side_run();
                ck(cudaStreamSynchronize(st3), "sync");
                const double u = usd(t0, clk());
                whole = whole && busy(st);
                ck(cudaStreamSynchronize(st), "sync");
                beside_us = std::min(beside_us, u);
            }
            const int n_main = std::max(1, (int) std::ceil(side_us / g_us));
            for (int i = 0; i < 2; ++i) {
                for (int k = 0; k < 5; ++k) side_run();
                t0 = clk();
                run_main(n_main);
                ck(cudaStreamSynchronize(st), "sync");
                const double u = usd(t0, clk()) / n_main;
                whole2 = whole2 && busy(st3);
                ck(cudaStreamSynchronize(st3), "sync");
                main_us = std::min(main_us, u);
            }
            // (after the side stream's non-temporal reads the staged range is still in the cache if the last line shows the hit rate again)
            t0 = clk();
            run_main(RUNS);
            ck(cudaStreamSynchronize(st), "sync");
            const double g2_us = usd(t0, clk()) / RUNS;
            std::printf("pfhit %-22s main alone %6.2f us a launch = %4.0f GB/s; beside the side stream %6.2f = %4.0f GB/s%s; side stream beside it %4.0f GB/s (%3.0f %% "
                        "of alone%s); main alone again %4.0f GB/s\n",
                        kind == 0 ? "hits (staged once)" : kind == 1 ? "memory (new ranges)" : "same range, not staged", g_us / REP, hit_mb * 1e3 / (g_us / REP),
                        main_us / REP, hit_mb * 1e3 / (main_us / REP), whole2 ? "" : " (side ended first)", side_mb * 1e3 / beside_us, 100.0 * side_us / beside_us,
                        whole ? "" : ", main ended first", hit_mb * 1e3 / (g2_us / REP));
            cudaGraphExecDestroy(ex);
            cudaGraphDestroy(g);
        }
        cudaStreamDestroy(st3);
        cudaFree(big);
        cudaFree(sbuf);
        cudaFree(sink);
    }
    if (which == "pfside") {
        // pfside (not part of `all`): what limits the side stream beside a window, the bus or the compute units. The side stream
        // (mall_pf_stream, as the engine's) stages KBENCH_SIDE_MB (256) of its own bytes KBENCH_SIDE_N (12) times in one launch (mall_pf,
        // KBENCH_PF_WG workgroups (8), every 16 bytes: always from memory, the buffer is 4 caches). The main stream runs launches of one of:
        //   spin: ALU work only, no loads (KBENCH_SPIN_BLOCKS blocks of 256 threads, default 64, 512 and 2048; KBENCH_SPIN_TURNS, 8000)
        //   cold: the expert gate|up, every launch its own experts (bus bound)
        //   warm: the expert gate|up, the same experts in every launch (from the card's cache)
        // Prints the side stream's rate alone and beside each (the main stream still ran at its end: "whole"), and each main launch alone and
        // beside the side stream (which still ran at the main stream's end).
        if (!mall_pf_built()) {
            std::printf("pfside: this build has no prefetch (STRATA_SPLIT_FAST)\n");
            return 3;
        }
        const int card = std::min(std::max(env_n("KBENCH_CARD", NCARD - 1), 0), NCARD - 1);
        const int rows = 2 * (ff_hi(NCARD, card) - ff_lo(NCARD, card)), E = NEXP;
        const size_t row_a = (size_t) N / 256 * 144, sa = (size_t) rows * row_a;
        std::vector<unsigned char> hw((size_t) E * sa);
        for (size_t i = 0; i + 8 <= hw.size(); i += 8) {
            const unsigned long long r = rng();
            std::memcpy(hw.data() + i, &r, 8);
        }
        for (size_t b = 0; b < hw.size(); b += 144) {
            const uint16_t d = f2h(1e-3f + (float) (rng() % 1024) * 1e-6f), dmin = f2h(2e-3f + (float) (rng() % 1024) * 1e-6f);
            std::memcpy(hw.data() + b, &d, 2);
            std::memcpy(hw.data() + b + 2, &dmin, 2);
        }
        unsigned char* wts = dev<unsigned char>(hw.size());
        ck(cudaMemcpy(wts, hw.data(), hw.size(), cudaMemcpyHostToDevice), "copy");
        const float* x = dev_f32((size_t) MAXT * N, -1.f, 1.f);
        quantize_q8_1(x, xq, N, MAXT, st);
        ck(cudaStreamSynchronize(st), "sync");
        DW A;
        A.p = wts, A.type = T_Q4_K, A.n_in = N, A.n_out = rows, A.row_bytes = row_a;
        float* gu = dev<float>((size_t) MAXT * TOPK * rows);
        const size_t side_b = (size_t) std::max(64, env_n("KBENCH_SIDE_MB", 256)) << 20;
        const int side_n = std::min(MALL_PF_MAXR, std::max(1, env_n("KBENCH_SIDE_N", 12))), swg = std::max(1, env_n("KBENCH_PF_WG", 8));
        unsigned char* sbuf = dev<unsigned char>(side_b);
        ck(cudaMemset(sbuf, 1, side_b), "memset");
        unsigned* ssink = dev<unsigned>(1);
        PfSet side;
        side.n = side_n, side.line = 128, side.full = 1, side.sink = ssink;
        for (int k = 0; k < side_n; ++k) side.r[k] = {sbuf, (long long) side_b};
        const double side_mb = (double) side_b * side_n / 1e6;
        bool own = false;
        const cudaStream_t st3 = (cudaStream_t) mall_pf_stream(env_n("KBENCH_PF_PRIO", 1), &own);
        const int turns = std::max(1, env_n("KBENCH_SPIN_TURNS", 8000));
        float* spin_out = dev<float>(8192);
        const auto clk = [] { return std::chrono::steady_clock::now(); };
        const auto usd = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
        const auto busy = [](cudaStream_t s) { return cudaStreamQuery(s) != cudaSuccess; };
        // the side stream alone (the best of 3)
        double side_us = 1e30;
        for (int i = 0; i < 4; ++i) {
            const auto t0 = clk();
            mall_pf(side, swg, st3);
            ck(cudaStreamSynchronize(st3), "sync");
            if (i) side_us = std::min(side_us, usd(t0, clk()));
        }
        std::printf("pfside: card %d, side stream (%s queue, %d workgroups) alone: %.0f MB in %.2f ms = %.0f GB/s\n", card, own ? "own" : "shared", swg, side_mb,
                    side_us / 1e3, side_mb * 1e3 / side_us);
        for (int T : Ts) {
            if (T < 1 || T > MAXT) continue;
            const int nj = T * TOPK;
            std::vector<Job> hj((size_t) REP * nj);
            std::vector<int> perm((size_t) E);
            for (int i = 0; i < E; ++i) perm[(size_t) i] = i;
            for (int l = 0; l < REP; ++l) {
                for (int j = 0; j < nj; ++j) std::swap(perm[(size_t) j], perm[(size_t) j + rng() % (size_t) (E - j)]);
                for (int j = 0; j < nj; ++j) {
                    Job& jb = hj[(size_t) l * nj + j];
                    std::memset(&jb, 0, sizeof jb);
                    jb.w = perm[(size_t) j], jb.n = 1, jb.y[0] = j / TOPK, jb.out[0] = j;
                }
            }
            Job* jobs = dev<Job>(hj.size());
            ck(cudaMemcpy(jobs, hj.data(), hj.size() * sizeof(Job), cudaMemcpyHostToDevice), "copy");
            int* njobs = dev<int>(2);
            const int hn[2] = {nj, 0};
            ck(cudaMemcpy(njobs, hn, sizeof hn, cudaMemcpyHostToDevice), "copy");
            struct Load { std::string name; int kind, blocks; };
            std::vector<Load> loads;
            if (const int sb = env_n("KBENCH_SPIN_BLOCKS", 0)) loads.push_back({"spin " + std::to_string(sb) + " blocks", 0, std::min(8192, std::max(1, sb))});
            else
                for (int sb : {64, 512, 2048}) loads.push_back({"spin " + std::to_string(sb) + " blocks", 0, sb});
            loads.push_back({"gate|up cold", 1, 0});
            loads.push_back({"gate|up warm", 2, 0});
            for (const Load& ld : loads) {
                cudaGraph_t g;
                cudaGraphExec_t ex;
                ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed), "begin capture");
                for (int l = 0; l < REP; ++l) {
                    if (ld.kind == 0) k_kb_spin<<<ld.blocks, 256, 0, st>>>(spin_out, turns);
                    else moe_gate_up(A, sa, jobs + (ld.kind == 1 ? (size_t) l * nj : 0), njobs, nj, xq, gu, st);
                }
                ck(cudaStreamEndCapture(st, &g), "end capture");
                ck(cudaGraphInstantiate(&ex, g, nullptr, nullptr, 0), "instantiate");
                const auto run_main = [&](int n) {
                    for (int i = 0; i < n; ++i) ck(cudaGraphLaunch(ex, st), "launch");
                };
                run_main(3);
                ck(cudaStreamSynchronize(st), "sync");
                // the main launches alone
                auto t0 = clk();
                run_main(RUNS);
                ck(cudaStreamSynchronize(st), "sync");
                const double g_us = usd(t0, clk()) / RUNS;   // us a graph launch
                // the side stream beside the main stream: main launches for 3 times the side stream's time alone, queued first
                const int n_long = (int) std::min(100000.0, std::ceil(3.0 * side_us / g_us)) + 1;
                double beside_us = 1e30;
                bool whole = true;
                for (int i = 0; i < 2; ++i) {
                    run_main(n_long);
                    t0 = clk();
                    mall_pf(side, swg, st3);
                    ck(cudaStreamSynchronize(st3), "sync");
                    const double u = usd(t0, clk());
                    whole = whole && busy(st);
                    ck(cudaStreamSynchronize(st), "sync");
                    beside_us = std::min(beside_us, u);
                }
                // the main launches beside the side stream: side launches for ~4 times the main run's time, queued first
                const int n_main = std::max(1, (int) std::ceil(side_us / g_us)), n_side = 5;
                double main_us = 1e30;
                bool whole2 = true;
                for (int i = 0; i < 2; ++i) {
                    for (int k = 0; k < n_side; ++k) mall_pf(side, swg, st3);
                    t0 = clk();
                    run_main(n_main);
                    ck(cudaStreamSynchronize(st), "sync");
                    const double u = usd(t0, clk()) / n_main;
                    whole2 = whole2 && busy(st3);
                    ck(cudaStreamSynchronize(st3), "sync");
                    main_us = std::min(main_us, u);
                }
                std::printf("pfside T=%2d %-18s main %7.2f us a launch alone, %7.2f beside the side stream (%+5.1f %%%s); side stream beside it %4.0f GB/s (%3.0f %% of "
                            "alone%s)\n",
                            T, ld.name.c_str(), g_us / REP, main_us / REP, 100.0 * (main_us / g_us - 1.0), whole2 ? "" : ", side ended first",
                            side_mb * 1e3 / beside_us, 100.0 * side_us / beside_us, whole ? "" : ", main ended first");
                cudaGraphExecDestroy(ex);
                cudaGraphDestroy(g);
            }
            cudaFree(jobs);
            cudaFree(njobs);
        }
        cudaFree(wts);
        cudaFree(gu);
        cudaFree(sbuf);
        cudaFree(ssink);
        cudaFree(spin_out);
    }
    if (which == "tkgemm") {
#if FNX_HIP
        // tkgemm (generator; not part of `all`): the dense 8-bit GEMM family (gen/tk_triton.py gemm8) launched from C++ through src/tk.hpp, on
        // random int8 or fp8 operands: c[m, n] = sa[m] * sw[n] * sum_k a[m, k] w[n, k]. For every key of the selected variant (FNX_TK_GEMM8): KBENCH_GEMM_M
        // rows (4096) against the key's N x K (key 0, any shape: KBENCH_GEMM_N x KBENCH_GEMM_K, 4096 x 2560). Prints ms a launch, the rate in T(FL)OPS
        // (2 M N K a launch), the largest relative error of 256 sampled outputs against the exact sum on the host, and a hash of the output.
        const tk::Family* f = tk::family("gemm8");
        if (!f) {
            std::fprintf(stderr, "kbench tkgemm: no variant selected (FNX_TK_DIR and FNX_TK_GEMM8)\n");
            return 2;
        }
        const int M = env_n("KBENCH_GEMM_M", 4096), runs = std::max(1, env_n("KBENCH_GEMM_RUNS", 20));
        static const char* const names[] = {"any", "qkvz", "wq", "wkv", "out", "hcdn", "hcup", "shgu", "shdn"};
        const auto fp8_val = [](unsigned char b) {   // e4m3: sign, 4 exponent bits (bias 7), 3 mantissa bits
            const int e = b >> 3 & 15, m = b & 7;
            const float v = e ? std::ldexp(1.f + m / 8.f, e - 7) : std::ldexp(m / 8.f, -6);
            return b & 128 ? -v : v;
        };
        int fails = 0;
        // KBENCH_GEMM_REF=n (0: off): each key's timing is taken n times, and before every one and after the last the engine's f16 product of the same
        // shape runs with the same number of launches (prefill.cu pf_gemm: W f16 [N][K], X f16 [M][K], fp32 sums and output, the BLAS library): lines
        // "tkgemm ref". The key's own line then gives the mean of its n timings. A 20-launch burst ends before the card's power cap acts: for the
        // rate the engine sees in a long read set KBENCH_GEMM_RUNS=2000.
        const int refs = env_n("KBENCH_GEMM_REF", 0);
        cublasHandle_t blas = nullptr;
        if (refs > 0 && (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas, st) != CUBLAS_STATUS_SUCCESS)) {
            std::fprintf(stderr, "kbench tkgemm: no BLAS handle for the reference\n");
            return 2;
        }
        for (const auto& kv : f->kerns) {
            const tk::Kern& k = *kv.second;
            const int BM = k.meta[0], BN = k.meta[1], fp8 = k.meta[3], out16 = k.meta[4], wt = k.meta[7] >> 1 & 1, ep = k.meta[7] >> 2 & 1;   // wt: the weights as [K][N]; ep: the scales applied
            const int mm = k.meta[7] & 1, nm = k.meta[7] >> 3 & 1;   // the kernel masks the rows past M / the columns past N
            const int at = k.meta[7] >> 4 & 1;   // built with the pointer facts: 16-byte aligned pointers (hipMalloc's are), every operand under 2^31 bytes
            const int N = k.meta[5] ? k.meta[5] : env_n("KBENCH_GEMM_N", 4096), K = k.meta[6] ? k.meta[6] : env_n("KBENCH_GEMM_K", 2560);
            // the kernel's contract: whole K tiles; whole M / N tiles where it does not mask them (its loads would run past the buffers); every
            // operand under 2^31 elements with a tile of slack (the kernel's element offsets are 32-bit)
            const long long lim = (1ll << 31) - 1;
            if (M < 1 || N < 1 || K < 1 || K % k.meta[2] || (!mm && M % BM) || (!nm && N % BN) || ((long long) M + BM) * K > lim || ((long long) N + BN) * K > lim ||
                ((long long) M + BM) * ((long long) N + BN) * (at ? (out16 ? 2 : 4) : 1) > lim) {
                std::fprintf(stderr, "kbench tkgemm: key %d (tiles %dx%dx%d, row mask %d, column mask %d) does not take M %d N %d K %d\n", k.key, BM, BN, k.meta[2], mm, nm, M, N, K);
                return 2;
            }
            rng.seed(12345 + (unsigned) k.key);   // a key's operands are the same for every variant (their hashes compare)
            std::vector<unsigned char> ha((size_t) M * K), hw((size_t) N * K);   // hw: [N][K]
            for (auto* v : {&ha, &hw})
                for (auto& x : *v) x = fp8 ? (unsigned char) ((rng() & 128) | ((4 + rng() % 5) << 3) | (rng() & 7)) : (unsigned char) (int8_t) ((int) (rng() % 255) - 127);
            std::vector<float> hsa((size_t) M), hsw((size_t) N);
            for (auto& x : hsa) x = 0.5f + (float) (rng() % 1024) / 1024.f;
            for (auto& x : hsw) x = (0.01f + (float) (rng() % 1024) / 16384.f) * (out16 ? 1e-3f : 1.f);   // (f16 outputs: the products stay under 65504)
            unsigned char* a = dev<unsigned char>(ha.size() + 4096);   // (slack: a tile's rows past M are masked, not read)
            unsigned char* w = dev<unsigned char>(hw.size() + 4096);
            float* sa = dev<float>(hsa.size());
            float* sw = dev<float>(hsw.size());
            const size_t ob = (size_t) M * N * (out16 ? 2 : 4);
            unsigned char* c = dev<unsigned char>(ob);
            ck(cudaMemcpy(a, ha.data(), ha.size(), cudaMemcpyHostToDevice), "copy");
            if (wt) {
                std::vector<unsigned char> t(hw.size());
                for (int n = 0; n < N; ++n)
                    for (int j = 0; j < K; ++j) t[(size_t) j * N + n] = hw[(size_t) n * K + j];
                ck(cudaMemcpy(w, t.data(), t.size(), cudaMemcpyHostToDevice), "copy");
            } else {
                ck(cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice), "copy");
            }
            ck(cudaMemcpy(sa, hsa.data(), hsa.size() * 4, cudaMemcpyHostToDevice), "copy");
            ck(cudaMemcpy(sw, hsw.data(), hsw.size() * 4, cudaMemcpyHostToDevice), "copy");
            const unsigned grid = (unsigned) (((M + BM - 1) / BM) * ((N + BN - 1) / BN));
            using GemmSig = void(const unsigned char*, const unsigned char*, const float*, const float*, unsigned char*, int, int, int, void*, void*);
            const auto launch = [&] { tk::Sig<GemmSig>::launch(k, dim3(grid), dim3(k.threads), k.smem, st, a, w, sa, sw, c, M, N, K, nullptr, nullptr); };
            const char* const kname = k.key >= 0 && k.key < 9 ? names[k.key] : "?";
            uint16_t *x16 = nullptr, *w16 = nullptr;   // f16 bits
            float* o32 = nullptr;
            if (refs > 0) {   // the reference's operands: the same integers as f16 (x / 64: exact)
                std::vector<uint16_t> hx(ha.size()), hw16(hw.size());
                const auto h16 = [](unsigned char b) {
                    const _Float16 v = (_Float16) ((float) (int8_t) b / 64.f);
                    uint16_t h;
                    std::memcpy(&h, &v, 2);
                    return h;
                };
                for (size_t i = 0; i < ha.size(); ++i) hx[i] = h16(ha[i]);
                for (size_t i = 0; i < hw.size(); ++i) hw16[i] = h16(hw[i]);
                x16 = dev<uint16_t>(hx.size());
                w16 = dev<uint16_t>(hw16.size());
                o32 = dev<float>((size_t) M * N);
                ck(cudaMemcpy(x16, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice), "copy");
                ck(cudaMemcpy(w16, hw16.data(), hw16.size() * 2, cudaMemcpyHostToDevice), "copy");
            }
            const auto ref_run = [&] {
                const float one = 1.0f, zero = 0.0f;
                const auto call = [&] {
                    if (cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, w16, CUDA_R_16F, K, x16, CUDA_R_16F, K, &zero, o32, CUDA_R_32F, N,
                                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS)
                        throw std::runtime_error("kbench tkgemm: the reference GEMM failed");
                };
                for (int i = 0; i < 3; ++i) call();
                ck(cudaStreamSynchronize(st), "sync");
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < runs; ++i) call();
                ck(cudaStreamSynchronize(st), "sync");
                const double rus = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / runs;
                std::printf("tkgemm ref %-5s f16 lib M %d N %d K %d  %8.3f ms  %6.1f TFLOPS\n", kname, M, N, K, rus / 1e3, 2.0 * M * N * K / rus / 1e6);
            };
            double us = 0;
            for (int pass = 0; pass < std::max(1, refs); ++pass) {
                if (refs > 0) ref_run();
                const double pus = graph_us(st, 1, runs, launch);
                if (refs > 0) std::printf("tkgemm pass %d %-5s %8.3f ms  %6.1f %s\n", pass, kname, pus / 1e3, 2.0 * M * N * K / pus / 1e6, fp8 ? "TFLOPS" : "TOPS");
                us += pus / std::max(1, refs);
            }
            if (refs > 0) ref_run();
            std::vector<unsigned char> hc(ob);
            ck(cudaMemcpy(hc.data(), c, ob, cudaMemcpyDeviceToHost), "host copy");
            double worst = 0;
            for (int i = 0; i < 256; ++i) {
                const size_t m = rng() % (size_t) M, n = rng() % (size_t) N;
                double ref = 0;
                for (int j = 0; j < K; ++j) {
                    const unsigned char x = ha[m * K + j], y = hw[n * K + j];
                    ref += fp8 ? (double) fp8_val(x) * fp8_val(y) : (double) (int8_t) x * (int8_t) y;
                }
                if (ep) ref *= (double) hsa[m] * hsw[n];
                double got;
                if (out16) {
                    uint16_t h;
                    std::memcpy(&h, hc.data() + (m * N + n) * 2, 2);
                    const int e = h >> 10 & 31, mant = h & 1023;
                    got = e == 31 ? NAN : (h & 0x8000 ? -1.0 : 1.0) * (e ? std::ldexp(1.0 + mant / 1024.0, e - 15) : std::ldexp(mant / 1024.0, -14));
                } else {
                    float v;
                    std::memcpy(&v, hc.data() + (m * N + n) * 4, 4);
                    got = v;
                }
                const double rel = std::fabs(got - ref) / std::max(std::fabs(ref), 1e-3 * (fp8 ? 1.0 : 1000.0) * (ep ? 1.0 : 100.0));
                worst = std::isfinite(rel) ? std::max(worst, rel) : INFINITY;   // an output that is not finite fails
            }
            const bool ok = worst < (out16 ? 2e-3 : fp8 ? 1e-3 : 2e-6);
            fails += !ok;
            std::printf("tkgemm key %d %-5s %s%s M %d N %d K %d tiles %dx%dx%d  %8.3f ms  %6.1f %s  max rel err %.2e%s  hash %016llx\n", k.key,
                        k.key >= 0 && k.key < 9 ? names[k.key] : "?", fp8 ? "fp8" : "int8", wt ? " W[K][N]" : "", M, N, K, BM, BN, k.meta[2], us / 1e3,
                        2.0 * M * N * K / us / 1e6, fp8 ? "TFLOPS" : "TOPS", worst, ok ? "" : "  TOO LARGE", fnv(c, ob));
            for (void* p : {(void*) a, (void*) w, (void*) sa, (void*) sw, (void*) c, (void*) x16, (void*) w16, (void*) o32}) cudaFree(p);
        }
        if (blas) cublasDestroy(blas);
        if (fails) return 1;
#else
        std::printf("tkgemm: HIP only\n");
#endif
    }
    if (which == "pfattn") {
#if FNX_ATT_MM && FNX_HIP
        // pfattn (generator; not part of `all`): the prompt attention's kernel alone (attn_prompt: the union walk and its lists' builder, or the
        // generator's variant of them, FNX_TK_PF_ATTN_U) on one card: a chunk of KBENCH_PF_C rows (4096) that ends at each depth of KBENCH_PF_DEPTHS
        // (8192,65536,109000), over KBENCH_LAYERS (4) layers' K/V. A row past 2048 positions attends 512 blocks and its tail; the rows of a group of 16
        // share KBENCH_PF_OVERLAP percent (75) of their blocks (neighbouring rows select nearly the same blocks; the walk's cost depends on it).
        // Prints us a launch and a hash of the partials: two variants compare for speed and for equal bits on the same inputs.
        const int C = env_n("KBENCH_PF_C", 4096), L = std::max(1, std::min(8, env_n("KBENCH_LAYERS", 4)));
        const int ov = std::max(0, std::min(100, env_n("KBENCH_PF_OVERLAP", 75))), runs = std::max(1, env_n("KBENCH_PF_RUNS", 10));
        std::vector<int> depths;
        {
            const char* dv = std::getenv("KBENCH_PF_DEPTHS");
            const std::string ds = dv ? dv : "8192,65536,109000";
            for (size_t i = 0; i < ds.size();) {
                const size_t j = ds.find(',', i);
                depths.push_back(std::atoi(ds.substr(i, j - i).c_str()));
                if (j == std::string::npos) break;
                i = j + 1;
            }
        }
        const int maxd = depths.empty() ? 0 : *std::max_element(depths.begin(), depths.end());
        if (C < 1 || depths.empty() || *std::min_element(depths.begin(), depths.end()) < C) {
            std::fprintf(stderr, "kbench pfattn: every depth of KBENCH_PF_DEPTHS at least KBENCH_PF_C (%d)\n", C);
            return 2;
        }
        const KvPool pool = kv_pool(maxd / kvp_rows(0) + 1, L);
        const int maxsel = qsa_maxsel();
        const float* qg = dev_f32((size_t) C * QG_W, -2.f, 2.f);
        const size_t pf = attn_prompt_block_floats(C) * NCARD, ui = attn_prompt_union_ints(C);
        float* part = dev<float>(pf);
        int* uscr = ui ? dev<int>(ui) : nullptr;
        int* cells = dev<int>((size_t) C * maxsel);
        int* ncells = dev<int>((size_t) C);
        int* pos0p = dev<int>(1);
        std::printf("pfattn: chunks of %d rows, groups of %d rows in the walk, %d layers' K/V, rows of a 16-row group share %d %% of their blocks\n", C,
                    attn_prompt_union_rows(), L, ov);
        for (int depth : depths) {
            const int pos0 = depth - C;
            std::vector<int> hc((size_t) C * maxsel, 0), hn((size_t) C), common;
            std::vector<char> in;
            for (int t = 0; t < C; ++t) {
                const int p = pos0 + t, nv = (p + 1) / 4, n_tail = (p + 1) % 4;
                if (nv <= 512) {
                    hn[(size_t) t] = -1;
                    continue;
                }
                if (t % 16 == 0 || common.empty()) {   // the group's common blocks: 512 of the first row's
                    std::vector<int> all((size_t) nv);
                    for (int j = 0; j < nv; ++j) all[(size_t) j] = j;
                    for (int j = 0; j < 512; ++j) std::swap(all[(size_t) j], all[(size_t) j + rng() % (size_t) (nv - j)]);
                    common.assign(all.begin(), all.begin() + 512);
                }
                std::vector<int> blocks = common;
                in.assign((size_t) nv, 0);
                for (int b : blocks) in[(size_t) b] = 1;
                for (int j = 0, nrep = 512 * (100 - ov) / 100; j < nrep && nv > 512; ++j) {   // the row's own blocks in place of some common ones
                    const size_t at = rng() % 512;
                    int nb = (int) (rng() % (size_t) nv);
                    while (in[(size_t) nb]) nb = (nb + 1) % nv;
                    in[(size_t) blocks[at]] = 0, in[(size_t) nb] = 1;
                    blocks[at] = nb;
                }
                std::sort(blocks.begin(), blocks.end());
                int n = 0;
                int* row = hc.data() + (size_t) t * maxsel;
                for (int j = 0; j < 512; ++j)
                    for (int m = 0; m < 4; ++m) row[n++] = blocks[(size_t) j] * 4 + m;
                for (int m = 0; m < n_tail; ++m) row[n++] = (p / 4) * 4 + m;
                hn[(size_t) t] = n;
            }
            ck(cudaMemcpy(cells, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice), "copy");
            ck(cudaMemcpy(ncells, hn.data(), hn.size() * 4, cudaMemcpyHostToDevice), "copy");
            ck(cudaMemcpy(pos0p, &pos0, 4, cudaMemcpyHostToDevice), "copy");
            const double us = graph_us(st, L, runs, [&] {
                for (int i = 0; i < L; ++i) attn_prompt(qg, pool.kc(i), pool.vc(i), pool.g, pos0p, cells, ncells, 0, C, 0, C, part, st, uscr);
            });
            ck(cudaMemset(part, 0, pf * 4), "memset");
            graph_us(st, 1, 1, [&] { attn_prompt(qg, pool.kc(0), pool.vc(0), pool.g, pos0p, cells, ncells, 0, C, 0, C, part, st, uscr); });
            std::printf("pfattn depth %6d  %9.2f us a launch  hash %016llx\n", depth, us, fnv(part, pf * 4));
        }
#else
        std::printf("pfattn: not in this build (needs STRATA_SPLIT_APPROX on HIP)\n");
#endif
    }
    if (which == "attn" || which == "qsasel") {
#if FNX_ATT_MM
        const int ctx = env_n("KBENCH_CTX", 109000), L = std::max(1, std::min(REP, env_n("KBENCH_LAYERS", 12))), only = env_n("KBENCH_ATT_MM", -1);
        const int Tmax = *std::max_element(Ts.begin(), Ts.end());
        if (ctx < 64 || Tmax > MAXT || *std::min_element(Ts.begin(), Ts.end()) < 1) {
            std::fprintf(stderr, "kbench %s: KBENCH_CTX >= 64 and rows 1..%d\n", which.c_str(), MAXT);
            return 2;
        }
        int fails = 0;   // comparisons that failed: exit 1
        std::vector<int> modes;
        if (only != 1) modes.push_back(0);
        if (only != 0) modes.push_back(1);
        const KvPool pool = kv_pool(ctx / kvp_rows(0) + 1, L);
        int* pos0p = dev<int>(1);
        const int maxsel = qsa_maxsel();
        std::printf("%s: %d positions, %d layers' K/V at addresses of their own (%.0f MB each)\n", which.c_str(), ctx, L,
                    (double) (ctx / kvp_rows(0) + 1) * pool.g.page * 2 / 1e6);
        if (which == "attn") {
            const float* qg = dev_f32((size_t) Tmax * QG_W, -2.f, 2.f);
            float* scratch = dev<float>(attn_scratch_floats(NQH_LOC));
            const size_t on = (size_t) Tmax * NQH_LOC * HD;
            float* out[2] = {dev<float>(on), dev<float>(on)};
            float* out_l = dev<float>(on);
            // the output as q8_1 rows with their twins: [0] by attn itself, [1] by a quantize after it (the reference), [2] by attn_finish
            const int nqv = NQH_LOC * HD;
            const size_t aq_b = q8_1_bytes(nqv) * (size_t) Tmax;
            void* aq[3];
            int8_t* aq_t[3];
            float* aq_d[3];
            for (int k = 0; k < 3; ++k) {
                aq[k] = dev<char>(aq_b), aq_t[k] = dev<int8_t>((size_t) Tmax * nqv), aq_d[k] = dev<float>((size_t) Tmax * (nqv / 32));
                q8_twin(aq[k], aq_b, aq_t[k], aq_d[k]);
            }
            int* cells = dev<int>((size_t) Tmax * maxsel);
            int* ncells = dev<int>((size_t) Tmax);
            int* posv = dev<int>((size_t) Tmax);
            const int** pos_ptrs = dev<const int*>((size_t) Tmax);
            const int** pt_ptrs = dev<const int*>((size_t) Tmax);
            unsigned* tag = dev<unsigned>(1);
            AttRing ar;
            for (int T : Ts) {
                // the rows are the last T positions; a row past 2048 positions attends 512 blocks (random here, ascending) and its tail, as the selection writes them
                const int pos0 = ctx - T;
                std::vector<int> hc((size_t) T * maxsel, 0), hn((size_t) T), hp((size_t) T);
                std::vector<const int*> hpp((size_t) T), hpt((size_t) T, pool.g.pt);
                for (int t = 0; t < T; ++t) {
                    const int p = pos0 + t, nv = (p + 1) / 4, n_tail = (p + 1) % 4;
                    hp[(size_t) t] = p;
                    hpp[(size_t) t] = posv + t;
                    if (nv <= 512) {
                        hn[(size_t) t] = -1;
                        continue;
                    }
                    std::vector<int> blocks((size_t) nv);
                    for (int j = 0; j < nv; ++j) blocks[(size_t) j] = j;
                    for (int j = 0; j < 512; ++j) std::swap(blocks[(size_t) j], blocks[(size_t) j + rng() % (size_t) (nv - j)]);
                    std::sort(blocks.begin(), blocks.begin() + 512);
                    int n = 0;
                    int* row = hc.data() + (size_t) t * maxsel;
                    for (int j = 0; j < 512; ++j)
                        for (int m = 0; m < 4; ++m) row[n++] = blocks[(size_t) j] * 4 + m;
                    for (int m = 0; m < n_tail; ++m) row[n++] = (p / 4) * 4 + m;
                    hn[(size_t) t] = n;
                }
                ck(cudaMemcpy(cells, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice), "copy");
                ck(cudaMemcpy(ncells, hn.data(), hn.size() * 4, cudaMemcpyHostToDevice), "copy");
                ck(cudaMemcpy(posv, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice), "copy");
                ck(cudaMemcpy(pos_ptrs, hpp.data(), hpp.size() * sizeof(void*), cudaMemcpyHostToDevice), "copy");
                ck(cudaMemcpy(pt_ptrs, hpt.data(), hpt.size() * sizeof(void*), cudaMemcpyHostToDevice), "copy");
                ck(cudaMemcpy(pos0p, &pos0, 4, cudaMemcpyHostToDevice), "copy");
                LaneRows lr;
                lr.pos = pos_ptrs;
                lr.pt = pt_ptrs;
                const size_t ob = (size_t) T * NQH_LOC * HD * 4;
                for (int mode : modes) {
                    att_mm_force(mode);
                    // part + combine (attn, one sequence's window), then the part alone in the batched window's form (attn_part_lanes)
                    const double us_attn = graph_us(st, REP, RUNS, [&] {
                        for (int i = 0; i < REP; ++i) attn(qg, pool.kc(i % L), pool.vc(i % L), pool.g, out[mode], ar, tag, 0, pos0p, T, scratch, st, cells, ncells);
                    });
                    const double us_part = graph_us(st, REP, RUNS, [&] {
                        for (int i = 0; i < REP; ++i) attn_part_lanes(qg, pool.kc(i % L), pool.vc(i % L), pool.g, lr, T, scratch, st, cells, ncells, 0);
                    });
                    // layer 0's output by both forms: the same bits
                    graph_us(st, 1, 1, [&] { attn(qg, pool.kc(0), pool.vc(0), pool.g, out[mode], ar, tag, 0, pos0p, T, scratch, st, cells, ncells); });
                    graph_us(st, 1, 1, [&] {
                        attn_part_lanes(qg, pool.kc(0), pool.vc(0), pool.g, lr, T, scratch, st, cells, ncells, 0);
                        attn_finish(qg, out_l, ar, tag, 0, T, scratch, st);
                    });
                    // the q8_1 form of the output: attn and attn_finish write it themselves (the combine's launch with FNX_ATT_Q8; a quantize after it
                    // in any other form): the same bytes as a quantize of the output, rows and twins
                    for (int k = 0; k < 3; ++k) {
                        ck(cudaMemset(aq[k], 0, aq_b), "memset");
                        ck(cudaMemset(aq_t[k], 0, (size_t) Tmax * nqv), "memset");
                        ck(cudaMemset(aq_d[k], 0, (size_t) Tmax * (nqv / 32) * 4), "memset");
                    }
                    graph_us(st, 1, 1, [&] { attn(qg, pool.kc(0), pool.vc(0), pool.g, out[mode], ar, tag, 0, pos0p, T, scratch, st, cells, ncells, aq[0]); });
                    graph_us(st, 1, 1, [&] {
                        attn(qg, pool.kc(0), pool.vc(0), pool.g, out[mode], ar, tag, 0, pos0p, T, scratch, st, cells, ncells);
                        quantize_q8_1(out[mode], aq[1], nqv, T, st);
                    });
                    graph_us(st, 1, 1, [&] {
                        attn_part_lanes(qg, pool.kc(0), pool.vc(0), pool.g, lr, T, scratch, st, cells, ncells, 0);
                        attn_finish(qg, out_l, ar, tag, 0, T, scratch, st, aq[2]);
                    });
                    {
                        const size_t qb = q8_1_bytes(nqv) * (size_t) T, tb = (size_t) T * nqv, db = (size_t) T * (nqv / 32) * 4;
                        const unsigned long long hq[3] = {fnv(aq[0], qb) ^ fnv(aq_t[0], tb) * 3 ^ fnv(aq_d[0], db) * 5, fnv(aq[1], qb) ^ fnv(aq_t[1], tb) * 3 ^ fnv(aq_d[1], db) * 5,
                                                          fnv(aq[2], qb) ^ fnv(aq_t[2], tb) * 3 ^ fnv(aq_d[2], db) * 5};
                        const bool okq = hq[0] == hq[1] && hq[2] == hq[1];
                        // the time of a call with its q8_1 output: made by attn itself, and by a quantize launch after it
                        const double us_in = graph_us(st, REP, RUNS, [&] {
                            for (int i = 0; i < REP; ++i) attn(qg, pool.kc(i % L), pool.vc(i % L), pool.g, out[mode], ar, tag, 0, pos0p, T, scratch, st, cells, ncells, aq[0]);
                        });
                        const double us_after = graph_us(st, REP, RUNS, [&] {
                            for (int i = 0; i < REP; ++i) {
                                attn(qg, pool.kc(i % L), pool.vc(i % L), pool.g, out[mode], ar, tag, 0, pos0p, T, scratch, st, cells, ncells);
                                quantize_q8_1(out[mode], aq[1], nqv, T, st);
                            }
                        });
                        std::printf("attn mm=%d ctx %d T=%2d  with the q8_1 output: by attn %7.2f us, by a quantize after it %7.2f us (%+.2f)\n", mode, ctx, T, us_in, us_after,
                                    us_in - us_after);
                        std::printf("attn mm=%d ctx %d T=%2d  q8_1 output %016llx: attn %s, attn_finish %s\n", mode, ctx, T, hq[1], hq[0] == hq[1] ? "equal" : "DIFFERS",
                                    hq[2] == hq[1] ? "equal" : "DIFFERS");
                        fails += !okq;
                    }
                    const unsigned long long h = fnv(out[mode], ob), hl = fnv(out_l, ob);
                    // (the two parts are timed in different launch forms: their difference only estimates the combine)
                    std::printf("attn mm=%d ctx %d T=%2d  part+combine %7.2f us  part (lanes form) %7.2f us  (difference %5.2f)  hash %016llx  lanes form %s\n",
                                mode, ctx, T, us_attn, us_part, us_attn - us_part, h, h == hl ? "equal" : "DIFFERS");
                    fails += h != hl;
                }
                if (modes.size() == 2) {
                    const std::vector<float> a = host(out[0], ob / 4), b = host(out[1], ob / 4);
                    double dmax = 0, amax = 0, d2 = 0, a2 = 0;
                    for (size_t i = 0; i < a.size(); ++i) {
                        const double d = std::fabs((double) a[i] - b[i]);
                        dmax = std::max(dmax, d);
                        amax = std::max(amax, std::fabs((double) a[i]));
                        d2 += d * d;
                        a2 += (double) a[i] * a[i];
                    }
                    const double rel = std::sqrt(d2 / std::max(a2, 1e-300));
                    std::printf("attn ctx %d T=%2d  mm against the old kernel: max |diff| %.3g (largest |value| %.3g), rms diff / rms value %.3g%s\n", ctx, T, dmax,
                                amax, rel, rel < 1e-4 ? "" : "  TOO LARGE");
                    fails += !(rel < 1e-4);
                }
            }
            att_mm_force(-1);
        } else {
            const int mp_big = 65536 + 16, mp = std::max(mp_big, ctx / 4 + 16);
            const float* q = dev_f32((size_t) Tmax * QSA_HEADS * QSA_KEY, -1.f, 1.f);
            float* sc[2] = {dev<float>((size_t) Tmax * mp), dev<float>((size_t) Tmax * mp)};
            int* cells[2] = {dev<int>((size_t) Tmax * maxsel), dev<int>((size_t) Tmax * maxsel)};
            int* ncells[2] = {dev<int>((size_t) Tmax), dev<int>((size_t) Tmax)};
            // the lists of the two selections of T rows: equal (n_cells and every cell)?
            const auto lists_equal = [&](int T, std::string& why) {
                const std::vector<int> na = host(ncells[0], (size_t) T), nb = host(ncells[1], (size_t) T);
                const std::vector<int> ca = host(cells[0], (size_t) T * maxsel), cb = host(cells[1], (size_t) T * maxsel);
                for (int t = 0; t < T; ++t) {
                    if (na[(size_t) t] != nb[(size_t) t]) {
                        why = "row " + std::to_string(t) + ": n " + std::to_string(na[(size_t) t]) + " / " + std::to_string(nb[(size_t) t]);
                        return false;
                    }
                    for (int i = 0; i < na[(size_t) t]; ++i)
                        if (ca[(size_t) t * maxsel + i] != cb[(size_t) t * maxsel + i]) {
                            why = "row " + std::to_string(t) + " cell " + std::to_string(i) + ": " + std::to_string(ca[(size_t) t * maxsel + i]) + " / " +
                                  std::to_string(cb[(size_t) t * maxsel + i]);
                            return false;
                        }
                }
                return true;
            };
            const auto clear_lists = [&](int k) {
                ck(cudaMemset(cells[k], 0xee, (size_t) Tmax * maxsel * 4), "memset");
                ck(cudaMemset(ncells[k], 0xee, (size_t) Tmax * 4), "memset");
            };
            for (int T : Ts) {
                const int pos0 = ctx - T;
                ck(cudaMemcpy(pos0p, &pos0, 4, cudaMemcpyHostToDevice), "copy");
                double us_sc[2] = {0, 0}, us_sel[2] = {0, 0};
                for (int mode : modes) {
                    ck(cudaMemset(sc[mode], 0, (size_t) Tmax * mp * 4), "memset");
                    us_sc[mode] = graph_us(st, REP, RUNS, [&] {
                        for (int i = 0; i < REP; ++i) kb_idx_score(q, pool.pooled(i % L), pool.g, sc[mode], mp, pos0p, T, mode, st);
                    });
                    graph_us(st, 1, 1, [&] { kb_idx_score(q, pool.pooled(0), pool.g, sc[mode], mp, pos0p, T, mode, st); });
                    clear_lists(mode);
                    us_sel[mode] = graph_us(st, REP, RUNS, [&] {
                        for (int i = 0; i < REP; ++i) kb_idx_select(sc[mode], cells[mode], ncells[mode], mp, pos0p, T, 0, mode, st);
                    });
                    std::printf("qsasel mm=%d ctx %d (%d blocks) T=%2d  score %7.2f us  select %7.2f us  scores %016llx  cells %016llx\n", mode, ctx, ctx / 4, T,
                                us_sc[mode], us_sel[mode], fnv(sc[mode], (size_t) T * mp * 4), fnv(cells[mode], (size_t) T * maxsel * 4));
                }
                if (modes.size() == 2) {
                    std::string why;
                    const bool sc_eq = fnv(sc[0], (size_t) T * mp * 4) == fnv(sc[1], (size_t) T * mp * 4), l_eq = lists_equal(T, why);
                    std::printf("qsasel ctx %d T=%2d  scores %s, lists %s%s\n", ctx, T, sc_eq ? "equal bits" : "DIFFER", l_eq ? "equal" : "DIFFER: ", why.c_str());
                    fails += !sc_eq + !l_eq;
                    // FNX_ATT_MM_SCORE_TC's scorer (the prompt path's matrix products: other last bits): its time, and the cells its selection changes
                    const double us_tc = graph_us(st, REP, RUNS, [&] {
                        for (int i = 0; i < REP; ++i) kb_idx_score(q, pool.pooled(i % L), pool.g, sc[1], mp, pos0p, T, 2, st);
                    });
                    graph_us(st, 1, 1, [&] {
                        kb_idx_score(q, pool.pooled(0), pool.g, sc[1], mp, pos0p, T, 2, st);
                        kb_idx_select(sc[1], cells[1], ncells[1], mp, pos0p, T, 0, 1, st);
                    });
                    const std::vector<int> ca = host(cells[0], (size_t) T * maxsel), cb = host(cells[1], (size_t) T * maxsel);
                    const std::vector<int> na = host(ncells[0], (size_t) T);
                    size_t diff = 0, all = 0;
                    for (int t = 0; t < T; ++t) {   // cells of the exact scores' list that the other list does not hold
                        const int n = std::max(na[(size_t) t], 0);
                        const int* a = ca.data() + (size_t) t * maxsel;
                        const int* b = cb.data() + (size_t) t * maxsel;
                        all += (size_t) n;
                        for (int i = 0; i < n; ++i) diff += !std::binary_search(b, b + n, a[i]);
                    }
                    std::printf("qsasel ctx %d T=%2d  score tc %7.2f us (FNX_ATT_MM_SCORE_TC=1): %zu of %zu cells of the exact scores' lists are not in its lists\n",
                                ctx, T, us_tc, diff, all);
                }
            }
            if (modes.size() == 2) {
                // the two selections on made-up scores: ties at the threshold, a sample that misses or holds only the high scores, sizes around the kernel's
                // limits (513: just past dense; 4096 / 4097: all scores as candidates / the first sampled size)
                const int T = std::min(Tmax, 8);
                const char* names[] = {"uniform", "64 values", "all equal", "half zero", "ascending", "descending", "high at 8k", "high at 8k+1", "all zero", "two values"};
                int bad = 0, cases = 0;
                for (int nvs : {513, 600, 2047, 4096, 4097, 9000, 27250, 40000, 65536}) {
                    const int pos0 = 4 * nvs - 1;   // row 0 sees nvs blocks and no tail; rows 1..3 the same blocks and a tail
                    ck(cudaMemcpy(pos0p, &pos0, 4, cudaMemcpyHostToDevice), "copy");
                    for (int dist = 0; dist < 10; ++dist) {
                        std::vector<float> h((size_t) T * mp, 0.f);
                        std::uniform_real_distribution<float> u(0.f, 1.f);
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
                        put(sc[0], h);
                        for (int mode = 0; mode < 2; ++mode) {
                            clear_lists(mode);
                            graph_us(st, 1, 1, [&] { kb_idx_select(sc[0], cells[mode], ncells[mode], mp, pos0p, T, 0, mode, st); });
                        }
                        std::string why;
                        const bool eq = lists_equal(T, why);
                        ++cases;
                        bad += !eq;
                        if (!eq) std::printf("qsasel made-up scores: %d blocks, %s, T=%d: lists DIFFER: %s\n", nvs, names[dist], T, why.c_str());
                    }
                }
                std::printf("qsasel made-up scores: %d of %d cases with equal lists (T=%d)\n", cases - bad, cases, T);
                fails += bad;
            }
        }
        if (fails) {
            std::fprintf(stderr, "kbench %s: %d comparison(s) FAILED\n", which.c_str(), fails);
            return 1;
        }
#else
        std::fprintf(stderr, "kbench %s: this build has no STRATA_SPLIT_APPROX\n", which.c_str());
        return 2;
#endif
    }
    return 0;
}
