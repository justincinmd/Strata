// The prompt path's routed experts (moe_mma.cu: moe_gate_up_mma, swiglu_h16, moe_down_mma16) on one card, on the 4-card
// shapes of Qwen3.8-Flash-Next UD-Q4_K_XL: gate|up Q4_K 320 x 2560 an expert (the card's FF slice of gate and up), down Q5_1
// 2560 x 160. Random weights (sane scales), random f16 activations, routing like the model's (10 distinct experts a row, skewed
// expert popularity), the jobs as k_pf_fill builds them. Prints each kernel's time and an FNV hash of its output (a variant
// that keeps the sums' order must keep the hash).   strata-split-moemm-test-4 [C [reps [skew]]]   default 4096 20 0.8
// (HIP gfx12, not a ctest: needs a GPU, no model)
#define FNX_MM_ATTR 1   // MM_SKIP=bits (moe_mma.cuh): attribution runs
#include "../src/moe_mma.cu"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {
using namespace fnx;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); std::exit(2); } } while (0)

uint64_t fnv(const void* p, size_t n) {
    const unsigned char* b = (const unsigned char*) p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}
uint16_t h16(float f) { __half h = __float2half(f); return *reinterpret_cast<uint16_t*>(&h); }

}  // namespace

int main(int argc, char** argv) {
    const int C = argc > 1 ? std::atoi(argv[1]) : 4096, reps = argc > 2 ? std::atoi(argv[2]) : 20;
    const double skew = argc > 3 ? std::atof(argv[3]) : 0.8;
    constexpr int NE = 512, K = 10, NIN = 2560, GU = 320, FFS = 160;
    const int P = C * K;
    std::mt19937_64 g(12345);
    std::uniform_int_distribution<int> byte(0, 255);
    // weights: Q4_K (144 bytes a 256-block: d, dmin halves, 12 scale bytes, 128 quant bytes), Q5_1 (24 bytes a 32-block: d, m halves, ...)
    const size_t gu_row = NIN / 256 * 144, gu_sa = gu_row * GU, dn_row = FFS / 32 * 24, dn_sa = dn_row * NIN;
    std::vector<unsigned char> wgu(gu_sa * NE), wdn(dn_sa * NE);
    for (auto& b : wgu) b = (unsigned char) byte(g);
    for (auto& b : wdn) b = (unsigned char) byte(g);
    for (size_t o = 0; o < wgu.size(); o += 144) { const uint16_t d = h16(0.004f), m = h16(0.002f); std::memcpy(&wgu[o], &d, 2); std::memcpy(&wgu[o + 2], &m, 2); }
    for (size_t o = 0; o < wdn.size(); o += 24) { const uint16_t d = h16(0.01f), m = h16(-0.15f); std::memcpy(&wdn[o], &d, 2); std::memcpy(&wdn[o + 2], &m, 2); }
    std::vector<uint16_t> x((size_t) C * NIN);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (auto& v : x) v = h16(u(g));
    // routing: expert e has weight 1 / (rank + 8)^skew under a random permutation; 10 distinct a row
    std::vector<int> perm(NE);
    for (int e = 0; e < NE; ++e) perm[e] = e;
    std::shuffle(perm.begin(), perm.end(), g);
    std::vector<double> wt(NE);
    for (int r = 0; r < NE; ++r) wt[perm[r]] = 1.0 / std::pow(r + 8.0, skew);
    std::discrete_distribution<int> pick(wt.begin(), wt.end());
    std::vector<int> topi(P);
    for (int t = 0; t < C; ++t)
        for (int k = 0; k < K; ++k) {
            int e;
            bool dup;
            do {
                e = pick(g), dup = false;
                for (int j = 0; j < k; ++j) dup |= topi[t * K + j] == e;
            } while (dup);
            topi[t * K + k] = e;
        }
    // jobs as k_pf_fill: per expert in index order, PF_JOB pairs a job
    std::vector<int> cnt(NE, 0);
    for (int p = 0; p < P; ++p) ++cnt[topi[p]];
    std::vector<int> jbase(NE);
    int nj = 0;
    for (int e = 0; e < NE; ++e) jbase[e] = nj, nj += (cnt[e] + PF_JOB - 1) / PF_JOB;
    std::vector<BigJob> gu(nj), dn(nj);
    std::vector<int> fill(NE, 0);
    for (int p = 0; p < P; ++p) {
        const int e = topi[p], s = fill[e]++, j = jbase[e] + s / PF_JOB, k = s % PF_JOB;
        gu[j].y[k] = p / K, gu[j].out[k] = p, dn[j].y[k] = p, dn[j].out[k] = p;
        if (k == 0) gu[j].w = dn[j].w = e, gu[j].n = dn[j].n = std::min(PF_JOB, cnt[e] - (s / PF_JOB) * PF_JOB);
    }
    int full = 0, used = 0, mx = 0;
    for (int e = 0; e < NE; ++e) used += cnt[e] > 0, mx = std::max(mx, cnt[e]);
    for (int j = 0; j < nj; ++j) full += gu[j].n == PF_JOB;
    std::printf("C %d: %d pairs, %d experts used (largest %d rows), %d jobs (%d full), fill %.1f %%\n", C, P, used, mx, nj, full,
                100.0 * P / ((double) nj * PF_JOB));
    const int cap = std::max(nj, (int) (1.15 * nj));   // the engine launches njobs_cap blocks a column; the rest return at once
    unsigned char *dwgu, *dwdn;
    uint16_t *dx, *dh, *dpart;
    float* dgu;
    BigJob *djgu, *djdn;
    int* dnj;
    CK(cudaMalloc(&dwgu, wgu.size()));
    CK(cudaMalloc(&dwdn, wdn.size()));
    CK(cudaMalloc(&dx, x.size() * 2));
    CK(cudaMalloc(&dgu, (size_t) P * GU * 4));
    CK(cudaMalloc(&dh, (size_t) P * FFS * 2));
    CK(cudaMalloc(&dpart, (size_t) P * NIN * 2));
    CK(cudaMalloc(&djgu, sizeof(BigJob) * cap));
    CK(cudaMalloc(&djdn, sizeof(BigJob) * cap));
    CK(cudaMalloc(&dnj, 4));
    CK(cudaMemcpy(dwgu, wgu.data(), wgu.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dwdn, wdn.data(), wdn.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(djgu, gu.data(), sizeof(BigJob) * nj, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(djdn, dn.data(), sizeof(BigJob) * nj, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dnj, &nj, 4, cudaMemcpyHostToDevice));
    DW A_gu, A_dn;
    A_gu.p = dwgu, A_gu.type = T_Q4_K, A_gu.n_in = NIN, A_gu.n_out = GU, A_gu.row_bytes = gu_row;
    A_dn.p = dwdn, A_dn.type = T_Q5_1, A_dn.n_in = FFS, A_dn.n_out = NIN, A_dn.row_bytes = dn_row;
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    const int skip = std::getenv("MM_SKIP") ? std::atoi(std::getenv("MM_SKIP")) : 0;
    CK(cudaMemcpyToSymbol(HIP_SYMBOL(mm_skip), &skip, sizeof(int)));
    if (skip) std::printf("MM_SKIP %d (attribution: outputs not valid)\n", skip);
    cudaEvent_t e0, e1, e2, e3;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&e2)); CK(cudaEventCreate(&e3));
    float tg = 0, ts = 0, td = 0;
    for (int r = 0; r < reps + 2; ++r) {
        CK(cudaEventRecord(e0, s));
        moe_gate_up_mma(A_gu, gu_sa, djgu, dnj, cap, dx, dgu, s);
        CK(cudaEventRecord(e1, s));
        swiglu_h16(dgu, FFS, P, dh, s);
        CK(cudaEventRecord(e2, s));
        moe_down_mma16(A_dn, dn_sa, djdn, dnj, cap, dh, dpart, s);
        CK(cudaEventRecord(e3, s));
        CK(cudaStreamSynchronize(s));
        float a, b, c;
        CK(cudaEventElapsedTime(&a, e0, e1)); CK(cudaEventElapsedTime(&b, e1, e2)); CK(cudaEventElapsedTime(&c, e2, e3));
        if (r >= 2) tg += a, ts += b, td += c;
    }
    std::vector<float> hgu((size_t) P * GU);
    std::vector<uint16_t> hpart((size_t) P * NIN);
    CK(cudaMemcpy(hgu.data(), dgu, hgu.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hpart.data(), dpart, hpart.size() * 2, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (float v : hgu) bad += !std::isfinite(v);
    const double fl_gu = 2.0 * P * GU * NIN, fl_dn = 2.0 * P * NIN * FFS;
    std::printf("gate|up %.3f ms (%.1f TFLOP/s useful)  swiglu %.3f ms  down %.3f ms (%.1f TFLOP/s useful)  non-finite gu %zu\n",
                tg / reps, fl_gu / (tg / reps * 1e9), ts / reps, td / reps, fl_dn / (td / reps * 1e9), bad);
    std::printf("hash gu %016llx part16 %016llx\n", (unsigned long long) fnv(hgu.data(), hgu.size() * 4),
                (unsigned long long) fnv(hpart.data(), hpart.size() * 2));
    return 0;
}
