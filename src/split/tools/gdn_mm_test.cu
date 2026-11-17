// FNX_PF_GDN_MM=1 (k_gdn_ch_rec_mm, the prompt GDN recurrence on WMMA) against k_gdn_ch_rec and the row-by-row k_gdn_pf_rec, on one card
// (HIP gfx12, not a ctest: needs a GPU, no model).   strata-split-gdnmm-test-4 [nk nv [C ...]]   default nk 4, nv 12 (one card of four)
// Random inputs of the model's shapes as tools/gdn_tc_test.cu makes them; prints each sub-chunk kernel's distance to the row-by-row
// reference (outputs, end state: largest difference over largest value) and the time of prep + recurrence for both.
#define FNX_KPART 4
#include "../src/kernels.cu"

#include <cstdint>
#include <random>

namespace fnx { std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> v; return v; } }

namespace {
using namespace fnx;

struct Rng {
    std::mt19937_64 g{12345};
    float uni() { return (float) std::uniform_real_distribution<double>(0.0, 1.0)(g); }
    float nrm() { return (float) std::normal_distribution<double>(0.0, 1.0)(g); }
};

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); std::exit(2); } } while (0)

double rel_diff(const std::vector<float>& r, const std::vector<float>& x, double& maxval) {
    double md = 0, mv = 0;
    for (size_t i = 0; i < r.size(); ++i) {
        if (!std::isfinite(r[i]) || !std::isfinite(x[i])) { maxval = NAN; return NAN; }   // a non-finite value fails the check
        md = std::max(md, (double) std::fabs(r[i] - x[i]));
        mv = std::max(mv, (double) std::fabs(r[i]));
    }
    maxval = mv;
    return md / std::max(mv, 1e-30);
}
}  // namespace

int main(int argc, char** argv) {
    int nk = 4, nv = 12;
    std::vector<int> cases = {33, 1000, 2048, 16384};
    if (argc >= 3) { nk = std::atoi(argv[1]); nv = std::atoi(argv[2]); }
    if (argc > 3) { cases.clear(); for (int i = 3; i < argc; ++i) cases.push_back(std::atoi(argv[i])); }
    const int vpk = nv / nk;
    const int ch = (2 * nk + nv) * HS;
    int dev = 0;
    CK(cudaGetDevice(&dev));
    cudaDeviceProp pr{};
    CK(cudaGetDeviceProperties(&pr, dev));
    std::printf("card %s (%d CUs), a slice of %d key heads and %d value heads\n", pr.name, pr.multiProcessorCount, nk, nv);
    CK(cudaFuncSetAttribute(k_gdn_ch_rec, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) GCH_REC_SMEM));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    std::vector<int> vmap(nv);
    for (int lv = 0; lv < nv; ++lv) vmap[lv] = (lv / vpk) | (lv % vpk == 0 ? 1 << 8 : 0);
    int* d_vmap;
    CK(cudaMalloc(&d_vmap, nv * 4));
    CK(cudaMemcpy(d_vmap, vmap.data(), nv * 4, cudaMemcpyHostToDevice));
    int fails = 0;
    for (int variant = 0; variant < 2; ++variant) {   // 0: random state, 1: zero state (a fresh prompt)
        for (int C : cases) {
            Rng rng;
            std::vector<float> xc((size_t) C * ch), ab((size_t) C * 2 * nv), dt(nv), a(nv), S0((size_t) nv * HS * HS);
            for (int t = 0; t < C; ++t) {
                float* row = xc.data() + (size_t) t * ch;
                for (int h = 0; h < 2 * nk; ++h) {   // q, k: unit rows per head
                    double ss = 0;
                    for (int i = 0; i < HS; ++i) { row[h * HS + i] = rng.nrm(); ss += (double) row[h * HS + i] * row[h * HS + i]; }
                    const float inv = (float) (1.0 / std::sqrt(ss + 1e-6));
                    for (int i = 0; i < HS; ++i) row[h * HS + i] *= inv;
                }
                for (int i = 0; i < nv * HS; ++i) row[2 * nk * HS + i] = 0.5f * rng.nrm();
                for (int i = 0; i < 2 * nv; ++i) ab[(size_t) t * 2 * nv + i] = rng.nrm();
            }
            for (int i = 0; i < nv; ++i) { dt[i] = 0.5f * rng.nrm(); a[i] = -std::exp(-3.5f + 5.5f * rng.uni()); }
            for (auto& x : S0) x = variant ? 0.0f : 0.05f * rng.nrm();
            float *d_xc, *d_ab, *d_dt, *d_a, *d_S, *d_y0, *d_y1, *d_yr, *d_sq, *d_chw, *d_S0;
            const size_t ny = (size_t) C * nv * HS, ns = (size_t) nv * HS * HS;
            const size_t chw_n = (size_t) ((C + GCH - 1) / GCH) * nv * GCH_PREP_FLOATS;
            CK(cudaMalloc(&d_xc, xc.size() * 4)); CK(cudaMalloc(&d_ab, ab.size() * 4)); CK(cudaMalloc(&d_dt, nv * 4)); CK(cudaMalloc(&d_a, nv * 4));
            CK(cudaMalloc(&d_S, ns * 4)); CK(cudaMalloc(&d_S0, ns * 4)); CK(cudaMalloc(&d_y0, ny * 4)); CK(cudaMalloc(&d_y1, ny * 4)); CK(cudaMalloc(&d_chw, chw_n * 4));
            CK(cudaMalloc(&d_yr, ny * 4)); CK(cudaMalloc(&d_sq, (size_t) C * nv * GDN_Q * 4));
            CK(cudaMemcpy(d_xc, xc.data(), xc.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_ab, ab.data(), ab.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_dt, dt.data(), nv * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_a, a.data(), nv * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_S0, S0.data(), ns * 4, cudaMemcpyHostToDevice));
            auto prep = [&] { k_gdn_ch_prep<<<dim3((C + GCH - 1) / GCH, nv), GCH_THREADS, 0, s>>>(d_xc, d_ab, d_dt, d_a, d_vmap, nk, nv, C, d_chw); };
            auto run_old = [&] { prep(); k_gdn_ch_rec<<<nv * GCH_REC_BLOCKS_PER_HEAD, GCR_THREADS, GCH_REC_SMEM, s>>>(d_xc, d_chw, d_S, d_y0, nk, nv, d_vmap, C); };
            auto prep_mm = [&] { k_gdn_ch_prep<true><<<dim3((C + GCH - 1) / GCH, nv), GCH_THREADS, 0, s>>>(d_xc, d_ab, d_dt, d_a, d_vmap, nk, nv, C, d_chw); };
            auto run_new = [&] { prep_mm(); k_gdn_ch_rec_mm<0><<<nv * GCH_REC_BLOCKS_PER_HEAD, GMM_THREADS, 0, s>>>(d_xc, d_chw, d_S, d_y1, nk, nv, d_vmap, C); };
            std::vector<float> yr(ny), s_ref(ns), y0(ny), s_old(ns), y1(ny), s_new(ns);
            CK(cudaMemcpy(d_S, d_S0, ns * 4, cudaMemcpyDeviceToDevice));
            k_gdn_pf_rec<<<nv * GDN_Q, GDN_PF_THREADS, 0, s>>>(d_xc, d_ab, d_dt, d_a, d_S, d_yr, d_sq, nk, nv, d_vmap, C);
            CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
            CK(cudaMemcpy(yr.data(), d_yr, ny * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(s_ref.data(), d_S, ns * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(d_S, d_S0, ns * 4, cudaMemcpyDeviceToDevice));
            run_old();
            CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
            CK(cudaMemcpy(y0.data(), d_y0, ny * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(s_old.data(), d_S, ns * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(d_S, d_S0, ns * 4, cudaMemcpyDeviceToDevice));
            CK(cudaMemset(d_y1, 0, ny * 4));
            run_new();
            CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
            CK(cudaMemcpy(y1.data(), d_y1, ny * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(s_new.data(), d_S, ns * 4, cudaMemcpyDeviceToHost));
            double my, ms_, d1, d2;
            const double dy0 = rel_diff(yr, y0, my), ds0 = rel_diff(s_ref, s_old, ms_);
            const double dy1 = rel_diff(yr, y1, d1), ds1 = rel_diff(s_ref, s_new, d2);
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            float t_old = 0, t_new = 0;
            const int reps = C >= 8192 ? 5 : 20;
            run_old(); run_new();
            CK(cudaEventRecord(e0, s));
            for (int r = 0; r < reps; ++r) run_old();
            CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&t_old, e0, e1));
            CK(cudaEventRecord(e0, s));
            for (int r = 0; r < reps; ++r) run_new();
            CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&t_new, e0, e1));
            float t_prep = 0;
            CK(cudaEventRecord(e0, s));
            for (int r = 0; r < reps; ++r) prep();
            CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&t_prep, e0, e1));
            std::printf("  prep alone %.3f ms\n", t_prep / reps);
            if (std::getenv("GMM_SKIP")) {   // timing only: rec_mm with parts left out (wrong results)
                prep_mm();
                auto skip_run = [&](auto kern, const char* name) {
                    float t = 0;
                    CK(cudaEventRecord(e0, s));
                    for (int r = 0; r < reps; ++r) { CK(cudaMemcpyAsync(d_S, d_S0, ns * 4, cudaMemcpyDeviceToDevice, s)); kern<<<nv * GCH_REC_BLOCKS_PER_HEAD, GMM_THREADS, 0, s>>>(d_xc, d_chw, d_S, d_y1, nk, nv, d_vmap, C); }
                    CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&t, e0, e1));
                    std::printf("  rec_mm %-28s %.3f ms\n", name, t / reps);
                };
                {
                    float t = 0;
                    CK(cudaEventRecord(e0, s));
                    for (int r = 0; r < reps; ++r) k_gdn_ch_prep<true, 1><<<dim3((C + GCH - 1) / GCH, nv), GCH_THREADS, 0, s>>>(d_xc, d_ab, d_dt, d_a, d_vmap, nk, nv, C, d_chw);
                    CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&t, e0, e1));
                    std::printf("  prep without the T solve      %.3f ms\n", t / reps);
                    prep_mm();
                }
                skip_run(k_gdn_ch_rec_mm<0>, "all");
                skip_run(k_gdn_ch_rec_mm<1>, "no next v/T/P loads");
                skip_run(k_gdn_ch_rec_mm<2>, "no next K/Q loads");
                skip_run(k_gdn_ch_rec_mm<3>, "no next loads");
                skip_run(k_gdn_ch_rec_mm<7>, "no loads, no stores");
            }
            {
                uint64_t h = 1469598103934665603ull;
                auto mix = [&](const std::vector<float>& v) { const unsigned char* p = (const unsigned char*) v.data(); for (size_t i = 0; i < v.size() * 4; ++i) h = (h ^ p[i]) * 1099511628211ull; };
                mix(y0); mix(s_old);
                std::printf("  old path outputs + state hash %016llx\n", (unsigned long long) h);
            }
            const bool ok = std::isfinite(dy1) && std::isfinite(ds1) && dy1 < 4 * dy0 + 1e-5 && ds1 < 4 * ds0 + 1e-5;
            fails += !ok;
            std::printf("%s C %6d: vs row-by-row: y old %.2e new %.2e (max |y| %.3g), state old %.2e new %.2e (max |S| %.3g) | prep+rec old %.3f ms, mm %.3f ms (%.2fx) %s\n",
                        variant ? "zero state  " : "random state", C, dy0, dy1, my, ds0, ds1, ms_, t_old / reps, t_new / reps, t_old / std::max(t_new, 1e-9f), ok ? "ok" : "FAIL");
            cudaEventDestroy(e0); cudaEventDestroy(e1);
            cudaFree(d_yr); cudaFree(d_sq);
            cudaFree(d_xc); cudaFree(d_ab); cudaFree(d_dt); cudaFree(d_a); cudaFree(d_S); cudaFree(d_S0); cudaFree(d_y0); cudaFree(d_y1); cudaFree(d_chw);
        }
    }
    std::printf("%d failures\n", fails);
    return fails ? 1 : 0;
}
