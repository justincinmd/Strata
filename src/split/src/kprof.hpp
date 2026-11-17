// FNX_KPROF=1 (a measuring aid, off by default): the one-user window runs eagerly and every kernel launch on the LAST card is timed with a
// device synchronize after it (launch call + kernel + the wait's return; the smallest entry shows that overhead). Only the last card is timed: its peers'
// work for the same half is already queued, so its exchange kernels never wait for work that is not there, and the peers run ahead and spin in theirs.
// The report (every FNX_KPROF windows of one row count, to stderr) lists, per kernel name: launches a window, mean us, ms a window. Decode is slow
// while it is on, and the answers are those of the captured window.
// The draft layer's step (Engine::draft_rows) is timed the same way, in its own table ("kprof: draft T=.."), and the attention's three kernels are
// marked one by one ("attn part", "attn local", "attn merge").
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "env.hpp"
#include "model.hpp"

namespace fnx {
inline int g_kprof_dev = -1;             // the device being timed, -1: off (the default)
struct KprofRow { long n = 0; double us = 0; };
struct KprofT { int dev = -1; long windows = 0; double window_us = 0; std::vector<std::pair<std::string, KprofRow>> rows; };
inline KprofT g_kprof[2][MAXT + 1];     // [0] the window, [1] the draft layer's step
inline int g_kprof_T = 0, g_kprof_kind = 0;
inline std::chrono::steady_clock::time_point g_kprof_last, g_kprof_w0;
inline void kprof_mark(const char* what) {
    int dev = -1;
    cudaGetDevice(&dev);
    if (dev != g_kprof_dev || !g_kprof_T) {              // a peer's launch (not timed): the timed card's clock starts after it
        g_kprof_last = std::chrono::steady_clock::now();
        return;
    }
    cudaDeviceSynchronize();
    const auto now = std::chrono::steady_clock::now();
    auto& rows = g_kprof[g_kprof_kind][g_kprof_T].rows;
    auto it = std::find_if(rows.begin(), rows.end(), [&](const std::pair<std::string, KprofRow>& r) { return r.first == what; });
    if (it == rows.end()) it = rows.insert(rows.end(), {what, KprofRow{}});
    ++it->second.n;
    it->second.us += std::chrono::duration<double, std::micro>(now - g_kprof_last).count();
    g_kprof_last = now;
}
inline void kprof_window(int T, bool begin, int kind = 0) {
    if (begin) {
        g_kprof_T = T, g_kprof_kind = kind;
        g_kprof[kind][T].dev = g_kprof_dev;
        g_kprof_w0 = g_kprof_last = std::chrono::steady_clock::now();
        return;
    }
    KprofT& k = g_kprof[kind][T];
    const char* tab = kind ? "draft " : "";
    g_kprof_T = 0;
    ++k.windows;
    k.window_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - g_kprof_w0).count();
    static const long every = std::max(1LL, env_int("FNX_KPROF", 0));
    if (k.windows % every) return;
    std::vector<std::pair<std::string, KprofRow>> rows = k.rows;
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.us > b.second.us; });
    double sum = 0, launches = 0;
    for (const auto& r : rows) { sum += r.second.us; launches += r.second.n; }
    std::fprintf(stderr, "fnx: kprof: %sT=%d, %ld windows, card %d: %.0f launches a window, %.2f ms of timed launches, %.2f ms eager window\n", tab, T,
                 k.windows, k.dev, launches / k.windows, sum / k.windows / 1e3, k.window_us / k.windows / 1e3);
    for (const auto& r : rows)
        std::fprintf(stderr, "fnx: kprof: %sT=%d %-22s %6.1f a window  %7.1f us each  %7.3f ms a window  %4.1f%%\n", tab, T, r.first.c_str(),
                     (double) r.second.n / k.windows, r.second.us / r.second.n, r.second.us / k.windows / 1e3, 100.0 * r.second.us / sum);
}
}  // namespace fnx
