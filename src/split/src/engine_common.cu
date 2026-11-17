// The parts of the Engine (engine.hpp) that do not depend on the model, for every model's build: see engine_parts.hpp.
#include "engine_parts.hpp"
#include "env.hpp"
#include "kprof.hpp"

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <ctime>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace fnx {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

float h2f(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s << 31;
        else {
            int ee = 0;
            uint32_t mm = m;
            while (!(mm & 1024)) { mm <<= 1; --ee; }
            u = (s << 31) | ((uint32_t) (ee + 127 - 14) << 23) | ((mm & 1023) << 13);
        }
    } else if (e == 31) u = (s << 31) | 0x7f800000u | (m << 13);
    else u = (s << 31) | ((e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

namespace {
// a state's pieces between a card and pageable host memory (laid end to end there), through the card's two
// page-locked halves in blocks of STAGE_BYTES: the copy engine fills (or empties) one half while the host copies the
// other. Pageable, because page-locked memory is mapped into every card: 2 MiB of each card's VRAM per GiB (64 MiB for
// 32 GiB, tools/pin_probe), 256 MiB for the 128 GiB cache, about what cards 0 and 1 have left
constexpr size_t STAGE_BYTES = (size_t) 32 << 20;
}  // namespace

void install_graph(CardBuf::GraphSlot& s, cudaGraph_t g, const std::vector<int>& key) {
    if (s.exec) {
        cudaGraphExecUpdateResultInfo info{};
        if (cudaGraphExecUpdate(s.exec, g, &info) == cudaSuccess) {
            s.key = key;
            ck(cudaGraphDestroy(g), "graph destroy");
            return;
        }
        cudaGetLastError();   // a different shape: a new executable in its place
        ck(cudaGraphExecDestroy(s.exec), "graph exec destroy");
        s.exec = nullptr;
    }
    ck(cudaGraphInstantiate(&s.exec, g, 0), "instantiate");
    s.key = key;
    ck(cudaGraphDestroy(g), "graph destroy");
}

long host_available_mib() {
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "MemAvailable: %ld kB", &kb) == 1) break;
    std::fclose(f);
    return kb < 0 ? -1 : kb / 1024;
}

// FNX_POOL_SPIN_MS=n (HIP; 20 on 4 cards in a FAST build, else 0 = off): the pool's threads do not sleep at once after a job.
// Each spins for up to n ms for the next one, and the caller spins for its threads' end, before either falls back to the condition variable. A solo
// round launches the pool for a window, 2-3 draft steps and the commit, with the cards' work between them; with sleeping threads the host needed 0.46 ms
// to queue each of them and the caller's card ran ahead of the others (4 x R9700, Q4, FNX_DRAFT_TIMING: a card's own draft step 0.71 / 0.52 / 0.52 /
// 0.52 ms; spinning: queue 0.04 ms, 0.47 on every card; one user 193.6 -> 206.6 tok/s, many users not slower, the same answers). n must bridge a
// window (~13 ms; 40 was not better). NCARD - 1 cores spin only while launches keep coming: n ms after the last one the threads sleep again. The jobs
// and their order per card are unchanged
static inline void pool_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

struct CardPool {
    std::vector<std::thread> th;
    std::mutex mu;
    std::condition_variable go, done;
    const std::function<void(int)>* job = nullptr;
    unsigned gen = 0;
    int left = 0;
    bool quit = false;
    std::atomic<unsigned> agen{0};   // gen and left again, for the spinning side (written under mu, read without it)
    std::atomic<int> aleft{0};
    std::atomic<bool> aquit{false};
    std::chrono::nanoseconds spin{0};
    std::exception_ptr err[NCARD];
    int dev0 = 0;
    explicit CardPool(const std::vector<int>& devs) : dev0(devs[0]) {
#if FNX_HIP
        const long long ms = env_int("FNX_POOL_SPIN_MS", FAST && NCARD == 4 ? 20 : 0);
        if (ms < 0 || ms > 1000) throw std::runtime_error("FNX_POOL_SPIN_MS: want 0 (off) to 1000 ms");
        spin = std::chrono::milliseconds(ms);
#endif
        for (int c = 1; c < NCARD; ++c)
            th.emplace_back([this, c, d = devs[(size_t) c]] {
                const cudaError_t set = cudaSetDevice(d);
                unsigned seen = 0;
                for (;;) {
                    if (spin.count() > 0) {   // the next job is often a few ms away: do not sleep through it
                        const auto end = std::chrono::steady_clock::now() + spin;
                        while (agen.load(std::memory_order_acquire) == seen && !aquit.load(std::memory_order_acquire) &&
                               std::chrono::steady_clock::now() < end)
                            pool_relax();
                    }
                    const std::function<void(int)>* f;
                    {
                        std::unique_lock<std::mutex> l(mu);
                        go.wait(l, [&] { return quit || gen != seen; });
                        if (quit) return;
                        seen = gen;
                        f = job;
                    }
                    try {
                        if (set != cudaSuccess) throw std::runtime_error(std::string("card pool: cudaSetDevice: ") + cudaGetErrorString(set));
                        (*f)(c);
                    } catch (...) { err[c] = std::current_exception(); }
                    std::lock_guard<std::mutex> l(mu);
                    aleft.store(--left, std::memory_order_release);
                    if (left == 0) done.notify_one();
                }
            });
    }
    ~CardPool() {
        {
            std::lock_guard<std::mutex> l(mu);
            quit = true;
            aquit.store(true, std::memory_order_release);
        }
        go.notify_all();
        for (auto& t : th) t.join();
    }
    void run(const std::function<void(int)>& f) {
        {
            std::lock_guard<std::mutex> l(mu);
            job = &f;
            left = NCARD - 1;
            aleft.store(left, std::memory_order_release);
            ++gen;
            agen.store(gen, std::memory_order_release);
            for (auto& e : err) e = nullptr;
        }
        go.notify_all();
        std::exception_ptr e0;
        try {
            ck(cudaSetDevice(dev0), "cudaSetDevice");
            f(0);
        } catch (...) { e0 = std::current_exception(); }
        if (spin.count() > 0) {   // the threads end within microseconds of the caller's own job as a rule
            const auto end = std::chrono::steady_clock::now() + spin;
            while (aleft.load(std::memory_order_acquire) != 0 && std::chrono::steady_clock::now() < end) pool_relax();
        }
        {
            std::unique_lock<std::mutex> l(mu);
            done.wait(l, [&] { return left == 0; });
        }
        if (e0) std::rethrow_exception(e0);
        for (auto& e : err)
            if (e) std::rethrow_exception(e);
    }
};

// the cards' error words behind a window error's message (bits: 1 ring take timed out, 2 ring value out of range, 4 hc barrier timed out,
// 16 non-finite logits). The message names the first bit set on ANY card: a card whose hc barriers stall makes its peers' ring takes time out
std::string Engine::err_words() const {
    std::string s = " (error words by card:";
    for (int c = 0; c < NCARD; ++c) s += " " + std::to_string(h_.err[c]);
    return s + ")";
}

// the hc read's time-out in cycles, as kernels.cu's hc_read sets it (FNX_HC_TIMEOUT_CYCLES; else HcArgs' default)
static long long diag_hc_timeout() {
    try {
        const long long v = env_int("FNX_HC_TIMEOUT_CYCLES", 0);
        return v > 0 ? v : 2000000000LL;
    } catch (...) { return 2000000000LL; }
}

// KFD's evicted_ms of this process, the largest of its cards (-1: no file). Reads the directory anew: for the stall hook's own thread
static long long diag_kfd_max() {
    long long best = -1;
    try {
        const std::string dir = "/sys/class/kfd/kfd/proc/" + std::to_string((long) getpid());
        std::vector<std::string> names;
        if (DIR* d = opendir(dir.c_str())) {
            try {
                while (const dirent* e = readdir(d))
                    if (!std::strncmp(e->d_name, "stats_", 6)) names.push_back(dir + "/" + e->d_name + "/evicted_ms");
            } catch (...) { names.clear(); }
            closedir(d);
        }
        for (const std::string& p : names)
            if (FILE* f = std::fopen(p.c_str(), "r")) {
                long long v = 0;
                if (std::fscanf(f, "%lld", &v) == 1) best = std::max(best, v);
                std::fclose(f);
            }
    } catch (...) {}
    return best;
}

// the host's eviction watch (engine.hpp EvictStats; kernels.hpp WAIT_EVICT_WINDOW_MS says how the cards use the word).
// KFD adds an eviction's time to a card's evicted_ms when that card's queues are back, so a rise says "an eviction ended on a card just now"; the cards
// come back one after another, and under a run of evictions a card can stay stopped while another has come back. The thread and the page live as long
// as the process (a few file reads a sample; evicted_ms is a counter read: it wakes no device). The thread only sleeps, reads and stores: it prints
// nothing (a full pipe must not stop it), and when it ends on an error it leaves "no rise" in the word.
namespace {
struct EvictWatch {
    unsigned long long* word = nullptr;
    int files = 0;
    std::atomic<unsigned long long> bursts{0}, evicted_ms{0}, longest_ms{0}, samples{0}, read_fails{0};
    std::atomic<long long> age_ms{-1};   // from the last rise to the last sample (-1: no rise yet)
    std::atomic<long long> busy_until_ms{0};
    std::atomic<bool> ended{false};
    std::vector<int> fds;
    const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    long long clock_ms() const { return (long long) std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count(); }
    static constexpr unsigned NEVER = 0x3fffffffu;   // the age the word shows for "no rise", and for a rise longer ago than this (12 days)
    void publish(long long now, long long last) {    // last < 0: no rise yet
        const long long age = last < 0 ? (long long) NEVER : std::min<long long>(now - last, (long long) NEVER);
        const unsigned n = (unsigned) now;           // the clock wraps; the cards take the difference
        *(volatile unsigned long long*) word = (unsigned long long) n << 32 | (unsigned) (n - (unsigned) age);   // one aligned store: one sample's two halves
        age_ms.store(last < 0 ? -1 : now - last, std::memory_order_relaxed);
    }

    void run() {
        const size_t n = fds.size();
        std::vector<unsigned long long> prev(n, 0);
        std::vector<char> have(n, 0);
        auto read = [&](size_t i, unsigned long long& v) {
            char buf[32];
            const ssize_t k = pread(fds[i], buf, sizeof buf - 1, 0);
            if (k <= 0) return false;
            buf[k] = 0;
            char* end = nullptr;
            v = std::strtoull(buf, &end, 10);
            return end != buf;
        };
        for (size_t i = 0; i < n; ++i)
            if (!(have[i] = read(i, prev[i]))) read_fails.fetch_add(1, std::memory_order_relaxed);
        long long last = -1;
        bool in_burst = false;
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(clock_ms() < busy_until_ms.load(std::memory_order_relaxed) ? 10 : 50));
            const long long now = clock_ms();
            unsigned long long rise = 0;
            for (size_t i = 0; i < n; ++i) {
                unsigned long long v = 0;
                if (!read(i, v)) {
                    read_fails.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                if (have[i] && v > prev[i]) rise = std::max(rise, v - prev[i]);
                prev[i] = v, have[i] = 1;
            }
            if (rise) {
                last = now;
                evicted_ms.fetch_add(rise, std::memory_order_relaxed);
                if (rise > longest_ms.load(std::memory_order_relaxed)) longest_ms.store(rise, std::memory_order_relaxed);
                if (!in_burst) in_burst = true, bursts.fetch_add(1, std::memory_order_relaxed);
            } else if (in_burst && now - last > 40) in_burst = false;
            publish(now, last);
            samples.fetch_add(1, std::memory_order_relaxed);
        }
    }
};
std::atomic<EvictWatch*> g_evict_watch{nullptr};
std::once_flag g_evict_once;
}  // namespace

const unsigned long long* evict_watch_word() {
    std::call_once(g_evict_once, [] {
        EvictWatch* w = nullptr;
        void* hp = nullptr;
        try {
            w = new EvictWatch;
            const std::string dir = "/sys/class/kfd/kfd/proc/" + std::to_string((long) getpid());
            std::vector<std::string> names;
            if (DIR* d = opendir(dir.c_str())) {
                try {
                    while (const dirent* e = readdir(d))
                        if (!std::strncmp(e->d_name, "stats_", 6)) names.push_back(dir + "/" + e->d_name + "/evicted_ms");
                } catch (...) { names.clear(); }
                closedir(d);
            }
            w->fds.reserve(names.size());
            for (const std::string& p : names) {
                const int fd = open(p.c_str(), O_RDONLY | O_CLOEXEC);
                if (fd >= 0) w->fds.push_back(fd);
            }
            // the cards read the word: page-locked and mapped for every device; kept for the process's life
            if (w->fds.empty() || cudaHostAlloc(&hp, 4096, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess || !hp) throw 0;
            std::memset(hp, 0, 4096);
            w->word = (unsigned long long*) hp;
            w->files = (int) w->fds.size();
            w->publish(0, -1);
            std::thread([w] {
                try {
                    w->run();
                } catch (...) {}
                w->publish(w->clock_ms(), -1);   // the thread ends (it never does but on an error): no rise for the cards from here on
                w->ended.store(true);
            }).detach();
            g_evict_watch.store(w);
        } catch (...) {
            cudaGetLastError();
            if (hp) cudaFreeHost(hp);
            if (w) {
                for (int fd : w->fds) close(fd);
                delete w;
            }
        }
    });
    const EvictWatch* w = g_evict_watch.load();
    return w ? w->word : nullptr;
}

void evict_watch_touch() {
    if (EvictWatch* w = g_evict_watch.load(std::memory_order_relaxed)) w->busy_until_ms.store(w->clock_ms() + 3000, std::memory_order_relaxed);
}

EvictStats evict_watch_stats() {
    EvictStats s;
    if (const EvictWatch* w = g_evict_watch.load()) {
        s.on = !w->ended.load(), s.files = w->files;
        s.bursts = w->bursts.load(), s.evicted_ms = w->evicted_ms.load(), s.longest_ms = w->longest_ms.load(), s.samples = w->samples.load();
        s.read_fails = w->read_fails.load();
        s.age_ms = w->age_ms.load();
    }
    return s;
}

static int diag_us(std::chrono::steady_clock::duration d) { return (int) std::chrono::duration_cast<std::chrono::microseconds>(d).count(); }

void Engine::on_cards(const std::function<void(int)>& f) {
    if (!pool_) {
        std::vector<int> devs;
        for (CardBuf* b : cb_) devs.push_back(b->dev);
        pool_ = std::make_shared<CardPool>(devs);
    }
    // when the call started and each card's job began and ended, kept for the last DIAG_RUNS calls (diag_report prints them)
    DiagRun& r = diag_runs_[diag_run_n_ % DIAG_RUNS];
    r.t0 = std::chrono::steady_clock::now();
    r.wall_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    for (int c = 0; c < NCARD; ++c) r.start_us[c] = r.took_us[c] = r.gl_start_us[c] = r.gl_took_us[c] = r.sy_start_us[c] = r.sy_took_us[c] = -1;
    r.win = diag_win_pending_;   // not 0: this call launches main window r.win's graphs (diag_graph_launch and diag_sync stamp them)
    diag_win_pending_ = 0;
    if (r.win) {
        diag_win_run_ = &r;
        if (diag_ext_pending_) diag_ext_log();   // the cards' counts of waits that began again, brought home by an earlier main window
        if (diag_kfd_last_.n < 0 || r.t0 - diag_kfd_last_.at > std::chrono::seconds(DIAG_KFD_S)) {   // the queues' evicted time, a few times a minute
            diag_kfd_prev_ = diag_kfd_last_;
            diag_kfd_sample(diag_kfd_last_);
            if (diag_kfd_first_.n < 0) diag_kfd_first_ = diag_kfd_last_;
            diag_ext_due_ = diag_ext_h_ != nullptr;   // and this window's launch jobs queue a copy of the cards' counts
        }
    }
    ++diag_run_n_;
    evict_watch_touch();
    struct Ctx {
        DiagRun* r;
        const std::function<void(int)>* f;
    } ctx{&r, &f};
    const std::function<void(int)> timed = [p = &ctx](int c) {   // one pointer: no allocation
        const auto a = std::chrono::steady_clock::now();
        p->r->start_us[c] = diag_us(a - p->r->t0);
        (*p->f)(c);
        p->r->took_us[c] = diag_us(std::chrono::steady_clock::now() - a);
    };
    pool_->run(timed);
    if (r.win && diag_ext_due_) {
        diag_ext_due_ = false;
        diag_ext_pending_ = true;
    }
    int end_us = 0;
    for (int c = 0; c < NCARD; ++c) end_us = std::max(end_us, r.start_us[c] + r.took_us[c]);
    if (end_us > 20000) {
        ++diag_slow_n_;
        diag_slow_max_us_ = std::max(diag_slow_max_us_, end_us);
    }
}

// FNX_DIAG_STALL="card,window,ms[,mode]" (a test hook, off unless set): once, at card `card`'s first graph launch after `window` main
// windows were launched, mode host (default): the card's launch thread sleeps `ms` before the launch (a late host launch); dev: a kernel holds the
// card's stream for `ms` before the graph (a card that stalls before its window); tag: the card's tag word is one too high for that window (a
// protocol fault: its words carry a tag no peer waits for; single-lane windows only, `ms` is not used); evict: for `ms` a thread throws away one page
// of a host exchange buffer of its own every millisecond (madvise MADV_DONTNEED, then a write). When the driver follows that buffer as a user pointer,
// each time is a real eviction: the driver stops this process's queues on EVERY card and restores them (what page migration does). No root needed.
// The thread's last line says how far the largest evicted_ms of the cards rose.
void Engine::diag_stall(int c, CardBuf& b, bool single) {
    struct Cfg {
        int card = -1, ms = 0, mode = 0;
        unsigned long window = 0;
    };
    static const Cfg cfg = [] {
        Cfg k;
        const char* e = std::getenv("FNX_DIAG_STALL");
        if (!e || !*e) return k;
        char mode[16] = "host";
        long card = -1, ms = 0;
        unsigned long win = 0;
        const int n = std::sscanf(e, "%ld,%lu,%ld,%15s", &card, &win, &ms, mode);
        const int m = !std::strcmp(mode, "host") ? 0 : !std::strcmp(mode, "dev") ? 1 : !std::strcmp(mode, "tag") ? 2 : !std::strcmp(mode, "evict") ? 3 : -1;
        if (n < 3 || card < 0 || card >= NCARD || ms < 0 || ms > (m == 1 ? 30000 : 60000) || m < 0)
            throw std::runtime_error(std::string("FNX_DIAG_STALL: want card,window,ms[,host|dev|tag|evict] with card 0..") + std::to_string(NCARD - 1) +
                                     " and ms 0..60000 (dev: 0..30000), got " + e);
        k.card = (int) card, k.window = win, k.ms = (int) ms, k.mode = m;
        return k;
    }();
    static std::atomic<bool> fired{false};
    if (c < 0) return;   // diag_load's call: the setting is read (and refused) at load
    if (cfg.card != c || diag_win_ < cfg.window || fired.load(std::memory_order_relaxed)) return;
    if (cfg.mode == 2 && !single) return;   // the batched windows' tags are the lanes' own
    if (fired.exchange(true)) return;
    std::fprintf(stderr, "sfnx: diag: FNX_DIAG_STALL fires: card %d, main window %lu, mode %s, %d ms\n", c, diag_win_,
                 cfg.mode == 0 ? "host" : cfg.mode == 1 ? "dev" : cfg.mode == 2 ? "tag" : "evict", cfg.ms);
    if (cfg.mode == 0) std::this_thread::sleep_for(std::chrono::milliseconds(cfg.ms));
    else if (cfg.mode == 1) diag_spin((long long) ((double) cfg.ms * (diag_wall_khz_[c] > 0 ? diag_wall_khz_[c] : 100000.0)), b.st);
    else if (cfg.mode == 2) diag_poke((unsigned*) (b.ctl + 1), 1u, b.st);
    else {
        // a buffer of the kind the cards' exchange buffers are (page-locked for the runtime, mapped to the cards); never freed (a test hook, once a process)
        void* hp = nullptr;   // not the engine's to release: the thread below can outlive a window error
        ck(cudaHostAlloc(&hp, (size_t) 1 << 20, cudaHostAllocPortable | cudaHostAllocMapped), "stall hook's buffer");
        std::memset(hp, 0, (size_t) 1 << 20);
        char* pad = (char*) hp;
        const int ms = cfg.ms;
        std::thread([pad, ms] {
            const long pg = sysconf(_SC_PAGESIZE);
            char* p = (char*) (((uintptr_t) pad + ((size_t) 1 << 19)) & ~(uintptr_t) (pg - 1));   // one whole page in the middle of the buffer
            const long long e0 = diag_kfd_max();
            const auto t0 = std::chrono::steady_clock::now();
            long n = 0, bad = 0;
            int bad_errno = 0;
            while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(ms)) {
                if (madvise(p, (size_t) pg, MADV_DONTNEED) != 0) ++bad, bad_errno = errno;
                *(volatile char*) p = 1;   // a new page comes in
                ++n;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(300));   // the last restore
            const long long e1 = diag_kfd_max();
            std::fprintf(stderr, "sfnx: diag: FNX_DIAG_STALL evict: %ld madvise calls in %d ms (%ld refused, last errno %d); the largest evicted_ms of the "
                                 "cards rose by %lld ms (-1: no file)%s\n", n, ms, bad, bad_errno, e0 >= 0 && e1 >= 0 ? e1 - e0 : -1LL,
                         e0 >= 0 && e1 == e0 ? ": that maximum did not rise in this time" : "");
            std::fflush(stderr);
        }).detach();
    }
}

// a main window's graph launch on card c, with the call's start and length kept in the on_cards() call's record (two clock reads)
void Engine::diag_graph_launch(int c, CardBuf& b, void* exec, bool single) {
    diag_stall(c, b, single);
    DiagRun& r = diag_runs_[(diag_run_n_ - 1) % DIAG_RUNS];
    const auto a = std::chrono::steady_clock::now();
    ck(cudaGraphLaunch((cudaGraphExec_t) exec, b.st), "graph launch");
    if (r.win) {
        r.gl_start_us[c] = diag_us(a - r.t0);
        r.gl_took_us[c] = diag_us(std::chrono::steady_clock::now() - a);
        if (diag_ext_due_)   // once in DIAG_KFD_S seconds: the card's counts of waits that began again, behind the window on its stream
            ck(cudaMemcpyAsync(diag_ext_h_ + c, err_word(b) + DG_EXT_N, sizeof(unsigned), cudaMemcpyDeviceToHost, b.st), "extended waits");
    }
}

// one line when the driver evicted, or waits began again after an eviction, since the last look (the cards' words came home with
// an earlier, synchronized window; the watch's numbers are its own)
void Engine::diag_ext_log() {
    diag_ext_pending_ = false;
    unsigned d[NCARD], any = 0;
    for (int c = 0; c < NCARD; ++c) {
        d[c] = diag_ext_h_[c] - diag_ext_seen_[c];   // unsigned: a wrapped count still gives the difference
        diag_ext_seen_[c] = diag_ext_h_[c];
        any |= d[c];
    }
    const EvictStats e = evict_watch_stats();
    if (!any && e.evicted_ms == diag_ext_evicted_seen_) return;
    char t[400];
    std::snprintf(t, sizeof t, "sfnx: diag: the driver stopped this process's queues: kfd evicted_ms rose by %llu ms in %llu runs of evictions since the last "
                               "line (the watch's sum of each sample's largest rise; in all %llu ms in %llu runs, the largest rise of one sample %llu ms); "
                               "waits (threads) that began again at their time-out, by card:",
                  e.evicted_ms - diag_ext_evicted_seen_, e.bursts - diag_ext_bursts_seen_, e.evicted_ms, e.bursts, e.longest_ms);
    diag_ext_evicted_seen_ = e.evicted_ms, diag_ext_bursts_seen_ = e.bursts;
    std::string line = t;
    for (int c = 0; c < NCARD; ++c) line += " +" + std::to_string(d[c]);
    std::fprintf(stderr, "%s\n", line.c_str());
}

// a main window's wait for card c's stream, stamped in the window's on_cards() record
void Engine::diag_sync(int c, CardBuf& b, const char* what) {
    const auto a = std::chrono::steady_clock::now();
    ck(cudaStreamSynchronize(b.st), what);
    if (DiagRun* r = diag_win_run_) {
        r->sy_start_us[c] = diag_us(a - r->t0);
        r->sy_took_us[c] = diag_us(std::chrono::steady_clock::now() - a);
    }
}

// KFD's evicted_ms of this process, one value a card it has open (engine.hpp DiagKfd). Never throws; n = 0 where the files do not exist.
void Engine::diag_kfd_sample(DiagKfd& k) {
    k.at = std::chrono::steady_clock::now();
    k.win = diag_win_;
    k.n = 0;
    try {
        if (!diag_kfd_scanned_) {
            diag_kfd_scanned_ = true;
            const std::string dir = "/sys/class/kfd/kfd/proc/" + std::to_string((long) getpid());
            std::vector<std::string> names;
            if (DIR* d = opendir(dir.c_str())) {
                try {
                    while (const dirent* e = readdir(d))
                        if (!std::strncmp(e->d_name, "stats_", 6)) names.push_back(dir + "/" + e->d_name + "/evicted_ms");
                } catch (...) { names.clear(); }
                closedir(d);
            }
            std::sort(names.begin(), names.end());
            if (names.size() > (size_t) DIAG_KFD_MAX) names.resize(DIAG_KFD_MAX);
            diag_kfd_paths_ = std::move(names);
        }
        for (const std::string& p : diag_kfd_paths_) {
            unsigned long long v = 0;
            bool ok = false;
            if (FILE* f = std::fopen(p.c_str(), "r")) {
                ok = std::fscanf(f, "%llu", &v) == 1;
                std::fclose(f);
            }
            if (!ok) {   // a file that cannot be read: no sample (a 0 would read as a rise later)
                k.n = -1;
                return;
            }
            k.ms[k.n++] = v;
        }
        if (k.n == 0) k.n = -1;
    } catch (...) { k.n = -1; }
}

// a window error's report, printed before the error is thrown. One line a card from its error block (kernels.hpp ErrDiag: the first
// time-out on that card), then the host's side: the last on_cards() calls and each card's memory. Never throws.
void Engine::diag_report(const char* where) {
    try {
        using namespace std::chrono;
        const auto now_s = steady_clock::now();
        const auto now_w = system_clock::now();
        const std::time_t tt = system_clock::to_time_t(now_w);
        struct tm tmv {};
        localtime_r(&tt, &tmv);
        char ts[48];
        std::strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S", &tmv);
        const int tms = (int) (duration_cast<milliseconds>(now_w.time_since_epoch()).count() % 1000);
        std::fprintf(stderr, "sfnx: diag: %s.%03d %s error: main windows launched %lu, largest tag %u, on_cards calls %lu\n", ts, tms, where, diag_win_,
                     tag_max_, diag_run_n_);
        static const char* const kinds[] = {"?", "decode ring sum", "decode ring sum (fp32)", "attention merge", "chunk ring flag", "prompt sum flag",
                                            "hc grid barrier", "hc ticket gate"};
        const auto site = [](unsigned tag) {   // a ring tag: window tag * 256 + call + 1
            char b[96];
            const int call = (int) (tag & 255u) - 1;
            if (tag == 0) std::snprintf(b, sizeof b, "none");
            else if (call >= 128) std::snprintf(b, sizeof b, "window tag %u call %d (layer %d's attention merge)", tag >> 8, call, call - 128);
            else std::snprintf(b, sizeof b, "window tag %u call %d (layer %d, %s half)", tag >> 8, call, call / 2, call & 1 ? "FFN" : "mixer");
            return std::string(b);
        };
        unsigned blk[NCARD][ERR_WORDS];
        bool have[NCARD];
        double ago_ms[NCARD];   // the card's first time-out, ms before its clock was read for this report (by the card's steady counter); < 0: not known
        double host_ago_ms[NCARD], host_ago_err_ms[NCARD];   // and before this report began: at least, and the width of the host's uncertainty
        long long over_cyc[NCARD];
        int dev0 = -1;
        if (cudaGetDevice(&dev0) != cudaSuccess) dev0 = -1;
        for (int c = 0; c < NCARD; ++c) {
            CardBuf& b = *cb_[c];
            ago_ms[c] = host_ago_ms[c] = -1.0;
            host_ago_err_ms[c] = 0.0;
            over_cyc[c] = 0;
            have[c] = cudaSetDevice(b.dev) == cudaSuccess;
            steady_clock::time_point read_a = steady_clock::now(), read_b = read_a;
            if (have[c]) {
                try {   // the card's steady counter now (the window is over: the stream is idle), to date the record on the host's time line
                    diag_clock(err_word(b), b.st);
                    have[c] = cudaStreamSynchronize(b.st) == cudaSuccess;
                    read_b = steady_clock::now();
                } catch (...) { cudaGetLastError(); }
            }
            have[c] = have[c] && cudaMemcpy(blk[c], err_word(b), sizeof blk[c], cudaMemcpyDeviceToHost) == cudaSuccess;
            float win_ms = -1.f;
            if (cudaEventElapsedTime(&win_ms, b.ev0, b.ev1) != cudaSuccess) win_ms = -1.f;
            cudaGetLastError();
            char pci[64] = "?";
            long vram = -1, gtt = -1;
#if FNX_HIP
            if (hipDeviceGetPCIBusId(pci, (int) sizeof pci, b.dev) != hipSuccess) std::snprintf(pci, sizeof pci, "?");
            for (char* q = pci; *q; ++q) *q = (char) std::tolower((unsigned char) *q);
            for (int k = 0; k < 2; ++k) {
                const std::string path = std::string("/sys/bus/pci/devices/") + pci + (k ? "/mem_info_gtt_used" : "/mem_info_vram_used");
                if (FILE* f = std::fopen(path.c_str(), "r")) {
                    unsigned long long v = 0;
                    if (std::fscanf(f, "%llu", &v) == 1) (k ? gtt : vram) = (long) (v >> 20);
                    std::fclose(f);
                }
            }
#endif
            if (!have[c]) {
                std::fprintf(stderr, "sfnx: diag: card %d (device %d, %s): the error block could not be read; vram %ld MiB, gtt %ld MiB\n", c, b.dev, pci, vram, gtt);
                continue;
            }
            const unsigned* d = blk[c];
            std::string line;
            char t[512];
            std::snprintf(t, sizeof t, "sfnx: diag: card %d (device %d, %s): error word %u (read now from the card); ", c, b.dev, pci, d[0]);
            line += t;
            if (d[DG_CLAIM] == 0) line += "no time-out record; ";
            else {
                const unsigned kind = (d[DG_CLAIM] >> 28) <= DK_RING_CHAIN ? d[DG_CLAIM] >> 28 : 0, src = (d[DG_CLAIM] >> 24) & 15u, elem = d[DG_CLAIM] & 0xffffffu;
                const long long waited = (long long) (((unsigned long long) d[DG_CYC_HI] << 32) | d[DG_CYC_LO]);
                const unsigned long long wall = ((unsigned long long) d[DG_WALL_HI] << 32) | d[DG_WALL_LO];
                const unsigned long long now_c = ((unsigned long long) d[DG_NOW_HI] << 32) | d[DG_NOW_LO];
                const double khz = diag_wall_khz_[c];
                // clock64's rate in this report's spin (the card is idle now: the rate during the wait can have been another). A spin that the driver
                // stopped in the middle reads far off (clock64 does not keep its rate while the queue is off the card): then the rate at load is used
                const double spin_mhz = khz > 0 && d[DG_CAL0_WALL] > 0 ? (double) d[DG_CAL0_CYC] * khz / d[DG_CAL0_WALL] / 1000.0 : 0.0;
                const double load_mhz = diag_clock_mhz_[c];
                const bool spin_ok = spin_mhz > 0 && (load_mhz <= 0 || (spin_mhz > 0.6 * load_mhz && spin_mhz < 1.5 * load_mhz));
                const double mhz = spin_ok ? spin_mhz : load_mhz;
                const bool ringk = kind == DK_RING || kind == DK_RING_F32 || kind == DK_RING_CHAIN;
                const long long limit = ringk ? diag_ring_timeout_[c] : kind == DK_HC_BARRIER || kind == DK_HC_GATE ? diag_hc_timeout() : RING_TIMEOUT_CYCLES;
                std::snprintf(t, sizeof t, "first time-out: %s, ", kinds[kind]);
                line += t;
                if (ringk || kind == DK_ATT_MERGE) {
                    line += "waited for " + site(d[DG_TAG]) + ", the word held " + (d[DG_SEEN_HI] ? site(d[DG_SEEN_HI]) : std::string("no tag (never written)"));
                    std::snprintf(t, sizeof t, " payload 0x%08x, ", d[DG_SEEN_LO]);
                    line += t;
                    if (src != DG_NO_SRC) {
                        std::snprintf(t, sizeof t, "from card %u, element %u (row %u, pair %u), ", src, elem, elem / (unsigned) (N / 2), elem % (unsigned) (N / 2));
                        line += t;
                    }
                } else if (kind == DK_PROMPT_SUM) {
                    std::snprintf(t, sizeof t, "waited for card %u's flag to reach sequence %u, it held %u, block %u, ", src, d[DG_TAG], d[DG_SEEN_HI], elem);
                    line += t;
                } else if (kind == DK_HC_BARRIER) {
                    std::snprintf(t, sizeof t, "the barrier's counter stood at %u, the generation began at %u, grid %u (%u of %u blocks had come), block %u, ",
                                  d[DG_SEEN_HI], d[DG_TAG], d[DG_SEEN_LO], d[DG_SEEN_HI] >= d[DG_TAG] ? d[DG_SEEN_HI] - d[DG_TAG] : 0u, d[DG_SEEN_LO], elem);
                    line += t;
                } else if (kind == DK_HC_GATE) {
                    std::snprintf(t, sizeof t, "%u of %u down tickets done, %u tickets taken, block %u, ", d[DG_SEEN_HI], d[DG_TAG], d[DG_SEEN_LO], elem);
                    line += t;
                } else {
                    std::snprintf(t, sizeof t, "waited for %u, saw %u (%u), block %u, ", d[DG_TAG], d[DG_SEEN_HI], d[DG_SEEN_LO], elem);
                    line += t;
                }
                // a wait that spins checks its clock every few hundred cycles: one that gives up far past its time-out did not run in between
                const long long over = waited - limit;
                std::snprintf(t, sizeof t, "waited %lld cycles, %lld over its time-out of %lld", waited, over, limit);
                line += t;
                if (mhz > 0) {
                    if (spin_ok)
                        std::snprintf(t, sizeof t, " (%.0f ms and %.1f ms over at %.0f MHz, clock64's rate in this report's spin; %.0f MHz at load)",
                                      (double) waited / (mhz * 1000.0), (double) over / (mhz * 1000.0), mhz, load_mhz);
                    else
                        std::snprintf(t, sizeof t, " (%.0f ms and %.1f ms over at %.0f MHz, clock64's rate at load; this report's spin read %.0f MHz, too "
                                                   "far from it to be used)",
                                      (double) waited / (mhz * 1000.0), (double) over / (mhz * 1000.0), mhz, spin_mhz);
                    line += t;
                }
                over_cyc[c] = over;
                // (how long the stop was in wall time is not in the cycles: clock64 does not keep its rate while the queue is off the card)
                // with the eviction watch on, the cycles count from the wait's last start (it begins again after an eviction)
                line += over > 2000000 ? ": its last poll came that late, so the waiting thread did not run right before it (likely the card's queue was off "
                                         "the card: an eviction or a preemption)"
                                       : ": the thread was polling when it gave up (the word was late or lost; stops earlier in the wait do not show here)";
                if (khz > 0 && now_c >= wall) {
                    ago_ms[c] = (double) (now_c - wall) / khz;
                    // the card's clock was read between read_a and read_b on the host's clock
                    host_ago_ms[c] = ago_ms[c] - duration<double, std::milli>(read_b - now_s).count();
                    host_ago_err_ms[c] = duration<double, std::milli>(read_b - read_a).count();
                    std::snprintf(t, sizeof t, "; it timed out %.1f to %.1f ms before this report began (the card's steady counter)", host_ago_ms[c],
                                  host_ago_ms[c] + host_ago_err_ms[c]);
                    line += t;
                }
                std::snprintf(t, sizeof t, "; waits timed out on this card %u, cards whose words were missed: mask 0x%x; ", d[DG_COUNT], d[DG_SRCMASK]);
                line += t;
            }
            if (d[DG_WATCH_LO] | d[DG_WATCH_HI]) {
                std::snprintf(t, sizeof t, "waits (threads) on this card began again after an eviction %u times; ", d[DG_EXT_N]);
                line += t;
            } else line += "waits fail at their time-out (no eviction watch); ";
            line += "last ring sum entered: " + site(d[DG_PROG]);
            std::snprintf(t, sizeof t, "; the window on the card %.1f ms; vram %ld MiB, gtt %ld MiB", win_ms, vram, gtt);
            line += t;
            std::fprintf(stderr, "%s\n", line.c_str());
        }
        if (dev0 >= 0) (void) cudaSetDevice(dev0);
        cudaGetLastError();
        // a reading of the decode ring records. After a card's first ring time-out its later sums return at once and push nothing, so the cards that
        // time out later only follow; the earliest call's records are the ones that say something.
        {
            // the one-hop sums only (DK_RING): on the ring that passes sums from card to card one late card shows as several sources
            const auto ring = [&](int c) { return have[c] && (blk[c][DG_CLAIM] >> 28) == DK_RING; };
            bool stopped = false;
            for (int c = 0; c < NCARD; ++c) stopped = stopped || (have[c] && blk[c][DG_CLAIM] && over_cyc[c] > 2000000);
            if (stopped)
                std::fprintf(stderr, "sfnx: diag: reading: a wait's last poll came late, so its thread did not run right before it gave up: likely the card's "
                                     "queue was stopped (see the kfd line: a driver eviction stops all cards). The lines below then say only which words were "
                                     "open.\n");
            int n_ring = 0;
            unsigned wtag = 0, lowest = 0xffffffffu;
            bool tags_differ = false, newer = false;
            for (int c = 0; c < NCARD; ++c) {
                if (!ring(c)) continue;
                const unsigned* d = blk[c];
                if (n_ring++ == 0) wtag = d[DG_TAG] >> 8;
                else tags_differ = tags_differ || (d[DG_TAG] >> 8) != wtag;
                newer = newer || d[DG_SEEN_HI] > d[DG_TAG];   // the word held a tag past the one waited for
                lowest = std::min(lowest, d[DG_TAG] & 255u);
            }
            if (n_ring) {
                std::string say = "sfnx: diag: reading: ";
                if (tags_differ || newer) {
                    say += tags_differ ? "the cards waited under different window tags" : "a word held a newer tag than the one its reader waited for";
                    say += ": a tag or protocol fault (a card with a wrong tag, or words of another window), not a card that was only late; compare "
                           "'waited for' and 'the word held' in the cards' lines";
                } else {
                    std::string at, later;
                    unsigned mask = 0;
                    int n_at = 0;
                    for (int c = 0; c < NCARD; ++c) {
                        if (!ring(c)) continue;
                        if ((blk[c][DG_TAG] & 255u) == lowest) at += " " + std::to_string(c), mask |= blk[c][DG_SRCMASK], ++n_at;
                        else later += " " + std::to_string(c);
                    }
                    const bool one_src = mask && !(mask & (mask - 1u));
                    int src = 0;
                    while (one_src && !((mask >> src) & 1u)) ++src;
                    char h[16];
                    std::snprintf(h, sizeof h, "%x", mask);
                    say += "window tag " + std::to_string(wtag) + ": the lowest call with a ring time-out is " + std::to_string((int) lowest - 1) + ", on card(s)" + at +
                           " (cards whose words they missed: mask 0x" + h + ")";
                    if (!later.empty()) say += "; card(s)" + later + " timed out at a later call (they follow: a card with a time-out stops pushing)";
                    if (one_src && n_at >= 2)
                        say += ". This fits: card " + std::to_string(src) + " was late at that sum by more than the time-out (a stall on the card, or a late launch: "
                               "see its job and graph launch times below)" +
                               (lowest == 1u ? "; call 0 is the window's first sum: the card had not reached it (a late launch, or a stall in the work before it)"
                                             : "; the call is past 0, so the card stopped inside its window");
                    else if (one_src && n_at == 1)
                        say += ". This fits: words from card " + std::to_string(src) + " to that one card did not arrive while the other cards got through the sum "
                               "(a lost or late word on one path, or card " + std::to_string(src) + " ran only a part of its sum)";
                    else say += ". No single late card or path fits (see the cards' lines)";
                }
                std::fprintf(stderr, "%s\n", say.c_str());
            }
        }
        {   // the cards' first time-outs on one time line: each card's steady counter, read in this report between two host clock reads
            std::string say;
            int first = -1;
            for (int c = 0; c < NCARD; ++c) {
                if (ago_ms[c] < 0) continue;
                char t[64];
                std::snprintf(t, sizeof t, " card %d %.1f..%.1f", c, host_ago_ms[c], host_ago_ms[c] + host_ago_err_ms[c]);
                say += t;
                if (first < 0 || host_ago_ms[c] > host_ago_ms[first]) first = c;
            }
            if (first >= 0) {
                bool clear = true;   // the first one's earliest bound is before every other card's latest bound
                for (int c = 0; c < NCARD; ++c)
                    if (c != first && ago_ms[c] >= 0 && host_ago_ms[c] + host_ago_err_ms[c] >= host_ago_ms[first]) clear = false;
                std::fprintf(stderr, "sfnx: diag: order: a card's first time-out, ms before this report began:%s; %s\n", say.c_str(),
                             clear ? ("the first was card " + std::to_string(first) + "'s").c_str() : "the ranges overlap: the order is not resolved");
            }
        }
        {   // the driver's evicted time: now against the last samples
            DiagKfd now;
            diag_kfd_sample(now);
            if (now.n <= 0) std::fprintf(stderr, "sfnx: diag: kfd: no evicted_ms files for this process (/sys/class/kfd/kfd/proc/%ld/stats_*/evicted_ms)\n", (long) getpid());
            else {
                const auto line = [&](const DiagKfd& k, const char* name) {
                    if (k.n <= 0 || k.n != now.n) return std::string();
                    unsigned long long worst = 0;
                    for (int i = 0; i < now.n; ++i) worst = std::max(worst, now.ms[i] - std::min(now.ms[i], k.ms[i]));
                    char t[160];
                    std::snprintf(t, sizeof t, " +%llu ms since %s (%.1f s and %lu main windows ago);", worst, name, duration<double>(now.at - k.at).count(),
                                  diag_win_ - k.win);
                    return std::string(t);
                };
                std::string say = "sfnx: diag: kfd: time the driver kept this process's queues off the cards (evicted_ms, by card file):";
                for (int i = 0; i < now.n; ++i) say += " " + std::to_string(now.ms[i]);
                say += " ms in all;" + line(diag_kfd_last_, "the last sample") + line(diag_kfd_prev_, "the sample before") + line(diag_kfd_first_, "the first main window");
                say += " evicted_ms rises when an eviction ends on a card. An eviction stops this process's kernels on every card; the cards stop and come "
                       "back one after another, so a card can run and wait while its peer is still stopped";
                std::fprintf(stderr, "%s\n", say.c_str());
            }
        }
        {   // the eviction watch's own numbers
            const EvictStats e = evict_watch_stats();
            if (e.files == 0) std::fprintf(stderr, "sfnx: diag: eviction watch: none (it could not start): a wait fails at its time-out\n");
            else {
                char a[64] = "no rise yet";
                if (e.age_ms >= 0) std::snprintf(a, sizeof a, "the last rise was %lld ms before its last sample", e.age_ms);
                std::fprintf(stderr, "sfnx: diag: eviction watch (%d files, %llu samples, %llu failed reads%s): %s; its sum of each sample's largest rise is "
                                     "%llu ms in %llu runs, the largest rise of one sample %llu ms. Waits begin again after a rise: %s\n",
                             e.files, e.samples, e.read_fails, e.on ? "" : "; its thread ENDED: the cards see no rise from then on", a, e.evicted_ms, e.bursts,
                             e.longest_ms, diag_ext_on_ ? "on" : "off (FNX_WAIT_EVICT=0)");
            }
        }
        // the host: the last on_cards() calls (the newest first)
        const unsigned long n = std::min<unsigned long>(diag_run_n_, 8);
        for (unsigned long k = 0; k < n; ++k) {
            const DiagRun& r = diag_runs_[(diag_run_n_ - 1 - k) % DIAG_RUNS];
            std::string line;
            char t[256];
            if (r.win) std::snprintf(t, sizeof t, "main window %lu's launch", r.win);
            else std::snprintf(t, sizeof t, "not a main window's launch");
            const std::string what = t;
            std::snprintf(t, sizeof t, "sfnx: diag: host: on_cards call -%lu (%s) started %.3f ms before this report; by card, ms from the call's start:", k,
                          what.c_str(), duration<double, std::milli>(now_s - r.t0).count());
            line += t;
            bool slow = false;
            for (int c = 0; c < NCARD; ++c) {
                if (r.win && r.gl_start_us[c] >= 0)
                    std::snprintf(t, sizeof t, " [%d: job +%.3f took %.3f, graph launch +%.3f took %.3f, sync +%.3f waited %.3f]", c, r.start_us[c] / 1000.0,
                                  r.took_us[c] / 1000.0, r.gl_start_us[c] / 1000.0, r.gl_took_us[c] / 1000.0, r.sy_start_us[c] / 1000.0, r.sy_took_us[c] / 1000.0);
                else std::snprintf(t, sizeof t, " [%d: job +%.3f took %.3f]", c, r.start_us[c] / 1000.0, r.took_us[c] / 1000.0);
                line += t;
                slow = slow || r.start_us[c] > 20000 || r.took_us[c] > 20000 || r.took_us[c] < 0;
            }
            if (slow) line += "  <-- a job that began or took over 20 ms (-0.001: it did not begin or return)";
            std::fprintf(stderr, "%s\n", line.c_str());
        }
        std::fprintf(stderr, "sfnx: diag: host: on_cards calls that ended over 20 ms after their start, since the engine's start: %lu (the longest %.1f ms)\n",
                     diag_slow_n_, diag_slow_max_us_ / 1000.0);
    } catch (...) { std::fprintf(stderr, "sfnx: diag: the report itself failed\n"); }
    std::fflush(stderr);
}

unsigned Engine::ring_tag() {
    tags_room(1);
    return ++tag_max_;
}

unsigned Engine::take_tags(int n) {
    tags_room((unsigned) n + 1u);   // the parity step below can take one tag more than n
    unsigned base = tag_max_ + 1u;
    if ((base & 1u) != ((tag_ + 1u) & 1u)) ++base;
    tag_ = base + (unsigned) n - 1u;
    tag_max_ = tag_;
    return base;
}

void Engine::release_pages() {
    LaneHost& L = lanes_[(size_t) cur_];
    for (int p : L.pages) free_pages_.push_back(p);
    L.pages.clear();
    std::sort(free_pages_.begin(), free_pages_.end(), std::greater<int>());
}

void Engine::use_lane(int i) {
    if (i < 0 || i >= (int) lanes_.size()) throw std::runtime_error("use_lane: no lane " + std::to_string(i));
    if (i == cur_) return;
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("use_lane: a window is not committed");
    LaneHost& was = lanes_[(size_t) cur_];
    was.pos = pos_;
    was.tag = tag_;
    was.history = std::move(history_);
    cur_ = i;
    LaneHost& now = lanes_[(size_t) i];
    pos_ = now.pos;
    tag_ = now.tag;
    history_ = std::move(now.history);
    now.history.clear();
    for (CardBuf* b : cb_) card_lane(*b, b->lanes[(size_t) i]);
}

void Engine::set_images(const std::vector<Image>& images) {
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("set_images: a window is not committed");
    int end = 0;
    for (const Image& im : images) {
        if (im.nx < 1 || im.ny < 1 || im.n != im.nx * im.ny) throw std::runtime_error("set_images: an image's cells are not its grid");
        if (im.cell < end || im.cell + im.n > max_ctx_) throw std::runtime_error("set_images: images out of order or past the context");
        end = im.cell + im.n;
    }
    LaneHost& L = lanes_[(size_t) cur_];
    const bool same = L.images.size() == images.size() &&
                      std::equal(images.begin(), images.end(), L.images.begin(), [](const Image& a, const Image& b) {
                          return a.cell == b.cell && a.n == b.n && a.nx == b.nx && a.ny == b.ny;
                      });
    L.images = images;
    if (same) return;   // the rows change; the positions do not
    if (images.empty()) {
        for (CardBuf* b : cb_) b->mrope = b->lanes[(size_t) cur_].mrope = nullptr;
        return;
    }
    std::vector<int> tab((size_t) max_ctx_ * 3);
    int p = 0;
    size_t k = 0;
    for (int c = 0; c < max_ctx_;) {
        if (k < images.size() && c == images[k].cell) {
            const Image& im = images[k++];
            for (int j = 0; j < im.n; ++j) {
                tab[(size_t) (c + j) * 3] = p;
                tab[(size_t) (c + j) * 3 + 1] = p + j / im.nx;
                tab[(size_t) (c + j) * 3 + 2] = p + j % im.nx;
            }
            c += im.n;
            p += std::max(im.nx, im.ny);
        } else {
            tab[(size_t) c * 3] = tab[(size_t) c * 3 + 1] = tab[(size_t) c * 3 + 2] = p++;
            ++c;
        }
    }
    for (CardBuf* b : cb_) {
        ck(cudaSetDevice(b->dev), "set");
        LaneDev& D = b->lanes[(size_t) cur_];
        if (!D.mrope_buf) D.mrope_buf = dalloc<int>(tab.size());
        ck(cudaMemcpy(D.mrope_buf, tab.data(), tab.size() * sizeof(int), cudaMemcpyHostToDevice), "rotary table");
        b->mrope = D.mrope = D.mrope_buf;
    }
}

bool Engine::precap() const {
#if FNX_HIP
    static const bool on = env_int("FNX_PRECAP", 1) != 0;   // AMD: on (measured +18% with FNX_CAPTURE_AFTER=2 at 8 users, same bits; RESULTS-HIP.md section 8)
#else
    static const bool on = env_int("FNX_PRECAP", 0) != 0;   // NVIDIA: off in 0.2.9 (frozen)
#endif
    return on;
}

namespace {
void swap_slot(CardBuf& b, int T, CardBuf::GraphEntry& e) {
    CardBuf::GraphSlot& s = b.batch_graphs[(size_t) T];
    std::swap(s.exec, e.exec);
    std::swap(s.key, e.key);
    std::swap(b.lane_rows[(size_t) T], e.lane_rows);
    std::swap(b.lane_gdn[(size_t) T], e.lane_gdn);
    std::swap(b.seg_ctl[(size_t) T], e.seg_ctl);
    std::swap(b.seg_row0[(size_t) T], e.seg_row0);
    std::swap(b.batch_slot_used[(size_t) T], e.used);
}
}  // namespace

bool Engine::batch_graph_ready(int T, const std::vector<int>& key) {
    CardBuf& b0 = *cb_[0];
    if (b0.batch_graphs[(size_t) T].exec && b0.batch_graphs[(size_t) T].key == key) {
        if (precap()) for (CardBuf* b : cb_) b->batch_slot_used[(size_t) T] = ++graph_tick_;
        return true;
    }
    if (!precap()) return false;
    std::vector<CardBuf::GraphEntry>& cache = b0.batch_cache[(size_t) T];
    for (size_t i = 0; i < cache.size(); ++i)
        if (cache[i].key == key) {
            for (CardBuf* b : cb_) {
                swap_slot(*b, T, b->batch_cache[(size_t) T][i]);
                b->batch_slot_used[(size_t) T] = ++graph_tick_;
            }
            return true;
        }
    return false;
}

bool Engine::lazy_capture(const std::vector<int>& key) {
#if FNX_HIP
    static const long long after = env_int("FNX_CAPTURE_AFTER", 2);
#else
    static const long long after = env_int("FNX_CAPTURE_AFTER", 0);
#endif
    if (after < 2 || warming_) return false;
    if (seen_.size() > 8192) seen_.clear();
    return ++seen_[key] < after;
}

void Engine::batch_graph_stash(int T) {
    if (!precap() || !cb_[0]->batch_graphs[(size_t) T].exec) return;
    static const size_t max_graphs = (size_t) std::max<long long>(1, env_int("FNX_PRECAP_GRAPHS", 64));
    size_t total = 0;
    for (const auto& v : cb_[0]->batch_cache) total += v.size();
    int vt = -1;
    size_t vi = 0;
    if (total >= max_graphs) {   // the least recently used entry goes; its tables are the next capture's
        unsigned long best = ~0UL;
        for (int t = 0; t <= MAXT; ++t)
            for (size_t i = 0; i < cb_[0]->batch_cache[(size_t) t].size(); ++i)
                if (cb_[0]->batch_cache[(size_t) t][i].used < best) best = cb_[0]->batch_cache[(size_t) t][i].used, vt = t, vi = i;
    }
    for (CardBuf* b : cb_) {
        ck(cudaSetDevice(b->dev), "set");
        CardBuf::GraphEntry nu;
        CardBuf::GraphSlot& s = b->batch_graphs[(size_t) T];
        nu.exec = s.exec;
        nu.key = std::move(s.key);
        nu.lane_rows = b->lane_rows[(size_t) T];
        nu.lane_gdn = b->lane_gdn[(size_t) T];
        nu.seg_ctl = b->seg_ctl[(size_t) T];
        nu.seg_row0 = b->seg_row0[(size_t) T];
        nu.used = b->batch_slot_used[(size_t) T];
        s.exec = nullptr;
        s.key.clear();
        b->lane_rows[(size_t) T] = nullptr, b->lane_gdn[(size_t) T] = nullptr, b->seg_ctl[(size_t) T] = nullptr, b->seg_row0[(size_t) T] = nullptr;
        if (vt >= 0) {
            CardBuf::GraphEntry& v = b->batch_cache[(size_t) vt][vi];
            if (v.exec) cudaGraphExecDestroy(v.exec);
            b->lane_rows[(size_t) T] = v.lane_rows, b->lane_gdn[(size_t) T] = v.lane_gdn;
            b->seg_ctl[(size_t) T] = v.seg_ctl, b->seg_row0[(size_t) T] = v.seg_row0;
            b->batch_cache[(size_t) vt].erase(b->batch_cache[(size_t) vt].begin() + (long) vi);
        }
        b->batch_cache[(size_t) T].push_back(std::move(nu));
    }
}

void Engine::warm_graphs() {
    const int was = cur_;
    warming_ = true;
    std::vector<int> lanes;
    for (int i = 0; i < (int) lanes_.size(); ++i) lanes.push_back(i);
    const int tok = 248045;   // any token: every lane is reset after
    use_lane(0);
    reset();
    // the prompt path's buffers (and cuBLAS's) with a read of two chunks, every lane's rotary table
    std::fprintf(stderr, "fnx: warm: prompt path (1100 tokens) ...\n");
    prefill(std::vector<int>(1100, tok), tok);
    std::fprintf(stderr, "fnx: warm: prompt path done\n");
    reset();
    for (CardBuf* b : cb_) {
        ck(cudaSetDevice(b->dev), "set");
        for (LaneDev& D : b->lanes)
            if (!D.mrope_buf) D.mrope_buf = dalloc<int>((size_t) max_ctx_ * 3);
    }
    std::fprintf(stderr, "fnx: warm: verify windows ...\n");
    for (int T = 1; T <= MAXT; ++T) {
        verify(std::vector<int>((size_t) T, tok));
        commit(1);
        if (T == 1 || T == MAXT) std::fprintf(stderr, "fnx: warm: verify %d rows done\n", T);
    }
    if (mtp_)
        for (int T = 1; T <= MAXT; ++T) draft(T, true, 0, std::vector<int>((size_t) T, tok), 0);
    for (int n = 1; n <= std::min((int) lanes.size(), MAXT); ++n)
        step(std::vector<int>(lanes.begin(), lanes.begin() + n), std::vector<int>((size_t) n, tok));
    // batched windows with drafts (they share step()'s graph slots, one per row count): 2 .. MAXT rows over as many lanes
    // as there are, up to the rows, dealt out evenly
    if (batch_drafts() != BatchDrafts::off && precap()) {
        // FNX_PRECAP: every uniform layout (k rows a lane) of the first n lanes, n = 1 .. lanes, kept as graphs: a window of the lanes
        // in order with the same drafts each (or none) then never captures. The mean capture is what the planner is charged
        const auto t0 = std::chrono::steady_clock::now();
        unsigned long n_cap = 0;
        double ms_cap = 0.0;
        std::fprintf(stderr, "fnx: warm: precapture of the uniform batched windows ...\n");
        for (int n = 1; n <= (int) lanes.size(); ++n)
            for (int k = 1; n * k <= MAXT; ++k) {
                if (n == 1 && k == 1) continue;   // the one-row window is step()'s, warmed above
                const std::vector<int> ids(lanes.begin(), lanes.begin() + n), counts((size_t) n, k);
                verify_rows(ids, counts, std::vector<int>((size_t) (n * k), tok));
                commit_rows(std::vector<int>((size_t) n, 1));
                if (last_capture_ms > 0.0) ++n_cap, ms_cap += last_capture_ms;
            }
        if (n_cap) warm_capture_ms = ms_cap / (double) n_cap;
        long total = 0;
        for (const auto& v : cb_[0]->batch_cache) total += (long) v.size();
        std::fprintf(stderr, "fnx: warm: precapture done: %lu captures, %.1f ms each, %ld graphs held, %.1f s\n", n_cap, warm_capture_ms, total,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    } else if (batch_drafts() != BatchDrafts::off)
        for (int T = 2; T <= MAXT; ++T) {
            const int n = std::min((int) lanes.size(), T);
            std::vector<int> counts((size_t) n, T / n);
            for (int i = 0; i < T % n; ++i) ++counts[(size_t) i];
            verify_rows(std::vector<int>(lanes.begin(), lanes.begin() + n), counts, std::vector<int>((size_t) T, tok));
            commit_rows(std::vector<int>((size_t) n, 1));
        }
    for (int i : lanes) {
        use_lane(i);
        reset();
    }
    use_lane(was);
    warming_ = false;
}

std::vector<long> Engine::free_vram_mib() const {
    std::vector<long> v;
    for (CardBuf* b : cb_) {
        ck(cudaSetDevice(b->dev), "set");
        size_t f = 0, t = 0;
        ck(cudaMemGetInfo(&f, &t), "cudaMemGetInfo");
        v.push_back((long) (f >> 20));
    }
    return v;
}

namespace {
std::mutex owned_mu;
std::vector<std::function<void()>>& owned_list() {
    static std::vector<std::function<void()>> list;
    return list;
}
}  // namespace

void own(std::function<void()> release) {
    std::lock_guard<std::mutex> l(owned_mu);
    owned_list().push_back(std::move(release));
}

std::vector<std::function<void()>> take_owned() {
    std::lock_guard<std::mutex> l(owned_mu);
    return std::move(owned_list());
}

// every card idle, the window graphs destroyed, then what the engine made released, newest first (errors ignored: a
// destructor must not throw, and the process may be on its way out)
Engine::~Engine() {
    for (CardBuf* b : cb_) {
        cudaSetDevice(b->dev);
        cudaDeviceSynchronize();
        for (auto* slots : {&b->win_graphs, &b->mtp_graphs, &b->batch_graphs})
            for (CardBuf::GraphSlot& s : *slots)
                if (s.exec) cudaGraphExecDestroy(s.exec);
        for (auto& v : b->batch_cache)
            for (CardBuf::GraphEntry& e : v)
                if (e.exec) cudaGraphExecDestroy(e.exec);
        for (CardBuf::GraphEntry& e : b->dl_cache)
            if (e.exec) cudaGraphExecDestroy(e.exec);
    }
    std::vector<std::function<void()>> o = take_owned();
    for (auto it = o.rbegin(); it != o.rend(); ++it) (*it)();
    cudaGetLastError();
    for (CardBuf* b : cb_) delete b;
}

void Engine::embed_token(int tok, int cell, float* out) const {
    if (tok >= IMAGE_TOK) {
        for (const Image& im : lanes_[(size_t) cur_].images)
            if (cell >= im.cell && cell < im.cell + im.n) {
                if (!im.rows) break;
                std::memcpy(out, im.rows + (size_t) (cell - im.cell) * N, N * sizeof(float));
                return;
            }
        throw std::runtime_error("embed: image cell " + std::to_string(cell) + " without its embedding row");
    }
    if (tok < 0 || tok >= m_.vocab) throw std::runtime_error("embed: token " + std::to_string(tok) + " outside the vocabulary");
    const Tensor& te = *m_.tok_embd;
    const uint8_t* row = te.data + (size_t) tok * te.row_bytes();
    for (int b = 0; b < N / 32; ++b) {
        uint16_t dh;
        std::memcpy(&dh, row + b * 34, 2);
        const float d = h2f(dh);
        for (int j = 0; j < 32; ++j) out[b * 32 + j] = d * (float) (int8_t) row[b * 34 + 2 + j];
    }
}

void Engine::dump_dev(const std::string& name, int layer, const float* p, int T, int n) {
    if (!dump) return;
    std::vector<float> h((size_t) T * n);
    ck(cudaSetDevice(cb_[0]->dev), "set");
    ck(cudaStreamSynchronize(cb_[0]->st), ("dump sync before " + name + " of layer " + std::to_string(layer)).c_str());
    ck(cudaMemcpy(h.data(), p, h.size() * 4, cudaMemcpyDeviceToHost), "dump copy");
    dump(name, layer, T, n, h);
}

std::vector<int> Engine::window(const std::vector<int>& tokens) {
    std::vector<int> out = verify(tokens);
    commit((int) tokens.size());
    return out;
}

size_t Engine::SeqState::total() const {
    size_t t = 0;
    for (size_t b : bytes) t += b;
    return t;
}

// FNX_CKPT_POOL (on in a FAST build): a checkpoint's host memory (56 MiB a card with a 262k context's key rings) comes from the checkpoints freed
// before it. Fresh memory costs its page faults at the first copy and its unmapping when the oldest checkpoint goes: 4 cards, ~45 of a checkpoint's
// ~60 ms (an agent turn takes one checkpoint and drops one). The pool keeps the memory of two checkpoints at most; it is not in the cache's budget
namespace {
struct CkptPool {
    std::mutex mu;
    std::array<std::pair<void*, size_t>, (size_t) 2 * NCARD> free_{};   // (memory, payload bytes). Fixed: a return (from ~SeqState) allocates nothing
    size_t n = 0;
};
CkptPool& ckpt_pool() {
    static CkptPool* const p = new CkptPool;   // never destroyed: a checkpoint that dies at the process's end still finds it
    return *p;
}
bool ckpt_pool_on() {
    static const bool on = env_int("FNX_CKPT_POOL", FAST) != 0;
    return on;
}
void* ckpt_take(size_t bytes) {
    if (!ckpt_pool_on()) return nullptr;
    CkptPool& P = ckpt_pool();
    std::lock_guard<std::mutex> l(P.mu);
    for (size_t i = 0; i < P.n; ++i)
        if (P.free_[i].second == bytes) {
            void* p = P.free_[i].first;
            P.free_[i] = P.free_[--P.n];
            return p;
        }
    return nullptr;
}
void ckpt_give(void* p, size_t bytes) {
    if (!p) return;
    if (ckpt_pool_on()) {
        CkptPool& P = ckpt_pool();
        std::lock_guard<std::mutex> l(P.mu);
        if (P.n < P.free_.size()) {
            P.free_[P.n++] = {p, bytes};
            return;
        }
    }
    std::free(p);
}
}  // namespace

Engine::SeqState::~SeqState() {
    for (size_t c = 0; c < host.size(); ++c) {
        if (!kv && c < bytes.size()) ckpt_give(host[c], bytes[c]);
        else std::free(host[c]);
    }
}

void stage_copy(CardBuf& b, const std::vector<std::pair<char*, size_t>>& pieces, char* host, size_t total, bool to_host) {
    if (!b.stage[0]) {
        for (int k = 0; k < 2; ++k) {
            const cudaError_t e = cudaHostAlloc((void**) &b.stage[k], STAGE_BYTES, cudaHostAllocPortable);
            if (e != cudaSuccess) {
                cudaGetLastError();
                b.stage[k] = nullptr;
                throw std::runtime_error("state staging: page-locked host memory for 32 MiB: " + std::string(cudaGetErrorString(e)));
            }
            ck(cudaEventCreateWithFlags(&b.stage_ev[k], cudaEventDisableTiming), "event");
            own_pinned(b.stage[k]);
            own_event(b.stage_ev[k]);
        }
    }
    const size_t nblk = (total + STAGE_BYTES - 1) / STAGE_BYTES;
    size_t pi = 0, poff = 0;   // the next piece byte to move
    // the DMAs of block k between the device pieces and half k % 2
    auto dma = [&](size_t k) {
        const size_t blk = std::min(STAGE_BYTES, total - k * STAGE_BYTES);
        char* half = b.stage[k % 2];
        for (size_t done = 0; done < blk;) {
            const size_t take = std::min(blk - done, pieces[pi].second - poff);
            char* dev = pieces[pi].first + poff;
            ck(cudaMemcpyAsync(to_host ? half + done : dev, to_host ? dev : half + done, take,
                               to_host ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, b.st), "state copy");
            done += take;
            poff += take;
            if (poff == pieces[pi].second) {
                ++pi;
                poff = 0;
            }
        }
        ck(cudaEventRecord(b.stage_ev[k % 2], b.st), "event");
    };
    auto host_part = [&](size_t k) {
        const size_t blk = std::min(STAGE_BYTES, total - k * STAGE_BYTES);
        if (to_host) std::memcpy(host + k * STAGE_BYTES, b.stage[k % 2], blk);
        else std::memcpy(b.stage[k % 2], host + k * STAGE_BYTES, blk);
    };
    for (size_t k = 0; k < nblk; ++k) {
        if (to_host) {
            dma(k);
            if (k > 0) {
                ck(cudaEventSynchronize(b.stage_ev[(k - 1) % 2]), "state copy");
                host_part(k - 1);
            }
        } else {
            if (k > 1) ck(cudaEventSynchronize(b.stage_ev[k % 2]), "state copy");   // the half's last DMA is done
            host_part(k);
            dma(k);
        }
    }
    if (to_host && nblk > 0) {
        ck(cudaEventSynchronize(b.stage_ev[(nblk - 1) % 2]), "state copy");
        host_part(nblk - 1);
    }
    ck(cudaStreamSynchronize(b.st), "state copy");
}

// f(card) on a thread per card, the first failure rethrown after all of them end
void each_card_thread(int n, const std::function<void(int)>& f) {
    std::vector<std::exception_ptr> err((size_t) n);
    std::vector<std::thread> th;
    for (int c = 0; c < n; ++c)
        th.emplace_back([&, c] {
            try {
                f(c);
            } catch (...) {
                err[(size_t) c] = std::current_exception();
            }
        });
    for (std::thread& t : th) t.join();
    for (const std::exception_ptr& e : err)
        if (e) std::rethrow_exception(e);
}

std::shared_ptr<Engine::SeqState> Engine::save_state(bool kv) {
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("save_state: a window is not committed");
    auto s = std::make_shared<SeqState>();
    s->tokens = history_;
    for (const Image& im : lanes_[(size_t) cur_].images)
        if (im.cell + im.n <= pos_) s->images.push_back(Image{im.cell, im.n, im.nx, im.ny, nullptr});
    if (kv) ensure_pages(pos_);
    s->kv = kv;
    s->host.assign(NCARD, nullptr);
    s->bytes.assign(NCARD, 0);
    const unsigned half = (tag_ + 1u) & 1u;
    std::vector<std::vector<std::pair<char*, size_t>>> pieces(NCARD);
    for (int c = 0; c < NCARD; ++c) {
        pieces[(size_t) c] = state_pieces(*cb_[c], m_.cards[c], mtp_ != nullptr, pos_, kv, half, lanes_[(size_t) cur_].pages);
        size_t n = 0;
        for (const auto& p : pieces[(size_t) c]) n += p.second;
        s->host[c] = kv ? nullptr : ckpt_take(n);
        if (!s->host[c]) s->host[c] = std::aligned_alloc(4096, (n + 4095) / 4096 * 4096);
        if (!s->host[c])
            throw std::runtime_error("save_state: host memory for " + std::to_string(n >> 20) + " MiB (" + std::to_string(host_available_mib()) +
                                     " MiB available)");
        s->bytes[c] = n;
    }
    static const bool timing = env_int("FNX_PF_TIMING", 0) != 0;
    const auto t0 = std::chrono::steady_clock::now();
    each_card_thread(NCARD, [&](int c) {
        ck(cudaSetDevice(cb_[c]->dev), "set");
        stage_copy(*cb_[c], pieces[(size_t) c], (char*) s->host[c], s->bytes[c], true);
    });
    if (timing)
        std::fprintf(stderr, "save_state(%s): card 0: %zu pieces, %zu KiB, all cards in %.1f ms\n", kv ? "kv" : "checkpoint", pieces[0].size(), s->bytes[0] >> 10,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return s;
}

void Engine::load_state(const SeqState& s) {
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("load_state: a window is not committed");
    if ((int) s.tokens.size() + 1 > max_ctx_) throw std::runtime_error("load_state: the state does not fit the context");
    if (!s.kv && (history_.size() < s.tokens.size() || !std::equal(s.tokens.begin(), s.tokens.end(), history_.begin())))
        throw std::runtime_error("load_state: a checkpoint of another sequence (the engine does not hold its K/V)");
    const int pos = (int) s.tokens.size();
    if (s.kv) ensure_pages(pos);
    const unsigned half = (tag_ + 1u) & 1u;
    std::vector<std::vector<std::pair<char*, size_t>>> pieces(NCARD);
    for (int c = 0; c < NCARD; ++c) {
        pieces[(size_t) c] = state_pieces(*cb_[c], m_.cards[c], mtp_ != nullptr, pos, s.kv, half, lanes_[(size_t) cur_].pages);
        size_t n = 0;
        for (const auto& p : pieces[(size_t) c]) n += p.second;
        if (n != s.bytes[c]) throw std::runtime_error("load_state: a state saved with another layout");
    }
    static const bool timing = env_int("FNX_PF_TIMING", 0) != 0;
    const auto t0 = std::chrono::steady_clock::now();
    each_card_thread(NCARD, [&](int c) {
        ck(cudaSetDevice(cb_[c]->dev), "set");
        stage_copy(*cb_[c], pieces[(size_t) c], (char*) s.host[c], s.bytes[c], false);
    });
    if (timing)
        std::fprintf(stderr, "load_state(%s): card 0: %zu pieces, %zu KiB, all cards in %.1f ms\n", s.kv ? "kv" : "checkpoint", pieces[0].size(), s.bytes[0] >> 10,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    history_ = s.tokens;
    pos_ = pos;
    if (df2_) df2_forget(cur_);
    set_images(s.images);
    // pages past the state's length go back to the pool
    LaneHost& L = lanes_[(size_t) cur_];
    const size_t keep = (size_t) (pos / KVP_POS + 1);
    if (L.pages.size() > keep) {
        free_pages_.insert(free_pages_.end(), L.pages.begin() + (long) keep, L.pages.end());
        L.pages.resize(keep);
        std::sort(free_pages_.begin(), free_pages_.end(), std::greater<int>());
    }
}

void Engine::reset() {
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("reset: a window is not committed");
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        ck(cudaSetDevice(b.dev), "set");
        for (int l = 0; l < NL; ++l) {
            if (is_attn(l)) continue;
            ck(cudaMemsetAsync(b.S[l], 0, (size_t) w.nv() * HS * HS * 4, b.st), "reset state");
            ck(cudaMemsetAsync(b.chist[l], 0, (size_t) 2 * 3 * w.conv_ch() * 4, b.st), "reset conv history");
        }
        ck(reset_hists(b), "reset histories");
        ck(cudaStreamSynchronize(b.st), "reset sync");
    }
    release_pages();
    history_.clear();
    pos_ = 0;
    if (df2_) df2_forget(cur_);
    set_images({});
}

void Engine::commit(int n) {
    const int T = (int) pending_.size();
    if (T == 0) throw std::runtime_error("commit: no window to commit");
    if (n < 1 || n > T) throw std::runtime_error("commit: keep 1..T rows");
    if (n < T) {
        // the window read history parity tag & 1 and wrote the other: that one becomes the kept count's history. The
        // kept row's states and histories in one copy_many per card (Flash-Next's as cudaMemcpyAsync calls: 73 a card,
        // ~0.5 ms of host time a commit)
        const unsigned wrote = (tag_ & 1u) ^ 1u;
        constexpr int CJ = 2 * NL + HIST_JOBS;
        if (!h_.commit_jobs) h_.commit_jobs = halloc<CopyJob>((size_t) NCARD * CJ);
        for (CardBuf* b : cb_)
            if (!b->commit_jobs) {
                ck(cudaSetDevice(b->dev), "set");
                b->commit_jobs = dalloc<CopyJob>(CJ);
                ck(cudaEventCreateWithFlags(&b->commit_ev, cudaEventDisableTiming), "commit event");
                own_event(b->commit_ev);
            }
        on_cards([&](int c) {
            CardBuf& b = *cb_[c];
            const CardW& w = m_.cards[c];
            const size_t sbytes = (size_t) w.nv() * HS * HS * 4, hfloats = (size_t) 3 * w.conv_ch();
            // the page-locked table again only once the last commit's upload from it is done
            ck(cudaEventSynchronize(b.commit_ev), "commit table");
            CopyJob* jobs = static_cast<CopyJob*>(h_.commit_jobs) + (size_t) c * CJ;
            int k = 0;
            for (int l = 0; l < NL; ++l) {
                if (is_attn(l)) continue;
                jobs[k++] = {b.S[l], b.snap[l] + (size_t) (n - 1) * w.nv() * HS * HS, sbytes};
                jobs[k++] = {b.chist[l] + wrote * hfloats, b.hsnap[l] + (size_t) (n - 1) * hfloats, hfloats * 4};
            }
            k += hist_jobs(jobs + k, b, nullptr, (size_t) (n - 1));
            for (int i = 0; i < k; ++i)
                if (((uintptr_t) jobs[i].dst | (uintptr_t) jobs[i].src | jobs[i].bytes) & 15u)
                    throw std::runtime_error("commit: a copy not 16-byte aligned");
            ck(cudaMemcpyAsync(b.commit_jobs, jobs, (size_t) k * sizeof(CopyJob), cudaMemcpyHostToDevice, b.st), "commit jobs");
            ck(cudaEventRecord(b.commit_ev, b.st), "commit event");
            copy_many(b.commit_jobs, k, b.st);
        });
    }
    history_.insert(history_.end(), pending_.begin(), pending_.begin() + n);
    pos_ += n;
    pending_.clear();
}

std::vector<float> Engine::last_logits(int T) const {
    const int vocab = m_.cards[NCARD - 1].v1;
    std::vector<float> out((size_t) T * vocab);
    for (int c = 0; c < NCARD; ++c) {
        const CardW& w = m_.cards[c];
        ck(cudaSetDevice(cb_[c]->dev), "set");
        const int nv = w.v1 - w.v0;
        for (int t = 0; t < T; ++t)
            ck(cudaMemcpy(out.data() + (size_t) t * vocab + w.v0, cb_[c]->head_logits + (size_t) t * nv, (size_t) nv * 4,
                          cudaMemcpyDeviceToHost), "logits");
    }
    return out;
}

std::vector<std::vector<std::pair<float, int>>> Engine::top_tokens(int T, const Penalties& pen) {
    if (T < 1 || T > MAXT) throw std::runtime_error("top_tokens: 1..MAXT rows");
    if (!pen.rows.empty() && (int) pen.rows.size() != T) throw std::runtime_error("top_tokens: one penalty list per row");
    if (!tk_.pen_tok) {
        tk_.pen_tok = halloc<int>((size_t) MAXT * PEN_CAP);
        tk_.pen_cnt = halloc<int>((size_t) MAXT * PEN_CAP);
        tk_.pen_n = halloc<int>(MAXT);
        tk_.v = halloc<float>((size_t) NCARD * MAXT * TOPK_N);
        tk_.i = halloc<int>((size_t) NCARD * MAXT * TOPK_N);
        tk_.err = halloc<unsigned>(NCARD);
        tk_.par = halloc<float>((size_t) MAXT * 3);
        for (CardBuf* b : cb_) {
            ck(cudaSetDevice(b->dev), "set");
            b->tk_pen_tok = dalloc<int>((size_t) MAXT * PEN_CAP);
            b->tk_pen_cnt = dalloc<int>((size_t) MAXT * PEN_CAP);
            b->tk_pen_n = dalloc<int>(MAXT);
            b->tk_v = dalloc<float>((size_t) MAXT * TOPK_N);
            b->tk_i = dalloc<int>((size_t) MAXT * TOPK_N);
            b->tk_err = dalloc<unsigned>(1);
            b->tk_par = dalloc<float>((size_t) MAXT * 3);
        }
    }
    // the rows' penalty lists at the stride of the longest (the copies carry only what is used)
    static const Penalties::Row none;
    int stride = 1;
    for (int t = 0; t < T; ++t) stride = std::max(stride, (int) (pen.rows.empty() ? none : pen.rows[(size_t) t]).counts.size());
    if (stride > PEN_CAP) throw std::runtime_error("top_tokens: more penalized tokens than " + std::to_string(PEN_CAP));
    for (int t = 0; t < T; ++t) {
        const Penalties::Row& row = pen.rows.empty() ? none : pen.rows[(size_t) t];
        const auto& r = row.counts;
        tk_.par[3 * t] = row.repeat;
        tk_.par[3 * t + 1] = row.freq;
        tk_.par[3 * t + 2] = row.present;
        tk_.pen_n[t] = (int) r.size();
        for (size_t k = 0; k < r.size(); ++k) {
            tk_.pen_tok[(size_t) t * stride + k] = r[k].first;
            tk_.pen_cnt[(size_t) t * stride + k] = r[k].second;
        }
    }
    on_cards([&](int c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        ck(cudaMemcpyAsync(b.tk_pen_tok, tk_.pen_tok, (size_t) T * stride * 4, cudaMemcpyHostToDevice, b.st), "penalties");
        ck(cudaMemcpyAsync(b.tk_pen_cnt, tk_.pen_cnt, (size_t) T * stride * 4, cudaMemcpyHostToDevice, b.st), "penalties");
        ck(cudaMemcpyAsync(b.tk_pen_n, tk_.pen_n, (size_t) T * 4, cudaMemcpyHostToDevice, b.st), "penalties");
        ck(cudaMemcpyAsync(b.tk_par, tk_.par, (size_t) T * 3 * 4, cudaMemcpyHostToDevice, b.st), "penalties");
        ck(cudaMemsetAsync(b.tk_err, 0, 4, b.st), "top err");
        fnx::top_tokens(b.head_logits, w.v1 - w.v0, T, w.v0, b.tk_pen_tok, b.tk_pen_cnt, b.tk_pen_n, stride, b.tk_par,
                        b.tk_v, b.tk_i, b.tk_err, b.st);
        ck(cudaMemcpyAsync(tk_.v + (size_t) c * MAXT * TOPK_N, b.tk_v, (size_t) T * TOPK_N * 4, cudaMemcpyDeviceToHost, b.st), "top v");
        ck(cudaMemcpyAsync(tk_.i + (size_t) c * MAXT * TOPK_N, b.tk_i, (size_t) T * TOPK_N * 4, cudaMemcpyDeviceToHost, b.st), "top i");
        ck(cudaMemcpyAsync(tk_.err + c, b.tk_err, 4, cudaMemcpyDeviceToHost, b.st), "top err");
        ck(cudaStreamSynchronize(b.st), "top sync");
    });
    for (int c = 0; c < NCARD; ++c)
        if (tk_.err[c] & 16u) throw std::runtime_error("top_tokens: the head's logits hold a NaN or inf");
    std::vector<std::vector<std::pair<float, int>>> out(T);
    for (int t = 0; t < T; ++t) {
        auto& r = out[t];
        r.reserve((size_t) NCARD * TOPK_N);
        for (int c = 0; c < NCARD; ++c)
            for (int k = 0; k < TOPK_N; ++k) {
                const size_t at = ((size_t) c * MAXT + t) * TOPK_N + k;
                r.emplace_back(tk_.v[at], tk_.i[at]);
            }
        std::sort(r.begin(), r.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first || (a.first == b.first && a.second < b.second);
        });
        r.resize(TOPK_N);
    }
    return out;
}

std::vector<int> Engine::verify(const std::vector<int>& tokens) {
    const int T = (int) tokens.size();
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("verify: the last window is not committed");
    if (T < 1 || T > MAXT) throw std::runtime_error("window: 1..MAXT tokens");
    if (pos_ + T > max_ctx_) throw std::runtime_error("window: context full");
    arena_churn();
    ensure_pages(pos_ + T);
    const auto t0 = std::chrono::steady_clock::now();
    if (!h_.R0) {
        h_.R0 = halloc<float>((size_t) MAXT * ROW_W);
        h_.ctl = halloc<int>(2);
    }
    if (!h_.bv) {   // shared with the draft layer, which can run first (over a prompt read by prefill)
        h_.bv = halloc<float>(NCARD * MAXT);
        h_.bi = halloc<int>(NCARD * MAXT);
        h_.err = halloc<unsigned>(NCARD);
    }
    const int h0 = (int) history_.size();
    embed_into(0, T, pos_, [&](int q) -> int { return q < h0 ? history_[(size_t) q] : tokens[(size_t) (q - h0)]; }, nullptr);
    const auto t1 = std::chrono::steady_clock::now();
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    h_.ctl[0] = pos_;
    h_.ctl[1] = (int) take_tags(1);
    float* bv = h_.bv;
    int* bi = h_.bi;
    auto upload = [&](int, CardBuf& b, const CardW&) {
        ck(cudaEventRecord(b.ev0, b.st), "event");
        rows_upload(b, T);
        ck(cudaMemcpyAsync(b.ctl, h_.ctl, 2 * sizeof(int), cudaMemcpyHostToDevice, b.st), "ctl");
    };
    auto download = [&](int c, CardBuf& b, const CardW&) {
        ck(cudaMemcpyAsync(&bv[c * T], b.best_v, T * 4, cudaMemcpyDeviceToHost, b.st), "best v");
        ck(cudaMemcpyAsync(&bi[c * T], b.best_i, T * 4, cudaMemcpyDeviceToHost, b.st), "best i");
        ck(cudaMemcpyAsync(&h_.err[c], err_word(b), 4, cudaMemcpyDeviceToHost, b.st), "window err");
        ck(cudaEventRecord(b.ev1, b.st), "event");
    };
    static const bool kprof = env_int("FNX_KPROF", 0) > 0;
    ++diag_win_;               // every main window counts; the last window's record is dropped before any path can stamp into it
    diag_win_run_ = nullptr;
    if (dump) {
        each(upload);
        dump_dev("R0", -1, cb_[0]->R, T, ROW_W);
        each([&](int, CardBuf& b, const CardW&) { mall_pf_window(b, false); });
        enqueue(T, true);
        each(download);
    } else if (kprof) {   // the window eagerly, the last card's launches timed (kprof.hpp)
        each(upload);
        each([&](int, CardBuf& b, const CardW&) { ck(cudaStreamSynchronize(b.st), "kprof sync"); });
        g_kprof_dev = cb_[NCARD - 1]->dev;
        kprof_window(T, true);
        each([&](int, CardBuf& b, const CardW&) { mall_pf_window(b, false); });
        enqueue(T, false);
        g_kprof_dev = -1;
        each(download);
        each([&](int, CardBuf& b, const CardW&) { ck(cudaStreamSynchronize(b.st), "kprof sync"); });
        kprof_window(T, false);
    } else {
        const std::vector<int> key = graph_key({cur_});
        if (cb_[0]->win_graphs[T].key != key || !cb_[0]->win_graphs[T].exec) {   // capturing queues no work
            const auto tc = std::chrono::steady_clock::now();
            each([&](int, CardBuf& b, const CardW&) {
                ck(cudaStreamBeginCapture(b.st, cudaStreamCaptureModeRelaxed), "begin capture");
            });
            enqueue(T, false);
            each([&](int, CardBuf& b, const CardW&) {
                cudaGraph_t g;
                ck(cudaStreamEndCapture(b.st, &g), "end capture");
                install_graph(b.win_graphs[T], g, key);
            });
            ++cap_stat[2].n;
            cap_stat[2].ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
        }
        const auto l0 = std::chrono::steady_clock::now();
        diag_win_pending_ = diag_win_;
        on_cards([&](int c) {
            CardBuf& b = *cb_[c];
            upload(c, b, m_.cards[c]);
            mall_pf_window(b, true, T);
            diag_graph_launch(c, b, b.win_graphs[T].exec, true);
            download(c, b, m_.cards[c]);
        });
        last_launch_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - l0).count();
    }
    unsigned err = 0;
    float gpu_ms = 0.f;
    each([&](int c, CardBuf& b, const CardW&) {
        diag_sync(c, b, "window sync");
        err |= h_.err[c];
        float ms = 0.f;
        ck(cudaEventElapsedTime(&ms, b.ev0, b.ev1), "event time");
        gpu_ms = std::max(gpu_ms, ms);
        last_card_ms[c] = ms;
    });
    if (const char* e = window_error(err, false)) {
        diag_report("window");
        throw std::runtime_error(std::string("window: ") + e + err_words());
    }
    std::vector<int> out(T);
    for (int t = 0; t < T; ++t) {
        int best = 0;
        for (int c = 1; c < NCARD; ++c)
            if (bv[c * T + t] > bv[best * T + t]) best = c;
        out[t] = bi[best * T + t];
    }
    pending_ = tokens;
    last_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    last_embed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    last_gpu_ms = gpu_ms;
    if (!dump && !kprof) cost_fit[0].add(T, gpu_ms);
    // FNX_DRAFT_TIMING (see draft_rows): the window's host queue time and each card's own time, one line per 300 windows
    if (static const bool wtim = env_int("FNX_DRAFT_TIMING", 0) > 0; wtim && !dump && !kprof) {
        static double a_queue = 0.0, a_card[NCARD] = {};
        static long n = 0;
        a_queue += last_launch_ms;
        for (int c = 0; c < NCARD; ++c) a_card[c] += last_card_ms[c];
        if (++n % 300 == 0) {
            std::string per;
            for (int c = 0; c < NCARD; ++c) {
                char b[32];
                std::snprintf(b, sizeof b, " %.3f", a_card[c] / (double) n);
                per += b;
            }
            std::fprintf(stderr, "fnx: window timing: %ld windows, host queue %.3f ms, a card's own ms:%s\n", n, a_queue / (double) n, per.c_str());
        }
    }
    // FNX_SOLO_STATS=1: the one-user window's GPU ms by row count (the judge of a kernel change), one line per 300 windows
    if (static const bool solo_stats = env_int("FNX_SOLO_STATS", 0) != 0; solo_stats && !dump) {
        static unsigned long n_t[MAXT + 1] = {}, n = 0;
        static double ms_t[MAXT + 1] = {};
        ++n_t[T], ms_t[T] += gpu_ms;
        if (++n % 300 == 0) {
            std::string row = "fnx: solo windows gpu ms by rows (rows:count:ms):";
            for (int k = 1; k <= MAXT; ++k)
                if (n_t[k]) {
                    char buf[48];
                    std::snprintf(buf, sizeof buf, " %d:%lu:%.2f", k, n_t[k], ms_t[k] / (double) n_t[k]);
                    row += buf;
                }
            double fa = 0, fb = 0;
            if (cost_fit[0].get(fa, fb)) {
                char buf[96];
                std::snprintf(buf, sizeof buf, "; fit %.2f + %.2f a row, draft call %.2f ms", fa, fb, draft_call_fit);
                row += buf;
            }
            std::fprintf(stderr, "%s\n", row.c_str());
            // FNX_SOLO_STATS=2: each line is the last 300 windows alone (the cards' clocks drift with their temperature under a power cap: a running
            // mean since the start mixes the cold minutes with the warm ones)
            if (static const bool fresh = env_int("FNX_SOLO_STATS", 0) > 1; fresh)
                for (int k = 0; k <= MAXT; ++k) n_t[k] = 0, ms_t[k] = 0.0;
        }
    }
    return out;
}

std::vector<int> Engine::step(const std::vector<int>& lanes, const std::vector<int>& tokens) {
    const int T = (int) lanes.size();
    if (T < 1 || T > MAXT || (int) tokens.size() != T) throw std::runtime_error("step: 1..MAXT lanes, a token each");
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("step: the current lane's window is not committed");
    for (int r = 0; r < T; ++r) {
        if (lanes[r] < 0 || lanes[r] >= (int) lanes_.size()) throw std::runtime_error("step: no lane " + std::to_string(lanes[r]));
        for (int q = 0; q < r; ++q)
            if (lanes[q] == lanes[r]) throw std::runtime_error("step: a lane twice in one window");
    }
    arena_churn();
    const auto t0 = std::chrono::steady_clock::now();
    // the current lane's scalars to its slot, so every lane reads from lanes_
    LaneHost& cur = lanes_[(size_t) cur_];
    cur.pos = pos_;
    cur.tag = tag_;
    cur.history = std::move(history_);
    if (!h_.R0) {
        h_.R0 = halloc<float>((size_t) MAXT * ROW_W);
        h_.ctl = halloc<int>(2);
    }
    if (!h_.bv) {
        h_.bv = halloc<float>(NCARD * MAXT);
        h_.bi = halloc<int>(NCARD * MAXT);
        h_.err = halloc<unsigned>(NCARD);
    }
    std::vector<int> lane_ctl((size_t) 2 * T);
    try {
        for (int r = 0; r < T; ++r) {
            LaneHost& L = lanes_[(size_t) lanes[r]];
            if (L.pos + 1 > max_ctx_) throw std::runtime_error("step: lane " + std::to_string(lanes[r]) + "'s context is full");
            ensure_pages_of(lanes[r], L.pos + 1);
            const std::vector<int>& h = L.history;
            const int tok = tokens[r], h0 = (int) h.size();
            embed_into(r, 1, L.pos, [&](int q) -> int { return q < h0 ? h[(size_t) q] : tok; }, "step");
            // the lane's recurrent history half flips, as a window of its own would
            L.tag = L.tag + 1u;
            lane_ctl[2 * r] = L.pos;
            lane_ctl[2 * r + 1] = (int) L.tag;
        }
    } catch (...) {
        history_ = std::move(cur.history);
        throw;
    }
    h_.ctl[0] = 0;
    h_.ctl[1] = (int) ring_tag();
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    each([&](int, CardBuf& b, const CardW&) {
        ck(cudaEventRecord(b.ev0, b.st), "event");
        rows_upload(b, T);
        ck(cudaMemcpyAsync(b.ctl, h_.ctl, 2 * sizeof(int), cudaMemcpyHostToDevice, b.st), "ctl");
        for (int r = 0; r < T; ++r)
            ck(cudaMemcpyAsync(b.lanes[(size_t) lanes[r]].ctl, &lane_ctl[(size_t) 2 * r], 2 * sizeof(int), cudaMemcpyHostToDevice, b.st),
               "lane ctl");
    });
    const std::vector<int> bkey = graph_key(lanes);
    if (!batch_graph_ready(T, bkey)) {
        batch_graph_stash(T);
        const auto tc = std::chrono::steady_clock::now();
        each([&](int, CardBuf& b, const CardW&) {
            if (!b.lane_rows[(size_t) T]) b.lane_rows[(size_t) T] = dalloc<const int*>((size_t) LANE_ROW_COLS * MAXT);
            std::vector<const int*> rows((size_t) LANE_ROW_COLS * MAXT, nullptr);
            for (int r = 0; r < T; ++r) {
                const LaneDev& LD = b.lanes[(size_t) lanes[r]];
                rows[(size_t) r] = LD.ctl;
                rows[(size_t) (MAXT + r)] = LD.pt;
                lane_row_extra(rows.data(), r, LD);
            }
            ck(cudaMemcpy(b.lane_rows[(size_t) T], rows.data(), rows.size() * sizeof(const int*), cudaMemcpyHostToDevice), "lane rows");
            if (!b.lane_gdn[(size_t) T]) b.lane_gdn[(size_t) T] = dalloc<float*>((size_t) 2 * NL * MAXT);
            std::vector<float*> gdn((size_t) 2 * NL * MAXT, nullptr);
            for (int l = 0; l < NL; ++l)
                for (int r = 0; r < T; ++r) {
                    gdn[(size_t) l * MAXT + r] = b.lanes[(size_t) lanes[r]].S[(size_t) l];
                    gdn[(size_t) (NL + l) * MAXT + r] = b.lanes[(size_t) lanes[r]].chist[(size_t) l];
                }
            ck(cudaMemcpy(b.lane_gdn[(size_t) T], gdn.data(), gdn.size() * sizeof(float*), cudaMemcpyHostToDevice), "lane recurrent rows");
            ck(cudaStreamBeginCapture(b.st, cudaStreamCaptureModeRelaxed), "begin capture");
        });
        enqueue_batch(lanes);
        each([&](int, CardBuf& b, const CardW&) {
            cudaGraph_t g;
            ck(cudaStreamEndCapture(b.st, &g), "end capture");
            install_graph(b.batch_graphs[T], g, bkey);
        });
        ++cap_stat[3].n;
        cap_stat[3].ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
    }
    // FNX_ROWS_THREADS (on in a FAST build): the cards' graphs launched from a thread each, as verify() does (a launch of ~600-1,000 nodes is 0.4-0.7 ms
    // of host time, and in card order the last card's window started that much after the first's and every exchange waited for it)
    static const bool rows_threads = fast_opt("FNX_ROWS_THREADS");
    ++diag_win_;
    diag_win_run_ = nullptr;
    const auto launch_tail = [&](int c, CardBuf& b, bool launch) {
        if (launch) {
            mall_pf_window(b, true, T);
            diag_graph_launch(c, b, b.batch_graphs[T].exec, false);
        }
        ck(cudaMemcpyAsync(&h_.bv[c * T], b.best_v, T * 4, cudaMemcpyDeviceToHost, b.st), "best v");
        ck(cudaMemcpyAsync(&h_.bi[c * T], b.best_i, T * 4, cudaMemcpyDeviceToHost, b.st), "best i");
        ck(cudaMemcpyAsync(&h_.err[c], err_word(b), 4, cudaMemcpyDeviceToHost, b.st), "window err");
        ck(cudaEventRecord(b.ev1, b.st), "event");
    };
    if (rows_threads) {
        diag_win_pending_ = diag_win_;
        on_cards([&](int c) { launch_tail(c, *cb_[c], true); });
    }
    else {
        each([&](int, CardBuf& b, const CardW&) {
            mall_pf_window(b, true, T);
            ck(cudaGraphLaunch(b.batch_graphs[T].exec, b.st), "graph launch");
        });
        each([&](int c, CardBuf& b, const CardW&) { launch_tail(c, b, false); });
    }
    unsigned err = 0;
    float gpu_ms = 0.f;
    each([&](int c, CardBuf& b, const CardW&) {
        diag_sync(c, b, "step sync");
        err |= h_.err[c];
        float ms = 0.f;
        ck(cudaEventElapsedTime(&ms, b.ev0, b.ev1), "event time");
        gpu_ms = std::max(gpu_ms, ms);
    });
    if (const char* e = window_error(err, false)) {
        diag_report("step");
        throw std::runtime_error(std::string("step: ") + e + err_words());
    }
    std::vector<int> out(T);
    for (int r = 0; r < T; ++r) {
        int best = 0;
        for (int c = 1; c < NCARD; ++c)
            if (h_.bv[c * T + r] > h_.bv[best * T + r]) best = c;
        out[r] = h_.bi[best * T + r];
        LaneHost& L = lanes_[(size_t) lanes[r]];
        L.history.push_back(tokens[r]);
        ++L.pos;
    }
    pos_ = cur.pos;
    tag_ = cur.tag;
    history_ = std::move(cur.history);
    cur.history.clear();
    last_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    last_gpu_ms = gpu_ms;
    return out;
}

double rope_scale() {
    const char* e = std::getenv("FNX_ROPE_SCALE");
    if (!e) return 0.0;
    // 1 is off, so a profile's setting can be overridden without unsetting it
    const double S = env_num("FNX_ROPE_SCALE", 0.0);
    if (S == 1.0) return 0.0;
    if (!(S > 1.0)) throw std::runtime_error(std::string("FNX_ROPE_SCALE: want 1 (off) or a scale above 1, got ") + e);
    return S;
}

bool ring_f32() {
    // fp16 pairs by default on every count: fp32 costs ~30% of one request's decode on two cards. fp32's KL is lower
    // (two cards' code prompt 5.73e-4 against fp16's 7.48e-4, a bar of 6.06e-4, so fp16 is over it there; three cards
    // 4.38e-4 against 5.75e-4, both within)
    const long long v = env_int("FNX_RING_F32", 0);
    if (v != 0 && v != 1) throw std::runtime_error("FNX_RING_F32: 0 or 1, got " + std::to_string(v));
    return v == 1;
}

BatchDrafts batch_drafts() {
    static const BatchDrafts mode = [] {
        const char* e = std::getenv("FNX_BATCH_DRAFTS");
        if (!e || !std::strcmp(e, "1")) return BatchDrafts::mtp;
        if (!std::strcmp(e, "0")) return BatchDrafts::off;
        if (!std::strcmp(e, "lookup")) return BatchDrafts::lookup;
        throw std::runtime_error(std::string("FNX_BATCH_DRAFTS: 0, lookup or 1, not '") + e + "'");
    }();
    return mode;
}

RowSegs row_segs(const std::vector<int>& counts) {
    RowSegs sg;
    const int nl = (int) counts.size();
    for (int i = 0; i < nl; ++i) {
        sg.r0.push_back(sg.T);
        for (int j = 0; j < counts[(size_t) i]; ++j) sg.row_lane.push_back(i);
        sg.T += counts[(size_t) i];
        sg.order.push_back(i);
    }
    std::stable_sort(sg.order.begin(), sg.order.end(), [&](int a, int b) { return counts[(size_t) a] < counts[(size_t) b]; });
    for (int s = 0; s < nl; ++s) {
        const int k = counts[(size_t) sg.order[(size_t) s]];
        if (sg.groups.empty() || sg.groups.back().count != k) sg.groups.push_back({k, s, 0});
        ++sg.groups.back().n;
    }
    return sg;
}

std::vector<int> Engine::verify_rows(const std::vector<int>& lanes, const std::vector<int>& counts, const std::vector<int>& tokens) {
    const int nl = (int) lanes.size();
    if (nl < 1 || (int) counts.size() != nl) throw std::runtime_error("verify_rows: one lane or more, a row count each");
    int T = 0;
    for (int k : counts) {
        if (k < 1) throw std::runtime_error("verify_rows: a lane with no rows");
        T += k;
    }
    if (T > MAXT || (int) tokens.size() != T) throw std::runtime_error("verify_rows: at most MAXT rows in all, a token each");
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("verify_rows: the last window is not committed");
    for (int i = 0; i < nl; ++i) {
        if (lanes[i] < 0 || lanes[i] >= (int) lanes_.size()) throw std::runtime_error("verify_rows: no lane " + std::to_string(lanes[i]));
        for (int q = 0; q < i; ++q)
            if (lanes[q] == lanes[i]) throw std::runtime_error("verify_rows: a lane twice in one window");
    }
    arena_churn();
    const auto t0 = std::chrono::steady_clock::now();
    // the current lane's scalars to its slot, so every lane reads from lanes_
    LaneHost& cur = lanes_[(size_t) cur_];
    cur.pos = pos_;
    cur.tag = tag_;
    cur.history = std::move(history_);
    if (!h_.R0) {
        h_.R0 = halloc<float>((size_t) MAXT * ROW_W);
        h_.ctl = halloc<int>(2);
    }
    if (!h_.bv) {
        h_.bv = halloc<float>(NCARD * MAXT);
        h_.bi = halloc<int>(NCARD * MAXT);
        h_.err = halloc<unsigned>(NCARD);
    }
    for (CardBuf* b : cb_)
        if (!b->row_pos) {
            ck(cudaSetDevice(b->dev), "set");
            b->row_pos = dalloc<int>(MAXT);
        }
    const RowSegs sg = row_segs(counts);
    std::vector<int> lane_ctl((size_t) 2 * nl), row_pos((size_t) T);
    std::vector<unsigned> tags((size_t) nl);
    try {
        for (int i = 0; i < nl; ++i) {
            const LaneHost& L = lanes_[(size_t) lanes[i]];
            const int k = counts[(size_t) i], r = sg.r0[(size_t) i];
            if (L.pos + k > max_ctx_) throw std::runtime_error("verify_rows: lane " + std::to_string(lanes[i]) + "'s context is full");
            ensure_pages_of(lanes[i], L.pos + k);
            const std::vector<int>& h = L.history;
            const int h0 = (int) h.size();
            embed_into(r, k, L.pos, [&](int q) -> int { return q < h0 ? h[(size_t) q] : tokens[(size_t) (r + q - h0)]; }, "verify_rows");
            // the lane's recurrent history half flips, as a window of its own would
            tags[(size_t) i] = L.tag + 1u;
            lane_ctl[(size_t) 2 * i] = L.pos;
            lane_ctl[(size_t) 2 * i + 1] = (int) tags[(size_t) i];
            for (int j = 0; j < k; ++j) row_pos[(size_t) (r + j)] = L.pos + j;
        }
    } catch (...) {
        history_ = std::move(cur.history);
        throw;
    }
    for (int i = 0; i < nl; ++i) lanes_[(size_t) lanes[i]].tag = tags[(size_t) i];
    h_.ctl[0] = 0;
    h_.ctl[1] = (int) ring_tag();
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    each([&](int, CardBuf& b, const CardW&) {
        ck(cudaEventRecord(b.ev0, b.st), "event");
        rows_upload(b, T);
        ck(cudaMemcpyAsync(b.ctl, h_.ctl, 2 * sizeof(int), cudaMemcpyHostToDevice, b.st), "ctl");
        for (int i = 0; i < nl; ++i)
            ck(cudaMemcpyAsync(b.lanes[(size_t) lanes[i]].ctl, &lane_ctl[(size_t) 2 * i], 2 * sizeof(int), cudaMemcpyHostToDevice, b.st),
               "lane ctl");
        ck(cudaMemcpyAsync(b.row_pos, row_pos.data(), (size_t) T * sizeof(int), cudaMemcpyHostToDevice, b.st), "row positions");
    });
    // the graph slot step() uses for T rows; the layout (each lane's count) is part of the key, and -1 tells it from step()'s
    std::vector<int> key = graph_key(lanes);
    key.push_back(-1);
    key.insert(key.end(), counts.begin(), counts.end());
    last_capture_ms = 0.0;
    last_kind_ = 0;
    bool eager = false;
    if (!batch_graph_ready(T, key)) {
        eager = lazy_capture(key);
        last_kind_ = eager ? 2 : 1;
        if (!eager) batch_graph_stash(T);
        const auto tc = std::chrono::steady_clock::now();
        auto swap_eager = [&] {   // an eager window reads its tables from the eager set, not the slot's (a held graph points at those)
            for (CardBuf* b : cb_) {
                std::swap(b->lane_rows[(size_t) T], b->eager_tab.lane_rows);
                std::swap(b->lane_gdn[(size_t) T], b->eager_tab.lane_gdn);
                std::swap(b->seg_ctl[(size_t) T], b->eager_tab.seg_ctl);
                std::swap(b->seg_row0[(size_t) T], b->eager_tab.seg_row0);
            }
        };
        if (eager) swap_eager();
        each([&](int, CardBuf& b, const CardW&) {
            if (!b.lane_rows[(size_t) T]) b.lane_rows[(size_t) T] = dalloc<const int*>((size_t) LANE_ROW_COLS * MAXT);
            std::vector<const int*> rows((size_t) LANE_ROW_COLS * MAXT, nullptr);
            for (int r = 0; r < T; ++r) {
                const LaneDev& LD = b.lanes[(size_t) lanes[(size_t) sg.row_lane[(size_t) r]]];
                rows[(size_t) r] = b.row_pos + r;
                rows[(size_t) (MAXT + r)] = LD.pt;
                lane_row_extra(rows.data(), r, LD);
            }
            ck(cudaMemcpy(b.lane_rows[(size_t) T], rows.data(), rows.size() * sizeof(const int*), cudaMemcpyHostToDevice), "lane rows");
            if (!b.lane_gdn[(size_t) T]) b.lane_gdn[(size_t) T] = dalloc<float*>((size_t) 2 * NL * MAXT);
            if (!b.seg_ctl[(size_t) T]) b.seg_ctl[(size_t) T] = dalloc<const int*>(MAXT);
            if (!b.seg_row0[(size_t) T]) b.seg_row0[(size_t) T] = dalloc<int>(MAXT);
            std::vector<float*> gdn((size_t) 2 * NL * MAXT, nullptr);
            std::vector<const int*> sctl((size_t) MAXT, nullptr);
            std::vector<int> srow0((size_t) MAXT, 0);
            for (int s = 0; s < nl; ++s) {
                const int i = sg.order[(size_t) s];
                const LaneDev& LD = b.lanes[(size_t) lanes[i]];
                for (int l = 0; l < NL; ++l) {
                    gdn[(size_t) l * MAXT + s] = LD.S[(size_t) l];
                    gdn[(size_t) (NL + l) * MAXT + s] = LD.chist[(size_t) l];
                }
                sctl[(size_t) s] = LD.ctl;
                srow0[(size_t) s] = sg.r0[(size_t) i];
            }
            ck(cudaMemcpy(b.lane_gdn[(size_t) T], gdn.data(), gdn.size() * sizeof(float*), cudaMemcpyHostToDevice), "lane recurrent rows");
            ck(cudaMemcpy(b.seg_ctl[(size_t) T], sctl.data(), sctl.size() * sizeof(const int*), cudaMemcpyHostToDevice), "segment ctl");
            ck(cudaMemcpy(b.seg_row0[(size_t) T], srow0.data(), srow0.size() * sizeof(int), cudaMemcpyHostToDevice), "segment rows");
            if (!eager) ck(cudaStreamBeginCapture(b.st, cudaStreamCaptureModeRelaxed), "begin capture");
        });
        if (eager) each([&](int, CardBuf& b, const CardW&) { mall_pf_window(b, false); });
        enqueue_rows(lanes, counts);
        if (eager) {
            swap_eager();
            last_capture_ms = 0.0;
        } else {
            each([&](int, CardBuf& b, const CardW&) {
                cudaGraph_t g;
                ck(cudaStreamEndCapture(b.st, &g), "end capture");
                install_graph(b.batch_graphs[T], g, key);
            });
            last_capture_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
            ++cap_stat[0].n;
            cap_stat[0].ms += last_capture_ms;
        }
        static const bool trace = env_int("FNX_PRECAP_TRACE", 0) != 0;
        if (trace) {
            std::string t;
            for (size_t i = 0; i < lanes.size(); ++i) t += " " + std::to_string(lanes[i]) + ":" + std::to_string(counts[i]);
            std::fprintf(stderr, "fnx: %s verify_rows T=%d lanes:rows%s (%.1f ms)\n", eager ? "eager" : "capture", T, t.c_str(),
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count());
        }
    }
    {   // FNX_BATCH_STATS=1: windows, graph captures, and the time between one window's start and the next's by the first one's row count; one block per 200 windows
        static const bool stats = env_flag("FNX_BATCH_STATS");
        if (stats) {
            static unsigned long nw = 0, ncap = 0, rows_sum = 0, cnt_t[MAXT + 1] = {};
            static double cap_ms = 0, ms_t[MAXT + 1] = {};
            static int prev_t = 0;
            static std::chrono::steady_clock::time_point t_first{}, t_prev{};
            const auto now = std::chrono::steady_clock::now();
            if (nw == 0) t_first = now;
            if (prev_t > 0) {
                const double gap = std::chrono::duration<double, std::milli>(now - t_prev).count();
                if (gap < 400.0) cnt_t[prev_t]++, ms_t[prev_t] += gap;   // a gap past 400 ms is an idle server, not a window
            }
            prev_t = T;
            t_prev = now;
            ++nw;
            rows_sum += (unsigned long) T;
            if (last_capture_ms > 0.0) ++ncap, cap_ms += last_capture_ms;
            if (nw % 200 == 0) {
                std::fprintf(stderr, "fnx: batch stats: %lu windows, %lu captured (%.1f ms each), %.1f rows a window, %.2f ms a window since the first\n", nw,
                             ncap, ncap ? cap_ms / (double) ncap : 0.0, (double) rows_sum / (double) nw,
                             std::chrono::duration<double, std::milli>(now - t_first).count() / (double) nw);
                std::fprintf(stderr, "fnx: captures since start (n, host ms): verify_rows %lu %.0f, draft %lu %.0f, window %lu %.0f, batch step %lu %.0f\n",
                             cap_stat[0].n, cap_stat[0].ms, cap_stat[1].n, cap_stat[1].ms, cap_stat[2].n, cap_stat[2].ms, cap_stat[3].n,
                             cap_stat[3].ms);
                if (dl_stat.graph + dl_stat.eager + dl_stat.captured)
                    std::fprintf(stderr, "fnx: draft_lanes calls since start: held graph %lu, eager %lu, captured %lu (%.1f ms each), graphs dropped %lu, held now %zu\n",
                                 dl_stat.graph, dl_stat.eager, dl_stat.captured, dl_stat.captured ? dl_stat.cap_ms / (double) dl_stat.captured : 0.0,
                                 dl_stat.evicted, cb_[0]->dl_cache.size());
                if (dl_stat.checked)
                    std::fprintf(stderr, "fnx: draft_lanes check: %lu calls compared with the launches a lane, %lu differ\n", dl_stat.checked,
                                 dl_stat.check_bad);
                std::fprintf(stderr, "fnx: verify_rows windows since start (n, mean host ms, mean gpu ms): held graph %lu %.1f %.1f, captured %lu %.1f %.1f, eager %lu %.1f %.1f\n",
                             win_stat[0].n, win_stat[0].ms / (double) std::max(1UL, win_stat[0].n), win_stat[0].gpu_ms / (double) std::max(1UL, win_stat[0].n),
                             win_stat[1].n, win_stat[1].ms / (double) std::max(1UL, win_stat[1].n), win_stat[1].gpu_ms / (double) std::max(1UL, win_stat[1].n),
                             win_stat[2].n, win_stat[2].ms / (double) std::max(1UL, win_stat[2].n), win_stat[2].gpu_ms / (double) std::max(1UL, win_stat[2].n));
                {
                    const char* nm[3] = {"held graph", "captured", "eager"};
                    for (int kd = 0; kd < 3; ++kd) {
                        std::string row = std::string("fnx: verify_rows host ms by rows, ") + nm[kd] + " (rows:count:ms):";
                        for (int k = 1; k <= MAXT; ++k)
                            if (win_stat_t[kd][k].n) {
                                char buf[48];
                                std::snprintf(buf, sizeof buf, " %d:%lu:%.1f", k, win_stat_t[kd][k].n, win_stat_t[kd][k].ms / (double) win_stat_t[kd][k].n);
                                row += buf;
                            }
                        std::fprintf(stderr, "%s\n", row.c_str());
                    }
                }
                std::string tab = "fnx: batch stats by rows (rows:count:ms):";
                for (int k = 1; k <= MAXT; ++k)
                    if (cnt_t[k]) {
                        char buf[48];
                        std::snprintf(buf, sizeof buf, " %d:%lu:%.1f", k, cnt_t[k], ms_t[k] / (double) cnt_t[k]);
                        tab += buf;
                    }
                std::fprintf(stderr, "%s\n", tab.c_str());
                if (double fa = 0, fb = 0; cost_fit[1].get(fa, fb))
                    std::fprintf(stderr, "fnx: batch windows (held graphs) fit: %.2f + %.2f ms a row (FNX_PLAN_BROUND_MS = %.2f, FNX_PLAN_BROW_MS = %.2f)\n", fa, fb, fa + fb, fb);
                nw = ncap = rows_sum = 0;
                cap_ms = 0;
                for (int k = 0; k <= MAXT; ++k) cnt_t[k] = 0, ms_t[k] = 0;
            }
        }
    }
    // FNX_ROWS_THREADS (on in a FAST build): as in step(), the held graphs launched from a thread a card (an eager window is already queued)
    static const bool rows_threads = fast_opt("FNX_ROWS_THREADS");
    ++diag_win_;
    diag_win_run_ = nullptr;
    const auto launch_tail = [&](int c, CardBuf& b, bool launch) {
        if (launch) {
            mall_pf_window(b, true, T);
            diag_graph_launch(c, b, b.batch_graphs[T].exec, false);
        }
        ck(cudaMemcpyAsync(&h_.bv[c * T], b.best_v, T * 4, cudaMemcpyDeviceToHost, b.st), "best v");
        ck(cudaMemcpyAsync(&h_.bi[c * T], b.best_i, T * 4, cudaMemcpyDeviceToHost, b.st), "best i");
        ck(cudaMemcpyAsync(&h_.err[c], err_word(b), 4, cudaMemcpyDeviceToHost, b.st), "window err");
        ck(cudaEventRecord(b.ev1, b.st), "event");
    };
    if (rows_threads && !eager) {
        diag_win_pending_ = diag_win_;
        on_cards([&](int c) { launch_tail(c, *cb_[c], true); });
    }
    else {
        if (!eager)
            each([&](int, CardBuf& b, const CardW&) {
                mall_pf_window(b, true, T);
                ck(cudaGraphLaunch(b.batch_graphs[T].exec, b.st), "graph launch");
            });
        each([&](int c, CardBuf& b, const CardW&) { launch_tail(c, b, false); });
    }
    unsigned err = 0;
    float gpu_ms = 0.f;
    each([&](int c, CardBuf& b, const CardW&) {
        diag_sync(c, b, "verify_rows sync");
        err |= h_.err[c];
        float ms = 0.f;
        ck(cudaEventElapsedTime(&ms, b.ev0, b.ev1), "event time");
        gpu_ms = std::max(gpu_ms, ms);
    });
    pos_ = cur.pos;
    tag_ = cur.tag;
    history_ = std::move(cur.history);
    cur.history.clear();
    // the window ran: its rows wait for commit_rows even when an error below ends the caller (the lanes' states moved)
    rows_pending_ = RowsWindow{lanes, counts, tokens};
    if (const char* e = window_error(err, false)) {
        diag_report("verify_rows");
        throw std::runtime_error(std::string("verify_rows: ") + e + err_words());
    }
    std::vector<int> out(T);
    for (int r = 0; r < T; ++r) {
        int best = 0;
        for (int c = 1; c < NCARD; ++c)
            if (h_.bv[c * T + r] > h_.bv[best * T + r]) best = c;
        out[r] = h_.bi[best * T + r];
    }
    last_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    last_gpu_ms = gpu_ms;
    if (last_kind_ == 0) cost_fit[1].add(T, gpu_ms);   // (a captured or eager window is not the held graph's cost)
    ++win_stat[last_kind_].n;
    win_stat[last_kind_].ms += last_ms;
    win_stat[last_kind_].gpu_ms += gpu_ms;
    ++win_stat_t[last_kind_][T].n;
    win_stat_t[last_kind_][T].ms += last_ms;
    return out;
}

std::vector<std::vector<int>> Engine::held_layouts(const std::vector<int>& lanes) const {
    std::vector<int> pre = graph_key(lanes);
    pre.push_back(-1);
    std::vector<std::vector<int>> out;
    for (int T = (int) lanes.size(); T <= MAXT; ++T) {
        const CardBuf::GraphSlot& s = cb_[0]->batch_graphs[(size_t) T];
        if (s.exec && s.key.size() == pre.size() + lanes.size() && std::equal(pre.begin(), pre.end(), s.key.begin()))
            out.emplace_back(s.key.begin() + (long) pre.size(), s.key.end());
        for (const CardBuf::GraphEntry& e : cb_[0]->batch_cache[(size_t) T])
            if (e.exec && e.key.size() == pre.size() + lanes.size() && std::equal(pre.begin(), pre.end(), e.key.begin()))
                out.emplace_back(e.key.begin() + (long) pre.size(), e.key.end());
    }
    return out;
}

void Engine::commit_rows(const std::vector<int>& keep) {
    const RowsWindow& w = rows_pending_;
    const int nl = (int) w.lanes.size();
    if (nl == 0) throw std::runtime_error("commit_rows: no window to commit");
    if ((int) keep.size() != nl) throw std::runtime_error("commit_rows: a kept count per lane");
    for (int i = 0; i < nl; ++i)
        if (keep[i] < 1 || keep[i] > w.counts[(size_t) i]) throw std::runtime_error("commit_rows: keep 1 .. its rows of each lane");
    const RowSegs sg = row_segs(w.counts);
    bool cut = false;
    for (int i = 0; i < nl; ++i) cut = cut || keep[i] < w.counts[(size_t) i];
    if (cut) {
        // a lane cut back takes its kept row's states and histories from the window's snapshots at its rows, as commit()
        // does (the history half its window wrote: the parity after its tag's)
        constexpr int CJ = 2 * NL + HIST_JOBS;
        if (!h_.rows_commit_jobs) h_.rows_commit_jobs = halloc<CopyJob>((size_t) NCARD * MAXT * CJ);
        for (CardBuf* b : cb_)
            if (!b->rows_commit_jobs) {
                ck(cudaSetDevice(b->dev), "set");
                b->rows_commit_jobs = dalloc<CopyJob>((size_t) MAXT * CJ);
                ck(cudaEventCreateWithFlags(&b->rows_commit_ev, cudaEventDisableTiming), "commit_rows event");
                own_event(b->rows_commit_ev);
            }
        on_cards([&](int c) {
            CardBuf& b = *cb_[c];
            const CardW& cw = m_.cards[c];
            const size_t sfloats = (size_t) cw.nv() * HS * HS, hfloats = (size_t) 3 * cw.conv_ch();
            // the page-locked table again only once the last commit_rows' upload from it is done
            ck(cudaEventSynchronize(b.rows_commit_ev), "commit_rows table");
            CopyJob* jobs = static_cast<CopyJob*>(h_.rows_commit_jobs) + (size_t) c * MAXT * CJ;
            int k = 0;
            for (int i = 0; i < nl; ++i) {
                if (keep[i] == w.counts[(size_t) i]) continue;
                const int lane = w.lanes[(size_t) i];
                const LaneDev& LD = b.lanes[(size_t) lane];
                const unsigned wrote = ((lane == cur_ ? tag_ : lanes_[(size_t) lane].tag) & 1u) ^ 1u;
                const size_t at = (size_t) (sg.r0[(size_t) i] + keep[i] - 1);
                for (int l = 0; l < NL; ++l) {
                    if (is_attn(l)) continue;
                    jobs[k++] = {LD.S[(size_t) l], b.snap[l] + at * sfloats, sfloats * 4};
                    jobs[k++] = {LD.chist[(size_t) l] + wrote * hfloats, b.hsnap[l] + at * hfloats, hfloats * 4};
                }
                k += hist_jobs(jobs + k, b, &LD, at);
            }
            for (int i = 0; i < k; ++i)
                if (((uintptr_t) jobs[i].dst | (uintptr_t) jobs[i].src | jobs[i].bytes) & 15u)
                    throw std::runtime_error("commit_rows: a copy not 16-byte aligned");
            ck(cudaMemcpyAsync(b.rows_commit_jobs, jobs, (size_t) k * sizeof(CopyJob), cudaMemcpyHostToDevice, b.st), "commit_rows jobs");
            ck(cudaEventRecord(b.rows_commit_ev, b.st), "commit_rows event");
            copy_many(b.rows_commit_jobs, k, b.st);
        });
    }
    for (int i = 0; i < nl; ++i) {
        const int lane = w.lanes[(size_t) i];
        const auto from = w.tokens.begin() + sg.r0[(size_t) i];
        std::vector<int>& h = lane == cur_ ? history_ : lanes_[(size_t) lane].history;
        h.insert(h.end(), from, from + keep[i]);
        (lane == cur_ ? pos_ : lanes_[(size_t) lane].pos) += keep[i];
    }
    rows_pending_ = RowsWindow{};
}

namespace {
bool draft_io_pack() {   // FNX_DRAFT_IO_PACK (draft_rows)
    static const bool on = env_int("FNX_DRAFT_IO_PACK", FAST) != 0;
    return on;
}
// FNX_DRAFT_EV_LAST (on in a FAST build; 0: the event before the downloads, as before): a draft call's second timing event is the last thing
// its card's worker queues, after the result downloads (the main windows queue theirs there already). With the downloads last, the calling
// thread's waits found cards 1 to 3 one after the other with work still to start: the call's four waits ended ~60 us later (4 cards, 210 W:
// a call 0.57 -> 0.50 ms). The cards' "own ms" of the statistics now covers the downloads too (~12 us)
bool draft_ev_last() {
    static const bool on = env_int("FNX_DRAFT_EV_LAST", FAST) != 0;
    return on;
}
}  // namespace

int Engine::draft(int T, bool from_main, int src_row, const std::vector<int>& tokens, int pos0, const Penalties::Row* pen,
                  bool cands) {
    if (!mtp_) throw std::runtime_error("draft: no draft layer attached");
    if (!h_.R0) throw std::runtime_error("draft: no window has run (it drafts from a window's rows)");
    if (src_row < 0 || src_row + T > MAXT) throw std::runtime_error("draft: source rows out of range");
    std::vector<const float*> src;
    for (CardBuf* b : cb_) src.push_back(draft_src(*b, from_main) + (size_t) src_row * ROW_W);
    return draft_rows(src, T, tokens, pos0, pen, cands);
}

int Engine::draft_rows(const std::vector<const float*>& src, int T, const std::vector<int>& tokens, int pos0,
                       const Penalties::Row* pen, bool cands) {
    if (!mtp_) throw std::runtime_error("draft: no draft layer attached");
    if (T < 1 || T > MAXT || (int) tokens.size() != T) throw std::runtime_error("draft: 1..MAXT rows, one token each");
    if (pos0 < 0 || pos0 + T > max_ctx_) throw std::runtime_error("draft: cells out of the context");
    ensure_pages(pos0 + T);
    if (!h_.mtp_emb) throw std::runtime_error("draft: mtp_attach first");
    const auto t0 = std::chrono::steady_clock::now();
    // the tokens' embeddings (Q8_0 rows of the main table; the token after row t sits at cell pos0 + t + 1)
    for (int t = 0; t < T; ++t) embed_token(tokens[t], pos0 + t + 1, h_.mtp_emb + (size_t) t * N);
    h_.mtp_ctl[0] = pos0;
    h_.mtp_ctl[1] = (int) ring_tag();
    // the penalized tokens in each card's slice of the draft vocabulary: {k, indexes, counts}
    int pen_k[NCARD] = {};
    if (pen) {
        if ((int) pen->counts.size() > PEN_CAP) throw std::runtime_error("draft: more penalized tokens than " + std::to_string(PEN_CAP));
        h_.mtp_pen_par[0] = pen->repeat, h_.mtp_pen_par[1] = pen->freq, h_.mtp_pen_par[2] = pen->present;
        for (int c = 0; c < NCARD; ++c) {
            const MtpCard& mc = mtp_->cards[c];
            int* hp = h_.mtp_pen + (size_t) c * (1 + 2 * PEN_CAP);
            int k = 0;
            for (const auto& [tok, cnt] : pen->counts) {
                const int di = tok >= 0 && tok < (int) draft_index_.size() ? draft_index_[(size_t) tok] : -1;
                if (di >= mc.dv0 && di < mc.dv1) hp[1 + k] = di - mc.dv0, hp[1 + PEN_CAP + k] = cnt, ++k;
            }
            std::memmove(hp + 1 + k, hp + 1 + PEN_CAP, (size_t) k * 4);   // counts right after the indexes
            hp[0] = pen_k[c] = k;
        }
    } else {
        for (int c = 0; c < NCARD; ++c) h_.mtp_pen[(size_t) c * (1 + 2 * PEN_CAP)] = 0;
    }
    // FNX_DRAFT_IO_PACK (on in a FAST build; 0: the copies one by one, as before): the call's words and embeddings go up in one copy and its
    // result words come down in one (draft_io, engine_impl.hpp). The penalty list goes up only when the card does not hold an empty one already.
    // The values the kernels read and the values the host reads are the same
    const bool io_pack = draft_io_pack();
    const int last = MTP_HEAD_EVERY_ROW ? T - 1 : 0;   // the last row's head row
    // what a card queues before the draft layer's launches (its inputs) and after them (the candidates, the results)
    auto draft_pre = [&](int c) {
        CardBuf& b = *cb_[c];
        ck(cudaMemcpyAsync(b.mtp_src, src[c], (size_t) T * ROW_W * 4, cudaMemcpyDeviceToDevice, b.st), "draft source rows");
        if (io_pack) {
            ck(cudaMemcpyAsync(b.mtp_io, h_.mtp_io, draft_io::EMB + (size_t) T * N * 4, cudaMemcpyHostToDevice, b.st), "draft inputs");
            if (pen_k[c] || b.mtp_pen_k != 0)
                ck(cudaMemcpyAsync(b.mtp_pen, h_.mtp_pen + (size_t) c * (1 + 2 * PEN_CAP), (size_t) (1 + 2 * pen_k[c]) * 4,
                                   cudaMemcpyHostToDevice, b.st), "draft penalties");
        } else {
            ck(cudaMemcpyAsync(b.mtp_emb, h_.mtp_emb, (size_t) T * N * 4, cudaMemcpyHostToDevice, b.st), "draft embeddings");
            ck(cudaMemcpyAsync(b.mtp_ctl, h_.mtp_ctl, 2 * sizeof(int), cudaMemcpyHostToDevice, b.st), "draft ctl");
            ck(cudaMemcpyAsync(b.mtp_pen, h_.mtp_pen + (size_t) c * (1 + 2 * PEN_CAP), (size_t) (1 + 2 * pen_k[c]) * 4,
                               cudaMemcpyHostToDevice, b.st), "draft penalties");
            if (pen_k[c]) ck(cudaMemcpyAsync(b.mtp_pen_par, h_.mtp_pen_par, 3 * 4, cudaMemcpyHostToDevice, b.st), "draft penalties");
        }
        b.mtp_pen_k = pen_k[c];
        ck(cudaEventRecord(b.ev0, b.st), "event");
    };
    auto draft_post = [&](int c) {
        CardBuf& b = *cb_[c];
        const MtpCard& mc = mtp_->cards[c];
        const int nv = mc.dv1 - mc.dv0;
        if (cands && nv > 0) {   // the last row's logits are penalized already
            if (!io_pack) ck(cudaMemsetAsync(b.mtp_tk_err, 0, 4, b.st), "draft top err");   // (packed: the upload cleared it)
            fnx::top_tokens(b.mtp_logits + (size_t) last * nv, nv, 1, mc.dv0, b.mtp_zero, b.mtp_zero, b.mtp_zero, 0, b.mtp_pen_par,
                       b.mtp_tk_v, b.mtp_tk_i, b.mtp_tk_err, b.st);
        }
        if (!draft_ev_last()) ck(cudaEventRecord(b.ev1, b.st), "event");
        if (io_pack) {
            ck(cudaMemcpyAsync(h_.mtp_res + (size_t) c * draft_io::RES_END, b.mtp_io, cands && nv > 0 ? draft_io::RES_END : draft_io::TK_V,
                               cudaMemcpyDeviceToHost, b.st), "draft results");
            ck(cudaMemcpyAsync(&h_.err[c], err_word(b), 4, cudaMemcpyDeviceToHost, b.st), "window err");
        } else {
            ck(cudaMemcpyAsync(&h_.bv[c], b.mtp_bv + last, 4, cudaMemcpyDeviceToHost, b.st), "draft v");
            ck(cudaMemcpyAsync(&h_.bi[c], b.mtp_bi + last, 4, cudaMemcpyDeviceToHost, b.st), "draft i");
            ck(cudaMemcpyAsync(&h_.err[c], err_word(b), 4, cudaMemcpyDeviceToHost, b.st), "window err");
            if (cands && nv > 0) {
                ck(cudaMemcpyAsync(h_.mtp_tk_v + (size_t) c * TOPK_N, b.mtp_tk_v, TOPK_N * 4, cudaMemcpyDeviceToHost, b.st), "draft top v");
                ck(cudaMemcpyAsync(h_.mtp_tk_i + (size_t) c * TOPK_N, b.mtp_tk_i, TOPK_N * 4, cudaMemcpyDeviceToHost, b.st), "draft top i");
                ck(cudaMemcpyAsync(h_.mtp_tk_err + c, b.mtp_tk_err, 4, cudaMemcpyDeviceToHost, b.st), "draft top err");
            }
        }
        if (draft_ev_last()) ck(cudaEventRecord(b.ev1, b.st), "event");   // the last thing the card's worker queues (draft_ev_last above)
    };
    // the packed results of card c into the host's words (after the card's stream has ended)
    auto draft_land = [&](int c) {
        if (!io_pack) return;
        const char* r = h_.mtp_res + (size_t) c * draft_io::RES_END;
        std::memcpy(&h_.bv[c], r + draft_io::BV + (size_t) last * 4, 4);
        std::memcpy(&h_.bi[c], r + draft_io::BI + (size_t) last * 4, 4);
        if (cands && mtp_->cards[c].dv1 > mtp_->cards[c].dv0) {
            std::memcpy(h_.mtp_tk_v + (size_t) c * TOPK_N, r + draft_io::TK_V, TOPK_N * 4);
            std::memcpy(h_.mtp_tk_i + (size_t) c * TOPK_N, r + draft_io::TK_I, TOPK_N * 4);
            std::memcpy(h_.mtp_tk_err + c, r + draft_io::TK_ERR, 4);
        }
    };
    static const bool kprof = env_int("FNX_KPROF", 0) > 0;
    const std::vector<int> key = graph_key({cur_});
    if (kprof) {   // the step eagerly, the last card's launches timed (kprof.hpp), as the window's
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            draft_pre(c);
        }
        for (CardBuf* b : cb_) ck(cudaStreamSynchronize(b->st), "kprof sync");
        g_kprof_dev = cb_[NCARD - 1]->dev;
        kprof_window(T, true, 1);
        mtp_enqueue(T);
        g_kprof_dev = -1;
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            draft_post(c);
        }
        for (CardBuf* b : cb_) ck(cudaStreamSynchronize(b->st), "kprof sync");
        kprof_window(T, false, 1);
    } else if (cb_[0]->mtp_graphs[T].key != key || !cb_[0]->mtp_graphs[T].exec) {   // capturing queues no work
        const auto tc = std::chrono::steady_clock::now();
        for (CardBuf* b : cb_) {
            ck(cudaSetDevice(b->dev), "set");
            ck(cudaStreamBeginCapture(b->st, cudaStreamCaptureModeRelaxed), "begin capture");
        }
        mtp_enqueue(T);
        for (CardBuf* b : cb_) {
            ck(cudaSetDevice(b->dev), "set");
            cudaGraph_t g;
            ck(cudaStreamEndCapture(b->st, &g), "end capture");
            install_graph(b->mtp_graphs[T], g, key);
        }
        ++cap_stat[1].n;
        cap_stat[1].ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
    }
    // FNX_DRAFT_TIMING (a measuring aid, off by default): where a step's time goes. 1: the host's time to queue the step (uploads, graph
    // launch, downloads; a thread a card) and to wait for it, and each card's own time from its uploads' end to its head's end. 2: the same with the
    // step's launches queued one by one from this thread in place of the graph launch (the same kernels in the same order: the same answers)
    static const int dtim = (int) env_int("FNX_DRAFT_TIMING", 0);
    const auto th0 = std::chrono::steady_clock::now();
    if (!kprof && dtim >= 2) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            draft_pre(c);
        }
        mtp_enqueue(T);
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            draft_post(c);
        }
    } else if (!kprof)
        on_cards([&](int c) {
            draft_pre(c);
            ck(cudaGraphLaunch(cb_[c]->mtp_graphs[T].exec, cb_[c]->st), "graph launch");
            draft_post(c);
        });
    const auto th1 = std::chrono::steady_clock::now();
    unsigned err = 0;
    float gpu_ms = 0.f, card_ms[NCARD], end_ms[NCARD] = {};
    for (int c = 0; c < NCARD; ++c) {
        ck(cudaSetDevice(cb_[c]->dev), "set");
        ck(cudaStreamSynchronize(cb_[c]->st), "draft sync");
        if (dtim > 0) end_ms[c] = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - th1).count();
        draft_land(c);
        err |= h_.err[c];
        float ms = 0.f;
        ck(cudaEventElapsedTime(&ms, cb_[c]->ev0, cb_[c]->ev1), "event time");
        gpu_ms = std::max(gpu_ms, ms);
        card_ms[c] = ms;
    }
    last_draft_gpu_ms = gpu_ms;
    if (dtim > 0 && !kprof) {
        static double a_queue = 0.0, a_wait = 0.0, a_card[NCARD] = {}, a_end[NCARD] = {};
        static long n = 0;
        a_queue += std::chrono::duration<double, std::milli>(th1 - th0).count();
        a_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - th1).count();
        for (int c = 0; c < NCARD; ++c) a_card[c] += card_ms[c], a_end[c] += end_ms[c];
        if (++n % 500 == 0) {
            std::string per;
            for (int c = 0; c < NCARD; ++c) {
                char b[32];
                std::snprintf(b, sizeof b, " %.3f", a_card[c] / (double) n);
                per += b;
            }
            std::string ends;   // (the waits run one after the other: where each one ended, from the first one's start)
            for (int c = 0; c < NCARD; ++c) {
                char b[32];
                std::snprintf(b, sizeof b, " %.3f", a_end[c] / (double) n);
                ends += b;
            }
            std::fprintf(stderr, "fnx: draft timing (%s): %ld steps, host queue %.3f ms, host wait %.3f ms, a card's own ms:%s, the cards' waits end at ms:%s\n",
                         dtim >= 2 ? "launches one by one" : "graph", n, a_queue / (double) n, a_wait / (double) n, per.c_str(), ends.c_str());
        }
    }
    if (const char* e = window_error(err, true)) {
        diag_report("draft");
        throw std::runtime_error(std::string("draft: ") + e + err_words());
    }
    int best = 0;
    for (int c = 1; c < NCARD; ++c)
        if (mtp_->cards[c].dv1 > mtp_->cards[c].dv0 && h_.bv[c] > h_.bv[best]) best = c;
    if (cands) {
        last_draft_cands.clear();
        for (int c = 0; c < NCARD; ++c) {
            if (mtp_->cards[c].dv1 <= mtp_->cards[c].dv0) continue;
            if (h_.mtp_tk_err[c] & 16u) throw std::runtime_error("draft: the draft layer's logits hold a NaN or inf");
            for (int k = 0; k < TOPK_N; ++k)
                last_draft_cands.emplace_back(h_.mtp_tk_v[(size_t) c * TOPK_N + k], mtp_->vocab[h_.mtp_tk_i[(size_t) c * TOPK_N + k]]);
        }
        std::sort(last_draft_cands.begin(), last_draft_cands.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first || (a.first == b.first && a.second < b.second);
        });
        last_draft_cands.resize(TOPK_N);
    }
    last_draft_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (T == 1) draft_call_fit = draft_calls_fit++ ? draft_call_fit * 0.99 + last_draft_ms * 0.01 : last_draft_ms;
    return mtp_->vocab[h_.bi[best]];
}


std::vector<int> Engine::draft_lanes(const std::vector<DraftReq>& reqs) {
    if (!mtp_) throw std::runtime_error("draft_lanes: no draft layer attached");
    if (!h_.R0 || !h_.mtp_emb) throw std::runtime_error("draft_lanes: no window has run (it drafts from a window's rows)");
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("draft_lanes: a window is not committed");
    const int nl = (int) reqs.size();
    int T = 0;
    for (const DraftReq& q : reqs) {
        if (q.T < 1 || (int) q.tokens.size() != q.T) throw std::runtime_error("draft_lanes: a lane's rows and tokens differ");
        if (q.lane < 0 || q.lane >= (int) lanes_.size()) throw std::runtime_error("draft_lanes: no lane " + std::to_string(q.lane));
        if (q.src_row < 0 || q.src_row + q.T > MAXT) throw std::runtime_error("draft_lanes: source rows out of range");
        if (q.pos0 < 0 || q.pos0 + q.T > max_ctx_ || q.pos0 + q.T > lane_pages(q.lane) * KVP_POS)
            throw std::runtime_error("draft_lanes: cells out of the lane's pages");
        T += q.T;
    }
    if (nl < 1 || T > MAXT) throw std::runtime_error("draft_lanes: 1.." + std::to_string(MAXT) + " rows in all");
    const auto t0 = std::chrono::steady_clock::now();
    const int was = cur_;
    const bool dl_tab = draft_lanes_tab();   // (see below)
    if (dl_tab)   // the table form's launches run all the lanes' rows at once: a lane twice would write its cells in no order
        for (int i = 0; i < nl; ++i)
            for (int j = 0; j < i; ++j)
                if (reqs[(size_t) i].lane == reqs[(size_t) j].lane) throw std::runtime_error("draft_lanes: a lane twice");
    bool any_pen = false, any_cands = false;
    for (const DraftReq& q : reqs) any_pen = any_pen || q.pen, any_cands = any_cands || q.cands;
    std::vector<int> off((size_t) nl);
    int o = 0;
    for (int i = 0; i < nl; ++i) {
        const DraftReq& q = reqs[(size_t) i];
        off[(size_t) i] = o;
        use_lane(q.lane);   // the embeddings read the lane's images
        for (int t = 0; t < q.T; ++t) {
            embed_token(q.tokens[(size_t) t], q.pos0 + t + 1, h_.mtp_emb + (size_t) (o + t) * N);
            h_.mtp_row_pos[o + t] = q.pos0 + t;
        }
        h_.mtp_lctl[2 * i] = q.pos0;
        // (table form: the lane's last row and whether it penalizes, for penalize_lanes / gather_lanes; the word is not read otherwise)
        h_.mtp_lctl[2 * i + 1] = dl_tab ? (o + q.T - 1) | (q.pen ? 0x10000 : 0) : 0;
        o += q.T;
    }
    use_lane(was);
    h_.mtp_ctl[0] = 0;
    h_.mtp_ctl[1] = (int) ring_tag();
    // the penalized tokens of the lanes that penalize their drafts, in each card's slice of the draft vocabulary: {k, indexes, counts}, as draft() lays them out
    std::vector<std::array<int, NCARD>> pen_k((size_t) nl);
    for (int i = 0; i < nl; ++i) {
        pen_k[(size_t) i].fill(0);
        const Penalties::Row* pen = reqs[(size_t) i].pen;
        if (!pen) continue;
        if ((int) pen->counts.size() > PEN_CAP) throw std::runtime_error("draft_lanes: more penalized tokens than " + std::to_string(PEN_CAP));
        h_.mtp_pen_par_l[3 * i] = pen->repeat, h_.mtp_pen_par_l[3 * i + 1] = pen->freq, h_.mtp_pen_par_l[3 * i + 2] = pen->present;
        for (int c = 0; c < NCARD; ++c) {
            const MtpCard& mc = mtp_->cards[c];
            int* hp = h_.mtp_pen_l + ((size_t) i * NCARD + c) * (1 + 2 * PEN_CAP);
            int k = 0;
            for (const auto& [tok, cnt] : pen->counts) {
                const int di = tok >= 0 && tok < (int) draft_index_.size() ? draft_index_[(size_t) tok] : -1;
                if (di >= mc.dv0 && di < mc.dv1) hp[1 + k] = di - mc.dv0, hp[1 + PEN_CAP + k] = cnt, ++k;
            }
            std::memmove(hp + 1 + k, hp + 1 + PEN_CAP, (size_t) k * 4);   // counts right after the indexes
            hp[0] = pen_k[(size_t) i][c] = k;
        }
    }
    last_lane_cands.assign((size_t) nl, {});
    for (int i = 0; i < nl; ++i)   // (once here, not in every card's uploads: those may run on the cards' threads)
        if (!reqs[(size_t) i].pen && reqs[(size_t) i].cands)
            h_.mtp_pen_par_l[3 * i] = 1.f, h_.mtp_pen_par_l[3 * i + 1] = 0.f, h_.mtp_pen_par_l[3 * i + 2] = 0.f;
    // the call's launches as a graph a card (FNX_DRAFT_LANES_GRAPH), as draft_rows' and the windows', and launches that take the
    // lanes from the card's tables (FNX_DRAFT_LANES_TAB, draft_lanes_tab()). Both on for 4 AMD cards in a FAST build; both 0: the call as it was.
    // Eagerly the call was 36 (2 lanes) to 54 (8 lanes) launches a card at ~45 us each on the cards (4 x R9700, Q4, FNX_BATCH_STATS: 2.1 / 2.5 /
    // 3.3 ms a call at 2 / 4 / 8 users, of windows of 11.9 / 16.4 / 26.8 ms; the one-lane call's graph is 0.59 ms).
    // A graph is valid for the calls whose launches have its arguments. With the launches a lane (TAB=0) that is one LAYOUT: the lanes in the call's
    // order, each one's rows, whether it penalizes, whether it takes candidates (row offsets and counts, the lanes' page tables / rings / rope
    // tables are launch arguments). The layouts are too many: a first call has each lane's kept tokens (2^8 layouts at 8 users with one draft
    // each), a chain step a subset of the lanes; measured, 8 users met a held graph in half of the calls and a capture costs 3.8 ms.
    // So the table form (TAB=1) takes the layout out of the launches:
    //   - the indexer rows, the norms + rotary and the K/V append run once over all rows with the lanes' pointers from the row table
    //     (qsa_index_lanes / attn_prep_lanes / kv_append_lanes, the kernels of a batched window's attention layers; the table's column 3 holds
    //     the lane's draft-layer ring). The kernels are the same as a launch a lane; a row reads its position from its own word, not pos0 + t.
    //   - the lanes' source rows come by one copy_many launch (the jobs are uploaded), not by a copy a lane before the call;
    //   - the penalties go by one launch that reads each lane's last row from the lane's second control word (penalize_lanes: k_penalize_row's
    //     arithmetic); the candidates come from the last rows gathered by lane (gather_lanes: a copy) by one launch with a block a lane.
    //   What is left in the key: the number of lanes, the rows, whether any lane penalizes, whether any takes candidates, cache_live().
    //   Every value a kernel computes comes from the same inputs by the same arithmetic as before (rows are independent of each other; copies
    //   are copies), so the texts are the same as the eager call's. Not chosen: a first call padded to the window's layout (rows of rejected
    //   tokens would write draft-layer cells and index keys: other cells than today's).
    // The cache: FNX_DRAFT_LANES_GRAPHS graphs (default 384), the least recently used one out; a key is captured at its FNX_DRAFT_LANES_AFTER'th
    // sight (default 2; eagerly before, as lazy_capture does for the windows). The uploads and downloads stay outside the graph.
    static const bool dl_graph = env_int("FNX_DRAFT_LANES_GRAPH", FAST) != 0;
    std::vector<int> dl_key;
    if (dl_graph) {
        std::vector<int> dl_lanes;
        for (const DraftReq& q : reqs) dl_lanes.push_back(q.lane);
        dl_key = graph_key(dl_lanes);
        dl_key.push_back(-3);   // (not a window's key: verify_rows' has -1 here)
        for (const DraftReq& q : reqs) dl_key.push_back(q.T), dl_key.push_back(q.pen != nullptr), dl_key.push_back(q.cands);
    }
    // dl_key: the layout (the row table's content follows from it, and with the launches a lane the graph). dl_gkey: the graph's key
    std::vector<int> dl_gkey;
    if (dl_graph && dl_tab) {
        dl_gkey = {(int) cache_live(), -4, nl, T, (int) any_pen, (int) any_cands};
    } else {
        dl_gkey = dl_key;
    }
    // the cards' row table (each row's position word and its lane's tables) is the layout's: written when the layout is another one than the last
    // call's. (A blocking copy before this call's launches; what may still be queued on the streams, commit_rows' work, does not read the table.)
    const bool new_tab = !dl_graph || dl_tab_key_ != dl_key;   // (with the switch off: every call, as before)
    if (new_tab) dl_tab_key_.clear();   // (set again once every card has the table)
    int dl_at = -1;                // the layout's graph in the cards' caches
    if (dl_graph) {
        static const size_t dl_max = (size_t) std::max<long long>(1, env_int("FNX_DRAFT_LANES_GRAPHS", 384));
        static const long long dl_after = env_int("FNX_DRAFT_LANES_AFTER", 2);
        std::vector<CardBuf::GraphEntry>& c0 = cb_[0]->dl_cache;
        for (size_t i = 0; i < c0.size() && dl_at < 0; ++i)
            if (c0[i].key == dl_gkey) dl_at = (int) i;
        bool cap = false;
        if (dl_at < 0 && !dl_capture_off_) {
            if (warming_ || dl_after <= 1) {
                cap = true;
            } else {
                if (dl_seen_.size() >= 8192 && !dl_seen_.count(dl_gkey)) dl_seen_.clear();   // (bounded: the counts start again)
                cap = ++dl_seen_[dl_gkey] >= dl_after;
            }
        }
        if (dl_at >= 0) {
            ++dl_stat.graph;
        } else if (cap) {
            if (c0.size() >= dl_max) {   // the least recently used graph goes: the same index on every card, whatever a destroy returns
                size_t v = 0;
                for (size_t i = 1; i < c0.size(); ++i)
                    if (c0[i].used < c0[v].used) v = i;
                bool bad = false;
                for (CardBuf* b : cb_) {
                    bad |= cudaSetDevice(b->dev) != cudaSuccess;
                    if (b->dl_cache[v].exec) bad |= cudaGraphExecDestroy(b->dl_cache[v].exec) != cudaSuccess;
                    b->dl_cache.erase(b->dl_cache.begin() + (long) v);
                }
                if (bad) {
                    cudaGetLastError();
                    dl_capture_off_ = true;
                    std::fprintf(stderr, "fnx: draft_lanes: a graph could not be destroyed: no more captures\n");
                }
                ++dl_stat.evicted;
            }
        }
        if (dl_at < 0 && cap && !dl_capture_off_) {
            const auto tc = std::chrono::steady_clock::now();
            // capturing runs nothing. Whatever throws in here: no stream stays in capture mode, no graph of a part of the cards is kept, no more
            // captures, and the exception goes on as it did from the eager call's launches
            std::vector<cudaGraphExec_t> ex((size_t) NCARD, nullptr);
            int began = 0;   // the streams of cards [0, began) capture
            bool made = true;
            try {
                for (; began < NCARD; ++began) {
                    CardBuf& b = *cb_[began];
                    ck(cudaSetDevice(b.dev), "set");
                    ck(cudaStreamBeginCapture(b.st, cudaStreamCaptureModeRelaxed), "begin capture");
                }
                mtp_enqueue_lanes(reqs, T);
                while (began > 0) {
                    const int c = began - 1;
                    CardBuf& b = *cb_[c];
                    ck(cudaSetDevice(b.dev), "set");
                    cudaGraph_t g = nullptr;
                    const cudaError_t ended = cudaStreamEndCapture(b.st, &g);
                    --began;   // (the stream's capture is over, whatever the call returned)
                    ck(ended, "end capture");
                    if (made && cudaGraphInstantiate(&ex[(size_t) c], g, 0) != cudaSuccess) {
                        cudaGetLastError();
                        ex[(size_t) c] = nullptr;
                        made = false;
                    }
                    ck(cudaGraphDestroy(g), "graph destroy");
                }
            } catch (...) {
                for (int c = 0; c < began; ++c) {
                    cudaGraph_t g = nullptr;
                    if (cudaSetDevice(cb_[c]->dev) == cudaSuccess && cudaStreamEndCapture(cb_[c]->st, &g) == cudaSuccess && g) cudaGraphDestroy(g);
                }
                for (int c = 0; c < NCARD; ++c)
                    if (ex[(size_t) c] && cudaSetDevice(cb_[c]->dev) == cudaSuccess) cudaGraphExecDestroy(ex[(size_t) c]);
                dl_capture_off_ = true;
                throw;
            }
            if (made) {
                const size_t n0 = c0.size();
                try {
                    for (int c = 0; c < NCARD; ++c) {
                        CardBuf::GraphEntry e;
                        e.exec = ex[(size_t) c], e.key = dl_gkey;
                        cb_[c]->dl_cache.push_back(std::move(e));
                    }
                } catch (...) {   // (no memory for an entry: every card's cache as it was, the graphs gone)
                    for (int c = 0; c < NCARD; ++c) {
                        if (cb_[c]->dl_cache.size() > n0) cb_[c]->dl_cache.pop_back();
                        if (cudaSetDevice(cb_[c]->dev) == cudaSuccess) cudaGraphExecDestroy(ex[(size_t) c]);
                    }
                    dl_capture_off_ = true;
                    throw;
                }
                dl_at = (int) c0.size() - 1;
                dl_seen_.erase(dl_gkey);
                ++dl_stat.captured;
                dl_stat.cap_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
            } else {   // no room for a graph on some card: this call and the layouts without a graph run eagerly from here on
                for (int c = 0; c < NCARD; ++c)
                    if (ex[(size_t) c]) {
                        cudaSetDevice(cb_[c]->dev);
                        cudaGraphExecDestroy(ex[(size_t) c]);
                    }
                cudaGetLastError();
                dl_capture_off_ = true;
                std::fprintf(stderr, "fnx: draft_lanes: a graph could not be made (%zu held): no more captures, layouts without one run eagerly\n", c0.size());
                ++dl_stat.eager;
            }
        } else if (dl_at < 0) {
            ++dl_stat.eager;
        }
        if (dl_at >= 0)
            for (CardBuf* b : cb_) b->dl_cache[(size_t) dl_at].used = ++graph_tick_;
    } else {
        ++dl_stat.eager;
    }
    // FNX_DRAFT_LANES_CHECK=1 (a test, with the table form): after the call (table form, its graph, on the state as a normal call finds it)
    // the launches a lane run again eagerly as the reference, from a copy of the source rows taken before the call (a chain step's source,
    // mtp_R, is overwritten by the call) and with a new ring tag; the rows' best tokens and values and the candidates of both are compared bit
    // for bit. A write the table form missed (source rows, K/V cells, keys) shows as a difference: the reference makes its own. The reference
    // rewrites the call's cells with the same values when the forms agree; the call's results are the ones returned.
    static const bool dl_check = env_int("FNX_DRAFT_LANES_CHECK", 0) != 0;
    bool tf = dl_tab, src_saved = false;   // the pass's form; its source rows come from the check's copy
    const bool io_pack = draft_io_pack();   // (draft_rows: the call's small copies as one upload and one download a card)
    const auto up = [&](int c, bool set) {
        CardBuf& b = *cb_[c];
        if (set) ck(cudaSetDevice(b.dev), "set");
        std::vector<const int*> rows((size_t) LANE_ROW_COLS * MAXT, nullptr);
        if (new_tab)
        for (int i = 0; i < nl; ++i)
            for (int t = 0; t < reqs[(size_t) i].T; ++t) {
                const int r = off[(size_t) i] + t;
                const LaneDev& LD = b.lanes[(size_t) reqs[(size_t) i].lane];
                rows[(size_t) r] = b.mtp_row_pos + r;
                rows[(size_t) (MAXT + r)] = LD.pt;
                lane_row_extra(rows.data(), r, LD);
                if (dl_tab) rows[(size_t) 3 * MAXT + r] = reinterpret_cast<const int*>(LD.mtp_ring);   // (this table's column 3: the draft layer's ring)
            }
        if (new_tab) ck(cudaMemcpy(b.mtp_lane_rows, rows.data(), rows.size() * sizeof(const int*), cudaMemcpyHostToDevice), "draft lane rows");
        if (tf) {   // the source rows' copies as jobs of the call's copy_many launch
            CopyJob* jobs = static_cast<CopyJob*>(h_.mtp_jobs) + (size_t) c * MAXT;
            for (int i = 0; i < nl; ++i) {
                const DraftReq& q = reqs[(size_t) i];
                jobs[i] = CopyJob{b.mtp_src + (size_t) off[(size_t) i] * ROW_W, draft_src(b, q.from_main) + (size_t) q.src_row * ROW_W, (size_t) q.T * ROW_W * 4};
            }
            ck(cudaMemcpyAsync(b.mtp_jobs, jobs, (size_t) nl * sizeof(CopyJob), cudaMemcpyHostToDevice, b.st), "draft source jobs");
        }
        for (int i = 0; i < (tf ? 0 : nl); ++i) {
            const DraftReq& q = reqs[(size_t) i];
            ck(cudaMemcpyAsync(b.mtp_src + (size_t) off[(size_t) i] * ROW_W,
                               src_saved ? b.mtp_ck_src + (size_t) off[(size_t) i] * ROW_W : draft_src(b, q.from_main) + (size_t) q.src_row * ROW_W,
                               (size_t) q.T * ROW_W * 4, cudaMemcpyDeviceToDevice, b.st), "draft source rows");
        }
        if (io_pack) {   // the embeddings, both control blocks, the rows' positions and every lane's parameters in one copy (draft_io)
            ck(cudaMemcpyAsync(b.mtp_io, h_.mtp_io, draft_io::EMB + (size_t) T * N * 4, cudaMemcpyHostToDevice, b.st), "draft inputs");
        } else {
            ck(cudaMemcpyAsync(b.mtp_emb, h_.mtp_emb, (size_t) T * N * 4, cudaMemcpyHostToDevice, b.st), "draft embeddings");
            ck(cudaMemcpyAsync(b.mtp_ctl, h_.mtp_ctl, 2 * sizeof(int), cudaMemcpyHostToDevice, b.st), "draft ctl");
            ck(cudaMemcpyAsync(b.mtp_lctl, h_.mtp_lctl, (size_t) 2 * nl * sizeof(int), cudaMemcpyHostToDevice, b.st), "draft lane ctl");
            ck(cudaMemcpyAsync(b.mtp_row_pos, h_.mtp_row_pos, (size_t) T * sizeof(int), cudaMemcpyHostToDevice, b.st), "draft row positions");
            if (tf && (any_pen || any_cands))   // every lane's parameters in one copy (a lane that neither penalizes nor samples: not read)
                ck(cudaMemcpyAsync(b.mtp_pen_par_l, h_.mtp_pen_par_l, (size_t) 3 * nl * 4, cudaMemcpyHostToDevice, b.st), "draft penalties");
        }
        for (int i = 0; i < nl; ++i)
            if (reqs[(size_t) i].pen) {
                ck(cudaMemcpyAsync(b.mtp_pen_l + (size_t) i * (1 + 2 * PEN_CAP), h_.mtp_pen_l + ((size_t) i * NCARD + c) * (1 + 2 * PEN_CAP),
                                   (size_t) (1 + 2 * pen_k[(size_t) i][c]) * 4, cudaMemcpyHostToDevice, b.st), "draft penalties");
                if (!tf && !io_pack) ck(cudaMemcpyAsync(b.mtp_pen_par_l + 3 * i, h_.mtp_pen_par_l + 3 * i, 3 * 4, cudaMemcpyHostToDevice, b.st), "draft penalties");
            } else if (reqs[(size_t) i].cands && !tf && !io_pack) {
                ck(cudaMemcpyAsync(b.mtp_pen_par_l + 3 * i, h_.mtp_pen_par_l + 3 * i, 3 * 4, cudaMemcpyHostToDevice, b.st), "draft penalties");
            }
        ck(cudaEventRecord(b.ev0, b.st), "event");
    };
    const auto down = [&](int c, bool set) {
        CardBuf& b = *cb_[c];
        if (set) ck(cudaSetDevice(b.dev), "set");
        if (!draft_ev_last()) ck(cudaEventRecord(b.ev1, b.st), "event");
        if (io_pack) {   // the rows' best tokens in one copy (land() below moves them to the host's rows)
            ck(cudaMemcpyAsync(h_.mtp_res + (size_t) c * draft_io::RES_END + draft_io::BV, b.mtp_bv, (size_t) 2 * MAXT * 4, cudaMemcpyDeviceToHost,
                               b.st), "draft results");
        } else {
            ck(cudaMemcpyAsync(h_.bv + (size_t) c * MAXT, b.mtp_bv, (size_t) T * 4, cudaMemcpyDeviceToHost, b.st), "draft v");
            ck(cudaMemcpyAsync(h_.bi + (size_t) c * MAXT, b.mtp_bi, (size_t) T * 4, cudaMemcpyDeviceToHost, b.st), "draft i");
        }
        ck(cudaMemcpyAsync(&h_.err[c], err_word(b), 4, cudaMemcpyDeviceToHost, b.st), "window err");
        const MtpCard& mc = mtp_->cards[c];
        if (tf && any_cands && mc.dv1 > mc.dv0) {   // every lane's candidates in three copies: the host's rows by card, then lane (cand_at)
            const size_t at = (size_t) c * MAXT;
            ck(cudaMemcpyAsync(h_.mtp_tk_v_l + at * TOPK_N, b.mtp_tk_v_l, (size_t) nl * TOPK_N * 4, cudaMemcpyDeviceToHost, b.st), "draft top v");
            ck(cudaMemcpyAsync(h_.mtp_tk_i_l + at * TOPK_N, b.mtp_tk_i_l, (size_t) nl * TOPK_N * 4, cudaMemcpyDeviceToHost, b.st), "draft top i");
            ck(cudaMemcpyAsync(h_.mtp_tk_err_l + at, b.mtp_tk_err_l, (size_t) nl * 4, cudaMemcpyDeviceToHost, b.st), "draft top err");
        }
        for (int i = 0; i < (tf ? 0 : nl); ++i)
            if (reqs[(size_t) i].cands && mc.dv1 > mc.dv0) {
                const size_t at = (size_t) i * NCARD + c;
                ck(cudaMemcpyAsync(h_.mtp_tk_v_l + at * TOPK_N, b.mtp_tk_v_l + (size_t) i * TOPK_N, TOPK_N * 4, cudaMemcpyDeviceToHost, b.st), "draft top v");
                ck(cudaMemcpyAsync(h_.mtp_tk_i_l + at * TOPK_N, b.mtp_tk_i_l + (size_t) i * TOPK_N, TOPK_N * 4, cudaMemcpyDeviceToHost, b.st), "draft top i");
                ck(cudaMemcpyAsync(h_.mtp_tk_err_l + at, b.mtp_tk_err_l + i, 4, cudaMemcpyDeviceToHost, b.st), "draft top err");
            }
        if (draft_ev_last()) ck(cudaEventRecord(b.ev1, b.st), "event");   // the last thing queued on the card (draft_ev_last above)
    };
    // FNX_DRAFT_THREADS=1 (default off: 2 cards measured +1.5 % at 4 users, inside the noise; an arm for 4 cards): each card's uploads, its ~50-60 eager launches and its downloads from that card's pool thread. In card
    // order on one thread a card had its part queued only after the cards before it (about 0.5 ms each), and each exchange waited for the last card's
    // launches to be queued: the call took about a host launch pass per card. The launches of a card and their order are unchanged.
    static const bool draft_threads = env_int("FNX_DRAFT_THREADS", 0) != 0;
    const bool dl_checking = dl_check && dl_tab;
    if (dl_checking)   // the source rows, before the call overwrites a chain step's
        for (int c = 0; c < NCARD; ++c) {
            CardBuf& b = *cb_[c];
            ck(cudaSetDevice(b.dev), "set");
            if (!b.mtp_ck_src) b.mtp_ck_src = dalloc<float>((size_t) MAXT * ROW_W);
            for (int i = 0; i < nl; ++i) {
                const DraftReq& q = reqs[(size_t) i];
                ck(cudaMemcpyAsync(b.mtp_ck_src + (size_t) off[(size_t) i] * ROW_W, draft_src(b, q.from_main) + (size_t) q.src_row * ROW_W,
                                   (size_t) q.T * ROW_W * 4, cudaMemcpyDeviceToDevice, b.st), "check source rows");
            }
        }
    if (dl_at >= 0) {   // the layout's graph, a thread a card (as draft_rows)
        on_cards([&](int c) {
            up(c, false);
            ck(cudaGraphLaunch(cb_[c]->dl_cache[(size_t) dl_at].exec, cb_[c]->st), "graph launch");
            down(c, false);
        });
    } else if (draft_threads) {
        on_cards([&](int c) {
            up(c, false);
            mtp_enqueue_lanes(reqs, T, c);
            down(c, false);
        });
    } else {
        for (int c = 0; c < NCARD; ++c) up(c, true);
        mtp_enqueue_lanes(reqs, T);
        for (int c = 0; c < NCARD; ++c) down(c, true);
    }
    // the packed results of card c into the host's rows (after the card's stream has ended)
    const auto land = [&](int c) {
        if (!io_pack) return;
        const char* r = h_.mtp_res + (size_t) c * draft_io::RES_END;
        std::memcpy(h_.bv + (size_t) c * MAXT, r + draft_io::BV, (size_t) T * 4);
        std::memcpy(h_.bi + (size_t) c * MAXT, r + draft_io::BI, (size_t) T * 4);
    };
    unsigned err = 0;
    float gpu_ms = 0.f;
    for (int c = 0; c < NCARD; ++c) {
        ck(cudaSetDevice(cb_[c]->dev), "set");
        ck(cudaStreamSynchronize(cb_[c]->st), "draft sync");
        land(c);
        err |= h_.err[c];
        float ms = 0.f;
        ck(cudaEventElapsedTime(&ms, cb_[c]->ev0, cb_[c]->ev1), "event time");
        gpu_ms = std::max(gpu_ms, ms);
    }
    last_draft_gpu_ms = gpu_ms;
    if (dl_checking) {   // FNX_DRAFT_LANES_CHECK: the reference pass (the launches a lane), then the comparison
        const size_t nb = (size_t) NCARD * MAXT, nt = (size_t) MAXT * NCARD * TOPK_N;
        const std::vector<float> t_bv(h_.bv, h_.bv + nb), t_tv(h_.mtp_tk_v_l, h_.mtp_tk_v_l + nt);
        const std::vector<int> t_bi(h_.bi, h_.bi + nb), t_ti(h_.mtp_tk_i_l, h_.mtp_tk_i_l + nt);
        const std::vector<unsigned> t_te(h_.mtp_tk_err_l, h_.mtp_tk_err_l + nb), t_err(h_.err, h_.err + NCARD);
        for (CardBuf* b : cb_) {   // the call's results off the cards: the reference must write its own
            ck(cudaSetDevice(b->dev), "set");
            ck(cudaMemset(b->mtp_bv, 0xff, (size_t) MAXT * 4), "check");
            ck(cudaMemset(b->mtp_bi, 0xff, (size_t) MAXT * 4), "check");
            ck(cudaMemset(b->mtp_tk_v_l, 0xff, (size_t) MAXT * TOPK_N * 4), "check");
            ck(cudaMemset(b->mtp_tk_i_l, 0xff, (size_t) MAXT * TOPK_N * 4), "check");
        }
        tf = false, src_saved = true;
        h_.mtp_ctl[1] = (int) ring_tag();
        for (int c = 0; c < NCARD; ++c) {
            up(c, true);
            if (io_pack)   // (the packed upload wrote zeros over the marks above: mark the rows' results again, after it on the stream)
                ck(cudaMemsetAsync(cb_[c]->mtp_bv, 0xff, (size_t) 2 * MAXT * 4, cb_[c]->st), "check");
        }
        mtp_enqueue_lanes(reqs, T, -1, 0);
        for (int c = 0; c < NCARD; ++c) down(c, true);
        unsigned r_err = 0;
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            ck(cudaStreamSynchronize(cb_[c]->st), "draft sync");
            land(c);
            r_err |= h_.err[c];
        }
        tf = true, src_saved = false;
        ++dl_stat.checked;
        int bad = r_err != err;
        for (int c = 0; c < NCARD; ++c) {
            if (mtp_->cards[c].dv1 <= mtp_->cards[c].dv0) continue;
            bad += std::memcmp(&t_bv[(size_t) c * MAXT], h_.bv + (size_t) c * MAXT, (size_t) T * 4) != 0;
            bad += std::memcmp(&t_bi[(size_t) c * MAXT], h_.bi + (size_t) c * MAXT, (size_t) T * 4) != 0;
            for (int i = 0; i < nl; ++i) {
                if (!reqs[(size_t) i].cands) continue;
                const size_t ref = ((size_t) i * NCARD + c) * TOPK_N, tab = ((size_t) c * MAXT + i) * TOPK_N;
                bad += std::memcmp(h_.mtp_tk_v_l + ref, &t_tv[tab], TOPK_N * 4) != 0;
                bad += std::memcmp(h_.mtp_tk_i_l + ref, &t_ti[tab], TOPK_N * 4) != 0;
            }
        }
        if (bad && ++dl_stat.check_bad <= 8)
            std::fprintf(stderr, "fnx: draft_lanes check: call %lu differs from the launches a lane (%d parts; %d lanes, %d rows, graph %d, err %u / %u)\n",
                         dl_stat.checked, bad, nl, T, dl_at, err, r_err);
        // the call's results back (the table form's host rows)
        std::memcpy(h_.bv, t_bv.data(), nb * 4), std::memcpy(h_.bi, t_bi.data(), nb * 4);
        std::memcpy(h_.mtp_tk_v_l, t_tv.data(), nt * 4), std::memcpy(h_.mtp_tk_i_l, t_ti.data(), nt * 4);
        std::memcpy(h_.mtp_tk_err_l, t_te.data(), nb * 4), std::memcpy(h_.err, t_err.data(), NCARD * 4);
        err |= r_err;   // an error of the reference pass fails the call too (its launches changed the cards' state)
    }
    if (new_tab) dl_tab_key_ = dl_key;   // every card's copy of the table went through
    if (const char* e = window_error(err, true)) {
        diag_report("draft_lanes");
        throw std::runtime_error(std::string("draft_lanes: ") + e + err_words());
    }
    for (int i = 0; i < nl; ++i) {
        if (!reqs[(size_t) i].cands) continue;
        std::vector<std::pair<float, int>>& cd = last_lane_cands[(size_t) i];
        for (int c = 0; c < NCARD; ++c) {
            if (mtp_->cards[c].dv1 <= mtp_->cards[c].dv0) continue;
            const size_t at = dl_tab ? (size_t) c * MAXT + i : (size_t) i * NCARD + c;
            // (table form: the candidates' one launch sets the bit in the first lane's word)
            if (h_.mtp_tk_err_l[dl_tab ? (size_t) c * MAXT : at] & 16u) throw std::runtime_error("draft_lanes: the draft layer's logits hold a NaN or inf");
            for (int k = 0; k < TOPK_N; ++k) cd.emplace_back(h_.mtp_tk_v_l[at * TOPK_N + k], mtp_->vocab[h_.mtp_tk_i_l[at * TOPK_N + k]]);
        }
        std::sort(cd.begin(), cd.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first || (a.first == b.first && a.second < b.second);
        });
        cd.resize(TOPK_N);
    }
    std::vector<int> out((size_t) nl);
    for (int i = 0; i < nl; ++i) {
        const int row = off[(size_t) i] + reqs[(size_t) i].T - 1;
        int best = 0;
        for (int c = 1; c < NCARD; ++c)
            if (mtp_->cards[c].dv1 > mtp_->cards[c].dv0 && h_.bv[(size_t) c * MAXT + row] > h_.bv[(size_t) best * MAXT + row]) best = c;
        out[(size_t) i] = mtp_->vocab[h_.bi[(size_t) best * MAXT + row]];
    }
    last_draft_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

// ---- DFlash2 ------------------------------------------------------------------------------------------------------------------------------------
#if FNX_DF2
void Engine::df2_forget(int lane) {
    static const bool trace = env_flag("FNX_DF2_TRACE");
    if (trace) std::fprintf(stderr, "df2: lane %d forgotten (drafter ring_hi %ld, host ring [%ld, %ld))\n", lane, (long) df2_->ring_hi(lane), df2_lane_[(size_t) lane].lo, df2_lane_[(size_t) lane].hi);
    df2_->reset(lane);
    df2_lane_[(size_t) lane].lo = df2_lane_[(size_t) lane].hi = 0;
}

Engine::Df2Info Engine::df2_info(int lane) const {
    Df2Info i;
    if (!df2_) return i;
    i.lo = df2_lane_[(size_t) lane].lo, i.hi = df2_lane_[(size_t) lane].hi, i.ring_hi = (long) df2_->ring_hi(lane);
    return i;
}

// the lane's drafter brought up to position pos0 (the first row of the window about to be drafted from) with what the prompt reads tapped: its ring holds
// the positions below pos0 in the host ring [lo, hi) that it has not got yet, the last 2047 at most. A ring that is ahead of pos0 (the sequence went back) or
// does not touch the host ring's run starts over; whatever is missing is left out (the drafts are poorer for a while, never wrong: the verify decides)
void Engine::df2_catch_up(int lane, int pos0) {
    Df2Lane& L = df2_lane_[(size_t) lane];
    df2::Bridge& B = *df2_;
    long have = (long) B.ring_hi(lane);
    static const bool trace = env_flag("FNX_DF2_TRACE");
    if (trace && have != pos0) std::fprintf(stderr, "df2: lane %d catch-up at %d: drafter ring_hi %ld, host ring [%ld, %ld)\n", lane, pos0, have, L.lo, L.hi);
    if (have == pos0) return;
    if (have > pos0) {
        B.reset(lane);
        have = 0;
    }
    if (!L.ring || L.hi <= L.lo) return;
    long from = std::max({L.lo, have, (long) pos0 - df2::kRing + 1});
    const long to = std::min<long>(L.hi, pos0);
    if (from >= to) return;
    constexpr size_t rowf = (size_t) df2::kTaps * df2::kNEmbd;
    std::string err;
    while (from < to) {   // (the ring wraps at kRing positions: at most two runs)
        const long slot = from % df2::kRing, n = std::min<long>(to - from, df2::kRing - slot);
        if (!B.context(lane, L.ring + (size_t) slot * rowf, n, from, err)) throw std::runtime_error("dflash2: " + err);
        from += n;
    }
}

void Engine::df2_defer(const Df2Req& r) {
    if (!df2_) throw std::runtime_error("df2_defer: no DFlash2 attached");
    if (r.used < 1 || r.row0 < 0 || r.row0 + r.used > MAXT) throw std::runtime_error("df2_defer: rows outside the window");
    Df2Lane& L = df2_lane_[(size_t) r.lane];
    if (!L.ring) {
        ck(cudaSetDevice(cb_[0]->dev), "set");
        ck(cudaHostAlloc((void**) &L.ring, (size_t) df2::kRing * df2::kTaps * df2::kNEmbd * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "df2 tap ring");
        own_pinned(L.ring);
    }
    constexpr size_t rowf = (size_t) df2::kTaps * df2::kNEmbd;
    ck(cudaSetDevice(cb_[0]->dev), "set");
    long pos = r.pos0;
    for (int done = 0; done < r.used;) {
        const int at = (int) (pos % df2::kRing), k = std::min(r.used - done, df2::kRing - at);
        ck(cudaMemcpy(L.ring + (size_t) at * rowf, df2_tapw_ + (size_t) (r.row0 + done) * rowf, (size_t) k * rowf * 4, cudaMemcpyDeviceToHost), "df2 defer");
        done += k;
        pos += k;
    }
    const long e = (long) r.pos0 + r.used;
    if (L.hi == r.pos0 && L.hi > L.lo) L.hi = e, L.lo = std::max<long>(L.lo, e - df2::kRing);
    else L.lo = r.pos0, L.hi = e;
}

void Engine::df2_draft(std::vector<Df2Req>& reqs) {
    if (!df2_) throw std::runtime_error("df2_draft: no DFlash2 attached");
    const auto t0 = std::chrono::steady_clock::now();
    df2::Bridge& B = *df2_;
    ck(cudaSetDevice(cb_[0]->dev), "set");
    constexpr size_t rowf = (size_t) df2::kTaps * df2::kNEmbd;
    std::string err;
    for (Df2Req& r : reqs) {
        if (r.used < 1 || r.row0 < 0 || r.row0 + r.used > MAXT) throw std::runtime_error("df2_draft: rows outside the window");
        if (r.lane < 0 || r.lane >= B.lanes()) throw std::runtime_error("df2_draft: no drafter for this lane");
        df2_catch_up(r.lane, r.pos0);
        const float* src = df2_tapw_ + (size_t) r.row0 * rowf;
        int n = r.used;
        long p0 = r.pos0;
        if (n > df2::kRows) {   // a lookup window kept more rows than a round takes: the first ones only update the ring
            if (!B.context(r.lane, src, n - df2::kRows, p0, err)) throw std::runtime_error("dflash2: " + err);
            src += (size_t) (n - df2::kRows) * rowf;
            p0 += n - df2::kRows;
            n = df2::kRows;
        }
        ck(cudaMemcpyAsync(B.landing(r.lane), src, (size_t) n * rowf * 4, cudaMemcpyDeviceToDevice, (cudaStream_t) B.stream(r.lane)), "df2 taps");
        if (!B.launch(r.lane, n, p0, r.anchor, err)) throw std::runtime_error("dflash2: " + err);
    }
    for (Df2Req& r : reqs)
        if (!B.finish(r.lane, r.tokens, r.probs, err)) throw std::runtime_error("dflash2: " + err);
    last_df2_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    static const bool trace = env_flag("FNX_DF2_TRACE");
    if (trace) {   // the mean wall time of a call by the number of lanes in it, every 400 calls
        static double ms[MAXT + 1] = {};
        static long n[MAXT + 1] = {}, calls = 0;
        const size_t k = std::min<size_t>(reqs.size(), MAXT);
        ms[k] += last_df2_ms, ++n[k];
        if (++calls % 400 == 0) {
            std::fprintf(stderr, "df2: drafter calls by lanes (mean ms):");
            for (int i = 1; i <= MAXT; ++i)
                if (n[i]) std::fprintf(stderr, " %d lanes %.2f (%ld)", i, ms[i] / (double) n[i], n[i]);
            std::fprintf(stderr, "%c", 10);
        }
    }
}
#else
void Engine::df2_forget(int) {}
Engine::Df2Info Engine::df2_info(int) const { return {}; }
void Engine::df2_catch_up(int, int) {}
void Engine::df2_draft(std::vector<Df2Req>&) { throw std::runtime_error("this engine was built without DFlash2"); }
void Engine::df2_defer(const Df2Req&) {}
#endif

}  // namespace fnx
