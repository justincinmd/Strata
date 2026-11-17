// tk.hpp (generator): kernels launched from code objects that gen/tkgen.py made ahead of time, in place of a kernel compiled into this
// binary. A FAMILY is one launch site's kernel (pf_attn_u, hc_mm, ...). A VARIANT is one setting of the family's parameters (template arguments and -D
// constants, or a Triton configuration): tkgen compiles it into one code object with a kernel per KEY (the window's rows for a decode kernel, 0 when the
// family has one kernel). An experiment picks a variant by name at run time: no rebuild of the engine.
//   FNX_TK_DIR=<dir>            the directory of manifest.tk and its code objects (unset: every family is the kernel in the binary)
//   FNX_TK_<FAMILY>=<variant>   the family's variant; 0 or builtin: the kernel in the binary. Unset: the manifest's `use` line, else the binary's.
//                               Set to an empty string: an error (it would drop a `use` line without a word)
//   FNX_TK_LOG=1                a stderr line per family at its first use
// stderr: `fnx: tk: <family> = <variant> (...)` at the start for every selected variant, and `fnx: tk: <family> = <variant> runs` at its first
// launch (a selected family with no `runs` line did not launch in that run: e.g. moe_gu_q4k on a file with no Q4_K gate|up weights).
// A variant that is named and cannot be used (not in the manifest, a manifest made for another build, a load error, a key it lacks) is an error, never a
// silent fall back: an arm must run what its name says. An FNX_TK_<X> that names no family of the manifest is an error too (a mistyped name).
// A launch site:
//   static const tk::Family* f = tk::family("pf_attn_u");          // nullptr: built in
//   if (f) { const tk::Kern& k = f->kern(0); FNX_TK_LAUNCH(k, k_pf_attn_u<4>, groups, k.threads, k.smem, s, qg, k16, ...); }
// FNX_TK_LAUNCH takes the built-in kernel's name only for its parameter types (decltype: nothing is instantiated): the arguments convert as in a direct
// launch, and their count and byte sizes are checked against the code object's kernel (its metadata's arguments, without the hidden ones the runtime
// fills), so a kernel made with another argument list does not run.
// manifest.tk (text, fields split at white space; tkgen writes it next to manifest.json):
//   tk 1 | cfg <FNX_TK_CFG of the target> | src <id> | use <family> <variant> | f <file id> <file> <sha256> <bytes> <FNV-1a 64 of the bytes, hex>
//   v <family> <variant> <parameters>
//   k <family> <variant> <key> <file id> <symbol> <threads> <shared bytes> <the arguments' byte sizes, a,b,c> <meta 0..7>
// (a kernel names its code object, so a variant can take each key's kernel from another one: the tuner's pick per shape)
// A line that is not exactly one of these, a second tk / cfg / src line, a family's second `use`, a variant's second `v`, a `k` before its `v`,
// more fields than the line has: errors. A family's name is lower case letters, digits and `_`.
// The bytes of a code object are checked against its f line when it is read (a file replaced after the manifest was written does not run); a
// selected variant's files are read and checked at the start (tk::init), so that check does not first fail inside a graph capture.
// A selected variant whose launch path is off in this run is an error at that path (tk::refuse), not a silent run of the built-in kernel.
// HIP only. Loading is lazy, per card, under the module's lock; a launch after that is a map-free pointer read.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "env.hpp"
#include "tk_cfg.hpp"

#if FNX_HIP
#include <unistd.h>   // environ
#endif

namespace fnx {
namespace tk {

constexpr int MAXDEV = 16, NMETA = 8;

#if FNX_HIP
struct Mod {
    std::string path;
    size_t bytes = 0;              // the manifest's: the file's size and the FNV-1a 64 of its bytes
    unsigned long long fnv = 0;
    std::vector<char> image;
    hipModule_t mod[MAXDEV] = {};
    std::mutex mu;
};
#endif

struct Kern {
    std::string family, variant, symbol;
    int key = 0, threads = 0;
    unsigned smem = 0;
    std::vector<unsigned> args;   // the byte size of each argument a launch passes
    int meta[NMETA] = {};
#if FNX_HIP
    Mod* mod = nullptr;
    std::atomic<bool>* ran = nullptr;   // the variant's: set at its first launch
    mutable std::atomic<hipFunction_t> fn[MAXDEV];
    Kern() {
        for (auto& f : fn) f.store(nullptr, std::memory_order_relaxed);
    }
#endif
};

struct Family {
    std::string name, variant;
    std::map<int, const Kern*> kerns;
    const Kern* find(int key) const {
        const auto it = kerns.find(key);
        return it == kerns.end() ? nullptr : it->second;
    }
    const Kern& kern(int key) const {
        const Kern* k = find(key);
        if (!k) throw std::runtime_error("tk: family " + name + " variant " + variant + " has no kernel for key " + std::to_string(key));
        return *k;
    }
};

#if FNX_HIP
namespace detail {

struct State {
    std::string dir, src;
    std::deque<Mod> mods;
    std::deque<Kern> kerns;
    std::deque<std::atomic<bool>> ran;                                           // a flag a variant (Kern::ran)
    std::map<std::pair<std::string, std::string>, std::atomic<bool>*> ran_of;
    std::map<std::string, std::map<std::string, std::vector<const Kern*>>> by;   // family -> variant -> kernels
    std::map<std::string, std::string> use;                                      // the manifest's defaults
    std::map<std::string, std::unique_ptr<Family>> fams;                         // resolved (null: built in)
    std::mutex mu;
};

inline std::string env_name(const std::string& family) {
    std::string n = "FNX_TK_";
    for (const char c : family) n += std::isalnum((unsigned char) c) ? (char) std::toupper((unsigned char) c) : '_';
    return n;
}

inline void load(State& st) {
    const char* dir = std::getenv("FNX_TK_DIR");
    st.dir = dir ? dir : "";
    if (!st.dir.empty()) {
        const std::string path = st.dir + "/manifest.tk";
        std::ifstream f(path);
        if (!f) throw std::runtime_error("tk: FNX_TK_DIR: cannot read " + path);
        std::map<std::string, Mod*> mod_of;   // file id -> the code object
        std::string line;
        bool head = false, cfg = false, src = false;
        int ln = 0;
        const auto fam_name = [](const std::string& n) {
            return !n.empty() && n.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_") == std::string::npos;
        };
        while (std::getline(f, line)) {
            ++ln;
            std::istringstream is(line);
            std::string w;
            if (!(is >> w) || w[0] == '#') continue;
            const auto bad = [&](const char* what) { return std::runtime_error("tk: " + path + " line " + std::to_string(ln) + ": " + what); };
            const auto done = [&] {   // the line has no more fields
                std::string more;
                if (is >> more) throw bad("more fields than this line has");
            };
            if (w == "tk") {
                int ver = 0;
                if (head) throw bad("a second tk line");
                if (!(is >> ver) || ver != 1) throw bad("this binary reads manifest version 1");
                done();
                head = true;
            } else if (w == "cfg") {
                std::string c;
                if (cfg) throw bad("a second cfg line");
                is >> c;
                if (c != FNX_TK_CFG)
                    throw std::runtime_error("tk: " + path + " was made for another build (its kernels' argument layouts may differ):\n  manifest " + c +
                                             "\n  binary   " FNX_TK_CFG);
                done();
                cfg = true;
            } else if (w == "src") {
                if (src) throw bad("a second src line");
                if (!(is >> st.src)) throw bad("src <id>");
                done();
                src = true;
            } else if (w == "use") {
                std::string fam, var;
                if (!(is >> fam >> var) || !fam_name(fam)) throw bad("use <family> <variant>");
                done();
                if (!st.use.emplace(fam, var).second) throw bad("a family's second use line");
            } else if (w == "f") {
                std::string id, file, sha, fnv;
                size_t bytes = 0;
                if (!(is >> id >> file >> sha >> bytes >> fnv) || !bytes || fnv.size() != 16 || fnv.find_first_not_of("0123456789abcdef") != std::string::npos)
                    throw bad("f <file id> <file> <sha256> <bytes> <fnv64>");
                done();
                if (mod_of.count(id)) throw bad("a file id twice");
                st.mods.emplace_back();
                st.mods.back().path = file[0] == '/' ? file : st.dir + "/" + file;
                st.mods.back().bytes = bytes;
                st.mods.back().fnv = std::stoull(fnv, nullptr, 16);
                mod_of[id] = &st.mods.back();
            } else if (w == "v") {
                std::string fam, var;   // (the parameters after the names are free text for a reader)
                if (!(is >> fam >> var) || !fam_name(fam)) throw bad("v <family> <variant> ... (a family is lower case letters, digits and _)");
                if (!st.by[fam].emplace(var, std::vector<const Kern*>()).second) throw bad("a variant's second v line");
                st.ran.emplace_back(false);
                st.ran_of[{fam, var}] = &st.ran.back();
            } else if (w == "k") {
                st.kerns.emplace_back();
                Kern& k = st.kerns.back();
                std::string fid, sizes;
                if (!(is >> k.family >> k.variant >> k.key >> fid >> k.symbol >> k.threads >> k.smem >> sizes)) throw bad("k: too few fields");
                // the sizes: digits and commas only, each 1..65536
                if (sizes.empty() || sizes.find_first_not_of("0123456789,") != std::string::npos || sizes.front() == ',' || sizes.back() == ',' ||
                    sizes.find(",,") != std::string::npos)
                    throw bad("k: the argument sizes are a,b,c");
                for (size_t i = 0; i < sizes.size();) {
                    const size_t j = std::min(sizes.find(',', i), sizes.size());
                    if (j - i > 5) throw bad("k: an argument size past 65536");
                    const unsigned long v = std::stoul(sizes.substr(i, j - i));
                    if (v < 1 || v > 65536) throw bad("k: an argument size out of 1..65536");
                    k.args.push_back((unsigned) v);
                    i = j + 1;
                }
                if (k.threads < 1 || k.threads > 1024) throw bad("k: block threads out of 1..1024");
                for (int& m : k.meta)
                    if (!(is >> m)) throw bad("k: too few meta fields");
                done();
                const auto it = mod_of.find(fid);
                if (it == mod_of.end()) throw bad("k names a file id with no f line before it");
                k.mod = it->second;
                const auto rn = st.ran_of.find({k.family, k.variant});
                if (rn == st.ran_of.end()) throw bad("k names a variant with no v line before it");
                k.ran = rn->second;
                for (const Kern* o : st.by[k.family][k.variant])
                    if (o->key == k.key) throw bad("k: a key twice in one variant");
                st.by[k.family][k.variant].push_back(&k);
            } else {
                throw bad("unknown line");
            }
        }
        if (!head || !cfg) throw std::runtime_error("tk: " + path + ": no `tk 1` or no `cfg` line");
    }
    // every FNX_TK_<X> names a family of the manifest
    for (char** e = ::environ; e && *e; ++e) {
        if (std::strncmp(*e, "FNX_TK_", 7) != 0) continue;
        const char* eq = std::strchr(*e, '=');
        const std::string name(*e, eq ? (size_t) (eq - *e) : std::strlen(*e)), val = eq ? eq + 1 : "";
        if (name == "FNX_TK_DIR" || name == "FNX_TK_LOG") continue;
        if (val.empty()) throw std::runtime_error("tk: " + name + " is set to an empty string: name a variant, or 0 / builtin for the kernel in the binary");
        bool known = false;
        for (const auto& fam : st.by) known = known || env_name(fam.first) == name;
        if (!known && val != "0" && val != "builtin")
            throw std::runtime_error("tk: " + name + "=" + val + " names no family of " + (st.dir.empty() ? std::string("a manifest (FNX_TK_DIR is not set)") : st.dir + "/manifest.tk"));
    }
}

inline State& state() {
    static State st;
    static std::once_flag once;
    std::call_once(once, [] {
        try {
            load(st);
        } catch (...) {   // a later call reads the manifest again: nothing of the failed read stays
            st.mods.clear(), st.kerns.clear(), st.ran.clear(), st.ran_of.clear(), st.by.clear(), st.use.clear(), st.fams.clear();
            st.dir.clear(), st.src.clear();
            throw;
        }
    });
    return st;
}

// the code object's bytes, checked against the manifest's (under m.mu). The image stays: every card's module is made from these bytes
inline void read_image(Mod& m) {
    if (!m.image.empty()) return;
    std::ifstream in(m.path, std::ios::binary);
    if (!in) throw std::runtime_error("tk: cannot read " + m.path);
    m.image.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    unsigned long long h = 1469598103934665603ull;
    for (const char c : m.image) h = (h ^ (unsigned char) c) * 1099511628211ull;
    if (m.image.size() != m.bytes || h != m.fnv) {
        m.image.clear();
        throw std::runtime_error("tk: " + m.path + " is not the code object of the manifest (its bytes changed after the manifest was written): build again");
    }
}

inline hipFunction_t fn_of(const Kern& k) {
    int dev = -1;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= MAXDEV) throw std::runtime_error("tk: no current card (or one past the " + std::to_string(MAXDEV) + " it tracks)");
    hipFunction_t f = k.fn[dev].load(std::memory_order_acquire);
    if (f) return f;
    Mod& m = *k.mod;
    std::lock_guard<std::mutex> lk(m.mu);
    if (!m.mod[dev]) {
        read_image(m);
        const hipError_t e = hipModuleLoadData(&m.mod[dev], m.image.data());
        if (e != hipSuccess) {
            m.mod[dev] = nullptr;
            throw std::runtime_error("tk: hipModuleLoadData " + m.path + " on card " + std::to_string(dev) + ": " + hipGetErrorString(e));
        }
    }
    const hipError_t e = hipModuleGetFunction(&f, m.mod[dev], k.symbol.c_str());
    if (e != hipSuccess || !f)
        throw std::runtime_error("tk: " + m.path + " has no kernel " + k.symbol + " (" + k.family + " " + k.variant + " key " + std::to_string(k.key) + "): " + hipGetErrorString(e));
    k.fn[dev].store(f, std::memory_order_release);
    return f;
}

inline void launch_ptrs(const Kern& k, dim3 grid, dim3 block, unsigned smem, cudaStream_t s, void** args, const unsigned* sizes, size_t n) {
    bool same = n == k.args.size();
    for (size_t i = 0; same && i < n; ++i) same = sizes[i] == k.args[i];
    if (!same) {
        std::string a, b;
        for (size_t i = 0; i < n; ++i) a += (i ? "," : "") + std::to_string(sizes[i]);
        for (const unsigned v : k.args) b += (b.empty() ? "" : ",") + std::to_string(v);
        throw std::runtime_error("tk: " + k.family + " " + k.variant + " key " + std::to_string(k.key) + ": the launch passes arguments of " + a +
                                 " bytes, the code object's kernel takes " + b + " (made from another source?)");
    }
    const hipError_t e = hipModuleLaunchKernel(fn_of(k), grid.x, grid.y, grid.z, block.x, block.y, block.z, smem, s, args, nullptr);
    if (e != hipSuccess) throw std::runtime_error("tk: launch of " + k.family + " " + k.variant + ": " + hipGetErrorString(e));
    if (!k.ran->load(std::memory_order_relaxed) && !k.ran->exchange(true))   // once a variant: the run's log says that it launched
        std::fprintf(stderr, "fnx: tk: %s = %s runs (first launch: key %d)\n", k.family.c_str(), k.variant.c_str(), k.key);
}

}  // namespace detail

namespace detail {
// the family's selection: its variant's kernels, or nullptr (the kernel in the binary). FNX_TK_<FAMILY>, else the manifest's `use` line
inline const std::vector<const Kern*>* selected(State& st, const std::string& name, std::string& var) {
    const std::string en = env_name(name);
    const char* ev = std::getenv(en.c_str());
    if (ev) {
        var = ev;
        if (var.empty()) throw std::runtime_error("tk: " + en + " is set to an empty string: name a variant, or 0 / builtin for the kernel in the binary");
    } else {
        const auto u = st.use.find(name);
        var = u != st.use.end() ? u->second : "";
    }
    if (var.empty() || var == "0" || var == "builtin") return nullptr;
    const std::string how = ev ? en + "=" + var : "manifest `use " + name + " " + var + "`";
    const auto fi = st.by.find(name);
    if (fi == st.by.end()) throw std::runtime_error("tk: " + how + ": the manifest has no family " + name);
    const auto vi = fi->second.find(var);
    if (vi == fi->second.end()) {
        std::string have;
        for (const auto& v : fi->second) have += " " + v.first;
        throw std::runtime_error("tk: " + how + ": no such variant; the manifest has:" + have);
    }
    if (vi->second.empty()) throw std::runtime_error("tk: " + name + " " + var + ": no kernels in the manifest");
    return &vi->second;
}
}  // namespace detail

// the family's selected variant, or nullptr: the kernel in the binary. A launch site keeps the answer in a static
inline const Family* family(const char* name) {
    detail::State& st = detail::state();
    std::lock_guard<std::mutex> lk(st.mu);
    const auto done = st.fams.find(name);
    if (done != st.fams.end()) return done->second.get();
    std::string var;
    std::unique_ptr<Family> fam;
    if (const auto* ks = detail::selected(st, name, var)) {
        fam = std::make_unique<Family>();
        fam->name = name, fam->variant = var;
        for (const Kern* k : *ks) fam->kerns[k->key] = k;
    }
    static const bool log = env_int("FNX_TK_LOG", 0) != 0;
    if (log) std::fprintf(stderr, "tk: %s = %s\n", name, fam ? (fam->variant + " (" + st.dir + ", src " + st.src + ")").c_str() : "built in");
    return (st.fams[name] = std::move(fam)).get();
}

// a launch path that cannot take the family's variant (its switch is off in this run, another weight type): an error when a variant is selected, so
// that an arm does not run the built-in kernel under a variant's name. Call it once at that path (a static)
inline bool refuse(const char* name, const char* why) {
    if (const Family* f = family(name))
        throw std::runtime_error("tk: " + std::string(name) + " = " + f->variant + " is selected, but " + why);
    return true;
}

// the engine's start: reads the manifest and checks every selection (an FNX_TK_<FAMILY>, a `use` line) against it, so a mistyped arm fails at the
// start and not at its family's first launch. sites: the families this binary has a launch site for; a selected family that is not one of them is
// an error (it would be logged and never run). A selected variant's code objects are read and their bytes checked here. A line on stderr per
// selected variant (an arm's log says what was picked; the `runs` line at the first launch says that it ran)
inline void init(std::initializer_list<const char*> sites) {
    detail::State& st = detail::state();
    std::lock_guard<std::mutex> lk(st.mu);
    for (const auto& u : st.use)
        if (!st.by.count(u.first)) throw std::runtime_error("tk: manifest `use " + u.first + " " + u.second + "`: the manifest has no family " + u.first);
    for (const auto& f : st.by) {
        std::string var;
        if (const auto* ks = detail::selected(st, f.first, var)) {
            bool site = false;
            for (const char* s : sites) site = site || f.first == s;
            if (!site) throw std::runtime_error("tk: " + f.first + " = " + var + " is selected, but this binary has no launch site for the family");
            for (const Kern* k : *ks) {
                std::lock_guard<std::mutex> ml(k->mod->mu);
                detail::read_image(*k->mod);
            }
            std::fprintf(stderr, "fnx: tk: %s = %s (%zu kernels; %s, src %s)\n", f.first.c_str(), var.c_str(), ks->size(), st.dir.c_str(), st.src.c_str());
        }
    }
}

template <typename F> struct Sig;
template <typename... P> struct Sig<void(P...)> {
    static void launch(const Kern& k, dim3 grid, dim3 block, unsigned smem, cudaStream_t s, P... p) {
        void* a[] = {(void*) &p...};
        static constexpr unsigned sizes[] = {(unsigned) sizeof(P)...};
        detail::launch_ptrs(k, grid, block, smem, s, a, sizes, sizeof...(P));
    }
};
#define FNX_TK_LAUNCH(K, KERNEL, GRID, BLOCK, SMEM, STREAM, ...) \
    ::fnx::tk::Sig<decltype(KERNEL)>::launch(K, GRID, BLOCK, SMEM, STREAM, __VA_ARGS__)

#else   // !FNX_HIP: every family is the kernel in the binary
inline const Family* family(const char*) { return nullptr; }
inline bool refuse(const char*, const char*) { return true; }
inline void init(std::initializer_list<const char*>) {}
#define FNX_TK_LAUNCH(...) throw std::logic_error("tk: code objects are HIP only")
#endif

}  // namespace tk
}  // namespace fnx
