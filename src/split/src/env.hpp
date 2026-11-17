// Numbers parsed whole, for sFNX's settings from the environment and its request lines: a value that is not a number
// is an error naming the variable and the value (std::atoi and friends read garbage as 0, which turned a setting off
// without a word, and std::stoi / stof stop at the first character they cannot read, so "0.7x" was 0.7).
#pragma once

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace fnx {

// s as a whole number in base 10: an optional sign, digits, nothing else, within long long
inline bool parse_whole(const char* s, long long& x) {
    if (!(s[0] == '-' || s[0] == '+' || (s[0] >= '0' && s[0] <= '9'))) return false;
    char* end = nullptr;
    errno = 0;
    x = std::strtoll(s, &end, 10);
    return *end == '\0' && end != s && errno != ERANGE;
}

// s as a finite decimal number (strtod also reads hex floats, inf and nan, and skips leading spaces: none of them pass)
inline bool parse_number(const char* s, double& x) {
    if (!(s[0] == '-' || s[0] == '+' || s[0] == '.' || (s[0] >= '0' && s[0] <= '9')) || std::strpbrk(s, "xX")) return false;
    char* end = nullptr;
    errno = 0;
    x = std::strtod(s, &end);
    return *end == '\0' && end != s && errno != ERANGE && std::isfinite(x);
}

// the variable as a whole number (parse_whole); def when it is not set
inline long long env_int(const char* name, long long def) {
    const char* v = std::getenv(name);
    if (!v) return def;
    long long x = 0;
    if (!parse_whole(v, x)) throw std::runtime_error(std::string(name) + ": '" + v + "' is not a whole number");
    return x;
}

// the variable as a finite decimal number (parse_number); def when it is not set
inline double env_num(const char* name, double def) {
    const char* v = std::getenv(name);
    if (!v) return def;
    double x = 0;
    if (!parse_number(v, x)) throw std::runtime_error(std::string(name) + ": '" + v + "' is not a number");
    return x;
}

// an on/off switch: a whole number, on when not 0; off when it is not set
inline bool env_flag(const char* name) { return env_int(name, 0) != 0; }

// The engine's two opt-in sets of faster paths (build options STRATA_SPLIT_FAST and STRATA_SPLIT_APPROX; docs/SPLIT_ENGINE.md):
//   FAST    the paths that give the default build's output to the last bit
//   APPROX  the paths that change the output's last bits (sums in another order, f16 for values that go between the cards); needs FAST
// Neither (the default build): the engine takes the paths it took before them. A path's own variable still overrides its default, for a
// measurement: FNX_X=0 turns one path off, FNX_X=1 turns one on where the build holds its code. A path's finer settings (a variable that only
// acts while the path is on, or that sits in code only an option builds) keep plain defaults. Call sites keep the answer in a static: the
// environment is read once.
// Not behind an option (the same bits by construction, checked on every row count and call shape with kbench hchash): the hc read's inject-row
// weights in shared memory and its up pass skipped by the waves with no part in it (hc.cuh).
#ifndef FNX_FAST
#define FNX_FAST 0
#endif
#ifndef FNX_APPROX
#define FNX_APPROX 0
#endif
#if FNX_APPROX && !FNX_FAST
#error "FNX_APPROX needs FNX_FAST (STRATA_SPLIT_APPROX needs STRATA_SPLIT_FAST)"
#endif
constexpr bool FAST = FNX_FAST != 0, APPROX = FNX_APPROX != 0;
// a FAST path's switch NAME: on in a FAST build unless the variable says otherwise
inline bool fast_opt(const char* name) { return env_int(name, FAST ? 1 : 0) != 0; }

}  // namespace fnx
