// what a wait that timed out leaves behind its card's error word (kernels.hpp: ErrDiag, DiagKind). Device code, header only.
// Called only inside a time-out branch: a window without a time-out runs none of it.
#pragma once

#include <cuda_runtime.h>

#include "kernels.hpp"

namespace fnx {

// the card's steady counter (constant rate: hipDeviceAttributeWallClockRate); clock64 is the core clock's cycle counter, whose rate moves with the clock
__device__ __forceinline__ long long diag_wall() {
#if FNX_HIP
    return wall_clock64();
#else
    return 0;
#endif
}

// The wait of `kind` for `tag` gave up after `waited` cycles; the word held (seen_hi, seen_lo); src: the card whose word it was (DG_NO_SRC: none);
// elem: the element. Every caller counts itself; the first on the card (a compare-and-swap that claims the record with kind, source and element)
// writes the rest. Kept to a few stores: this is inlined into every wait's time-out branch.
__device__ __forceinline__ void diag_timeout(unsigned* err, unsigned kind, unsigned tag, unsigned seen_hi, unsigned seen_lo, unsigned src, unsigned elem,
                                             long long waited) {
    atomicAdd(err + DG_COUNT, 1u);
    atomicOr(err + DG_SRCMASK, 1u << (src & 15u));
    if (atomicCAS(err + DG_CLAIM, 0u, diag_claim(kind, src, elem)) != 0u) return;
    volatile unsigned* d = err;
    const unsigned long long w = (unsigned long long) diag_wall();
    d[DG_TAG] = tag;
    d[DG_SEEN_HI] = seen_hi;
    d[DG_SEEN_LO] = seen_lo;
    d[DG_CYC_LO] = (unsigned) waited;
    d[DG_CYC_HI] = (unsigned) ((unsigned long long) waited >> 32);
    d[DG_WALL_LO] = (unsigned) w;
    d[DG_WALL_HI] = (unsigned) (w >> 32);
}

// one value for the whole wave (the first running lane's): a value that every lane loads from the same address. The compiler takes a load as a
// per-lane value; what is decided on it would turn the wait's clock into a per-lane value too, and that costs vector instructions in every poll
__device__ __forceinline__ unsigned diag_uniform(unsigned x) {
#if FNX_HIP
    return __builtin_amdgcn_readfirstlane(x);
#else
    return x;
#endif
}

// A wait that began at `ts` is `dt` cycles past its start t0 (ts, or later after an extension), over its `timeout`. true: wait on (t0 was moved);
// false: give up. See kernels.hpp WAIT_EVICT_WINDOW_MS for the rule. All of the wait's state is in t0 - ts, what the wait was extended by: its bit 0
// says that the short extension is used. Kept small: it is inlined into every wait's time-out branch, and it runs only there.
__device__ __forceinline__ bool diag_wait_on(unsigned* err, long long dt, long long timeout, long long ts, long long& t0) {
    const unsigned lo = diag_uniform(*(const volatile unsigned*) (err + DG_WATCH_LO)), hi = diag_uniform(*(const volatile unsigned*) (err + DG_WATCH_HI));
    const long long ext = t0 - ts;
    if ((lo | hi) == 0u || ext >= WAIT_EXT_CAP) return false;   // no watch (as before the watch), or extended enough
    const unsigned long long w = *(const volatile unsigned long long*) (((unsigned long long) hi << 32) | lo);   // host memory, one aligned word
    // ms from the last rise of evicted_ms to the host's last sample, against about the time-out in ms (2^21 cycles: 0.6 to 1 ms; a minute at most)
    // plus the window. "No rise" is an age far above any of this
    const unsigned long long t_ms = (unsigned long long) timeout >> 21;
    if (diag_uniform((unsigned) (w >> 32)) - diag_uniform((unsigned) w) <= (unsigned) (t_ms < 60000ull ? t_ms : 60000ull) + WAIT_EVICT_WINDOW_MS) {
        atomicAdd(err + DG_EXT_N, 1u);
        t0 = ts + ((ext + dt) & ~1LL);   // the driver had the queues off the cards: begin again (and a later time-out can get the short extension again)
        return true;
    }
    if (ext & 1LL) return false;   // no rise, and the short extension is used
    t0 = ts + ((ext + dt - timeout + WAIT_GRACE_CYCLES) | 1LL);   // the time-out comes again WAIT_GRACE_CYCLES from now, then the word is read again
    return true;
}

// a decode ring sum's start: the tag this card has reached, kept until the card has a ring time-out of its own (one thread of the launch: a load, a store)
__device__ __forceinline__ void diag_prog(unsigned* err, unsigned tag) {
    if ((blockIdx.x | threadIdx.x) == 0 && !(*(volatile const unsigned*) err & 1u)) *(volatile unsigned*) (err + DG_PROG) = tag;
}

}  // namespace fnx
