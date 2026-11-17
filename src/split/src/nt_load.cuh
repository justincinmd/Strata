// the Infinity Cache (MALL, 64 MB on an R9700) policy of the weight streams. FNX_MALL_NT (build option STRATA_SPLIT_FAST):
//   0  every load as it was (__ldg / plain)
//   1  the routed experts' weights with non-temporal loads (TH_LOAD_NT): a window reads ~25 MB of them a layer and nothing reads them again, so they
//      should not take the cache from what does get read again or was staged (k_mall_pf)
//   2  also the dense read-once weight streams (Q8 projections, the hc read's matrices, the router, the indexer's projections, the shared expert)
// Activations and small tables keep plain loads. Measured on gfx1201: a 256 MB plain stream evicts a warm 32 MB range,
// a non-temporal one leaves it whole, at the same GB/s. HIP only; elsewhere the macros are the loads they replace.
#pragma once

#ifndef FNX_MALL_NT
#define FNX_MALL_NT 0
#endif

#if FNX_MALL_NT && FNX_HIP
namespace fnx_nt {
typedef unsigned v4 __attribute__((ext_vector_type(4)));
typedef v4 v4_u __attribute__((aligned(1)));
typedef unsigned long long u64_u __attribute__((aligned(1)));
typedef unsigned u32_u __attribute__((aligned(1)));
typedef unsigned short u16_u __attribute__((aligned(1)));
// *p with a non-temporal load of T's size (1, 2, 4, 8 or 16 bytes), at any alignment
template <typename T> __device__ __forceinline__ T load(const T* p) {
    T r;
    if constexpr (sizeof(T) == 16) {
        const v4 v = __builtin_nontemporal_load(reinterpret_cast<const v4_u*>(p));
        __builtin_memcpy(&r, &v, 16);
    } else if constexpr (sizeof(T) == 8) {
        const unsigned long long v = __builtin_nontemporal_load(reinterpret_cast<const u64_u*>(p));
        __builtin_memcpy(&r, &v, 8);
    } else if constexpr (sizeof(T) == 4) {
        const unsigned v = __builtin_nontemporal_load(reinterpret_cast<const u32_u*>(p));
        __builtin_memcpy(&r, &v, 4);
    } else if constexpr (sizeof(T) == 2) {
        const unsigned short v = __builtin_nontemporal_load(reinterpret_cast<const u16_u*>(p));
        __builtin_memcpy(&r, &v, 2);
    } else {
        static_assert(sizeof(T) == 1, "nt load: 1, 2, 4, 8 or 16 bytes");
        const unsigned char v = __builtin_nontemporal_load(reinterpret_cast<const unsigned char*>(p));
        __builtin_memcpy(&r, &v, 1);
    }
    return r;
}
// NB bytes at src into dst: a 2-byte head when NB % 4 is 2, then 4 bytes at a time (the compiler joins neighbours into wider loads). With dst + head a
// multiple of 4 in a local array, every later 4-byte read of a block whose words start after a 2-byte header is one whole slice of the array (the copy
// stays in registers; with the words astride the slices it went to the stack)
template <int NB> __device__ __forceinline__ void copy(char* dst, const char* src) {
    static_assert(NB % 2 == 0, "nt copy: an even size");
    constexpr int H = NB % 4;
    if constexpr (H) {
        const unsigned short v = __builtin_nontemporal_load(reinterpret_cast<const u16_u*>(src));
        __builtin_memcpy(dst, &v, 2);
    }
#pragma unroll
    for (int k = H; k < NB; k += 4) {
        const unsigned v = __builtin_nontemporal_load(reinterpret_cast<const u32_u*>(src + k));
        __builtin_memcpy(dst + k, &v, 4);
    }
}
}  // namespace fnx_nt
#define FNX_WLD1(p) fnx_nt::load(p)
#else
#define FNX_WLD1(p) __ldg(p)
#endif

// FNX_WLD2P: for a site that was a plain load (not __ldg), so that level 0 and 1 leave it exactly as it was
#if FNX_MALL_NT >= 2 && FNX_HIP
#define FNX_WLD2(p) fnx_nt::load(p)
#define FNX_WLD2P(p) fnx_nt::load(p)
#else
#define FNX_WLD2(p) __ldg(p)
#define FNX_WLD2P(p) (*(p))
#endif
