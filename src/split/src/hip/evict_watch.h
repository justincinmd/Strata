// the address of the host's eviction watch word for the HIP memory-operation kernels (hip/cuda.h fnx_memop_wait), set once by the
// engine when its eviction watch runs (engine.cu diag_load, engine_common.cu EvictWatch); null: off. kernels.hpp WAIT_EVICT_WINDOW_MS has the rule.
#pragma once
#include <atomic>
inline std::atomic<const unsigned long long*> fnx_evict_watch{nullptr};
