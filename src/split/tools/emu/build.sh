#!/bin/bash
# tools/emu/build.sh: builds the host emulation test of the FNX_ATT_MM kernels (attn_mm_emu.cpp) into $1 (default /tmp/attn_mm_emu). It takes the old kernels
# (k_attn_part, k_attn_combine, k_idx_score, k_idx_select) from the sources by their first lines, so the test runs what the engine runs.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); src=$here/../../src; out=${1:-/tmp/attn_mm_emu}
gen=$(mktemp -d)
{
  awk '/^__device__ __forceinline__ float wsum/{p=1} p{print} /^__device__ __forceinline__ float sigm/{p=0}' "$src/kernels.cu"
  awk '/^constexpr int QSA_POOL/{p=1} p{print} /^constexpr int QSA_MAXSEL/{p=0}' "$src/qsa.cuh"
  awk '/^\/\/ scores\[t\]\[j\] for the visible blocks/{p=1} /^\/\/ the same for a prompt.s rows/{p=0} p{print}' "$src/qsa.cuh"
  awk '/^constexpr int QSA_SEL_THREADS/{p=1} /^}  \/\/ namespace/{p=0} p{print}' "$src/qsa.cuh"
  awk '/^#ifndef FNX_ATT_SPLITS_H/{p=1} /^\/\/ this card.s splits of head h/{p=0} p{print}' "$src/kernels.cu"
  awk '/^__global__ void __launch_bounds__\(HD\) k_attn_combine\(/{p=1} /^#include "attn_mm.cuh"/{p=0} p{print}' "$src/kernels.cu"
} > "$gen/emu_gen.inc"
${CXX:-/opt/rocm/llvm/bin/clang++} -std=c++17 -O1 -g -Wno-unknown-pragmas -Wno-duplicate-decl-specifier -Wno-unused-function \
  -DFNX_NCARD=${NCARD:-4} -DFNX_27B=0 -DFNX_ATT_HEADS=1 -DFNX_ATT_MM=1 -DFNX_HIP=0 -I"$here" -I"$gen" -I"$src" "$here/attn_mm_emu.cpp" -o "$out"
rm -rf "$gen"
echo "built $out"
