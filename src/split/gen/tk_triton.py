#!/usr/bin/env python3
"""tk_triton (generator): the Triton backend's worker. tkgen.py runs it in Triton's own python:

  tk_triton.py --kernel NAME --params JSON --keys JSON --out DIR/variant.hsaco
      -> one code object per key (DIR/variant.k<key>.hsaco), and on the last stdout line the keys' manifest entries as JSON

A kernel is compiled for gfx1201 ahead of time, with no card and no torch: triton.compile on the function, its argument types and its constants.
The C++ side (src/tk.hpp) launches it with the function's run-time arguments in order, then Triton's two hidden pointers (null): KERNELS[..]["args"]
is that list. A configuration that spills registers to scratch memory is refused (it is slow: 256 x 128 tiles on 8-bit GEMMs).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tkgen import NMETA, Elf, digest  # noqa: E402  (stdlib only)

ARCH = "gfx1201"


# ------------------------------------------------------------------ dense 8-bit GEMM, scales in the epilogue
# c[m, n] = sa[m] * sw[n] * sum_k a[m, k] * w[n, k]      a [M][K] (a row = a token), w [N][K] (a row = an output), int8 (exact integer sum) or
# fp8 e4m3; sa a scale a token, sw a scale a weight row; c f32 or f16 [M][N].
# The caller's contract (the launch site checks it; tools/kbench.cu tkgemm does): K is a multiple of BK (no K mask: a tile's loads would run past the
# row); M a multiple of BM when MM = 0; N a multiple of BN when NM = 0; M * K, N * K and M * N under 2^31 elements with a tile of slack (the
# element offsets are 32-bit; this model's largest operand is 8192 x 10240). Grid: ceil(M / BM) * ceil(N / BN) blocks.
# NC, KC: the site's N and K as constants (0: the run-time N, K). MM: mask the rows past M (0: M is a multiple of BM, or the buffers are padded).
# WT: the weights' layout: 0 = [N][K] as the files hold them (a tile's loads then stride over rows), 1 = [K][N] (made at load: a tile's N run is
# contiguous, as a BLAS B operand).
# AT (gemm8_spec, not a constant of the function): 1 = the facts Triton's JIT gives its backend for torch tensors, which an ahead-of-time build
# lacks: every pointer is a multiple of 16 bytes and every operand's bytes are under 2^31 (the AMD backend then uses buffer loads and stores:
# 32-bit offsets, masks without a branch), and M, N, K are multiples of 16 where the contract says so. The caller's contract with AT = 1: the five
# pointers are 16-byte aligned and M * K, N * K, M (+ BM) * N (+ BN) * the output's element size are under 2^31 bytes.
@triton.jit
def gemm8(a, w, sa, sw, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, FP8: tl.constexpr, OUT16: tl.constexpr,
          NC: tl.constexpr, KC: tl.constexpr, MM: tl.constexpr, WT: tl.constexpr, DA: tl.constexpr, EP: tl.constexpr, NM: tl.constexpr):
    # DA: 1 the accumulator as the dot's third operand, 0 acc += dot. EP: 1 the scales in the epilogue, 0 none (the raw sums: the epilogue's cost)
    # NM: 1 mask the columns past N when N is not a known multiple of BN; 0 never (the caller's N is a multiple of BN)
    full_n = (NC and NC % BN == 0) or not NM   # no column of a tile is past N: no column masks
    n_all = NC if NC else N
    k_all = KC if KC else K
    pid = tl.program_id(0)
    nb = tl.cdiv(n_all, BN)
    pm = pid // nb
    pn = pid % nb
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mn = rn < n_all
    if FP8:
        acc = tl.zeros((BM, BN), dtype=tl.float32)
    else:
        acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k in range(0, k_all, BK):
        if MM:
            x = tl.load(a + rm[:, None] * k_all + (k + rk)[None, :], mask=(rm < M)[:, None], other=0)
        else:
            x = tl.load(a + rm[:, None] * k_all + (k + rk)[None, :])
        if WT:
            wp = w + (k + rk)[:, None] * n_all + rn[None, :]
        else:
            wp = w + rn[None, :] * k_all + (k + rk)[:, None]
        if full_n:
            y = tl.load(wp)
        else:
            y = tl.load(wp, mask=mn[None, :], other=0)
        if FP8:   # the operands are loaded as bytes (a masked byte is 0 = +0.0 in e4m3) and reread as fp8
            x = x.to(tl.float8e4nv, bitcast=True)
            y = y.to(tl.float8e4nv, bitcast=True)
        if DA:
            acc = tl.dot(x, y, acc, out_dtype=tl.float32 if FP8 else tl.int32)
        else:
            acc += tl.dot(x, y, out_dtype=tl.float32 if FP8 else tl.int32)
    out = acc.to(tl.float32)
    if EP:
        if MM:
            s1 = tl.load(sa + rm, mask=rm < M, other=0.0)
        else:
            s1 = tl.load(sa + rm)
        if full_n:
            s2 = tl.load(sw + rn)
        else:
            s2 = tl.load(sw + rn, mask=mn, other=0.0)
        out = out * (s1[:, None] * s2[None, :])
    if OUT16:
        out = out.to(tl.float16)
    cp = c + rm[:, None] * n_all + rn[None, :]
    if MM and full_n:
        tl.store(cp, out, mask=(rm < M)[:, None])
    elif MM:
        tl.store(cp, out, mask=(rm < M)[:, None] & mn[None, :])
    elif full_n:
        tl.store(cp, out)
    else:
        tl.store(cp, out, mask=mn[None, :])


def gemm8_spec(pv, shape):
    a8 = "*u8" if pv["FP8"] else "*i8"
    nc, kc = (shape or {}).get("N", 0), (shape or {}).get("K", 0)
    if kc and kc % pv["BK"]:
        raise SystemExit(f"gemm8: K {kc} is not a multiple of BK {pv['BK']}")
    if nc and nc % pv["BN"] and not pv["NM"]:
        raise SystemExit(f"gemm8: N {nc} is not a multiple of BN {pv['BN']} (NM=0)")
    attrs = None
    if pv.get("AT", 0):
        ptr = [["tt.divisibility", 16], ["tt.pointer_range", 32]]
        d16 = [["tt.divisibility", 16]]
        attrs = {(i,): ptr for i in range(5)}   # a w sa sw c
        if not pv["MM"]:
            attrs[(5,)] = d16   # M: a multiple of BM
        if not pv["NM"] or (nc and nc % 16 == 0):
            attrs[(6,)] = d16   # N: a multiple of BN, or the site's constant
        attrs[(7,)] = d16       # K: a multiple of BK
    return dict(
        attrs=attrs,
        signature=dict(a=a8, w=a8, sa="*fp32", sw="*fp32", c="*fp16" if pv["OUT16"] else "*fp32", M="i32", N="i32", K="i32"),
        constexprs=dict(BM=pv["BM"], BN=pv["BN"], BK=pv["BK"], FP8=pv["FP8"], OUT16=pv["OUT16"], NC=nc if pv["SN"] else 0, KC=kc if pv["SK"] else 0, MM=pv["MM"], WT=pv["WT"], DA=pv["DA"], EP=pv["EP"], NM=pv["NM"]),
        options=dict(num_warps=pv["W"], num_stages=pv["ST"]),
        meta=[pv["BM"], pv["BN"], pv["BK"], pv["FP8"], pv["OUT16"], nc, kc,
              pv["MM"] | pv["WT"] << 1 | pv["EP"] << 2 | pv["NM"] << 3 | (1 if pv.get("AT", 0) else 0) << 4],   # meta 7: bit 0 MM, bit 1 WT, bit 2 EP, bit 3 NM, bit 4 AT
    )


KERNELS = {
    "gemm8": dict(fn=gemm8, spec=gemm8_spec, args="a w sa sw c M N K"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--keys", required=True, help="JSON: {key: shape or null}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kern = KERNELS[a.kernel]
    pv, keys = json.loads(a.params), json.loads(a.keys)
    out = Path(a.out)
    entries = {}
    skipped = {}
    for key, shape in keys.items():
        try:
            sp = kern["spec"](pv, shape)
        except SystemExit as e:   # this key's shape does not take the configuration: the variant has no kernel for it
            skipped[key] = str(e)
            continue
        src = ASTSource(fn=kern["fn"], signature=sp["signature"], constexprs=sp["constexprs"], attrs=sp.get("attrs"))
        ck = triton.compile(src, target=GPUTarget("hip", ARCH, 32), options=sp["options"])
        asm = ck.asm["amdgcn"]
        spill = max([int(x) for x in re.findall(r"\.vgpr_spill_count:\s*(\d+)", asm)] + [0])
        scratch = max([int(x) for x in re.findall(r"\.private_segment_fixed_size:\s*(\d+)", asm)] + [0])
        if spill or scratch:
            skipped[key] = f"{a.kernel} key {key}: {spill} spilled registers, {scratch} scratch bytes a lane (refused: use smaller tiles)"
            continue
        vgpr = max([int(x) for x in re.findall(r"\.vgpr_count:\s*(\d+)", asm)] + [0])
        blob = ck.asm["hsaco"]
        f = out.with_name(out.stem + f".k{key}.hsaco")
        f.write_bytes(blob)
        elf = Elf(blob)
        name = ck.metadata.name
        entries[key] = dict(symbol=name, threads=ck.metadata.num_warps * 32, smem=ck.metadata.shared, args=elf.kernels()[name]["args"],
                            meta=(sp["meta"] + [0] * NMETA)[:NMETA], file=f"{out.parent.name}/{f.name}", vgpr=vgpr,
                            digest=digest(blob))   # (of the bytes the symbol and the argument sizes were read from: tkgen keeps it as the file's record)
    if not entries:
        raise SystemExit("no key takes this configuration: " + "; ".join(skipped.values()))
    print(json.dumps(dict(keys=entries, triton=triton.__version__, args=kern["args"], skipped=skipped)))


if __name__ == "__main__":
    main()
