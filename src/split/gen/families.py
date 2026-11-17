"""The generator's families (tkgen.py list). A family = one launch site's kernel, with its parameters as a grid.

hip families take their kernel from the engine's sources: a parameter is a template argument or a -D constant whose default (in the .cuh, under
#ifndef) is today's value, so the all-default variant `base` is the kernel in the binary, bit for bit. Add a family when a step touches its kernel:
  1. the kernel's constants that are worth a sweep become `#ifndef FNX_TK_<X>` defaults in its .cuh
  2. the family here: parameters, keys, K(...) per key (name, template arguments, threads, shared bytes, meta), benches
  3. the launch site: `static const tk::Family* f = tk::family("<name>")`, FNX_TK_LAUNCH when f is set (src/tk.hpp)
"""
import hashlib
import re

from tkgen import Bench, K, P, family

MAXT = 16


# ------------------------------------------------------------------ bench output parsers (tools/kbench.cu's lines)
def _digest(lines):
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:12] if lines else None


def parse_hc(out):
    m = {}
    for T, alone, win, err in re.findall(r"hc_read T=\s*(\d+) stub\s+0\s+alone\s+([\d.]+) us\s+in a window\s+([\d.]+) us.*?err (\d+)", out):
        m[f"T{T} alone us"] = float(alone)
        m[f"T{T} window us"] = float(win)
        if err != "0":
            m["err"] = f"err {err}"
    return m


def parse_hchash(out):
    lines = re.findall(r"^hchash .*$", out, re.M)
    m = {"hashes": len(lines), "hash": _digest(lines)}
    if any(not re.search(r"err 0 ", ln) for ln in lines):
        m["err"] = "err"
    return m


def parse_chain(out):
    m = {}
    hs = []
    for T, us, err, h in re.findall(r"chain T=\s*(\d+)\s+([\d.]+) us a layer.*?err (\d+)\s+hash ([0-9a-f]+)", out):
        m[f"T{T} us a layer"] = float(us)
        hs.append(f"{T} {h} {err}")
    m["hash"] = _digest(hs)
    return m


def parse_moegu(out):
    m = {}
    hs = []
    for T, us, h in re.findall(r"moegu T=\s*(\d+)\s+([\d.]+) us a launch.*?hash ([0-9a-f]+)", out):
        m[f"T{T} us"] = float(us)
        hs.append(f"{T} {h}")
    m["hash"] = _digest(hs)
    return m


def parse_pfattn(out):
    m = {}
    hs = []
    for depth, us, h in re.findall(r"pfattn depth\s*(\d+)\s+([\d.]+) us a launch.*?hash ([0-9a-f]+)", out):
        m[f"{int(depth) // 1000}K us"] = float(us)
        hs.append(f"{depth} {h}")
    m["hash"] = _digest(hs)
    return m


KB = ["{kbench}", "{dev}"]

# ------------------------------------------------------------------ the target's cfg alone (tkgen build uses it when a directory holds only triton variants)
family("tk_probe", "hip", "none", "no kernel: the target's FNX_TK_CFG", params={}, keys=[], kernels={})

# ------------------------------------------------------------------ prefill
# the prompt attention's union walk (attn_prompt.cuh). Key 0 the walk, key 1 the builder of each group's position list (the same R).
family(
    "pf_attn_u", "hip", "prefill",
    "prompt attention: a group of R rows walks the union of its selected positions (k_pf_attn_u + k_pf_union)",
    params=dict(
        R=P("r", [2, 4, 8], 4),          # rows a group
        REG=P("reg", [0, 1], 1),         # O in accumulator tiles
        PC=P("pc", [32, 64], 32),        # positions a tile
        LEAN=P("lean", [0, 1], 1),       # < 32 KB shared memory, 4 blocks a WGP
        GW=P("gw", [0, 1, 2], 2),        # the gather: 0 four 8-byte loads of four positions a thread; 1 one (position, group) a thread, 16-byte loads; 2 + V at K's places. Same bits
        AT=P("at", [0, 1, 2, 4, 8, 12], 0),   # attribution (results not valid): 1 no K/V dequant, 2 no K/V loads, 4 no score products, 8 no P V products
    ),
    valid=lambda p: (p["REG"] or (p["PC"] == 32 and not p["LEAN"])) and not (p["LEAN"] and p["PC"] == 64) and (p["GW"] == 0 or (p["PC"] == 32 and not p["AT"] & 3)) and not (p["AT"] & 8 and not p["REG"]),
    keys=[0, 1], coupled=True,
    kernels={
        0: K("k_pf_attn_u", [("int", "{R}"), ("bool", "{REG}"), ("int", "{PC}"), ("bool", "{LEAN}"), ("int", "{AT}"), ("int", "{GW}")], threads="AP_THREADS",
             smem="AuShape<{R}, {REG}, {PC}, {LEAN}>::SMEM", meta=["{R}", "{PC}", "{REG}", "{LEAN}"]),
        1: K("k_pf_union", [("int", "{R}")], threads="256", smem="0", meta=["{R}"]),
    },
    presets={"r4-plain": dict(REG=0, LEAN=0, GW=0), "reg": dict(LEAN=0, GW=0), "pc64": dict(PC=64, LEAN=0, GW=0), "r8": dict(R=8, REG=0, LEAN=0, GW=0)},   # (the built-in's other layouts)
    bench={
        "pfattn": Bench(KB + ["1", "pfattn"], parse_pfattn, doc="the walk alone on one card at KBENCH_CTX depths (us a launch of a 4096-row chunk, hash)"),
    },
)

# ------------------------------------------------------------------ decode
# the hyper-connection read as a ticket queue (hc_mm.cuh): a kernel per window row count
family(
    "hc_mm", "hip", "decode",
    "decode hc read, ticket queue on int8 WMMA (k_hc_mm<T>); UD = a down wave's units in flight, POLL = the gate's sleep",
    params=dict(
        UD=P("ud", [2, 4, 8, 16], 8),
        POLL=P("poll", [16, 64, 256], 64),
        NT=P("nt", [None, 0, 1, 2], None),   # FNX_MALL_NT of this kernel alone (None: the target's): its weight loads' cache hint
    ),
    defines={"FNX_TK_HM_UD": "{UD}", "FNX_TK_HM_POLL": "{POLL}", "FNX_MALL_NT": "{NT}"},
    keys=list(range(1, MAXT + 1)),
    kernels=lambda key: K("k_hc_mm", [("int", "{key}")], threads="HM_T", smem="0", meta=["HM_UD"]),
    presets={"ud4": dict(UD=4), "ud2": dict(UD=2), "nt0": dict(NT=0), "poll16": dict(POLL=16), "poll256": dict(POLL=256)},   # (UD=16 spills at 4..12 rows)
    bench={
        "hc": Bench(KB + ["{rows}", "hc"], parse_hc, doc="us a read alone and in a window (--arg rows=1,4,8)"),
        "hchash": Bench(KB + ["1", "hchash"], parse_hchash, doc="every shape and row count: the reference hashes"),
        "chain": Bench(KB + ["{rows}", "chain"], parse_chain, doc="the decode chain's us a layer and its hash (--arg rows=1,4,8)"),
    },
)

# the routed experts' gate|up on Q4_K weights (moe_q4k.cuh): a kernel per window row count. The bus-bound kernel of a decode round.
family(
    "moe_gu_q4k", "hip", "decode",
    "decode routed experts gate|up, Q4_K (k_moe_gu_q4k<T>); LPR = lanes a row, PASS = load passes of a lane's groups (0: a group at a time), NT",
    params=dict(
        LPR=P("lpr", [2, 4, 8], 8),
        PASS=P("pass", [0, 1, 2], 2),
        NT=P("nt", [None, 0, 1, 2], None),   # FNX_MALL_NT: 0 plain loads, 1 the experts' loads non-temporal, 2 also the dense weights'
    ),
    defines={"FNX_TK_Q4K_LPR": "{LPR}", "FNX_TK_Q4K_PASSES": "{PASS}", "FNX_MALL_NT": "{NT}"},
    keys=list(range(1, MAXT + 1)),
    kernels=lambda key: K("k_moe_gu_q4k", [("int", "{key}")], threads="Q4K_THREADS", smem="0", meta=["Q4K_ROWS", "Q4K_LPR"]),
    presets={"lpr4": dict(LPR=4), "pass1": dict(PASS=1), "pass0": dict(PASS=0), "nt0": dict(NT=0)},
    bench={
        "moegu": Bench(KB + ["{rows}", "moegu"], parse_moegu, doc="the gate|up launch alone, every launch its own experts: us a launch, hash (--arg rows=1,4,8)"),
    },
)


# ------------------------------------------------------------------ both paths: the dense 8-bit GEMM, Triton
# c[m, n] = sa[m] * sw[n] * sum_k a[m, k] * w[n, k]: int8 (exact) or fp8 products on WMMA, the scales in the epilogue. A key = a site's (N, K) as
# constants (0: any N, K at run time). The parameters are Triton's tile sizes, waves and pipeline stages: the tuner's grid.
def parse_tkgemm(out):
    m = {}
    hs = []
    for key, name, ms, tops, err, h in re.findall(r"tkgemm key\s*(\d+) (\S+)\s.*?([\d.]+) ms\s+([\d.]+) T\S*OPS\s+max rel err ([\d.e+-]+)\s+hash ([0-9a-f]+)", out):
        m[f"{name} TOPS"] = float(tops)
        hs.append(f"{key} {h}")
        if float(err) > 2e-3:
            m["err"] = f"{name} rel err {err}"
    m["hash"] = _digest(hs)
    return m


GEMM_SHAPES = {
    0: None,
    1: dict(N=4096, K=2560, name="qkvz"),
    2: dict(N=3072, K=2560, name="wq"),
    3: dict(N=256, K=2560, name="wkv"),
    4: dict(N=2560, K=1536, name="out"),
    5: dict(N=320, K=10240, name="hcdn"),
    6: dict(N=10240, K=320, name="hcup"),
    7: dict(N=320, K=2560, name="shgu"),
    8: dict(N=2560, K=160, name="shdn"),
}
family(
    "gemm8", "triton", "both",
    "dense 8-bit GEMM with a scale a token and a scale a weight row in the epilogue (int8 exact or fp8), a kernel per site shape",
    params=dict(
        BM=P("m", [32, 64, 128, 256], 128),
        BN=P("n", [32, 64, 128, 256], 128),
        BK=P("k", [32, 64, 128], 64),
        FP8=P("fp", [0, 1], 0),
        OUT16=P("h", [0, 1], 0),
        MM=P("mm", [0, 1], 1),           # mask the rows past M (0: M a multiple of BM)
        WT=P("wt", [0, 1], 0),           # the weights' layout: 0 [N][K] (the files'), 1 [K][N] (made at load)
        DA=P("da", [0, 1], 1),           # the accumulator as the dot's third operand (0: acc += dot)
        EP=P("ep", [0, 1], 1),           # the scales in the epilogue (0: the raw sums; for the epilogue's cost only)
        NM=P("nm", [0, 1], 1),           # mask the columns past N (0: N a multiple of BN)
        SN=P("sn", [0, 1], 1),           # the site's N as a constant of the kernel (0: read at run time)
        SK=P("sk", [0, 1], 1),           # the site's K as a constant (0: read at run time)
        AT=P("at", [0, 1], 1),           # the JIT's pointer facts for the backend: 16-byte aligned pointers, operands under 2^31 bytes (buffer loads); 0: none
        W=P("w", [4, 8], 8),
        ST=P("st", [1, 2, 3], 2),
    ),
    valid=lambda p: p["BM"] * p["BN"] <= 256 * 128,   # (a configuration that spills registers is refused at its build)
    keys=GEMM_SHAPES,
    triton="gemm8", builtin=False,
    # (card 3, 2026-11-16, int8 at 4096 rows, 2000 launches = at the 210 W cap, against the f16 BLAS product of the same shape in the same run
    # (kbench KBENCH_GEMM_REF): AT=1 is the step: qkvz 141 TOPS = 1.48x, wq 143 = 1.61x, the row mask and run-time N / K then cost 1-3 %
    # (AT=0: 104 = 1.09x, and the row mask halved the rate). BN=256 is +6 % on qkvz / wq and worse on N <= 320; OUT16 is +9 % (n256-h1: 163 TOPS =
    # 1.71x / 1.85x); fp8 = int8; [K][N] weights -11 %; BK=32 -18 %; BM=256 and W=4 spill. A 20-launch burst reads 207-226 TOPS: not the engine's rate)
    presets={"n256": dict(BN=256), "n256-h": dict(BN=256, OUT16=1), "h": dict(OUT16=1), "fp8": dict(FP8=1), "k128": dict(BK=128), "m64": dict(BM=64),
             "mm0": dict(MM=0), "at0": dict(AT=0)},
    bench={
        "tkgemm": Bench(KB + ["1", "tkgemm"], parse_tkgemm, doc="each key's GEMM at KBENCH_GEMM_M rows (4096): ms, T(FL)OPS, error against the exact sum, hash"),
    },
)
