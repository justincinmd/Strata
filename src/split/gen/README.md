# gen: the split engine's kernel generator

A launch site's kernel is a **family**. One setting of the family's parameters is a **variant**. `tkgen.py` compiles variants ahead of time
into code objects and writes a manifest. The engine loads a variant by name at run time (`src/tk.hpp`), so an experiment needs no engine
rebuild. The all-default variant `base` gives the results of the kernel in the binary, bit for bit.

Two backends, one manifest and one loader:

- **hip**: the engine's own kernel source. A parameter is a template argument or a `-D` constant with today's value as its `#ifndef` default.
  tkgen compiles `src/kernels.cu` with the target's own compile line (read from the build's `build.ninja`), device code only.
- **triton**: a Triton function in `tk_triton.py`, compiled for gfx1201 with no card present. One code object a key.

## Use

```
B=<cmake build dir>; O=<output dir>
gen/tkgen.py list                                             # families, parameters, presets, benches
gen/tkgen.py build --build-dir $B --target strata-split-4 --out $O hc_mm pf_attn_u:all moe_gu_q4k:LPR=4,PASS=1
gen/tkgen.py build --build-dir $B --target strata-split-kbench-4 --out $O-kb --triton-python <python with triton> gemm8:all
gen/tkgen.py verify --out $O --binary <the target's binary>   # base against the binary's own kernels, no card (below)
gen/tkgen.py sweep --out $O-kb --family hc_mm --bench hc --kbench <kbench binary> --device 3 --arg rows=1,4,8
gen/tkgen.py compose --out $O --family hc_mm --name tuned 1-4=ud4 5-16=base     # a variant with each key's kernel from another one
gen/tkgen.py use --out $O hc_mm=tuned                         # the manifest's default (unset: the kernel in the binary)
gen/tkgen.py arms --out $O --family hc_mm --bin <suffix>      # the variants as one line each ("name suffix K=V ...") for a test queue
FNX_TK_DIR=$O FNX_TK_HC_MM=ud4 strata-split-4 ...             # run a variant (0 or builtin: the kernel in the binary)
```

A code object is made for one target: the engine and kbench differ (kbench's `HcArgs` is 192 bytes, the engine's 184), so each has its own
output directory. Four checks keep a wrong kernel from running:

- **Build stamp.** The manifest records the target's `FNX_TK_CFG` (`src/tk_cfg.hpp`). The loader refuses another build's manifest, and
  `tkgen build` refuses a directory made with other build settings before it changes a file.
- **Bytes.** The manifest records each code object's size and FNV-1a 64, taken at build time from the bytes the kernels' symbols and
  argument sizes were read from. tkgen never takes them again from the file on disk. The loader checks the bytes it reads against them, for
  every selected variant at the engine's start.
- **Arguments.** Every launch checks its argument count and byte sizes against the code object's kernel (the ELF's AMDGPU metadata, without
  the hidden arguments the runtime fills).
- **Launch path.** At the engine's start (`tk_start`), a selected variant whose launch path is off in that run is an error (`hc_mm` needs
  `FNX_MM_HC=1`; `pf_attn_u` needs the union walk on), and so is a selected family with no launch site in the binary.

A named variant that cannot be used is an error, never a silent fall back to the built-in kernel. `FNX_TK_<FAMILY>` set to an empty string
is an error too. The engine's log has a line `fnx: tk: <family> = <variant> (...)` for each selected variant at the start and a line
`fnx: tk: <family> = <variant> runs` at its first launch. One limit: `moe_gu_q4k` runs for the layers whose experts' gate|up weights are
Q4_K. A file with no such layer never launches it: its log has no `runs` line.

One tkgen at a time changes an output directory (a lock file). A directory made by an older tkgen (no record of the bytes in its entries)
is refused: build it again with `--force`.

`verify` compares a variant's kernels with the binary's kernels of the same names, with no card. It is a screen, not a proof. `base` and
the built-in kernel come from one source in two translation units, and the compiler schedules them in two ways. A kernel can be in several
of the binary's code objects; `verify` compares with every copy and the worst result counts:

- `same code`: the same instructions with the same immediates.
- `same ops`: the same count of each float operation and dot product; order, registers and integer address math differ.
- `OPS DIFFER` (exit code 1): the arithmetic compiled in two ways, for example a product and a sum in place of a multiply-add.

Run it for `base` after every build of a target. Equal result bits are proven on a card: the family's bench hash (kbench target) and the
engine's texts and deep-answer hash. A variant with another setting differs, as it must. A variant or family that is
asked for and cannot be compared is an error.

A kernel that spills registers to scratch memory is left out of its variant (`--keep-spills` keeps it). It is slow, and the kernels that
hold registers by name in inline assembly are wrong with spills (`hc_mm:UD=16` at 4 rows: an illegal memory access). `compose` builds a
variant from the keys that are good.

## Attribution before a build

To learn what a kernel's time is made of, give it an integer template parameter that leaves parts out (a variant somebody selects by name;
its results are not valid) and time the variants in the family's bench. `pf_attn_u` has `AT`: 1 no K/V dequant, 2 no K/V loads, 4 no score
products, 8 no P V products. It showed that the walk's loads (41 %) and not its dequant (6 %) or products (19 %) were the cost, and the
gather `GW` followed from that (walk -25 %, same bits). The parts overlap and the rest runs at another clock under the power cap, so read
the numbers as bounds.

## Add a family

1. Make the kernel's constants that are worth a sweep `#ifndef FNX_TK_<X>` defaults in its `.cuh` (today's values).
2. Add the family to `families.py`: parameters, keys, `K(...)` per key (kernel name, template arguments, threads, shared bytes, meta
   integers the launch site needs), presets, benches.
3. At the launch site: `static const tk::Family* f = tk::family("<name>");` and `FNX_TK_LAUNCH(f->kern(key), <built-in kernel>, grid, block,
   smem, stream, args...)` when `f` is set. The built-in kernel's name gives only the parameter types.
4. Give the constants `static_assert`s for the values the kernel takes: a setting outside them must fail to compile (tkgen prints REJECTED).
5. Check: the `base` variant gives the built-in's hash and speed in the family's bench.

A code object holds its own copy of every `__device__` global. Kernels that share one (today only `g_moe_hq` in `moe.cuh`: `k_moe_h` writes
it, `k_moe_down` reads it) must be in one family and one variant, or stay built in. A family that takes only one of them reads zeros.

A key is the kernel's slot in the family: the window's rows for a decode kernel, a site's shape for a GEMM, or a kernel number when a family
holds more than one kernel (`pf_attn_u`: 0 the walk, 1 the lists' builder). Such a family is `coupled=True`: its kernels share one layout,
so `compose` refuses to mix settings.

## Families (2026-11-15)

| Family | Path | Backend | Parameters | Bench (kbench case) |
|---|---|---|---|---|
| `pf_attn_u` | prefill | hip | R rows a group, REG, PC positions a tile, LEAN, GW the gather (2: a thread a (position, group), 16-byte loads, V at K's places), AT attribution | `pfattn` |
| `hc_mm` | decode | hip | UD units in flight, POLL, NT load hint | `hc`, `hchash`, `chain` |
| `moe_gu_q4k` | decode | hip | LPR lanes a row, PASS load passes, NT | `moegu` |
| `gemm8` | both | triton | tiles BM BN BK, FP8, OUT16, masks MM NM, layout WT, constants SN SK, pointer facts AT, warps W, stages ST | `tkgemm` |

`gemm8` in the engine: `FNX_TK_GEMM8=<variant>` sends the prompt path's dense products of the sites in `FNX_PF_TK_SITES` (a comma list of
`qkvz wq wkv ssmout wo shgu`, or `all`; default `qkvz`) through the 8-bit GEMM in place of the f16 BLAS product (prefill.cu `pf_dense_site`).
A site takes its key's kernel when the kernel fits the site's shape, else key 0; none is an error at the start. The engine makes the
operands (`k_pf_row8`): activations 8 bits with a scale a token, the Q8_0 matrix 8 bits with a scale a row (once a layer, in a cache),
int8 or fp8 by the variant. `FNX_PF_TK_ROT=1` rotates both operands first (a Hadamard transform of each 32 columns; the product is the
same, the rows lose their outliers). `FNX_PF_TK_CHECK=1` (slow) prints each product's error against the f16 product. The texts change:
it is an arm, not a default. Not selected: the f16 path, the same bits as without the family.

The caller of a `gemm8` kernel keeps a contract the kernel does not check: K a multiple of
BK; M a multiple of BM when MM=0; N a multiple of BN when NM=0; each operand (M x K, N x K, M x N) under 2^31 elements (32-bit offsets).
With `AT=1` (the default) also: the five pointers are 16-byte aligned and each operand is under 2^31 bytes. `AT=1` gives Triton's backend the
facts its JIT takes from a torch tensor (an ahead-of-time build has none): the AMD backend then uses buffer loads and stores. It is the
difference between 1.09x and 1.48x the f16 BLAS product on the qkvz shape at the power cap.

Measure a GEMM variant against the engine's f16 product in one run: `KBENCH_GEMM_REF=1 KBENCH_GEMM_RUNS=2000` (kbench `tkgemm`). 2000
launches hold the card at its power cap, as a long prompt read does; a 20-launch burst reads ~45 % higher for both and is not the engine's rate.
The reference is a timing reference only (host BLAS calls against a replayed graph: small shapes favor the graph a little; its operands are
not the scaled ones).

## Files

- `tkgen.py`: list, build, verify, compose, use, arms, show, sweep. Standard library only. Tests: `tests/split/test_split_tkgen.py`.
- `families.py`: the registry and the bench output parsers.
- `tk_triton.py`: the Triton kernels and their worker (run by tkgen in Triton's python).
- `../src/tk.hpp`, `../src/tk_cfg.hpp`: the loader and the build stamp.
- Output directory: `manifest.json` (all details), `manifest.tk` (what the engine reads), `<family>/<variant>.hsaco` + `.cu` + `.log`,
  `sweeps/*.md`.
