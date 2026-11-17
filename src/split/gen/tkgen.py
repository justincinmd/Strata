#!/usr/bin/env python3
"""tkgen: the split engine's kernel generator.

A FAMILY is one launch site's kernel. A VARIANT is one setting of the family's parameters. tkgen compiles variants ahead of time into code
objects and writes a manifest; the engine (src/tk.hpp) launches a family's variant by name at run time (FNX_TK_DIR, FNX_TK_<FAMILY>), so an
experiment needs no engine rebuild. Two backends, one manifest:
  hip     the engine's own kernel sources (src/*.cuh), compiled by the build's clang with the variant's template arguments and -D constants
  triton  a Triton function (gen/tk_triton.py), compiled by Triton to a code object for the same loader

  tkgen.py list                                         the families, their parameters and presets
  tkgen.py build  --build-dir B --target T --out DIR [SPEC ...]   compile variants into DIR and write DIR/manifest.{json,tk}
  tkgen.py verify --out DIR --binary BIN [family ...]   a variant's kernels (base) against the binary's own kernels: same code / same float ops
  tkgen.py use    --out DIR family=variant ...          the manifest's default variant of a family (family=0: the kernel in the binary)
  tkgen.py compose --out DIR --family F --name N KEYS=variant ...   a variant with each key's kernel from another variant (the tuner's picks)
  tkgen.py arms   --out DIR --family F --bin SUFFIX     the variants as lines for a test queue (one engine start a variant, no rebuild)
  tkgen.py show   --out DIR                             what the manifest holds
  tkgen.py sweep  --out DIR --family F --bench NAME ... run a bench once per variant (FNX_TK_<F>=<variant>) and print a table

SPEC: family (its presets), family:all (the whole grid), family:name (one preset or variant name), family:P=v,P=v (one setting).
Families: gen/families.py.
"""
import argparse
import concurrent.futures
import difflib
import fcntl
import hashlib
import itertools
import json
import os
import re
import shlex
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent   # src/split/gen
SPLIT = HERE.parent
SRC = SPLIT / "src"
NMETA = 8
BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"


def die(msg):
    sys.exit("tkgen: " + msg)


# ------------------------------------------------------------------ family specs (gen/families.py fills REG)
class P:
    """A parameter: its short code in variant names, the values of its grid, and the built-in's value (None: the target's own -D)."""

    def __init__(self, code, values, default):
        self.code, self.values, self.default = code, list(values), default


class K:
    """One kernel of a hip family: its template name, the template arguments as (kind, C++ expression) with kind int | bool | uint | enum:<type>
    | type, and the C++ expressions of its block threads, dynamic shared bytes and meta integers (formatted with the parameters and {key})."""

    def __init__(self, name, targs=(), threads="1", smem="0", meta=()):
        self.name, self.targs, self.threads, self.smem, self.meta = name, list(targs), threads, smem, list(meta)


class Bench:
    """A sweep's bench: the command (formatted with {kbench} {engine} {dev} and the sweep's --arg values) and parse(stdout) -> {metric: value}."""

    def __init__(self, cmd, parse, env=None, doc=""):
        self.cmd, self.parse, self.env, self.doc = cmd, parse, dict(env or {}), doc


class Family:
    def __init__(self, name, backend, path, doc, params, keys=(0,), kernels=None, defines=None, valid=None, presets=None, bench=None,
                 triton=None, builtin=True, coupled=False):
        self.name, self.backend, self.path, self.doc = name, backend, path, doc
        self.builtin = builtin   # False: the binary has no kernel of this family (a sweep has no built-in arm)
        self.coupled = coupled   # True: the keys are parts of one layout (a variant's kernels all come from one setting: no compose)
        self.params, self.keys, self.kernels = params, list(keys), kernels
        self.shapes = dict(keys) if isinstance(keys, dict) else {k: None for k in keys}   # a triton family: key -> its shape constants
        self.defines, self.valid, self.presets = dict(defines or {}), valid, dict(presets or {})
        self.bench, self.triton = dict(bench or {}), triton

    def default(self):
        return {n: p.default for n, p in self.params.items()}

    def vname(self, pv):
        parts = [f"{p.code}{pv[n]}" for n, p in self.params.items() if pv[n] != p.default]
        return "-".join(parts).replace(".", "p") or "base"

    def grid(self):
        names = list(self.params)
        for vals in itertools.product(*(self.params[n].values for n in names)):
            pv = dict(zip(names, vals))
            if self.valid is None or self.valid(pv):
                yield pv

    def kernel(self, key):
        return self.kernels(key) if callable(self.kernels) else self.kernels[key]


REG = {}


def family(*a, **kw):
    f = Family(*a, **kw)
    REG[f.name] = f
    return f


def load_families():
    if not REG:
        sys.modules.setdefault("tkgen", sys.modules[__name__])   # families.py imports this module by name
        sys.path.insert(0, str(HERE))
        import families  # noqa: F401  (registers into REG)
    return REG


def parse_value(fam, name, text):
    if name not in fam.params:
        die(f"{fam.name}: no parameter {name} (has: {', '.join(fam.params)})")
    for v in fam.params[name].values + [fam.params[name].default]:
        if str(v) == text:
            return v
    try:
        return int(text)   # a value outside the grid: allowed, the compiler decides
    except ValueError:
        return text


def expand_spec(spec):
    """SPEC -> (family, [parameter settings])."""
    fams = load_families()
    name, _, rest = spec.partition(":")
    if name not in fams:
        die(f"no family {name} (has: {', '.join(fams)})")
    fam = fams[name]
    if not rest:
        sets = [dict(fam.default(), **ov) for ov in fam.presets.values()] or [fam.default()]
        if fam.default() not in sets:
            sets.insert(0, fam.default())
        return fam, sets
    if rest == "all":
        return fam, list(fam.grid())
    if rest in fam.presets:
        return fam, [dict(fam.default(), **fam.presets[rest])]
    if rest == "base":
        return fam, [fam.default()]
    if "=" in rest:
        pv = fam.default()
        for kv in rest.split(","):
            k, _, v = kv.partition("=")
            pv[k] = parse_value(fam, k, v)
        return fam, [pv]
    for pv in fam.grid():   # a variant name
        if fam.vname(pv) == rest:
            return fam, [pv]
    die(f"{name}: no preset or grid variant named {rest}")


# ------------------------------------------------------------------ the target's compile line (the cmake build's build.ninja)
def target_flags(build_dir, target):
    nin = Path(build_dir) / "build.ninja"
    if not nin.exists():
        die(f"{nin}: not a cmake/ninja build directory")
    txt = nin.read_text()
    m = re.search(r"^build src/split/CMakeFiles/%s\.dir/kernels_p0\.cu\.o:[^\n]*\n((?:  [^\n]*\n)+)" % re.escape(target), txt, re.M)
    if not m:
        die(f"{nin}: no target {target} (configure it: its kernels_p0.cu.o rule is what tkgen copies)")
    v = dict(re.findall(r"^  (\w+) = (.*)$", m.group(1), re.M))
    flags = [f for f in shlex.split(v.get("FLAGS", "")) if f not in ("-Wall", "-Wextra")]
    cxx = "/opt/rocm/llvm/bin/clang++"
    rules = Path(build_dir) / "CMakeFiles" / "rules.ninja"
    if rules.exists():
        r = re.search(r"rule HIP_COMPILER__%s_\w+\n(?:  [^\n]*\n)*?  command = (?:\$\{LAUNCHER\}\$\{CODE_CHECK\})?(\S+)" % re.escape(target), rules.read_text())
        if r:
            cxx = r.group(1)
    return cxx, shlex.split(v.get("DEFINES", "")) + shlex.split(v.get("INCLUDES", "")) + flags


def src_stamp():
    h = hashlib.sha256()
    for p in sorted(SRC.rglob("*")):
        if p.suffix in (".cu", ".cuh", ".hpp", ".h", ".cpp"):
            h.update(p.name.encode())
            h.update(p.read_bytes())
    try:
        g = subprocess.run(["git", "-C", str(SPLIT), "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        g = ""
    return f"{g or 'nogit'}+{h.hexdigest()[:10]}"


# ------------------------------------------------------------------ code objects
def bundle_entries(buf, at):
    """A clang offload bundle's entries at buf[at:]: (target triple, offset in buf, size). Every extent is checked against the buffer."""
    try:
        (n,) = struct.unpack_from("<Q", buf, at + len(BUNDLE_MAGIC))
        off = at + len(BUNDLE_MAGIC) + 8
        if n > 64:
            die(f"an offload bundle with {n} entries")
        out = []
        for _ in range(n):
            o, sz, tl = struct.unpack_from("<QQQ", buf, off)
            off += 24
            if tl > 256 or off + tl > len(buf) or at + o + sz > len(buf):
                die("an offload bundle's entry runs past the file")
            out.append((buf[off:off + tl].decode(errors="replace"), at + o, sz))
            off += tl
    except struct.error:
        die("an offload bundle's header runs past the file")
    return out


def unbundle(buf, arch=None):
    """A clang offload bundle's GPU code object (the ELF as it is when buf is not a bundle). arch (gfx1201): the entry of that target when the
    bundle holds several."""
    if not buf.startswith(BUNDLE_MAGIC):
        return buf
    found = [(t, buf[o:o + sz]) for t, o, sz in bundle_entries(buf, 0) if "amdgcn" in t and sz]
    if arch:
        found = [f for f in found if f[0].endswith(arch)]
    if len(found) != 1:
        die(f"the bundle holds {len(found)} amdgcn code objects" + (f" for {arch}" if arch else "") + " (one wanted)")
    return found[0][1]


class Elf:
    def __init__(self, buf):
        if buf[:4] != b"\x7fELF" or buf[4] != 2:
            die("not a 64-bit ELF code object")
        self.buf = buf
        if len(buf) < 64:
            die("an ELF shorter than its header")
        shoff, = struct.unpack_from("<Q", buf, 0x28)
        shentsize, shnum, self.shstrndx = struct.unpack_from("<HHH", buf, 0x3A)
        if shentsize < 64 or shoff + shnum * shentsize > len(buf) or self.shstrndx >= max(shnum, 1):
            die("the ELF's section table runs past the file")
        self.sec = []
        for i in range(shnum):
            name, typ, _flags, addr, off, size, link, _info, _align, entsize = struct.unpack_from("<IIQQQQIIQQ", buf, shoff + i * shentsize)
            if typ != 8 and off + size > len(buf):   # (8 = SHT_NOBITS: no bytes in the file)
                die("an ELF section runs past the file")
            self.sec.append(dict(name=name, type=typ, addr=addr, off=off, size=size, link=link, entsize=entsize))
        self.syms = []   # dynsym, in index order: (name, value, size, type)
        for s in self.sec:
            if s["type"] == 11:   # SHT_DYNSYM
                if s["link"] >= shnum:
                    die("the ELF's symbol table names no string table")
                st = self.sec[s["link"]]
                for i in range(s["size"] // 24):
                    n, info, _other, _shndx, value, size = struct.unpack_from("<IBBHQQ", buf, s["off"] + i * 24)
                    self.syms.append((self.cstr(st, n), value, size, info & 15))
        self.by = {s[0]: s for s in self.syms}

    def cstr(self, table, n):
        """The string at offset n of a string table section (it ends inside the table)."""
        end = self.buf.find(b"\0", table["off"] + n, table["off"] + table["size"])
        if n >= table["size"] or end < 0:
            die("an ELF name runs past its string table")
        return self.buf[table["off"] + n:end].decode(errors="replace")

    def read(self, addr, n):
        for s in self.sec:
            if s["type"] != 8 and s["addr"] and s["addr"] <= addr and addr + n <= s["addr"] + s["size"]:
                o = s["off"] + addr - s["addr"]
                return self.buf[o:o + n]
        die(f"address {addr:#x} is in no section")

    def section(self, name):
        """A section's bytes by its name (None when the file has none)."""
        st = self.sec[self.shstrndx]
        for x in self.sec:
            if x["type"] != 8 and self.cstr(st, x["name"]) == name:
                return self.buf[x["off"]:x["off"] + x["size"]]
        return None

    def data(self, name):
        if name not in self.by:
            die(f"the code object has no symbol {name}")
        _, value, size, _ = self.by[name]
        return self.read(value, size)

    def kernels(self):
        """The code object's kernels from its AMDGPU metadata note: symbol -> dict(args = the byte size of each argument the launch passes, in
        order (the hidden ones the runtime fills are left out), vgpr, spill, scratch)."""
        for s in self.sec:
            if s["type"] != 7:   # SHT_NOTE
                continue
            off, end = s["off"], s["off"] + s["size"]
            while off + 12 <= end:
                namesz, descsz, typ = struct.unpack_from("<III", self.buf, off)
                name = self.buf[off + 12:off + 12 + namesz].rstrip(b"\0")
                d0 = off + 12 + (namesz + 3) // 4 * 4
                off = d0 + (descsz + 3) // 4 * 4
                if name == b"AMDGPU" and typ == 32:   # NT_AMDGPU_METADATA: msgpack
                    if d0 + descsz > len(self.buf):
                        die("the AMDGPU metadata note runs past the file")
                    md, used = msgpack(self.buf[d0:d0 + descsz], 0)
                    if not isinstance(md, dict) or not isinstance(md.get("amdhsa.kernels", []), list):
                        die("the AMDGPU metadata is not a map with a kernel list")
                    out = {}
                    for k in md.get("amdhsa.kernels", []):
                        args = sorted((a[".offset"], a[".size"]) for a in k.get(".args", []) if not str(a.get(".value_kind", "")).startswith("hidden_"))
                        out[k[".name"]] = dict(args=[sz for _, sz in args], vgpr=k.get(".vgpr_count", 0), spill=k.get(".vgpr_spill_count", 0),
                                               scratch=k.get(".private_segment_fixed_size", 0))
                    return out
        die("the code object has no AMDGPU metadata note")


def msgpack(b, i):
    """A value of a msgpack buffer at i -> (value, the next offset). The types the AMDGPU metadata uses. A value past the buffer is an error
    (struct.error from a short read is caught by the caller's die)."""
    if i >= len(b):
        die("msgpack: a value starts past the buffer")
    t = b[i]
    if t <= 0x7F:
        return t, i + 1
    if t >= 0xE0:
        return t - 256, i + 1
    if 0x80 <= t <= 0x8F or t in (0xDE, 0xDF):
        n, i = (t & 15, i + 1) if t <= 0x8F else (struct.unpack_from(">H", b, i + 1)[0], i + 3) if t == 0xDE else (struct.unpack_from(">I", b, i + 1)[0], i + 5)
        m = {}
        for _ in range(n):
            k, i = msgpack(b, i)
            m[k], i = msgpack(b, i)
        return m, i
    if 0x90 <= t <= 0x9F or t in (0xDC, 0xDD):
        n, i = (t & 15, i + 1) if t <= 0x9F else (struct.unpack_from(">H", b, i + 1)[0], i + 3) if t == 0xDC else (struct.unpack_from(">I", b, i + 1)[0], i + 5)
        v = []
        for _ in range(n):
            x, i = msgpack(b, i)
            v.append(x)
        return v, i
    if 0xA0 <= t <= 0xBF or t in (0xD9, 0xDA, 0xDB, 0xC4, 0xC5, 0xC6):
        if t <= 0xBF:
            n, i = t & 31, i + 1
        elif t in (0xD9, 0xC4):
            n, i = b[i + 1], i + 2
        elif t in (0xDA, 0xC5):
            n, i = struct.unpack_from(">H", b, i + 1)[0], i + 3
        else:
            n, i = struct.unpack_from(">I", b, i + 1)[0], i + 5
        if i + n > len(b):
            die("msgpack: a string runs past the buffer")
        raw = bytes(b[i:i + n])
        return (raw if t in (0xC4, 0xC5, 0xC6) else raw.decode()), i + n
    if t == 0xC0:
        return None, i + 1
    if t in (0xC2, 0xC3):
        return t == 0xC3, i + 1
    fmt = {0xCC: ">B", 0xCD: ">H", 0xCE: ">I", 0xCF: ">Q", 0xD0: ">b", 0xD1: ">h", 0xD2: ">i", 0xD3: ">q", 0xCA: ">f", 0xCB: ">d"}.get(t)
    if not fmt:
        die(f"msgpack: type {t:#x} is not handled")
    return struct.unpack_from(fmt, b, i + 1)[0], i + 1 + struct.calcsize(fmt)


def demangled(path, tools):
    """dynsym index -> demangled name (llvm-readelf -C)."""
    out = subprocess.run([str(tools / "llvm-readelf"), "--dyn-syms", "-W", "-C", str(path)], capture_output=True, text=True, check=True).stdout
    names = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\d+):\s+[0-9a-f]+\s+\d+\s+\w+\s+\w+\s+\w+\s+\S+\s+(.*)$", line)
        if m:
            names[int(m.group(1))] = m.group(2)
    return names


def render_targ(kind, v):
    if kind == "bool":
        return "true" if v else "false"
    if kind == "uint":
        return f"{v}u"
    if kind.startswith("enum:"):
        return f"({kind[5:]}){v}"
    return str(v)


# ------------------------------------------------------------------ the hip backend
def hip_tu(fam, pv):
    lines = [f"// generated by tkgen: family {fam.name}, variant {fam.vname(pv)}"]
    for macro, fmt in fam.defines.items():
        val = fmt.format(**pv) if isinstance(fmt, str) else fmt(pv)
        if val is None or val == "None":
            continue   # the target's own setting
        lines += [f"#undef {macro}", f"#define {macro} {val}"]
    lines += ["#define FNX_KPART 99   // no launchers: only the kernels named below are instantiated", "#define FNX_TK_GEN 1",
              f'#include "{SRC}/kernels.cu"', f'#include "{SRC}/tk_cfg.hpp"', "namespace fnx {", "namespace {", "[[maybe_unused]] void tk_ref() {"]
    rows = []
    for i, key in enumerate(fam.keys):
        k = fam.kernel(key)
        f = dict(pv, key=key)
        targs = [(kind, expr.format(**f)) for kind, expr in k.targs]
        ident = k.name + ("<" + ", ".join(e for _, e in targs) + ">" if targs else "")
        lines.append(f"    (void) &{ident};")
        ints = [e for kind, e in targs if kind != "type"]
        meta = [m.format(**f) for m in k.meta]
        if len(meta) > NMETA:
            die(f"{fam.name}: more than {NMETA} meta integers")
        vals = [str(key), k.threads.format(**f), k.smem.format(**f), str(len(ints))] + ints + [str(len(meta))] + meta
        rows.append(f'extern "C" __device__ __attribute__((used)) const long long tk_k{i}[] = {{' + ", ".join(f"(long long) ({v})" for v in vals) + "};")
    # (extern "C" on each declaration: it also gives a const object external linkage, which a brace block does not)
    lines += ["}", "}  // namespace", 'extern "C" __device__ __attribute__((used)) const char tk_cfg[] = FNX_TK_CFG;'] + rows + ["}  // namespace fnx", ""]
    return "\n".join(lines)


def deps_hash(paths):
    """A hash of the contents of the files a variant was compiled from (missing file: its name alone, so the hash changes)."""
    h = hashlib.sha256()
    for p in paths:
        h.update(p.encode() + b"\0")
        try:
            h.update(Path(p).read_bytes())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()[:16]


_CXX_ID = {}


def cxx_id(cxx):
    """The compiler's identity (its --version text): a new compiler at the same path makes new code objects."""
    if cxx not in _CXX_ID:
        try:
            _CXX_ID[cxx] = hashlib.sha256(subprocess.run([cxx, "--version"], capture_output=True, text=True).stdout.encode()).hexdigest()[:12]
        except OSError:
            _CXX_ID[cxx] = "?"
    return _CXX_ID[cxx]


def target_arch(flags):
    for f in flags:
        m = re.match(r"--offload-arch=(\w+)", f) or re.match(r"-mcpu=(gfx\w+)", f)
        if m:
            return m.group(1)
    return None


def hip_build(fam, pv, out, cxx, flags, stamp, force, keep_spills=False, want_cfg=None):
    """Compile one variant. Returns its manifest entry, or {'error': first lines} when the compiler refuses the setting. A kernel that spills
    registers to scratch memory is left out of the variant (its key is then missing) unless keep_spills: it is slow, and a kernel that holds
    registers by name in inline assembly (the WMMA units) is wrong with them (hc_mm UD=16 at T=4: an illegal memory access).
    The code object is compiled beside its final name and takes that name only when it is good (and its cfg is want_cfg, the directory's): a
    refused build leaves the files of the manifest as they were. The cache holds for the same unit text, flags, compiler and the same bytes
    in every file the compiler read (its dependency list)."""
    var = fam.vname(pv)
    d = out / fam.name
    d.mkdir(parents=True, exist_ok=True)
    tu = hip_tu(fam, pv)
    cache = hashlib.sha256((tu + "\0" + " ".join(flags) + "\0" + cxx + "\0" + cxx_id(cxx) + "\0" + str(keep_spills)).encode()).hexdigest()[:16]
    hsaco, meta_path = d / f"{var}.hsaco", d / f"{var}.json"
    if not force and hsaco.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old.get("cache") == cache and old.get("deps") and deps_hash(old["deps"]) == old.get("dephash") and want_cfg in (None, old.get("cfg")) \
                and old.get("files", {}).get(f"{fam.name}/{var}.hsaco") == digest(hsaco.read_bytes()):
            return old
    src = d / f"{var}.cu"
    src.write_text(tu)
    co, dep, stage = d / f"{var}.co.tmp", d / f"{var}.d.tmp", d / f"{var}.hsaco.tmp"
    t0 = time.time()
    # the files the unit reads: the host pass's list (a device-only compile writes no dependency file; the two passes include the same headers
    # but for what a header guards by the pass)
    rd = subprocess.run([cxx] + flags + ["-w", "-x", "hip", "-M", "-MF", str(dep), str(src)], capture_output=True, text=True)
    r = subprocess.run([cxx] + flags + ["-w", "-x", "hip", "--cuda-device-only", "-c", str(src), "-o", str(co)], capture_output=True, text=True)
    (d / f"{var}.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0 or rd.returncode != 0 or not dep.exists():
        for f in (co, dep):
            f.unlink(missing_ok=True)
        text = r.stderr if r.returncode != 0 else rd.stderr or "no dependency list"
        err = [ln for ln in text.splitlines() if "error" in ln][:2] or text.splitlines()[-2:]
        return {"error": " | ".join(e.strip()[-240:] for e in err), "params": pv}
    deps = sorted({str(Path(p).resolve()) for p in shlex.split(dep.read_text().replace("\\\n", " ")) if not p.endswith(":")} - {str(src.resolve())})
    dep.unlink()
    elf_bytes = unbundle(co.read_bytes(), target_arch(flags))
    co.unlink()
    stage.write_bytes(elf_bytes)

    def refuse(msg):
        stage.unlink(missing_ok=True)
        return {"error": msg, "params": pv}

    elf = Elf(elf_bytes)
    names = demangled(stage, Path(cxx).parent)
    cfg = elf.data("tk_cfg").split(b"\0")[0].decode()
    if want_cfg not in (None, cfg):
        return refuse(f"its cfg {cfg} is not the directory's {want_cfg} (another target or build setting: use another --out)")
    kmeta = elf.kernels()
    keys = {}
    skipped = {}
    for i, key in enumerate(fam.keys):
        k = fam.kernel(key)
        raw = elf.data(f"tk_k{i}")
        v = list(struct.unpack(f"<{len(raw) // 8}q", raw))
        kkey, threads, smem, nt = v[0], v[1], v[2], v[3]
        tvals, nm = v[4:4 + nt], v[4 + nt]
        meta = v[5 + nt:5 + nt + nm] + [0] * (NMETA - nm)
        kinds = [kind for kind, _ in k.targs]
        it = iter(tvals)
        rendered = [expr.format(**dict(pv, key=key)) if kind == "type" else render_targ(kind, next(it)) for kind, expr in k.targs]
        want = "::" + k.name + ("<" + ", ".join(rendered) + ">" if kinds else "") + "("
        hits = [idx for idx, (sn, _, _, typ) in enumerate(elf.syms) if typ == 2 and want in names.get(idx, "")]
        if len(hits) != 1:
            return refuse(f"{len(hits)} kernels match {want} in the code object")
        sym = elf.syms[hits[0]][0]
        if sym not in kmeta:
            return refuse(f"{sym} is not a kernel of the code object's metadata")
        if (kmeta[sym]["spill"] or kmeta[sym]["scratch"]) and not keep_spills:
            skipped[str(kkey)] = f"{kmeta[sym]['spill']} spilled registers, {kmeta[sym]['scratch']} scratch bytes a lane (--keep-spills keeps it)"
            continue
        keys[str(kkey)] = dict(symbol=sym, threads=threads, smem=smem, args=kmeta[sym]["args"], meta=meta, vgpr=kmeta[sym]["vgpr"],
                               spill=kmeta[sym]["spill"], scratch=kmeta[sym]["scratch"])
    if fam.keys and not keys:
        return refuse("every kernel of it spills registers: " + next(iter(skipped.values())))
    entry = dict(params=pv, file=f"{fam.name}/{var}.hsaco", files={f"{fam.name}/{var}.hsaco": digest(elf_bytes)}, src=stamp, cache=cache, cfg=cfg,
                 bytes=len(elf_bytes), seconds=round(time.time() - t0, 1), keys=keys, skipped=skipped, deps=deps, dephash=deps_hash(deps))
    os.replace(stage, hsaco)   # published: the code object, then its record
    meta_path.write_text(json.dumps(entry, indent=1))
    return entry


# ------------------------------------------------------------------ the triton backend (a worker in Triton's own python: gen/tk_triton.py)
_PY_ID = {}


def py_id(py):
    """Triton's version in that python (a new Triton at the same path makes new code objects)."""
    if py not in _PY_ID:
        try:
            _PY_ID[py] = subprocess.run([py, "-c", "import triton; print(triton.__version__)"], capture_output=True, text=True).stdout.strip() or "?"
        except OSError:
            _PY_ID[py] = "?"
    return _PY_ID[py]


def triton_build(fam, pv, out, py, stamp, force):
    var = fam.vname(pv)
    d = out / fam.name
    d.mkdir(parents=True, exist_ok=True)
    worker = HERE / "tk_triton.py"
    shapes = json.dumps({str(k): v for k, v in fam.shapes.items()}, sort_keys=True)
    cache = hashlib.sha256((json.dumps(pv, sort_keys=True) + "\0" + shapes + "\0" + str(fam.triton) + "\0" + worker.read_text() + (HERE / "tkgen.py").read_text()
                            + py + py_id(py)).encode()).hexdigest()[:16]
    hsaco, meta_path = d / f"{var}.hsaco", d / f"{var}.json"
    if not force and meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old.get("cache") == cache and old.get("files") and all((out / f).exists() and digest((out / f).read_bytes()) == rec
                                                                  for f, rec in old["files"].items()):
            return old
    t0 = time.time()
    r = subprocess.run([py, str(worker), "--kernel", fam.triton, "--params", json.dumps(pv), "--keys", json.dumps({str(k): v for k, v in fam.shapes.items()}), "--out", str(hsaco)],
                       capture_output=True, text=True)
    (d / f"{var}.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        return {"error": " | ".join(r.stderr.strip().splitlines()[-2:])[-400:], "params": pv}
    info = json.loads(r.stdout.strip().splitlines()[-1])
    # a code object per key (each k names its file); its record is the worker's, of the bytes the worker read the kernel's metadata from
    files = {k["file"]: k.pop("digest", None) for k in info["keys"].values()}
    for f, rec in files.items():
        if not rec or digest((out / f).read_bytes()) != rec:
            return {"error": f"{f} is not the code object the worker made (its bytes changed after the worker wrote it)", "params": pv}
    size = sum(rec["bytes"] for rec in files.values())
    entry = dict(params=pv, file=None, files=files, src=stamp, cache=cache, cfg=None, bytes=size, seconds=round(time.time() - t0, 1), keys=info["keys"],
                 triton=info.get("triton"), skipped=info.get("skipped", {}))
    meta_path.write_text(json.dumps(entry, indent=1))
    return entry


# ------------------------------------------------------------------ the manifest
def load_manifest(out):
    p = out / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else dict(version=1, target=None, cfg=None, src=None, families={}, use={})


def fnv64(buf):
    h = 1469598103934665603
    for b in buf:
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def digest(buf):
    """A code object's record: sha256, bytes and FNV-1a 64 (what src/tk.hpp checks when it reads the file). It is taken from the bytes the
    kernels' symbols, argument sizes and cfg were read from, at build time, and stays with them in the variant's entry: the manifest never
    takes a new one from the file on disk (a file replaced later does not run: the loader's check fails)."""
    return dict(sha256=hashlib.sha256(buf).hexdigest(), bytes=len(buf), fnv=f"{fnv64(buf):016x}")


_LOCK = {}


def lock_out(out):
    """One tkgen at a time changes an output directory (the lock is this process's until it ends)."""
    if str(out) in _LOCK:
        return
    out.mkdir(parents=True, exist_ok=True)
    f = open(out / ".lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        die(f"another tkgen works on {out}")
    _LOCK[str(out)] = f


def write_manifest(out, man):
    lines = ["# written by tkgen.py; read by src/tk.hpp", "tk 1", f"cfg {man['cfg']}", f"src {man['src']}"]
    for fam, var in sorted(man.get("use", {}).items()):
        lines.append(f"use {fam} {var}")
    files, recs = {}, {}   # file -> id, file -> its record; a kernel names its code object (a composed variant takes its keys from several)
    body = []
    for fname, f in sorted(man["families"].items()):
        for var, e in sorted(f["variants"].items()):
            ps = ",".join(f"{k}={v}" for k, v in e["params"].items()) or "-"
            body.append(f"v {fname} {var} {ps}")
            for key, k in sorted(e["keys"].items(), key=lambda kv: int(kv[0])):
                file = k.get("file", e.get("file"))
                rec = e.get("files", {}).get(file)
                if not rec:
                    die(f"{fname} {var}: no record of the bytes of {file} (an entry of an older tkgen: build it again with --force)")
                if file not in files:
                    if re.search(r"\s", file):
                        die(f"a code object's path has white space: {file!r}")
                    files[file] = f"f{len(files)}"
                    recs[file] = rec
                    lines.append(f"f {files[file]} {file} {rec['sha256']} {rec['bytes']} {rec['fnv']}")
                elif recs[file] != rec:
                    die(f"{fname} {var}: {file} has two records (two builds of it in one manifest): build the family again with --force")
                body.append(f"k {fname} {var} {key} {files[file]} {k['symbol']} {k['threads']} {k['smem']} {','.join(str(x) for x in k['args'])} " + " ".join(str(m) for m in k["meta"]))
    lines += body
    man.pop("files", None)
    for name, text in (("manifest.json", json.dumps(man, indent=1)), ("manifest.tk", "\n".join(lines) + "\n")):
        tmp = out / (name + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, out / name)


def regs(e):
    """A variant's register use: the most vector registers a kernel of it takes, and the kernels that spill to scratch memory (slow)."""
    ks = e["keys"]
    spilled = sorted((int(k) for k, v in ks.items() if v.get("spill") or v.get("scratch")))
    return f"vgpr {max((v.get('vgpr', 0) for v in ks.values()), default=0):3d}" + (f"  SPILLS at keys {','.join(map(str, spilled))}" if spilled else "")


def cmd_build(a):
    out = Path(a.out).resolve()
    lock_out(out)
    cxx, flags = target_flags(a.build_dir, a.target)
    stamp = src_stamp()
    man = load_manifest(out)
    if man["target"] not in (None, a.target):
        die(f"{out} holds variants of target {man['target']}; use another --out for {a.target}")
    # the target's cfg now (a unit with no kernel): a directory made with other build settings is refused before anything in it changes
    probe = hip_build(load_families()["tk_probe"], {}, out, cxx, flags, stamp, a.force)
    if "error" in probe:
        die("tk_probe (the target's cfg): " + probe["error"])
    if man["cfg"] not in (None, probe["cfg"]):
        die(f"{out} was made for another build of {a.target} (its kernels' argument layouts may differ); use another --out or remove it\n"
            f"  directory {man['cfg']}\n  target    {probe['cfg']}")
    man["target"], man["src"], man["cfg"] = a.target, stamp, probe["cfg"]
    jobs, seen = [], set()
    for spec in a.spec or [n for n in load_families() if n != "tk_probe"]:
        fam, sets = expand_spec(spec)
        for pv in sets:
            if (fam.name, fam.vname(pv)) not in seen:   # (a variant named by two specs is one job: two would share its staging files)
                seen.add((fam.name, fam.vname(pv)))
                jobs.append((fam, pv))
    print(f"tkgen: {len(jobs)} variants, target {a.target}, src {stamp}", flush=True)
    ok = bad = 0

    def run(job):
        fam, pv = job
        if fam.backend == "hip":
            return job, hip_build(fam, pv, out, cxx, flags, stamp, a.force, a.keep_spills, man["cfg"])
        return job, triton_build(fam, pv, out, a.triton_python, stamp, a.force)

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for (fam, pv), e in ex.map(run, jobs):
            var = fam.vname(pv)
            if "error" in e:
                bad += 1
                print(f"  {fam.name:14s} {var:28s} REJECTED: {e['error']}", flush=True)
                continue
            ok += 1
            f = man["families"].setdefault(fam.name, dict(backend=fam.backend, path=fam.path, variants={}))
            f["variants"][var] = e
            skip = "".join(f"\n      no key {k}: {why}" for k, why in e.get("skipped", {}).items())
            print(f"  {fam.name:14s} {var:28s} {e['bytes'] // 1024:5d} KB  {len(e['keys'])} kernels  {regs(e)}  {e['seconds']} s{skip}", flush=True)
    write_manifest(out, man)
    print(f"tkgen: {ok} built, {bad} rejected -> {out}/manifest.tk")
    return 1 if bad and not ok else 0


def cmd_use(a):
    out = Path(a.out).resolve()
    lock_out(out)
    man = load_manifest(out)
    for kv in a.pick:
        fam, _, var = kv.partition("=")
        if fam not in man["families"]:
            die(f"the manifest has no family {fam}")
        if var in ("0", "builtin", ""):
            man["use"].pop(fam, None)
        elif var not in man["families"][fam]["variants"]:
            die(f"{fam}: no variant {var} (has: {', '.join(man['families'][fam]['variants'])})")
        else:
            man["use"][fam] = var
    write_manifest(out, man)
    print("use:", man["use"] or "every family built in")


def cmd_compose(a):
    """A variant whose keys come from other variants of the family: the tuner's pick per shape (KEYS=variant; KEYS = 3, 1-4 or *)."""
    out = Path(a.out).resolve()
    lock_out(out)
    man = load_manifest(out)
    if a.family not in man["families"]:
        die(f"the manifest has no family {a.family}")
    have = man["families"][a.family]["variants"]
    fam = load_families().get(a.family)
    if fam is not None and fam.coupled:
        die(f"{a.family}: its keys are the parts of one layout (a variant's kernels come from one setting): no compose")
    if a.name in have and not have[a.name].get("composed"):
        die(f"{a.family}: {a.name} is a built variant's name")
    keys, files = {}, {}
    parts = []
    for pick in a.pick:
        ks, _, var = pick.partition("=")
        if var not in have:
            die(f"{a.family}: no variant {var}")
        src = have[var]
        if ks == "*":
            sel = list(src["keys"])
        elif "-" in ks:
            lo, hi = ks.split("-")
            sel = [str(k) for k in range(int(lo), int(hi) + 1)]
        else:
            sel = [ks]
        for k in sel:
            if k not in src["keys"]:
                die(f"{a.family} {var}: no key {k}")
            file = src["keys"][k].get("file", src.get("file"))
            if file not in src.get("files", {}):
                die(f"{a.family} {var}: no record of the bytes of {file} (an entry of an older tkgen: build it again with --force)")
            keys[k] = dict(src["keys"][k], file=file)
            files[file] = src["files"][file]
        parts.append(f"{ks}:{var}")
    have[a.name] = dict(params={"composed": "+".join(parts)}, file=None, files=files, src=man["src"], cache=None, cfg=None, composed=True,
                        bytes=0, seconds=0, keys=keys)
    write_manifest(out, man)
    print(f"{a.family} {a.name}: {len(keys)} keys from {', '.join(parts)}")


def cmd_arms(a):
    """The family's variants as lines for a test queue ("name binsuffix K=V ...")."""
    out = Path(a.out).resolve()
    man = load_manifest(out)
    if a.family not in man["families"]:
        die(f"the manifest has no family {a.family}")
    have = man["families"][a.family]["variants"]
    variants = list(have) if a.variants in (None, "all") else a.variants.split(",")
    envname = "FNX_TK_" + re.sub(r"[^A-Za-z0-9]", "_", a.family).upper()
    tag = re.sub(r"[^a-z0-9]", "", a.family.lower())[:6]
    print(f'"{tag}-builtin {a.bin}"')
    for v in variants:
        if v not in have:
            die(f"{a.family}: no variant {v}")
        print(f'"{tag}-{v} {a.bin} FNX_TK_DIR={out} {envname}={v}"')


ARITH_OP = re.compile(r"_f(16|32|64)|fma|v_dot")


def norm_asm(text):
    """llvm-objdump -d output as instruction lines that do not depend on where the function is: no addresses, no symbol names, no comments
    (a branch keeps its relative count, an instruction keeps its immediates)."""
    out = []
    for line in text.splitlines():
        if line[:1] not in ("\t", " "):
            continue
        line = re.sub(r"\s+", " ", re.sub(r"\s*//.*$", "", re.sub(r"<[^>]*>", "", line))).strip()
        if line:
            out.append(line)
    return out


def arith_ops(lines):
    """How many of each float operation and dot product a function holds, as a sorted list of (operation, count). The names are made
    independent of the choices that do not change a result: a pair of operations in one instruction (v_dual_*, two halves around `::`)
    counts as its two operations, the encoding suffix is dropped, an accumulating multiply-add is a multiply-add, a reversed subtract is a
    subtract and a compare's direction is left out (the operands swap with it). The count of multiply-adds against multiplies and adds is
    what a different inlining would change (a product and a sum contract to one operation only inside one function). It is a screen, not a
    proof: operands, constants and order are not compared."""
    n = {}
    for line in lines:
        for half in line.split("::"):
            op = (half.split() or [""])[0]
            if not ARITH_OP.search(op):
                continue
            op = re.sub(r"_e(32|64)(_dpp)?$", "", op.replace("v_dual_", "v_"))
            op = op.replace("fmac", "fma").replace("subrev", "sub")
            op = re.sub(r"_cmp_(n?)(lt|gt)_", r"_cmp_\1ltgt_", re.sub(r"_cmp_(n?)(le|ge)_", r"_cmp_\1lege_", op))
            n[op] = n.get(op, 0) + 1
    return sorted(n.items())


def fatbin_objects(buf, arch):
    """The GPU code objects of a host binary: its .hip_fatbin section is a row of clang offload bundles (one a kernel source file) with zero
    bytes between them. The row is walked by each bundle's own extent (a bundle's bytes are not searched for the next one's start)."""
    fat = Elf(buf).section(".hip_fatbin")
    if fat is None:
        die("the binary has no .hip_fatbin section (not a HIP build)")
    objs, at = [], 0
    while True:
        m = re.compile(rb"[^\0]").search(fat, at)
        if not m:
            break
        at = m.start()
        if not fat.startswith(BUNDLE_MAGIC, at):
            die(f".hip_fatbin: bytes that are no offload bundle at {at:#x}")
        entries = bundle_entries(fat, at)
        end = max([o + sz for _, o, sz in entries] + [at + len(BUNDLE_MAGIC) + 8])
        objs.append(unbundle(fat[at:end], arch))
        at = end
    if not objs:
        die(".hip_fatbin holds no offload bundle")
    return objs


def cmd_verify(a):
    """A variant's kernels against the binary's kernels of the same names, with no card: a screen, not a proof. For `base` (all defaults) the
    two come from the same source with the same flags, in two translation units. A kernel can be in several of the binary's code objects
    (one a kernel source file that uses it); the variant's is compared with each copy and the worst result counts.
      same code       the same instructions with the same immediates (data the code reads from other sections is not compared)
      same ops        the same count of each float operation and dot product; order, registers, integer address math differ (the compiler
                      schedules two units in two ways)
      OPS DIFFER      the units compiled the arithmetic in two ways (a product and a sum in place of a multiply-add, ...): exit code 1
    Equal result bits are proven by the family's bench hash on a card and by the engine's texts; this check finds a different contraction
    early and covers the engine target, which kbench does not run. A variant or family that is asked for and cannot be compared is an error."""
    out = Path(a.out).resolve()
    man = load_manifest(out)
    tools = Path(a.tools)
    arch = "gfx" + dict(x.split("=", 1) for x in (man["cfg"] or "").split(";") if "=" in x).get("arch", "1201")
    objs = fatbin_objects(Path(a.binary).read_bytes(), arch)
    bad = compared = 0
    with tempfile.TemporaryDirectory() as td:
        builtin = {}   # symbol -> [(file, address, size)]: every copy
        for i, o in enumerate(objs):
            f = Path(td) / f"part{i}.elf"
            f.write_bytes(o)
            for name, value, size, typ in Elf(o).syms:
                if typ == 2 and size:
                    builtin.setdefault(name, []).append((f, value, size))

        def dis(f, addr, size):
            r = subprocess.run([str(tools / "llvm-objdump"), "-d", "--no-show-raw-insn", "--no-leading-addr", f"--start-address={addr}",
                                f"--stop-address={addr + size}", str(f)], capture_output=True, text=True, check=True)
            text = norm_asm(r.stdout)
            if not text:
                die(f"llvm-objdump gave no instructions for {f} at {addr:#x} (+{size}): nothing to compare")
            return text

        for fname in a.family or sorted(n for n, f in man["families"].items() if f["backend"] == "hip" and n != "tk_probe"):
            fam = man["families"].get(fname) or die(f"{fname}: not in {out}/manifest.json")
            if fam["backend"] != "hip":
                die(f"{fname}: a {fam['backend']} family has no kernel in the binary to compare with")
            e = fam["variants"].get(a.variant)
            if not e or not e["keys"]:
                die(f"{fname}: no variant {a.variant} with kernels in {out}/manifest.json (has: {', '.join(fam['variants'])})")
            n = dict(same=0, ops=0, differ=0, missing=0)
            for key, k in sorted(e["keys"].items(), key=lambda kv: int(kv[0])):
                sym = k["symbol"]
                vf = out / k.get("file", e.get("file"))
                ve = Elf(unbundle(vf.read_bytes(), arch))
                copies = builtin.get(sym, [])
                if not copies or sym not in ve.by:
                    n["missing"] += 1
                    verdict = "no kernel of this name in the binary" if not copies else "the variant's code object lacks its kernel"
                else:
                    _, value, size, _ = ve.by[sym]
                    tv = dis(vf, value, size)
                    ov = arith_ops(tv)
                    texts = {}   # distinct copies
                    for c in copies:
                        t = dis(*c)
                        texts.setdefault("\n".join(t), t)
                    worst, note = 0, ""
                    for t in texts.values():
                        if t == tv:
                            continue
                        if arith_ops(t) == ov:
                            worst = max(worst, 1)
                        else:
                            worst = 2
                            ob = dict(arith_ops(t))
                            note = ": " + ", ".join(f"{o} {dict(ov).get(o, 0)} / {ob.get(o, 0)}" for o in sorted(set(dict(ov)) | set(ob))
                                                    if dict(ov).get(o, 0) != ob.get(o, 0))
                    n[("same", "ops", "differ")[worst]] += 1
                    compared += 1
                    verdict = ("same code", f"same ops ({sum(c for _, c in ov)})", "OPS DIFFER (variant / binary)" + note)[worst] + \
                              f"; {len(copies)} copies in the binary, {len(texts)} distinct"
                if a.verbose:
                    print(f"  {fname} {a.variant} key {key}: {verdict}")
            bad += n["differ"] + n["missing"]
            print(f"{fname} {a.variant} vs {Path(a.binary).name}: {len(e['keys'])} kernels: {n['same']} same code, {n['ops']} same ops, "
                  f"{n['differ']} OPS DIFFER, {n['missing']} not compared")
    if not compared:
        die("nothing was compared")
    return 1 if bad else 0


def cmd_show(a):
    man = load_manifest(Path(a.out).resolve())
    print(f"target {man['target']}  src {man['src']}\ncfg {man['cfg']}")
    for fname, f in sorted(man["families"].items()):
        print(f"{fname} ({f['backend']}, {f['path']}): default {man['use'].get(fname, 'built in')}")
        for var, e in sorted(f["variants"].items()):
            if not e["keys"]:
                continue
            k0 = next(iter(e["keys"].values()))
            print(f"  {var:28s} keys {len(e['keys']):2d}  threads {k0['threads']:4d}  smem {k0['smem']:6d}  {e['bytes'] // 1024:5d} KB  {regs(e)}  src {e['src']}")


def cmd_list(a):
    for f in load_families().values():
        if f.name == "tk_probe":
            continue
        print(f"{f.name} ({f.backend}, {f.path}): {f.doc}")
        for n, p in f.params.items():
            print(f"    {n:8s} code {p.code:6s} values {p.values}  built-in {p.default}")
        if f.presets:
            print(f"    presets: {', '.join(f.presets)}")
        if f.bench:
            print(f"    benches: {', '.join(f.bench)}")
        print(f"    grid: {sum(1 for _ in f.grid())} settings, keys {f.keys if len(f.keys) < 6 else str(f.keys[0]) + '..' + str(f.keys[-1])}")


# ------------------------------------------------------------------ sweeps
def cmd_sweep(a):
    out = Path(a.out).resolve()
    man = load_manifest(out)
    fam = load_families().get(a.family) or die(f"no family {a.family}")
    if a.family not in man["families"]:
        die(f"{out}: no variants of {a.family}; run build first")
    bench = fam.bench.get(a.bench) or die(f"{a.family}: no bench {a.bench} (has: {', '.join(fam.bench)})")
    have = man["families"][a.family]["variants"]
    variants = list(have) if a.variants in (None, "all") else a.variants.split(",")
    for v in variants:
        if v not in have:
            die(f"{a.family}: the manifest has no variant {v}")
    fmt = dict(kbench=a.kbench or "", engine=a.engine or "", dev=a.device)
    for kv in a.arg or []:
        k, _, v = kv.partition("=")
        fmt[k] = v
    cmd = [c.format(**fmt) for c in bench.cmd]
    envname = "FNX_TK_" + re.sub(r"[^A-Za-z0-9]", "_", a.family).upper()
    ref = "builtin" if fam.builtin else variants[0]   # the arm the others are compared with
    arms = ([ref] if fam.builtin else []) + variants + ([ref] if a.bracket else [])
    rows = []
    for rep in range(a.reps):
        for arm in arms:
            env = dict(os.environ, FNX_TK_DIR=str(out), **bench.env)
            env[envname] = "0" if arm == "builtin" else arm
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=a.timeout)
            m = bench.parse(r.stdout) if r.returncode == 0 else {}
            row = dict(arm=arm, rep=rep, rc=r.returncode, seconds=round(time.time() - t0, 1), metrics=m)
            if r.returncode != 0:
                row["error"] = (r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ["?"])[-1][-300:]
            rows.append(row)
            print(f"  {arm:28s} rep {rep} rc {r.returncode} " + " ".join(f"{k}={v}" for k, v in m.items()) + (" " + row.get("error", "")), flush=True)
    # the table: each metric's mean per arm; numbers against the built-in's mean, strings (hashes) as equal / differs
    metrics = []
    for row in rows:
        for k in row["metrics"]:
            if k not in metrics:
                metrics.append(k)
    base = {}
    table = []
    for arm in dict.fromkeys(arms):
        cells = {}
        for k in metrics:
            vals = [r["metrics"][k] for r in rows if r["arm"] == arm and k in r["metrics"]]
            if not vals:
                cells[k] = None
            elif all(isinstance(v, (int, float)) for v in vals):
                cells[k] = sum(vals) / len(vals)
            else:
                cells[k] = vals[0] if len(set(vals)) == 1 else ("UNSTABLE", tuple(dict.fromkeys(str(v) for v in vals)))
        if arm == ref:
            base = cells
        table.append((arm, cells))
    lines = ["| variant | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    for arm, cells in table:
        out_cells = []
        for k in metrics:
            v, b = cells[k], base.get(k)
            if v is None:
                out_cells.append("-")
            elif isinstance(v, float) or isinstance(v, int):
                rel = f" ({(v / b - 1) * 100:+.1f} %)" if isinstance(b, (int, float)) and b and arm != ref else ""
                out_cells.append(f"{v:.2f}{rel}")
            elif isinstance(v, tuple):   # its runs gave different strings (hashes): never "same"
                out_cells.append("UNSTABLE " + " / ".join(v[1]))
            elif arm == ref:
                out_cells.append(str(v))
            else:
                out_cells.append("same" if v == b else f"DIFFERS {v}" + (" (the reference is unstable)" if isinstance(b, tuple) else ""))
        lines.append(f"| {arm} | " + " | ".join(out_cells) + " |")
    text = "\n".join(lines)
    print(text)
    sd = out / "sweeps"
    sd.mkdir(exist_ok=True)
    tag = time.strftime("%Y%m%d-%H%M%S")
    (sd / f"{a.family}-{a.bench}-{tag}.json").write_text(json.dumps(dict(family=a.family, bench=a.bench, cmd=cmd, src=man["src"], rows=rows), indent=1))
    (sd / f"{a.family}-{a.bench}-{tag}.md").write_text(f"# sweep {a.family} / {a.bench} ({tag})\n\n`{' '.join(cmd)}`\n\n{text}\n")
    print(f"tkgen: {sd}/{a.family}-{a.bench}-{tag}.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    b = sub.add_parser("build")
    b.add_argument("--build-dir", required=True)
    b.add_argument("--target", default="strata-split-4")
    b.add_argument("--out", required=True)
    b.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 4))
    b.add_argument("--force", action="store_true")
    b.add_argument("--keep-spills", action="store_true", help="keep a hip kernel that spills registers to scratch memory")
    b.add_argument("--triton-python", default=os.environ.get("TKGEN_TRITON_PY", "python3"))
    b.add_argument("spec", nargs="*")
    b.set_defaults(fn=cmd_build)
    u = sub.add_parser("use")
    u.add_argument("--out", required=True)
    u.add_argument("pick", nargs="+")
    u.set_defaults(fn=cmd_use)
    s = sub.add_parser("show")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_show)
    c = sub.add_parser("compose")
    c.add_argument("--out", required=True)
    c.add_argument("--family", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("pick", nargs="+", help="KEYS=variant (KEYS: 3, 1-4 or *)")
    c.set_defaults(fn=cmd_compose)
    r = sub.add_parser("arms")
    r.add_argument("--out", required=True)
    r.add_argument("--family", required=True)
    r.add_argument("--variants")
    r.add_argument("--bin", required=True, help="the engine binary's suffix (bin/strata-split-4-<suffix>)")
    r.set_defaults(fn=cmd_arms)
    y = sub.add_parser("verify", help="a variant's kernels against the binary's kernels of the same names (no GPU)")
    y.add_argument("--out", required=True)
    y.add_argument("--binary", required=True, help="the target's binary (strata-split-4 or the kbench the directory was made for)")
    y.add_argument("--variant", default="base")
    y.add_argument("--tools", default="/opt/rocm/llvm/bin", help="the directory of llvm-objdump")
    y.add_argument("-v", "--verbose", action="store_true")
    y.add_argument("family", nargs="*")
    y.set_defaults(fn=cmd_verify)
    w = sub.add_parser("sweep")
    w.add_argument("--out", required=True)
    w.add_argument("--family", required=True)
    w.add_argument("--bench", required=True)
    w.add_argument("--variants")
    w.add_argument("--kbench")
    w.add_argument("--engine")
    w.add_argument("--device", default="3")
    w.add_argument("--arg", action="append", help="name=value for the bench command")
    w.add_argument("--reps", type=int, default=1)
    w.add_argument("--bracket", action="store_true", help="run the built-in again after the variants")
    w.add_argument("--timeout", type=int, default=900)
    w.set_defaults(fn=cmd_sweep)
    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
