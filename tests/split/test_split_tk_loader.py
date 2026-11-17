# (generator) src/split/src/tk.hpp's manifest loader, run on the host: a small program that calls tk::init and tk::family, built
# with the ROCm clang (no card and no HIP library: the loader's read and selection paths make no HIP call). Skipped where the compiler or the
# HIP headers are missing. What it holds: a named variant that cannot be used is an error, never a silent run of the built-in kernel.
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "split" / "src"
CXX = Path("/opt/rocm/llvm/bin/clang++")
HIP_INC = Path("/opt/rocm/include")

PROGRAM = r"""
#include "tk.hpp"
int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--cfg") return std::printf("%s\n", FNX_TK_CFG), 0;
    try {
        fnx::tk::init({"pf_attn_u", "hc_mm", "moe_gu_q4k"});
        for (int i = 1; i < argc; ++i) {
            const fnx::tk::Family* f = fnx::tk::family(argv[i]);
            std::printf("%s=%s\n", argv[i], f ? f->variant.c_str() : "builtin");
        }
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
        return 1;
    }
    return 0;
}
"""


def load_tkgen():   # (one tkgen module for the test files of a run: its families register into the module they import by name)
    if "tkgen" not in sys.modules:
        spec = importlib.util.spec_from_file_location("tkgen", ROOT / "src" / "split" / "gen" / "tkgen.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["tkgen"] = mod
        spec.loader.exec_module(mod)
    sys.modules["tkgen"].load_families()
    return sys.modules["tkgen"]


@unittest.skipUnless(CXX.exists() and (HIP_INC / "hip" / "hip_runtime.h").exists(), "no ROCm compiler or HIP headers")
class Loader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = load_tkgen()
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        (d / "t.cpp").write_text(PROGRAM)
        r = subprocess.run([str(CXX), "-x", "c++", "-std=c++17", "-w", "-DFNX_HIP=1", "-DFNX_NCARD=4", "-DFNX_ARCH=1201", "-D__HIP_PLATFORM_AMD__=1",
                            f"-I{ROOT / 'include' / 'strata' / 'hip_compat'}", f"-I{SRC / 'hip'}", f"-I{SRC}", f"-I{HIP_INC}", str(d / "t.cpp"), "-o", str(d / "t")],
                           capture_output=True, text=True)
        if r.returncode != 0:   # (the compiler and the HIP headers are here: a build error is tk.hpp's, not the machine's)
            raise AssertionError("the loader program does not build: " + r.stderr[-1500:])
        cls.exe = str(d / "t")
        cls.cfg = subprocess.run([cls.exe, "--cfg"], capture_output=True, text=True).stdout.strip()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def manifest(self, d, body, cfg=None, head=True):
        """A directory with manifest.tk: the head lines, an f line for a.hsaco (id f0) and the body's lines."""
        d = Path(d)
        obj = b"code object bytes"
        (d / "a.hsaco").write_bytes(obj)
        lines = (["tk 1", f"cfg {cfg or self.cfg}", "src s1"] if head else []) + [f"f f0 a.hsaco {'0' * 64} {len(obj)} {self.t.fnv64(obj):016x}"] + body
        (d / "manifest.tk").write_text("\n".join(lines) + "\n")
        return d

    def run_tk(self, d, env=None, fams=("hc_mm",)):
        e = {k: v for k, v in os.environ.items() if not k.startswith("FNX_TK_")}
        if d is not None:
            e["FNX_TK_DIR"] = str(d)
        e.update(env or {})
        r = subprocess.run([self.exe, *fams], capture_output=True, text=True, env=e)
        return r.returncode, r.stdout.strip(), r.stderr.strip()

    K = "k hc_mm {v} {key} f0 sym{key} 256 0 184,8 8 0 0 0 0 0 0 0"
    GOOD = ["v hc_mm ud4 UD=4", K.format(v="ud4", key=1), K.format(v="ud4", key=2), "v hc_mm base -", K.format(v="base", key=1)]

    def test_no_directory_is_every_family_built_in(self):
        self.assertEqual(self.run_tk(None), (0, "hc_mm=builtin", ""))
        rc, out, _ = self.run_tk(None, {"FNX_TK_HC_MM": "ud4"})   # a variant is named and no manifest holds it
        self.assertEqual(rc, 1)
        self.assertIn("names no family", out)

    def test_a_variant_is_picked_by_name_and_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.manifest(tmp, self.GOOD)
            self.assertEqual(self.run_tk(d), (0, "hc_mm=builtin", ""))
            rc, out, err = self.run_tk(d, {"FNX_TK_HC_MM": "ud4"})
            self.assertEqual((rc, out), (0, "hc_mm=ud4"))
            self.assertIn("fnx: tk: hc_mm = ud4 (2 kernels;", err)
            for v in ("0", "builtin"):
                self.assertEqual(self.run_tk(d, {"FNX_TK_HC_MM": v})[:2], (0, "hc_mm=builtin"))

    def test_the_manifests_use_line_and_what_overrides_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.manifest(tmp, ["use hc_mm ud4"] + self.GOOD)
            self.assertEqual(self.run_tk(d)[:2], (0, "hc_mm=ud4"))
            self.assertEqual(self.run_tk(d, {"FNX_TK_HC_MM": "base"})[:2], (0, "hc_mm=base"))
            self.assertEqual(self.run_tk(d, {"FNX_TK_HC_MM": "0"})[:2], (0, "hc_mm=builtin"))
            rc, out, _ = self.run_tk(d, {"FNX_TK_HC_MM": ""})   # an empty value would drop the use line without a word
            self.assertEqual(rc, 1)
            self.assertIn("empty string", out)

    def test_a_name_that_cannot_be_used_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.manifest(tmp, self.GOOD)
            for env, what in (({"FNX_TK_HC_MM": "ud5"}, "no such variant"), ({"FNX_TK_HC_NN": "ud4"}, "names no family"),
                              ({"FNX_TK_NOPE": ""}, "empty string")):
                rc, out, _ = self.run_tk(d, env)
                self.assertEqual(rc, 1, env)
                self.assertIn(what, out)
        with tempfile.TemporaryDirectory() as tmp:   # a family with no launch site in this binary (the engine has none for gemm8)
            d = self.manifest(tmp, ["v gemm8 base -", "k gemm8 base 0 f0 g 128 0 8,8 0 0 0 0 0 0 0 0"])
            rc, out, _ = self.run_tk(d, {"FNX_TK_GEMM8": "base"})
            self.assertEqual(rc, 1)
            self.assertIn("no launch site", out)
            self.assertEqual(self.run_tk(d)[0], 0)   # (not selected: no error)
        with tempfile.TemporaryDirectory() as tmp:   # a use line for a family the manifest lacks
            d = self.manifest(tmp, ["use moe_gu_q4k base"] + self.GOOD)
            rc, out, _ = self.run_tk(d)
            self.assertEqual(rc, 1)
            self.assertIn("no family moe_gu_q4k", out)

    def test_another_builds_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.manifest(tmp, self.GOOD, cfg=self.cfg.replace("ncard=4", "ncard=2"))
            rc, out, _ = self.run_tk(d)
            self.assertEqual(rc, 1)
            self.assertIn("another build", out)

    def test_a_code_object_that_is_not_the_manifests_is_refused_at_the_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.manifest(tmp, self.GOOD)
            (d / "a.hsaco").write_bytes(b"code object bytez")   # same size, other bytes
            rc, out, _ = self.run_tk(d, {"FNX_TK_HC_MM": "ud4"})
            self.assertEqual(rc, 1)
            self.assertIn("not the code object of the manifest", out)
            self.assertEqual(self.run_tk(d)[0], 0)   # (no variant selected: the file is not read)
            (d / "a.hsaco").unlink()
            rc, out, _ = self.run_tk(d, {"FNX_TK_HC_MM": "ud4"})
            self.assertEqual(rc, 1)
            self.assertIn("cannot read", out)

    def test_lines_that_are_not_exactly_the_format_are_errors(self):
        k1 = self.K.format(v="ud4", key=1)
        cases = {
            "a second tk line": ["tk 1"] + self.GOOD,
            "a second cfg line": [f"cfg {self.cfg}"] + self.GOOD,
            "a second src line": ["src s2"] + self.GOOD,
            "a family's second use line": ["use hc_mm ud4", "use hc_mm builtin"] + self.GOOD,
            "a variant's second v line": self.GOOD + ["v hc_mm ud4 UD=4"],
            "no v line before it": [k1],
            "a key twice": ["v hc_mm ud4 -", k1, k1],
            "a file id twice": ["f f0 a.hsaco " + "0" * 64 + " 17 0000000000000000"] + self.GOOD,
            "no f line before it": ["v hc_mm ud4 -", k1.replace(" f0 ", " f9 ")],
            "more fields": ["v hc_mm ud4 -", k1 + " 7"],
            "more fields than": ["use hc_mm ud4 extra"] + self.GOOD,
            "lower case": ["v HC_MM ud4 -"],
            "unknown line": ["w hc_mm"] + self.GOOD,
            "argument sizes": ["v hc_mm ud4 -", k1.replace("184,8", "184,,8")],
            "block threads": ["v hc_mm ud4 -", k1.replace(" 256 0 ", " 4096 0 ")],
            "too few meta": ["v hc_mm ud4 -", k1.rsplit(" ", 1)[0]],
        }
        for what, body in cases.items():
            with tempfile.TemporaryDirectory() as tmp:
                d = self.manifest(tmp, body)
                rc, out, _ = self.run_tk(d)
                self.assertEqual(rc, 1, what)
                self.assertIn(what, out, what)
        with tempfile.TemporaryDirectory() as tmp:   # no head
            d = self.manifest(tmp, self.GOOD, head=False)
            rc, out, _ = self.run_tk(d)
            self.assertEqual(rc, 1)
            self.assertIn("no `tk 1` or no `cfg` line", out)

    def test_tkgens_manifest_is_what_the_loader_reads(self):
        # write_manifest's lines through the loader: the two sides of the format agree
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "hc_mm").mkdir()
            obj = b"object"
            (d / "hc_mm" / "ud4.hsaco").write_bytes(obj)
            e = dict(params={"UD": 4}, file="hc_mm/ud4.hsaco", files={"hc_mm/ud4.hsaco": self.t.digest(obj)}, src="s", cache="c", cfg=self.cfg, bytes=len(obj), seconds=0,
                     keys={str(k): dict(symbol=f"sym{k}", threads=256, smem=0, args=[184, 8], meta=[8] + [0] * 7) for k in (1, 2, 3)})
            man = dict(version=1, target="strata-split-4", cfg=self.cfg, src="abc", use={"hc_mm": "ud4"},
                       families={"hc_mm": dict(backend="hip", path="decode", variants={"ud4": e})})
            self.t.write_manifest(d, man)
            rc, out, err = self.run_tk(d)
            self.assertEqual((rc, out), (0, "hc_mm=ud4"))
            self.assertIn("fnx: tk: hc_mm = ud4 (3 kernels;", err)


if __name__ == "__main__":
    unittest.main()
