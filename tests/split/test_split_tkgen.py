"""src/split/gen/tkgen.py (the kernel generator): family specs, variant names, the generated HIP translation unit, the manifest the engine
reads (src/split/src/tk.hpp), the code object readers. No compiler and no card."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "src" / "split" / "gen"


def load_tkgen():
    if "tkgen" not in sys.modules:
        spec = importlib.util.spec_from_file_location("tkgen", GEN / "tkgen.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["tkgen"] = mod
        spec.loader.exec_module(mod)
    sys.modules["tkgen"].load_families()
    return sys.modules["tkgen"]


def pack(v):
    """A msgpack encoder for the test (the subset the reader handles)."""
    if isinstance(v, bool):
        return b"\xc3" if v else b"\xc2"
    if isinstance(v, int):
        return bytes([v]) if 0 <= v < 128 else b"\xce" + struct.pack(">I", v)
    if isinstance(v, str):
        b = v.encode()
        return (bytes([0xA0 | len(b)]) if len(b) < 32 else b"\xd9" + bytes([len(b)]) if len(b) < 256 else b"\xda" + struct.pack(">H", len(b))) + b
    if isinstance(v, list):
        return (bytes([0x90 | len(v)]) if len(v) < 16 else b"\xdc" + struct.pack(">H", len(v))) + b"".join(pack(x) for x in v)
    if isinstance(v, dict):
        return (bytes([0x80 | len(v)]) if len(v) < 16 else b"\xde" + struct.pack(">H", len(v))) + b"".join(pack(k) + pack(x) for k, x in v.items())
    raise TypeError(type(v))


class Families(unittest.TestCase):
    def setUp(self):
        self.t = load_tkgen()

    def test_both_paths_have_families(self):
        paths = {f.path for f in self.t.REG.values()}
        self.assertTrue({"prefill", "decode"} <= paths)
        self.assertEqual({f.backend for f in self.t.REG.values()}, {"hip", "triton"})

    def test_the_default_setting_is_base_and_in_the_grid(self):
        for f in self.t.REG.values():
            if not f.params:
                continue
            self.assertEqual(f.vname(f.default()), "base", f.name)
            self.assertIn(f.default(), list(f.grid()), f.name)

    def test_variant_names_are_unique_in_a_grid(self):
        for f in self.t.REG.values():
            names = [f.vname(pv) for pv in f.grid()]
            self.assertEqual(len(names), len(set(names)), f.name)

    def test_presets_are_valid_settings(self):
        for f in self.t.REG.values():
            for name, ov in f.presets.items():
                pv = dict(f.default(), **ov)
                self.assertTrue(set(ov) <= set(f.params), f"{f.name} {name}")
                self.assertTrue(f.valid is None or f.valid(pv), f"{f.name} {name}")

    def test_specs(self):
        fam, sets = self.t.expand_spec("hc_mm")
        self.assertEqual(sets[0], fam.default())
        self.assertEqual(len(sets), 1 + len(fam.presets))
        self.assertEqual(self.t.expand_spec("hc_mm:UD=4,POLL=16")[1], [dict(fam.default(), UD=4, POLL=16)])
        self.assertEqual(self.t.expand_spec("hc_mm:ud4")[1], [dict(fam.default(), UD=4)])
        self.assertEqual(len(self.t.expand_spec("hc_mm:all")[1]), sum(1 for _ in fam.grid()))
        with self.assertRaises(SystemExit):
            self.t.expand_spec("no_such_family")

    def test_a_decode_family_has_a_kernel_per_window_row_count(self):
        for name in ("hc_mm", "moe_gu_q4k"):
            self.assertEqual(self.t.REG[name].keys, list(range(1, 17)), name)


class HipUnit(unittest.TestCase):
    def setUp(self):
        self.t = load_tkgen()

    def test_the_unit_names_each_kernel_and_sets_only_changed_constants(self):
        fam = self.t.REG["hc_mm"]
        base = self.t.hip_tu(fam, fam.default())
        self.assertIn("#define FNX_KPART 99", base)
        self.assertIn("(void) &k_hc_mm<16>;", base)
        self.assertIn("#define FNX_TK_HM_UD 8", base)
        self.assertNotIn("FNX_MALL_NT", base)   # None: the target's own setting stays
        nt0 = self.t.hip_tu(fam, dict(fam.default(), NT=0))
        self.assertIn("#undef FNX_MALL_NT\n#define FNX_MALL_NT 0", nt0)

    def test_template_arguments_follow_the_setting(self):
        fam = self.t.REG["pf_attn_u"]
        tu = self.t.hip_tu(fam, dict(fam.default(), R=8, REG=0, LEAN=0, GW=0))
        self.assertIn("(void) &k_pf_attn_u<8, 0, 32, 0, 0, 0>;", tu)
        self.assertIn("(void) &k_pf_union<8>;", tu)

    def test_the_union_walk_site_and_source_defaults_agree(self):
        # the built-in walk is <4, true, APC, true, 0, 2> (FNX_PF_ATT_GW's default 2): the family's defaults must stay that kernel
        fam = self.t.REG["pf_attn_u"]
        self.assertEqual(fam.default(), dict(R=4, REG=1, PC=32, LEAN=1, GW=2, AT=0))
        src = (ROOT / "src" / "split" / "src" / "kernels.cu").read_text()
        self.assertIn("k_pf_attn_u<4, true, APC, true, 0, 2><<<", src)
        self.assertIn('env_int("FNX_PF_ATT_GW", 2)', src)
        # the attribution settings never pair with a gather they do not act on, and the other layouts' presets keep the first gather
        self.assertFalse(fam.valid(dict(fam.default(), AT=1)))
        self.assertFalse(fam.valid(dict(fam.default(), REG=0, LEAN=0, GW=0, AT=8)))
        self.assertTrue(fam.valid(dict(fam.default(), GW=0, AT=2)))
        for name in ("r4-plain", "reg", "pc64", "r8"):
            self.assertEqual(fam.presets[name]["GW"], 0, name)
        for macro, name in (("FNX_TK_HM_UD", "hc_mm.cuh"), ("FNX_TK_HM_POLL", "hc_mm.cuh"), ("FNX_TK_Q4K_LPR", "moe_q4k.cuh"), ("FNX_TK_Q4K_PASSES", "moe_q4k.cuh")):
            text = (ROOT / "src" / "split" / "src" / name).read_text()
            self.assertRegex(text, rf"#ifndef {macro}\n#define {macro} \d+\n#endif", macro)


class Manifest(unittest.TestCase):
    def setUp(self):
        self.t = load_tkgen()

    def entry(self, d, name, keys):
        (d / "fam").mkdir(exist_ok=True)
        (d / "fam" / f"{name}.hsaco").write_bytes(name.encode())
        return dict(params={"A": 1}, file=f"fam/{name}.hsaco", files={f"fam/{name}.hsaco": self.t.digest(name.encode())}, src="s", cache="c", cfg="cfg", bytes=1,
                    seconds=0,
                    keys={str(k): dict(symbol=f"sym{k}", threads=256, smem=0, args=[8, 4], meta=[k] + [0] * 7) for k in keys})

    def test_lines_and_composed_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            man = dict(version=1, target="t", cfg="iface=1;x=2", src="abc", use={"fam": "two"},
                       families={"fam": dict(backend="hip", path="decode", variants={"one": self.entry(d, "one", [1, 2]), "two": self.entry(d, "two", [1, 2])})})
            (d / "manifest.json").write_text(json.dumps(man))

            class A:
                out, family, name, pick = str(d), "fam", "mix", ["1=one", "2=two"]

            self.t.cmd_compose(A)
            lines = (d / "manifest.tk").read_text().splitlines()
            self.assertEqual(lines[1:4], ["tk 1", "cfg iface=1;x=2", "src abc"])
            self.assertIn("use fam two", lines)
            files = {ln.split()[2]: ln.split()[1] for ln in lines if ln.startswith("f ")}
            self.assertEqual(set(files), {"fam/one.hsaco", "fam/two.hsaco"})
            ks = [ln.split() for ln in lines if ln.startswith("k fam mix ")]
            self.assertEqual([(k[3], k[4], k[5]) for k in ks], [("1", files["fam/one.hsaco"], "sym1"), ("2", files["fam/two.hsaco"], "sym2")])
            for ln in lines:   # f id file sha256 bytes fnv64: src/tk.hpp checks the bytes it loads against them
                if ln.startswith("f "):
                    f = ln.split()
                    buf = (d / f[2]).read_bytes()
                    self.assertEqual((int(f[4]), f[5]), (len(buf), f"{self.t.fnv64(buf):016x}"))
            self.assertEqual(len([ln for ln in lines if ln.startswith("v fam ")]), 3)   # tk.hpp wants a variant's v line before its k lines
            self.assertLess(lines.index(next(ln for ln in lines if ln.startswith("v fam mix "))), lines.index(next(ln for ln in lines if ln.startswith("k fam mix "))))
            for k in ks:   # k family variant key file symbol threads smem args meta0..7
                self.assertEqual(len(k), 9 + 8)
                self.assertEqual(k[8], "8,4")
            first_k = next(i for i, ln in enumerate(lines) if ln.startswith("k "))
            last_f = max(i for i, ln in enumerate(lines) if ln.startswith("f "))
            self.assertLess(last_f, first_k)   # tk.hpp wants a kernel's file line before it

    def test_the_manifest_keeps_the_record_of_the_bytes_the_kernels_were_read_from(self):
        # a code object replaced on disk after its build: a later manifest write (use, compose) must not take the new bytes' record under the old
        # kernels' symbols and argument sizes; the loader then refuses the file (its FNV is not the manifest's)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            man = dict(version=1, target="t", cfg="c", src="s", use={}, families={"fam": dict(backend="hip", path="decode", variants={"one": self.entry(d, "one", [1])})})
            (d / "manifest.json").write_text(json.dumps(man))
            (d / "fam" / "one.hsaco").write_bytes(b"another build's object")

            class A:
                out, pick = str(d), ["fam=one"]

            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):
                self.t.cmd_use(A)
            f = next(ln.split() for ln in (d / "manifest.tk").read_text().splitlines() if ln.startswith("f "))
            self.assertEqual((int(f[4]), f[5]), (3, f"{self.t.fnv64(b'one'):016x}"))
            # an entry with no record (an older tkgen's) is refused, not hashed from the disk
            del man["families"]["fam"]["variants"]["one"]["files"]
            (d / "manifest.json").write_text(json.dumps(man))
            with self.assertRaises(SystemExit):
                self.t.cmd_use(A)

    def test_fnv64(self):
        # FNV-1a's steps from the basis src/tk.hpp and kbench use (1469598103934665603): the two sides must agree, byte for byte
        basis, prime = 1469598103934665603, 1099511628211
        self.assertEqual(self.t.fnv64(b""), basis)
        self.assertEqual(self.t.fnv64(b"a"), ((basis ^ 0x61) * prime) % (1 << 64))
        tk = (ROOT / "src" / "split" / "src" / "tk.hpp").read_text()
        self.assertIn("unsigned long long h = 1469598103934665603ull;", tk)
        self.assertIn("h = (h ^ (unsigned char) c) * 1099511628211ull;", tk)

    def test_a_coupled_family_takes_no_composed_variant(self):
        # pf_attn_u: the walk (key 0) and its lists' builder (key 1) share the rows a group; kernels of two settings must not mix
        self.assertTrue(self.t.REG["pf_attn_u"].coupled)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            man = dict(version=1, target="t", cfg="c", src="s", use={},
                       families={"pf_attn_u": dict(backend="hip", path="prefill", variants={"base": self.entry(d, "base", [0, 1]), "r8": self.entry(d, "r8", [0, 1])})})
            (d / "manifest.json").write_text(json.dumps(man))

            class A:
                out, family, name, pick = str(d), "pf_attn_u", "badmix", ["0=base", "1=r8"]

            with self.assertRaises(SystemExit):
                self.t.cmd_compose(A)

    def test_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            man = dict(version=1, target="t", cfg="c", src="s", use={},
                       families={"hc_mm": dict(backend="hip", path="decode", variants={"ud4": self.entry(d, "ud4", [1])})})
            (d / "manifest.json").write_text(json.dumps(man))

            class A:
                out, family, variants, bin = str(d), "hc_mm", None, "x"

            import contextlib
            import io

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.t.cmd_arms(A)
            self.assertEqual(buf.getvalue().splitlines(), ['"hcmm-builtin x"', f'"hcmm-ud4 x FNX_TK_DIR={d.resolve()} FNX_TK_HC_MM=ud4"'])


class CodeObjects(unittest.TestCase):
    def setUp(self):
        self.t = load_tkgen()

    def test_msgpack_reader(self):
        md = {"amdhsa.kernels": [{".name": "k", ".vgpr_count": 300, ".args": [{".offset": 8, ".size": 4, ".value_kind": "by_value"},
                                                                         {".offset": 0, ".size": 8, ".value_kind": "global_buffer"},
                                                                         {".offset": 16, ".size": 4, ".value_kind": "hidden_block_count_x"}]}],
              "amdhsa.version": [1, 2], "s": "x" * 40, "flag": True}
        got, end = self.t.msgpack(pack(md), 0)
        self.assertEqual(got, md)
        self.assertEqual(end, len(pack(md)))

    def test_kernel_arguments_leave_out_the_hidden_ones(self):
        md = {"amdhsa.kernels": [{".name": "k", ".vgpr_count": 40, ".vgpr_spill_count": 3, ".private_segment_fixed_size": 16,
                                  ".args": [{".offset": 192, ".size": 8, ".value_kind": "global_buffer"}, {".offset": 0, ".size": 192, ".value_kind": "by_value"},
                                            {".offset": 200, ".size": 4, ".value_kind": "hidden_block_count_x"}]}]}
        desc = pack(md)
        note = struct.pack("<III", 7, len(desc), 32) + b"AMDGPU\0\0" + desc + b"\0" * (-len(desc) % 4)
        # a minimal ELF64: header, the note's bytes, two section headers (null, the note)
        shoff = 64 + len(note)
        hdr = b"\x7fELF\x02\x01\x01" + b"\0" * 9 + struct.pack("<HHIQQQIHHHHHH", 3, 224, 1, 0, 0, shoff, 0, 64, 0, 0, 64, 2, 0)
        sh = b"\0" * 64 + struct.pack("<IIQQQQIIQQ", 0, 7, 0, 0, 64, len(note), 0, 0, 4, 0)
        k = self.t.Elf(hdr + note + sh).kernels()
        self.assertEqual(k, {"k": dict(args=[192, 8], vgpr=40, spill=3, scratch=16)})

    def test_unbundle(self):
        triple, elf = b"hipv4-amdgcn-amd-amdhsa--gfx1201", b"\x7fELFpayload"
        host = b"host-x86_64-unknown-linux-gnu-"
        head = self.t.BUNDLE_MAGIC + struct.pack("<Q", 2)
        entries = 24 * 2 + len(host) + len(triple)
        off = len(head) + entries
        buf = head + struct.pack("<QQQ", off, 0, len(host)) + host + struct.pack("<QQQ", off, len(elf), len(triple)) + triple + elf
        self.assertEqual(self.t.unbundle(buf), elf)
        self.assertEqual(self.t.unbundle(buf, "gfx1201"), elf)
        self.assertEqual(self.t.unbundle(elf), elf)
        with self.assertRaises(SystemExit):   # no code object of that target
            self.t.unbundle(buf, "gfx1200")

    def test_msgpack_reader_refuses_a_short_buffer(self):
        with self.assertRaises(SystemExit):
            self.t.msgpack(b"\xaaA", 0)   # a 10-byte string with one byte


def bundle(t, elf):
    triple = b"hipv4-amdgcn-amd-amdhsa--gfx1201"
    head = t.BUNDLE_MAGIC + struct.pack("<Q", 1)
    return head + struct.pack("<QQQ", len(head) + 24 + len(triple), len(elf), len(triple)) + triple + elf


def host_elf(sections):
    """A 64-bit ELF with the named sections' bytes (what Elf reads: the section table and its names)."""
    names = b"\0" + b"".join(n.encode() + b"\0" for n in sections) + b".shstrtab\0"
    body, heads, off = b"", [struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)], 64
    for n, data in list(sections.items()) + [(".shstrtab", names)]:
        heads.append(struct.pack("<IIQQQQIIQQ", names.index(n.encode() + b"\0"), 1, 0, 0, off, len(data), 0, 0, 1, 0))
        body += data
        off += len(data)
    head = bytearray(64)
    head[:5] = b"\x7fELF\x02"
    struct.pack_into("<Q", head, 0x28, off)
    struct.pack_into("<HHH", head, 0x3A, 64, len(heads), len(heads) - 1)
    return bytes(head) + body + b"".join(heads)


class Verify(unittest.TestCase):
    def setUp(self):
        self.t = load_tkgen()

    def test_asm_lines_do_not_depend_on_the_address(self):
        a = "\nDisassembly of section .text:\n\n0000000000052100 <k>:\n\ts_load_b64 s[2:3], s[0:1], 0x10 // 000000052104: F4040080\n\ts_cbranch_scc1 265 <k+0x430>\n"
        b = "0000000000071b00 <k>:\n\ts_load_b64 s[2:3], s[0:1], 0x10   // 000000071B04: F4040080\n\ts_cbranch_scc1 265 <k+0x430>\n"
        self.assertEqual(self.t.norm_asm(a), ["s_load_b64 s[2:3], s[0:1], 0x10", "s_cbranch_scc1 265"])
        self.assertEqual(self.t.norm_asm(a), self.t.norm_asm(b))

    def test_arith_ops_count_the_operations_not_their_order(self):
        one = ["v_mul_f32_e32 v1, v2, v3", "v_fmac_f32_e32 v4, v5, v6", "v_add_nc_u32_e32 v0, v0, v1", "v_cmp_lt_f32_e32 vcc_lo, v1, v2", "v_dot4_i32_iu8 v1, v2, v3, v1"]
        two = ["v_dot4_i32_iu8 v5, v2, v3, v5", "v_dual_fmac_f32 v9, v5, v6 :: v_dual_mul_f32 v7, v2, v3", "v_cmp_gt_f32_e64 s0, v2, v1"]
        self.assertEqual(self.t.arith_ops(one), [("v_cmp_ltgt_f32", 1), ("v_dot4_i32_iu8", 1), ("v_fma_f32", 1), ("v_mul_f32", 1)])
        self.assertEqual(self.t.arith_ops(one), self.t.arith_ops(two))
        # (a product and a sum in place of a multiply-add: other bits)
        self.assertNotEqual(self.t.arith_ops(one), self.t.arith_ops(["v_mul_f32_e32 v1, v2, v3", "v_mul_f32_e32 v4, v5, v6", "v_add_f32_e32 v4, v4, v7"]))

    def test_a_binary_holds_a_code_object_a_kernel_file(self):
        pad = b"\0" * 5
        fat = bundle(self.t, b"\x7fELFone") + pad + bundle(self.t, b"\x7fELFtwo")
        binary = host_elf({".text": b"\x90" * 8, ".hip_fatbin": fat})
        self.assertEqual(self.t.Elf(binary).section(".text"), b"\x90" * 8)
        self.assertIsNone(self.t.Elf(binary).section(".nope"))
        self.assertEqual([o.rstrip(b"\0") for o in self.t.fatbin_objects(binary, "gfx1201")], [b"\x7fELFone", b"\x7fELFtwo"])
        with self.assertRaises(SystemExit):
            self.t.fatbin_objects(host_elf({".text": b""}), "gfx1201")
        # a bundle's payload that holds the magic is not a second bundle: the row is walked by each bundle's extent
        inner = b"\x7fELF" + self.t.BUNDLE_MAGIC + b"tail"
        self.assertEqual(self.t.fatbin_objects(host_elf({".hip_fatbin": bundle(self.t, inner) + b"\0" * 3 + bundle(self.t, b"\x7fELFtwo")}), "gfx1201"),
                         [inner, b"\x7fELFtwo"])
        for fat in (b"garbage", b"", bundle(self.t, b"\x7fELFone")[:-4], bundle(self.t, b"\x7fELFone") + b"junk"):
            with self.assertRaises(SystemExit):
                self.t.fatbin_objects(host_elf({".hip_fatbin": fat}), "gfx1201")

    def test_an_elf_whose_tables_run_past_the_file_is_refused(self):
        good = host_elf({".text": b"\x90" * 8})
        self.assertEqual(self.t.Elf(good).section(".text"), b"\x90" * 8)
        for cut in (good[:40], good[:-10]):
            with self.assertRaises(SystemExit):
                self.t.Elf(cut)
        big = bytearray(good)   # the .text section's size past the file
        shoff, = struct.unpack_from("<Q", big, 0x28)
        struct.pack_into("<Q", big, shoff + 64 + 32, 1 << 20)
        with self.assertRaises(SystemExit):
            self.t.Elf(bytes(big))


if __name__ == "__main__":
    unittest.main()
