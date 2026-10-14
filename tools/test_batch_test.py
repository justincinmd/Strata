"""tools/batch_test.py: when the dual lanes are asked for, from every place the engine reads them (the config's "args"
and "lanes", --extra, the environment), with the engine's precedence.  A fake engine script (no GPU) speaks the
GEN / BGEN lines and reports lanes=2 in its INFO line unless told to turn them off.

    python -m unittest tools.test_batch_test
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import batch_test as BT  # noqa: E402

# The fake engine: records its arguments and STRATA_LANES, reads the lanes as the engine does (last --lanes N, then the
# larger of that and STRATA_LANES), says lanes=2 unless FAKE_LANES_OFF=1, batch_slots = the last --batch.  GEN answers
# two tokens; BGEN <slot> answers one, admits the slot (BADM 1), then one BT and BDONE.
FAKE = r'''import json, os, re, sys
args = sys.argv[1:]
def atoi(x):
    m = re.match(r"\s*([+-]?\d+)", x or "")
    return int(m.group(1)) if m else 0
lanes = 0
batch = 0
for i, x in enumerate(args[:-1]):
    if x == "--lanes":
        lanes = atoi(args[i + 1])
    if x == "--batch":
        batch = atoi(args[i + 1])
if os.environ.get("STRATA_LANES"):
    lanes = max(lanes, atoi(os.environ["STRATA_LANES"]))
on = lanes >= 2 and os.environ.get("FAKE_LANES_OFF") != "1"
gens = []
def record():
    with open(os.environ["FAKE_REC"], "w") as f:
        json.dump({"args": args, "env_lanes": os.environ.get("STRATA_LANES"), "gens": gens}, f)
record()
print("INFO engine=0.1.43" + (f" batch_slots={batch}" if batch >= 2 else "") + (" lanes=2" if on else ""), flush=True)
print("READY 4096 stop", flush=True)
for line in sys.stdin:
    f = line.split()
    if not f or f[0] == "QUIT":
        break
    gens.append(f[0])
    record()
    if f[0] == "GEN":
        print("T 65", flush=True)
        print("T 66", flush=True)
        print("DONE 2 10 1.0 1.0 length 0 0 0", flush=True)
    elif f[0] == "BGEN":
        b = f[1]
        print("T 65", flush=True)
        print(f"BADM {b} 1", flush=True)
        print(f"BT {b} 66", flush=True)
        print(f"BDONE {b} 2 length 1.0", flush=True)
'''


class StubTokenizer:
    def encode(self, text, parse_special=False):
        return list(text.encode("utf-8"))[:16]

    def decode(self, ids):
        return bytes(i & 255 for i in ids).decode("utf-8", "replace")


@unittest.skipIf(os.name == "nt", "the fake engine is started as a script with a shebang")
class LanesIntent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.exe = d / "fake_engine"
        self.exe.write_text(f"#!{sys.executable}\n{FAKE}")
        self.exe.chmod(self.exe.stat().st_mode | stat.S_IXUSR)
        self.rec = d / "rec.json"
        self.dir = d
        env = {k: v for k, v in os.environ.items() if k not in ("STRATA_LANES", "FAKE_LANES_OFF")}
        env.update(FAKE_REC=str(self.rec), BATCH_TEST_LOG=str(d / "engine.log"))
        self.env = mock.patch.dict(os.environ, env, clear=True)
        self.env.start()
        self.tok = mock.patch.object(BT, "tokenizer", lambda path: StubTokenizer())
        self.tok.start()

    def tearDown(self):
        self.tok.stop()
        self.env.stop()
        self.tmp.cleanup()

    def run_main(self, cfg, *argv):
        cfg = {"tokenizer": "unused", **cfg}
        path = self.dir / "cfg.json"
        path.write_text(json.dumps(cfg))
        with mock.patch.object(sys, "argv", ["batch_test.py", "--exe", str(self.exe), "--config", str(path),
                                             "--max-new", "2", "--mt-min", "", *argv]):
            with contextlib.redirect_stdout(io.StringIO()) as self.out, warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)   # (batch_test leaves its engine log to the exit)
                rc = BT.main()
        return rc, json.loads(self.rec.read_text()) if self.rec.exists() else None

    def assert_lanes_run(self, rc, rec):
        self.assertEqual(rc, 0)
        a = rec["args"]
        self.assertEqual(a[len(a) - 1 - a[::-1].index("--batch") + 1], "2", a)   # --batch 2 by default
        self.assertEqual(rec["gens"].count("BGEN"), 2, rec)                      # --n 2 by default

    def test_lanes_in_the_config_args(self):
        """The review's case: "--lanes", "2" in the config's "args" sets the lanes defaults."""
        self.assert_lanes_run(*self.run_main({"args": ["--lanes", "2"]}))

    def test_config_args_lanes_turned_off_by_the_engine_fail(self):
        """The review's repro: lanes in the config's "args" with --batch 2 --n 2 and an engine that turns them off
        used to pass (both slots IDENTICAL); now the missing INFO lanes=2 fails the test (exit 3)."""
        os.environ["FAKE_LANES_OFF"] = "1"
        rc, rec = self.run_main({"args": ["--lanes", "2"]}, "--batch", "2", "--n", "2")
        self.assertEqual(rc, 3)
        self.assertNotIn("BGEN", rec["gens"])

    def test_config_args_lanes_refuse_four_slots(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_main({"args": ["--lanes", "2"]}, "--batch", "4")
        self.assertIn("--batch 2 --n 2", str(cm.exception))

    def test_top_level_lanes(self):
        rc, rec = self.run_main({"args": [], "lanes": 2})
        self.assert_lanes_run(rc, rec)
        self.assertIn("--lanes", rec["args"])

    def test_top_level_lanes_turned_off_by_the_engine_fail(self):
        os.environ["FAKE_LANES_OFF"] = "1"
        self.assertEqual(self.run_main({"args": [], "lanes": 2})[0], 3)

    def test_lanes_in_extra(self):
        self.assert_lanes_run(*self.run_main({"args": []}, "--extra", "--lanes 2"))

    def test_lanes_from_env_option(self):
        rc, rec = self.run_main({"args": []}, "--env", "STRATA_LANES=2")
        self.assert_lanes_run(rc, rec)
        self.assertEqual(rec["env_lanes"], "2")

    def test_lanes_from_the_shell(self):
        os.environ["STRATA_LANES"] = "2"
        self.assert_lanes_run(*self.run_main({"args": []}))

    def test_lanes_from_the_config_env_with_config_env(self):
        rc, rec = self.run_main({"args": [], "env": {"STRATA_LANES": "2"}}, "--config-env")
        self.assert_lanes_run(rc, rec)
        self.assertEqual(rec["env_lanes"], "2")

    def test_the_config_env_without_config_env_is_not_asked(self):
        """Without --config-env the engine does not get the config's env block, so no lanes are asked for."""
        rc, rec = self.run_main({"args": [], "env": {"STRATA_LANES": "2"}})
        self.assertEqual(rc, 0)
        self.assertEqual(rec["gens"].count("BGEN"), 4)

    def test_the_last_lanes_flag_wins(self):
        """As the engine: a later --lanes 0 in --extra overrides the config's --lanes 2 (four-slot defaults, no lanes
        check); STRATA_LANES=2 still asks for them (the engine takes the larger)."""
        rc, rec = self.run_main({"args": ["--lanes", "2"]}, "--extra", "--lanes 0")
        self.assertEqual(rc, 0)
        self.assertEqual(rec["gens"].count("BGEN"), 4)
        os.environ["STRATA_LANES"] = "2"
        self.assert_lanes_run(*self.run_main({"args": ["--lanes", "2"]}, "--extra", "--lanes 0"))

    def test_no_lanes(self):
        rc, rec = self.run_main({"args": []})
        self.assertEqual(rc, 0)
        self.assertEqual(rec["gens"].count("BGEN"), 4)
        self.assertNotIn("--lanes", rec["args"])


class LanesWanted(unittest.TestCase):
    def test_precedence_and_atoi(self):
        self.assertTrue(BT.lanes_wanted(["--lanes", "2"], {}))
        self.assertFalse(BT.lanes_wanted(["--lanes", "2", "--lanes", "1"], {}))
        self.assertTrue(BT.lanes_wanted(["--lanes", "0"], {"STRATA_LANES": "2"}))
        self.assertFalse(BT.lanes_wanted([], {"STRATA_LANES": ""}))
        self.assertTrue(BT.lanes_wanted(["--lanes", "2x"], {}))         # atoi: the leading digits
        self.assertFalse(BT.lanes_wanted(["--lanes"], {}))              # no value: nothing read

    def test_engine_args_order(self):
        cfg = {"args": ["--lanes", "0"], "lanes": 2, "gpu": [0, 1], "layer_split": "26"}
        self.assertEqual(BT.engine_args(cfg, 2, ["--x"]),
                         ["--lanes", "0", "--layer-split", "26", "--batch", "2", "--x"])
        self.assertFalse(BT.lanes_wanted(BT.engine_args(cfg, None, []), {}))   # the config's own --lanes 0 stays


if __name__ == "__main__":
    unittest.main()
