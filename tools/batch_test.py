#!/usr/bin/env python3
"""Batching test: the same prompts decoded alone (GEN, the usual path with drafts) and together in the batch
windows (BGEN), greedy.  Every slot's tokens must equal its solo tokens; prints the aggregate decode rate.

  python3 tools/batch_test.py --exe engine/strata --config strata-<model>.json --batch 4 --n 4 \
      --extra "--layer-split 12,24,36 --trim-stage-weights --pcie-frac 0 --adapt-every 1000000"

The same test covers the dual lanes (docs/BATCHING.md "Dual lanes"): each slot then decodes as a solo request.  The
lanes are asked for by a config with "lanes": 2 (as the server), by --lanes 2 in --extra, or by --env STRATA_LANES=2.
Then --batch and --n default to 2, and the test fails (exit 3) unless the engine's INFO line says lanes=2 (an engine
that turns the lanes off says why in its log: "WARNING: --lanes is off: ...").  For a two-GPU config:

  python3 tools/batch_test.py --exe build/strata --config strata-<model>-lanes.json --config-env --max-new 1000 \
      --extra "--no-prefill-borrow --pcie-frac 0 --adapt-every 1000000 --kv-resident 32768"

(--no-prefill-borrow holds back more VRAM on the later stage; on 2x 32 GB cards --kv-resident 32768 keeps every expert
resident, which the pipelined lanes need: the log's "strata lanes: GO" line.)  In every mode the prompt count must not
exceed the engine's batch_slots.
"""
import argparse, json, os, subprocess, sys, tempfile, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_tokenizer as ST  # noqa: E402

QUESTIONS = [
    "Explique en detail le fonctionnement d'un B-tree.",
    "Write a Python function that merges overlapping intervals, then explain it.",
    "Compare TCP and QUIC: handshake, congestion control, multiplexing.",
    "Raconte l'histoire du calcul de pi, d'Archimede a nos jours.",
    "What are the main causes of the French Revolution?",
    "Explain how a transformer's attention works, step by step.",
    "Donne une recette de tarte aux pommes, etape par etape.",
    "Describe the life cycle of a star like the Sun.",
]


def tokenizer(path):
    t = Path(path)
    vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for s, i in vocab.items():
        tokens[i] = s
    merges = (t / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((t / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)


class Engine:
    def __init__(self, exe, cfg, batch, extra_env, extra_args):
        args = list(cfg["args"])
        if len(cfg.get("gpu") or []) > 1:
            args += ["--layer-split", str(cfg.get("layer_split") or "auto")]
        if batch:
            args += ["--batch", str(batch)]
        if cfg.get("lanes") == 2 and "--lanes" not in args:   # the server's "lanes": 2
            args += ["--lanes", "2"]
        args += extra_args
        env = dict(os.environ, **extra_env)
        env["LD_LIBRARY_PATH"] = ":".join(cfg.get("lib_dirs", []) + [env.get("LD_LIBRARY_PATH", "")])
        if os.name == "nt":
            env["PATH"] = os.pathsep.join(cfg.get("lib_dirs", []) + [env.get("PATH", "")])
        self.log_path = os.environ.get("BATCH_TEST_LOG") or os.path.join(tempfile.gettempdir(), "batch_test_engine.log")
        self.log = open(self.log_path, "w")
        self.p = subprocess.Popen([exe, "--serve", *args], cwd=cfg.get("cwd"), stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1, env=env)
        self.info = {}      # the INFO line's key=value fields (batch_slots, batch_groups, lanes, ...)
        for line in self.p.stdout:
            if line.startswith("INFO "):
                self.info.update(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
            if line.startswith("READY"):
                return
        raise SystemExit(f"the engine ended before READY - see {self.log_path}")

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def lines(self):
        for line in self.p.stdout:
            yield line.rstrip("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, default=None, help="batch slots (default 4; 2 when the lanes are asked for)")
    ap.add_argument("--n", type=int, default=None,
                    help="prompts (default: --batch; at most the engine's batch_slots, which the test checks)")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--skip-solo", action="store_true")
    ap.add_argument("--keys", default="", help='sampling keys for every request, e.g. "temperature=0.7 top_k=20"')
    ap.add_argument("--extra", default="", help='more engine arguments in one string, e.g. "--adapt-every 1000000"')
    ap.add_argument("--dump", default="", help="write every slot's tokens (solo and batch) as JSON: an A/B of two engine builds")
    ap.add_argument("--long-tokens", type=int, default=0,
                    help="the LAST prompt is read from about this many tokens of filler text first: its chunks run "
                         "beside the earlier slots' windows (the prompt path lends experts then)")
    ap.add_argument("--mt-min", default="1", help="STRATA_IQ_MT_MIN for the engine (1: exact; empty: the default)")
    ap.add_argument("--config-env", action="store_true",
                    help="start the engine with the config's \"env\" block too (as the server does); the test's own "
                         "variables (STRATA_DECODE_TIMING, STRATA_IQ_MT_MIN) still win")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="one more engine environment variable (repeatable), e.g. --env STRATA_LANES=2 for the dual lanes")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    env = {}
    if a.config_env:
        env.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    # the dual lanes are asked for (as the engine reads it: --lanes N or STRATA_LANES=N, N >= 2; the config's "lanes": 2)
    extra = a.extra.split()
    def as_int(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return 0
    lanes_flag = max([as_int(extra[i + 1]) for i, x in enumerate(extra[:-1]) if x == "--lanes"] or [0])
    lanes_env = as_int(env.get("STRATA_LANES", os.environ.get("STRATA_LANES")))
    want_lanes = cfg.get("lanes") == 2 or lanes_flag >= 2 or lanes_env >= 2
    if a.batch is None:
        a.batch = 2 if want_lanes else 4
    if a.n is None:
        a.n = a.batch
    if want_lanes and (a.batch != 2 or a.n > 2):
        raise SystemExit(f"lanes: two slots, one request per lane - run with --batch 2 --n 2 (got --batch {a.batch} "
                         f"--n {a.n})")
    if a.n > a.batch or a.n > len(QUESTIONS):
        raise SystemExit(f"--n {a.n}: at most --batch ({a.batch}) and {len(QUESTIONS)} prompts")
    tok = tokenizer(cfg["tokenizer"])
    prompts = []
    for q in QUESTIONS[: a.n]:
        text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        if a.long_tokens and q is QUESTIONS[a.n - 1]:
            para = ("The committee reviewed item %d of the long agenda, noting the budget, the schedule and the open "
                    "risks, and asked the staff to report back next quarter. ")
            per = max(1, len(tok.encode(para % 100)))
            filler = "".join(para % i for i in range(a.long_tokens // per + 1))
            text = f"<|im_start|>user\n{filler}\n\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prompts.append(tok.encode(text, parse_special=True))
    env.update({"STRATA_DECODE_TIMING": "1", **({"STRATA_VERIFY_PROFILE": "1"} if os.environ.get("PROF") else {})})
    if a.mt_min:
        env["STRATA_IQ_MT_MIN"] = a.mt_min
    eng = Engine(a.exe, cfg, a.batch, env, extra)
    out = eng.lines()

    def refuse(why):
        print(f"FAILED: {why} (engine log: {eng.log_path})", flush=True)
        eng.send("QUIT")
        eng.p.wait(timeout=180)
        return 3
    slots = as_int(eng.info.get("batch_slots"))
    print(f"engine: batch_slots={slots} batch_groups={eng.info.get('batch_groups', '-')} "
          f"lanes={eng.info.get('lanes', '0')}", flush=True)
    if len(prompts) > slots:
        return refuse(f"{len(prompts)} prompts but the engine runs {slots} batch slots (INFO batch_slots)")
    if want_lanes and eng.info.get("lanes") != "2":
        return refuse("the lanes were asked for but the engine's INFO line has no lanes=2: it turned them off "
                      "(see its \"WARNING: --lanes is off\" line)")

    solo = []
    if not a.skip_solo:
        for i, ids in enumerate(prompts):
            eng.send(" ".join(x for x in ("GEN", str(a.max_new), a.keys, ",".join(map(str, ids))) if x))
            got, t0 = [], time.time()
            for line in out:
                if line.startswith("T "):
                    got.append(int(line.split()[1]))
                elif line.startswith("DONE") or line.startswith("ERR"):
                    print(f"solo {i}: {len(got)} tokens in {time.time() - t0:.1f}s  {line[:60]}", flush=True)
                    break
            solo.append(got)

    # batch: admit every prompt, then the slots decode together
    got = {i: [] for i in range(len(prompts))}
    done = {}
    t_admit = time.time()
    first_bt = None
    for i, ids in enumerate(prompts):
        eng.send(" ".join(x for x in ("BGEN", str(i), str(a.max_new), a.keys, ",".join(map(str, ids))) if x))
        for line in out:
            if line.startswith("T "):
                got[i].append(int(line.split()[1]))
            elif line.startswith("BT "):
                _, s, y = line.split()
                got[int(s)].append(int(y))
                first_bt = first_bt or time.time()
            elif line.startswith("BDONE "):
                done[int(line.split()[1])] = line
            elif line.startswith("ERR"):
                print("batch:", line)
                return 1
            elif line.startswith("BADM "):
                if line.split()[2] == "0":
                    done[i] = line
                break
    t_admitted = time.time()
    for line in out:
        if line.startswith("BT "):
            _, s, y = line.split()
            got[int(s)].append(int(y))
            first_bt = first_bt or time.time()
        elif line.startswith("BDONE "):
            done[int(line.split()[1])] = line
            if len(done) == len(prompts):
                break
        elif line.startswith("ERR"):
            print("batch:", line)
            return 1
    t_end = time.time()
    total = sum(len(v) for v in got.values())
    print(f"batch: {len(prompts)} slots, {total} tokens; admissions {t_admitted - t_admit:.1f}s; "
          f"then {t_end - t_admitted:.1f}s -> aggregate {total / max(t_end - t_admit, 1e-9):.1f} tok/s overall, "
          f"{sum(len(v) for v in got.values()) / max(t_end - (first_bt or t_admit), 1e-9):.1f} tok/s from the first batch token",
          flush=True)
    ok = True
    for i in range(len(prompts)):
        b = got[i]
        if solo:
            same = b == solo[i]
            ok &= same
            first_diff = next((k for k in range(min(len(b), len(solo[i]))) if b[k] != solo[i][k]), None)
            print(f"slot {i}: {len(b)} tokens, solo {len(solo[i])}: {'IDENTICAL' if same else f'DIFFERS at {first_diff}'}")
        print("   ", ascii(tok.decode(b)[:160]))
    if a.dump:
        Path(a.dump).write_text(json.dumps({"solo": solo, "batch": [got[i] for i in range(len(prompts))]}), encoding="utf-8")
    eng.send("QUIT")
    eng.p.wait(timeout=180)   # the next run needs the GPUs back
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
