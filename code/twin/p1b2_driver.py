"""P1b' driver — runs C1 (night) and C2 (turbulence) exactly as pre-registered in
notes/P1b2_preregistration.md, then the held-out evaluation.

Greedy step selection on the TUNING scenes only (results/p1b2_split.json):
  config = base (C0 twin, no effects)
  for step in steps: test config+step; KEEP the step iff pooled D-hat at OP* (A>=256, variant mean) on the
  tuning scenes decreases vs the current config; otherwise drop it.
Then the final config (and the base, for reference) is run ONCE on the held-out scenes.
Every run: results/twin_<tag>/ + results/twin_discordance_<tag>.json + results/checkpoint_<C?>_<tag>.json.
Usage: python p1b2_driver.py night|turbulence [--workers N]
       python p1b2_driver.py night --workers 3 --holdout-only --profile b   (P1b'-resume, decision 28-09-2026)
       python p1b2_driver.py turbulence --summary-only                     (turbulence stopped, no held-out run)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import RESULTS  # noqa: E402

STEPS = {"night": ["a", "b", "c"], "turbulence": ["t1", "t2"]}
CK = {"night": "C1", "turbulence": "C2"}


def run(tag, profile, scenes, workers, preview=10):
    cmd = [sys.executable, str(HERE / "build_twin.py"), "v17", "--workers", str(workers), "--preview", str(preview),
           "--select", "aspect", "--profile", profile or "none", "--tag", tag, "--only", ",".join(scenes)]
    subprocess.run(cmd, check=True, cwd=str(HERE), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([sys.executable, str(HERE / "m2_discordance.py"), "full", "--tag", tag], check=True, cwd=str(HERE),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    d = json.load(open(RESULTS / f"twin_discordance_{tag}.json", encoding="utf-8"))
    a = d["by_definition"]["a256"]
    op = a["operating_point"]
    per = {s["video"]: s for s in a["per_scene"]}
    conf = d["confidence"]
    rec = dict(tag=tag, profile=profile, scenes=scenes, D_hat_op=op["D_hat"], cp95_op=op["cp95"], N_op=op["N"],
               U_fleet_op=op["c4a_fleet_U_expected"], p_hat_op=op["p_hat"], q_hat_op=op["q_hat"],
               D_op_per_scene={v: per[v]["D_OP"] for v in scenes if v in per},
               twin_conf_median=conf["twin_score_median_pooled"], real_conf_median=conf["real_score_median_pooled"],
               conf_per_scene={c["video"]: dict(twin=c["twin_score_median"], real=c["real_score_median"],
                                                twin_hit=c["twin_hit_rate"], real_hit=c["real_hit_rate"])
                               for c in conf["per_scene"]})
    return rec


def summary_from_checkpoints(group, tune, held, pfx, stopped):
    """Decision 28-09-2026 (P1b'-resume): tuning is stopped. Rebuild the greedy log from the tuning checkpoints
    already on disk (nothing is re-run); steps listed in `stopped` are recorded as not run."""
    base = json.load(open(RESULTS / f"checkpoint_{CK[group]}_{pfx}tune_base.json", encoding="utf-8"))
    log = dict(group=group, tuning=tune, held_out=held, rule="keep a step iff pooled D-hat at OP* on tuning decreases",
               base_tuning=base, steps=[], steps_not_run=stopped,
               decision_20260928="tuning stopped; remaining steps dropped (anh Dat, 28-09-2026)")
    cur_prof = ""
    for st in STEPS[group]:
        if st in stopped:
            continue
        rec = json.load(open(RESULTS / f"checkpoint_{CK[group]}_{pfx}tune_{cur_prof + st}.json", encoding="utf-8"))
        log["steps"].append(rec)
        if rec["decision"] == "KEEP":
            cur_prof = cur_prof + st
    log["final_profile"] = cur_prof
    return log


def main():
    group = sys.argv[1]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    split = json.load(open(RESULTS / "p1b2_split.json", encoding="utf-8"))["groups"][group]
    tune, held = split["tuning"], split["held_out"]
    pfx = "n" if group == "night" else "t"
    if "--holdout-only" in sys.argv:
        # P1b'-resume (28-09-2026): no further tuning; run ONLY the held-out scenes with the given profile, which
        # must equal the greedy result on the tuning checkpoints; the held-out base run is not made.
        prof = sys.argv[sys.argv.index("--profile") + 1].replace("none", "")
        log = summary_from_checkpoints(group, tune, held, pfx, stopped={"night": ["c"], "turbulence": ["t2"]}[group])
        if log["final_profile"] != prof:
            raise SystemExit(f"--profile {prof!r} != greedy result {log['final_profile']!r} on the tuning checkpoints")
        log["held_base"] = None
        log["held_final"] = run(f"{pfx}held_{prof or 'base'}", prof, held, workers)
        json.dump(log["held_final"], open(RESULTS / f"checkpoint_{CK[group]}_{pfx}held_{prof or 'base'}.json", "w"), indent=1)
        json.dump(log, open(RESULTS / f"checkpoint_{CK[group]}_summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(json.dumps(dict(final_profile=prof, held_D=log["held_final"]["D_hat_op"], N=log["held_final"]["N_op"]), indent=1))
        return
    if "--summary-only" in sys.argv:
        # turbulence (decision 28-09-2026): stopped, no held-out run; summary of the tuning checkpoints only
        log = summary_from_checkpoints(group, tune, held, pfx, stopped={"night": ["c"], "turbulence": ["t2"]}[group])
        log["held_base"] = log["held_final"] = None
        json.dump(log, open(RESULTS / f"checkpoint_{CK[group]}_summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(json.dumps(dict(final_profile=log["final_profile"]), indent=1))
        return
    log = dict(group=group, tuning=tune, held_out=held, rule="keep a step iff pooled D-hat at OP* on tuning decreases",
               steps=[])
    cur_prof = ""
    cur = run(f"{pfx}tune_base", "", tune, workers)
    json.dump(cur, open(RESULTS / f"checkpoint_{CK[group]}_{pfx}tune_base.json", "w"), indent=1)
    log["base_tuning"] = cur
    for st in STEPS[group]:
        prof = cur_prof + st
        rec = run(f"{pfx}tune_{prof}", prof, tune, workers)
        keep = rec["D_hat_op"] < cur["D_hat_op"] - 1e-12
        rec["decision"] = "KEEP" if keep else "DROP"
        rec["vs_D_hat_op"] = cur["D_hat_op"]
        json.dump(rec, open(RESULTS / f"checkpoint_{CK[group]}_{pfx}tune_{prof}.json", "w"), indent=1)
        log["steps"].append(rec)
        if keep:
            cur_prof, cur = prof, rec
    log["final_profile"] = cur_prof
    # held-out: final config once, plus the base for reference
    log["held_base"] = run(f"{pfx}held_base", "", held, workers)
    log["held_final"] = run(f"{pfx}held_{cur_prof or 'base2'}", cur_prof, held, workers) if cur_prof else log["held_base"]
    json.dump(log, open(RESULTS / f"checkpoint_{CK[group]}_summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in log.items() if k in ("final_profile",)}, indent=1))


if __name__ == "__main__":
    main()
