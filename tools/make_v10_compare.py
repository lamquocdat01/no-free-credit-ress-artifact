"""v1.0 / v1.1 comparison at OP* (supplement table; §6 sentence).

v1.0 static domain = 78 scenes = v1.1 static domain (76) + camera-jitter tier (boulevard, traffic).
1. Recomputes the v1.0 numbers with the CURRENT code on the rows main + jitter (code/f1_worst_phase_site.py,
   code/m3_validity_audit.py functions; no detector, no twin run).
2. Checks them against the numbers published at tag ress-v1.0 (data/results/*.json of the repo, via `git show`).
3. Locates every v1.0 disagreement (discordant events/scenes, LOSO violations, false acceptances) by tier.
Output: results/v10_v11_compare.json (data/results/ in the repo); feeds supplement Table S19
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from common import RESULTS, dump  # noqa: E402
from e0_checks import event_rows  # noqa: E402
from f1_worst_phase_site import exchange, tier_record, worst_table  # noqa: E402
from m3_validity_audit import event_table, loso, scene_table  # noqa: E402

D_MIN, K, EPS = 32, 16, 0.20        # OP*
REPO = ROOT if (ROOT / ".git").exists() else ROOT / "artifact_repo"   # works in the project and in the repo clone
TAG10 = "ress-v1.0"


def tagged(tag, name):
    out = subprocess.run(["git", "-C", str(REPO), "show", f"{tag}:data/results/{name}"], capture_output=True, check=True)
    return json.loads(out.stdout.decode("utf-8"))


def block(rows_main, rows_jit):
    rows = rows_main + rows_jit
    jit = {r["video"] for r in rows_jit}
    ev = worst_table(rows, D_MIN, K)
    t, x = tier_record(ev), exchange(ev, EPS)
    L = loso(scene_table(event_table(rows, D_MIN, K)), EPS)
    Sw = ev.groupby("video").D_w.max()
    disc_scenes = sorted(Sw[Sw > 0].index)
    viol = sorted({r["video"] for r in L["violating_scenes"] if r["viol_D"]})
    # false acceptances: recompute per camera (exchange() returns counts only)
    S = ev.groupby("video").agg(p_w=("p_w", "mean"), q_w=("q_w", "mean"), w=("D_w", "max"))
    from c4_power import ucp  # noqa: E402
    J, m = int(S.w.sum()), len(S)
    fa = sorted(v for v, r in S.iterrows() if r.q_w + float(ucp(J - int(r.w), m - 1, 0.025)) <= EPS and r.p_w > EPS)
    rec = dict(m=t["m"], N=t["N"], D_w=t["D_w"], U_fleet=t["U_fleet"], J=t["J"], U_pop=t["U_pop"],
               loso_viol=L["violation_rate_D"]["num"], n_direct=x["n_direct"], accepted=x["cameras_accepted"],
               false_accept=x["false_accept"])
    where = dict(discordant_events_in_jitter=int(ev[ev.video.isin(jit)].D_w.sum()),
                 discordant_scenes=disc_scenes, loso_violating_scenes=viol, false_accept_cameras=fa,
                 all_in_jitter=bool(set(disc_scenes) | set(viol) | set(fa) <= jit))
    return rec, where


def main():
    rows_main, rows_jit = event_rows("c0_main", "main"), event_rows("c0_main", "jitter")
    v10, where10 = block(rows_main, rows_jit)
    v11, where11 = block(rows_main, [])

    wp = tagged(TAG10, "worst_phase.json")["points"]["OP*"]
    va = tagged(TAG10, "validity_audit.json")["points"]["OP*"]
    pub = dict(m=wp["tiers"]["main"]["m"], N=wp["tiers"]["main"]["N"], D_w=wp["tiers"]["main"]["D_w"],
               U_fleet=wp["tiers"]["main"]["U_fleet"], J=wp["tiers"]["main"]["J"], U_pop=wp["tiers"]["main"]["U_pop"],
               loso_viol=va["loso"]["violation_rate_D"]["num"], n_direct=wp["exchange_main"]["n_direct"],
               accepted=wp["exchange_main"]["cameras_accepted"], false_accept=wp["exchange_main"]["false_accept"])
    match = {k: (abs(v10[k] - pub[k]) < 1e-12) for k in pub}

    cur = json.loads((RESULTS / "worst_phase.json").read_text(encoding="utf-8"))["points"]["OP*"]
    match11 = dict(N=v11["N"] == cur["tiers"]["main"]["N"], D_w=v11["D_w"] == cur["tiers"]["main"]["D_w"],
                   false_accept=v11["false_accept"] == cur["exchange_main"]["false_accept"])

    out = dict(meta=dict(doc=__doc__.strip(), op=dict(d_min=D_MIN, K=K, eps=EPS), tag_v10=TAG10,
                         jitter_scenes=sorted({r["video"] for r in rows_jit if r["duration"] >= D_MIN})),
               v10_recomputed=v10, v10_published=pub, v10_reproduced=match, v10_reproduced_all=all(match.values()),
               v11=v11, v11_matches_results=match11, v10_disagreements=where10, v11_disagreements=where11)
    dump(out, "v10_v11_compare.json")
    print(json.dumps({k: out[k] for k in ("v10_recomputed", "v10_reproduced_all", "v11", "v11_matches_results",
                                          "v10_disagreements")}, indent=1, default=str))


if __name__ == "__main__":
    main()
