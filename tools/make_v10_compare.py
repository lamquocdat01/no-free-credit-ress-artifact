"""Static-domain versions at OP* (supplement table S19; Section 6 sentence).

v1.0 static domain = 78 scenes; v1.1 = 76 (CDnet cameraJitter boulevard, traffic moved to the jitter tier);
v1.2 = 48 (LASIESTA SM, simulated camera motion, moved to the jitter tier; LASIESTA MC, moving camera, excluded).
1. Recomputes each version with the CURRENT code on the corresponding rows of the same twin run (code/f1_worst_phase_site.py,
   code/m3_validity_audit.py functions; no detector, no twin run).
2. Checks v1.0 and v1.1 against the numbers published at tags ress-v1.0 / ress-v1.1 (data/results/*.json via `git show`).
3. Locates every disagreement of the earlier domains (discordant scenes, LOSO violations, false acceptances).
Output: results/v10_v11_compare.json (data/results/ in the repo); feeds supplement Table S19
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from c4_power import ucp  # noqa: E402
from common import RESULTS, dump  # noqa: E402
from e0_checks import event_rows  # noqa: E402
from f1_worst_phase_site import exchange, tier_record, worst_table  # noqa: E402
from m3_validity_audit import event_table, loso, scene_table  # noqa: E402

D_MIN, K, EPS = 32, 16, 0.20        # OP*
REPO = ROOT if (ROOT / ".git").exists() else ROOT / "artifact_repo"   # works in the project and in the repo clone
TAGS = {"v1.0": "ress-v1.0", "v1.1": "ress-v1.1"}
CDNET_JITTER = {"boulevard", "traffic"}


def tagged(tag, name):
    out = subprocess.run(["git", "-C", str(REPO), "show", f"{tag}:data/results/{name}"], capture_output=True, check=True)
    return json.loads(out.stdout.decode("utf-8"))


def block(rows):
    ev = worst_table(rows, D_MIN, K)
    t, x = tier_record(ev), exchange(ev, EPS)
    L = loso(scene_table(event_table(rows, D_MIN, K)), EPS)
    Sw = ev.groupby("video").D_w.max()
    S = ev.groupby("video").agg(p_w=("p_w", "mean"), q_w=("q_w", "mean"), w=("D_w", "max"))
    J, m = int(S.w.sum()), len(S)
    fa = sorted(v for v, r in S.iterrows() if r.q_w + float(ucp(J - int(r.w), m - 1, 0.025)) <= EPS and r.p_w > EPS)
    rec = dict(m=t["m"], N=t["N"], D_w=t["D_w"], U_fleet=t["U_fleet"], J=t["J"], U_pop=t["U_pop"],
               loso_viol=L["violation_rate_D"]["num"], n_direct=x["n_direct"], accepted=x["cameras_accepted"],
               false_accept=x["false_accept"])
    where = dict(discordant_scenes=sorted(Sw[Sw > 0].index),
                 loso_violating_scenes=sorted({r["video"] for r in L["violating_scenes"] if r["viol_D"]}),
                 false_accept_cameras=fa)
    return rec, where


def published(tag):
    wp = tagged(tag, "worst_phase.json")["points"]["OP*"]
    va = tagged(tag, "validity_audit.json")["points"]["OP*"]
    t, x = wp["tiers"]["main"], wp["exchange_main"]
    return dict(m=t["m"], N=t["N"], D_w=t["D_w"], U_fleet=t["U_fleet"], J=t["J"], U_pop=t["U_pop"],
                loso_viol=va["loso"]["violation_rate_D"]["num"], n_direct=x["n_direct"], accepted=x["cameras_accepted"],
                false_accept=x["false_accept"])


def main():
    main_, jit, mc = (event_rows("c0_main", t) for t in ("main", "jitter", "excluded_MC"))
    sm = [r for r in jit if r["video"] not in CDNET_JITTER]
    rows = {"v1.0": main_ + jit + mc, "v1.1": main_ + sm + mc, "v1.2": main_}
    out = dict(meta=dict(doc=__doc__.strip(), op=dict(d_min=D_MIN, K=K, eps=EPS), tags=TAGS,
                         removed={"v1.0->v1.1": sorted(CDNET_JITTER),
                                  "v1.1->v1.2": sorted({r["video"] for r in sm + mc})}),
               versions={}, reproduced={}, disagreements={})
    for v, r in rows.items():
        out["versions"][v], out["disagreements"][v] = block(r)
    for v, tag in TAGS.items():
        pub = published(tag)
        out["reproduced"][v] = all(abs(out["versions"][v][k] - pub[k]) < 1e-12 for k in pub)
    cur = json.loads((RESULTS / "worst_phase.json").read_text(encoding="utf-8"))["points"]["OP*"]
    out["reproduced"]["v1.2_vs_results"] = (out["versions"]["v1.2"]["N"] == cur["tiers"]["main"]["N"]
                                            and out["versions"]["v1.2"]["D_w"] == cur["tiers"]["main"]["D_w"]
                                            and out["versions"]["v1.2"]["false_accept"] == cur["exchange_main"]["false_accept"])
    out["reproduced_all"] = all(out["reproduced"].values())
    dump(out, "v10_v11_compare.json")
    print(json.dumps({k: out[k] for k in ("versions", "reproduced", "disagreements")}, indent=1, default=str))


if __name__ == "__main__":
    main()
