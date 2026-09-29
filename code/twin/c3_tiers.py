"""P1b' C3 — discordance by tier and the C4 bounds for the main domain and for everything.

Tiers (results/p1_background17.json): main (static camera, not night/turbulence), night, turbulence;
PTZ excluded. Per tier: n scenes, n events at OP*, pooled D-hat [CP95], C4a-fleet U, C4a-pop U,
detector confidence twin vs real, and the pre-registered criterion (median per-scene D-hat on the HELD-OUT
scenes <= 5%; per-scene CP95 reported; the "median CP upper bound" reading reported as well).
Sources: main = C0 run (results/twin_discordance_c0_main.json);
night / turbulence = final configuration of C1 / C2 (checkpoint_C?_summary.json): held-out run + tuning run
of the same configuration; baseline (no effects) runs for comparison. P1b daytime D for the +0.5 pt check.
P1b'-resume (decision 28-09-2026): night = profile b on the 2 held-out scenes (no held-out base run);
turbulence = existing tuning base only (no held-out run), flagged n < 10 at OP* (not certifiable).
Output: results/twin_tiers.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from c4_power import ucp  # noqa: E402
from common import RESULTS, cp_ci, dump  # noqa: E402

EPS = 0.20


def m2(tag):
    subprocess.run([sys.executable, str(HERE / "m2_discordance.py"), "full", "--tag", tag], check=True, cwd=str(HERE),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.load(open(RESULTS / f"twin_discordance_{tag}.json", encoding="utf-8"))


def pooled(parts):
    """Combine disjoint scene sets at OP*: X, N, J, m, per-scene rows, confidence."""
    X = sum(p["by_definition"]["a256"]["operating_point"]["X_expected"] for p in parts)
    N = sum(p["by_definition"]["a256"]["operating_point"]["N"] for p in parts)
    J = sum(p["by_definition"]["a256"]["operating_point"]["J_possible"] for p in parts)
    m = sum(p["by_definition"]["a256"]["operating_point"]["m_scenes"] for p in parts)
    rows = [r for p in parts for r in p["by_definition"]["a256"]["per_scene"] if r.get("n_OP")]
    ts = [c for p in parts for c in p["confidence"]["per_scene"]]
    tw = [c["twin_score_median"] for c in ts if c["twin_score_median"] is not None]
    rl = [c["real_score_median"] for c in ts if c["real_score_median"] is not None]
    return dict(n_scenes=m, n_events_OP=N, X_expected=X, D_hat=X / N if N else None, cp95=cp_ci(X, N) if N else None,
                c4a_fleet_U=float(ucp(X, N)) if N else None, J=J, c4a_pop_U=float(ucp(J, m)) if m else None,
                per_scene=[dict(video=r["video"], n_OP=r["n_OP"], D_OP=r["D_OP"], cp95_OP=r["cp95_OP"]) for r in rows],
                median_scene_D=float(np.median([r["D_OP"] for r in rows])) if rows else None,
                median_scene_cp_upper=float(np.median([r["cp95_OP"][1] for r in rows])) if rows else None,
                twin_conf_scene_median=float(np.median(tw)) if tw else None,
                real_conf_scene_median=float(np.median(rl)) if rl else None,
                G2=dict(fleet_ok=bool(N and ucp(X, N) <= EPS / 2), pop_ok=bool(m and ucp(J, m) <= EPS / 2)))


def tier_from_driver(ck):
    s = json.load(open(RESULTS / f"checkpoint_{ck}_summary.json", encoding="utf-8"))
    prof = s["final_profile"]
    pfx = "n" if ck == "C1" else "t"
    tune_tag = f"{pfx}tune_{prof}" if prof else f"{pfx}tune_base"
    tune, tb = m2(tune_tag), m2(f"{pfx}tune_base")
    steps = [dict(profile=r["profile"], decision=r["decision"], D_hat_op=r["D_hat_op"], vs=r["vs_D_hat_op"],
                  twin_conf=r["twin_conf_median"], real_conf=r["real_conf_median"]) for r in s["steps"]]
    rec = dict(final_profile=prof, steps_on_tuning=steps, steps_not_run=s.get("steps_not_run", []),
               decision_20260928=s.get("decision_20260928"), base_tuning_D=s["base_tuning"]["D_hat_op"],
               tuning_final=pooled([tune]), tuning_base=pooled([tb]))
    # P1b'-resume (28-09-2026): held-out base is no longer run (night); turbulence has no held-out run at all
    held = m2(s["held_final"]["tag"]) if s.get("held_final") else None
    hb = m2(s["held_base"]["tag"]) if s.get("held_base") else None
    parts = [held, tune] if held else [tune]
    rec.update(held_out=pooled([held]) if held else None, held_out_base=pooled([hb]) if hb else None,
               all_final=pooled(parts), all_base=pooled([hb, tb]) if hb else None,
               criterion_median_heldout_D_le_5pct=(bool(pooled([held])["median_scene_D"] is not None
                                                        and pooled([held])["median_scene_D"] <= 0.05) if held else None))
    n_op = rec["all_final"]["n_events_OP"]
    rec["n_events_OP_lt_10"] = bool(n_op < 10)
    rec["status"] = ("n < 10 at OP*: not certifiable (tier reported for its measured D only)" if n_op < 10 else
                     "measured tier, outside the main domain")
    return rec, parts


def main():
    main_ = m2("c0_main")
    p1b = json.load(open(RESULTS / "twin_discordance.json", encoding="utf-8"))
    cats = p1b["by_definition"]["a256"]["operating_point_by_category"]
    day_keys = [k for k in cats if k not in ("nightVideos", "turbulence", "PTZ")]
    Np = sum(cats[k]["N"] for k in day_keys)
    D_p1b_day = sum(cats[k]["D_hat"] * cats[k]["N"] for k in day_keys) / Np
    out = dict(meta=dict(op="OP* (eps 20%, delta 5%, d_min 32, K 16), definition A>=256, variant mean",
                         G2="U <= eps/2 = 10% (fleet and pop)", preregistration="notes/P1b2_preregistration.md"))
    M = pooled([main_])
    M["c0_geometry"] = main_["confidence"]["geometry"]
    M["c0_geometry_by_kind"] = main_["confidence"].get("geometry_by_kind")
    M["D_p1b_daytime_same_categories"] = D_p1b_day
    M["delta_vs_p1b_pts"] = 100 * (M["D_hat"] - D_p1b_day)
    M["day_ok_plus_0_5pt"] = bool(M["delta_vs_p1b_pts"] <= 0.5)
    M["by_category"] = main_["by_definition"]["a256"].get("operating_point_by_category")
    M["by_kind"] = main_["by_definition"]["a256"].get("operating_point_by_kind")
    out["main"] = M
    night, nparts = tier_from_driver("C1")
    turb, tparts = tier_from_driver("C2")
    out["night"], out["turbulence"] = night, turb
    out["all_tiers_final"] = pooled([main_] + nparts + tparts)
    out["all_tiers_final_note"] = "main (C0) + night (held-out profile b + tuning profile b) + turbulence (tuning base only)"
    out["criterion"] = dict(night=night["criterion_median_heldout_D_le_5pct"], turbulence=turb["criterion_median_heldout_D_le_5pct"],
                            day=M["day_ok_plus_0_5pt"],
                            tunable=False,
                            note="decision 28-09-2026: night and turbulence tuning stopped -> both declared OUTSIDE the "
                                 "main domain regardless of the criterion; turbulence has no held-out run (None)")
    dump(out, "twin_tiers.json")
    print(json.dumps({k: out[k] for k in ("criterion",)}, indent=1))
    print("main", {k: M[k] for k in ("n_scenes", "n_events_OP", "D_hat", "c4a_fleet_U", "J", "c4a_pop_U", "delta_vs_p1b_pts")})


if __name__ == "__main__":
    main()
