"""B1 — A1/A2/A4 under two event definitions and the operating-point rule of G1b.

Definitions
  a256 (PRIMARY, G1b-1b): Mode-G* gap1 events whose largest GT object (largest 8-connected component;
       LASIESTA: largest colour instance), taken at the event's largest frame, has area >= 256 px
       (column L_area_max >= 256).
  orig (sensitivity): every event with >= 1 GT pixel == 255 (P1a definition).
Data: CDnet (A1, gap1), LASIESTA frame-level (A4; temporal order verified in B0), BMC-synth (A4).
A1 does not need a re-run: the per-event table already carries L_area_max; B1 filters it.
OP rule (G1b, applied mechanically, no re-asking): among cells with K <= d_min, Mode G*, definition a256,
CDnet (the 44-scene calibration population is CDnet), choose the smallest eps in the A2 grid
{5, 10, 20}% such that n_required(80% power, at p_hat) <= 150 AND C4a-pop with 44 scenes gives
U <= eps/2; ties -> larger K, then smaller d_min, then smaller n. If nothing beats OP-A (20%, 5%, 32, 8),
keep OP-A. Bootstrap B = 1000 by scene (= video), seed 42.
Outputs: results/miss_matrix_Gstar_a256.csv, results/m1_gstar_summary_v2.json,
         results/operating_points_v2.json, results/checkpoint_B1.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from a2_operating_points import D_MIN, EPS_GRID, cell_stats, n_required, power
from c4_power import ucp
from common import B_BOOT, K_GRID, RESULTS, SEED, dump, n0

A_MIN = 256
N_CAP = 150
EPS_FINE = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20]   # sensitivity only; the rule uses EPS_GRID


def load_events():
    C = pd.read_csv(RESULTS / "miss_matrix_Gstar.csv")
    C = C[C.merge_gap == 1].assign(dataset="CDnet2014")
    X = pd.read_csv(RESULTS / "miss_matrix_Gstar_ext.csv")
    X = X[X.level == "frame"]
    cols = ["dataset", "video", "event_id", "onset_frame", "duration", "n_fg", "n_hit", "L_area_max", "cls",
            "cls_group"] + [f"pGs_K{K}" for K in K_GRID]
    E = pd.concat([C[cols + [f"pG_K{K}" for K in K_GRID]], X[cols]], ignore_index=True)
    E["def_a256"] = E.L_area_max >= A_MIN
    return E


def table(D, col, rng, eps_grid):
    tab = {}
    for d in D_MIN:
        sub = D[D.duration >= d]
        row = {}
        for K in K_GRID:
            st = cell_stats(sub[f"{col}_K{K}"].values, sub.video.values, rng) if len(sub) else None
            if st is None:
                continue
            st["K_le_dmin"] = K <= d
            st["n_required"] = {f"eps{e}": dict(at_p_hat=n_required(st["p_hat"], e),
                                                at_p_upper=n_required(st["boot95_video"][1], e))
                                for e in eps_grid}
            row[f"K{K}"] = st
        tab[f"d{d}"] = dict(n_events=len(sub), n_videos=int(sub.video.nunique()), cells=row)
    return tab


def apply_rule(tab, eps_grid, u_pop):
    cands = []
    for d, r in tab.items():
        for k, c in r["cells"].items():
            if not c["K_le_dmin"]:
                continue
            for e in eps_grid:
                n = c["n_required"][f"eps{e}"]["at_p_hat"]
                if n is not None and n <= N_CAP and u_pop <= e / 2:
                    cands.append(dict(eps=e, d_min=int(d[1:]), K=int(k[1:]), n_required=n, p_hat=c["p_hat"],
                                      boot95_video=c["boot95_video"], n_events=c["n_events"]))
    cands.sort(key=lambda x: (x["eps"], -x["K"], x["d_min"], x["n_required"]))
    return cands


def main():
    E = load_events()
    E.to_csv(RESULTS / "miss_matrix_Gstar_a256.csv", index=False)
    rng = np.random.default_rng(SEED)
    bg = json.load(open(RESULTS / "p1_background17.json", encoding="utf-8"))  # corpus17 (raw data only)
    cal = [r["video"] for r in bg["scenes"] if r["dataset"] == "CDnet2014"]
    m = len(cal)
    u_pop = float(ucp(0, m))

    # ── m1 summary v2 (A1 + A4 under both definitions) ──
    summ = {}
    for ds in ("CDnet2014", "LASIESTA", "BMC"):
        for dfn, mask in (("a256", True), ("orig", False)):
            D = E[(E.dataset == ds) & (E.def_a256 if mask else True)]
            rec = dict(n_events=len(D), n_videos=int(D.video.nunique()),
                       events_per_video_median=float(D.groupby("video").size().median()) if len(D) else 0,
                       Gstar={f"K{K}": cell_stats(D[f"pGs_K{K}"].values, D.video.values, rng) for K in (1, 2, 4, 8, 16)})
            if ds == "CDnet2014":
                rec["G"] = {f"K{K}": cell_stats(D[f"pG_K{K}"].values, D.video.values, rng) for K in (1, 2, 4, 8, 16)}
                Dc = D[D.video.isin(cal)]
                rec["calibration44"] = dict(n_scenes=int(Dc.video.nunique()), n_events=len(Dc))
            summ[f"{ds}|{dfn}"] = rec
    realA = E[E.dataset.isin(["CDnet2014", "LASIESTA"]) & E.def_a256]
    summ["real_pooled|a256"] = dict(n_events=len(realA), n_videos=int(realA.video.nunique()),
                                    Gstar={f"K{K}": cell_stats(realA[f"pGs_K{K}"].values, realA.video.values, rng)
                                           for K in (1, 2, 4, 8, 16)})
    dump(dict(meta=dict(definitions={"a256": f"L_area_max >= {A_MIN} px (primary)", "orig": "any GT 255 pixel"},
                        events="Mode G* gap1 (CDnet), frame-level (LASIESTA, BMC)", bootstrap=f"B={B_BOOT} by scene, seed {SEED}"),
              summary=summ), "m1_gstar_summary_v2.json")

    # ── operating points v2 ──
    cd = E[E.dataset == "CDnet2014"]
    tabs = {"Gstar|a256|CDnet": table(cd[cd.def_a256], "pGs", rng, EPS_FINE),
            "Gstar|orig|CDnet": table(cd, "pGs", rng, EPS_FINE),
            "G|a256|CDnet": table(cd[cd.def_a256], "pG", rng, EPS_GRID),
            "Gstar|a256|real_pooled": table(realA, "pGs", rng, EPS_FINE)}
    rule = apply_rule(tabs["Gstar|a256|CDnet"], EPS_GRID, u_pop)
    fine = apply_rule(tabs["Gstar|a256|CDnet"], EPS_FINE, u_pop)
    opA = dict(eps=0.20, d_min=32, K=8)
    chosen = rule[0] if rule else None
    better = chosen is not None and (chosen["eps"], -chosen["K"], chosen["d_min"]) < (opA["eps"], -opA["K"], opA["d_min"])
    final = chosen if better else dict(opA, **{k: tabs["Gstar|a256|CDnet"]["d32"]["cells"]["K8"][k] for k in ("p_hat", "boot95_video", "n_events")},
                                       n_required=tabs["Gstar|a256|CDnet"]["d32"]["cells"]["K8"]["n_required"]["eps0.2"]["at_p_hat"])
    ref = {}
    for tag, (d, K, e) in {"OP-A": (32, 8, 0.2), "OP-B": (16, 4, 0.2), "OP-C": (32, 1, 0.1)}.items():
        c = tabs["Gstar|a256|CDnet"][f"d{d}"]["cells"][f"K{K}"]
        ref[tag] = dict(eps=e, d_min=d, K=K, p_hat=c["p_hat"], boot95_video=c["boot95_video"],
                        n_required=c["n_required"][f"eps{e}"]["at_p_hat"], n0=n0(e, 0.05),
                        n_events=c["n_events"])
    out = dict(meta=dict(rule=__doc__.split("OP rule")[1].split("Outputs")[0].strip(), n_cap=N_CAP,
                         c4a_pop_U_44=u_pop, calibration_scenes=m, eps_grid_rule=EPS_GRID, eps_grid_sensitivity=EPS_FINE,
                         d_min=D_MIN, K=K_GRID, power=0.8, delta=0.05),
               tables=tabs, rule_candidates=rule[:10], rule_candidates_fine_eps=fine[:10],
               chosen_by_rule=chosen, beats_OP_A=bool(better), final_operating_point=final, reference_points=ref)
    dump(out, "operating_points_v2.json")
    ck = dict(step="B1", final_operating_point=final, beats_OP_A=bool(better), c4a_pop_U_44=u_pop,
              n_rule_candidates=len(rule), first_fine_eps_candidate=fine[0] if fine else None,
              n_events={k: v["n_events"] for k, v in summ.items()})
    dump(ck, "checkpoint_B1.json")
    print(json.dumps(ck, indent=1))
    for name in ("Gstar|a256|CDnet", "Gstar|orig|CDnet"):
        print("\n==", name)
        for d in D_MIN:
            cells = tabs[name][f"d{d}"]["cells"]
            print(f"d{d:>2} n={tabs[name][f'd{d}']['n_events']:3d} " + " ".join(
                f"K{K}:{cells[f'K{K}']['p_hat']*100:4.1f}{'*' if K <= d else ' '}({cells[f'K{K}']['n_required']['eps0.2']['at_p_hat']})"
                for K in K_GRID))


if __name__ == "__main__":
    main()
