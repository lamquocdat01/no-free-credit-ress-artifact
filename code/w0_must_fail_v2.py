"""P3 W0.1 — must-fail scenario 2 (detector-driven): out-of-domain (night) scenes leak into the calibration set.

Scenario 1 (P1c, validity_audit.json) used K = 48, where most real misses are STRUCTURAL (event shorter than K: no
sampled frame at all) and the twin reproduces them by construction. Scenario 2 keeps the operating point (OP* and the
post-hoc point (16,16)) and adds k of the 6 CDnet night scenes (twin profile b: held-out fluidHighway,
streetCornerAtNight + tuning scenes) to the 78 main scenes as if they belonged to the main domain, k = 0..6, all
C(6, k) subsets. Unit = event, twin catch = >= 2/3 variants, worst-phase integer counts (E0.1).
Per subset: C4a-fleet U = U_CP(X_w, N), C4a-pop U = U_CP(J, m), the fleet decision (U <= eps/2, the G2 rule) and the
per-camera rule (certify camera s iff q_s + U_pop(-s) <= eps, U_pop(-s) from the other calibration scenes).
Two readings of "the procedure refuses":
  (A) fleet: the calibration no longer yields U_pop <= eps/2 (the domain-level certificate is withdrawn);
  (B) per camera: the leaked night cameras are not certified.
Also the dangerous case: night cameras certified with a calibration made of MAIN scenes only (night not in the
calibration) -> false certificates if p_s > eps.
Every table splits real misses into DETECTOR misses (the sampled residue class contains event frames, the detector
fails on all of them) and TEMPORAL misses (duration < K: no sampled frame).
Output: results/must_fail_v2.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c4_power import ucp  # noqa: E402
from common import RESULTS, SEED, dump  # noqa: E402
from e0_checks import event_rows  # noqa: E402
from m3_validity_audit import event_table, scene_table  # noqa: E402

DELTA = 0.05
POINTS = {"OP*": dict(eps=0.20, d_min=32, K=16), "challenge_16_16": dict(eps=0.20, d_min=16, K=16, note="post-hoc")}


def per_camera(S, eps, cal_mask):
    """Certify each camera with U_pop from the calibration scenes other than itself."""
    cal_mask = pd.Series(np.asarray(cal_mask), index=S.index)
    cal = S[cal_mask]
    m, J = len(cal), int(cal.w.sum())
    out = {}
    for v, r in S.iterrows():
        inside = bool(cal_mask.loc[v])
        Jm, mm = (J - int(r.w), m - 1) if inside else (J, m)
        U = float(ucp(Jm, mm))
        out[v] = dict(U_pop_minus=U, certified=bool(r.q + U <= eps), p_s=float(r.p), q_s=float(r.q),
                      false_cert=bool(r.q + U <= eps and r.p > eps))
    return out


def evaluate(S, eps, night, leaked):
    cal_mask = ~S.index.isin(night) | S.index.isin(leaked)
    cal = S[cal_mask]
    N, m = int(cal.n.sum()), len(cal)
    Xw, J = float((cal.n * cal.xw).sum()), int(cal.w.sum())
    Uf, Up = float(ucp(Xw, N)), float(ucp(J, m))
    pc = per_camera(S, eps, cal_mask)
    lk = [pc[v] for v in leaked]
    main = [pc[v] for v in S.index if v not in night]
    return dict(k=len(leaked), leaked=list(leaked), N=N, m=m, X_worst=Xw, J=J, U_fleet=Uf, U_pop=Up,
                fleet_refused=bool(Up > eps / 2 or Uf > eps / 2),
                leaked_certified=int(sum(r["certified"] for r in lk)), leaked_false_cert=int(sum(r["false_cert"] for r in lk)),
                main_certified=int(sum(r["certified"] for r in main)), main_false_cert=int(sum(r["false_cert"] for r in main)))


def main():
    rows_m = event_rows("c0_main", "main")
    rows_n = [r for tag in ("nheld_b", "ntune_b") for r in event_rows(tag, "night")]
    out = dict(meta=dict(doc=__doc__.split("Output")[0].strip(), seed=SEED, delta=DELTA), points={})
    for name, op in POINTS.items():
        evm = event_table(rows_m, op["d_min"], op["K"])
        evn = event_table(rows_n, op["d_min"], op["K"])
        ev = pd.concat([evm, evn], ignore_index=True)
        S = scene_table(ev)
        S["xw"] = ev.groupby("video").w.mean()                  # worst-phase discordant share per scene
        night = sorted(evn.video.unique())
        eps = op["eps"]
        rec = dict(op=op, night_scenes=night,
                   night_table=[dict(video=v, n=int(S.loc[v, "n"]), p_s=float(S.loc[v, "p"]), p_detector=float(S.loc[v, "p_app"]),
                                     p_temporal=float(S.loc[v, "p_struct"]), q_s=float(S.loc[v, "q"]), D_s=float(S.loc[v, "D"]),
                                     worst_phase_discordant=int((evn[evn.video == v].w > 0).sum())) for v in night],
                   main_miss_split=dict(p=float(evm.p.mean()), p_detector=float(evm.p_app.mean()), p_temporal=float(evm.p_struct.mean()),
                                        N=len(evm)))
        curve = []
        for k in range(0, len(night) + 1):
            subs = [evaluate(S, eps, night, c) for c in itertools.combinations(night, k)]
            U = np.array([s["U_pop"] for s in subs])
            Uf = np.array([s["U_fleet"] for s in subs])
            curve.append(dict(k=k, n_subsets=len(subs), U_pop_median=float(np.median(U)), U_pop_min=float(U.min()), U_pop_max=float(U.max()),
                              U_fleet_median=float(np.median(Uf)), U_fleet_min=float(Uf.min()), U_fleet_max=float(Uf.max()),
                              share_fleet_refused=float(np.mean([s["fleet_refused"] for s in subs])),
                              leaked_certified_total=int(sum(s["leaked_certified"] for s in subs)),
                              leaked_cameras_total=int(k * len(subs)),
                              leaked_false_cert_total=int(sum(s["leaked_false_cert"] for s in subs)),
                              main_false_cert_max=int(max(s["main_false_cert"] for s in subs)),
                              subsets=subs if k in (1, 3, 6) else None))
        rec["curve"] = curve
        # dangerous case: night cameras judged with a MAIN-only calibration (k = 0)
        pc0 = per_camera(S, eps, ~S.index.isin(night))
        rec["night_with_main_only_calibration"] = {v: pc0[v] for v in night}
        rec["night_false_cert_main_only"] = int(sum(pc0[v]["false_cert"] for v in night))
        rec["night_p_gt_eps"] = int(sum(pc0[v]["p_s"] > eps for v in night))
        out["points"][name] = rec
    dump(out, "must_fail_v2.json")
    for name, rec in out["points"].items():
        print(name, "main split", {k: round(v, 4) for k, v in rec["main_miss_split"].items()})
        for t in rec["night_table"]:
            print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()})
        for c in rec["curve"]:
            print("  k", c["k"], "Upop med", round(c["U_pop_median"], 3), "[", round(c["U_pop_min"], 3), round(c["U_pop_max"], 3), "] Ufleet med",
                  round(c["U_fleet_median"], 3), "refused", c["share_fleet_refused"], "leaked cert", c["leaked_certified_total"], "/", c["leaked_cameras_total"],
                  "false", c["leaked_false_cert_total"], "main false max", c["main_false_cert_max"])
        print("  night judged with main-only calibration: false cert", rec["night_false_cert_main_only"], "of", len(rec["night_scenes"]),
              "(p>eps:", rec["night_p_gt_eps"], ")")


if __name__ == "__main__":
    main()
