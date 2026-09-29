"""P3 W2 — sensitivity of the static-domain certificate to the event definition (planned in the outline, §6.6).
Same data, same method as E0.1 / M3 (event unit, majority twin, worst-phase integer counts); only the event filter
changes: primary = largest object >= 256 px ('a256'), original = every ground-truth event ('orig').
Output: results/sens_event_definition.json
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import dump  # noqa: E402
from e0_checks import confusion, ev_metrics, event_rows, summary  # noqa: E402
from m3_validity_audit import event_table, loso, scene_table  # noqa: E402

POINTS = {"OP*": (32, 16, 0.20), "OP-A": (32, 8, 0.20), "OP-C": (32, 1, 0.10), "challenge_16_16": (16, 16, 0.20)}


def main():
    out = dict(meta=dict(doc=__doc__.strip()), definitions={})
    for dfn in ("a256", "orig"):
        rows = event_rows("c0_main", "main", definition=dfn)
        rec = dict(n_events_all_durations=len(rows), points={})
        for name, (d, K, eps) in POINTS.items():
            s = summary(ev_metrics(rows, d, K), eps)
            c = confusion(ev_metrics(rows, d, K))
            lo = loso(scene_table(event_table(rows, d, K)), eps)
            rec["points"][name] = dict(N=s["N_events"], m=s["m_scenes"], disc_worst=s["D_worst_phase"]["num"],
                                       U_fleet=s["D_worst_phase"]["U_fleet"], J=s["J"]["num"], U_pop=s["J"]["U_pop"],
                                       p_hat=c["p_hat"], real_miss_events=c["appearance"]["events_real_miss_any_phase"],
                                       loso_viol=lo["violation_rate_D"]["num"], loso_false_cert=lo["false_certificates"]["num"])
        out["definitions"][dfn] = rec
    dump(out, "sens_event_definition.json")
    for dfn, r in out["definitions"].items():
        print(dfn, r["n_events_all_durations"], {k: (v["N"], v["disc_worst"], round(v["U_fleet"], 4), v["J"], round(v["U_pop"], 4), round(v["p_hat"], 4), v["loso_viol"]) for k, v in r["points"].items()})


if __name__ == "__main__":
    main()
