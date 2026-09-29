"""P3 W0.2 — exchange rate in EVENTS (primary) + CPU cost per saved event; waiting time only as a formula.

Built on m4_exchange_rate.point (same rule, same data: 78 main scenes, main-domain p-hat, eps = 20% grid, OP-C at
10%). Per grid point (d_min x K) and camera s:
  n_saved_s        = n_direct - n_real_twin,s   (real events saved; 0 when the twin route fails)
                     (a) q known (twin run long enough), (b) finite twin as built (3 variants x the scene's events)
  cpu_h_per_saved  = n_twin_needed,s * c_s / n_saved_s, c_s = measured CPU-hours per twin event run of scene s
                     (C0 build time / (3 x events)); defined only where n_saved_s > 0 and the twin route is feasible.
Waiting time is NOT estimated: T_saved = n_saved / lambda_s, lambda_s = the camera's real events per hour of
operation, is a deployment parameter. CDnet 2014 and LASIESTA do not publish per-video frame rates; the benchmark
clips (annotated spans, event-dense) are reported only as a sensitivity with fps in {15, 25, 30} (4 CDnet videos state
their fps in the name).
Output: results/exchange_rate_v2.json, manuscript/figs/fig_exchange_rate.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RESULTS, SEED, dump  # noqa: E402
from e0_checks import event_rows  # noqa: E402
from m4_exchange_rate import D_GRID, K_GRID, MARKED, point, twin_cost, video_frames  # noqa: E402

FIG = HERE.parent / "manuscript" / "figs"
C_KNOWN, C_FIN = "#2a78d6", "#eb6834"          # categorical slots 1-2 (validated default palette)


def q(x, p):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return None if not len(x) else float(np.percentile(x, p))


def summarise(g, recs):
    nd = g["n_direct_at_p_main"]
    out = dict(d_min=g["d_min"], K=g["K"], eps=g["eps"], N_events=g["N_events"], p_main=g["p_main"], J=g["J"],
               n_direct=nd, n0=g["n0"])
    for tag in ("q_known", "q_finite"):
        sv = [r[tag]["saving_vs_direct"] for r in recs]
        sv = [0 if v is None else v for v in sv]
        cph = [r["n_twin_events_needed"] * r["twin_cpu_hours_per_twin_event"] / s
               for r, s in zip(recs, sv) if s > 0 and r["n_twin_events_needed"] and r["twin_cpu_hours_per_twin_event"]]
        wf = {}
        for f in ("15.0", "25.0", "30.0"):
            wh = [s / r["lambda_per_hour"][f] for r, s in zip(recs, sv)]
            wf[f] = dict(median=q(wh, 50), iqr=[q(wh, 25), q(wh, 75)])
        out[tag] = dict(events_saved_per_camera=dict(median=q(sv, 50), iqr=[q(sv, 25), q(sv, 75)], mean=float(np.mean(sv)),
                                                     unit="real events per camera (78 cameras)"),
                        cameras_saving=int(sum(s > 0 for s in sv)), m=len(recs),
                        share_cameras_saving_ge_50pct=(float(np.mean([s >= 0.5 * nd for s in sv])) if nd else None),
                        cpu_hours_per_saved_event=dict(median=q(cph, 50), iqr=[q(cph, 25), q(cph, 75)], n_cameras=len(cph)),
                        benchmark_wait_hours_sensitivity=wf)
    return out


def figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "pdf.fonttype": 42})
    R = [r for r in rows if r["eps"] == 0.20 and r["n_direct"] is not None]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for tag, col, mk, lab, dx in (("q_known", C_KNOWN, "o", "twin run to convergence (q known)", 1.0),
                                  ("q_finite", C_FIN, "s", "twin as built (3 runs per event)", 1.04)):
        x = np.array([r["p_main"] for r in R]) * 100 * dx
        y = np.array([r[tag]["events_saved_per_camera"]["median"] for r in R])
        lo = y - np.array([r[tag]["events_saved_per_camera"]["iqr"][0] for r in R])
        hi = np.array([r[tag]["events_saved_per_camera"]["iqr"][1] for r in R]) - y
        ax.errorbar(x, y, yerr=[lo, hi], fmt=mk, ms=3.5, mfc=col, mec="#fcfcfb", mew=0.6, ecolor=col, elinewidth=0.8,
                    capsize=0, color=col, label=lab, zorder=3)
    for name, lab in (("OP*", "OP*"), ("challenge_16_16", "(16,16) post-hoc")):
        d, K, e = MARKED[name]
        r = [r for r in R if (r["d_min"], r["K"]) == (d, K)][0]
        x, y = r["p_main"] * 100, r["q_known"]["events_saved_per_camera"]["median"]
        ax.scatter([x], [y], s=60, facecolors="none", edgecolors="#0b0b0b", linewidths=0.9, zorder=4)
        ax.annotate(lab, (x, y), xytext=(6, 8 if name == "OP*" else -12), textcoords="offset points", fontsize=7, color="#0b0b0b")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("main-domain real miss rate $\\hat p$ (%)")
    ax.set_ylim(-1, 2e4)
    ax.set_ylabel("real events saved per camera")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_exchange_rate.pdf")
    fig.savefig(FIG / "fig_exchange_rate.png", dpi=200)


def main():
    rows_ev = event_rows("c0_main", "main")
    frames, cost = video_frames(), twin_cost()
    rows, marked = [], {}
    for eps in (0.20, 0.10):
        for d in D_GRID:
            for K in K_GRID:
                g, recs = point(rows_ev, d, K, eps, frames, cost)
                s = summarise(g, recs)
                rows.append(s)
                for name, (dm, kk, ee) in MARKED.items():
                    if (dm, kk, ee) == (d, K, eps):
                        marked[name] = s
    out = dict(meta=dict(doc=__doc__.split("Output")[0].strip(), seed=SEED,
                         waiting_time_formula="T_saved,s = n_saved,s / lambda_s  (lambda_s: real events per hour at camera s; "
                                              "a deployment parameter, not estimated)",
                         fps_note="CDnet 2014 / LASIESTA publish no per-video fps; benchmark clips used only as fps sensitivity {15,25,30}"),
               marked=marked, grid=rows)
    dump(out, "exchange_rate_v2.json")
    figure(rows)
    for k, v in marked.items():
        print(k, "p", round(v["p_main"], 4), "n_direct", v["n_direct"],
              {t: (v[t]["events_saved_per_camera"]["median"], v[t]["events_saved_per_camera"]["iqr"], v[t]["cameras_saving"],
                   v[t]["cpu_hours_per_saved_event"]["median"]) for t in ("q_known", "q_finite")})


if __name__ == "__main__":
    main()
