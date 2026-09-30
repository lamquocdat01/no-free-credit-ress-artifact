"""Tables moved from the supplement into the main text (P3 W3 prep), generated from results/*.json:
manuscript/tab_grid_main.tex   condensed operating-point grid (d_min in {32,16,8,1}, K in {1,8,16,24,48})
manuscript/tab_mustfail.tex    must-fail 1 (temporal) and must-fail 2 (night leak, per scene + curve)
manuscript/tab_eventdef.tex    event-definition sensitivity
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data" / "results"
M = ROOT / "manuscript"


def J(n):
    return json.load(open(R / n, encoding="utf-8"))


def P(x, d=1):
    return "--" if x is None else f"{100 * x:.{d}f}\\%"


def grid():
    g = {(r["d_min"], r["K"]): r for r in J("exchange_rate_v2.json")["grid"] if r["eps"] == 0.20}
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Operating-point grid at $\varepsilon=20\%$ (condensed; full grid in the supplement). Each cell: "
         r"static-domain real miss rate $\hat p$ / real events for direct certification / median real events saved per "
         r"camera by a twin run to convergence. The median saving of the twin as built is 0 at every cell. OP$^\ast$ is $(32,16)$.}",
         r"\label{tab:grid}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}rccccc@{}}", r"\toprule",
         r"$d_{\min}\backslash K$ & 1 & 8 & 16 & 24 & 48 \\", r"\midrule"]
    for d in (32, 24, 16, 12, 8, 4, 1):
        cells = []
        for K in (1, 8, 16, 24, 48):
            r = g[(d, K)]
            nd = "--" if r["n_direct"] is None else str(r["n_direct"])
            sv = "--" if r["q_known"]["events_saved_per_camera"]["median"] is None else f"{r['q_known']['events_saved_per_camera']['median']:.0f}"
            cells.append(f"{P(r['p_main'])}/{nd}/{sv}")
        L.append(f"{d} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (M / "tab_grid_main.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def mustfail():
    va = J("validity_audit.json")["must_fail"]["d1_K48"]
    m2 = J("must_fail_v2.json")["points"]["OP*"]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Must-fail scenarios. Top: temporal ($d_{\min}=1$, $K=48$), static scenes with $p_s>\varepsilon$, real "
         r"misses split into detector and temporal misses; all refused in every Monte-Carlo run. Middle: night scenes at "
         r"OP$^\ast$ (every miss is a detector miss), and whether each is falsely certified when judged alone (phase-averaged); "
         r"then the same for the camera-jitter scenes (CDnet scenes with real shake listed; the LASIESTA scenes with "
         r"simulated camera motion pooled, event-weighted). Bottom: "
         r"population bound when $k$ night scenes leak into the calibration set (median over subsets; withdrawn = share of "
         r"subsets with a bound above $\varepsilon/2$).}",
         r"\label{tab:mustfail}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"\multicolumn{7}{@{}l}{\emph{1: temporal}} \\",
         r"scene & $n$ & $p_s$ & detector & temporal & $q_s$ & refused \\", r"\midrule"]
    rr = va["mc_restricted_to_p_gt_eps"]["refuse_rate_given_p_gt_eps"]
    for s in va["scenes_p_gt_eps"]:
        L.append(f"{s['video'].replace('_', chr(92) + '_')} & {s['n']} & {P(s['p'])} & {P(s['p_app'])} & {P(s['p_struct'])} & {P(s['q'])} & {P(rr, 0)} \\\\")
    L += [r"\midrule", r"\multicolumn{7}{@{}l}{\emph{2: detector (night)}} \\",
          r"scene & $n$ & $p_s$ & $q_s$ & $D_s$ & disc. & false cert. \\", r"\midrule"]
    for t in m2["night_table"]:
        fc = "yes" if m2["night_with_main_only_calibration"][t["video"]]["false_cert"] else "no"
        L.append(f"{t['video']} & {t['n']} & {P(t['p_s'])} & {P(t['q_s'])} & {P(t['D_s'])} & {t['worst_phase_discordant']} & {fc} \\\\")
    L += [r"\midrule", r"\multicolumn{7}{@{}l}{\emph{3: detector (camera jitter)}} \\", r"\midrule"]
    sm = [t for t in m2["jitter_table"] if t["video"][:2] in ("I_", "O_")]      # LASIESTA SM: pooled row
    for t in m2["jitter_table"]:
        if t in sm:
            continue
        fc = "yes" if m2["jitter_with_main_only_calibration"][t["video"]]["false_cert"] else "no"
        L.append(f"{t['video']} & {t['n']} & {P(t['p_s'])} & {P(t['q_s'])} & {P(t['D_s'])} & {t['worst_phase_discordant']} & {fc} \\\\")
    if sm:
        n = sum(t["n"] for t in sm)
        w = lambda k: sum(t[k] * t["n"] for t in sm) / n
        fcs = sum(m2["jitter_with_main_only_calibration"][t["video"]]["false_cert"] for t in sm)
        L.append(f"LASIESTA SM ({len(sm)} scenes) & {n} & {P(w('p_s'))} & {P(w('q_s'))} & {P(w('D_s'))} & "
                 f"{sum(t['worst_phase_discordant'] for t in sm)} & {fcs} of {len(sm)} \\\\")
    L += [r"\midrule", r"$k$ leaked & 0 & 1 & 2 & 3 & 4 & 6 \\", r"\midrule"]
    cur = {c["k"]: c for c in m2["curve"]}
    L.append(r"$U_{\mathrm{pop}}$ & " + " & ".join(P(cur[k]["U_pop_median"]) for k in (0, 1, 2, 3, 4, 6)) + r" \\")
    L.append(r"withdrawn & " + " & ".join(P(cur[k]["share_fleet_refused"], 0) for k in (0, 1, 2, 3, 4, 6)) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (M / "tab_mustfail.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def eventdef():
    s = J("sens_event_definition.json")["definitions"]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Event-definition sensitivity (static domain). Primary: largest object $\ge 256$ px; original: every "
         r"ground-truth event. Discordant events at the worst phase; LOSO: scenes above the bound computed without them.}",
         r"\label{tab:eventdef}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"definition & point & $N$ & $\hat p$ & disc. & $\UCP(X,N)$ & $J$ / $\UCP(J,m)$ & LOSO \\", r"\midrule"]
    names = {"OP*": r"OP$^\ast$", "OP-A": "OP-A", "OP-C": "OP-C"}
    for dfn, lab in (("a256", "primary"), ("orig", "original")):
        for k in ("OP*", "OP-A", "OP-C"):
            r = s[dfn]["points"][k]
            L.append(f"{lab} & {names[k]} & {r['N']} & {P(r['p_hat'], 2)} & {r['disc_worst']} & {P(r['U_fleet'], 2)} & "
                     f"{r['J']} / {P(r['U_pop'], 1)} & {r['loso_viol']} \\\\")
            lab = ""
        if dfn == "a256":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (M / "tab_eventdef.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    grid(); mustfail(); eventdef()
    print("[write] tab_grid_main.tex, tab_mustfail.tex, tab_eventdef.tex")


def c4b():
    c = J("c4_power.json")["c4b_requirements"]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Data needed for the per-camera form of Result 4(c) with no discordance ($J=0$): minimum number of scenes "
         r"$m_{\min}$, events per scene $n$ at $m=m_{\min}$ and at $m=2m_{\min}$, total events, and the asymptotic floor "
         r"$\ln(1/\delta)/(\gamma t)$.}",
         r"\label{tab:c4b}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}rrrrrrr@{}}", r"\toprule",
         r"$t$ & $\gamma$ & $m_{\min}$ & $n$ at $m_{\min}$ & $n$ at $2m_{\min}$ & total at $2m_{\min}$ & floor \\", r"\midrule"]
    for t in (0.025, 0.05, 0.10):
        for g in (0.05, 0.10):
            r = c[f"t{t}_gamma{g}"]
            tab = {x["m"]: x for x in r["table"]}
            m0 = r["m_min_scenes"]
            a, b = tab[m0], tab[2 * m0]
            L.append(f"{100 * t:g}\% & {100 * g:g}\% & {m0} & {a['n_per_scene']} & {b['n_per_scene']}, {b['total_events']:,} & "
                     f"{r['total_events_floor']:,.0f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (M / "tab_c4b.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    c4b()
    print("[write] tab_c4b.tex")



def confusion2x2():
    cf = J("e0_confusion.json")
    cols = [("static, OP$^\\ast$", cf["by_op"]["OP*"]["main"]), ("night, OP$^\\ast$", cf["by_op"]["OP*"]["night"])]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Real versus twin outcomes (expected event counts under a uniform sampling phase; twin = majority of three "
         r"variants). Recall: share of expected real misses that the twin also misses.}",
         r"\label{tab:confusion}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
         " & " + " & ".join(c[0] for c in cols) + r" \\", r"\midrule"]
    keys = [("real miss, twin miss", "real_miss & twin_miss"), ("real miss, twin catch ($D$)", "real_miss & twin_catch (D)"),
            ("real catch, twin miss", "real_catch & twin_miss (pessimistic)"), ("real catch, twin catch", "real_catch & twin_catch")]
    for lab, k in keys:
        L.append(lab + " & " + " & ".join(f"{c[1]['cells'][k]:.2f}" for c in cols) + r" \\")
    L.append(r"\midrule")
    L.append("events & " + " & ".join(str(c[1]["N_events"]) for c in cols) + r" \\")
    L.append("events missed at some phase & " + " & ".join(str(c[1]["real_miss_events_any_phase"]) for c in cols) + r" \\")
    L.append("twin recall of real misses & " + " & ".join(P(c[1]["twin_recall_on_real_miss"], 0) for c in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (M / "tab_confusion.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def fidelity():
    geo = J("e0_geometry_sensitivity.json")["frame_level"]["by_kind_aspect"]
    conf = J("twin_discordance_c0_main_main48.json")["confidence"]["by_kind"]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Twin fidelity on the same frames (static domain): hit rates by object kind and sprite aspect error before "
         r"the width clamp, and median detector confidence on hits. Negative differences make the twin pessimistic.}",
         r"\label{tab:fidelity}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"kind & $|$aspect error$|$ & frames & twin hit & real hit & diff.\ (pts) \\", r"\midrule"]
    for k, v in geo.items():
        kind, b = k.split("/")
        b = b.replace("%", r"\%").replace("<=", r"$\le$").replace(">", "$>$")
        L.append(f"{kind} & {b} & {v['n']:,} & {P(v['twin_hit'])} & {P(v['real_hit'])} & {100 * v['twin_minus_real']:+.1f} \\\\")
    L.append(r"\midrule")
    for kind, v in conf.items():
        L.append(f"{kind} & confidence & & {v['twin']:.2f} & {v['real']:.2f} & \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (M / "tab_fidelity.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def loso_table():
    va = J("validity_audit.json")["points"]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Leave-one-scene-out: scenes whose discordance $D_s$ exceeds the bound $U_{-s}$ computed without them. None "
         r"is falsely certified, because $p_s\le\varepsilon$.}",
         r"\label{tab:loso}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"point & scene & $n$ & $D_s$ & $p_s$ & $q_s$ & $U_{-s}$ & $p_s>q_s+U_{-s}$ \\", r"\midrule"]
    for key, lab in (("OP*", r"OP$^\ast$"),):
        for s in va[key]["loso"]["violating_scenes"]:
            L.append(f"{lab} & {s['video'].replace('_', chr(92) + '_')} & {s['n']} & {P(s['D_s'])} & {P(s['p_s'])} & {P(s['q_s'])} & "
                     f"{P(s['U_pop_minus'])} & {'yes' if s['viol_C3'] else 'no'} \\\\")
        if not va[key]["loso"]["violating_scenes"]:
            L.append(f"{lab} & none & & & & & & \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (M / "tab_loso.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    confusion2x2(); fidelity(); loso_table()
    print("[write] tab_confusion.tex, tab_fidelity.tex, tab_loso.tex")
