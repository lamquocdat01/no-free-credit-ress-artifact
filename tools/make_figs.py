"""Figures F3-F6 of the manuscript (F2 = code/w0_exchange_rate_v2.py; F1 = TikZ in main.tex).
F3 tier certificates (e0_units.json), F4 U_pop vs number of leaked night scenes (must_fail_v2.json),
F5 LOSO per scene at OP* and (16,16) (recomputed with code/m3_validity_audit.py functions; same numbers as
validity_audit.json), F6 sprite gallery on grey (CC0 MPFB person sprites only; no dataset frames are reproduced
because CDnet 2014 / LASIESTA publish no reuse terms we could record).
Palette: categorical slots 1-2 of the validated default palette; text in neutral inks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
R = ROOT / "data" / "results"
FIG = ROOT / "manuscript" / "figs"
C1, C2, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def J(n):
    return json.load(open(R / n, encoding="utf-8"))


def style(plt):
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)


def f3(plt):
    u = J("e0_units.json")["recomputed"]
    lab = lambda t, name: f"{name} ({u[t]['rule_2of3']['J']['den']}" + (" scenes)" if t == "main" else ")")  # noqa: E731
    rows = [(lab("main", "static"), u["main"]["rule_2of3"]), (lab("jitter", "camera jitter"), u["jitter"]["rule_2of3"]),
            (lab("night", "night"), u["night"]["rule_2of3"]), (lab("turbulence", "turbulence"), u["turbulence"]["rule_2of3"]),
            (lab("all_tiers", "all tiers"), u["all_tiers"]["rule_2of3"])]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for i, (lab, r) in enumerate(rows):
        y = len(rows) - 1 - i
        d = r["D_worst_phase"]["value"]
        ax.plot([d, r["D_worst_phase"]["U_fleet"]], [y + 0.12] * 2, color=C1, lw=2, solid_capstyle="round")
        ax.plot([d], [y + 0.12], "|", color=C1, ms=7)
        ax.plot([r["J"]["num"] / r["J"]["den"], r["J"]["U_pop"]], [y - 0.12] * 2, color=C2, lw=2, solid_capstyle="round")
        ax.plot([r["J"]["num"] / r["J"]["den"]], [y - 0.12], "|", color=C2, ms=7)
    ax.axvline(0.10, color=INK2, lw=0.8, ls="--")
    ax.text(0.105, len(rows) - 1.55, r"$\varepsilon/2$", fontsize=7, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    ax.set_xticklabels(["0", "2%", "5%", "10%", "20%", "50%", "100%"])
    ax.plot([], [], color=C1, lw=2, label="events: estimate | to the fleet bound")
    ax.plot([], [], color=C2, lw=2, label="scenes: $J/m$ | to the population bound")
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=1)
    ax.grid(axis="x", color=GRID, lw=0.5)
    save(fig, "fig_tiers")


def f4(plt):
    m = J("must_fail_v2.json")["points"]
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    for key, col, mk, lab in (("OP*", C1, "o", "OP* $(d_{\\min}=32,K=16)$"), ("challenge_16_16", C2, "s", "$(16,16)$, post hoc")):
        c = m[key]["curve"]
        k = [r["k"] for r in c]
        ax.fill_between(k, [r["U_pop_min"] for r in c], [r["U_pop_max"] for r in c], color=col, alpha=0.15, lw=0)
        ax.plot(k, [r["U_pop_median"] for r in c], marker=mk, ms=4, color=col, lw=2, label=lab, mec="#fcfcfb", mew=0.6)
    ax.axhline(0.10, color=INK2, lw=0.8, ls="--")
    ax.text(6.0, 0.092, r"$\varepsilon/2$ (withdrawn above)", fontsize=7, color=INK2, ha="right")
    ax.set_xlabel("night scenes leaked into the calibration set, $k$ (of 6)")
    ax.set_ylabel("population bound $U_{\\mathrm{pop}}$")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, "fig_leak")


def f5(plt):
    from e0_checks import event_rows
    from m3_validity_audit import event_table, loso, scene_table
    rows = event_rows("c0_main", "main")
    va = J("validity_audit.json")["points"]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
    for ax, (key, d, K, lab) in zip(axes, (("OP*", 32, 16, "OP*"), ("challenge_16_16", 16, 16, "(16,16) post hoc"))):
        S = scene_table(event_table(rows, d, K))
        m, J_ = len(S), int(S.w.sum())
        from c4_power import ucp
        U = np.array([float(ucp(J_ - int(w), m - 1)) for w in S.w])
        D = S.D.values
        assert int((D > U).sum()) == va[key]["loso"]["violation_rate_D"]["num"]
        jit = (np.arange(m) % 7 - 3) * 0.0006
        ax.scatter(U + jit, D, s=10, color=C1, edgecolors="#fcfcfb", linewidths=0.4, zorder=3)
        lim = max(0.12, D.max() * 1.1)
        ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls="--")
        n_lab = 0
        for v, x, y in sorted(zip(S.index, U, D), key=lambda t: -t[2]):
            if y > x:
                ax.annotate(v.replace("_1fps", ""), (x, y), xytext=(-4, 5 - 11 * n_lab), textcoords="offset points",
                            fontsize=7, color=INK, ha="right")
                n_lab += 1
        ax.set_title(lab, fontsize=7.5, color=INK)
        ax.set_xlabel("$U_{\\mathrm{pop}}$ without the scene", fontsize=7)
        ax.set_xlim(0, lim)
        ax.set_ylim(-0.005, lim)
        ax.grid(color=GRID, lw=0.5)
    axes[0].set_ylabel("scene discordance $D_s$", fontsize=7)
    save(fig, "fig_loso")


def f6(plt):
    import cv2
    spr = ROOT / "assets" / "sprites" / "person"
    if not spr.exists():                       # artifact repository layout
        spr = ROOT / "data" / "sprites" / "person"
    names = ["char00_az000_p0.png", "char03_az090_p1.png", "char05_az045_p2.png", "char07_az135_p0.png",
             "char09_az000_p2.png", "char11_az090_p1.png"]
    names = [n for n in names if (spr / n).exists()][:6]
    fig, axes = plt.subplots(2, len(names), figsize=(3.5, 1.7))
    for j, n in enumerate(names):
        im = cv2.imread(str(spr / n), cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2RGB).astype(float) / 255
        a = im[:, :, 3:4].astype(float) / 255
        for i, (bg, gain) in enumerate(((0.55, 1.0), (0.12, 0.35))):
            comp = a * np.clip(rgb * gain, 0, 1) + (1 - a) * bg
            ax = axes[i, j]
            ax.imshow(comp)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    axes[0, 0].set_ylabel("day gain", fontsize=7)
    axes[1, 0].set_ylabel("night gain", fontsize=7)
    save(fig, "fig_sprites")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    FIG.mkdir(parents=True, exist_ok=True)
    for fn in (f3, f4, f5, f6):
        fn(plt)
        print("ok", fn.__name__)


if __name__ == "__main__":
    main()
