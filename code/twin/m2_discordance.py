"""M2 — discordance between real events and their paired twins (exact over the K phases).

For event e, period K, hit-residue sets R_real = {t mod K : real frame t is a G* hit},
R_twin = {t mod K : twin frame t is a G* hit}. Under a uniform random phase:
  p_e = 1 - |R_real|/K (real miss), q_e = 1 - |R_twin|/K (twin miss),
  D_e = |R_twin \\ R_real| / K = P(real miss AND twin catch),   disagreement = |R_real xor R_twin| / K.
Lemma C3 holds per event: p_e <= q_e + D_e (checked).
Scene level: D_s = mean_e D_e, Clopper-Pearson on the expected count (fractional, as M1). Heterogeneity:
Cochran Q / I^2 on scene proportions and a permutation test of homogeneity (events shuffled across scenes,
10,000 permutations, seed 42) at K = 1 (binary D_e).
Certificates on the real numbers:
  C4a-fleet: U_CP(X, N) with X = sum_e D_e (expected count; plus one seeded random-phase draw, seed 42).
  C4a-pop:   J = #scenes with >= 1 discordant event: J_possible (some D_e > 0) and J at the seeded draw.
Reported for definition a256 (primary) and orig, per sprite variant and averaged over the 3 variants,
at the chosen operating point (d_min, K) and over the K grid. Detector confidence twin vs real on hit frames.
Usage: python m2_discordance.py pilot|full [--tag TAG (results/twin_TAG)] [--only v1,v2 --name NAME]
Output: results/twin_discordance[_pilot].json, results/gstar_frames_twin[_pilot].parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from c4_power import ucp  # noqa: E402
from common import B_BOOT, K_GRID, RESULTS, SEED, cp_ci, dump  # noqa: E402


def residues(frames, K):
    return set(int(t) % K for t in frames)


def real_hits(E, T):
    """Real G* hit frames of every event from the FULL per-frame tables (A1/A4), not from the twin rows
    (the twin stops early; the real residue set must be complete)."""
    out = {}
    for ds in T.dataset.unique():
        F = pd.read_parquet(RESULTS / f"gstar_frames_{ds}.parquet", columns=["video", "frame", "has_fg", "hit"])
        F = F[F.video.isin(T[T.dataset == ds].video.unique())]
        hit = F.has_fg.astype(bool) & F.hit.fillna(False).astype(bool)
        byv = {v: set(g.frame.values[hit.loc[g.index].values]) for v, g in F.groupby("video")}
        for e in E[(E.dataset == ds) & E.video.isin(byv)].itertuples():
            fr = range(int(e.onset_frame), int(e.onset_frame) + int(e.duration))
            out[(ds, e.video, int(e.event_id))] = np.array(sorted(t for t in fr if t in byv[e.video]))
    return out


def event_table(T, E):
    """T: twin per-frame rows; E: event table (miss_matrix_Gstar_a256.csv) for the same scenes."""
    RH = real_hits(E, T)
    rows = []
    for (ds, v, eid, var), g in T.groupby(["dataset", "video", "event_id", "variant"]):
        e = E[(E.dataset == ds) & (E.video == v) & (E.event_id == eid)].iloc[0]
        fr_real = RH[(ds, v, int(eid))]
        chk = g.frame.values[g.real_hit.values.astype(bool)]
        assert set(chk) <= set(fr_real), "real hits in twin rows must be a subset of the A1/A4 table"
        fr_twin = g.frame.values[g.twin_hit.values.astype(bool)]
        r = dict(dataset=ds, video=v, event_id=int(eid), variant=int(var), def_a256=bool(e.def_a256),
                 duration=int(e.duration), kind=g.kind.iloc[0], cls_imputed=bool(g.cls_imputed.iloc[0]),
                 n_frames_twin=len(g))
        for K in K_GRID:
            Rr, Rt = residues(fr_real, K), residues(fr_twin, K)
            r[f"p_K{K}"] = 1 - len(Rr) / K
            r[f"q_K{K}"] = 1 - len(Rt) / K
            r[f"D_K{K}"] = len(Rt - Rr) / K
            r[f"dis_K{K}"] = len(Rr ^ Rt) / K
            assert r[f"p_K{K}"] <= r[f"q_K{K}"] + r[f"D_K{K}"] + 1e-12      # lemma C3, per event
        rows.append(r)
    return pd.DataFrame(rows)


def heterogeneity(ev, col, rng, n_perm=10000):
    s = ev.groupby("video")[col].agg(["sum", "count"])
    x, n = s["sum"].values, s["count"].values
    p = x.sum() / n.sum()
    if p <= 0 or p >= 1 or len(s) < 2:
        return dict(pooled=float(p), Q=0.0, df=int(len(s) - 1), I2=0.0, perm_p=1.0, note="no variation" if p <= 0 else "")
    Q = float(np.sum(n * (x / n - p) ** 2) / (p * (1 - p)))
    df = len(s) - 1
    I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    vals = ev[col].values
    lab = ev.video.values
    uv, inv = np.unique(lab, return_inverse=True)
    cnt = np.bincount(inv)

    def q_of(vv):
        xs = np.bincount(inv, weights=vv)
        return np.sum(cnt * (xs / cnt - p) ** 2) / (p * (1 - p))
    ge = sum(q_of(rng.permutation(vals)) >= Q - 1e-12 for _ in range(n_perm))
    return dict(pooled=float(p), Q=Q, df=int(df), I2=float(I2), perm_p=float((ge + 1) / (n_perm + 1)))


def certificates(ev, K, rng):
    col = f"D_K{K}"
    X = float(ev[col].sum())
    N = len(ev)
    draw = rng.random(N) < ev[col].values          # one random-phase realisation (seed 42)
    by = ev.assign(d=draw).groupby("video")
    J_possible = int((ev.groupby("video")[col].max() > 0).sum())
    J_draw = int((by.d.max() > 0).sum())
    m = ev.video.nunique()
    return dict(N=N, m_scenes=m, X_expected=X, D_hat=X / N if N else None, cp95=cp_ci(X, N),
                c4a_fleet_U_expected=float(ucp(X, N)) if N else None,
                X_draw=int(draw.sum()), c4a_fleet_U_draw=float(ucp(int(draw.sum()), N)) if N else None,
                J_possible=J_possible, J_draw=J_draw,
                c4a_pop_U_possible=float(ucp(J_possible, m)), c4a_pop_U_draw=float(ucp(J_draw, m)),
                p_hat=float(ev[f"p_K{K}"].mean()), q_hat=float(ev[f"q_K{K}"].mean()),
                disagreement=float(ev[f"dis_K{K}"].mean()))


def geometry_check(T):
    """Sprite box vs GT box: height is exact; width = w_ratio * W_gt. Tolerance +-10% of W_gt, plus 1 px
    for integer rounding (tiny boxes). W_gt re-read from the A1/A4 frame tables (largest GT object)."""
    bw = {}
    for ds in T.dataset.unique():
        F = pd.read_parquet(RESULTS / f"gstar_frames_{ds}.parquet", columns=["video", "frame", "L_bbox"]).dropna()
        for v, t, b in F.values:
            x = list(map(int, b.split(",")))
            bw[(v, int(t))] = max(2, x[2] - x[0])
    W = np.array([bw[(v, int(t))] for v, t in T[["video", "frame"]].values])
    dev = np.abs(T.w_ratio.values * W - W)
    return dict(area_ratio_median=float(T.area_ratio.median()), w_ratio_median=float(T.w_ratio.median()),
                h_ratio_exact=bool(np.allclose(T.h_ratio.values, 1.0)),
                frac_width_within_10pct_strict=float(np.mean(dev <= 0.1 * W + 1e-9)),
                frac_width_within_10pct_plus_1px=float(np.mean(dev <= 0.1 * W + 1 + 1e-9)),
                frac_width_at_clip_bound=float(np.mean(dev >= 0.1 * W - 0.5)),
                note="box area ratio = width ratio (height exact); 'at clip bound' = sprite aspect differs from the "
                     "GT box by >= 10% and was clamped")


def main():
    mode = sys.argv[1]
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else None
    tw = RESULTS / (f"twin_{tag}" if tag else ("twin_pilot" if mode == "pilot" else "twin"))
    suffix = f"_{tag}" if tag else ("_pilot" if mode == "pilot" else "")
    T = pd.concat([pd.read_parquet(p) for p in sorted(tw.glob("frames_*.parquet"))], ignore_index=True)
    if "--only" in sys.argv:
        T = T[T.video.isin(sys.argv[sys.argv.index("--only") + 1].split(","))]
        suffix += "_" + sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else "_subset"
    T.to_parquet(RESULTS / f"gstar_frames_twin{suffix}.parquet", index=False)
    E = pd.read_csv(RESULTS / "miss_matrix_Gstar_a256.csv")
    ev = event_table(T, E)
    op = json.load(open(RESULTS / "operating_points_v2.json", encoding="utf-8"))["final_operating_point"]
    rng = np.random.default_rng(SEED)
    out = dict(meta=dict(mode=mode, n_frames=len(T), n_scenes=int(T.video.nunique()),
                         operating_point=dict(eps=op["eps"], d_min=op["d_min"], K=op["K"]),
                         lemma_C3_per_event="asserted p_e <= q_e + D_e for every event x K x variant",
                         permutation=f"10,000 perms, seed {SEED}", note=__doc__.split("Usage")[0].strip()),
               by_definition={})
    for dfn in ("a256", "orig"):
        evd = ev[ev.def_a256] if dfn == "a256" else ev
        rec = dict(n_events=int(evd.groupby(["video", "event_id"]).ngroups), n_scenes=int(evd.video.nunique()))
        mean_ev = evd.groupby(["dataset", "video", "event_id", "duration", "kind"], as_index=False)[
            [c for c in evd.columns if c[:2] in ("p_", "q_", "D_", "di")]].mean()
        for K in K_GRID:
            rec[f"K{K}"] = dict(variant_mean=certificates(mean_ev, K, rng),
                                per_variant={int(vv): certificates(evd[evd.variant == vv], K, rng) for vv in sorted(evd.variant.unique())})
        sub = mean_ev[mean_ev.duration >= op["d_min"]]
        rec["operating_point"] = dict(n_events=len(sub), **certificates(sub, op["K"], rng),
                                      per_variant={int(vv): certificates(evd[(evd.variant == vv) & (evd.duration >= op["d_min"])], op["K"], rng)
                                                   for vv in sorted(evd.variant.unique())})
        cat = T.drop_duplicates("video").set_index("video")["category"].to_dict()
        sub2 = sub.assign(category=sub.video.map(cat).fillna("LASIESTA"))
        rec["operating_point_by_kind"] = {k: dict(N=len(g), p_hat=float(g[f"p_K{op['K']}"].mean()), q_hat=float(g[f"q_K{op['K']}"].mean()),
                                                  D_hat=float(g[f"D_K{op['K']}"].mean()), U_fleet=float(ucp(g[f"D_K{op['K']}"].sum(), len(g))))
                                          for k, g in sub2.groupby("kind")}
        rec["operating_point_by_category"] = {k: dict(N=len(g), n_scenes=int(g.video.nunique()), p_hat=float(g[f"p_K{op['K']}"].mean()),
                                                      q_hat=float(g[f"q_K{op['K']}"].mean()), D_hat=float(g[f"D_K{op['K']}"].mean()))
                                              for k, g in sub2.groupby("category")}
        k1 = evd.assign(D_bin=(evd["D_K1"] > 0).astype(float))
        rec["heterogeneity_K1"] = {f"variant{vv}": heterogeneity(k1[k1.variant == vv], "D_bin", rng) for vv in sorted(evd.variant.unique())}
        per_scene = []
        for v, g in mean_ev.groupby("video"):
            x = float(g["D_K1"].sum())
            per_scene.append(dict(video=v, dataset=g.dataset.iloc[0], n_events=len(g), D_K1=x / len(g), cp95_K1=cp_ci(x, len(g)),
                                  p_K1=float(g["p_K1"].mean()), q_K1=float(g["q_K1"].mean()),
                                  D_OP=float(g[g.duration >= op["d_min"]][f"D_K{op['K']}"].mean()) if (g.duration >= op["d_min"]).any() else None,
                                  n_OP=int((g.duration >= op["d_min"]).sum()),
                                  X_OP=float(g[g.duration >= op["d_min"]][f"D_K{op['K']}"].sum()),
                                  cp95_OP=cp_ci(float(g[g.duration >= op["d_min"]][f"D_K{op['K']}"].sum()), int((g.duration >= op["d_min"]).sum()))
                                  if (g.duration >= op["d_min"]).any() else None))
        rec["per_scene"] = per_scene
        sv = evd.groupby(["video", "event_id"])["D_K1"].agg(["min", "max"])
        rec["sprite_variance_K1"] = dict(frac_events_variants_disagree=float((sv["max"] > sv["min"]).mean()))
        out["by_definition"][dfn] = rec
    # detector confidence on hit frames, twin vs real, per scene
    conf = []
    for v, g in T.groupby("video"):
        conf.append(dict(video=v, kind=g.kind.mode().iloc[0],
                         twin_score_median=float(g.loc[g.twin_hit, "twin_score"].median()) if g.twin_hit.any() else None,
                         real_score_median=float(g.drop_duplicates(["event_id", "frame"]).loc[lambda x: x.real_hit, "real_score"].median())
                         if g.real_hit.any() else None,
                         twin_hit_rate=float(g.twin_hit.mean()), real_hit_rate=float(g.drop_duplicates(["event_id", "frame"]).real_hit.mean()),
                         area_ratio_median=float(g.area_ratio.median()), w_ratio_median=float(g.w_ratio.median()),
                         plate_source=g.plate_source.iloc[0] if "plate_source" in g else None))
    C = pd.DataFrame(conf)
    out["confidence"] = dict(per_scene=conf,
                             twin_score_median_pooled=float(T.loc[T.twin_hit, "twin_score"].median()),
                             real_score_median_pooled=float(T.drop_duplicates(["video", "event_id", "frame"]).loc[lambda x: x.real_hit, "real_score"].median()),
                             twin_hit_rate_pooled=float(T.twin_hit.mean()),
                             real_hit_rate_pooled=float(T.drop_duplicates(["video", "event_id", "frame"]).real_hit.mean()),
                             criterion="twin median matched-box confidence >= 0.6",
                             pass_=bool(T.loc[T.twin_hit, "twin_score"].median() >= 0.6),
                             by_kind={k: dict(twin=float(g.loc[g.twin_hit, "twin_score"].median()) if g.twin_hit.any() else None,
                                              real=float(g.drop_duplicates(["video", "event_id", "frame"]).loc[lambda x: x.real_hit, "real_score"].median()) if g.real_hit.any() else None,
                                              twin_hit=float(g.twin_hit.mean()))
                                      for k, g in T.groupby("kind")},
                             geometry=geometry_check(T),
                             geometry_by_kind={k: geometry_check(g) for k, g in T.groupby("kind")})
    out["meta"]["twin_dir"] = str(tw.name)
    dump(out, f"twin_discordance{suffix}.json")
    a = out["by_definition"]["a256"]
    print("events", a["n_events"], "scenes", a["n_scenes"])
    for K in (1, 4, 16):
        c = a[f"K{K}"]["variant_mean"]
        print(f"K{K} p={c['p_hat']:.3f} q={c['q_hat']:.3f} D={c['D_hat']:.3f} U_fleet={c['c4a_fleet_U_expected']:.3f} "
              f"J={c['J_possible']}/{c['m_scenes']} U_pop={c['c4a_pop_U_possible']:.3f}")
    print("OP", {k: a["operating_point"][k] for k in ("n_events", "D_hat", "c4a_fleet_U_expected", "J_possible", "c4a_pop_U_possible")})
    print(C.to_string())
    print("conf", out["confidence"]["twin_score_median_pooled"], out["confidence"]["real_score_median_pooled"], out["confidence"]["by_kind"])


if __name__ == "__main__":
    main()
