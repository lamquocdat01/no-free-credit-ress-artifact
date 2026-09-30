"""P1c E0 — three opening conditions, recomputed from the existing twin/real per-frame tables (no detector run).

E0.1 units. Every tier number of twin_tiers.json is re-derived with numerator / denominator / unit stated.
     The printed D-hat (m2_discordance.py) is  X / N  with N = #events (d >= d_min) and
     X = sum_e mean_v |R_twin,v(K) \\ R_real(K)| / K  = EXPECTED number of discordant events under a uniform
     random sampling phase, averaged over the 3 sprite variants (fractional numerator).
     Canonical unit fixed here = EVENT. Twin decision at phase phi = majority of the 3 sprite variants
     (twin catches if >= r of 3 variants catch; r = 2 canonical, r = 1 and r = 3 sensitivity).
       expected form : x_e = P_phi(real miss AND twin catch_r)          (exact C3 quantity, fractional count)
       worst-phase   : w_e = max_phi 1[real miss AND twin catch_r]      (integer, conservative: x_e <= w_e)
     C4a-fleet U = U_CP(sum, N) (one-sided, delta 5%); C4a-pop U = U_CP(J, m), J = #scenes with some w_e = 1.
     Event waterfall P1b (257 / 64) -> P1b' (207 / 86), one number per step.
E0.2 "right because empty?": 2x2 real miss / catch x twin miss / catch (expected event counts under the
     uniform phase, majority twin) per tier and per scene at OP*, OP-A, OP-C; recall of the twin on real misses.
     Misses are split into STRUCTURAL (the event has no frame in the sampled residue class: duration < K) and
     APPEARANCE (frames exist, the detector fails on all of them). Only appearance misses test the twin.
     Grid search of the nearest operating point (larger K / smaller d_min) with >= 20 real miss events.
E0.3 geometry: D-hat and q-hat by share of width-clamped frames per event (0, <= 10 %, > 10 %) and by class;
     |aspect error| (sprite aspect / GT-box aspect - 1, before the clamp) by camera group.
Outputs: results/e0_units.json, results/e0_event_waterfall.json, results/e0_confusion.json,
         results/e0_geometry_sensitivity.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c4_power import ucp  # noqa: E402
from common import RESULTS, SEED, dump  # noqa: E402

KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]          # divisors of 48: exact under the twin's mod-48 early stop
OPS = {"OP*": dict(eps=0.20, d_min=32, K=16), "OP-A": dict(eps=0.20, d_min=32, K=8),
       "OP-C": dict(eps=0.10, d_min=32, K=1)}
# v1.1 (post-F1 correction, 30-09-2026): the C0 run covered the 78 scenes of the v1.0 main domain; its 2 cameraJitter
# scenes (boulevard, traffic) now form the jitter tier. event_rows keeps only the videos of the requested tier
# (results/p1_background17.json by_tier), so "main" = 76 scenes and "jitter" = 2 scenes of the same run.
TIER_TAGS = {"main": ["c0_main"], "jitter": ["c0_main"], "night": ["nheld_b", "ntune_b"], "turbulence": ["ttune_base"]}
CHALLENGE = (16, 16)       # post-hoc point fixed in v1.0 (E0.2); kept fixed in v1.1, not re-searched

# Camera group for E0.3 (judgement from the dataset descriptions / first frame; no metadata exists):
# 'elevated' = looks down on the scene from a pole / building / bridge; 'level' = roughly eye/body height.
ELEVATED = {"boulevard", "busStation", "overpass", "park", "parking", "pedestrians", "streetLight", "traffic",
            "tramCrossroad_1fps", "tunnelExit_0_35fps", "turnpike_0_5fps", "winterDriveway", "blizzard", "snowFall",
            "wetSnow", "skating", "port_0_17fps", "boats", "canoe", "fall", "fountain01", "fountain02", "lakeSide",
            "backdoor", "bungalows"}


# ----------------------------------------------------------------------------------------------- data
def real_hit_frames(E, BW):
    out = {}
    for ds in E.dataset.unique():
        F = pd.read_parquet(RESULTS / f"gstar_frames_{ds}.parquet", columns=["video", "frame", "has_fg", "hit", "L_bbox"])
        F = F[F.video.isin(E[E.dataset == ds].video.unique())]
        hit = F.has_fg.astype(bool) & F.hit.astype("boolean").fillna(False).astype(bool)
        byv = {v: set(g.frame.values[hit.loc[g.index].values]) for v, g in F.groupby("video")}
        for v, t, b in F[["video", "frame", "L_bbox"]].dropna().values:
            x = list(map(int, b.split(",")))
            BW[(v, int(t))] = max(2, x[2] - x[0])
        for e in E[E.dataset == ds].itertuples():
            fr = range(int(e.onset_frame), int(e.onset_frame) + int(e.duration))
            out[(ds, e.video, int(e.event_id))] = np.array([t for t in fr if t in byv.get(e.video, ())], int)
    return out


def tier_videos(tier):
    return set(json.load(open(RESULTS / "p1_background17.json", encoding="utf-8"))["by_tier"][tier])


def event_rows(tag, tier, definition="a256"):
    """One row per (event); arrays per K of shape (K,) for real hit and (3, K) for twin hit by phase.
    Only the videos of `tier` (p1_background17.json by_tier) are kept.
    definition: 'a256' (primary: largest object >= 256 px) or 'orig' (all GT events, sensitivity)."""
    T = pd.read_parquet(RESULTS / f"gstar_frames_twin_{tag}.parquet")
    T = T[T.video.isin(tier_videos(tier))]
    E = pd.read_csv(RESULTS / "miss_matrix_Gstar_a256.csv")
    if definition == "a256":
        T = T[T.def_a256]
        E = E[E.def_a256]
    E = E[E.video.isin(T.video.unique())]
    BW = {}
    RH = real_hit_frames(E, BW)
    rows = []
    for (ds, v, eid), g in T.groupby(["dataset", "video", "event_id"]):
        e = E[(E.dataset == ds) & (E.video == v) & (E.event_id == eid)].iloc[0]
        fr_all = np.arange(int(e.onset_frame), int(e.onset_frame) + int(e.duration))
        rr = RH[(ds, v, int(eid))]
        r = dict(tier=tier, tag=tag, dataset=ds, category=g.category.iloc[0], video=v, event_id=int(eid),
                 duration=int(e.duration), kind=g.kind.iloc[0], n_var=int(g.variant.nunique()))
        # geometry per event (variant-pooled): share of frames whose width was clamped, |aspect error|
        W = np.array([BW[(v, int(t))] for t in g.frame.values])      # same rule as m2_discordance.geometry_check
        r["clamp_share"] = float(np.mean(np.abs(g.w_ratio.values * W - W) >= 0.1 * W - 0.5))
        r["abs_aspect_err"] = float(np.median(np.abs(g.aspect_err.values)))
        for K in KS:
            cov = np.zeros(K, bool)
            cov[np.unique(fr_all % K)] = True                       # residue classes that contain a frame
            rh = np.zeros(K, bool)
            rh[np.unique(rr % K)] = True
            th = np.zeros((3, K), bool)
            for vi, gv in g.groupby("variant"):
                th[int(vi), np.unique(gv.frame.values[gv.twin_hit.values.astype(bool)] % K)] = True
            r[f"cov{K}"], r[f"rh{K}"], r[f"th{K}"] = cov, rh, th
        rows.append(r)
    return rows


# ----------------------------------------------------------------------------------------------- metrics
def cells(r, K, rule=2):
    """Per-phase indicators for one event: real miss, twin catch (majority rule), structural phase."""
    rm = ~r[f"rh{K}"]
    tc = r[f"th{K}"].sum(0) >= rule
    st = ~r[f"cov{K}"]
    return rm, tc, st


def ev_metrics(rows, d_min, K, rule=2):
    out = []
    for r in rows:
        if r["duration"] < d_min:
            continue
        rm, tc, st = cells(r, K, rule)
        disc = rm & tc
        out.append(dict(video=r["video"], tier=r["tier"], category=r["category"], kind=r["kind"],
                        clamp_share=r["clamp_share"], abs_aspect_err=r["abs_aspect_err"],
                        x=disc.mean(), w=float(disc.any()),
                        rm=rm.mean(), rm_any=float(rm.any()), tm=(~tc).mean(),
                        rm_tm=(rm & ~tc).mean(), rc_tc=(~rm & tc).mean(), rc_tm=(~rm & ~tc).mean(),
                        rm_app=(rm & ~st).mean(), rm_app_any=float((rm & ~st).any()),
                        rm_tm_app=(rm & ~tc & ~st).mean(), rm_tc_app=(rm & tc & ~st).mean(),
                        rm_struct=(rm & st).mean(),
                        ev_rm_app=float((rm & ~st).any()), ev_rm_app_twin_any=float((rm & ~st & ~tc).any()),
                        ev_rm_app_twin_all=float((rm & ~st).any() and not (rm & ~st & tc).any()),
                        x_var=float(np.mean([(rm & r[f"th{K}"][v]).mean() for v in range(3)]))))
    return pd.DataFrame(out)


def summary(df, eps=0.20):
    if not len(df):
        return dict(N=0)
    N, m = len(df), df.video.nunique()
    Xe, Xw = float(df.x.sum()), int(df.w.sum())
    J = int((df.groupby("video").w.max() > 0).sum())
    Xv = float(df.x_var.sum())
    return dict(
        N_events=N, m_scenes=m,
        D_expected=dict(num=Xe, den=N, unit="expected discordant events (uniform phase, majority twin) / events",
                        value=Xe / N, U_fleet=float(ucp(Xe, N))),
        D_worst_phase=dict(num=Xw, den=N, unit="events discordant at >= 1 phase (majority twin) / events",
                           value=Xw / N, U_fleet=float(ucp(Xw, N))),
        D_old_variant_mean=dict(num=Xv, den=N, unit="expected discordant events, mean over 3 variants / events "
                                                    "(= printed P1b' number)", value=Xv / N, U_fleet=float(ucp(Xv, N))),
        J=dict(num=J, den=m, unit="scenes with >= 1 worst-phase discordant event / scenes", U_pop=float(ucp(J, m))),
        G2=dict(fleet_ok=bool(max(ucp(Xe, N), ucp(Xw, N)) <= eps / 2), pop_ok=bool(ucp(J, m) <= eps / 2)))


def confusion(df):
    if not len(df):
        return dict(N=0)
    N = len(df)
    c = {k: float(df[k].sum()) for k in ("rm_tm", "x", "rc_tc", "rc_tm")}
    RM = c["rm_tm"] + c["x"]
    app = float(df.rm_app.sum())
    return dict(
        N_events=N, m_scenes=int(df.video.nunique()),
        unit="expected event counts under a uniform random sampling phase; twin = majority of 3 sprite variants",
        cells={"real_miss & twin_miss": c["rm_tm"], "real_miss & twin_catch (D)": c["x"],
               "real_catch & twin_catch": c["rc_tc"], "real_catch & twin_miss (pessimistic)": c["rc_tm"]},
        p_hat=RM / N, q_hat=(c["rm_tm"] + c["rc_tm"]) / N, D_hat=c["x"] / N, reverse_hat=c["rc_tm"] / N,
        real_miss_events_expected=RM, real_miss_events_any_phase=int(df.rm_any.sum()),
        twin_recall_on_real_miss=(c["rm_tm"] / RM) if RM > 0 else None,
        appearance=dict(real_miss_expected=app, real_miss_any_phase=int(df.rm_app_any.sum()),
                        twin_miss=float(df.rm_tm_app.sum()), twin_catch=float(df.rm_tc_app.sum()),
                        twin_recall=(float(df.rm_tm_app.sum()) / app) if app > 0 else None,
                        note="real miss at a phase whose residue class contains >= 1 event frame (detector failure)",
                        events_real_miss_any_phase=int(df.ev_rm_app.sum()),
                        events_twin_also_miss_at_some_missed_phase=int(df.ev_rm_app_twin_any.sum()),
                        events_twin_miss_at_every_missed_phase=int(df.ev_rm_app_twin_all.sum()),
                        scenes_with_real_miss=int(df[df.ev_rm_app > 0].video.nunique()),
                        scenes_where_twin_sees_a_real_miss=int(df[df.ev_rm_app_twin_any > 0].video.nunique())),
        structural_real_miss_expected=float(df.rm_struct.sum()))


# ----------------------------------------------------------------------------------------------- waterfall
def waterfall(tiers_rows):
    p1b = json.load(open(RESULTS / "twin_discordance.json", encoding="utf-8"))
    bg = json.load(open(RESULTS / "p1_background17.json", encoding="utf-8"))
    T1 = pd.read_parquet(RESULTS / "gstar_frames_twin.parquet", columns=["dataset", "category", "video", "event_id", "def_a256"])
    ev1 = T1.drop_duplicates(["dataset", "video", "event_id"])
    E = pd.read_csv(RESULTS / "miss_matrix_Gstar_a256.csv")
    tier = {v: t for t, vs in bg["by_tier"].items() for v in vs}
    las_old = set(bg["comparison_with_P1b_lists"]["lasiesta_P1b_list"])
    cd_p1b = set(ev1[ev1.dataset == "CDnet2014"].video)
    a = p1b["by_definition"]
    steps = [dict(step="P1b twin, original definition (all GT events, all durations)", events=a["orig"]["n_events"],
                  scenes=a["orig"]["n_scenes"], source="twin_discordance.json by_definition.orig.n_events"),
             dict(step="P1b, A >= 256 (primary definition)", events=a["a256"]["n_events"], scenes=a["a256"]["n_scenes"],
                  source="twin_discordance.json by_definition.a256.n_events (= the '257 events / 64 scenes' of P1b)")]
    A = E[E.def_a256]

    def cnt(mask, d_min=0):
        s = A[mask & (A.duration >= d_min)]
        return int(len(s)), int(s.video.nunique())
    in_cd = (A.dataset == "CDnet2014") & A.video.isin(cd_p1b)
    n, m = cnt(in_cd | ((A.dataset == "LASIESTA") & A.video.isin(las_old)))
    steps.append(dict(step="same 64 scenes, LASIESTA events rebuilt in TRUE frame order (B0 fix)", events=n, scenes=m,
                      source="miss_matrix_Gstar_a256.csv (corpus17), CDnet 44 + LASIESTA old 20"))
    n, m = cnt(in_cd | (A.dataset == "LASIESTA") & A.video.map(tier).eq("main"))
    steps.append(dict(step="LASIESTA list 20 -> 48 scenes (GT-empty criterion, true order)", events=n, scenes=m,
                      source="p1_background17.json lasiesta_gt_empty_list"))
    keep = A.video.map(tier).isin(["main", "jitter", "night", "turbulence"])
    n, m = cnt(keep)
    steps.append(dict(step="PTZ excluded (4 scenes)", events=n, scenes=m, source="p1_background17.json by_tier"))
    run = set(r["video"] for rows in tiers_rows.values() for r in rows)
    n, m = cnt(keep & A.video.isin(run))
    steps.append(dict(step="scenes actually run in P1b' (turbulence held-out turbulence2/3 not run, decision 28-09)",
                      events=n, scenes=m, source="twin runs c0_main + nheld_b + ntune_b + ttune_base"))
    n, m = cnt(A.video.map(tier).eq("jitter"), d_min=32)
    jitter_OP = dict(events=n, scenes=m)
    n, m = cnt(keep & A.video.isin(run), d_min=32)
    steps.append(dict(step="OP* event definition: duration >= d_min 32 frames", events=n, scenes=m,
                      source="= twin_tiers.json all_tiers_final.n_events_OP"))
    n_main, m_main = cnt(A.video.map(tier).eq("main"), d_min=32)
    note = ("The two headline numbers are in DIFFERENT units: 257 = all A>=256 events of any duration (P1b), "
            "207 = A>=256 events with duration >= 32 (P1b' OP*). On the same basis P1b had "
            f"{p1b['by_definition']['a256']['operating_point']['N']} OP* events / 64 scenes.")
    return dict(steps=steps, main_domain_OP=dict(events=n_main, scenes=m_main), jitter_tier_OP=jitter_OP, note=note,
                p1b_OP_events=p1b["by_definition"]["a256"]["operating_point"]["N"])


def frame_level_geometry(tag="c0_main"):
    """Twin vs real G* hit on the SAME frames, split by clamp and by |aspect error| (main domain, A>=256).
    Paired test: per (event, variant) twin hit rate on clamped vs unclamped frames (Wilcoxon), real on the same frames."""
    from scipy.stats import wilcoxon
    T = pd.read_parquet(RESULTS / f"gstar_frames_twin_{tag}.parquet")
    T = T[T.def_a256].copy()
    BW = {}
    for ds in T.dataset.unique():
        F = pd.read_parquet(RESULTS / f"gstar_frames_{ds}.parquet", columns=["video", "frame", "L_bbox"]).dropna()
        for v, t, b in F.values:
            x = list(map(int, b.split(",")))
            BW[(v, int(t))] = max(2, x[2] - x[0])
    W = np.array([BW[(v, int(t))] for v, t in T[["video", "frame"]].values])
    T["clamped"] = np.abs(T.w_ratio.values * W - W) >= 0.1 * W - 0.5
    T["abin"] = pd.cut(T.aspect_err.abs(), [-1, 0.1, 0.2, 0.4, np.inf], labels=["<=10%", "10-20%", "20-40%", ">40%"])
    T["real_hit"] = T.real_hit.astype(bool)
    T["twin_hit"] = T.twin_hit.astype(bool)
    out = dict(unit="twin frame rows (event x variant x frame, early-stopped at 48 covered residues)",
               n_rows=len(T), clamp_share=float(T.clamped.mean()),
               by_clamp={str(k): dict(n=len(g), twin_hit=float(g.twin_hit.mean()), real_hit=float(g.real_hit.mean()))
                         for k, g in T.groupby("clamped")},
               by_kind_aspect={f"{a}/{b}": dict(n=len(g), twin_hit=float(g.twin_hit.mean()), real_hit=float(g.real_hit.mean()),
                                                  twin_minus_real=float(g.twin_hit.mean() - g.real_hit.mean()))
                               for (a, b), g in T.groupby(["kind", "abin"], observed=True)})
    pt = T.groupby(["video", "event_id", "variant", "clamped"]).twin_hit.mean().unstack().dropna()
    pr = T.groupby(["video", "event_id", "variant", "clamped"]).real_hit.mean().unstack().dropna()
    w = wilcoxon(pt[True], pt[False])
    out["paired_within_event"] = dict(n_event_variants=len(pt), twin_unclamped=float(pt[False].mean()),
                                      twin_clamped=float(pt[True].mean()), real_unclamped=float(pr[False].mean()),
                                      real_clamped=float(pr[True].mean()),
                                      extra_twin_drop_pts=float(100 * ((pt[False].mean() - pt[True].mean()) -
                                                                       (pr[False].mean() - pr[True].mean()))),
                                      wilcoxon_twin_p=float(w.pvalue))
    return out


# ----------------------------------------------------------------------------------------------- main
def main():
    tiers_rows = {t: [r for tag in tags for r in event_rows(tag, t)] for t, tags in TIER_TAGS.items()}
    allrows = [r for rows in tiers_rows.values() for r in rows]
    tiers_rows_all = dict(tiers_rows, all_tiers=allrows)

    # ---------------- E0.1
    old = json.load(open(RESULTS / "twin_tiers.json", encoding="utf-8"))
    units = dict(meta=dict(doc=__doc__.split("E0.2")[0].strip(), seed=SEED,
                           canonical_unit="EVENT; twin catch = >= 2 of 3 sprite variants; phase uniform (expected form) "
                                          "and worst phase (integer, conservative)"),
                 printed_numbers_explained={}, recomputed={})
    for t in ("night", "turbulence"):                     # v1.0 "main" (78 scenes) no longer exists as a tier
        o = old[t]["all_final"]
        units["printed_numbers_explained"][t] = dict(
            D_hat=dict(value=o["D_hat"], num=o["X_expected"], den=o["n_events_OP"],
                       unit="events (d >= 32); numerator = sum over events of mean over 3 sprite variants of the "
                            "fraction of the K=16 phases with real miss AND twin catch (expected count, fractional)"),
            c4a_fleet_U=dict(value=o["c4a_fleet_U"], formula="U_CP(X_expected, N) one-sided delta 5%, fractional X"),
            J=dict(value=o["J"], den=o["n_scenes"], unit="scenes with some event having D_e > 0 in ANY variant (1/3 rule)"),
            c4a_pop_U=dict(value=o["c4a_pop_U"], formula="U_CP(J, m)"))
    for t, rows in tiers_rows_all.items():
        units["recomputed"][t] = {f"rule_{r}of3": summary(ev_metrics(rows, 32, 16, r)) for r in (2, 1, 3)}
    mm = units["recomputed"]["main"]["rule_2of3"]
    disc_ev = []
    for r in tiers_rows["main"]:
        if r["duration"] >= 32:
            rm, tc, _ = cells(r, 16)
            if (rm & tc).any():
                disc_ev.append(dict(video=r["video"], event_id=r["event_id"], duration=r["duration"], kind=r["kind"],
                                    phases_discordant=int((rm & tc).sum()), phases_real_miss=int(rm.sum()),
                                    per_variant_phases=[int((rm & r["th16"][v]).sum()) for v in range(3)]))
    v10 = summary(ev_metrics(tiers_rows["main"] + tiers_rows["jitter"], 32, 16, 1))
    ok = abs(v10["D_old_variant_mean"]["value"] - old["main"]["D_hat"]) < 1e-12
    units["check_reproduces_printed_main_D"] = bool(ok)
    units["check_reproduces_printed_main_D_note"] = "v1.0 main (78) = v1.1 main (76) + jitter (2)"
    units["erratum"] = dict(
        main=dict(printed_v10_78_scenes=dict(D_hat=old["main"]["D_hat"], U_fleet=old["main"]["c4a_fleet_U"], J=old["main"]["J"],
                               U_pop=old["main"]["c4a_pop_U"]),
                  canonical=dict(D_expected=mm["D_expected"]["value"], U_fleet_expected=mm["D_expected"]["U_fleet"],
                                 D_worst=mm["D_worst_phase"]["value"], U_fleet_worst=mm["D_worst_phase"]["U_fleet"],
                                 J=mm["J"]["num"], U_pop=mm["J"]["U_pop"])),
        discordant_events_main=disc_ev,
        note="0.17% is a per-EVENT rate with a fractional numerator: 0.3125 EXPECTED discordant events / 180 events "
             "(expectation over the 16 sampling phases and the 3 sprite variants). The majority rule does not change it "
             "(the 3 variants agree on these events). In integer form (event discordant at >= 1 phase) the numerator "
             "is the count in D_worst_phase; the certificate is then computed on that integer count.")
    dump(units, "e0_units.json")
    dump(waterfall(tiers_rows), "e0_event_waterfall.json")

    # ---------------- E0.2
    conf = dict(meta=dict(doc=__doc__.split("E0.2")[1].split("E0.3")[0].strip()), by_op={}, grid_main={}, per_scene_OPstar={})
    for name, op in OPS.items():
        conf["by_op"][name] = dict(op=op, **{t: confusion(ev_metrics(rows, op["d_min"], op["K"]))
                                              for t, rows in tiers_rows_all.items()})
    grid = []
    for d_min in (32, 24, 16, 12, 8, 4, 1):
        for K in (8, 16, 24, 48):
            c = confusion(ev_metrics(tiers_rows["main"], d_min, K))
            s = summary(ev_metrics(tiers_rows["main"], d_min, K))
            grid.append(dict(d_min=d_min, K=K, N=c["N_events"], real_miss_exp=c["real_miss_events_expected"],
                             real_miss_app_exp=c["appearance"]["real_miss_expected"],
                             real_miss_app_any=c["appearance"]["real_miss_any_phase"],
                             twin_recall=c["twin_recall_on_real_miss"], twin_recall_app=c["appearance"]["twin_recall"],
                             D_hat=c["D_hat"], U_fleet_worst=s["D_worst_phase"]["U_fleet"], J=s["J"]["num"], U_pop=s["J"]["U_pop"]))
    G = pd.DataFrame(grid)
    conf["grid_main"] = dict(rows=grid, criterion=">= 20 real APPEARANCE-miss events in the main domain, counted as "
                             "events missed at >= 1 phase (the expected count never reaches 20 on this grid: max "
                             f"{G.real_miss_app_exp.max():.2f})")
    hit = G[G.real_miss_app_any >= 20]
    near_K = hit[hit.d_min == 32].sort_values("K").head(1)
    near_d = hit[hit.K == 16].sort_values("d_min", ascending=False).head(1)
    conf["nearest_challenged"] = dict(
        along_K_at_dmin32=near_K.to_dict("records")[0] if len(near_K) else None,
        along_dmin_at_K16=near_d.to_dict("records")[0] if len(near_d) else None,
        any=hit.sort_values(["real_miss_app_exp"]).head(3).to_dict("records"))
    conf["nearest_challenged"]["v11_note"] = ("challenged_along_dmin is FIXED at the v1.0 post-hoc point (16,16); "
                                              "the search result above is reported only")
    near_d = G[(G.d_min == CHALLENGE[0]) & (G.K == CHALLENGE[1])]
    for label, rec in (("along_K", near_K), ("along_dmin", near_d)):
        if len(rec):
            d, K = int(rec.d_min.iloc[0]), int(rec.K.iloc[0])
            conf[f"challenged_{label}"] = dict(op=dict(d_min=d, K=K), main=confusion(ev_metrics(tiers_rows["main"], d, K)),
                                               main_cert=summary(ev_metrics(tiers_rows["main"], d, K)))
    ps = ev_metrics(tiers_rows["main"], 32, 16).groupby("video")[["rm", "x", "rm_tm", "rc_tm", "rm_app"]].sum()
    conf["per_scene_OPstar"] = ps[(ps > 0).any(axis=1)].reset_index().to_dict("records")
    c0 = conf["by_op"]["OP*"]["main"]
    conf["verdict"] = dict(
        real_miss_events_main_OPstar=c0["real_miss_events_expected"],
        real_miss_events_main_OPstar_appearance=c0["appearance"]["real_miss_expected"],
        challenged=bool(c0["real_miss_events_expected"] >= 10),
        text=("chứng chỉ miền chính chưa được thử thách ở OP*" if c0["real_miss_events_expected"] < 10 else "tested"))
    dump(conf, "e0_confusion.json")

    # ---------------- E0.3
    geo = dict(meta=dict(doc=__doc__.split("E0.3")[1].split("Outputs")[0].strip(),
                         camera_groups="elevated = " + ", ".join(sorted(ELEVATED)) + "; LASIESTA and the rest = level "
                                       "(judgement from dataset descriptions, no metadata)"), by_op={})
    for name, (d, K) in {"OP*": (32, 16), "challenged": (None, None)}.items():
        if name == "challenged":
            ch = conf.get("challenged_along_dmin") or conf.get("challenged_along_K")
            if not ch:
                continue
            d, K = ch["op"]["d_min"], ch["op"]["K"]
        df = ev_metrics(tiers_rows["main"], d, K)
        df["clamp_grp"] = pd.cut(df.clamp_share, [-1e-9, 1e-9, 0.10, 1.0], labels=["a_none", "b_le10pct", "c_gt10pct"])
        df["camera"] = np.where(df.video.isin(ELEVATED), "elevated", "level")

        def grp(col):
            return {str(k): dict(N=len(g), D_hat=float(g.x.sum() / len(g)), D_num=float(g.x.sum()),
                                 q_hat=float(g.tm.mean()), p_hat=float(g.rm.mean()),
                                 twin_recall_app=(float(g.rm_tm_app.sum() / g.rm_app.sum()) if g.rm_app.sum() > 0 else None),
                                 abs_aspect_err_median=float(g.abs_aspect_err.median()))
                    for k, g in df.groupby(col, observed=True)}
        geo["by_op"][name] = dict(op=dict(d_min=d, K=K), by_clamp=grp("clamp_grp"), by_kind=grp("kind"),
                                  by_camera=grp("camera"),
                                  by_camera_kind={f"{a}/{b}": v for (a, b), g in df.groupby(["camera", "kind"])
                                                  for v in [dict(N=len(g), abs_aspect_err_median=float(g.abs_aspect_err.median()),
                                                                 clamp_share_mean=float(g.clamp_share.mean()),
                                                                 q_hat=float(g.tm.mean()), D_hat=float(g.x.mean()))]})
    geo["note_groups"] = ("the pre-specified event groups (a) no clamp / (b) <= 10% / (c) > 10% of frames clamped are "
                          "degenerate: almost every event has > 10% clamped frames (counts in by_clamp) -> the "
                          "frame-level paired analysis below is the usable test")
    geo["frame_level"] = frame_level_geometry()
    dump(geo, "e0_geometry_sensitivity.json")
    print(json.dumps(dict(units=units["erratum"], wf=[(s["events"], s["scenes"]) for s in waterfall(tiers_rows)["steps"]],
                          verdict=conf["verdict"], near=conf["nearest_challenged"]), indent=1, default=float))


if __name__ == "__main__":
    main()
