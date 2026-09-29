"""A3 — is the negative IW result real or a feature artefact?

Re-runs P0 T4 (BMC-synth 261 Mode-T events -> each real video with >= 30 Mode-T events) with four
feature groups:
  (a) logD                                  temporal only
  (b) logD + area_frac + speed              + geometry
  (c) (b) + score_mean + score_max          + appearance (detector confidence)
  (d) full P0 set (adds onset_pos, video_inevent — both video-level / position artefacts)
video_inevent is dropped from (a)-(c); onset_pos only in (d).
Per group x target x classifier: AUC, mean RAW weight (~1 iff sim covers the target support),
PSIS Pareto k-hat of the raw weights (Zhang & Stephens 2009 fit, Vehtari et al. PSIS tail rule),
n_eff/N (weights clipped at p99 as P0), bias at K=8 with video-level bootstrap CI.
NEW overlap criterion (replaces n_eff/N): PASS iff AUC <= 0.80 AND mean raw weight in [0.5, 2]
AND k-hat < 0.7.
P1b' (standalone #17): events and features are rebuilt from the RAW detection dumps in true temporal order
(corpus17 Mode T); "appearance" = the detector's max box confidence per frame; geometry uses the true frame
size (area_frac == area_true). The P1a run (other project's derived table) is kept as iw_ablation.json.
Output: results/iw_ablation_v2.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import json

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import B_BOOT, DET_ROOT, K_GRID, REAL, RESULTS, SEED, dump, runs
from gstar_common import lasiesta_order, load_dets

MIN_TGT_EVENTS = 30


def crossfit_pi(X, y, kind):
    """5-fold cross-fitted P(target | x); sigmoid-calibrated logistic regression or gradient boosting.
    (Same classifier protocol as the P0 spike, re-implemented here so that #17 is self-contained.)"""
    pi = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, y):
        if kind == "logistic":
            base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        else:
            base = GradientBoostingClassifier(random_state=SEED, n_estimators=150, max_depth=2,
                                              learning_rate=0.05, subsample=0.8)
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        clf.fit(X[tr], y[tr])
        pi[te] = clf.predict_proba(X[te])[:, 1]
    return pi


def frame_sizes():
    size = {}
    for ds in ("CDnet2014", "LASIESTA", "BMC"):
        F = pd.read_parquet(RESULTS / f"gstar_frames_{ds}.parquet", columns=["video", "H", "W"]).dropna().drop_duplicates()
        for v, h, w_ in F.values:
            size[(ds, str(v).split("/")[-1])] = (float(w_), float(h))
    return size


def event_features():
    """Mode-T events (maximal runs of frames with >= 1 box, true temporal order) + per-event features."""
    size = frame_sizes()
    rows = []
    for ds in ("BMC", "CDnet2014", "LASIESTA"):
        for f in sorted((DET_ROOT / ds).glob("*.json")):
            v = f.stem.split("__", 1)[1]
            d = load_dets(ds, f.name)
            idx = sorted(d)
            if ds == "LASIESTA":
                nums = lasiesta_order(v)
                num = np.array([nums[i] for i in idx])
            else:
                num = np.array(idx) + 1
            o = np.argsort(num)
            num = num[o]
            boxes = [d[idx[i]] for i in o]
            pos = np.array([len(b) > 0 for b in boxes])
            W, H = size[(ds, v)]
            diag = np.hypot(W, H)
            inev = float(pos.mean())
            for eid, (i0, i1) in enumerate(runs(num, pos)):
                seg = boxes[i0:i1 + 1]
                conf = [max(b[4] for b in bb) for bb in seg]
                areas = [sum((b[2] - b[0]) * (b[3] - b[1]) for b in bb) / (W * H) for bb in seg]
                cen = [((max(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))[0] + max(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))[2]) / 2,
                        (max(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))[1] + max(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))[3]) / 2) for bb in seg]
                sp = float(np.mean(np.hypot(*np.diff(np.array(cen), axis=0).T))) if len(cen) > 1 else 0.0
                D = int(num[i1] - num[i0] + 1)
                rows.append(dict(dataset=ds, video=v, event_id=eid, D=D, logD=np.log1p(D),
                                 score_mean=float(np.mean(conf)), score_max=float(np.max(conf)),
                                 onset_pos=i0 / len(num), video_inevent=inev,
                                 area_frac=float(np.mean(areas)), area_true=float(np.mean(areas)),
                                 speed=sp, speed_norm=sp / diag, p_miss_K8=max(0.0, 1 - D / 8)))
    return pd.DataFrame(rows)

GROUPS = {
    "a_temporal": ["logD"],
    "b_geometry": ["logD", "area_frac", "speed"],
    "c_appearance": ["logD", "area_frac", "speed", "score_mean", "score_max"],
    "d_full_P0": ["logD", "score_mean", "score_max", "onset_pos", "video_inevent", "area_frac", "speed"],
    # added after the first run: P0 geometry is in raw pixel units (speed px/frame; area / proxy frame size)
    # while BMC is 640x480 and CDnet mostly 320x240 -> check with TRUE frame sizes
    "b1_area_only": ["logD", "area_frac"],
    "b2_geometry_norm": ["logD", "area_true", "speed_norm"],
    "c2_appearance_norm": ["logD", "area_true", "speed_norm", "score_mean", "score_max"],
    # nested order hides redundancy: appearance WITHOUT geometry
    "e_appearance_only": ["logD", "score_mean", "score_max"],
}
AUC_MAX, W_LO, W_HI, K_MAX = 0.80, 0.5, 2.0, 0.7


def gpdfit(x):
    """Zhang & Stephens (2009) generalized Pareto fit with the PSIS weakly-informative prior on k."""
    x = np.sort(x)
    n = len(x)
    prior_bs, prior_k = 3, 10
    m = 30 + int(np.sqrt(n))
    b = 1 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    b /= prior_bs * x[int(n / 4 + 0.5) - 1]
    b += 1 / x[-1]
    k = np.log1p(-b[:, None] * x).mean(axis=1)
    L = n * (np.log(-(b / k)) - k - 1)
    w = 1 / np.exp(L - L[:, None]).sum(axis=1)
    ok = w >= 10 * np.finfo(float).eps
    w, b = w[ok] / w[ok].sum(), b[ok]
    bp = np.sum(b * w)
    kp = np.log1p(-bp * x).mean()
    return (n * kp + prior_k * 0.5) / (n + prior_k)


def pareto_khat(w):
    """PSIS k-hat of raw importance weights w (> 0)."""
    lw = np.log(w) - np.log(w).max()
    S = len(lw)
    M = int(np.ceil(min(0.2 * S, 3 * np.sqrt(S))))
    srt = np.sort(lw)
    cutoff = srt[-M - 1]
    tail = lw[lw > cutoff]
    if len(tail) <= 4:
        return float("inf")
    exc = np.exp(tail) - np.exp(cutoff)
    exc = exc[exc > 0]
    if len(exc) <= 4 or np.allclose(exc, exc[0]):
        return float("nan")
    return float(gpdfit(exc))


def main():
    E = event_features()
    geom_note = dict(frame_size="true (from the raw frames)", n_events={ds: int((E.dataset == ds).sum()) for ds in E.dataset.unique()},
                     feature_medians={ds: {f: float(E.loc[E.dataset == ds, f].median())
                                           for f in ("logD", "area_true", "speed_norm", "score_mean")}
                                      for ds in ("CDnet2014", "LASIESTA", "BMC")})
    sim = E[E.dataset == "BMC"].reset_index(drop=True)
    cnt = E[E.dataset.isin(REAL)].groupby(["dataset", "video"]).size()
    targets = cnt[cnt >= MIN_TGT_EVENTS].index.tolist()
    sim_vids = sim.video.unique()
    sim_idx = {v: np.where(sim.video.values == v)[0] for v in sim_vids}
    P_sim = sim["p_miss_K8"].values
    rng = np.random.default_rng(SEED)

    # self-check of the k-hat code on known tails (exact Pareto with shape 1/alpha)
    r0 = np.random.default_rng(SEED)
    khat_check = {f"pareto_k{k}": float(np.median([pareto_khat((1 - r0.random(4000)) ** (-k)) for _ in range(20)]))
                  for k in (0.3, 0.5, 0.9)}
    khat_check["uniform_weights_(0.5,1.5)"] = pareto_khat(0.5 + r0.random(4000))

    per = []
    for ds, v in targets:
        tgt = E[(E.dataset == ds) & (E.video == v)].reset_index(drop=True)
        rec = dict(dataset=ds, video=v, n_tgt=len(tgt), n_sim=len(sim))
        for gname, feats in GROUPS.items():
            X = np.vstack([sim[feats].values, tgt[feats].values])
            y = np.r_[np.zeros(len(sim)), np.ones(len(tgt))]
            for kind in ("logistic", "gbm"):
                pi = crossfit_pi(X, y, kind)
                auc = float(roc_auc_score(y, pi))
                ps = np.clip(pi[: len(sim)], 1e-6, 1 - 1e-6)
                w_raw = ps / (1 - ps) * len(sim) / len(tgt)
                khat = pareto_khat(w_raw)
                w = np.minimum(w_raw, np.percentile(w_raw, 99))
                neff = w.sum() ** 2 / (w ** 2).sum()
                est = float((w * P_sim).sum() / w.sum())
                truth = float(tgt["p_miss_K8"].mean())
                bs = np.empty(B_BOOT)
                Pt = tgt["p_miss_K8"].values
                for b in range(B_BOOT):
                    ii = np.concatenate([sim_idx[x] for x in rng.choice(sim_vids, len(sim_vids))])
                    jj = rng.integers(0, len(tgt), len(tgt))
                    bs[b] = (w[ii] * P_sim[ii]).sum() / w[ii].sum() - Pt[jj].mean()
                mr = float(w_raw.mean())
                rec[f"{gname}|{kind}"] = dict(
                    auc=auc, mean_raw_weight=mr, khat=khat, n_eff_over_N=float(neff / len(sim)),
                    bias_K8=est - truth, bias_K8_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    overlap_pass=bool(auc <= AUC_MAX and W_LO <= mr <= W_HI and khat < K_MAX),
                    fail_auc=auc > AUC_MAX, fail_w=not (W_LO <= mr <= W_HI), fail_k=not (khat < K_MAX))
        per.append(rec)
        print(ds, v, " ".join(f"{g[0]}:{rec[f'{g}|logistic']['auc']:.2f}/{rec[f'{g}|gbm']['auc']:.2f}" for g in GROUPS))

    def agg(rows):
        out = {}
        for g in GROUPS:
            for kind in ("logistic", "gbm"):
                R = [r[f"{g}|{kind}"] for r in rows]
                a = np.array([x["auc"] for x in R])
                mw = np.array([x["mean_raw_weight"] for x in R])
                kh = np.array([x["khat"] for x in R])
                b8 = np.array([x["bias_K8"] for x in R])
                out[f"{g}|{kind}"] = dict(
                    n_targets=len(R), auc_median=float(np.median(a)), auc_min=float(a.min()), auc_max=float(a.max()),
                    mean_raw_weight_median=float(np.median(mw)),
                    khat_median=float(np.nanmedian(kh)), khat_frac_ge_0_7=float(np.mean(~(kh < K_MAX))),
                    n_eff_over_N_median=float(np.median([x["n_eff_over_N"] for x in R])),
                    bias_K8_abs_median=float(np.median(np.abs(b8))),
                    frac_bias_ci_covers_0=float(np.mean([x["bias_K8_ci95"][0] <= 0 <= x["bias_K8_ci95"][1] for x in R])),
                    frac_overlap_pass=float(np.mean([x["overlap_pass"] for x in R])),
                    frac_fail_auc=float(np.mean([x["fail_auc"] for x in R])),
                    frac_fail_w=float(np.mean([x["fail_w"] for x in R])),
                    frac_fail_k=float(np.mean([x["fail_k"] for x in R])))
        return out

    summ_all = agg(per)
    summ_cd = agg([r for r in per if r["dataset"] == "CDnet2014"])
    # where does the gap live? AUC increments between nested groups (median over targets, logistic)
    steps = {}
    for kind in ("logistic", "gbm"):
        med = {g: summ_all[f"{g}|{kind}"]["auc_median"] for g in GROUPS}
        steps[kind] = dict(temporal_alone=med["a_temporal"],
                           plus_geometry=med["b_geometry"] - med["a_temporal"],
                           plus_appearance=med["c_appearance"] - med["b_geometry"],
                           plus_video_level_artefacts=med["d_full_P0"] - med["c_appearance"])
        steps[kind + "_normalized_geometry"] = dict(
            temporal_alone=med["a_temporal"], plus_area_raw=med["b1_area_only"] - med["a_temporal"],
            plus_geometry_true_size=med["b2_geometry_norm"] - med["a_temporal"],
            plus_appearance=med["c2_appearance_norm"] - med["b2_geometry_norm"])
        steps[kind + "_appearance_first"] = dict(
            temporal_alone=med["a_temporal"], plus_appearance_only=med["e_appearance_only"] - med["a_temporal"],
            plus_geometry_after_appearance=med["c2_appearance_norm"] - med["e_appearance_only"])
    any_pass = {k: v["frac_overlap_pass"] for k, v in summ_all.items()}
    dump(dict(meta=dict(groups=GROUPS, criterion=f"PASS iff AUC <= {AUC_MAX} and mean raw weight in "
                        f"[{W_LO}, {W_HI}] and PSIS k-hat < {K_MAX}", source="BMC-synth 261 Mode-T events",
                        targets=f"real videos with >= {MIN_TGT_EVENTS} Mode-T events", n_targets=len(per),
                        bootstrap=f"B={B_BOOT} seed={SEED}", khat="Zhang-Stephens GPD fit on the top "
                        "min(0.2S, 3 sqrt S) raw weights, PSIS prior on k",
                        standalone="corpus17 Mode T from raw dumps, true order; appearance = detector max confidence"),
              khat_selfcheck=khat_check, geometry_units=geom_note, summary_all_targets=summ_all, summary_cdnet_targets=summ_cd,
              auc_steps=steps, frac_pass_by_group=any_pass, per_target=per), "iw_ablation_v2.json")
    for k, v in summ_all.items():
        print(f"{k:28s} AUC {v['auc_median']:.3f} w {v['mean_raw_weight_median']:.3f} k {v['khat_median']:.2f} "
              f"pass {v['frac_overlap_pass']:.2f} |bias8| {v['bias_K8_abs_median']:.3f}")
    print(steps, khat_check)


if __name__ == "__main__":
    main()
