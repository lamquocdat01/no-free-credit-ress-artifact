"""A1 — Mode G* (object-level) on CDnet2014, from the existing detection JSON + pixel GT.

GT event  = maximal run of frames with >= 1 GT pixel == 255 inside the temporalROI, gaps <= 1 merged
            (primary, as P0 "gap1"); gap0 reported as sensitivity.
Mode G    = frame-level definition (frame alert = the detector returned >= 1 box; TP = alert on a GT frame),
            re-evaluated on the SAME events. Computed from the raw detection JSON (P1b': no derived table of
            another project is read; P1a verified alert == any box on 118,173/118,173 frames).
Mode G*   = frame is a HIT iff a predicted box covers >= 50% of the largest GT component OR has
            IoU >= 0.3 with its bbox (gstar_common.eval_frame). Both rules reported separately.
p_miss(e, K) exact over the K phases (as P0). Detector is NOT re-run.
Outputs: results/miss_matrix_Gstar.csv, results/m1_gstar_summary.json,
         results/gstar_frames_CDnet2014.parquet (per-frame, for A6 / P1b).
"""
from __future__ import annotations

import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from common import B_BOOT, DET_ROOT, K_GRID, RESULTS, SEED, cp_ci, dump
from gstar_common import (CDNET_ROOT, class_group, eval_frame, fg_runs, label_binary, load_dets,
                          pmiss_residue)


def roi(cat, v):
    a, b = (CDNET_ROOT / cat / v / "temporalROI.txt").read_text().split()[:2]
    return int(a), int(b)


def video_frames(args):
    cat, v = args
    a, b = roi(cat, v)
    dets = load_dets("CDnet2014", f"{cat}__{v}.json")
    rows = []
    for t in range(a, b + 1):
        gt = cv2.imread(str(CDNET_ROOT / cat / v / "groundtruth" / f"gt{t:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            rows.append(dict(category=cat, video=v, frame=t, gt_missing=True, has_fg=False))
            continue
        fg = gt == 255
        has = bool(fg.any())
        boxes = dets.get(t - 1)                     # JSON is 0-based sort index -> file number t
        r = dict(category=cat, video=v, frame=t, gt_missing=False, has_fg=has,
                 det_missing=boxes is None, n_boxes=len(boxes) if boxes else 0,
                 H=gt.shape[0], W=gt.shape[1])
        if has:
            lab, n = label_binary(fg)
            e = eval_frame(lab, n, boxes or [])
            e["L_bbox"] = None if e["L_bbox"] is None else ",".join(map(str, e["L_bbox"]))
            r.update(e)
        rows.append(r)
    return rows


def build_events(F, P, merge_gap):
    """F: per-frame G* table of one video (sorted), P: frame -> 'TP' map of Mode G (raw boxes)."""
    fr = F.frame.values
    has = F.has_fg.values.astype(bool)
    ev = []
    for eid, (i0, i1) in enumerate(fg_runs(fr, has, merge_gap)):
        seg = F.iloc[i0:i1 + 1]
        segf = seg.frame.values
        g_tp = np.array([P.get(int(t)) == "TP" for t in segf])
        fgm = seg.has_fg.values.astype(bool)
        hit = seg.hit.fillna(False).values.astype(bool) & fgm
        hit_c = seg.hit_cover.fillna(False).values.astype(bool) & fgm
        hit_i = seg.hit_iou.fillna(False).values.astype(bool) & fgm
        hit_a = seg.hit_any.fillna(False).values.astype(bool) & fgm
        detm = seg.det_missing.fillna(False).values.astype(bool)
        cls = seg.cls.values[hit]
        cls_a = seg.cls_any.values[hit_a]
        cls_vote = pd.Series(cls).mode().iloc[0] if len(cls) else -1
        cls_vote_any = pd.Series(cls_a).mode().iloc[0] if len(cls_a) else -1
        r = dict(dataset="CDnet2014", category=F.category.iloc[0], video=F.video.iloc[0],
                 merge_gap=merge_gap, event_id=eid, onset_frame=int(segf[0]), end_frame=int(segf[-1]),
                 duration=int(segf[-1] - segf[0] + 1), n_fg=int(fgm.sum()),
                 n_tp_G=int(g_tp.sum()), n_hit=int(hit.sum()), n_hit_cover=int(hit_c.sum()),
                 n_hit_iou=int(hit_i.sum()), n_hit_any=int(hit_a.sum()),
                 n_tp_G_not_hit=int((g_tp & ~hit & ~detm).sum()), n_det_missing=int(detm.sum()),
                 cls=int(cls_vote), cls_group=class_group(int(cls_vote)),
                 cls_any=int(cls_vote_any), cls_group_any=class_group(int(cls_vote_any)),
                 L_area_median=float(np.nanmedian(seg.L_area.values[fgm].astype(float))),
                 L_area_max=float(np.nanmax(seg.L_area.values[fgm].astype(float))),
                 frame_area=int(F.H.iloc[0] * F.W.iloc[0]))
        for K in K_GRID:
            r[f"pG_K{K}"] = pmiss_residue(segf[g_tp], K)
            r[f"pGs_K{K}"] = pmiss_residue(segf[hit], K)
            r[f"pGsC_K{K}"] = pmiss_residue(segf[hit_c], K)
            r[f"pGsI_K{K}"] = pmiss_residue(segf[hit_i], K)
            r[f"pGsA_K{K}"] = pmiss_residue(segf[hit_a], K)
        ev.append(r)
    return ev


def pooled(df, col, rng, by="video"):
    """p_hat(K) with CP (fractional expected count) and video-level bootstrap CI."""
    vids = df[by].unique()
    idx = {v: np.where(df[by].values == v)[0] for v in vids}
    out = {}
    for K in K_GRID:
        x = df[f"{col}_K{K}"].values
        n = len(x)
        bs = np.empty(B_BOOT)
        for b in range(B_BOOT):
            ii = np.concatenate([idx[v] for v in rng.choice(vids, len(vids), replace=True)])
            bs[b] = x[ii].mean()
        out[f"K{K}"] = dict(p_hat=float(x.mean()), cp95=cp_ci(float(x.sum()), n),
                            boot95_video=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    return out


def main():
    pd.set_option("future.no_silent_downcasting", True)
    vids = sorted([p.stem.split("__", 1) for p in (DET_ROOT / "CDnet2014").glob("*__*.json")])
    fp = RESULTS / "gstar_frames_CDnet2014.parquet"
    if "--events-only" in sys.argv and fp.exists():
        F = pd.read_parquet(fp)
    else:
        with Pool(8) as pool:
            res = pool.map(video_frames, vids)
        F = pd.DataFrame([r for rr in res for r in rr])
        F.to_parquet(fp, index=False)

    # frame-level TP map (Mode G) from the raw boxes
    P_all = {}
    for (c, v), g in F.groupby(["category", "video"]):
        ok = ~g.det_missing.fillna(True).astype(bool).values
        tp = g.has_fg.values.astype(bool) & ok & (g.n_boxes.fillna(0).values > 0)
        P_all[(c, v)] = {int(t): ("TP" if x else "other") for t, x in zip(g.frame.values, tp)}

    ev = []
    for mg in (0, 1):
        for (c, v), g in F.groupby(["category", "video"], sort=True):
            ev += build_events(g.sort_values("frame").reset_index(drop=True), P_all[(c, v)], mg)
    E = pd.DataFrame(ev)
    E.to_csv(RESULTS / "miss_matrix_Gstar.csv", index=False)

    # regression against the P1a run (which read a derived per-frame state table for Mode G)
    reg = {}
    prev = RESULTS / "miss_matrix_Gstar_p1a.csv"
    if prev.exists():
        Pv = pd.read_csv(prev)
        a = E.sort_values(["merge_gap", "category", "video", "onset_frame"]).reset_index(drop=True)
        b = Pv.sort_values(["merge_gap", "category", "video", "onset_frame"]).reset_index(drop=True)
        reg = dict(same_events=bool(len(a) == len(b) and (a.onset_frame.values == b.onset_frame.values).all()
                                    and (a.duration.values == b.duration.values).all()))
        if reg["same_events"]:
            reg["max_abs_diff_pGstar"] = float(max(np.abs(a[f"pGs_K{K}"] - b[f"pGs_K{K}"]).max() for K in K_GRID))
            dG = np.max(np.abs(np.stack([a[f"pG_K{K}"] - b[f"pG_K{K}"] for K in K_GRID])), axis=0)
            reg["n_events_pG_changed"] = int((dG > 1e-12).sum())
            reg["max_abs_diff_pG"] = float(dG.max())
            reg["why_pG_can_change"] = ("first temporalROI frame has no JSON record; the derived table had an alert value "
                                        "for it, the raw JSON does not -> counted as no alert")

    rng = np.random.default_rng(SEED)
    summ = {}
    for mg in (0, 1):
        D = E[E.merge_gap == mg]
        fgm = F[F.has_fg]
        s = dict(n_events=len(D), n_videos=int(D.video.nunique()))
        for lab, col in [("G", "pG"), ("Gstar", "pGs"), ("Gstar_cover50_only", "pGsC"),
                         ("Gstar_iou30_only", "pGsI"), ("Gstar_any_object", "pGsA")]:
            s[lab] = pooled(D, col, rng)
        s["events_detected_G_but_no_Gstar_hit"] = int(((D.n_tp_G > 0) & (D.n_hit == 0)).sum())
        s["events_detected_G_K1"] = int((D.n_tp_G > 0).sum())
        s["events_detected_Gstar_K1"] = int((D.n_hit > 0).sum())
        s["events_with_det_missing_frame"] = int((D.n_det_missing > 0).sum())
        s["events_all_frames_det_missing"] = int((D.n_det_missing == D.n_fg).sum())
        # sensitivity (NOT the primary definition): drop events whose largest GT object never reaches
        # A_min pixels — annotation specks that Mode G "detects" only through background FPs
        s["min_area_sensitivity"] = {}
        for a_min in (16, 64, 256):
            Df = D[D.L_area_max >= a_min]
            s["min_area_sensitivity"][f"A{a_min}"] = dict(
                n_events=len(Df), n_videos=int(Df.video.nunique()),
                G=pooled(Df, "pG", rng), Gstar=pooled(Df, "pGs", rng))
        s["missed_events_Gstar_K1_by_Lmax"] = {
            "lt16": int(((D.n_hit == 0) & (D.L_area_max < 16)).sum()),
            "16_255": int(((D.n_hit == 0) & (D.L_area_max >= 16) & (D.L_area_max < 256)).sum()),
            "ge256": int(((D.n_hit == 0) & (D.L_area_max >= 256)).sum())}
        summ[f"gap{mg}"] = s
    fg = F[F.has_fg & ~F.det_missing.fillna(True).astype(bool)]
    P_tp = np.array([P_all[(c, v)].get(int(t)) == "TP" for c, v, t in fg[["category", "video", "frame"]].values])
    frame_level = dict(n_fg_frames=int(len(fg)), n_TP_G=int(P_tp.sum()),
                       n_hit_Gstar=int(fg.hit.astype(bool).sum()),
                       n_TP_G_not_Gstar_hit=int((P_tp & ~fg.hit.astype(bool).values).sum()),
                       frac_TP_G_that_are_background_FP=float((P_tp & ~fg.hit.astype(bool).values).sum() / P_tp.sum()),
                       n_hit_cover_only=int((fg.hit_cover.astype(bool) & ~fg.hit_iou.astype(bool)).sum()),
                       n_hit_iou_only=int((fg.hit_iou.astype(bool) & ~fg.hit_cover.astype(bool)).sum()),
                       n_hit_both=int((fg.hit_cover.astype(bool) & fg.hit_iou.astype(bool)).sum()),
                       n_hit_any_object=int(fg.hit_any.astype(bool).sum()),
                       n_fg_frames_det_missing=int((F.has_fg & F.det_missing.fillna(True).astype(bool)).sum()))

    # per-video G vs G* (gap1)
    D1 = E[E.merge_gap == 1]
    per_video = []
    for (c, v), g in D1.groupby(["category", "video"]):
        r = dict(category=c, video=v, n_events=len(g),
                 events_bg_fp_only=int(((g.n_tp_G > 0) & (g.n_hit == 0)).sum()),
                 frames_TP_G=int(g.n_tp_G.sum()), frames_TP_G_not_hit=int(g.n_tp_G_not_hit.sum()))
        for K in (1, 2, 4, 8):
            r[f"G_K{K}"] = float(g[f"pG_K{K}"].mean())
            r[f"Gstar_K{K}"] = float(g[f"pGs_K{K}"].mean())
        per_video.append(r)
    per_video = sorted(per_video, key=lambda r: -(r["Gstar_K1"] - r["G_K1"]))

    out = dict(
        meta=dict(hit_rule="box covers >=50% of largest GT component OR IoU(box, bbox(largest)) >= 0.3",
                  gt="CDnet groundtruth 255 = fg; 50/85/170 and 0 = not fg; frames in temporalROI only",
                  event="max run of GT-fg frames, gaps <= 1 merged (gap1 = primary); gap0 = sensitivity",
                  frame_map="JSON frame f <-> CDnet file number f+1 (0-based sort index)",
                  sources="raw: detection JSON (DET_ROOT) + CDnet groundtruth PNG; nothing else",
                  det_missing_rule="frames with no JSON record count as NOT hit (no inference); counted below",
                  bootstrap=f"B={B_BOOT}, seed={SEED}, resample videos", detector_rerun=False),
        self_checks=dict(
            frames_in_roi=int(len(F)),
            frames_det_missing=int(F.det_missing.fillna(True).astype(bool).sum()),
            gt_png_missing=int(F.gt_missing.sum()),
            regression_vs_P1a=reg),
        frame_level=frame_level, summary=summ, per_video_gap1=per_video)
    dump(out, "m1_gstar_summary.json")
    for mg in ("gap0", "gap1"):
        s = summ[mg]
        print(mg, s["n_events"], {k: (round(s["G"][k]["p_hat"], 4), round(s["Gstar"][k]["p_hat"], 4))
                                  for k in ("K1", "K2", "K4", "K8")})
    print(frame_level)
    print(out["self_checks"])


if __name__ == "__main__":
    main()
