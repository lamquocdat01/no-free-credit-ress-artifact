"""A4 — Mode G* on LASIESTA (real, 48 seq) and BMC-synth (SiVIC, 20 videos), same rule as A1.

LASIESTA GT: RGB PNG, one colour per object, black = background, grey 128 = ignore. Frame-level
events (any object present, gaps <= 1 merged) are primary (comparable to CDnet); instance-level
events (one colour track) are reported too. Frame map: JSON index -> lexicographically sorted file
list -> temporal number parsed from the file name (the dumps were sorted as text; see gstar_common).
BMC-synth GT: NNN/gt/%06d.png, grayscale (lossy: values near 0 / 255) -> fg = value > 127.
Detector NOT re-run. (The P1a audit of another project's Mode-T table was moved to ../legacy/, see B0.)
Outputs: results/m1_gstar_ext.json, results/miss_matrix_Gstar_ext.csv,
         results/gstar_frames_{LASIESTA,BMC}.parquet
"""
from __future__ import annotations

import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from common import B_BOOT, K_GRID, RESULTS, SEED, cp_ci, dump, runs
from gstar_common import (BMC_ROOT, DET_ROOT, LAS_ROOT, class_group, eval_frame, fg_runs, label_binary,
                          label_colours, lasiesta_order, load_dets, pmiss_residue)

D_MIN = [1, 2, 4, 8, 16, 32]


def las_seqs():
    return sorted(p.name for p in LAS_ROOT.iterdir() if p.is_dir() and not p.name.endswith("-GT")
                  and (LAS_ROOT / (p.name + "-GT")).is_dir())


def las_json(seq):
    return next((DET_ROOT / "LASIESTA").glob(f"*__{seq}.json")).name


def bmc_vids():
    return [(s, v.name) for s in ("synth1", "synth2") for v in sorted((BMC_ROOT / s).iterdir())
            if v.is_dir() and (v / "gt").is_dir()]


# ───────────────────────────── per-frame ─────────────────────────────
def las_frames(seq):
    nums = lasiesta_order(seq)
    dets = load_dets("LASIESTA", las_json(seq))
    num2idx = {n: i for i, n in enumerate(nums)}
    rows = []
    for n in sorted(nums):
        rgb = cv2.imread(str(LAS_ROOT / f"{seq}-GT" / f"{seq}-GT_{n}.png"))
        if rgb is None:
            rows.append(dict(dataset="LASIESTA", video=seq, frame=n, gt_missing=True, has_fg=False))
            continue
        rgb = rgb[:, :, ::-1]
        lab, nl, cols = label_colours(rgb)
        has = nl > 1
        boxes = dets.get(num2idx[n])
        r = dict(dataset="LASIESTA", video=seq, frame=n, gt_missing=False, has_fg=bool(has),
                 det_missing=boxes is None, n_boxes=len(boxes) if boxes else 0,
                 H=rgb.shape[0], W=rgb.shape[1])
        if has:
            e = eval_frame(lab, nl, boxes or [])
            e["L_bbox"] = None if e["L_bbox"] is None else ",".join(map(str, e["L_bbox"]))
            r.update(e)
            # instance-level: per colour, hit against that instance
            inst = {}
            for i, c in enumerate(cols, 1):
                sub = np.where(lab == i, 1, 0).astype(np.int32)
                ei = eval_frame(sub, 2, boxes or [])
                inst["%02x%02x%02x" % c] = (int(ei["L_area"]), bool(ei["hit"]), int(ei["cls"]))
            r["instances"] = repr(inst)
        rows.append(r)
    return rows


def bmc_frames(sv, shift=1):
    split, v = sv
    dets = load_dets("BMC", f"{split}__{v}.json")
    gdir = BMC_ROOT / split / v / "gt"
    nums = sorted(int(p.stem) for p in gdir.glob("*.png") if p.stem.isdigit())  # skips e.g. "000937 (1).png"
    rows = []
    for n in nums:
        g = cv2.imread(str(gdir / f"{n:06d}.png"), cv2.IMREAD_GRAYSCALE)
        fg = g > 127
        has = bool(fg.any())
        boxes = dets.get(n - shift)
        r = dict(dataset="BMC", video=f"{split}/{v}", frame=n, gt_missing=False, has_fg=has,
                 det_missing=boxes is None, n_boxes=len(boxes) if boxes else 0, H=g.shape[0], W=g.shape[1])
        if has:
            lab, nl = label_binary(fg)
            e = eval_frame(lab, nl, boxes or [])
            e["L_bbox"] = None if e["L_bbox"] is None else ",".join(map(str, e["L_bbox"]))
            r.update(e)
        rows.append(r)
    return rows


def _las(seq):
    return las_frames(seq)


def _bmc(sv):
    return bmc_frames(sv)


# ───────────────────────────── events ─────────────────────────────
def events_frame_level(F):
    ev = []
    for (ds, v), g in F.groupby(["dataset", "video"], sort=True):
        g = g.sort_values("frame").reset_index(drop=True)
        fr, has = g.frame.values, g.has_fg.values.astype(bool)
        for eid, (i0, i1) in enumerate(fg_runs(fr, has, 1)):
            seg = g.iloc[i0:i1 + 1]
            segf = seg.frame.values
            fgm = seg.has_fg.values.astype(bool)
            hit = seg.hit.fillna(False).values.astype(bool) & fgm
            hit_a = seg.hit_any.fillna(False).values.astype(bool) & fgm
            detm = seg.det_missing.fillna(False).values.astype(bool)
            cls = seg.cls.values[hit]
            cv = int(pd.Series(cls).mode().iloc[0]) if len(cls) else -1
            r = dict(dataset=ds, video=v, level="frame", event_id=eid, onset_frame=int(segf[0]),
                     duration=int(segf[-1] - segf[0] + 1), n_fg=int(fgm.sum()), n_hit=int(hit.sum()),
                     n_hit_any=int(hit_a.sum()), n_det_missing=int(detm.sum()),
                     L_area_max=float(np.nanmax(seg.L_area.values[fgm].astype(float))),
                     cls=cv, cls_group=class_group(cv))
            for K in K_GRID:
                r[f"pGs_K{K}"] = pmiss_residue(segf[hit], K)
                r[f"pGsA_K{K}"] = pmiss_residue(segf[hit_a], K)
            ev.append(r)
    return ev


def events_instance_level(F):
    ev = []
    L = F[(F.dataset == "LASIESTA") & F.has_fg]
    for v, g in L.groupby("video", sort=True):
        tracks = {}
        for t, s in zip(g.frame.values, g.instances.values):
            for c, (a, h, cl) in eval(s).items():
                tracks.setdefault(c, []).append((t, a, h, cl))
        for c, rows in sorted(tracks.items()):
            fr = np.array([r[0] for r in rows])
            hit = np.array([r[2] for r in rows])
            cls = [r[3] for r in rows if r[2]]
            for eid, (i0, i1) in enumerate(fg_runs(fr, np.ones(len(fr), bool), 1)):
                segf, segh = fr[i0:i1 + 1], hit[i0:i1 + 1]
                cc = [r[3] for r in rows[i0:i1 + 1] if r[2]]
                cv = int(pd.Series(cc).mode().iloc[0]) if cc else -1
                r = dict(dataset="LASIESTA", video=v, level="instance", instance=c, event_id=eid,
                         onset_frame=int(segf[0]), duration=int(segf[-1] - segf[0] + 1), n_fg=len(segf),
                         n_hit=int(segh.sum()), L_area_max=float(max(x[1] for x in rows[i0:i1 + 1])),
                         cls=cv, cls_group=class_group(cv))
                for K in K_GRID:
                    r[f"pGs_K{K}"] = pmiss_residue(segf[segh], K)
                ev.append(r)
    return ev


def pooled(df, col, rng, ks=K_GRID):
    vids = df.video.unique()
    idx = {v: np.where(df.video.values == v)[0] for v in vids}
    out = {}
    for K in ks:
        x = df[f"{col}_K{K}"].values
        bs = np.array([x[np.concatenate([idx[v] for v in rng.choice(vids, len(vids), replace=True)])].mean()
                       for _ in range(B_BOOT)]) if len(x) else np.array([np.nan])
        out[f"K{K}"] = dict(p_hat=float(x.mean()) if len(x) else None, cp95=cp_ci(float(x.sum()), len(x)),
                            boot95_video=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))])
    return out


def by_dmin(df, col, rng):
    out = {}
    for d in D_MIN:
        sub = df[df.duration >= d]
        out[f"d{d}"] = dict(n_events=len(sub), n_videos=int(sub.video.nunique()),
                            cells=pooled(sub, col, rng, [K for K in K_GRID if K <= d] + [8]) if len(sub) else {})
    return out


def detector_miss_compare(E, rng):
    """BMC-synth vs real (unpaired): detector-only miss in cells K <= d_min (temporal miss = 0 there).
    Difference BMC - real with a video-level bootstrap (each side resampled by video)."""
    out = {}
    real = E[E.dataset.isin(["CDnet2014", "LASIESTA"])]
    sim = E[E.dataset == "BMC"]
    for d in D_MIN:
        rs, ss = real[real.duration >= d], sim[sim.duration >= d]
        cell = {}
        for K in [K for K in K_GRID if K <= d]:
            col = f"pGs_K{K}"
            rv = {v: g[col].values for v, g in rs.groupby("video")}
            sv = {v: g[col].values for v, g in ss.groupby("video")}
            if not rv or not sv:
                continue
            diff = []
            rk, sk = list(rv), list(sv)
            for _ in range(B_BOOT):
                a = np.concatenate([sv[k] for k in rng.choice(sk, len(sk))]).mean()
                b = np.concatenate([rv[k] for k in rng.choice(rk, len(rk))]).mean()
                diff.append(a - b)
            cell[f"K{K}"] = dict(bmc=float(ss[col].mean()), real=float(rs[col].mean()),
                                 cdnet=float(rs[rs.dataset == "CDnet2014"][col].mean()) if (rs.dataset == "CDnet2014").any() else None,
                                 lasiesta=float(rs[rs.dataset == "LASIESTA"][col].mean()) if (rs.dataset == "LASIESTA").any() else None,
                                 n_bmc=len(ss), n_real=len(rs),
                                 diff_bmc_minus_real=float(ss[col].mean() - rs[col].mean()),
                                 diff_boot95=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))])
        out[f"d{d}"] = cell
    return out


def alignment_check():
    """Hit rate of G* on fg frames for candidate frame maps (the right map maximises it)."""
    res = {}
    for seq in ["I_BS_01", "I_SI_01", "O_CL_01"]:
        nums = lasiesta_order(seq)
        dets = load_dets("LASIESTA", las_json(seq))
        rr = {}
        for label, idx_of in [("lexicographic_sort_index (used)", {n: i for i, n in enumerate(nums)}),
                              ("naive numeric index n-1", {n: n - 1 for n in nums})]:
            h = t = 0
            for n in sorted(nums)[::3]:
                rgb = cv2.imread(str(LAS_ROOT / f"{seq}-GT" / f"{seq}-GT_{n}.png"))[:, :, ::-1]
                lab, nl, _ = label_colours(rgb)
                if nl > 1:
                    t += 1
                    h += eval_frame(lab, nl, dets.get(idx_of[n]) or [])["hit"]
            rr[label] = h / max(t, 1)
        res[f"LASIESTA/{seq}"] = rr
    for sv in [("synth1", "111"), ("synth1", "122"), ("synth2", "421"), ("synth2", "512")]:
        rr = {}
        for shift in (-1, 0, 1, 2, 3):
            rows = [r for r in bmc_frames(sv, shift)[::2] if r["has_fg"]]
            rr[f"json_frame = gt_number - {shift}"] = dict(
                hit_rate=float(np.mean([r["hit"] for r in rows])),
                mean_best_iou=float(np.mean([r["best_iou"] for r in rows])))
        res[f"BMC/{sv[0]}/{sv[1]}"] = rr
    res["verdict"] = ("LASIESTA: text-sort index map is right (hit rate ~0.92-0.95 vs <=0.21). BMC: hit rate is "
                      "flat across shifts (lenient rule, slow objects) but mean best IoU peaks at shift 1 "
                      "(json f <-> gt f+1), the map used.")
    return res


def main():
    pd.set_option("future.no_silent_downcasting", True)
    fl, fb = RESULTS / "gstar_frames_LASIESTA.parquet", RESULTS / "gstar_frames_BMC.parquet"
    if "--events-only" in sys.argv and fl.exists() and fb.exists():
        FL, FB = pd.read_parquet(fl), pd.read_parquet(fb)
    else:
        with Pool(8) as pool:
            FL = pd.DataFrame([r for rr in pool.map(_las, las_seqs()) for r in rr])
            FB = pd.DataFrame([r for rr in pool.map(_bmc, bmc_vids()) for r in rr])
        FL.to_parquet(fl, index=False)
        FB.to_parquet(fb, index=False)
    F = pd.concat([FL, FB], ignore_index=True)
    ev = events_frame_level(F) + events_instance_level(F)
    E = pd.DataFrame(ev)
    E.to_csv(RESULTS / "miss_matrix_Gstar_ext.csv", index=False)

    # CDnet G* (A1, gap1) for the real-vs-sim comparison
    C = pd.read_csv(RESULTS / "miss_matrix_Gstar.csv")
    C = C[C.merge_gap == 1].assign(level="frame")
    ALL = pd.concat([E[E.level == "frame"], C[E.columns.intersection(C.columns)]], ignore_index=True)

    rng = np.random.default_rng(SEED)
    out = dict(meta=dict(rule="same as A1 (cover>=0.5 of largest object OR IoU>=0.3 with its bbox)",
                         lasiesta_gt="RGB colour = instance; black bg; grey 128 ignored",
                         lasiesta_frame_map="JSON idx -> text-sorted file list -> number in file name",
                         bmc_gt="grayscale > 127 = fg; JSON frame f <-> gt number f+1; non-numeric duplicate "
                                "synth2/422/gt/000937 (1).png ignored (frames/ has no duplicate)",
                         events="frame-level: run of frames with any GT object, gaps <= 1 merged; "
                                "instance-level (LASIESTA): run of one colour track, gaps <= 1 merged",
                         bootstrap=f"B={B_BOOT} seed={SEED} by video", detector_rerun=False),
               alignment_check=alignment_check(), datasets={})
    for ds, sub in [("LASIESTA", E[(E.dataset == "LASIESTA") & (E.level == "frame")]),
                    ("LASIESTA_instance", E[(E.dataset == "LASIESTA") & (E.level == "instance")]),
                    ("BMC_synth", E[E.dataset == "BMC"])]:
        Fd = F[F.dataset == ("BMC" if ds == "BMC_synth" else "LASIESTA")]
        rec = dict(n_events=len(sub), n_videos=int(sub.video.nunique()),
                   n_videos_total=int(Fd.video.nunique()),
                   n_fg_frames=int(Fd.has_fg.sum()), n_fg_frames_det_missing=int((Fd.has_fg & Fd.det_missing.fillna(True).astype(bool)).sum()),
                   events_per_video_median=float(sub.groupby("video").size().median()) if len(sub) else 0,
                   p_hat=pooled(sub, "pGs", rng), by_dmin=by_dmin(sub, "pGs", rng))
        if "pGsA_K1" in sub and sub["pGsA_K1"].notna().any():
            rec["p_hat_any_object"] = {k: v["p_hat"] for k, v in pooled(sub, "pGsA", rng).items()}
        out["datasets"][ds] = rec
    out["bmc_vs_real_detector_miss"] = detector_miss_compare(ALL, rng)
    dump(out, "m1_gstar_ext.json")
    for ds, r in out["datasets"].items():
        print(ds, r["n_events"], r["n_videos"], {k: round(v["p_hat"], 3) for k, v in r["p_hat"].items() if k in ("K1", "K2", "K4", "K8")})
    print(out["alignment_check"])


if __name__ == "__main__":
    main()
