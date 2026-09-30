"""corpus17 — the #17 corpus rebuilt from RAW data only (P1b', decision 5: #17 is a standalone paper).

Sources (read-only, raw):
  * detection dumps of the fixed detector (DET_ROOT/<dataset>/*.json): boxes + confidences per frame;
  * dataset frames / pixel ground truth: CDnet2014 (groundtruth/gt%06d.png, temporalROI), LASIESTA (RGB
    instance masks), BMC-synth (gt/%06d.png).
Frame order = TRUE temporal order (CDnet/BMC: JSON index + 1; LASIESTA: number parsed from the file name).

Definitions
  Mode T  : detector-positive event = maximal run of frames on which the detector returned >= 1 box.
  Mode G* : GT event (maximal run of frames with a GT object, gaps <= 1 merged); frame HIT = a box covers
            >= 50% of the largest GT object or IoU >= 0.3 with its bbox (a1_gstar.py / a4_gstar_ext.py).
  A256    : primary event definition: largest GT object reaches >= 256 px at the event's largest frame.
Calibration scenes (twin): a GT-empty run >= 25 frames (background plate) and >= 1 Mode-G* event.
Tiers: PTZ (excluded: the twin assumes a static background), nightVideos (night tier), turbulence (tier),
       cameraJitter (jitter tier), everything else = main domain (static camera, daytime / non-night).
       Post-F1 correction (30-09-2026, v1.1): cameraJitter was in the main domain in v1.0. It is split off as its own
       tier, like PTZ, using only CDnet's own category label (a shaking camera violates the static-background
       assumption of the twin in the same way as PTZ). Decided after F1 had shown the only worst-phase false
       acceptance (traffic) and one of the two LOSO violations (traffic) in this category -> a post-hoc change,
       disclosed as such. Calibration scenes of the category: boulevard, traffic (badminton, sidewalk have no
       GT-empty background run >= 25 frames and were never calibration scenes).
Outputs: results/corpus17_events_T.csv, results/p1_background17.json, results/corpus17_manifest.json
"""
from __future__ import annotations

import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from common import DET_ROOT, REAL, RESULTS, dump, n0, runs
from gstar_common import BMC_ROOT, CDNET_ROOT, LAS_ROOT, lasiesta_order, load_dets

MIN_BG_RUN = 25
DATASETS = ("BMC", "CDnet2014", "LASIESTA")


def true_numbers(ds, name, idx):
    if ds == "LASIESTA":
        nums = lasiesta_order(name.split("__", 1)[1][:-5])
        return [nums[i] for i in idx]
    return [i + 1 for i in idx]


def mode_T():
    rows = []
    for ds in DATASETS:
        for f in sorted((DET_ROOT / ds).glob("*.json")):
            d = load_dets(ds, f.name)
            idx = sorted(d)
            num = np.array(true_numbers(ds, f.name, idx))
            pos = np.array([len(d[i]) > 0 for i in idx])
            o = np.argsort(num)
            num, pos = num[o], pos[o]
            v = f.stem.split("__", 1)[1]
            for eid, (i0, i1) in enumerate(runs(num, pos)):
                rows.append(dict(dataset=ds, video=v, event_id=eid, onset_frame=int(num[i0]),
                                 duration=int(num[i1] - num[i0] + 1)))
    return pd.DataFrame(rows)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha_tree(files):
    with Pool(8) as pool:
        hs = pool.map(sha_file, [str(f) for f in files])
    return hashlib.sha256("".join(f"{Path(f).name}:{h}\n" for f, h in sorted(zip(map(str, files), hs))).encode()).hexdigest()


def calibration():
    F = pd.read_parquet(RESULTS / "gstar_frames_CDnet2014.parquet", columns=["category", "video", "frame", "has_fg"])
    E = pd.read_csv(RESULTS / "miss_matrix_Gstar.csv")
    E1 = E[E.merge_gap == 1]
    cd = []
    for (c, v), g in F.groupby(["category", "video"]):
        L = max([j - i + 1 for i, j in runs(g.frame.values, ~g.has_fg.values.astype(bool))], default=0)
        ne = int((E1.video == v).sum())
        cd.append(dict(dataset="CDnet2014", category=c, video=v, longest_gt_empty_run=L, n_events_gap1=ne,
                       n_events_a256=int(((E1.video == v) & (E1.L_area_max >= 256)).sum())))
    cd = pd.DataFrame(cd)
    FL = pd.read_parquet(RESULTS / "gstar_frames_LASIESTA.parquet", columns=["video", "frame", "has_fg"])
    X = pd.read_csv(RESULTS / "miss_matrix_Gstar_ext.csv")
    XL = X[(X.dataset == "LASIESTA") & (X.level == "frame")]
    la = []
    for v, g in FL.groupby("video"):
        g = g.sort_values("frame")
        L = max([j - i + 1 for i, j in runs(g.frame.values, ~g.has_fg.values.astype(bool))], default=0)
        d = load_dets("LASIESTA", next((DET_ROOT / "LASIESTA").glob(f"*__{v}.json")).name)
        nums = lasiesta_order(v)
        num2idx = {n: i for i, n in enumerate(nums)}
        silent = np.array([len(d.get(num2idx[n], [])) == 0 for n in sorted(nums)])
        Ls = max([j - i + 1 for i, j in runs(np.array(sorted(nums)), silent)], default=0)
        la.append(dict(dataset="LASIESTA", category="LASIESTA", video=v, longest_gt_empty_run=L,
                       longest_detector_silent_run_true_order=Ls, n_events_gap1=int((XL.video == v).sum()),
                       n_events_a256=int(((XL.video == v) & (XL.L_area_max >= 256)).sum())))
    la = pd.DataFrame(la)
    return cd, la


def tier(cat):
    if cat == "PTZ":
        return "excluded_PTZ"
    if cat == "nightVideos":
        return "night"
    if cat == "turbulence":
        return "turbulence"
    if cat == "cameraJitter":            # post-F1 correction (v1.1), see module docstring
        return "jitter"
    return "main"


def main():
    T = mode_T()
    T.to_csv(RESULTS / "corpus17_events_T.csv", index=False)
    chk = {}
    b0 = RESULTS / "events_durations_corrected.csv"          # B0 (built from the legacy table) — consistency only
    if b0.exists():
        B = pd.read_csv(b0)
        B["video"] = B.video.str.split("__").str[-1]
        a = T.sort_values(["dataset", "video", "event_id"]).reset_index(drop=True)
        b = B.sort_values(["dataset", "video", "event_id"]).reset_index(drop=True)
        chk = dict(same_count=len(a) == len(b), identical_durations=bool(len(a) == len(b) and (a.duration.values == b.duration.values).all()))
    real = T[T.dataset.isin(REAL)].groupby(["dataset", "video"]).size()
    nvid_real = sum(len(list((DET_ROOT / ds).glob("*.json"))) for ds in REAL)
    counts = {f">={k}": int((real >= k).sum()) for k in (n0(), 2 * n0(), 3 * n0())}

    cd, la = calibration()
    cd_cal = cd[(cd.longest_gt_empty_run >= MIN_BG_RUN) & (cd.n_events_gap1 > 0)].copy()
    la_cal = la[(la.longest_gt_empty_run >= MIN_BG_RUN) & (la.n_events_gap1 > 0)].copy()
    la_old = la[la.longest_detector_silent_run_true_order >= MIN_BG_RUN]
    cd_cal["tier"] = cd_cal.category.map(tier)
    la_cal["tier"] = "main"
    old = json.load(open(RESULTS / "p1_background.json", encoding="utf-8")) if (RESULTS / "p1_background.json").exists() else None
    cmp = {}
    if old:
        o_cd = {r["video"] for r in old["cdnet_calibration_candidates_gt_empty"]["list"]}
        o_la = {r["video"] for r in old["per_video"] if r["dataset"] == "LASIESTA" and r["longest_bg_run"] >= 25}
        cmp = dict(cdnet_same=o_cd == set(cd_cal.video), cdnet_added=sorted(set(cd_cal.video) - o_cd), cdnet_removed=sorted(o_cd - set(cd_cal.video)),
                   lasiesta_P1b_list=sorted(o_la), lasiesta_gt_empty_list=sorted(la_cal.video),
                   lasiesta_same=o_la == set(la_cal.video),
                   note="P1b LASIESTA list came from a detector-silent run computed in the text-sorted (scrambled) order; "
                        "#17 now uses the GT-empty criterion (same as CDnet) in true order")
    scenes = pd.concat([cd_cal, la_cal], ignore_index=True)
    bg = dict(meta=dict(min_bg_run=MIN_BG_RUN, criterion="GT-empty run >= 25 frames (true order) and >= 1 Mode-G* event (gap1)",
                        tiers="PTZ excluded (static-background twin); nightVideos = night tier; turbulence = turbulence tier; "
                             "cameraJitter = jitter tier (post-F1 correction v1.1, CDnet category label); rest = main"),
              scenes=scenes.to_dict("records"),
              by_tier={t: sorted(g.video) for t, g in scenes.groupby("tier")},
              n_by_tier={t: int(len(g)) for t, g in scenes.groupby("tier")},
              comparison_with_P1b_lists=cmp, cdnet_all=cd.to_dict("records"), lasiesta_all=la.to_dict("records"))
    dump(bg, "p1_background17.json")

    # manifest (the paper's "Data" section)
    det = {}
    for ds in DATASETS:
        fs = sorted((DET_ROOT / ds).glob("*.json"))
        det[ds] = dict(path=str(DET_ROOT / ds), n_files=len(fs), sha256_tree=sha_tree(fs))
    gt = {}
    cdf = sorted(CDNET_ROOT.glob("*/*/groundtruth/gt*.png")) + sorted(CDNET_ROOT.glob("*/*/temporalROI.txt"))
    gt["CDnet2014"] = dict(path=str(CDNET_ROOT), n_files=len(cdf), sha256_tree=sha_tree(cdf))
    lf = sorted(LAS_ROOT.glob("*-GT/*.png"))
    gt["LASIESTA"] = dict(path=str(LAS_ROOT), n_files=len(lf), sha256_tree=sha_tree(lf))
    bf = sorted(BMC_ROOT.glob("synth*/*/gt/*.png"))
    gt["BMC_synth"] = dict(path=str(BMC_ROOT), n_files=len(bf), sha256_tree=sha_tree(bf))
    Gs = pd.read_csv(RESULTS / "miss_matrix_Gstar.csv")
    Gs = Gs[Gs.merge_gap == 1]
    Xs = pd.read_csv(RESULTS / "miss_matrix_Gstar_ext.csv")
    Xs = Xs[Xs.level == "frame"]
    man = dict(
        description="#17 corpus — built only from the raw sources below; every number of the paper is read from results/ of #17",
        detector=json.load(open(RESULTS / "a0_detector_guard.json"))["verified"],
        raw_detection_dumps=det, raw_ground_truth=gt,
        frame_order="true temporal order (CDnet/BMC: index+1; LASIESTA: number in file name — text order is NOT temporal)",
        definitions=dict(mode_T="maximal run of frames with >= 1 detector box",
                         mode_Gstar="GT event, gaps <= 1 merged; HIT = box covers >= 50% of largest GT object or IoU >= 0.3",
                         A256="largest GT object >= 256 px at the event's largest frame (primary)",
                         orig="any GT pixel (sensitivity)"),
        counts=dict(mode_T={ds: int((T.dataset == ds).sum()) for ds in DATASETS}, mode_T_total=int(len(T)),
                    mode_T_real_videos=int(nvid_real), mode_T_real_videos_reaching=counts,
                    mode_Gstar_CDnet=dict(orig=int(len(Gs)), a256=int((Gs.L_area_max >= 256).sum()), videos=int(Gs.video.nunique())),
                    mode_Gstar_LASIESTA=dict(orig=int((Xs.dataset == "LASIESTA").sum()),
                                             a256=int(((Xs.dataset == "LASIESTA") & (Xs.L_area_max >= 256)).sum())),
                    mode_Gstar_BMC_synth=dict(orig=int((Xs.dataset == "BMC").sum()),
                                              a256=int(((Xs.dataset == "BMC") & (Xs.L_area_max >= 256)).sum())),
                    calibration_scenes=bg["n_by_tier"]),
        consistency_with_B0=chk,
        files=dict(events_T="results/corpus17_events_T.csv", events_Gstar_CDnet="results/miss_matrix_Gstar.csv",
                   events_Gstar_ext="results/miss_matrix_Gstar_ext.csv", scenes="results/p1_background17.json"))
    dump(man, "corpus17_manifest.json")
    print(json.dumps(dict(counts=man["counts"], consistency=chk, lists=cmp, tiers=bg["n_by_tier"]), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
