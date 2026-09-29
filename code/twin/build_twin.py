"""B4 — composite (semi-synthetic, S1) twin, paired 1-1 with every real GT event, + detector + Mode G*.

Per scene
  plate   = per-pixel median of up to 60 GT-EMPTY frames (longest empty run first; if the longest empty
            run < 25 frames, empty frames spread over the video; if none: temporal median of all frames,
            flagged 'plate_temporal_median').
Per event (Mode-G* gap1 events; both definitions, flag def_a256) and sprite variant v in {0,1,2} (seed 42+v)
  class   = event class from A6 (person / vehicle; 'other' -> vehicle); 'unmatched' -> the scene's dominant
            class (flag cls_imputed).
  sprite  = vehicle: VKITTI2 sprite whose aspect is within +-10% of the event's median GT-bbox aspect (closest
            otherwise, flagged); person: one MPFB character, one azimuth, walking cycle p0,p1,p2,p1 every
            CYCLE frames, mirrored to the motion direction.
Per frame t of the event with a GT object
  base    = REAL frame t with the real object (GT 255, shadow 50; LASIESTA any non-black pixel), dilated
            DIL px, replaced by the plate  -> scene conditions at t are kept; only the object is swapped.
  placement: sprite height = GT bbox height (largest GT object at t); width = clip(h*aspect_sprite,
            0.9 W_gt, 1.1 W_gt)  (area and aspect within ~+-10%); bottom-centre on the GT bbox.
  appearance: scene-illumination gain g = clip(L_frame / L_REF, G_MIN, G_MAX) from the whole frame's mean
            luminance (NOT the local ring: pilot v1 showed ring matching bleaches sprites into the wall);
            a BLEND fraction of the frame's mean colour is mixed in; grey scenes -> grey sprite.
  shadow  = soft ellipse under the bbox bottom (darkening SHADOW); feather = alpha eroded 1 px + Gaussian
            sigma 1 (~2 px).
  twin GT = sprite alpha > 0.5 at the placed position; Mode G* on the twin uses the A1 rule against it.
Detector: yolo26s-seg through detector_guard (conf 0.25, iou 0.7, imgsz 640), BGR frames like the dumps.
Exact early stop: once the twin hits of an event cover all 48 residues mod 48 (every K of the grid divides
48), R_twin(K) is full for every K, so p_e, q_e and D_e can no longer change -> remaining frames skipped
(rows = processed frames; the real side is always evaluated on every frame, see m2_discordance.py).
Outputs (per scene, resumable): results/twin/frames_{ds}__{video}.parquet; previews in results/twin_preview/.
All parameters -> results/twin_manifest.json.
Usage: python build_twin.py pilot|full [--workers N] [--preview N]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import RESULTS, runs  # noqa: E402
import twin_fx  # noqa: E402
from gstar_common import (CDNET_ROOT, LAS_ROOT, eval_frame, label_binary, label_colours,  # noqa: E402
                          lasiesta_order, load_dets)

ROOT = HERE.parent.parent
SPR = ROOT / "assets" / "sprites"
TW = RESULTS / "twin"
PREV = RESULTS / "twin_preview"

P = dict(DIL=3, PLATE_N=60, MIN_EMPTY_RUN=25, CYCLE=4, L_REF=110.0, G_MIN=0.35, G_MAX=1.2, BLEND=0.05,
         SHADOW=0.35, LCM=48, ASPECT_TOL=0.10, SIZE_TOL=0.10, VARIANT_SEEDS=[42, 43, 44],
         VEHICLE_SPRITES="vehicle", PERSON_SPRITES="person",
         SELECT="random",      # P1b: random sprite within +-10% aspect; P1b' C0: "aspect" (nearest aspect, per-frame pose)
         PROFILE="")           # P1b' C1/C2 effect letters, e.g. "a", "ab", "abc", "t1", "t2" (see twin_fx.py)

PILOT = [("CDnet2014", "baseline", "highway"), ("CDnet2014", "baseline", "pedestrians"),
         ("CDnet2014", "baseline", "office"), ("CDnet2014", "baseline", "PETS2006"),
         ("LASIESTA", None, "O_SM_04")]


# ───────────────────────────── io ─────────────────────────────
def frame_path(ds, cat, v, t):
    if ds == "CDnet2014":
        return CDNET_ROOT / cat / v / "input" / f"in{t:06d}.jpg"
    return LAS_ROOT / v / f"{v}-{t}.bmp"


def gt_remove_mask(ds, cat, v, t, shape):
    if ds == "CDnet2014":
        g = cv2.imread(str(CDNET_ROOT / cat / v / "groundtruth" / f"gt{t:06d}.png"), cv2.IMREAD_GRAYSCALE)
        m = (g == 255) | (g == 50)
    else:
        g = cv2.imread(str(LAS_ROOT / f"{v}-GT" / f"{v}-GT_{t}.png"))
        m = g.max(axis=2) > 20
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * P["DIL"] + 1,) * 2, np.uint8)) > 0


def make_plate(ds, cat, v, F):
    fr = F.frame.values
    empty = ~F.has_fg.values.astype(bool)
    rr = runs(fr, empty)
    longest = max(rr, key=lambda r: r[1] - r[0], default=None)
    note = ""
    if longest is not None and longest[1] - longest[0] + 1 >= P["MIN_EMPTY_RUN"]:
        idx = np.arange(longest[0], longest[1] + 1)
        note = "longest_empty_run"
    elif empty.sum() >= 5:
        idx = np.where(empty)[0]
        note = "empty_frames_spread"
    else:
        idx = np.arange(len(fr))
        note = "plate_temporal_median"
    idx = idx[np.linspace(0, len(idx) - 1, min(P["PLATE_N"], len(idx))).astype(int)]
    imgs = [cv2.imread(str(frame_path(ds, cat, v, int(fr[i])))) for i in idx]
    plate = np.median(np.stack(imgs), axis=0).astype(np.uint8)
    return plate, dict(plate_source=note, plate_n=len(idx),
                       longest_empty_run=0 if longest is None else int(longest[1] - longest[0] + 1)), imgs


# ───────────────────────────── sprites ─────────────────────────────
class Sprites:
    def __init__(self):
        vm = SPR / P["VEHICLE_SPRITES"] / "metadata.csv"
        self.veh = pd.read_csv(vm) if vm.exists() else None      # person-only scenes can run without it
        self.per = pd.read_csv(SPR / P["PERSON_SPRITES"] / "metadata.csv")
        self.cache = {}

    def img(self, kind, f):
        key = (kind, f)
        if key not in self.cache:
            self.cache[key] = cv2.imread(str(SPR / (P["VEHICLE_SPRITES"] if kind == "vehicle" else P["PERSON_SPRITES"]) / f),
                                         cv2.IMREAD_UNCHANGED)
        return self.cache[key]

    def choose(self, kind, aspect, rng, vi=0):
        if P["SELECT"] == "aspect":
            return self.choose_aspect(kind, aspect, rng, vi)
        if kind == "vehicle":
            d = np.abs(self.veh.aspect.values / aspect - 1)
            ok = np.where(d <= P["ASPECT_TOL"])[0]
            flag = len(ok) == 0
            if flag:
                ok = np.argsort(d)[:5]
            f = self.veh.file.values[rng.choice(ok)]
            return dict(kind="vehicle", files=[f], aspect_miss=bool(flag))
        ch = int(rng.integers(self.per.char.nunique()))
        az = int(rng.choice(sorted(self.per.azimuth.unique())))
        files = [f"char{ch:02d}_az{az:03d}_{p}.png" for p in ("p0", "p1", "p2", "p1")]
        return dict(kind="person", files=files, char=ch, azimuth=az, aspect_miss=False)


    def choose_aspect(self, kind, aspect, rng, vi):
        """C0: sprite whose aspect is nearest to the event's median GT box aspect.
        vehicle: variant vi takes the vi-th nearest sprite (distinct sprites across the 3 variants);
        person : random character (per variant seed), azimuth whose mean aspect over the walking poses is
                 nearest; the pose is then chosen PER FRAME as the one nearest to that frame's box aspect."""
        if kind == "vehicle":
            d = np.abs(np.log(self.veh.aspect.values / aspect))
            i = np.argsort(d, kind="stable")[vi]
            f0 = self.veh.file.values[i]
            return dict(kind="vehicle", files=[f0], aspect_miss=bool(d[i] > np.log(1 + P["ASPECT_TOL"])),
                        pick=lambda k, a_t, f=f0: f)
        ch = int(rng.integers(self.per.char.nunique()))
        sub = self.per[self.per.char == ch]
        m = sub.groupby("azimuth").aspect.mean()
        az = int(m.index[np.argmin(np.abs(np.log(m.values / aspect)))])
        poses = sub[sub.azimuth == az][["file", "aspect"]].values

        def pick(k, a_t, poses=poses):
            return poses[int(np.argmin(np.abs(np.log(poses[:, 1].astype(float) / max(a_t, 1e-3)))))][0]
        return dict(kind="person", files=list(poses[:, 0]), char=ch, azimuth=az, aspect_miss=False, pick=pick)


# ───────────────────────────── compositing ─────────────────────────────
def place(base, sprite_rgba, bbox, is_grey, mirror, ctx=None):
    """Return (twin_bgr, twin_mask, info). bbox = (x0, y0, x1, y1) of the largest GT object at t."""
    H, W = base.shape[:2]
    x0, y0, x1, y1 = bbox
    bw, bh = max(2, x1 - x0), max(2, y1 - y0)
    sp = sprite_rgba[:, :, ::-1] if False else sprite_rgba
    if mirror:
        sp = sp[:, ::-1]
    asp = sp.shape[1] / sp.shape[0]
    th = bh
    tw = int(round(np.clip(th * asp, (1 - P["SIZE_TOL"]) * bw, (1 + P["SIZE_TOL"]) * bw)))
    tw = max(2, tw)
    s = cv2.resize(sp, (tw, th), interpolation=cv2.INTER_AREA if th < sp.shape[0] else cv2.INTER_LINEAR)
    prof = ctx["prof"] if ctx else ""
    if prof:
        s = twin_fx.apply_layer_fx(s, prof, ctx)
    cx = (x0 + x1) / 2
    px0, py0 = int(round(cx - tw / 2)), y1 - th
    a = s[:, :, 3].astype(np.float32) / 255.0
    a = cv2.erode(a, np.ones((3, 3), np.uint8)) if min(tw, th) > 24 else a
    a = cv2.GaussianBlur(a, (0, 0), 0.8)
    rgb = s[:, :, :3].astype(np.float32)
    if is_grey:
        rgb = np.repeat(cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2GRAY)[:, :, None], 3, 2).astype(np.float32)
    # scene illumination: gain from the WHOLE frame's mean luminance (pilot v1 used the ring around the
    # object, which bleached sprites on bright walls -> ghost-like twins); slight tint towards the frame colour
    mu = base.reshape(-1, 3).astype(np.float32).mean(0)
    L_ring = float(0.114 * mu[0] + 0.587 * mu[1] + 0.299 * mu[2])
    g = float(np.clip(L_ring / P["L_REF"], P["G_MIN"], P["G_MAX"]))
    dark = 0.0
    if "a" in prof:                    # histogram matching to the local background replaces the frame gain
        rgb, dark = twin_fx.apply_colour_fx(rgb, a, base, bbox, prof, ctx)
    else:
        rgb = np.clip((1 - P["BLEND"]) * rgb * g + P["BLEND"] * mu[None, None, :], 0, 255)
        if prof:
            rgb, dark = twin_fx.apply_colour_fx(rgb, a, base, bbox, prof, ctx)
    out = base.astype(np.float32).copy()
    # shadow
    sh = np.zeros((H, W), np.float32)
    cv2.ellipse(sh, (int(cx), int(y1)), (max(1, int(0.45 * tw)), max(1, int(0.08 * th))), 0, 0, 360, 1.0, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), max(1.0, 0.03 * th))
    out *= (1 - P["SHADOW"] * sh)[:, :, None]
    # paste with clipping
    sx0, sy0 = max(0, -px0), max(0, -py0)
    dx0, dy0 = max(0, px0), max(0, py0)
    dx1, dy1 = min(W, px0 + tw), min(H, py0 + th)
    mask = np.zeros((H, W), bool)
    if dx1 > dx0 and dy1 > dy0:
        aa = a[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0][:, :, None]
        out[dy0:dy1, dx0:dx1] = aa * rgb[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0] + (1 - aa) * out[dy0:dy1, dx0:dx1]
        mask[dy0:dy1, dx0:dx1] = aa[:, :, 0] > 0.5
    if "b" in prof and ctx["kind"] == "vehicle":
        out += twin_fx.lamp_layer((H, W), tw, th, px0, py0, *ctx["vel"], dark)
    info = dict(w_ratio=tw / bw, h_ratio=th / bh, area_ratio=float(mask.sum()) / max(1.0, bw * bh), gain=g, L_ring=L_ring,
                aspect_err=float(asp / (bw / bh) - 1), darkness=dark)
    return np.clip(out, 0, 255).astype(np.uint8), mask, info


# ───────────────────────────── per scene ─────────────────────────────
def real_eval(ds, cat, v, t, dets, num2idx):
    """A1 rule on the REAL frame with the dumped boxes -> (hit, matched-box score)."""
    if ds == "CDnet2014":
        g = cv2.imread(str(CDNET_ROOT / cat / v / "groundtruth" / f"gt{t:06d}.png"), cv2.IMREAD_GRAYSCALE)
        lab, n = label_binary(g == 255)
        boxes = dets.get(t - 1) or []
    else:
        rgb = cv2.imread(str(LAS_ROOT / f"{v}-GT" / f"{v}-GT_{t}.png"))[:, :, ::-1]
        lab, n, _ = label_colours(rgb)
        boxes = dets.get(num2idx[t]) or []
    e = eval_frame(lab, n, boxes)
    return bool(e["hit"]), float(e["score"])


def seed_for(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def scene_job(args):
    ds, cat, v, n_preview, torch_threads, veh_dir, tw_dir, prev_dir, sel, prof_ = args
    P["VEHICLE_SPRITES"], P["SELECT"], P["PROFILE"] = veh_dir, sel, prof_
    tw, prev = Path(tw_dir), Path(prev_dir)
    out_p = tw / f"frames_{ds}__{v}.parquet"
    if out_p.exists():
        return str(out_p), "cached"
    import torch
    torch.set_num_threads(torch_threads)
    from detector_guard import PREDICT_KW, load_detector
    model = load_detector()
    F = pd.read_parquet(RESULTS / f"gstar_frames_{ds}.parquet")
    F = F[F.video == v].sort_values("frame").reset_index(drop=True)
    if cat is None and "category" in F:
        cat = F.category.iloc[0]
    E = pd.read_csv(RESULTS / "miss_matrix_Gstar_a256.csv")
    E = E[(E.dataset == ds) & (E.video == v)]
    ec = json.load(open(RESULTS / "event_classes.json", encoding="utf-8"))
    dom = {}
    for r in ec.get("cdnet44_per_scene", []):
        cnt = {k: r.get(k, 0) for k in ("person", "vehicle", "other")}
        dom[r["video"]] = max(cnt, key=cnt.get) if sum(cnt.values()) else "vehicle"
    plate, pinfo, bg_imgs = make_plate(ds, cat, v, F)
    prof = P["PROFILE"]
    stats = twin_fx.scene_stats(bg_imgs, plate) if prof else {}
    pinfo.update({f"stat_{k}": v2 for k, v2 in stats.items()})
    if ds == "CDnet2014":
        dets, num2idx = load_dets("CDnet2014", f"{cat}__{v}.json"), None
    else:
        from a4_gstar_ext import las_json
        dets = load_dets("LASIESTA", las_json(v))
        num2idx = {n: i for i, n in enumerate(lasiesta_order(v))}
    real_cache = {}
    S = Sprites()
    Fi = F.set_index("frame")
    rows, prev_n = [], 0
    prev.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for e in E.itertuples():
        fr = [t for t in range(int(e.onset_frame), int(e.onset_frame) + int(e.duration)) if t in Fi.index]
        seg = Fi.loc[fr]
        seg = seg[seg.has_fg.astype(bool)]
        if len(seg) == 0:
            continue
        bbs = np.array([list(map(int, b.split(","))) for b in seg.L_bbox.values])
        asp = float(np.median((bbs[:, 2] - bbs[:, 0]) / np.maximum(1, bbs[:, 3] - bbs[:, 1])))
        cg = e.cls_group
        imputed = cg == "unmatched"
        if imputed:
            cg = dom.get(v, "person" if ds == "LASIESTA" else "vehicle")
        kind = "person" if cg == "person" else "vehicle"
        dx = float(((bbs[-1, 0] + bbs[-1, 2]) - (bbs[0, 0] + bbs[0, 2])) / 2)
        for vi, sd in enumerate(P["VARIANT_SEEDS"]):
            rng = np.random.default_rng(seed_for(sd, ds, v, e.event_id))
            ch = S.choose(kind, asp, rng, vi)
            turb = twin_fx.TurbField(stats.get("turb_sigma", 0), stats.get("turb_ell", 1), stats.get("turb_rho", 0), rng) if prof else None
            cxs = (bbs[:, 0] + bbs[:, 2]) / 2
            cys = (bbs[:, 1] + bbs[:, 3]) / 2
            mirror = bool(rng.random() < 0.5) if kind == "vehicle" else (dx < 0)
            covered = set()                      # twin-hit residues mod LCM (every K in K_GRID divides 48)
            for k, t in enumerate(seg.index.values):
                if len(covered) == P["LCM"]:
                    break                            # exact early stop: R_twin(K) is full for every K

                real = cv2.imread(str(frame_path(ds, cat, v, int(t))))
                if int(t) not in real_cache:
                    real_cache[int(t)] = real_eval(ds, cat, v, int(t), dets, num2idx)
                is_grey = bool(np.abs(real[:, :, 0].astype(int) - real[:, :, 2]).mean() < 2)
                rm = gt_remove_mask(ds, cat, v, int(t), real.shape)
                base = real.copy()
                base[rm] = plate[rm]
                a_t = (bbs[k, 2] - bbs[k, 0]) / max(1, bbs[k, 3] - bbs[k, 1])
                f = ch["pick"](k, a_t) if "pick" in ch else ch["files"][(k // P["CYCLE"]) % len(ch["files"])]
                k0, k1 = max(0, k - 1), min(len(bbs) - 1, k + 1)
                vel = (float(cxs[k1] - cxs[k0]) / max(1, k1 - k0), float(cys[k1] - cys[k0]) / max(1, k1 - k0))
                ctx = dict(prof=prof, kind=kind, vel=vel, stats=stats, turb=turb, rng=rng) if prof else None
                twin, tmask, info = place(base, S.img(kind, f), bbs[k], is_grey, mirror, ctx)
                res = model.predict(twin, **PREDICT_KW)[0]
                boxes = []
                if res.boxes is not None and len(res.boxes):
                    d = res.boxes.data.cpu().numpy()
                    boxes = [[int(b[0]), int(b[1]), int(b[2]), int(b[3]), round(float(b[4]), 4), int(b[5])] for b in d]
                if tmask.sum() > 0:
                    lab, n = label_binary(tmask)
                    ev = eval_frame(lab, n, boxes)
                else:
                    ev = dict(hit=False, hit_cover=False, hit_iou=False, cls=-1, score=float("nan"), best_iou=0.0, best_cover=0.0)
                if ev["hit"]:
                    covered.add(int(t) % P["LCM"])
                rows.append(dict(dataset=ds, category=cat, video=v, event_id=int(e.event_id), def_a256=bool(e.def_a256),
                                 variant=vi, seed=sd, frame=int(t), kind=kind, cls_imputed=bool(imputed),
                                 sprite=f, aspect_miss=ch["aspect_miss"], mirror=mirror,
                                 twin_hit=bool(ev["hit"]), twin_hit_cover=bool(ev["hit_cover"]), twin_hit_iou=bool(ev["hit_iou"]),
                                 twin_cls=int(ev["cls"]), twin_score=float(ev["score"]), twin_best_iou=float(ev["best_iou"]),
                                 n_boxes=len(boxes), max_score=max([b[4] for b in boxes], default=0.0),
                                 twin_mask_px=int(tmask.sum()), real_hit=real_cache[int(t)][0],
                                 real_score=real_cache[int(t)][1], real_hit_table=bool(seg.loc[t, "hit"]),
                                 **{k2: float(v2) for k2, v2 in info.items()}))
                n_ev = max(1, len(E))
                per_ev = max(1, -(-n_preview // n_ev))                 # previews per event (ceil)
                step = max(1, min(len(seg), P["LCM"]) // per_ev)
                if prev_n < n_preview and vi == 0 and k % step == 0 and k // step < per_ev:
                    vis = twin.copy()
                    for b in boxes:
                        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0) if ev["hit"] else (0, 0, 255), 1)
                    cv2.imwrite(str(prev / f"{ds}__{v}__e{int(e.event_id):03d}_f{int(t)}.jpg"),
                                np.hstack([real, twin, vis]))
                    prev_n += 1
    df = pd.DataFrame(rows)
    for k2, v2 in pinfo.items():
        df[k2] = v2
    tw.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_p, index=False)
    return str(out_p), f"{len(df)} rows {time.time() - t0:.0f}s plate={pinfo['plate_source']}"


def scene_list(mode):
    if mode == "pilot":
        return PILOT
    if mode == "v17":                  # P1b': scene list and tiers from the #17 corpus (corpus17.py)
        bg = json.load(open(RESULTS / "p1_background17.json", encoding="utf-8"))
        return [(r["dataset"], r["category"] if r["dataset"] == "CDnet2014" else None, r["video"]) for r in bg["scenes"]]
    raise ValueError("P1b' uses mode 'v17' (corpus17 scene list); the P1b list is in ../legacy/build_twin_p1b.py")


def scene_kinds(scenes):
    """Sprite kinds each scene needs (same rule as scene_job): lets person-only scenes run before the full
    VKITTI2 archive is available."""
    E = pd.read_csv(RESULTS / "miss_matrix_Gstar_a256.csv")
    ec = json.load(open(RESULTS / "event_classes.json", encoding="utf-8"))
    dom = {}
    for r in ec.get("cdnet44_per_scene", []):
        cnt = {k: r.get(k, 0) for k in ("person", "vehicle", "other")}
        dom[r["video"]] = max(cnt, key=cnt.get) if sum(cnt.values()) else "vehicle"
    out = {}
    for ds, cat, v in scenes:
        ks = set()
        for cg in E[(E.dataset == ds) & (E.video == v)].cls_group:
            if cg == "unmatched":
                cg = dom.get(v, "person" if ds == "LASIESTA" else "vehicle")
            ks.add("person" if cg == "person" else "vehicle")
        out[v] = ks
    return out


def main():
    mode = sys.argv[1]
    for flag, key in (("--select", "SELECT"), ("--profile", "PROFILE")):
        if flag in sys.argv:
            P[key] = sys.argv[sys.argv.index(flag) + 1].replace("none", "")
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else None
    if "--vehicle-sprites" in sys.argv:
        P["VEHICLE_SPRITES"] = sys.argv[sys.argv.index("--vehicle-sprites") + 1]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    n_prev = int(sys.argv[sys.argv.index("--preview") + 1]) if "--preview" in sys.argv else (20 if mode == "pilot" else 3)
    tw = RESULTS / (f"twin_{tag}" if tag else ("twin_pilot" if mode == "pilot" else "twin"))
    prev = (RESULTS / "twin_preview_v2" / tag) if tag else (PREV / ("pilot" if mode == "pilot" else "full"))
    tw.mkdir(parents=True, exist_ok=True)
    scenes = scene_list(mode)
    if "--only" in sys.argv:           # comma-separated video names
        keep = set(sys.argv[sys.argv.index("--only") + 1].split(","))
        scenes = [s for s in scenes if s[2] in keep]
    if "--tier" in sys.argv:
        bg17 = json.load(open(RESULTS / "p1_background17.json", encoding="utf-8"))["by_tier"]
        keep = set(sum((bg17.get(t, []) for t in sys.argv[sys.argv.index("--tier") + 1].split(",")), []))
        scenes = [s for s in scenes if s[2] in keep]
    print("scenes", len(scenes), flush=True)
    if "--scenes" in sys.argv:
        sel = sys.argv[sys.argv.index("--scenes") + 1]
        kinds = scene_kinds(scenes)
        scenes = [s for s in scenes if (sel == "person_only") == ("vehicle" not in kinds[s[2]])]
        print(sel, len(scenes), "scenes", flush=True)
    man = dict(mode=mode, params=P, scenes=[list(s) for s in scenes], workers=workers,
               detector=json.load(open(RESULTS / "a0_detector_guard.json"))["verified"],
               vehicle_sprites=len(pd.read_csv(SPR / P["VEHICLE_SPRITES"] / "metadata.csv"))
               if (SPR / P["VEHICLE_SPRITES"] / "metadata.csv").exists() else None,
               person_sprites=len(pd.read_csv(SPR / P["PERSON_SPRITES"] / "metadata.csv")),
               doc=__doc__)
    mtag = tag or (sys.argv[sys.argv.index("--scenes") + 1] if "--scenes" in sys.argv else "all")
    json.dump(man, open(RESULTS / f"twin_manifest_{mode}_{mtag}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    thr = max(1, (os.cpu_count() or 8) // workers)
    jobs = [(ds, cat, v, n_prev, thr, P["VEHICLE_SPRITES"], str(tw), str(prev), P["SELECT"], P["PROFILE"]) for ds, cat, v in scenes]
    with get_context("spawn").Pool(workers) as pool:
        for p, msg in pool.imap_unordered(scene_job, jobs):
            print(p, msg, flush=True)


if __name__ == "__main__":
    main()
