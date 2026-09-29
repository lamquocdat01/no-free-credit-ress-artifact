"""Mode G* (object-level) helpers shared by A1 (CDnet) and A4 (LASIESTA, BMC-synth).

Frame t with GT foreground is a HIT iff there is a predicted box b with
    cover(b, L) = |L ∩ b| / |L| >= 0.5          (rule "cover50")
 or IoU(b, bbox(L)) >= 0.3                       (rule "iou30")
where L is the LARGEST GT object of the frame (CDnet/BMC: largest 8-connected component of the
foreground mask; LASIESTA: largest colour instance). Sensitivity "any": the same rule against ANY
GT object with area >= MIN_OBJ_PX.
Boxes come from the per-frame detection dumps of the fixed detector; the detector is NOT re-run.

Frame-index conventions (verified, see a1_gstar.py self-checks):
  CDnet   : JSON frame f (0-based sort index, ROI applied to the 0-based index) <-> in/gt number f+1.
  BMC     : JSON frame f <-> frames/gt %06d number f+1 (zero-padded names, sort == numeric).
  LASIESTA: JSON frame f <-> f-th file of the LEXICOGRAPHICALLY sorted *.bmp list (names are NOT
            zero-padded: -1, -10, -100, ...); the temporal frame number is parsed from the name.
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from common import DET_ROOT

DATASETS = Path(os.environ.get("THS_DATASETS", "datasets"))  # datasets moved out of Drive sync 28-09-2026
CDNET_ROOT = DATASETS / "CDnet2014" / "archive" / "dataset"
LAS_ROOT = DATASETS / "LASIESTA"
BMC_ROOT = DATASETS / "BMC"

COVER_THR, IOU_THR = 0.5, 0.3
MIN_OBJ_PX = 20
MERGE_GAP = 1

COCO_PERSON = {0}
COCO_VEHICLE = {1, 2, 3, 4, 5, 6, 7, 8}          # bicycle, car, motorcycle, airplane, bus, train, truck, boat
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train",
              7: "truck", 8: "boat"}


def class_group(c):
    if c is None or c < 0:
        return "unmatched"
    if c in COCO_PERSON:
        return "person"
    if c in COCO_VEHICLE:
        return "vehicle"
    return "other"


# ───────────────────────────── per-frame matching ─────────────────────────────
def _objects_from_labels(lab, n):
    """lab: int32 label image (0 = background). Returns list of (area, bbox[x0,y0,x1,y1) , id)."""
    areas = np.bincount(lab.ravel(), minlength=n)
    objs = []
    for i, sl in enumerate(ndi.find_objects(lab, max_label=n - 1), 1):
        if sl is None or areas[i] == 0:
            continue
        objs.append((int(areas[i]), (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop), i))
    return objs


def eval_frame(lab, n_lab, boxes):
    """lab: label image of GT objects (0 bg), n_lab = max label + 1. boxes: [[x1,y1,x2,y2,score,cls],..].
    Returns dict of per-frame quantities (has_fg must be decided by the caller)."""
    objs = _objects_from_labels(lab, n_lab)
    out = dict(n_obj=len(objs), hit=False, hit_cover=False, hit_iou=False, hit_any=False,
               best_cover=0.0, best_iou=0.0, cls=-1, cls_any=-1, L_area=0, L_bbox=None, score=float("nan"))
    if not objs:
        return out
    objs.sort(key=lambda o: -o[0])
    H, W = lab.shape

    def score_obj(area, bb, oid):
        m = (lab == oid).astype(np.int32)
        ii = np.pad(m.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        best = []  # (cover, iou, score, cls)
        for b in boxes:
            x1, y1 = max(0, min(W, int(b[0]))), max(0, min(H, int(b[1])))
            x2, y2 = max(0, min(W, int(b[2]))), max(0, min(H, int(b[3])))
            inside = int(ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1]) if (x2 > x1 and y2 > y1) else 0
            cov = inside / area
            ix = max(0, min(b[2], bb[2]) - max(b[0], bb[0]))
            iy = max(0, min(b[3], bb[3]) - max(b[1], bb[1]))
            inter = ix * iy
            ua = (b[2] - b[0]) * (b[3] - b[1]) + (bb[2] - bb[0]) * (bb[3] - bb[1]) - inter
            iou = inter / ua if ua > 0 else 0.0
            best.append((cov, iou, float(b[4]), int(b[5])))
        return best

    area, bb, oid = objs[0]
    out["L_area"], out["L_bbox"] = int(area), list(bb)
    if boxes:
        sc = score_obj(area, bb, oid)
        out["best_cover"] = max(s[0] for s in sc)
        out["best_iou"] = max(s[1] for s in sc)
        hits = [s for s in sc if s[0] >= COVER_THR or s[1] >= IOU_THR]
        out["hit_cover"] = any(s[0] >= COVER_THR for s in sc)
        out["hit_iou"] = any(s[1] >= IOU_THR for s in sc)
        out["hit"] = bool(hits)
        if hits:
            out["cls"] = max(hits, key=lambda s: s[2])[3]
            out["score"] = max(s[2] for s in hits)      # confidence of the best matching box
        # sensitivity: any object of area >= MIN_OBJ_PX
        if out["hit"]:
            out["hit_any"], out["cls_any"] = True, out["cls"]
        else:
            for a2, bb2, oid2 in objs[1:]:
                if a2 < MIN_OBJ_PX:
                    break
                s2 = [s for s in score_obj(a2, bb2, oid2) if s[0] >= COVER_THR or s[1] >= IOU_THR]
                if s2:
                    out["hit_any"], out["cls_any"] = True, max(s2, key=lambda s: s[2])[3]
                    break
    return out


def label_binary(fg):
    n, lab = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
    return lab, n


def label_colours(rgb, ignore_grey=(128, 128, 128)):
    """LASIESTA: one colour per object; black = background; grey 128 = ignore (not an object)."""
    flat = rgb.reshape(-1, 3)
    key = (flat[:, 0].astype(np.int32) << 16) | (flat[:, 1].astype(np.int32) << 8) | flat[:, 2]
    grey = (ignore_grey[0] << 16) | (ignore_grey[1] << 8) | ignore_grey[2]
    lab = np.zeros(len(key), np.int32)
    cols = [c for c in np.unique(key) if c not in (0, grey)]
    for i, c in enumerate(cols, 1):
        lab[key == c] = i
    return lab.reshape(rgb.shape[:2]), len(cols) + 1, [(c >> 16, (c >> 8) & 255, c & 255) for c in cols]


# ───────────────────────────── events ─────────────────────────────
def fg_runs(frames, has_fg, merge_gap=MERGE_GAP):
    """Maximal runs of has_fg over consecutive frame numbers, merging gaps <= merge_gap.
    Returns (i0, i1) index pairs into `frames` (inclusive)."""
    frames = np.asarray(frames)
    out, i, n = [], 0, len(frames)
    while i < n:
        if not has_fg[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and has_fg[j + 1] and frames[j + 1] == frames[j] + 1:
            j += 1
        out.append([i, j])
        i = j + 1
    if merge_gap > 0 and out:
        merged = [out[0]]
        for i0, i1 in out[1:]:
            if frames[i0] - frames[merged[-1][1]] - 1 <= merge_gap:
                merged[-1][1] = i1
            else:
                merged.append([i0, i1])
        out = merged
    return [tuple(x) for x in out]


def pmiss_residue(hit_frames, K):
    """Exact phase-averaged miss probability under periodic sampling with period K."""
    if len(hit_frames) == 0:
        return 1.0
    return 1.0 - len(set(int(t) % K for t in hit_frames)) / K


# ───────────────────────────── detection JSON ─────────────────────────────
def load_dets(ds, name):
    p = DET_ROOT / ds / name
    if not p.exists():
        return None
    return {int(f["frame"]): f["boxes"] for f in json.load(open(p))["frames"]}


def lasiesta_order(seq):
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
        files += glob.glob(os.path.join(str(LAS_ROOT / seq), ext))
        files += glob.glob(os.path.join(str(LAS_ROOT / seq), ext.upper()))
    files = sorted(set(files))           # exactly as dump_p1_detections.iter_frames_dir
    nums = [int(re.search(r"-(\d+)\.[A-Za-z]+$", f).group(1)) for f in files]
    return nums                           # nums[json_idx] = temporal frame number
