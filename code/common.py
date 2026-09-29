"""Shared paths and small statistics helpers for #17 (self-contained pipeline, P1b' 2026-09-28).

#17 reads RAW data only: the per-frame detection dumps of the fixed detector (JSON, boxes + scores) and the
pixel ground truth / frames of the public datasets (CDnet2014, LASIESTA, BMC). No derived table, code or
number of any other project is read. Scripts of the P0-P1b phase that still used such tables are kept,
unmodified, in ../legacy/ (not part of the paper pipeline).
Outputs go to ../results/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "data" / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42
B_BOOT = 1000
K_GRID = [1, 2, 3, 4, 6, 8, 12, 16]
EPS, DELTA = 0.05, 0.05
REAL = ("CDnet2014", "LASIESTA")

# RAW detection dumps: one JSON per video, {"frames": [{"frame": idx, "boxes": [[x1,y1,x2,y2,score,cls],...]}]}
# produced by the fixed detector yolo26s-seg (conf 0.25, iou 0.7, imgsz 640); frame index = position in the
# text-sorted file list (see gstar_common.py for the per-dataset map to the true frame number).
# D0 (28-09-2026): copied out of Drive sync to THS_DATASETS/_derived/detections_cpu_yolo26s (+ MANIFEST.json with
# the sha256 of every JSON); byte-identical to the dumps used before (notes/D0_dataset_relocation.md).
DATASETS = Path(os.environ.get("THS_DATASETS", "datasets"))
DET_ROOT = DATASETS / "_derived" / "detections_cpu_yolo26s"


def n0(eps=EPS, delta=DELTA):
    """Classical success-run (zero-failure) sample size."""
    return int(np.ceil(np.log(delta) / np.log(1.0 - eps)))


def runs(frames, flag):
    """Maximal runs of flag==1 over consecutive frame numbers (an event = one maximal run).
    Returns list of (i0, i1) index pairs (inclusive)."""
    out, i, n = [], 0, len(flag)
    while i < n:
        if not flag[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flag[j + 1] and frames[j + 1] == frames[j] + 1:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def cp_ci(x, n, alpha=0.05):
    """Clopper-Pearson interval. x may be fractional (expected miss count under a
    uniform random phase); the beta-quantile form is used directly."""
    if n == 0:
        return [float("nan"), float("nan")]
    lo = 0.0 if x <= 0 else float(sps.beta.ppf(alpha / 2, x, n - x + 1))
    hi = 1.0 if x >= n else float(sps.beta.ppf(1 - alpha / 2, x + 1, n - x))
    return [lo, hi]


def dump(obj, name):
    p = RESULTS / name
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_conv)
    print(f"[write] {p}")
    return p


def _conv(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
