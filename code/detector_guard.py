"""A0 — detector guard for every #17 step that runs the detector (P1a onwards).

Loads ONLY the fixed detector weights yolo26s-seg.pt (copy inside folder 17: assets/models/, D0 28-09-2026), after checking its sha256 against the
recorded value (results/p0_inventory.json). Any mismatch or missing file raises. There is NO fallback to
another model (a silent fallback to yolov8n would change every number; it must never happen here).

Usage:
    from detector_guard import load_detector, verify_weights
    model = load_detector()          # ultralytics.YOLO, verified
    rec = verify_weights()           # dict(path, sha256, size) or raises
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from common import RESULTS

WEIGHTS = Path(__file__).resolve().parents[1] / "assets" / "models" / "yolo26s-seg.pt"  # 23 MB, self-contained
MODEL_NAME = "yolo26s-seg"
FORBIDDEN = ("yolov8n", "yolov8n-seg", "yolov8")
# Ultralytics defaults, the same settings as the per-frame detection dumps
PREDICT_KW = dict(conf=0.25, iou=0.7, imgsz=640, verbose=False)


class DetectorGuardError(RuntimeError):
    pass


def expected_sha256() -> str:
    inv = json.load(open(RESULTS / "p0_inventory.json", encoding="utf-8"))
    note = inv["raw_datasets"]["detector_P1_YOLO_Only"]["note"]
    m = re.search(r"sha256 ([0-9a-f]{64})", note)
    if not m:
        raise DetectorGuardError("p0_inventory.json has no sha256 for the P1_YOLO_Only weights")
    return m.group(1)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_weights(path: Path = WEIGHTS) -> dict:
    path = Path(path)
    if any(s in path.name.lower() for s in FORBIDDEN) or MODEL_NAME not in path.name:
        raise DetectorGuardError(f"refusing weights {path.name}: only {MODEL_NAME}.pt is allowed")
    if not path.is_file():
        raise DetectorGuardError(f"weights missing: {path} (no fallback allowed)")
    exp, got = expected_sha256(), sha256_file(path)
    if got != exp:
        raise DetectorGuardError(f"sha256 mismatch for {path}: got {got}, expected {exp}")
    return dict(path=str(path), sha256=got, size_bytes=path.stat().st_size, model=MODEL_NAME)


def load_detector(path: Path = WEIGHTS):
    """Return an ultralytics.YOLO model for the verified weights. Raises on any doubt."""
    rec = verify_weights(path)
    from ultralytics import YOLO  # imported only after the guard passes
    model = YOLO(rec["path"])
    ck = str(getattr(model, "ckpt_path", "") or rec["path"])
    if MODEL_NAME not in Path(ck).name:
        raise DetectorGuardError(f"loaded checkpoint {ck} is not {MODEL_NAME}")
    if getattr(model, "task", "segment") != "segment":
        raise DetectorGuardError(f"expected a segmentation model, got task={model.task}")
    return model


if __name__ == "__main__":
    import sys
    from common import dump
    rec = verify_weights()
    out = dict(verified=rec, predict_kwargs=PREDICT_KW, fallback="none (raises DetectorGuardError)")
    # negative tests: a wrong path and a forbidden name must both raise
    neg = {}
    for label, p in [("missing", WEIGHTS.with_name("does_not_exist_yolo26s-seg.pt")),
                     ("forbidden_name", WEIGHTS.with_name("yolov8n.pt"))]:
        try:
            verify_weights(p)
            neg[label] = "DID NOT RAISE"
        except DetectorGuardError as e:
            neg[label] = f"raised: {e}"
    out["negative_tests"] = neg
    if "--load" in sys.argv:
        m = load_detector()
        out["loaded"] = dict(task=m.task, n_classes=len(m.names), ckpt=str(getattr(m, "ckpt_path", "")))
    dump(out, "a0_detector_guard.json")
    print(json.dumps(out, indent=1))
    assert all(v.startswith("raised") for v in neg.values())
