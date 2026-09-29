"""B3 — vehicle sprites from Virtual KITTI 2 (CC BY-NC-SA 3.0; sprites are NOT redistributed).

Candidates from textgt bbox.txt (Camera_0, every STEP-th frame, all 5 scenes x 10 variations):
  truncation_ratio == 0, number_pixels >= MIN_PIX, occupancy_ratio >= MIN_OCC (visible pixels / bbox area;
  unoccluded cars fill ~0.6-0.85 of their box), bbox >= 2 px away from the image border,
  mask = ONE 8-connected component holding >= 97% of the instance pixels (occluders split masks),
  and no other instance covering > 3% of the bbox (nothing in front).
Diversity: at most one sprite per (scene, variation, trackID, 60-frame bucket); then a seeded sample of
N_TARGET balanced over variation. Output: assets/sprites/vehicle/*.png (RGBA, tight crop) +
assets/sprites/vehicle/metadata.csv (scene, variation, camera, frame, trackID, label, model, color, bbox,
area, aspect, mean luminance).
"""
from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
VK = Path(os.environ.get("THS_DATASETS", "datasets")) / "VKITTI2"   # D0 28-09-2026: out of Drive sync
X = VK / "x"
OUT = ROOT / "assets" / "sprites" / "vehicle"
STEP, MIN_PIX, MIN_OCC = 10, 1500, 0.55
N_TARGET = 600
SEED = 42


def candidates():
    rows = []
    for scene in sorted(p.name for p in X.iterdir() if p.name.startswith("Scene")):
        for var in sorted(p.name for p in (X / scene).iterdir() if p.is_dir()):
            b = pd.read_csv(X / scene / var / "bbox.txt", sep=" ")
            col = pd.read_csv(X / scene / var / "colors.txt", sep=" ") if (X / scene / var / "colors.txt").exists() else None
            info = pd.read_csv(X / scene / var / "info.txt", sep=" ")
            b = b[(b.cameraID == 0) & (b.frame % STEP == 0) & (b.truncation_ratio == 0) &
                  (b.number_pixels >= MIN_PIX) & (b.occupancy_ratio >= MIN_OCC) &
                  (b.left >= 2) & (b.top >= 2) & (b.right <= 1239) & (b.bottom <= 372)]
            b = b.merge(info, on="trackID", how="left").assign(scene=scene, variation=var)
            b["bucket"] = b.frame // 60
            b = b.drop_duplicates(["trackID", "bucket"])
            rows.append(b)
    return pd.concat(rows, ignore_index=True)


def main():
    global OUT
    import sys
    if len(sys.argv) > 1:
        OUT = ROOT / "assets" / "sprites" / sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    C = candidates()
    print("candidates", len(C), C.variation.value_counts().to_dict())
    rng = np.random.default_rng(SEED)
    C = C.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    need = {(r.scene, r.variation, int(r.frame)) for r in C.itertuples()}
    rgb = {}
    truncated = False
    try:
        with tarfile.open(VK / "vkitti_2.0.3_rgb.tar") as tf:
            for m in tf:
                p = m.name.split("/")
                if len(p) == 6 and p[3] == "rgb" and p[4] == "Camera_0":
                    fr = int(p[5].split("_")[1].split(".")[0])
                    if (p[0], p[1], fr) in need:
                        rgb[(p[0], p[1], fr)] = tf.extractfile(m).read()
    except (tarfile.ReadError, EOFError, OSError) as e:      # partially downloaded archive (pilot only)
        truncated = True
        print("tar truncated:", e)
    print("rgb frames loaded", len(rgb), "of", len(need))
    kept, per_var = [], {}
    avail = {k[1] for k in rgb}
    cap = int(np.ceil(N_TARGET / max(1, len(avail)))) + 5
    for r in C.itertuples():
        key = (r.scene, r.variation, int(r.frame))
        if key not in rgb or per_var.get(r.variation, 0) >= cap:
            continue
        inst = np.array(Image.open(X / r.scene / r.variation / "frames" / "instanceSegmentation" / "Camera_0" /
                                   f"instancegt_{int(r.frame):05d}.png"))
        img = cv2.imdecode(np.frombuffer(rgb[key], np.uint8), cv2.IMREAD_COLOR)
        x0, x1, y0, y1 = int(r.left), int(r.right) + 1, int(r.top), int(r.bottom) + 1
        m = inst == (int(r.trackID) + 1)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        if n < 2 or st[1:, cv2.CC_STAT_AREA].max() < 0.97 * m.sum():
            continue
        box = inst[y0:y1, x0:x1]
        other = (box > 0) & (box != int(r.trackID) + 1)
        if other.mean() > 0.03:
            continue
        a = (m[y0:y1, x0:x1] * 255).astype(np.uint8)
        crop = img[y0:y1, x0:x1]
        rgba = np.dstack([crop, a])
        name = f"{r.scene}_{r.variation}_f{int(r.frame):05d}_t{int(r.trackID):03d}.png"
        cv2.imwrite(str(OUT / name), rgba)
        lum = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)[a > 0].mean())
        kept.append(dict(file=name, scene=r.scene, variation=r.variation, camera=0, frame=int(r.frame),
                         trackID=int(r.trackID), label=r.label, model=r.model, color=r.color,
                         x0=x0, y0=y0, x1=x1, y1=y1, w=x1 - x0, h=y1 - y0, area_px=int(m.sum()),
                         aspect=(x1 - x0) / (y1 - y0), occupancy=float(r.occupancy_ratio), lum_mean=lum))
        per_var[r.variation] = per_var.get(r.variation, 0) + 1
        if len(kept) >= N_TARGET:
            break
    md = pd.DataFrame(kept)
    md["provisional_truncated_tar"] = truncated
    md.to_csv(OUT / "metadata.csv", index=False)
    print("sprites", len(md), md.variation.value_counts().to_dict(), md.label.value_counts().to_dict())
    print(md[["w", "h", "aspect", "lum_mean"]].describe().round(2))


if __name__ == "__main__":
    main()
