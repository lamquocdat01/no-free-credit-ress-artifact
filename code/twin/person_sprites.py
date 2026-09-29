"""B3 — crop the Blender renders (assets/sprites/person_raw) to tight RGBA sprites + metadata.csv."""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "assets" / "sprites" / "person_raw"
OUT = ROOT / "assets" / "sprites" / "person"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = json.load(open(RAW / "render_log.json"))
    chars = {c["i"]: c for c in log["chars"]}
    rows = []
    for f in sorted(RAW.glob("char*_az*_p*.png")):
        i, az, p = re.match(r"char(\d+)_az(\d+)_(p\d)", f.stem).groups()
        im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        a = im[:, :, 3]
        ys, xs = np.nonzero(a > 8)
        crop = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        cv2.imwrite(str(OUT / f.name), crop)
        c = chars[int(i)]
        rgb = crop[:, :, :3][crop[:, :, 3] > 128]
        rows.append(dict(file=f.name, char=int(i), azimuth=int(az), pose=p, gender=c["macro"]["gender"],
                         age=c["macro"]["age"], weight=c["macro"]["weight"], height=c["macro"]["height"],
                         long_sleeve=c["long_sleeve"], long_hair=c.get("long_hair"), w=crop.shape[1], h=crop.shape[0],
                         aspect=crop.shape[1] / crop.shape[0], area_px=int((crop[:, :, 3] > 128).sum()),
                         lum_mean=float(cv2.cvtColor(rgb.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).mean())))
    md = pd.DataFrame(rows)
    md.to_csv(OUT / "metadata.csv", index=False)
    print(len(md), md.char.nunique(), md[["w", "h", "aspect"]].describe().round(2).to_dict())


if __name__ == "__main__":
    main()
