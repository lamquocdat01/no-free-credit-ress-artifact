"""B3 — sha256 of every downloaded asset + sprite sets -> results/asset_inventory.json, checkpoint_B3.json."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import RESULTS, dump  # noqa: E402

ROOT = HERE.parent.parent
A = ROOT / "assets"
DS = Path(os.environ.get("THS_DATASETS", "datasets"))   # D0 28-09-2026: VKITTI2 + tools out of Drive sync
VK, TOOLS = DS / "VKITTI2", DS / "_tools"
EXPECTED = {"vkitti_2.0.3_rgb.tar": 7532472320, "vkitti_2.0.3_instanceSegmentation.tar": 172298240,
            "vkitti_2.0.3_textgt.tar.gz": 24451078}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main():
    files = []
    for p in [VK / n for n in EXPECTED] + [TOOLS / "blender" / "blender.zip", TOOLS / "makehuman" / "add-on-mpfb-v2.0.17.zip"]:
        if p.exists():
            rec = dict(file=str(p), bytes=p.stat().st_size, sha256=sha(p),
                       modified=datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"))
            if p.name in EXPECTED:
                rec["expected_bytes"] = EXPECTED[p.name]
                rec["complete"] = p.stat().st_size == EXPECTED[p.name]
            files.append(rec)
        else:
            files.append(dict(file=str(p), missing=True))   # the 2 extracted VKITTI2 tars were deleted in D0 (see notes)
    sprites = {}
    for d in sorted((A / "sprites").iterdir()):
        if d.is_dir() and (d / "metadata.csv").exists():
            md = pd.read_csv(d / "metadata.csv")
            hs = sorted(sha(d / f) for f in md.file)
            sprites[d.name] = dict(n=len(md), sha256_of_sorted_file_hashes=hashlib.sha256("".join(hs).encode()).hexdigest(),
                                   **({"by_variation": md.variation.value_counts().to_dict(), "by_label": md.label.value_counts().to_dict(),
                                       "n_scenes": int(md.scene.nunique())} if "variation" in md else
                                      {"n_characters": int(md.char.nunique()), "azimuths": sorted(md.azimuth.unique().tolist()),
                                       "poses": sorted(md.pose.unique().tolist())}))
    out = dict(generated=datetime.now().isoformat(timespec="seconds"), files=files, sprites=sprites,
               blender=dict(version="4.2.23 LTS", mode="portable (THS_DATASETS/_tools/blender/.../portable)",
                            render_test_seconds=3.43, render_test="Cycles CPU 16 spp 256x256"),
               licenses="assets/LICENSES.md")
    dump(out, "asset_inventory.json")
    dump(dict(step="B3", files_complete=all(f.get("complete", True) and not f.get("missing") for f in files),
              sprites={k: v["n"] for k, v in sprites.items()}), "checkpoint_B3.json")
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
