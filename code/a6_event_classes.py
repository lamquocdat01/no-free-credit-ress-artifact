"""A6 — object-class composition of the GT events (COCO class of the matched box, majority vote over
the event's G* hit frames) for the 44 CDnet calibration scenes (+ LASIESTA), to choose sprite sources.
Events with no G* hit have no matched box -> 'unmatched' (the class is NOT inferred).
Nothing is downloaded. Output: results/event_classes.json
"""
from __future__ import annotations

import json

import pandas as pd

from common import RESULTS, dump
from gstar_common import COCO_NAMES

SPRITES = {
    "vehicle": dict(source="Virtual KITTI 2 (rgb 7.01 GB + instanceSegmentation 166 MB)",
                    url="https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/",
                    license="CC BY-NC-SA 3.0 (non-commercial, share-alike, attribution; do not redistribute frames/sprites)",
                    status="NOT downloaded — waiting for G1b approval (>5 GB)"),
    "person": [
        dict(source="MakeHuman 1.x (CC0 exports) rendered in Blender (GPL tool; renders are the user's own)",
             url="https://static.makehumancommunity.org/makehuman/faq/can_i_sell_models_created_with_makehuman.html",
             license="exported models CC0 (FAQ: 'copy, modify, distribute ... even for commercial purposes, all "
                     "without asking permission'); direct download, no form",
             role="primary: parametric body shape/clothing variety, walk cycles rendered at CDnet scale",
             status="not downloaded (license read 2026-09-27)"),
        dict(source="Quaternius — Ultimate Animated Character Pack / Universal Base Characters",
             url="https://quaternius.com/packs/ultimatedanimatedcharacter.html",
             license="CC0 (per quaternius.com / itch.io pages); free tier downloadable without account",
             role="backup: rigged + animated low-poly people (less photoreal)",
             status="not downloaded (license read 2026-09-27)"),
    ],
    "excluded": ["Mixamo (Adobe account)", "SMPL/SMPL-X, RenderPeople, MOTSynth (registration / form)",
                 "CARLA pedestrians (D4: no CARLA before G2)"],
}


def comp(df, col):
    c = df[col].value_counts()
    n = int(len(df))
    return dict(n_events=n, counts={k: int(v) for k, v in c.items()},
                pct={k: round(100 * v / n, 1) for k, v in c.items()},
                pct_of_matched={k: round(100 * v / max(1, int((df[col] != "unmatched").sum())), 1)
                                for k, v in c.items() if k != "unmatched"})


def main():
    bg = json.load(open(RESULTS / "p1_background17.json", encoding="utf-8"))  # corpus17 (raw data only)
    cal = [r["video"] for r in bg["scenes"] if r["dataset"] == "CDnet2014"]
    las_bg = [r["video"] for r in bg["scenes"] if r["dataset"] == "LASIESTA"]
    E = pd.read_csv(RESULTS / "miss_matrix_Gstar.csv")
    C = E[(E.merge_gap == 1) & E.video.isin(cal)]
    X = pd.read_csv(RESULTS / "miss_matrix_Gstar_ext.csv")
    LF = X[(X.dataset == "LASIESTA") & (X.level == "frame")]
    LI = X[(X.dataset == "LASIESTA") & (X.level == "instance")]
    out = dict(meta=dict(rule="class of the highest-score box that satisfies the A1 hit rule, majority over hit frames",
                         groups="person = COCO 0; vehicle = COCO 1-8 (bicycle..boat); other = rest; unmatched = no G* hit",
                         events="CDnet gap1 Mode-G* events in the 44 calibration scenes; LASIESTA frame + instance level"),
               cdnet44=comp(C, "cls_group"), cdnet44_any_object=comp(C, "cls_group_any"),
               cdnet_all53=comp(E[E.merge_gap == 1], "cls_group"),
               lasiesta_frame=comp(LF, "cls_group"), lasiesta_instance=comp(LI, "cls_group"),
               lasiesta_bg20_frame=comp(LF[LF.video.isin(las_bg)], "cls_group"))
    out["cdnet44_coco_classes"] = {COCO_NAMES.get(int(k), str(int(k))): int(v)
                                   for k, v in C[C.cls >= 0].cls.value_counts().items()}
    per = C.groupby(["category", "video"]).cls_group.value_counts().unstack(fill_value=0)
    out["cdnet44_per_scene"] = per.reset_index().to_dict("records")
    dom = per.idxmax(axis=1)
    out["cdnet44_scene_dominant_class"] = {k: int(v) for k, v in dom.value_counts().items()}
    pv = out["cdnet44"]["pct_of_matched"]
    veh, per_ = pv.get("vehicle", 0), pv.get("person", 0)
    out["sprite_decision"] = dict(
        vehicle_share_of_matched_pct=veh, person_share_of_matched_pct=per_,
        vkitti2_recommended=bool(veh >= 25),
        rule="download VKITTI2 only if vehicles are a substantial share (>= 25% of matched events) of the "
             "calibration events (PLAN 8b)",
        sources=SPRITES)
    dump(out, "event_classes.json")
    print(json.dumps({k: out[k] for k in ("cdnet44", "cdnet44_any_object", "lasiesta_frame", "lasiesta_instance",
                                          "cdnet44_coco_classes", "cdnet44_scene_dominant_class")}, indent=0))
    print(out["sprite_decision"]["vehicle_share_of_matched_pct"], out["sprite_decision"]["person_share_of_matched_pct"])


if __name__ == "__main__":
    main()
