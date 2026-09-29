"""B3 — person sprites rendered headless in Blender 4.2 with MPFB2 (MakeHuman base mesh + targets, CC0 assets;
MPFB code GPLv3). The MakeHuman clothes/skin asset packs download at ~8 KB/s from every mirror, so clothing is
built procedurally: per-face materials by body region (shirt / trousers / shoes / skin / hair cap from the
MakeHuman 'helper-hair' geometry). Each character: random macro (gender, age, muscle, weight, height, race),
random clothing colours; 3 walking poses x 8 azimuths, camera elevated ELEV degrees (surveillance view).
Run:  %THS_DATASETS%\_toolslenderlender-4.2.23-windows-x64lender.exe --background --factory-startup
          --python render_people.py -- OUT_DIR N_CHAR SEED
      (portable Blender + MPFB add-on moved out of Drive sync to THS_DATASETS/_tools on 28-09-2026, D0)
Output: OUT_DIR/char{i:02d}_az{az:03d}_p{k}.png (RGBA) + OUT_DIR/render_log.json
"""
import json
import math
import random
import sys
import time

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
OUT, N_CHAR, SEED = argv[0], int(argv[1]), int(argv[2])
AZ = [0, 45, 90, 135, 180, 225, 270, 315]
ELEV = 25.0
RES = (160, 320)
SAMPLES = 16

bpy.ops.preferences.addon_enable(module="bl_ext.user_default.mpfb")
from bl_ext.user_default.mpfb.services.humanservice import HumanService  # noqa: E402

# axes probed on the MPFB default rig (probe 27-09): upperarm01 local Z lowers the arm from the A-pose
# (L: -, R: +); upperleg01 / upperarm01 local X swings forward-back. Knee sign is probed per character.
ARMS_DOWN = {"upperarm01.L": (0, 0, -40), "upperarm01.R": (0, 0, 40)}
SWING = {"p0": (22, -18), "p1": (0, -4), "p2": (-18, 22)}   # (left leg, right leg) degrees about local X


def mat(name, rgb, rough=0.7):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    return m


def rnd_col(rng, dark=False):
    if dark:
        v = rng.uniform(0.02, 0.15)
        return (v, v * rng.uniform(0.8, 1.1), v * rng.uniform(0.8, 1.2))
    h = rng.random()
    s = rng.uniform(0.1, 0.8)
    v = rng.uniform(0.08, 0.8)
    import colorsys
    return colorsys.hsv_to_rgb(h, s, v)


def build_character(i, rng):
    macro = {"gender": float(i % 2), "age": rng.uniform(0.3, 0.75), "muscle": rng.uniform(0.3, 0.7),
             "weight": rng.uniform(0.3, 0.85), "proportions": 0.5, "height": rng.uniform(0.25, 0.85),
             "cupsize": 0.5, "firmness": 0.5}
    r = [rng.random() for _ in range(3)]
    macro["race"] = {"asian": r[0] / sum(r), "caucasian": r[1] / sum(r), "african": r[2] / sum(r)}
    bm = HumanService.create_human(mask_helpers=False, macro_detail_dict=macro)
    # keep body + hair cap helper only
    vg = bm.vertex_groups
    keep = bm.vertex_groups.new(name="keep")
    idx_body = vg["body"].index
    idx_hair = vg["helper-hair"].index
    ids = [v.index for v in bm.data.vertices if any(g.group in (idx_body, idx_hair) and g.weight > 0.5 for g in v.groups)]
    long_hair = rng.random() < (0.7 if macro["gender"] < 0.5 else 0.1)
    if not long_hair:   # short hair: drop the long-hair helper shell, paint the scalp instead (below)
        ids = [v.index for v in bm.data.vertices if any(g.group == idx_body and g.weight > 0.5 for g in v.groups)]
    keep.add(ids, 1.0, "REPLACE")
    mod = bm.modifiers.new("keep", "MASK")
    mod.vertex_group = "keep"
    # region materials from rest-pose geometry
    zs = [v.co.z for v in bm.data.vertices]
    zmin, zmax = min(zs), max(zs)
    H = zmax - zmin
    skin_tone = rng.choice([(0.8, 0.6, 0.5), (0.6, 0.42, 0.3), (0.35, 0.22, 0.15), (0.9, 0.72, 0.6)])
    mats = dict(skin=mat("skin", skin_tone, 0.5), shirt=mat("shirt", rnd_col(rng)), pants=mat("pants", rnd_col(rng)),
                shoes=mat("shoes", rnd_col(rng, dark=True)), hair=mat("hair", rnd_col(rng, dark=True), 0.9))
    order = ["skin", "shirt", "pants", "shoes", "hair"]
    for k in order:
        bm.data.materials.append(mats[k])
    hair_set = set(v.index for v in bm.data.vertices if any(g.group == idx_hair and g.weight > 0.5 for g in v.groups))
    long_sleeve = rng.random() < 0.5
    for p in bm.data.polygons:
        c = p.center
        zr = (c.z - zmin) / H
        ax = abs(c.x) / H
        if all(v in hair_set for v in p.vertices) or (not long_hair and zr > 0.925 and c.y > -0.02 * H):
            k = "hair"
        elif zr < 0.045:
            k = "shoes"
        elif zr < 0.50 and ax < 0.13:
            k = "pants"
        elif zr < 0.83 and (ax < 0.11 or (long_sleeve and ax < 0.30) or (not long_sleeve and ax < 0.17)):
            k = "shirt"
        else:
            k = "skin"
        p.material_index = order.index(k)
    rig = HumanService.add_builtin_rig(bm, "default")
    return bm, rig, macro, long_sleeve, long_hair


def knee_sign(rig):
    """+1 or -1 such that bending the knee moves the foot AWAY from the toes (backwards)."""
    f = rig.pose.bones["foot.L"]
    toes = (rig.matrix_world @ f.tail - rig.matrix_world @ f.head).y
    pb = rig.pose.bones["lowerleg01.L"]
    pb.rotation_mode = "XYZ"
    y0 = (rig.matrix_world @ f.head).y
    pb.rotation_euler = (math.radians(30), 0, 0)
    bpy.context.view_layer.update()
    y1 = (rig.matrix_world @ f.head).y
    pb.rotation_euler = (0, 0, 0)
    bpy.context.view_layer.update()
    return 1 if (y1 - y0) * toes < 0 else -1


def pose(rig, name, ks):
    for b in rig.pose.bones:
        b.rotation_mode = "XYZ"
        b.rotation_euler = (0, 0, 0)
    l, r = SWING[name]
    rot = {"upperarm01.L": [0, 0, ARMS_DOWN["upperarm01.L"][2]], "upperarm01.R": [0, 0, ARMS_DOWN["upperarm01.R"][2]],
           "upperleg01.L": [l, 0, 0], "upperleg01.R": [r, 0, 0]}
    rot["upperarm01.L"][0] = -0.6 * l          # arms counter-swing
    rot["upperarm01.R"][0] = -0.6 * r
    back = "L" if l < r else "R"               # the trailing leg bends its knee
    rot["lowerleg01." + back] = [ks * 25 if name != "p1" else ks * 35, 0, 0]
    for bone, e in rot.items():
        rig.pose.bones[bone].rotation_euler = tuple(math.radians(v) for v in e)
    bpy.context.view_layer.update()


def setup_scene():
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = SAMPLES
    sc.render.resolution_x, sc.render.resolution_y = RES
    sc.render.film_transparent = True
    sc.world = bpy.data.worlds.new("w") if sc.world is None else sc.world
    sc.world.use_nodes = True
    sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.55, 0.6, 1)
    sc.world.node_tree.nodes["Background"].inputs[1].default_value = 0.8
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(40), math.radians(15), math.radians(30))
    sc.collection.objects.link(sun)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.type = "ORTHO"
    sc.collection.objects.link(cam)
    sc.camera = cam
    return cam


def aim(cam, az, target, dist, ortho):
    a, e = math.radians(az), math.radians(ELEV)
    cam.location = target + Vector((dist * math.cos(e) * math.sin(a), -dist * math.cos(e) * math.cos(a), dist * math.sin(e)))
    d = target - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    cam.data.ortho_scale = ortho


def main():
    rng = random.Random(SEED)
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    cam = setup_scene()
    log = dict(seed=SEED, n_char=N_CHAR, azimuths=AZ, elevation_deg=ELEV, res=RES, samples=SAMPLES,
               blender=bpy.app.version_string, chars=[])
    t0 = time.time()
    for i in range(N_CHAR):
        bm, rig, macro, ls, lh = build_character(i, rng)
        zs = [v.co.z for v in bm.data.vertices]
        H = max(zs) - min(zs)
        target = Vector((0, 0, min(zs) + 0.5 * H))
        ks = knee_sign(rig)
        for pname in SWING:
            pose(rig, pname, ks)
            for az in AZ:
                aim(cam, az, target, 10 * H, 1.15 * H)
                bpy.context.scene.render.filepath = f"{OUT}/char{i:02d}_az{az:03d}_{pname}.png"
                bpy.ops.render.render(write_still=True)
        log["chars"].append(dict(i=i, macro=macro, long_sleeve=ls, long_hair=lh, height_units=H))
        for o in (bm, rig):
            bpy.data.objects.remove(o)
        print(f"CHAR {i} done {time.time() - t0:.1f}s", flush=True)
    log["seconds"] = time.time() - t0
    json.dump(log, open(f"{OUT}/render_log.json", "w"), indent=1)


main()
