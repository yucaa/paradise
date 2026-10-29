"""Connect the brain simulation to the scene. Run inside Blender after
eden.py and flies.py:
    exec(open(r"<repo>/blender/wire_brain.py").read())

- Every heaven material's activity texture becomes the per-frame retinotopic
  activity sequence, sampled from the hero's eye (FlyPOV).
- Glow strength (scene["glow"]) is keyed to the hero's smoothed valence.
"""
import json
from pathlib import Path

import bpy

JOURNEY = Path(r"C:\Users\Eric\orca\workspaces\sixteen\seventeen\out\journey")
GLOW_BASE = 2.0
GLOW_PER_VALENCE = 0.6


def main():
    sc = bpy.context.scene
    frames = json.loads((JOURNEY / "journey.json").read_text())["frames"]
    seq = bpy.data.images.load(str(JOURNEY / "activity" / "0000.png"), check_existing=True)
    seq.source = "SEQUENCE"
    seq.colorspace_settings.is_data = True
    pov = bpy.data.objects["FlyPOV"]
    wired = 0
    for m in bpy.data.materials:
        if not m.node_tree or "Activity" not in m.node_tree.nodes:
            continue
        nodes = m.node_tree.nodes
        nodes["ActivityFrame"].object = pov
        tex = nodes["Activity"]
        tex.image = seq
        tex.image_user.frame_duration = len(frames)
        tex.image_user.frame_start = 0
        tex.image_user.frame_offset = -1
        tex.image_user.use_auto_refresh = True
        wired += 1
    if sc.animation_data:
        sc.animation_data_clear()
    for rec in frames:
        sc["glow"] = GLOW_BASE + GLOW_PER_VALENCE * max(rec["valence"], 0.0)
        sc.keyframe_insert('["glow"]', frame=rec["frame"])
    print("wired", wired, "materials,", len(frames), "frames")


main()
