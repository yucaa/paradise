"""Render what the hero fly sees: an equirectangular, body-frame view from its
head, in raw panorama values (fly_view=1), at the brain's input resolution.
Run inside Blender after build_scene.py and flies.py:
    exec(open(r"<repo>/blender/render_flyview.py").read())
    render_range(0, 120)
"""
from contextlib import contextmanager
from pathlib import Path

import bpy

OUT = Path(r"C:\Users\Eric\orca\workspaces\sixteen\seventeen\out\flyview")
RES = (128, 64)


@contextmanager
def fly_vision():
    sc = bpy.context.scene
    saved = dict(engine=sc.render.engine, camera=sc.camera, res=(sc.render.resolution_x, sc.render.resolution_y),
                 path=sc.render.filepath, pct=sc.render.resolution_percentage,
                 view=sc.view_settings.view_transform, look=sc.view_settings.look)
    eden = bpy.data.objects.get("Eden")
    ink = eden.modifiers.get("Ink") if eden else None
    hero_parts = [o for o in bpy.data.objects if o.name.startswith("Hero")]
    # The fly sees the raw panorama: no haze, no ink, no post-processing.
    hidden = hero_parts + [o for o in bpy.data.objects if o.name in ("Haze", "FarGround")]
    try:
        sc.render.engine = "CYCLES"
        sc.cycles.samples = 16
        sc.cycles.device = "GPU"
        sc.camera = bpy.data.objects["FlyPOV"]
        sc.render.resolution_x, sc.render.resolution_y = RES
        sc.render.resolution_percentage = 100
        sc.view_settings.view_transform = "Standard"
        sc["fly_view"] = 1.0
        if ink:
            ink.show_render = False
        sc.render.use_compositing = False
        for o in hidden:                # nor does a fly see its own body
            o.hide_render = True
        yield sc
    finally:
        sc.render.engine = saved["engine"]
        sc.camera = saved["camera"]
        sc.render.resolution_x, sc.render.resolution_y = saved["res"]
        sc.render.resolution_percentage = saved["pct"]
        sc.render.filepath = saved["path"]
        sc.view_settings.view_transform = saved["view"]
        sc.view_settings.look = saved["look"]
        sc["fly_view"] = 0.0
        if ink:
            ink.show_render = True
        sc.render.use_compositing = True
        for o in hidden:
            o.hide_render = False


def render_range(start, end):
    OUT.mkdir(parents=True, exist_ok=True)
    with fly_vision() as sc:
        for f in range(start, end + 1):
            sc.frame_set(f)
            sc.render.filepath = str(OUT / f"{f:04d}.png")
            bpy.ops.render.render(write_still=True)
    print("rendered", start, end)
