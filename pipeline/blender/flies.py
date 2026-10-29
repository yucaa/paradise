"""Low-poly flies, their flight paths, the chase camera and the hero fly's
point-of-view camera. Run inside Blender after build_scene.py:
    exec(open(r"<repo>/blender/flies.py").read())

The hero fly wanders the orchard and ends at the FlyEye point, the one place
where the world reproduces heaven exactly.
"""
import math
import random

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector, noise

FPS = 24
SECONDS = 20
FRAMES = FPS * SECONDS
EYE = Vector((0.0, 0.0, 1.2))
SWARM = 14
FLY_SCALE = 0.12      # world metres per fly body length unit
SEED = 11
HERO_SCALE = FLY_SCALE * 1.4
CUBE = 0.6
HERO_EYE = Vector((0, CUBE / 2 + 0.02, 0))   # eye camera: centre of the cube's front face


# --- model ---------------------------------------------------------------------

def fly_view_switch(n, l, shader, out):
    """Flies appear as dark silhouettes in the fly-view renders."""
    silhouette = n.new("ShaderNodeEmission")
    silhouette.inputs["Color"].default_value = (0.03, 0.03, 0.03, 1)
    sw = n.new("ShaderNodeMixShader")
    v = n.new("ShaderNodeValue")
    fc = v.outputs[0].driver_add("default_value")
    var = fc.driver.variables.new()
    var.targets[0].id_type = "SCENE"
    var.targets[0].id = bpy.context.scene
    var.targets[0].data_path = '["fly_view"]'
    fc.driver.expression = var.name
    l.new(v.outputs[0], sw.inputs["Fac"])
    l.new(shader, sw.inputs[1]); l.new(silhouette.outputs[0], sw.inputs[2])
    l.new(sw.outputs[0], out.inputs["Surface"])


def natural_material(name, color, roughness=0.35, coat=0.0, film=0.0, alpha=1.0, sheen=0.0, emit=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    n, l = m.node_tree.nodes, m.node_tree.links
    n.clear()
    out = n.new("ShaderNodeOutputMaterial")
    b = n.new("ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = color
    b.inputs["Roughness"].default_value = roughness
    b.inputs["Coat Weight"].default_value = coat
    b.inputs["Sheen Weight"].default_value = sheen
    if emit:
        b.inputs["Emission Color"].default_value = color
        b.inputs["Emission Strength"].default_value = emit
    if film:
        # Iridescent thin film, like a fly's wing membrane.
        b.inputs["Thin Film Thickness"].default_value = film
        b.inputs["Thin Film IOR"].default_value = 1.45
    if alpha < 1.0:
        b.inputs["Alpha"].default_value = alpha
        b.inputs["Transmission Weight"].default_value = 0.6
        m.surface_render_method = "BLENDED"
    fly_view_switch(n, l, b.outputs[0], out)
    return m


def fly_materials():
    return {
        # Graphite body against the pale meadow; pale, softly luminous wings.
        "body": natural_material("FlyBody", (0.035, 0.035, 0.04, 1), roughness=0.22, coat=1.0),
        "wing": natural_material("FlyWing", (1.0, 0.98, 0.94, 1), roughness=0.1, film=320, alpha=0.75, emit=0.6),
    }


def fly_mesh(mats):
    """The fly, abstracted: a softly bevelled cube. Body units; forward is +Y."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=CUBE)
    bmesh.ops.bevel(bm, geom=bm.edges[:] + bm.verts[:], offset=0.035, segments=3, affect="EDGES")
    me = bpy.data.meshes.new("FlyBody")
    bm.to_mesh(me)
    bm.free()
    me.shade_smooth()
    me.materials.append(mats["body"])
    return me


def wing_mesh(mats, side):
    """A thin rounded paddle hinged at the cube's top edge, extending sideways."""
    bm = bmesh.new()
    pts = [(0.0, 0.16), (0.7, 0.2), (1.05, 0.08), (1.05, -0.12), (0.7, -0.22), (0.0, -0.16)]
    vs = [bm.verts.new((x * side, y, 0)) for x, y in pts]
    bm.faces.new(vs if side > 0 else vs[::-1])
    me = bpy.data.meshes.new(f"Wing{side}")
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mats["wing"])
    return me


def make_fly(name, mats, body_me, wing_l, wing_r, scale):
    root = bpy.data.objects.new(name, body_me)
    root.scale = (scale,) * 3
    bpy.context.scene.collection.objects.link(root)
    wings = []
    for side, me in ((-1, wing_l), (1, wing_r)):
        w = bpy.data.objects.new(f"{name}_wing{side}", me)
        w.parent = root
        w.location = (CUBE / 2 * side, 0.0, CUBE / 2)
        bpy.context.scene.collection.objects.link(w)
        wings.append((side, w))
    return root, wings


# --- motion --------------------------------------------------------------------

def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def wander(t, seed, centre, radius, height):
    """Smooth 3D noise path."""
    p = Vector((noise.noise(Vector((t, seed, 0.0))), noise.noise(Vector((seed, t, 1.7))),
                noise.noise(Vector((2.3, seed, t)))))
    return centre + Vector((p.x * radius, p.y * radius, height[0] + (p.z + 1) / 2 * (height[1] - height[0])))


ARRIVE = (0.55, 0.85)   # fraction of the film over which the hero settles in


def arrival(f):
    return smoothstep((f / FRAMES - ARRIVE[0]) / (ARRIVE[1] - ARRIVE[0]))


def hero_position(f):
    t = f / FRAMES
    free = wander(t * 3.0, 4.2, Vector((0, 2.0, 0)), 9.0, (0.8, 3.0))
    # Land the hero's eye camera (not its body origin) on the heaven point.
    return free.lerp(EYE - HERO_EYE * HERO_SCALE, arrival(f))


def hero_orientation(f, travel):
    """Heaven is defined in the fly's body frame (forward = +Y), so the hero
    must arrive level and facing +Y to see it."""
    yaw, pitch = travel
    a = arrival(f)
    yaw = math.atan2(math.sin(yaw), math.cos(yaw)) * (1 - a)
    return pitch * (1 - a), yaw


def animate(obj, wings, pos_fn, rng, orient_fn=None):
    prev = pos_fn(0)
    phase = rng.uniform(0, math.tau)
    for f in range(0, FRAMES + 1):
        p = pos_fn(f)
        nxt = pos_fn(f + 1)
        v = nxt - p if (nxt - p).length > 1e-5 else (p - prev)
        yaw = math.atan2(-v.x, v.y)          # body +Y faces direction of travel
        pitch = math.atan2(v.z, math.hypot(v.x, v.y)) * 0.6
        if orient_fn:
            pitch, yaw = orient_fn(f, (yaw, pitch))
        obj.location = p
        obj.rotation_euler = Euler((pitch, 0, yaw))
        obj.keyframe_insert("location", frame=f)
        obj.keyframe_insert("rotation_euler", frame=f)
        # Wingbeat is ~200 Hz, far above frame rate: strobe through the stroke.
        beat = math.sin(phase + f * 2.39996)
        for side, w in wings:
            w.rotation_euler = Euler((0, side * beat * 0.9, -side * 0.15))
            w.keyframe_insert("rotation_euler", frame=f)
        prev = p


def cameras(hero):
    sc = bpy.context.scene
    # Chase camera: trails the hero, looks at it, eases in at the end.
    cam = sc.camera
    cam.data.lens = 50
    cam.data.dof.use_dof = True
    cam.data.dof.focus_object = hero
    cam.data.dof.aperture_fstop = 1.8
    cam.constraints.clear()
    cam.animation_data_clear()
    track = cam.constraints.new("TRACK_TO")
    track.target = hero
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    trail = []
    for f in range(0, FRAMES + 1):
        p = hero_position(f)
        trail.append(p)
        back = hero_position(max(f - 18, 0))
        d = (p - back)
        d = d.normalized() if d.length > 1e-4 else Vector((0, 1, 0))
        dist = 3.0 - 1.6 * arrival(f)
        cam.location = p - d * dist + Vector((0.5, 0, 0.3))
        cam.keyframe_insert("location", frame=f)
    # The hero's own eye: an equirectangular camera riding on its head,
    # used only for the fly-view renders that feed the brain simulation.
    pov = bpy.data.objects.new("FlyPOV", bpy.data.cameras.new("FlyPOV"))
    pov.data.type = "PANO"
    try:
        pov.data.panorama_type = "EQUIRECTANGULAR"
    except (AttributeError, TypeError):
        pov.data.cycles.panorama_type = "EQUIRECTANGULAR"
    pov.parent = hero
    pov.location = HERO_EYE
    # Blender cameras look down -Z; rotate so the camera looks along body +Y.
    pov.rotation_euler = Euler((math.radians(90), 0, 0))
    sc.collection.objects.link(pov)
    return cam, pov


def main():
    sc = bpy.context.scene
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 0, FRAMES
    for ob in [o for o in bpy.data.objects if o.name.startswith(("Fly_", "Hero", "FlyPOV"))]:
        bpy.data.objects.remove(ob, do_unlink=True)
    rng = random.Random(SEED)
    mats = fly_materials()
    body, wl, wr = fly_mesh(mats), wing_mesh(mats, -1), wing_mesh(mats, 1)
    hero, wings = make_fly("Hero", mats, body, wl, wr, HERO_SCALE)
    animate(hero, wings, hero_position, rng, hero_orientation)
    for i in range(SWARM):
        f, w = make_fly(f"Fly_{i:02d}", mats, body, wl, wr, FLY_SCALE * rng.uniform(0.8, 1.2))
        seed = rng.uniform(0, 100)
        centre = Vector((rng.uniform(-10, 10), rng.uniform(-6, 14), 0))
        speed = rng.uniform(2.0, 4.0)
        animate(f, w, lambda fr, s=seed, c=centre, k=speed:
                wander(fr / FRAMES * k, s, c, 5.0, (0.5, 3.5)), rng)
    cam, pov = cameras(hero)
    sc.frame_set(0)
    print("flies:", SWARM + 1, "frames:", FRAMES)


main()
