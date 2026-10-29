"""Build the low-poly Eden in Blender. Run inside Blender:
    exec(open(r"<repo>/blender/build_scene.py").read())

Every facet is coloured by the extracted heaven panorama, sampled in the
direction of that facet's centre as seen from the fly's eye (the FlyEye empty).
From the fly's vantage the world reproduces its brain's ideal image exactly.
"""
import math
import random
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector, noise

REPO = Path(r"C:\Users\Eric\orca\workspaces\sixteen\seventeen")
PANORAMA = REPO / "out" / "heaven_current.png"
EYE = Vector((0.0, 0.0, 1.2))
SEED = 7
CONTRAST = 2.6
TONE_LPI = 140        # screentone dots per screen height
INK = (0.03, 0.02, 0.06, 1.0)
OUTLINE = 0.035       # ink line thickness, metres
SATURATION = 2.2


def clear():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.node_groups, bpy.data.cameras,
                 bpy.data.lights, bpy.data.images, bpy.data.worlds):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


# --- geometry ----------------------------------------------------------------

def height(x, y):
    p = Vector((x, y, 0)) * 0.06
    return 2.2 * noise.noise(p) + 0.5 * noise.noise(p * 3.1) - 0.2


def jitter(bm, amount, rng):
    for v in bm.verts:
        v.co += Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))) * amount


def add(bm_all, bm, matrix):
    bm.transform(matrix)
    me = bpy.data.meshes.new("tmp")
    bm.to_mesh(me)
    bm.free()
    bm_all.from_mesh(me)
    bpy.data.meshes.remove(me)


def ground(bm_all, rng):
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=48, y_segments=48, size=40)
    for v in bm.verts:
        v.co.x += rng.uniform(-0.3, 0.3)
        v.co.y += rng.uniform(-0.3, 0.3)
        v.co.z = height(v.co.x, v.co.y)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    add(bm_all, bm, Matrix.Identity(4))


def tree(bm_all, rng, x, y):
    z = height(x, y) - 0.2
    h = rng.uniform(4, 8)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, segments=5, radius1=0.35, radius2=0.18, depth=h, cap_ends=True)
    bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, h / 2))
    add(bm_all, bm, Matrix.Translation((x, y, z)))
    for _ in range(rng.randint(2, 4)):
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=rng.uniform(1.6, 2.8))
        jitter(bm, 0.35, rng)
        s = (1.0, 1.0, rng.uniform(0.6, 0.9), 1.0)
        off = Vector((rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2), h + rng.uniform(-0.5, 1.5)))
        add(bm_all, bm, Matrix.Translation(Vector((x, y, z)) + off) @ Matrix.Diagonal(s))


def fruit(bm_all, rng, x, y, r):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=r)
    jitter(bm, r * 0.12, rng)
    # Overripe: slumped on the ground.
    add(bm_all, bm, Matrix.Translation((x, y, height(x, y) + r * 0.6))
        @ Matrix.Diagonal((1.0, 1.0, 0.8, 1.0)))


def flower(bm_all, rng, x, y):
    z = height(x, y)
    h = rng.uniform(0.6, 1.6)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, segments=3, radius1=0.03, radius2=0.02, depth=h, cap_ends=True)
    bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, h / 2))
    add(bm_all, bm, Matrix.Translation((x, y, z)))
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, segments=rng.choice([5, 6]), radius1=rng.uniform(0.18, 0.35),
                          radius2=0.0, depth=0.12, cap_ends=True)
    add(bm_all, bm, Matrix.Translation((x, y, z + h)) @ Matrix.Rotation(math.pi, 4, "X"))


def grass(bm_all, rng, x, y):
    z = height(x, y)
    h = rng.uniform(0.3, 1.1)
    w = 0.06
    lean = Vector((rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 0))
    bm = bmesh.new()
    a = bm.verts.new((-w, 0, 0))
    b = bm.verts.new((w, 0, 0))
    c = bm.verts.new(Vector((0, 0, h)) + lean)
    bm.faces.new((a, b, c))
    add(bm_all, bm, Matrix.Translation((x, y, z)) @ Matrix.Rotation(rng.uniform(0, math.tau), 4, "Z"))


def build_eden():
    rng = random.Random(SEED)
    bm_all = bmesh.new()
    ground(bm_all, rng)
    for _ in range(16):
        a, d = rng.uniform(0, math.tau), rng.uniform(7, 30)
        tree(bm_all, rng, d * math.cos(a), d * math.sin(a))
    for _ in range(45):
        a, d = rng.uniform(0, math.tau), rng.uniform(1.5, 14)
        fruit(bm_all, rng, d * math.cos(a), d * math.sin(a), rng.uniform(0.15, 0.5))
    for _ in range(70):
        a, d = rng.uniform(0, math.tau), rng.uniform(1, 16)
        flower(bm_all, rng, d * math.cos(a), d * math.sin(a))
    for _ in range(500):
        a, d = rng.uniform(0, math.tau), rng.uniform(0.6, 9)
        grass(bm_all, rng, d * math.cos(a), d * math.sin(a))
    me = bpy.data.meshes.new("Eden")
    bm_all.to_mesh(me)
    bm_all.free()
    ob = bpy.data.objects.new("Eden", me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


# --- facet centres (geometry nodes) -------------------------------------------

def facet_modifier(ob):
    ng = bpy.data.node_groups.new("FacetCentre", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    n, l = ng.nodes, ng.links
    gi, go = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
    pos = n.new("GeometryNodeInputPosition")
    store = n.new("GeometryNodeStoreNamedAttribute")
    store.data_type = "FLOAT_VECTOR"
    store.domain = "FACE"
    store.inputs["Name"].default_value = "facet"
    l.new(gi.outputs[0], store.inputs["Geometry"])
    # A face-domain read of a point field averages the face's corners: its centre.
    l.new(pos.outputs[0], store.inputs["Value"])
    l.new(store.outputs[0], go.inputs[0])
    ob.modifiers.new("FacetCentre", "NODES").node_group = ng


# --- heaven projection shader -------------------------------------------------

def projection_group():
    """Direction -> equirect UV matching the extraction's panorama:
    az 0 = fly's forward (+Y), positive toward its right (+X); el up = +Z."""
    ng = bpy.data.node_groups.new("HeavenUV", "ShaderNodeTree")
    ng.interface.new_socket("Direction", in_out="INPUT", socket_type="NodeSocketVector")
    ng.interface.new_socket("UV", in_out="OUTPUT", socket_type="NodeSocketVector")
    n, l = ng.nodes, ng.links
    gi, go = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
    norm = n.new("ShaderNodeVectorMath"); norm.operation = "NORMALIZE"
    sep = n.new("ShaderNodeSeparateXYZ")
    az = n.new("ShaderNodeMath"); az.operation = "ARCTAN2"
    el = n.new("ShaderNodeMath"); el.operation = "ARCSINE"
    u = n.new("ShaderNodeMath"); u.operation = "MULTIPLY_ADD"
    u.inputs[1].default_value = 1 / math.tau; u.inputs[2].default_value = 0.5
    v = n.new("ShaderNodeMath"); v.operation = "MULTIPLY_ADD"
    v.inputs[1].default_value = 1 / math.pi; v.inputs[2].default_value = 0.5
    comb = n.new("ShaderNodeCombineXYZ")
    l.new(gi.outputs[0], norm.inputs[0]); l.new(norm.outputs[0], sep.inputs[0])
    l.new(sep.outputs["X"], az.inputs[0]); l.new(sep.outputs["Y"], az.inputs[1])
    l.new(sep.outputs["Z"], el.inputs[0])
    l.new(az.outputs[0], u.inputs[0]); l.new(el.outputs[0], v.inputs[0])
    l.new(u.outputs[0], comb.inputs["X"]); l.new(v.outputs[0], comb.inputs["Y"])
    l.new(comb.outputs[0], go.inputs[0])
    return ng


def scene_value(n, prop):
    """A Value node driven by a scene custom property, so one switch drives
    every material and the world."""
    v = n.new("ShaderNodeValue")
    v.label = prop
    fc = v.outputs[0].driver_add("default_value")
    var = fc.driver.variables.new()
    var.targets[0].id_type = "SCENE"
    var.targets[0].id = bpy.context.scene
    var.targets[0].data_path = f'["{prop}"]'
    fc.driver.expression = var.name
    return v


def math_node(n, op, a=None, b=None):
    m = n.new("ShaderNodeMath"); m.operation = op
    if a is not None:
        m.inputs[0].default_value = a
    if b is not None:
        m.inputs[1].default_value = b
    return m


def ramp(n, stops, interp="CONSTANT"):
    r = n.new("ShaderNodeValToRGB")
    r.color_ramp.interpolation = interp
    els = r.color_ramp.elements
    while len(els) < len(stops):
        els.new(0.5)
    for el, (pos, col) in zip(els, stops):
        el.position = pos
        el.color = col if len(col) == 4 else (*col, 1)
    return r


def heaven_material(img, activity, uv_group, eye):
    """Anime/manga cel shader. Colour comes from heaven (sampled from the fly's
    eye); facet edges glow with the activity of the fly's neurons that look in
    that direction. Scene property fly_view=1 swaps everything for the raw,
    unlit panorama values: what the brain is actually shown."""
    m = bpy.data.materials.new("HeavenFacet")
    m.use_nodes = True
    n, l = m.node_tree.nodes, m.node_tree.links
    n.clear()
    out = n.new("ShaderNodeOutputMaterial")

    # Direction from the fly's eye to this facet -> panorama UV.
    attr = n.new("ShaderNodeAttribute"); attr.attribute_name = "facet"
    eye_pos = n.new("ShaderNodeCombineXYZ")
    for ax, val in zip("XYZ", eye):
        eye_pos.inputs[ax].default_value = val
    sub = n.new("ShaderNodeVectorMath"); sub.operation = "SUBTRACT"
    uv = n.new("ShaderNodeGroup"); uv.node_tree = uv_group
    l.new(attr.outputs["Vector"], sub.inputs[0]); l.new(eye_pos.outputs[0], sub.inputs[1])
    l.new(sub.outputs[0], uv.inputs[0])
    tex = n.new("ShaderNodeTexImage"); tex.image = img; tex.extension = "EXTEND"
    act = n.new("ShaderNodeTexImage"); act.image = activity; act.extension = "EXTEND"
    act.name = "Activity"
    l.new(uv.outputs[0], tex.inputs["Vector"])
    # Activity is retinotopic: sample it from the hero's moving eye, in its body
    # frame. The eye camera looks down its -Z with +Y up; body frame is
    # forward +Y, up +Z, so body = (x, -z, y). wire_brain.py sets the object.
    act_frame = n.new("ShaderNodeTexCoord"); act_frame.name = "ActivityFrame"
    cam_xyz = n.new("ShaderNodeSeparateXYZ")
    flip = math_node(n, "MULTIPLY", None, -1.0)
    body = n.new("ShaderNodeCombineXYZ")
    act_uv = n.new("ShaderNodeGroup"); act_uv.node_tree = uv_group
    l.new(act_frame.outputs["Object"], cam_xyz.inputs[0])
    l.new(cam_xyz.outputs["X"], body.inputs["X"]); l.new(cam_xyz.outputs["Z"], flip.inputs[0])
    l.new(flip.outputs[0], body.inputs["Y"]); l.new(cam_xyz.outputs["Y"], body.inputs["Z"])
    l.new(body.outputs[0], act_uv.inputs[0]); l.new(act_uv.outputs[0], act.inputs["Vector"])
    grade = n.new("ShaderNodeGroup"); grade.node_tree = grade_group()
    l.new(tex.outputs["Color"], grade.inputs[0])

    # Cel bands: quantise diffuse lighting into three tones.
    diff = n.new("ShaderNodeBsdfDiffuse")
    s2rgb = n.new("ShaderNodeShaderToRGB")
    bands = ramp(n, [(0.0, (0.52,) * 3), (0.12, (0.78,) * 3), (0.45, (1.0,) * 3)])
    l.new(diff.outputs[0], s2rgb.inputs[0]); l.new(s2rgb.outputs["Color"], bands.inputs["Fac"])

    # Manga screentone: screen-space dots whose radius grows with shadow.
    tc = n.new("ShaderNodeTexCoord")
    aspect = n.new("ShaderNodeVectorMath"); aspect.operation = "MULTIPLY"
    aspect.inputs[1].default_value = (TONE_LPI * 16 / 9, TONE_LPI, 0)
    frac = n.new("ShaderNodeVectorMath"); frac.operation = "FRACTION"
    centre = n.new("ShaderNodeVectorMath"); centre.operation = "SUBTRACT"
    centre.inputs[1].default_value = (0.5, 0.5, 0)
    dist = n.new("ShaderNodeVectorMath"); dist.operation = "LENGTH"
    l.new(tc.outputs["Window"], aspect.inputs[0]); l.new(aspect.outputs[0], frac.inputs[0])
    l.new(frac.outputs[0], centre.inputs[0]); l.new(centre.outputs[0], dist.inputs[0])
    shade = n.new("ShaderNodeRGBToBW"); l.new(bands.outputs["Color"], shade.inputs[0])
    dark = math_node(n, "SUBTRACT", 1.0); l.new(shade.outputs[0], dark.inputs[1])
    # Only the shadow band gets tone; lit and mid bands stay clean.
    radius = math_node(n, "MULTIPLY_ADD", None, 2.0); radius.inputs[2].default_value = -0.6
    radius.use_clamp = True
    l.new(dark.outputs[0], radius.inputs[0])
    dots = math_node(n, "LESS_THAN"); l.new(dist.outputs["Value"], dots.inputs[0])
    l.new(radius.outputs[0], dots.inputs[1])
    tone = math_node(n, "MULTIPLY_ADD", None, -0.3); tone.inputs[2].default_value = 1.0
    l.new(dots.outputs[0], tone.inputs[0])

    # Anime rim light on surfaces turning away from the camera.
    lw = n.new("ShaderNodeLayerWeight"); lw.inputs["Blend"].default_value = 0.35
    rim = ramp(n, [(0.0, (0, 0, 0)), (0.82, (0.35, 0.35, 0.35))])
    l.new(lw.outputs["Facing"], rim.inputs["Fac"])

    toon = n.new("ShaderNodeMix"); toon.data_type = "RGBA"; toon.blend_type = "MULTIPLY"
    toon.inputs["Factor"].default_value = 1.0
    l.new(grade.outputs[0], toon.inputs["A"]); l.new(bands.outputs["Color"], toon.inputs["B"])
    toned = n.new("ShaderNodeVectorMath"); toned.operation = "SCALE"
    l.new(toon.outputs["Result"], toned.inputs[0]); l.new(tone.outputs[0], toned.inputs["Scale"])
    lit = n.new("ShaderNodeVectorMath"); lit.operation = "ADD"
    l.new(toned.outputs[0], lit.inputs[0]); l.new(rim.outputs["Color"], lit.inputs[1])
    body = n.new("ShaderNodeEmission")
    l.new(lit.outputs[0], body.inputs["Color"])

    # Neural glow along facet edges: colour and strength from activity.
    wire = n.new("ShaderNodeWireframe"); wire.use_pixel_size = True
    wire.inputs["Size"].default_value = 2.0
    act_bw = n.new("ShaderNodeRGBToBW"); l.new(act.outputs["Color"], act_bw.inputs[0])
    palette = ramp(n, [(0.0, (0.0, 0.0, 0.0)), (0.25, (0.05, 0.2, 1.0)),
                       (0.6, (0.2, 1.0, 0.9)), (1.0, (1.0, 1.0, 1.0))], interp="LINEAR")
    l.new(act_bw.outputs[0], palette.inputs["Fac"])
    glow = n.new("ShaderNodeEmission")
    gain = scene_value(n, "glow")
    # Brightness tracks activity (squared, so quiet columns go dark).
    act_sq = math_node(n, "POWER", None, 2.0); l.new(act_bw.outputs[0], act_sq.inputs[0])
    lit_wire = math_node(n, "MULTIPLY"); l.new(wire.outputs[0], lit_wire.inputs[0])
    l.new(act_sq.outputs[0], lit_wire.inputs[1])
    strength = math_node(n, "MULTIPLY"); l.new(lit_wire.outputs[0], strength.inputs[0])
    l.new(gain.outputs[0], strength.inputs[1])
    l.new(palette.outputs["Color"], glow.inputs["Color"]); l.new(strength.outputs[0], glow.inputs["Strength"])
    anime = n.new("ShaderNodeAddShader")
    l.new(body.outputs[0], anime.inputs[0]); l.new(glow.outputs[0], anime.inputs[1])

    # Fly view: raw panorama values, unlit.
    raw = n.new("ShaderNodeGamma"); raw.inputs["Gamma"].default_value = 2.2
    l.new(tex.outputs["Color"], raw.inputs["Color"])
    raw_emit = n.new("ShaderNodeEmission"); l.new(raw.outputs[0], raw_emit.inputs["Color"])
    switch = n.new("ShaderNodeMixShader")
    l.new(scene_value(n, "fly_view").outputs[0], switch.inputs["Fac"])
    l.new(anime.outputs[0], switch.inputs[1]); l.new(raw_emit.outputs[0], switch.inputs[2])
    l.new(switch.outputs[0], out.inputs["Surface"])
    return m


def outline_material():
    """Inverted-hull ink line: back faces of a slightly inflated shell."""
    m = bpy.data.materials.new("Ink")
    m.use_nodes = True
    m.use_backface_culling = True
    n = m.node_tree.nodes
    n.clear()
    e = n.new("ShaderNodeEmission"); e.inputs["Color"].default_value = INK
    o = n.new("ShaderNodeOutputMaterial")
    m.node_tree.links.new(e.outputs[0], o.inputs["Surface"])
    return m


def add_outline(ob):
    ob.data.materials.append(outline_material())
    s = ob.modifiers.new("Ink", "SOLIDIFY")
    s.thickness = OUTLINE
    s.offset = 1.0
    s.use_flip_normals = True
    s.use_rim = False
    s.material_offset = 1


def grade_group():
    """The brain responds to deviations from grey, which in the raw panorama are
    subtle. Expand contrast and saturation around grey so humans can see them.
    Shared by surfaces and sky; the styling pass tunes these two numbers."""
    if "HeavenGrade" in bpy.data.node_groups:
        return bpy.data.node_groups["HeavenGrade"]
    ng = bpy.data.node_groups.new("HeavenGrade", "ShaderNodeTree")
    ng.interface.new_socket("Color", in_out="INPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket("Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    n, l = ng.nodes, ng.links
    gi, go = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
    centre = n.new("ShaderNodeVectorMath"); centre.operation = "MULTIPLY_ADD"
    centre.name = "Contrast"
    centre.inputs[1].default_value = (CONTRAST,) * 3
    centre.inputs[2].default_value = (0.5 * (1 - CONTRAST),) * 3
    hsv = n.new("ShaderNodeHueSaturation"); hsv.name = "Saturation"
    hsv.inputs["Saturation"].default_value = SATURATION
    # The panorama is loaded as raw data (the values the brain saw); convert the
    # graded result from display-encoded to linear light for rendering.
    to_linear = n.new("ShaderNodeGamma"); to_linear.inputs["Gamma"].default_value = 2.2
    l.new(gi.outputs[0], centre.inputs[0]); l.new(centre.outputs[0], hsv.inputs["Color"])
    l.new(hsv.outputs["Color"], to_linear.inputs["Color"])
    l.new(to_linear.outputs["Color"], go.inputs[0])
    return ng


def heaven_world(img, uv_group):
    w = bpy.data.worlds.new("Heaven")
    w.use_nodes = True
    n, l = w.node_tree.nodes, w.node_tree.links
    n.clear()
    out = n.new("ShaderNodeOutputWorld")
    tc = n.new("ShaderNodeTexCoord")
    grp = n.new("ShaderNodeGroup"); grp.node_tree = uv_group
    tex = n.new("ShaderNodeTexImage"); tex.image = img; tex.extension = "EXTEND"
    grade = n.new("ShaderNodeGroup"); grade.node_tree = grade_group()
    raw = n.new("ShaderNodeGamma"); raw.inputs["Gamma"].default_value = 2.2
    pick = n.new("ShaderNodeMix"); pick.data_type = "RGBA"
    bg = n.new("ShaderNodeBackground")
    l.new(tc.outputs["Generated"], grp.inputs[0]); l.new(grp.outputs[0], tex.inputs["Vector"])
    l.new(tex.outputs["Color"], grade.inputs[0]); l.new(tex.outputs["Color"], raw.inputs["Color"])
    l.new(scene_value(n, "fly_view").outputs[0], pick.inputs["Factor"])
    l.new(grade.outputs[0], pick.inputs["A"]); l.new(raw.outputs[0], pick.inputs["B"])
    l.new(pick.outputs["Result"], bg.inputs["Color"])
    l.new(bg.outputs[0], out.inputs["Surface"])
    bpy.context.scene.world = w


def setup_view():
    sc = bpy.context.scene
    eye = bpy.data.objects.new("FlyEye", None)
    eye.location = EYE
    eye.empty_display_type = "SPHERE"
    eye.empty_display_size = 0.05
    sc.collection.objects.link(eye)
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 2.0
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))
    sc.collection.objects.link(sun)
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    cam.data.lens = 18
    cam.location = (0, -9, 3.2)
    cam.rotation_euler = (math.radians(84), 0, 0)
    sc.collection.objects.link(cam)
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    try:
        sc.view_settings.view_transform = "Standard"
    except TypeError:
        pass


def activity_image():
    """Retinotopic neural activity on the panorama grid, 0..1. Replaced per
    frame by the brain simulation; starts dark."""
    img = bpy.data.images.new("activity", 128, 64, float_buffer=True)
    img.colorspace_settings.is_data = True
    img.pixels[:] = [0.0, 0.0, 0.0, 1.0] * (128 * 64)
    return img


def bloom():
    """Compositor glare so the neural edges bleed light."""
    sc = bpy.context.scene
    try:
        ng = bpy.data.node_groups.new("Bloom", "CompositorNodeTree")
        sc.compositing_node_group = ng
        ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        out = ng.nodes.new("NodeGroupOutput")
    except (AttributeError, TypeError):  # pre-5.0 compositor API
        sc.use_nodes = True
        ng = sc.node_tree
        ng.nodes.clear()
        out = ng.nodes.new("CompositorNodeComposite")
    rl = ng.nodes.new("CompositorNodeRLayers")
    glare = ng.nodes.new("CompositorNodeGlare")
    for attr, val in (("glare_type", "BLOOM"), ("quality", "HIGH")):
        try:
            setattr(glare, attr, val)
        except (AttributeError, TypeError):
            pass
    for name, val in (("Type", "Bloom"), ("Threshold", 0.9), ("Size", 0.6), ("Strength", 0.8)):
        if name in glare.inputs:
            try:
                glare.inputs[name].default_value = val
            except TypeError:
                pass
    ng.links.new(rl.outputs["Image"], glare.inputs["Image"])
    ng.links.new(glare.outputs["Image"], out.inputs[0])


def main():
    clear()
    sc = bpy.context.scene
    sc["glow"] = 4.0
    sc["fly_view"] = 0.0
    img = bpy.data.images.load(str(PANORAMA), check_existing=False)
    img.name = "heaven"
    img.colorspace_settings.is_data = True
    ob = build_eden()
    facet_modifier(ob)
    uv = projection_group()
    ob.data.materials.append(heaven_material(img, activity_image(), uv, EYE))
    add_outline(ob)
    heaven_world(img, uv)
    setup_view()
    bloom()
    print("Eden:", len(ob.data.polygons), "facets")


main()
