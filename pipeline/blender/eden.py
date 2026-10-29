"""The painterly Eden. Run inside Blender:
    exec(open(r"<repo>/blender/eden.py").read())

A soft meadow at the fly's scale: rolling ground, grass, wildflowers, trees of
brush-stroke leaf cards, overripe fruit, drifting pollen, haze and god rays.

Colour still comes only from the extracted heaven panorama: each blade, petal
and leaf takes one dab of colour sampled in the direction of its root as seen
from the fly's eye. Blade tips, petals and pollen glow with the activity of the
hero fly's visual neurons (wired by wire_brain.py).

Scene property fly_view=1 swaps every surface for the raw panorama sampled at
the shading point: from the eye point this reproduces heaven exactly.
"""
import math
from pathlib import Path

import bpy
import numpy as np

REPO = Path(r"C:\Users\Eric\orca\workspaces\sixteen\seventeen")
PANORAMA = REPO / "out" / "heaven_current.png"
EYE = np.array([0.0, 0.0, 1.2])
SEED = 3

CONTRAST = 1.9
SATURATION = 1.5
GLOW_COLORS = [(0.0, (0, 0, 0)), (0.35, (1.0, 0.62, 0.25)), (0.75, (1.0, 0.92, 0.7)), (1.0, (1, 1, 1))]


# --- terrain ---------------------------------------------------------------------

rng = np.random.default_rng(SEED)
_WAVES = [(rng.normal(size=2) * f, rng.uniform(0, math.tau), a)
          for f, a in ((0.03, 1.6), (0.05, 0.9), (0.09, 0.45), (0.17, 0.18))
          for _ in range(3)]


def height(x, y):
    z = sum(a * np.sin(k[0] * x + k[1] * y + p) for k, p, a in _WAVES)
    z0 = sum(a * math.sin(p) for k, p, a in _WAVES)
    # Keep a soft hollow around the eye point so the fly hovers above the grass.
    return z - z0 - 0.4 * np.exp(-(np.asarray(x) ** 2 + np.asarray(y) ** 2) / 30)


def scatter(n, r_min, r_max, density_falloff=1.0):
    """Points on the ground, denser toward the centre."""
    r = r_min + (r_max - r_min) * rng.uniform(0, 1, n) ** density_falloff
    a = rng.uniform(0, math.tau, n)
    x, y = r * np.cos(a), r * np.sin(a)
    return np.c_[x, y, height(x, y)]


# --- mesh helpers ----------------------------------------------------------------

def build_mesh(name, verts, faces, attrs=None, smooth=True, material=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], faces)
    for key, val in (attrs or {}).items():
        val = np.asarray(val, np.float32)
        kind = "FLOAT_VECTOR" if val.ndim == 2 else "FLOAT"
        a = me.attributes.new(key, kind, "POINT")
        a.data.foreach_set("vector" if kind == "FLOAT_VECTOR" else "value", val.ravel())
    if smooth:
        me.shade_smooth()
    if material:
        me.materials.append(material)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def blades(roots, heights, widths, lean, facing, segments=4):
    """Curved, tapering blades. Returns verts, faces, per-vertex root and tip."""
    n = len(roots)
    t = np.linspace(0, 1, segments + 1)[:-1]                    # rows below the tip
    side = np.c_[np.cos(facing), np.sin(facing), np.zeros(n)]
    rows = []
    for ti in t:
        centre = roots + np.c_[np.zeros(n), np.zeros(n), heights * ti] + lean * (heights * ti ** 2)[:, None]
        half = (widths * (1 - ti) ** 0.7)[:, None] * side * 0.5
        rows += [centre - half, centre + half]
    tip = roots + np.c_[np.zeros(n), np.zeros(n), heights] + lean * heights[:, None]
    per = 2 * segments + 1
    V = np.stack(rows + [tip], 1).reshape(-1, 3)
    base = (np.arange(n) * per)[:, None]
    faces = []
    for s in range(segments - 1):
        a, b = 2 * s, 2 * s + 1
        faces.append(np.c_[base + a, base + b, base + b + 2, base + a + 2])
    last = 2 * (segments - 1)
    tri = np.c_[base + last, base + last + 1, base + per - 1]
    F = [list(f) for blk in faces for f in blk] + [list(f) for f in tri]
    root = np.repeat(roots, per, 0)
    tipf = np.tile(np.r_[np.repeat(t, 2), 1.0], n)
    return V, F, root, tipf


# --- shaders ---------------------------------------------------------------------

def node(n, kind, **inputs):
    x = n.new(kind)
    for k, v in inputs.items():
        x.inputs[k].default_value = v
    return x


def scene_value(n, prop):
    v = n.new("ShaderNodeValue")
    v.label = prop
    fc = v.outputs[0].driver_add("default_value")
    var = fc.driver.variables.new()
    var.targets[0].id_type = "SCENE"
    var.targets[0].id = bpy.context.scene
    var.targets[0].data_path = f'["{prop}"]'
    fc.driver.expression = var.name
    return v


def math_node(n, op, a=None, b=None, clamp=False):
    m = n.new("ShaderNodeMath"); m.operation = op; m.use_clamp = clamp
    if a is not None:
        m.inputs[0].default_value = a
    if b is not None:
        m.inputs[1].default_value = b
    return m


def ramp(n, stops, interp="LINEAR"):
    r = n.new("ShaderNodeValToRGB")
    r.color_ramp.interpolation = interp
    els = r.color_ramp.elements
    while len(els) < len(stops):
        els.new(0.5)
    for el, (pos, col) in zip(els, stops):
        el.position = pos
        el.color = (*col, 1)
    return r


def uv_group():
    """Direction -> equirect UV matching the extraction's panorama:
    az 0 = fly's forward (+Y), positive toward its right (+X); el up = +Z."""
    if "HeavenUV" in bpy.data.node_groups:
        return bpy.data.node_groups["HeavenUV"]
    ng = bpy.data.node_groups.new("HeavenUV", "ShaderNodeTree")
    ng.interface.new_socket("Direction", in_out="INPUT", socket_type="NodeSocketVector")
    ng.interface.new_socket("UV", in_out="OUTPUT", socket_type="NodeSocketVector")
    n, l = ng.nodes, ng.links
    gi, go = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
    norm = n.new("ShaderNodeVectorMath"); norm.operation = "NORMALIZE"
    sep = n.new("ShaderNodeSeparateXYZ")
    az = math_node(n, "ARCTAN2"); el = math_node(n, "ARCSINE")
    u = math_node(n, "MULTIPLY_ADD", None, 1 / math.tau); u.inputs[2].default_value = 0.5
    v = math_node(n, "MULTIPLY_ADD", None, 1 / math.pi); v.inputs[2].default_value = 0.5
    comb = n.new("ShaderNodeCombineXYZ")
    l.new(gi.outputs[0], norm.inputs[0]); l.new(norm.outputs[0], sep.inputs[0])
    l.new(sep.outputs["X"], az.inputs[0]); l.new(sep.outputs["Y"], az.inputs[1])
    l.new(sep.outputs["Z"], el.inputs[0])
    l.new(az.outputs[0], u.inputs[0]); l.new(el.outputs[0], v.inputs[0])
    l.new(u.outputs[0], comb.inputs["X"]); l.new(v.outputs[0], comb.inputs["Y"])
    l.new(comb.outputs[0], go.inputs[0])
    return ng


def grade_group():
    """Expand the panorama's subtle deviations from grey so humans can see
    them, then convert display-encoded values to linear light."""
    if "HeavenGrade" in bpy.data.node_groups:
        return bpy.data.node_groups["HeavenGrade"]
    ng = bpy.data.node_groups.new("HeavenGrade", "ShaderNodeTree")
    ng.interface.new_socket("Color", in_out="INPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket("Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    n, l = ng.nodes, ng.links
    gi, go = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
    c = n.new("ShaderNodeVectorMath"); c.operation = "MULTIPLY_ADD"; c.name = "Contrast"
    c.inputs[1].default_value = (CONTRAST,) * 3
    c.inputs[2].default_value = (0.5 * (1 - CONTRAST),) * 3
    hsv = n.new("ShaderNodeHueSaturation"); hsv.name = "Saturation"
    hsv.inputs["Saturation"].default_value = SATURATION
    lin = node(n, "ShaderNodeGamma", Gamma=2.2)
    l.new(gi.outputs[0], c.inputs[0]); l.new(c.outputs[0], hsv.inputs["Color"])
    l.new(hsv.outputs["Color"], lin.inputs["Color"]); l.new(lin.outputs[0], go.inputs[0])
    return ng


class Heaven:
    """Shared images and helpers for building heaven-coloured materials."""

    def __init__(self):
        self.img = bpy.data.images.load(str(PANORAMA), check_existing=False)
        self.img.name = "heaven"
        self.img.colorspace_settings.is_data = True
        self.act = bpy.data.images.new("activity", 128, 64, float_buffer=True)
        self.act.colorspace_settings.is_data = True
        self.act.pixels[:] = [0.0, 0.0, 0.0, 1.0] * (128 * 64)

    def sample(self, n, l, point):
        """Panorama colour (raw) in the direction from the eye to `point`."""
        eye = node(n, "ShaderNodeCombineXYZ", X=EYE[0], Y=EYE[1], Z=EYE[2])
        d = n.new("ShaderNodeVectorMath"); d.operation = "SUBTRACT"
        uv = n.new("ShaderNodeGroup"); uv.node_tree = uv_group()
        tex = n.new("ShaderNodeTexImage"); tex.image = self.img; tex.extension = "EXTEND"
        l.new(point, d.inputs[0]); l.new(eye.outputs[0], d.inputs[1])
        l.new(d.outputs[0], uv.inputs[0]); l.new(uv.outputs[0], tex.inputs["Vector"])
        return tex.outputs["Color"]

    def activity(self, n, l):
        """Retinotopic activity (0..1) sampled from the hero's moving eye in its
        body frame. The eye camera looks down -Z with +Y up; body = (x, -z, y).
        wire_brain.py points the ActivityFrame at the FlyPOV camera and swaps
        in the per-frame image sequence."""
        tc = n.new("ShaderNodeTexCoord"); tc.name = "ActivityFrame"
        sep = n.new("ShaderNodeSeparateXYZ")
        flip = math_node(n, "MULTIPLY", None, -1.0)
        body = n.new("ShaderNodeCombineXYZ")
        uv = n.new("ShaderNodeGroup"); uv.node_tree = uv_group()
        tex = n.new("ShaderNodeTexImage"); tex.image = self.act; tex.extension = "EXTEND"
        tex.name = "Activity"
        l.new(tc.outputs["Object"], sep.inputs[0])
        l.new(sep.outputs["X"], body.inputs["X"]); l.new(sep.outputs["Z"], flip.inputs[0])
        l.new(flip.outputs[0], body.inputs["Y"]); l.new(sep.outputs["Y"], body.inputs["Z"])
        l.new(body.outputs[0], uv.inputs[0]); l.new(uv.outputs[0], tex.inputs["Vector"])
        bw = n.new("ShaderNodeRGBToBW"); l.new(tex.outputs["Color"], bw.inputs[0])
        return bw.outputs[0]

    def material(self, name, kind):
        """kind: ground, grass, flower, leaf, fruit, mote."""
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        n, l = m.node_tree.nodes, m.node_tree.links
        n.clear()
        out = n.new("ShaderNodeOutputMaterial")
        geo = n.new("ShaderNodeNewGeometry")
        attr_root = n.new("ShaderNodeAttribute"); attr_root.attribute_name = "root"
        attr_tip = n.new("ShaderNodeAttribute"); attr_tip.attribute_name = "tip"
        # One colour dab per blade/petal/leaf (sampled at its root); the ground
        # and fruit sample smoothly at the shading point.
        point = geo.outputs["Position"] if kind in ("ground", "fruit") else attr_root.outputs["Vector"]
        grade = n.new("ShaderNodeGroup"); grade.node_tree = grade_group()
        l.new(self.sample(n, l, point), grade.inputs[0])
        colour = grade.outputs[0]

        # Painterly variation: low-frequency brush noise nudges value per dab.
        brush = node(n, "ShaderNodeTexNoise", Scale=1.6, Detail=2.0, Roughness=0.6)
        l.new(point, brush.inputs["Vector"])
        vary = n.new("ShaderNodeMix"); vary.data_type = "RGBA"; vary.blend_type = "OVERLAY"
        vary.inputs["Factor"].default_value = 0.25
        l.new(colour, vary.inputs["A"]); l.new(brush.outputs["Fac"], vary.inputs["B"])
        colour = vary.outputs["Result"]
        if kind in ("grass", "flower", "leaf"):
            # Light catches the tips.
            lift = n.new("ShaderNodeMix"); lift.data_type = "RGBA"; lift.blend_type = "SCREEN"
            tipf = math_node(n, "MULTIPLY", None, 0.35)
            l.new(attr_tip.outputs["Fac"], tipf.inputs[0]); l.new(tipf.outputs[0], lift.inputs["Factor"])
            lift.inputs["B"].default_value = (1.0, 0.95, 0.85, 1)
            l.new(colour, lift.inputs["A"]); colour = lift.outputs["Result"]

        bsdf = n.new("ShaderNodeBsdfPrincipled")
        l.new(colour, bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = {"fruit": 0.25, "ground": 0.9}.get(kind, 0.6)
        if kind == "fruit":
            bsdf.inputs["Subsurface Weight"].default_value = 0.6
            bsdf.inputs["Coat Weight"].default_value = 0.4
        surface = bsdf.outputs[0]
        if kind in ("grass", "flower", "leaf"):
            # Backlit translucency: the ethereal glow of sunlit grass.
            tr = n.new("ShaderNodeBsdfTranslucent"); l.new(colour, tr.inputs["Color"])
            mix = n.new("ShaderNodeMixShader"); mix.inputs["Fac"].default_value = 0.45
            l.new(surface, mix.inputs[1]); l.new(tr.outputs[0], mix.inputs[2])
            surface = mix.outputs[0]

        # Neural bioluminescence: tips, petals and pollen glow with activity.
        act = self.activity(n, l)
        sq = math_node(n, "POWER", None, 2.0); l.new(act, sq.inputs[0])
        where = {"mote": 1.0, "flower": 1.0}.get(kind)
        tipw = math_node(n, "POWER", None, 3.0)
        l.new(attr_tip.outputs["Fac"], tipw.inputs[0])
        amount = math_node(n, "MULTIPLY"); l.new(sq.outputs[0], amount.inputs[0])
        if where is None:
            l.new(tipw.outputs[0], amount.inputs[1])
        else:
            amount.inputs[1].default_value = where
        gain = scene_value(n, "glow")
        strength = math_node(n, "MULTIPLY"); l.new(amount.outputs[0], strength.inputs[0])
        l.new(gain.outputs[0], strength.inputs[1])
        if kind == "mote":
            base = math_node(n, "ADD", None, 0.6); l.new(strength.outputs[0], base.inputs[0])
            strength = base
        if kind in ("ground", "fruit"):
            strength = math_node(n, "MULTIPLY", 0.0, 0.0)   # no glow on ground or fruit
        pal = ramp(n, GLOW_COLORS); l.new(act, pal.inputs["Fac"])
        glow = n.new("ShaderNodeEmission")
        l.new(pal.outputs["Color"], glow.inputs["Color"]); l.new(strength.outputs[0], glow.inputs["Strength"])
        lit = n.new("ShaderNodeAddShader")
        l.new(surface, lit.inputs[0]); l.new(glow.outputs[0], lit.inputs[1])
        shader = lit.outputs[0]

        if kind == "leaf":
            # Brush-stroke leaf card: a noise-ragged ellipse cut from the quad.
            uvn = n.new("ShaderNodeAttribute"); uvn.attribute_name = "card"
            ctr = n.new("ShaderNodeVectorMath"); ctr.operation = "SUBTRACT"
            ctr.inputs[1].default_value = (0.5, 0.5, 0)
            sc = n.new("ShaderNodeVectorMath"); sc.operation = "MULTIPLY"
            sc.inputs[1].default_value = (1.0, 1.9, 1)
            ln = n.new("ShaderNodeVectorMath"); ln.operation = "LENGTH"
            rag = node(n, "ShaderNodeTexNoise", Scale=7.0, Detail=4.0)
            l.new(uvn.outputs["Vector"], rag.inputs["Vector"])
            edge = math_node(n, "MULTIPLY_ADD", None, 0.25); edge.inputs[2].default_value = 0.32
            l.new(rag.outputs["Fac"], edge.inputs[0])
            alpha = math_node(n, "LESS_THAN")
            l.new(uvn.outputs["Vector"], ctr.inputs[0]); l.new(ctr.outputs[0], sc.inputs[0])
            l.new(sc.outputs[0], ln.inputs[0]); l.new(ln.outputs["Value"], alpha.inputs[0])
            l.new(edge.outputs[0], alpha.inputs[1])
            tr = n.new("ShaderNodeBsdfTransparent")
            cut = n.new("ShaderNodeMixShader")
            l.new(alpha.outputs[0], cut.inputs["Fac"])
            l.new(tr.outputs[0], cut.inputs[1]); l.new(shader, cut.inputs[2])
            shader = cut.outputs[0]
            leaf_alpha = alpha.outputs[0]
            m.surface_render_method = "DITHERED"

        # Fly view: raw panorama at the shading point, unlit.
        raw = node(n, "ShaderNodeGamma", Gamma=2.2)
        l.new(self.sample(n, l, geo.outputs["Position"]), raw.inputs["Color"])
        raw_emit = n.new("ShaderNodeEmission"); l.new(raw.outputs[0], raw_emit.inputs["Color"])
        switch = n.new("ShaderNodeMixShader")
        l.new(scene_value(n, "fly_view").outputs[0], switch.inputs["Fac"])
        l.new(shader, switch.inputs[1])
        fly_seen = raw_emit.outputs[0]
        if kind == "leaf":
            # The fly sees the same ragged leaf shapes, not square cards.
            tr2 = n.new("ShaderNodeBsdfTransparent")
            cut2 = n.new("ShaderNodeMixShader")
            l.new(leaf_alpha, cut2.inputs["Fac"])
            l.new(tr2.outputs[0], cut2.inputs[1]); l.new(raw_emit.outputs[0], cut2.inputs[2])
            fly_seen = cut2.outputs[0]
        l.new(fly_seen, switch.inputs[2])
        l.new(switch.outputs[0], out.inputs["Surface"])
        return m

    def world(self):
        w = bpy.data.worlds.new("Heaven")
        w.use_nodes = True
        n, l = w.node_tree.nodes, w.node_tree.links
        n.clear()
        out = n.new("ShaderNodeOutputWorld")
        tc = n.new("ShaderNodeTexCoord")
        uv = n.new("ShaderNodeGroup"); uv.node_tree = uv_group()
        tex = n.new("ShaderNodeTexImage"); tex.image = self.img; tex.extension = "EXTEND"
        l.new(tc.outputs["Generated"], uv.inputs[0]); l.new(uv.outputs[0], tex.inputs["Vector"])
        grade = n.new("ShaderNodeGroup"); grade.node_tree = grade_group()
        raw = node(n, "ShaderNodeGamma", Gamma=2.2)
        l.new(tex.outputs["Color"], grade.inputs[0]); l.new(tex.outputs["Color"], raw.inputs["Color"])
        # Soften the sky toward luminous haze.
        haze = n.new("ShaderNodeMix"); haze.data_type = "RGBA"; haze.blend_type = "SCREEN"
        haze.inputs["Factor"].default_value = 0.35
        haze.inputs["B"].default_value = (1.0, 0.93, 0.82, 1)
        l.new(grade.outputs[0], haze.inputs["A"])
        pick = n.new("ShaderNodeMix"); pick.data_type = "RGBA"
        l.new(scene_value(n, "fly_view").outputs[0], pick.inputs["Factor"])
        l.new(haze.outputs["Result"], pick.inputs["A"]); l.new(raw.outputs[0], pick.inputs["B"])
        bg = n.new("ShaderNodeBackground"); bg.inputs["Strength"].default_value = 1.0
        l.new(pick.outputs["Result"], bg.inputs["Color"]); l.new(bg.outputs[0], out.inputs["Surface"])
        bpy.context.scene.world = w


# --- the meadow -------------------------------------------------------------------

def ground(h):
    n = 220
    xs = np.linspace(-70, 70, n)
    X, Y = np.meshgrid(xs, xs)
    V = np.c_[X.ravel(), Y.ravel(), height(X.ravel(), Y.ravel())]
    idx = np.arange(n * n).reshape(n, n)
    F = np.c_[idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()]
    return build_mesh("Ground", V, F.tolist(), {"root": V, "tip": np.zeros(len(V))},
                      material=h.material("HeavenGround", "ground"))


def far_ground(h):
    """A ring of ground from the meadow's edge to the horizon, so the art camera
    never sees the world end. Hidden from the fly-view renders (render_flyview),
    where the panorama projection makes it irrelevant."""
    rings, segs = 24, 160
    r = np.geomspace(66, 400, rings)
    a = np.linspace(0, math.tau, segs, endpoint=False)
    R, A = np.meshgrid(r, a, indexing="ij")
    X, Y = (R * np.cos(A)).ravel(), (R * np.sin(A)).ravel()
    V = np.c_[X, Y, height(X, Y)]
    idx = np.arange(rings * segs).reshape(rings, segs)
    F = np.c_[idx[:-1].ravel(), np.roll(idx[:-1], -1, 1).ravel(),
              np.roll(idx[1:], -1, 1).ravel(), idx[1:].ravel()]
    return build_mesh("FarGround", V, F.tolist(), {"root": V, "tip": np.zeros(len(V))},
                      material=bpy.data.materials["HeavenGround"])


def grass(h, count=110000):
    roots = scatter(count, 0.4, 28, density_falloff=1.6)
    d = np.linalg.norm(roots[:, :2], axis=1)
    heights = rng.uniform(0.35, 1.0, count) * (1.1 + 0.6 * np.exp(-d / 6))
    widths = rng.uniform(0.012, 0.03, count)
    a = rng.uniform(0, math.tau, count)
    wind = np.array([0.35, 0.15, 0])            # a shared breeze direction
    lean = np.c_[np.cos(a), np.sin(a), np.zeros(count)] * rng.uniform(0.05, 0.35, count)[:, None] + wind * 0.4
    V, F, root, tip = blades(roots, heights, widths, lean, rng.uniform(0, math.tau, count))
    return build_mesh("Grass", V, F, {"root": root, "tip": tip}, material=h.material("HeavenGrass", "grass"))


def flowers(h, count=2200):
    feet = scatter(count, 0.8, 20, density_falloff=1.3)
    stem_h = rng.uniform(0.5, 1.5, count)
    lean = rng.normal(0, 0.08, (count, 3)) * [1, 1, 0]
    V, F, _, tip = blades(feet, stem_h, np.full(count, 0.018), lean, rng.uniform(0, math.tau, count), 3)
    top = feet + np.c_[np.zeros(count), np.zeros(count), stem_h] + lean * stem_h[:, None]
    per_stem = len(V) // count
    verts, faces = [V], list(F)
    roots, tips = [np.repeat(top, per_stem, 0)], [tip * 0.3]   # whole flower shares one dab
    petals = 9
    size = rng.uniform(0.035, 0.08, count)
    for k in range(petals):
        ang = k * math.tau / petals + rng.uniform(0, 0.4, count)
        out_dir = np.c_[np.cos(ang), np.sin(ang), np.full(count, 0.9)]   # cupped upward
        side = np.c_[-np.sin(ang), np.cos(ang), np.zeros(count)] * (size * 0.45)[:, None]
        mid = top + out_dir * (size * 0.6)[:, None]
        quad = np.stack([top, mid - side, top + out_dir * size[:, None], mid + side], 1).reshape(-1, 3)
        b = sum(len(x) for x in verts) + np.arange(count) * 4
        verts.append(quad)
        faces += np.c_[b, b + 1, b + 2, b + 3].tolist()
        roots.append(np.repeat(top, 4, 0))
        tips.append(np.tile([0.4, 0.9, 1.0, 0.9], count))
    return build_mesh("Flowers", np.concatenate(verts), faces,
                      {"root": np.concatenate(roots), "tip": np.concatenate(tips)},
                      material=h.material("HeavenFlower", "flower"))


def trees(h, count=11):
    tv, tf, troot = [], [], []
    lv, lf, lroot, ltip, lcard = [], [], [], [], []
    for _ in range(count):
        a, r = rng.uniform(0, math.tau), rng.uniform(11, 38)
        # Keep the fly's forward view open: fewer trees directly ahead.
        if abs(math.atan2(math.cos(a), math.sin(a))) < 0.5 and r < 20:
            r += 12
        base = np.array([r * math.cos(a), r * math.sin(a), 0.0])
        base[2] = float(height(base[0], base[1])) - 0.3
        hgt = rng.uniform(7, 13)
        bend = rng.normal(0, 0.8, 2)
        # Trunk: tapered, gently curved ring stack.
        rings, segs = 10, 9
        ts = np.linspace(0, 1, rings)
        start = sum(len(x) for x in tv)
        for t in ts:
            c = base + np.array([bend[0] * t ** 2, bend[1] * t ** 2, hgt * t])
            rad = 0.55 * (1 - 0.7 * t)
            ang = np.linspace(0, math.tau, segs, endpoint=False)
            tv.append(c + np.c_[rad * np.cos(ang), rad * np.sin(ang), np.zeros(segs)])
            troot.append(np.repeat(c[None], segs, 0))
        for i in range(rings - 1):
            for j in range(segs):
                a0 = start + i * segs + j
                a1 = start + i * segs + (j + 1) % segs
                tf.append([a0, a1, a1 + segs, a0 + segs])
        crown = base + np.array([bend[0], bend[1], hgt])
        # Foliage: clumps of brush-stroke leaf cards.
        for _ in range(rng.integers(7, 12)):
            cc = crown + rng.normal(0, [2.2, 2.2, 1.3])
            m = 260
            pos = cc + rng.normal(0, 1.0, (m, 3)) * [1.3, 1.3, 0.8]
            nrm = rng.normal(0, 1, (m, 3)); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
            u = np.cross(nrm, [0, 0, 1.0]); u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
            v = np.cross(nrm, u)
            s = rng.uniform(0.35, 0.7, m)[:, None]
            corners = [pos - u * s - v * s, pos + u * s - v * s, pos + u * s + v * s, pos - u * s + v * s]
            q = np.stack(corners, 1).reshape(-1, 3)
            b = sum(len(x) for x in lv) + np.arange(m) * 4
            lv.append(q)
            lf += np.c_[b, b + 1, b + 2, b + 3].tolist()
            lroot.append(np.repeat(pos, 4, 0))
            outward = np.linalg.norm(pos - crown, axis=1) / 4.0
            ltip.append(np.repeat(np.clip(outward, 0, 1), 4))
            lcard.append(np.tile([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], (m, 1)).astype(float))
    trunk = build_mesh("Trunks", np.concatenate(tv), tf,
                       {"root": np.concatenate(troot), "tip": np.zeros(sum(len(x) for x in tv))},
                       material=h.material("HeavenBark", "ground"))
    leaves = build_mesh("Leaves", np.concatenate(lv), lf,
                        {"root": np.concatenate(lroot), "tip": np.concatenate(ltip), "card": np.concatenate(lcard)},
                        smooth=False, material=h.material("HeavenLeaf", "leaf"))
    return trunk, leaves


def fruit(h, count=26):
    mat = h.material("HeavenFruit", "fruit")
    obs = []
    for i, p in enumerate(scatter(count, 1.8, 13)):
        r = rng.uniform(0.12, 0.35)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=14, radius=r,
                                             location=(p[0], p[1], p[2] + r * 0.55))
        ob = bpy.context.active_object
        ob.name = f"Fruit_{i:02d}"
        ob.scale = (1, 1, rng.uniform(0.7, 0.9))   # overripe, slumping
        ob.data.shade_smooth()
        ob.data.materials.append(mat)
        obs.append(ob)
    return obs


def motes(h, count=2600):
    c = rng.normal(0, 1, (count, 3)) * [7, 7, 1.4] + [0, 1.5, 1.8]
    c[:, 2] = np.maximum(c[:, 2], height(c[:, 0], c[:, 1]) + 0.2)
    t = (1 + 5 ** 0.5) / 2
    ico = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                    [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    ico /= np.linalg.norm(ico[0])
    faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
             [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5],
             [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    s = rng.uniform(0.005, 0.012, count)[:, None, None]
    V = (c[:, None, :] + ico[None] * s).reshape(-1, 3)
    F = (np.array(faces)[None] + (np.arange(count) * 12)[:, None, None]).reshape(-1, 3).tolist()
    return build_mesh("Pollen", V, F, {"root": np.repeat(c, 12, 0), "tip": np.ones(len(V))},
                      material=h.material("HeavenPollen", "mote"))


def haze():
    """Ground mist and god rays: a volume box, denser near the ground. (A world
    volume renders black in this Eevee build.) Hidden from the fly-view renders."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 10))
    box = bpy.context.active_object
    box.name = "Haze"
    box.scale = (800, 800, 26)
    m = bpy.data.materials.new("Haze")
    m.use_nodes = True
    n, l = m.node_tree.nodes, m.node_tree.links
    n.clear()
    out = n.new("ShaderNodeOutputMaterial")
    geo = n.new("ShaderNodeNewGeometry")
    z = n.new("ShaderNodeSeparateXYZ"); l.new(geo.outputs["Position"], z.inputs[0])
    fall = math_node(n, "MULTIPLY", None, -0.28); l.new(z.outputs["Z"], fall.inputs[0])
    ex = math_node(n, "EXPONENT"); l.new(fall.outputs[0], ex.inputs[0])
    dens = math_node(n, "MULTIPLY_ADD", None, 0.014); dens.inputs[2].default_value = 0.0015
    l.new(ex.outputs[0], dens.inputs[0])
    vol = node(n, "ShaderNodeVolumePrincipled", Anisotropy=0.6)
    vol.inputs["Color"].default_value = (1.0, 0.93, 0.84, 1)
    l.new(dens.outputs[0], vol.inputs["Density"])
    l.new(vol.outputs[0], out.inputs["Volume"])
    box.data.materials.append(m)
    return box


# --- light, camera, post ---------------------------------------------------------------

def light_and_render():
    sc = bpy.context.scene
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 4.5
    sun.data.color = (1.0, 0.86, 0.66)
    sun.data.angle = math.radians(4)
    # Low golden-hour sun from ahead-left: backlights the grass toward camera.
    sun.rotation_euler = (math.radians(80), 0, math.radians(165))
    sc.collection.objects.link(sun)
    eye = bpy.data.objects.new("FlyEye", None)
    eye.location = EYE.tolist()
    eye.empty_display_size = 0.05
    sc.collection.objects.link(eye)
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    cam.data.lens = 50
    cam.location = (0, -6, 1.8)
    sc.collection.objects.link(cam)
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    try:
        sc.view_settings.view_transform = "AgX"
        sc.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        sc.view_settings.view_transform = "Standard"
    ee = sc.eevee
    ee.volumetric_end = 90
    ee.volumetric_tile_size = "4"
    ee.volumetric_samples = 96
    ee.use_volumetric_shadows = True
    ee.taa_render_samples = 64
    sc["glow"] = 3.0
    sc["fly_view"] = 0.0


def painterly_post():
    """Oil-paint brushwork (anisotropic Kuwahara), soft bloom, warm grade,
    vignette."""
    sc = bpy.context.scene
    old = bpy.data.node_groups.get("Painterly")
    if old:
        bpy.data.node_groups.remove(old)
    ng = bpy.data.node_groups.new("Painterly", "CompositorNodeTree")
    sc.compositing_node_group = ng
    ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    n, l = ng.nodes, ng.links
    rl = n.new("CompositorNodeRLayers")
    out = n.new("NodeGroupOutput")
    kw = n.new("CompositorNodeKuwahara")
    for k, v in (("Size", 7), ("Uniformity", 4), ("Sharpness", 0.6), ("Eccentricity", 1.2)):
        if k in kw.inputs:
            kw.inputs[k].default_value = v
    try:
        kw.inputs["Type"].default_value = "Anisotropic"
    except (TypeError, KeyError):
        pass
    glare = n.new("CompositorNodeGlare")
    for k, v in (("Type", "Fog Glow"), ("Threshold", 0.75), ("Size", 0.7), ("Strength", 0.55)):
        if k in glare.inputs:
            try:
                glare.inputs[k].default_value = v
            except TypeError:
                pass
    l.new(rl.outputs["Image"], kw.inputs["Image"])
    l.new(kw.outputs[0], glare.inputs["Image"])
    l.new(glare.outputs["Image"], out.inputs[0])


def clear():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.node_groups, bpy.data.cameras,
                 bpy.data.lights, bpy.data.images, bpy.data.worlds):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


def main():
    clear()
    h = Heaven()
    ground(h)
    far_ground(h)
    grass(h)
    flowers(h)
    trees(h)
    fruit(h)
    motes(h)
    haze()
    h.world()
    light_and_render()
    painterly_post()
    tris = sum(len(o.data.polygons) for o in bpy.data.objects if o.type == "MESH")
    print("Eden built:", tris, "faces")


main()
