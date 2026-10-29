"""Export the meadow built by eden.py for the real-time viewer. Run in Blender:
    exec(open(r"<repo>/blender/export_meadow.py").read())

Each part becomes <name>.bin holding, back to back: positions f32x3, normals
f32x3, root f32x3, tip f32, [card f32x2], indices u32 — all in world space,
with Blender's Z-up axes (the viewer converts). meadow.json lists the parts.
"""
import json
from pathlib import Path

import bpy
import numpy as np

OUT = Path(r"C:\Users\Eric\orca\workspaces\sixteen\seventeen\viewer\live\data\meadow")
PARTS = {"Ground": "ground", "FarGround": "ground", "Grass": "grass", "Flowers": "flower",
         "Leaves": "leaf", "Trunks": "ground", "Pollen": "mote", "Fruit_": "fruit"}


def mesh_arrays(objs):
    dg = bpy.context.evaluated_depsgraph_get()
    P, N, R, T, C, I = [], [], [], [], [], []
    base = 0
    for ob in objs:
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        me.calc_loop_triangles()
        M = np.array(ob.matrix_world)
        nv = len(me.vertices)
        co = np.zeros(nv * 3, np.float32); me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        nr = np.zeros(nv * 3, np.float32); me.vertices.foreach_get("normal", nr)
        nr = nr.reshape(-1, 3) @ np.linalg.inv(M[:3, :3])
        nr /= np.linalg.norm(nr, axis=1, keepdims=True) + 1e-9
        def attr(name, width, default):
            a = me.attributes.get(name)
            if a is None:
                return default
            out = np.zeros(nv * width, np.float32)
            a.data.foreach_get("vector" if width == 3 else "value", out)
            return out.reshape(-1, width) if width > 1 else out
        root = attr("root", 3, co.copy())
        if ob.name.startswith("Fruit_"):
            root = co.copy()
        tip = attr("tip", 1, np.zeros(nv, np.float32))
        card = attr("card", 3, np.zeros((nv, 3), np.float32))[:, :2]
        tri = np.zeros(len(me.loop_triangles) * 3, np.uint32)
        me.loop_triangles.foreach_get("vertices", tri)
        P.append(co); N.append(nr); R.append(root); T.append(tip); C.append(card); I.append(tri + base)
        base += nv
        ev.to_mesh_clear()
    cat = lambda xs: np.concatenate(xs).astype(np.float32)
    return cat(P), cat(N), cat(R), cat(T), cat(C), np.concatenate(I).astype(np.uint32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = []
    for prefix, kind in PARTS.items():
        objs = [o for o in bpy.data.objects if o.type == "MESH" and
                (o.name == prefix or (prefix.endswith("_") and o.name.startswith(prefix)))]
        if not objs:
            continue
        P, N, R, T, C, I = mesh_arrays(objs)
        name = prefix.rstrip("_").lower()
        blobs = [P, N, R, T] + ([C] if kind == "leaf" else []) + [I]
        with open(OUT / f"{name}.bin", "wb") as f:
            for b in blobs:
                f.write(np.ascontiguousarray(b).tobytes())
        meta.append({"name": name, "kind": kind, "verts": int(len(P)), "indices": int(len(I)),
                     "card": kind == "leaf"})
        print(name, len(P), "verts", len(I) // 3, "tris")
    (OUT / "meadow.json").write_text(json.dumps({"parts": meta, "eye": [0.0, 0.0, 1.2]}))


main()
