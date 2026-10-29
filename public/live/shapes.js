// Geometry made in the browser on top of the exported meadow: a gentle terrain
// warp (so pools take irregular, lobed shapes), rocks along every shoreline,
// lily pads on the water, and stylized trees. All of it is b-space (Blender Z-up) and carries the same
// attributes as the exported parts (root, tip, card), so it is coloured by
// heaven like everything else.
import * as THREE from "three";

// --- noise --------------------------------------------------------------------------

function hash2(x, y) {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return s - Math.floor(s);
}
function vnoise2(x, y) {
  const i = Math.floor(x), j = Math.floor(y);
  let u = x - i, v = y - j;
  u = u * u * (3 - 2 * u); v = v * v * (3 - 2 * v);
  const a = hash2(i, j), b = hash2(i + 1, j), c = hash2(i, j + 1), d = hash2(i + 1, j + 1);
  return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v;
}
function fbm(x, y) {
  return (0.5 * vnoise2(x, y) + 0.25 * vnoise2(2.03 * x, 2.03 * y) + 0.125 * vnoise2(4.1 * x, 4.1 * y)) / 0.875;
}

// Domain-warped noise: soft coves, spits and lobes rather than ovals.
export function terrainWarp(x, y) {
  const wx = x + 4 * (fbm(x * 0.06 + 5.2, y * 0.06) - 0.5);
  const wy = y + 4 * (fbm(x * 0.06, y * 0.06 + 9.7) - 0.5);
  return 0.55 * (fbm(wx * 0.2, wy * 0.2) - 0.5) + 0.12 * (vnoise2(wx * 0.9, wy * 0.9) - 0.5);
}

// --- mesh helpers -------------------------------------------------------------------

// Builds a non-indexed (flat-shaded) BufferGeometry from triangle lists.
function geometry(tris, roots, tips) {
  const n = tris.length / 3;
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(tris), 3));
  g.setAttribute("root", new THREE.BufferAttribute(new Float32Array(roots), 3));
  g.setAttribute("tip", new THREE.BufferAttribute(new Float32Array(tips), 1));
  g.setAttribute("card", new THREE.BufferAttribute(new Float32Array(n * 2), 2));
  g.computeVertexNormals();
  return g;
}

// A jittered low-poly blob (icosahedron), transformed and appended to lists.
const ico = new THREE.IcosahedronGeometry(1, 1).toNonIndexed().attributes.position.array;
function blob(out, centre, scale, rot, jitter, rand, root, tip) {
  const m = new THREE.Matrix4().compose(centre, new THREE.Quaternion().setFromEuler(rot), scale);
  const v = new THREE.Vector3();
  // Jitter shared vertices consistently, so the blob stays closed.
  const key = (x, y, z) => `${x.toFixed(3)},${y.toFixed(3)},${z.toFixed(3)}`;
  const offs = new Map();
  for (let i = 0; i < ico.length; i += 3) {
    const k = key(ico[i], ico[i + 1], ico[i + 2]);
    if (!offs.has(k)) offs.set(k, 1 + (rand() - 0.5) * jitter);
    v.set(ico[i], ico[i + 1], ico[i + 2]).multiplyScalar(offs.get(k)).applyMatrix4(m);
    out.tris.push(v.x, v.y, v.z);
    out.roots.push(root.x, root.y, root.z);
    out.tips.push(typeof tip === "function" ? tip(v) : tip);
  }
}

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

// --- shore rocks ------------------------------------------------------------------

// Walk the terrain grid, find where it crosses the waterline, and set stones
// along those shores at irregular intervals, half-submerged, some in clusters.
export function shoreRocks(groundHeight, level, radius = 40) {
  const rand = rng(17);
  const step = 0.35;
  const pts = [];
  for (let x = -radius; x < radius; x += step) {
    for (let y = -radius; y < radius; y += step) {
      if (x * x + y * y > radius * radius) continue;
      const a = groundHeight(x, y) - level, b = groundHeight(x + step, y) - level, c = groundHeight(x, y + step) - level;
      if (a * b < 0) pts.push([x + step * a / (a - b), y]);
      else if (a * c < 0) pts.push([x, y + step * a / (a - c)]);
    }
  }
  // Irregular spacing: shuffle, then keep points at least a random gap apart.
  for (let i = pts.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [pts[i], pts[j]] = [pts[j], pts[i]]; }
  const kept = [];
  for (const p of pts) {
    const gap = 0.45 + rand() * 0.9;
    if (kept.every((q) => (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 > gap * gap)) kept.push(p);
  }
  const out = { tris: [], roots: [], tips: [] };
  for (const [x, y] of kept) {
    const count = rand() < 0.45 ? 2 + Math.floor(rand() * 3) : 1;
    for (let k = 0; k < count; k++) {
      const s = 0.1 + rand() ** 2 * 0.45;
      const c = new THREE.Vector3(x + (rand() - 0.5) * 0.6 * k, y + (rand() - 0.5) * 0.6 * k, level - s * 0.12);
      blob(out, c, new THREE.Vector3(s * (1 + rand() * 0.5), s * (0.8 + rand() * 0.4), s * (0.45 + rand() * 0.3)),
        new THREE.Euler(rand() * 0.4, rand() * 0.4, rand() * 6.28), 0.35, rand, c, 0);
    }
  }
  return { geometry: geometry(out.tris, out.roots, out.tips), count: kept.length };
}

// --- lily pads ----------------------------------------------------------------------

// Colonies of floating pads in the calm shallows, away from the stones at the
// shore: round leaves with the lily's notch, packed without overlapping.
// root = the pad's centre (it turns and bobs about it), tip = distance from the
// centre (0..1), card = (that distance, angle / 2pi) for the veins.
export function lilyPads(groundHeight, level, radius = 40) {
  const rand = rng(29);
  const pads = [];
  for (let x = -radius; x < radius; x += 0.3) {
    for (let y = -radius; y < radius; y += 0.3) {
      const px = x + (rand() - 0.5) * 0.3, py = y + (rand() - 0.5) * 0.3;
      const depth = level - groundHeight(px, py);
      if (depth < 0.1 || depth > 0.6 || px * px + py * py > radius * radius) continue;
      if (vnoise2(px * 0.3 + 3.1, py * 0.3 - 7.4) < 0.5 || rand() < 0.35) continue;   // in colonies
      const r = 0.1 + rand() ** 1.5 * 0.24;
      if (pads.every((q) => Math.hypot(q.x - px, q.y - py) > (q.r + r) * 1.05)) pads.push({ x: px, y: py, r });
    }
  }
  const tris = [], roots = [], tips = [], cards = [];
  const K = 20;
  for (const { x, y, r } of pads) {
    const notch = rand() * 6.283, gap = 0.22 + rand() * 0.12;
    const z = level + 0.006;
    const at = (k) => {
      const a = notch + gap + (k / K) * (6.283 - 2 * gap);
      const rr = r * (1 + (vnoise2(x * 9 + k * 0.7, y * 9) - 0.5) * 0.08);
      return [x + Math.cos(a) * rr, y + Math.sin(a) * rr, z + 0.004, (a - notch) / 6.283];
    };
    for (let k = 0; k < K; k++) {
      const a = at(k), b = at(k + 1);
      tris.push(x, y, z, ...a.slice(0, 3), ...b.slice(0, 3));
      for (let i = 0; i < 3; i++) roots.push(x, y, level);
      tips.push(0, 1, 1);
      cards.push(0, (a[3] + b[3]) / 2, 1, a[3], 1, b[3]);
    }
  }
  const g = geometry(tris, roots, tips);
  g.setAttribute("card", new THREE.BufferAttribute(new Float32Array(cards), 2));
  return { geometry: g, count: pads.length };
}

// --- trees grown like neurons ------------------------------------------------------

// Space colonization (Runions, Lane & Prusinkiewicz 2007): the crown is a cloud
// of attraction points; the skeleton grows step by step toward them, and points
// are consumed once reached. The same algorithm grows synthetic dendrites, so
// the trees of this Eden branch the way the fly's neurons do. Branch radii
// follow the pipe model: a parent carries the combined cross-section of its
// children.
function colonize(base, crown, radii, rand) {
  const D = 0.32, influence = 3.2, kill = 0.55;
  const attractors = [];
  while (attractors.length < 260) {
    const p = new THREE.Vector3(rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1);
    if (p.lengthSq() > 1) continue;
    attractors.push(p.multiply(radii).add(crown));
  }
  const nodes = [{ p: base.clone(), parent: -1 }];
  // Trunk: straight-ish growth until the crown is within reach.
  while (true) {
    const last = nodes[nodes.length - 1].p;
    if (attractors.some((a) => a.distanceToSquared(last) < influence * influence)) break;
    const dir = crown.clone().sub(last).normalize();
    nodes.push({ p: last.clone().addScaledVector(dir, D), parent: nodes.length - 1 });
    if (nodes.length > 80) break;
  }
  for (let iter = 0; iter < 140 && attractors.length; iter++) {
    const pull = new Map();
    for (const a of attractors) {
      let best = -1, bd = influence * influence;
      for (let i = 0; i < nodes.length; i++) {
        const d = nodes[i].p.distanceToSquared(a);
        if (d < bd) { bd = d; best = i; }
      }
      if (best < 0) continue;
      const v = pull.get(best) || new THREE.Vector3();
      pull.set(best, v.add(a.clone().sub(nodes[best].p).normalize()));
    }
    if (!pull.size) break;
    for (const [i, v] of pull) {
      v.normalize().add(new THREE.Vector3(0, 0, 0.12)).normalize();   // a little phototropism
      nodes.push({ p: nodes[i].p.clone().addScaledVector(v, D), parent: i });
    }
    for (let k = attractors.length - 1; k >= 0; k--)
      if (nodes.some((n) => n.p.distanceToSquared(attractors[k]) < kill * kill)) attractors.splice(k, 1);
  }
  // Pipe model radii, leaves upward.
  const children = nodes.map(() => []);
  nodes.forEach((n, i) => n.parent >= 0 && children[n.parent].push(i));
  const r = new Float32Array(nodes.length);
  for (let i = nodes.length - 1; i >= 0; i--) {
    r[i] = children[i].length ? Math.pow(children[i].reduce((s, c) => s + Math.pow(r[c], 2.4), 0), 1 / 2.4) : 0.018;
  }
  // Path length from the base, for pulses that travel trunk -> tips.
  const depth = new Float32Array(nodes.length);
  for (let i = 1; i < nodes.length; i++) depth[i] = depth[nodes[i].parent] + D;
  return { nodes, r, depth, children };
}

export function neuronTrees(trunkPos, ringsPerTree = 10, segs = 9) {
  const rand = rng(5);
  const perTree = ringsPerTree * segs;
  const branch = { tris: [], roots: [], tips: [] };
  const leaf = { pos: [], normal: [], root: [], tip: [], card: [] };
  const perches = [];   // twig ends in the outer crown, where a fly can rest
  const crowns = [];    // { base, crown, radii } per tree, for the map
  const ringCentre = (t, ring) => {
    const c = new THREE.Vector3();
    for (let s = 0; s < segs; s++) {
      const i = (t * perTree + ring * segs + s) * 3;
      c.x += trunkPos[i]; c.y += trunkPos[i + 1]; c.z += trunkPos[i + 2];
    }
    return c.divideScalar(segs);
  };
  const trees = trunkPos.length / 3 / perTree;
  const up = new THREE.Vector3(0, 1, 0), v = new THREE.Vector3();
  for (let t = 0; t < trees; t++) {
    const base = ringCentre(t, 0), top = ringCentre(t, ringsPerTree - 1);
    const h = top.z - base.z;
    const crown = new THREE.Vector3(top.x, top.y, base.z + h * (0.72 + rand() * 0.12));
    const radii = new THREE.Vector3(2.4 + rand() * 1.6, 2.4 + rand() * 1.6, 2.0 + rand() * 1.4);
    crowns.push({ base, crown, radii });
    const { nodes, r, depth, children } = colonize(base, crown, radii, rand);
    const maxDepth = Math.max(...depth);
    // Tapered tubes, one per segment.
    for (let i = 1; i < nodes.length; i++) {
      const a = nodes[nodes[i].parent].p, b = nodes[i].p;
      const dir = b.clone().sub(a);
      const len = dir.length();
      const cyl = new THREE.CylinderGeometry(r[i], r[nodes[i].parent], len * 1.04, 6, 1, true)
        .toNonIndexed().attributes.position.array;
      const m = new THREE.Matrix4().compose(a.clone().add(b).multiplyScalar(0.5),
        new THREE.Quaternion().setFromUnitVectors(up, dir.normalize()), new THREE.Vector3(1, 1, 1));
      const mid = a.clone().add(b).multiplyScalar(0.5);
      for (let k = 0; k < cyl.length; k += 3) {
        v.set(cyl[k], cyl[k + 1], cyl[k + 2]).applyMatrix4(m);
        branch.tris.push(v.x, v.y, v.z);
        branch.roots.push(mid.x, mid.y, mid.z);
        branch.tips.push(depth[i] / maxDepth);
      }
      // Foliage: soft billboard clusters at the twig ends. Normals point out
      // from the crown's centre, so light rolls smoothly over the whole mass.
      // Clusters on every twig end, and more sparsely along the outer crown.
      const twig = !children[i].length;
      if (twig && b.z > base.z + 1.2) perches.push(b.clone());
      const outer = depth[i] / maxDepth > 0.55 && children[i].length === 1 && rand() < 0.35;
      if (twig || outer) {
        const nC = twig ? 3 + Math.floor(rand() * 3) : 1 + Math.floor(rand() * 2);
        for (let c = 0; c < nC; c++) {
          const centre = b.clone().add(new THREE.Vector3(rand() - 0.5, rand() - 0.5, rand() - 0.5).multiplyScalar(0.8));
          const n = centre.clone().sub(crown).divide(radii).normalize();
          const size = 0.7 + rand() * 0.7;
          for (const [cu, cv] of [[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]]) {
            leaf.pos.push(centre.x, centre.y, centre.z);
            leaf.normal.push(n.x, n.y, n.z);
            leaf.root.push(centre.x, centre.y, centre.z);
            leaf.tip.push(size);
            leaf.card.push(cu, cv);
          }
        }
      }
    }
  }
  const lg = new THREE.BufferGeometry();
  lg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(leaf.pos), 3));
  lg.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(leaf.normal), 3));
  lg.setAttribute("root", new THREE.BufferAttribute(new Float32Array(leaf.root), 3));
  lg.setAttribute("tip", new THREE.BufferAttribute(new Float32Array(leaf.tip), 1));
  lg.setAttribute("card", new THREE.BufferAttribute(new Float32Array(leaf.card), 2));
  return { branches: geometry(branch.tris, branch.roots, branch.tips), foliage: lg, trees, perches, crowns };
}
