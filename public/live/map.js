// The garden in miniature: an isometric survey of the meadow, drawn as a
// surveyor would, in contour-like grid lines, with the ponds laid in flat, the
// trees as stems and crowns, the fruit as points, and the fly tracked across
// it with its path, its height above the ground, and its shadow. The map turns
// to look the same way as the main camera, so what is ahead in the view is up
// the map. All positions are b-space (Z up).

const R = 22;               // map radius, metres (the fly wanders within ~12, and rests in the trees beyond)
const STEP = 1.1;           // grid spacing, metres
const LIFT = 2.6;           // vertical exaggeration
const DEPTH = 1.4;          // how deep the block of ground is drawn below the water, metres

export class Minimap {
  constructor(canvas, { groundHeight, waterLevel, crowns, fruits }) {
    this.c = canvas;
    this.ctx = canvas.getContext("2d");
    this.water = waterLevel;
    this.crowns = crowns.filter((t) => Math.hypot(t.base.x, t.base.y) < R - 1.5);
    this.fruits = fruits.filter((f) => Math.hypot(f.x, f.y) < R);
    // Height grid (ponds flattened to their surface), and which cells are water.
    const n = Math.round((2 * R) / STEP) + 1;
    this.n = n;
    this.z = new Float32Array(n * n);
    this.wet = new Uint8Array(n * n);
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const x = -R + i * STEP, y = -R + j * STEP, g = groundHeight(x, y);
        this.z[j * n + i] = Math.max(g, waterLevel);
        this.wet[j * n + i] = g < waterLevel ? 1 : 0;
      }
    }
    this.ground = (x, y) => Math.max(groundHeight(x, y), waterLevel);
    this.angle = null;
  }

  resize() {
    const dpr = devicePixelRatio || 1;
    const w = Math.round(this.c.clientWidth * dpr), h = Math.round(this.c.clientHeight * dpr);
    if (this.c.width !== w || this.c.height !== h) { this.c.width = w; this.c.height = h; }
    return [w, h, dpr];
  }

  // flight: the Flight; camYaw: the main camera's heading in b-space (radians,
  // atan2 of its forward direction).
  update(dt, flight, camYaw) {
    const [w, h, dpr] = this.resize();
    if (!w || !h) return;
    const ctx = this.ctx;
    // Turn with the camera, smoothly: its forward direction points up the map.
    const want = 0.25 * Math.PI - camYaw;
    if (this.angle === null) this.angle = want;
    this.angle += Math.atan2(Math.sin(want - this.angle), Math.cos(want - this.angle)) * (1 - Math.exp(-dt * 1.5));
    const ca = Math.cos(this.angle), sa = Math.sin(this.angle);
    const s = Math.min(w / (2 * R), h / (R + LIFT * (DEPTH + 4))) * 0.94;
    const cx = w / 2, cy = h / 2 + s * LIFT * 1.6;
    const P = (x, y, z) => {
      const xr = x * ca - y * sa, yr = x * sa + y * ca;
      return [cx + (xr - yr) * 0.7071 * s, cy - (xr + yr) * 0.3536 * s - (z - this.water) * s * LIFT * 0.5];
    };
    ctx.clearRect(0, 0, w, h);
    const { n, z, wet } = this;
    const inside = (i, j) => { const x = -R + i * STEP, y = -R + j * STEP; return x * x + y * y <= R * R; };
    const at = (i, j) => P(-R + i * STEP, -R + j * STEP, z[j * n + i]);

    // Ponds, laid in flat.
    ctx.fillStyle = "rgba(130,155,215,0.16)";
    ctx.beginPath();
    for (let j = 0; j < n - 1; j++) {
      for (let i = 0; i < n - 1; i++) {
        if (!(wet[j * n + i] || wet[j * n + i + 1] || wet[(j + 1) * n + i] || wet[(j + 1) * n + i + 1])) continue;
        if (!inside(i, j) || !inside(i + 1, j + 1)) continue;
        const a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1), d = at(i, j + 1);
        ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.lineTo(c[0], c[1]); ctx.lineTo(d[0], d[1]); ctx.closePath();
      }
    }
    ctx.fill();

    // Terrain grid, both ways; brighter where the ground stands higher.
    ctx.lineWidth = Math.max(1, dpr * 0.6);
    for (const dir of [0, 1]) {
      for (let a = 0; a < n; a++) {
        let open = false;
        ctx.beginPath();
        for (let b = 0; b < n; b++) {
          const [i, j] = dir ? [a, b] : [b, a];
          if (!inside(i, j)) { open = false; continue; }
          const [px, py] = at(i, j);
          if (open) ctx.lineTo(px, py); else ctx.moveTo(px, py);
          open = true;
        }
        ctx.strokeStyle = "rgba(226,230,236,0.13)";
        ctx.stroke();
      }
    }
    // The rim of the survey.
    ctx.strokeStyle = "rgba(226,230,236,0.22)";
    ctx.beginPath();
    for (let k = 0; k <= 96; k++) {
      const t = (k / 96) * 2 * Math.PI, x = R * Math.cos(t), y = R * Math.sin(t);
      const [px, py] = P(x, y, this.ground(x, y));
      if (k) ctx.lineTo(px, py); else ctx.moveTo(px, py);
    }
    ctx.stroke();
    // The near side of the block of ground beneath it, as in a diorama.
    const base = this.water - DEPTH;
    ctx.strokeStyle = "rgba(226,230,236,0.1)";
    ctx.beginPath();
    let open = false;
    for (let k = 0; k <= 96; k++) {
      const t = (k / 96) * 2 * Math.PI, x = R * Math.cos(t), y = R * Math.sin(t);
      const near = x * (ca + sa) + y * (ca - sa) < 0;
      if (!near) { open = false; continue; }
      const [px, py] = P(x, y, base);
      if (open) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      open = true;
      if (k % 6 === 0) {
        const [tx, ty] = P(x, y, this.ground(x, y));
        ctx.moveTo(tx, ty); ctx.lineTo(px, py); ctx.moveTo(px, py);
      }
    }
    ctx.stroke();

    // Fruit.
    ctx.fillStyle = "rgba(233,196,106,0.55)";
    for (const f of this.fruits) {
      const [px, py] = P(f.x, f.y, f.z);
      ctx.fillRect(px - dpr, py - dpr, 2 * dpr, 2 * dpr);
    }

    // Trees, far to near: a stem and an elliptical crown.
    const trees = this.crowns.map((t) => ({ t, base: P(t.base.x, t.base.y, t.base.z), top: P(t.crown.x, t.crown.y, t.crown.z) }))
      .sort((a, b) => a.base[1] - b.base[1]);
    for (const { t, base, top } of trees) {
      ctx.strokeStyle = "rgba(226,230,236,0.35)";
      ctx.beginPath(); ctx.moveTo(base[0], base[1]); ctx.lineTo(top[0], top[1]); ctx.stroke();
      const rx = ((t.radii.x + t.radii.y) / 2) * s * 0.7071, ry = t.radii.z * s * LIFT * 0.32;
      ctx.fillStyle = "rgba(12,12,14,0.35)";
      ctx.strokeStyle = "rgba(226,230,236,0.3)";
      ctx.beginPath(); ctx.ellipse(top[0], top[1], rx, Math.max(ry, rx * 0.5), 0, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    }

    // The other flies, faintly.
    ctx.fillStyle = "rgba(226,230,236,0.35)";
    for (const f of flight.swarm) {
      const q = f.group.position;
      if (q.x * q.x + q.y * q.y > R * R) continue;
      const [px, py] = P(q.x, q.y, q.z);
      ctx.fillRect(px - 0.75 * dpr, py - 0.75 * dpr, 1.5 * dpr, 1.5 * dpr);
    }

    // Where it is going, when it has somewhere to go.
    const p = flight.pos;
    const hero = P(p.x, p.y, p.z);
    if (flight.target && (flight.state === "toEat" || flight.state === "toRest")) {
      const g = P(flight.target.x, flight.target.y, flight.target.z);
      ctx.setLineDash([3 * dpr, 3 * dpr]);
      ctx.strokeStyle = "rgba(233,196,106,0.45)";
      ctx.beginPath(); ctx.moveTo(hero[0], hero[1]); ctx.lineTo(g[0], g[1]); ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(g[0], g[1], 3 * dpr, 0, 2 * Math.PI); ctx.stroke();
    }

    // Its path, fading with age.
    const trail = flight.trail;
    ctx.lineWidth = dpr;
    for (let k = 1; k < trail.length; k++) {
      const a = trail[k - 1], b = trail[k];
      const pa = P(a.x, a.y, a.z), pb = P(b.x, b.y, b.z);
      ctx.strokeStyle = `rgba(233,196,106,${(0.6 * k) / trail.length})`;
      ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
    }

    // The fly: a drop line to its shadow on the ground, and a glowing point.
    const gz = this.ground(p.x, p.y);
    const shadow = P(p.x, p.y, gz);
    ctx.strokeStyle = "rgba(233,196,106,0.45)";
    ctx.beginPath(); ctx.moveTo(hero[0], hero[1]); ctx.lineTo(shadow[0], shadow[1]); ctx.stroke();
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.beginPath(); ctx.ellipse(shadow[0], shadow[1], 3 * dpr, 1.5 * dpr, 0, 0, 2 * Math.PI); ctx.fill();
    const glow = ctx.createRadialGradient(hero[0], hero[1], 0, hero[0], hero[1], 9 * dpr);
    glow.addColorStop(0, "rgba(233,196,106,0.55)");
    glow.addColorStop(1, "rgba(233,196,106,0)");
    ctx.fillStyle = glow;
    ctx.beginPath(); ctx.arc(hero[0], hero[1], 9 * dpr, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = "#f3d98c";
    ctx.beginPath(); ctx.arc(hero[0], hero[1], 2.4 * dpr, 0, 2 * Math.PI); ctx.fill();

    // Readout.
    ctx.font = `${9.5 * dpr}px "IBM Plex Mono", Consolas, monospace`;
    ctx.textBaseline = "bottom";
    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.fillText(`${(p.z - gz).toFixed(1)} m above ground`, 0, h - 2 * dpr);
  }
}
