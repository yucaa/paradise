// Cube flies. Positions and headings are in Blender b-space (Z up); the flies
// live inside the meadow's rotated world group.
//
// The hero steers by its own brain (run-and-tumble): after each brain reading,
// if valence fell it tumbles to a new random heading, otherwise it keeps going.
// It also slows where valence is high (orthokinesis), so it lingers where the
// world is good. Both are behaviours real flies use. Nothing tells it where
// heaven is; "guide" mode (key G) leads it there for a demonstration.
//
// It also keeps a fly's day: hunger builds while it wanders, until it flies
// down to the nearest fruit and feeds; fed, it flies up into a tree and rests
// on a twig, wings folded, swaying with the branch, before wandering again.
import * as THREE from "three";
import { EYE, WATER_LEVEL } from "./meadow.js";

export const CUBE = 0.6;
export const FLY_SCALE = 0.12;
export const HERO_SCALE = FLY_SCALE * 1.4;
const EYE_OFFSET = (CUBE / 2 + 0.02) * HERO_SCALE;   // eye sits on the cube's front face

function wingGeometry(side) {
  const s = new THREE.Shape();
  const pts = [[0, 0.16], [0.7, 0.2], [1.05, 0.08], [1.05, -0.12], [0.7, -0.22], [0, -0.16]];
  s.moveTo(pts[0][0] * side, pts[0][1]);
  pts.slice(1).forEach(([x, y]) => s.lineTo(x * side, y));
  return new THREE.ShapeGeometry(s);
}

const body = new THREE.MeshStandardMaterial({ color: 0x0a0a0c, roughness: 0.25, metalness: 0.1 });
const wing = new THREE.MeshStandardMaterial({ color: 0xfffaf0, emissive: 0xfff6e8, emissiveIntensity: 0.2,
  transparent: true, opacity: 0.72, side: THREE.DoubleSide, roughness: 0.1, depthWrite: false });
const silhouette = new THREE.MeshBasicMaterial({ color: 0x080808, side: THREE.DoubleSide });

function makeFly(scale) {
  const g = new THREE.Group();
  const cube = new THREE.Mesh(new THREE.BoxGeometry(CUBE, CUBE, CUBE), body);
  g.add(cube);
  const wings = [-1, 1].map((side) => {
    const hinge = new THREE.Group();
    hinge.position.set((CUBE / 2) * side, 0, CUBE / 2);
    const w = new THREE.Mesh(wingGeometry(side), wing);
    hinge.add(w);
    g.add(hinge);
    return { side, hinge, mesh: w };
  });
  g.scale.setScalar(scale);
  return { group: g, cube, wings, phase: Math.random() * 6.28 };
}

// Small pitch/roll into the wind, in the fly's own frame.
function bank(w, yaw) {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const across = w.x * c + w.y * s, along = -w.x * s + w.y * c;   // body right, forward
  return [-0.25 * along, 0.3 * across];
}

function flap(fly, t, folded = 0) {
  // ~200 Hz in life; strobed here. Folded: swept back over the body, with
  // the odd flick.
  const beat = Math.sin(fly.phase + t * 47.0);
  const flick = Math.max(0, Math.sin(t * 0.9 + fly.phase) ** 40) * 0.3;
  for (const w of fly.wings) {
    const fly_ = [0, w.side * beat * 0.9, -w.side * 0.15];
    const rest = [0.08, w.side * 0.05, -w.side * (1.35 - flick)];
    w.hinge.rotation.set(...fly_.map((a, k) => a + (rest[k] - a) * folded));
  }
}

const HUNGER_S = 50;     // seconds of wandering before it goes to feed
const EAT_S = 12;
const REST_S = 18;

export class Flight {
  constructor(world, spots, groundHeight) {
    this.world = world;
    this.spots = spots;
    this.floor = (x, y) => Math.max(groundHeight(x, y), WATER_LEVEL);   // ground, or the pond's surface
    this.state = "wander";       // wander | toEat | eat | toRest | rest
    this.stateT = 0;
    this.hunger = 0.4;
    this.target = null;
    this.folded = 0;
    this.hero = makeFly(HERO_SCALE);
    world.add(this.hero.group);
    this.swarm = Array.from({ length: 14 }, (_, i) => {
      const f = makeFly(FLY_SCALE * (0.8 + 0.4 * Math.random()));
      f.centre = new THREE.Vector3((Math.random() - 0.5) * 20, Math.random() * 20 - 6, 0);
      f.seed = Math.random() * 100;
      f.speed = 0.15 + Math.random() * 0.2;
      world.add(f.group);
      return f;
    });
    this.pos = new THREE.Vector3(6, -7, 1.6);
    this.yaw = 0.8;
    this.targetYaw = this.yaw;
    this.climb = 0;
    this.lastValence = null;
    this.valence = 0;
    this.mode = "seek";            // seek | guide
    this.drift = new THREE.Vector2();
    this.trail = [];
  }

  // Called after every brain reading.
  sense(valence) {
    this.valence = valence;
    if (this.mode === "seek" && this.state === "wander" && this.lastValence !== null && valence < this.lastValence - 0.04) {
      this.targetYaw = this.yaw + (Math.random() - 0.5) * 2.6;          // tumble
      this.climb = (Math.random() - 0.5) * 0.6;
    }
    this.lastValence = valence;
  }

  guide(on) {
    this.mode = on ? "guide" : "seek";
    this.setState("wander");
  }

  setState(s) {
    this.state = s;
    this.stateT = 0;
  }

  nearest(list, minZ = -Infinity) {
    let best = null, bd = Infinity;
    for (const q of list) {
      if (q.z < minZ) continue;
      const d = q.distanceTo(this.pos) * (0.8 + 0.4 * Math.random());
      if (d < bd) { bd = d; best = q; }
    }
    return best;
  }

  // Fly to a point: turn toward it, slow on the approach, settle onto it.
  approach(dt, target) {
    const p = this.pos;
    const to = target.clone().sub(p);
    const flat = Math.hypot(to.x, to.y);
    this.targetYaw = flat > 0.05 ? Math.atan2(-to.x, to.y) : this.yaw;
    this.yaw += Math.atan2(Math.sin(this.targetYaw - this.yaw), Math.cos(this.targetYaw - this.yaw)) * (1 - Math.exp(-dt * 3));
    const d = to.length();
    const speed = Math.min(1.2, 0.15 + d * 0.6);
    // Cruise above the grass (and the rise of the ground ahead), then drop
    // onto the target at the end.
    const ahead = p.clone().addScaledVector(this.forward(), Math.min(flat, 1.5));
    const g = Math.max(this.floor(p.x, p.y), this.floor(ahead.x, ahead.y));
    const cruise = Math.max(target.z + Math.min(1.2, flat * 0.35), g + 0.9);
    const wantZ = flat < 0.6 ? target.z : cruise;
    p.addScaledVector(this.forward(), Math.min(flat, speed * dt));
    p.z += (wantZ - p.z) * (1 - Math.exp(-dt * (flat < 0.6 ? 3 : 1.2)));
    return d;
  }

  forward() { return new THREE.Vector3(-Math.sin(this.yaw), Math.cos(this.yaw), 0); }

  update(dt, t, wind) {
    const p = this.pos;
    if (this.mode === "guide") {
      // Lead the eye onto the heaven point, level and facing forward.
      const target = EYE.clone().sub(new THREE.Vector3(0, EYE_OFFSET, 0));
      p.lerp(target, 1 - Math.exp(-dt * 0.6));
      this.targetYaw = 0;
      this.yaw += Math.atan2(Math.sin(this.targetYaw - this.yaw), Math.cos(this.targetYaw - this.yaw)) * (1 - Math.exp(-dt * 1.2));
    } else if (this.state !== "wander") {
      this.stateT += dt;
      const sit = HERO_SCALE * CUBE * 0.5;
      if (this.state === "toEat" || this.state === "toRest") {
        const at = this.target.clone().add(new THREE.Vector3(0, 0, sit));
        if (this.approach(dt, at) < 0.03 || this.stateT > 40) this.setState(this.state === "toEat" ? "eat" : "rest");
        this.drift.multiplyScalar(Math.exp(-dt * 2));
      } else {
        // Feeding or resting: settled on the spot. On a twig it rides the
        // branch's sway (as the tree shader moves twig ends).
        const w = wind.at(this.target.x, this.target.y);
        const lw = w.length(), wd = lw > 1e-4 ? w.clone().divideScalar(lw) : new THREE.Vector2();
        const sway = this.state === "rest" ? 1 : 0;
        const s = Math.sin(t * 0.95 - (this.target.x * wd.x + this.target.y * wd.y) * 0.6);
        p.set(this.target.x + sway * (w.x * 0.38 + wd.x * lw * 0.28 * s),
              this.target.y + sway * (w.y * 0.38 + wd.y * lw * 0.28 * s), this.target.z + sit);
        if (this.state === "eat") this.hunger = Math.max(0, this.hunger - dt / EAT_S);
        // Grooming turns, now and then.
        if (Math.random() < dt * 0.15) this.targetYaw = this.yaw + (Math.random() - 0.5) * 1.6;
        this.yaw += Math.atan2(Math.sin(this.targetYaw - this.yaw), Math.cos(this.targetYaw - this.yaw)) * (1 - Math.exp(-dt * 2));
        if (this.state === "eat" && this.stateT > EAT_S) {
          this.target = this.nearest(this.spots.perches, 1.5);
          this.setState(this.target ? "toRest" : "wander");
          p.z += 0.05;
        } else if (this.state === "rest" && this.stateT > REST_S) {
          this.setState("wander");
          this.climb = 0.3;
          this.targetYaw = this.yaw + Math.PI * (0.5 + Math.random());
        }
      }
    } else {
      this.hunger += dt / HUNGER_S;
      if (this.hunger >= 1 && this.spots.fruits.length) {
        this.target = this.nearest(this.spots.fruits);
        this.setState("toEat");
      }
      // Run: speed falls where the world is good (orthokinesis).
      const calm = 1 / (1 + Math.exp(-(this.valence - 6) / 0.5));
      const speed = 1.1 * (1 - 0.92 * calm);
      // Stay within the meadow and above the grass.
      const home = Math.hypot(p.x, p.y);
      if (home > 12) this.targetYaw = Math.atan2(p.x, -p.y) + (Math.random() - 0.5) * 0.3;
      const above = p.z - this.floor(p.x, p.y);
      const dz = (above < 0.7 ? 0.4 : above > 2.6 ? -0.4 : 0) + this.climb;
      this.yaw += Math.atan2(Math.sin(this.targetYaw - this.yaw), Math.cos(this.targetYaw - this.yaw)) * (1 - Math.exp(-dt * 2.5));
      p.addScaledVector(this.forward(), speed * dt);
      p.z += dz * speed * dt;
      // The breeze pushes it; its brain has to steer back.
      const w = wind.at(p.x, p.y);
      this.drift.lerp(w, 1 - Math.exp(-dt * 1.5));
      p.x += this.drift.x * 0.45 * dt; p.y += this.drift.y * 0.45 * dt;
      p.z += Math.sin(t * 2.3) * this.drift.length() * 0.05 * dt;
      this.climb *= Math.exp(-dt * 0.8);
    }
    const settled = this.mode === "seek" && (this.state === "eat" || this.state === "rest");
    // Never through the ground: the cube's underside stays above it.
    if (!settled) p.z = Math.max(p.z, this.floor(p.x, p.y) + 0.12);
    this.folded += ((settled ? 1 : 0) - this.folded) * (1 - Math.exp(-dt * 4));
    this.hero.group.position.copy(p);
    this.hero.group.rotation.set(...bank(this.drift, this.yaw).map((a) => a * (1 - this.folded)), this.yaw);
    flap(this.hero, t, this.folded);
    for (const f of this.swarm) {
      const s = t * f.speed;
      f.group.position.set(
        f.centre.x + 5 * Math.sin(s * 1.3 + f.seed) + 2 * Math.sin(s * 3.1 + f.seed * 2),
        f.centre.y + 5 * Math.cos(s * 1.1 + f.seed * 0.7) + 2 * Math.cos(s * 2.7 + f.seed),
        1.4 + 1.1 * Math.sin(s * 1.7 + f.seed * 1.3));
      // Tossed by the gusts, with a little inertia; banking into them.
      const w = wind.at(f.group.position.x, f.group.position.y);
      f.drift ??= new THREE.Vector2();
      f.offset ??= new THREE.Vector3();
      f.drift.lerp(w, 1 - Math.exp(-dt * 2.0));
      f.offset.x += (f.drift.x * 1.2 - f.offset.x) * (1 - Math.exp(-dt * 0.9));
      f.offset.y += (f.drift.y * 1.2 - f.offset.y) * (1 - Math.exp(-dt * 0.9));
      f.offset.z = Math.sin(t * 1.9 + f.seed) * f.drift.length() * 0.15;
      f.group.position.add(f.offset);
      const fp = f.group.position;
      fp.z = Math.max(fp.z, this.floor(fp.x, fp.y) + 0.3);
      const v = new THREE.Vector3(Math.cos(s * 1.3 + f.seed) * 1.3, -Math.sin(s * 1.1 + f.seed * 0.7) * 1.1, 0);
      const yaw = Math.atan2(-v.x, v.y);
      f.group.rotation.set(...bank(f.drift, yaw), yaw);
      flap(f, t);
    }
    if (!this.trail.length || this.trail[this.trail.length - 1].distanceTo(p) > 0.05) {
      this.trail.push(p.clone());
      if (this.trail.length > 600) this.trail.shift();
    }
  }

  // b-space eye position and body rotation (columns: right, forward, up).
  eye() { return this.pos.clone().addScaledVector(this.forward(), EYE_OFFSET); }
  bodyRotation() {
    const c = Math.cos(this.yaw), s = Math.sin(this.yaw);
    return new THREE.Matrix3().set(c, -s, 0, s, c, 0, 0, 0, 1);
  }

  // In the fly's own view: no hero body, other flies as dark silhouettes.
  flyView(on) {
    this.hero.group.visible = !on;
    for (const f of this.swarm) {
      f.cube.material = on ? silhouette : body;
      for (const w of f.wings) w.mesh.material = on ? silhouette : wing;
    }
  }
}
