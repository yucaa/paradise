// The camera as a slow, patient cinematographer. It holds one shot at a time
// (following, tracking alongside, waiting ahead, a high crane, and close
// orbits while the fly feeds or rests) and glides from one to the next rather
// than cutting. Movement is spring-damped, the framing leads the fly a little,
// and the lens lengthens for close shots. All positions are b-space (Z up).
import * as THREE from "three";

// [distance, height above the hero, angle relative to the hero's heading
// (0 = behind), orbit speed rad/s, field of view]
const SHOTS = {
  follow: [3.6, 1.0, 0.35, 0.0, 34],
  track: [2.8, 0.3, Math.PI / 2, 0.0, 30],
  ahead: [3.2, 0.45, Math.PI * 0.92, 0.0, 30],
  crane: [9.0, 4.2, 0.6, 0.035, 36],
  close: [1.35, 0.35, 0.9, 0.09, 24],
  perch: [2.2, -0.2, 1.4, 0.06, 26],
};
const WANDER = ["follow", "track", "crane", "ahead", "follow", "track"];

export class Director {
  constructor(groundHeight, waterLevel) {
    this.ground = (x, y) => Math.max(groundHeight(x, y), waterLevel);
    this.shot = "follow";
    this.shotT = 0;
    this.hold = 14;
    this.k = 0;
    this.orbit = 0;
    this.side = 1;
    this.pos = null;
    this.vel = new THREE.Vector3();
    this.look = new THREE.Vector3();
    this.fov = 34;
    this.zoom = 1;
  }

  choose(state) {
    const settled = state === "eat" || state === "rest";
    let next;
    if (settled) next = this.shot === "close" ? (state === "rest" ? "perch" : "crane") : "close";
    else next = WANDER[(this.k++) % WANDER.length];
    this.shot = next;
    this.shotT = 0;
    this.hold = settled ? 9 + Math.random() * 4 : 12 + Math.random() * 8;
    this.side = Math.random() < 0.5 ? -1 : 1;
    this.orbit = 0;
  }

  // hero: b-space position; yaw: heading; state: the hero's behaviour.
  update(dt, t, hero, yaw, state) {
    this.shotT += dt;
    const settled = state === "eat" || state === "rest";
    const wasSettled = this.shot === "close" || this.shot === "perch";
    if (this.shotT > this.hold || settled !== wasSettled && this.shotT > 2.5 && !(settled && this.shot === "crane")) this.choose(state);
    const [dist, h, ang, spin, fov] = SHOTS[this.shot];
    this.orbit += spin * dt;
    const a = yaw + Math.PI + this.side * ang + this.orbit;      // heading +PI = behind
    const d = dist * this.zoom;
    const want = new THREE.Vector3(hero.x - Math.sin(a) * d, hero.y + Math.cos(a) * d, hero.z + h * Math.max(1, this.zoom * 0.8));
    // Breathing, as from a hand-held rig.
    want.x += Math.sin(t * 0.31) * 0.06; want.y += Math.sin(t * 0.23 + 1) * 0.06; want.z += Math.sin(t * 0.37 + 2) * 0.04;
    want.z = Math.max(want.z, this.ground(want.x, want.y) + 0.35);
    if (!this.pos) { this.pos = want.clone(); this.look.copy(hero); }
    // Critically damped spring: glides between shots, no cuts.
    const w = 1.1;
    const acc = want.clone().sub(this.pos).multiplyScalar(w * w).addScaledVector(this.vel, -2 * w);
    this.vel.addScaledVector(acc, dt);
    this.pos.addScaledVector(this.vel, dt);
    this.pos.z = Math.max(this.pos.z, this.ground(this.pos.x, this.pos.y) + 0.25);
    // Frame a little ahead of the fly, and lag behind it slightly.
    const lead = settled ? 0 : 0.5;
    const target = new THREE.Vector3(hero.x - Math.sin(yaw) * lead, hero.y + Math.cos(yaw) * lead, hero.z);
    this.look.lerp(target, 1 - Math.exp(-dt * 2.2));
    this.fov += (fov - this.fov) * (1 - Math.exp(-dt * 0.8));
    return { pos: this.pos, look: this.look, fov: this.fov };
  }
}
