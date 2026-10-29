// The breeze ("the cool of the day", Gen. 3:8: literally "the wind of the
// day"). Three layers, summed:
//   prevailing  a weak drift whose direction veers slowly
//   eddies      curl noise: swirling, divergence-free, always changing
//   puffs       individual gusts that spawn, travel, swell and die out, each
//               with its own heading, size, strength and swirl; they arrive
//               in flurries separated by lulls
// The shaders evaluate the same field (WIND_GLSL), so a puff seen crossing the
// grass is the puff that tosses the flies and roughens the water there.
import * as THREE from "three";

export const MAX_PUFFS = 12;

export const WIND_GLSL = /* glsl */ `
  uniform vec2 uWind;                        // prevailing drift, b-space XY
  uniform vec4 uPuffA[${MAX_PUFFS}];         // centre xy, radius, current strength
  uniform vec4 uPuffB[${MAX_PUFFS}];         // heading xy, swirl, -
  vec2 eddy(vec2 p) {
    float e = 0.07;
    vec3 q = vec3(p * 0.11, uTime * 0.13);
    float a = vnoise(q + vec3(0.0, e, 0.0)), b = vnoise(q - vec3(0.0, e, 0.0));
    float c = vnoise(q + vec3(e, 0.0, 0.0)), d = vnoise(q - vec3(e, 0.0, 0.0));
    return vec2(a - b, d - c) / (2.0 * e) * 0.35;   // curl of the noise
  }
  vec2 windAt(vec2 p) {
    float lw = length(uWind);
    vec2 w = uWind + eddy(p) * (0.25 + 0.5 * lw);
    for (int i = 0; i < ${MAX_PUFFS}; i++) {
      vec4 a = uPuffA[i];
      if (a.w <= 0.0) continue;
      vec2 d = p - a.xy;
      float f = exp(-dot(d, d) / (a.z * a.z));
      vec4 b = uPuffB[i];
      w += a.w * f * (b.xy + b.z * vec2(-d.y, d.x) / a.z);
    }
    return w;
  }
`;

// JavaScript twin of the GLSL noise.
const fract = (x) => x - Math.floor(x);
const hash = (x, y, z) => fract(Math.sin(x * 12.9898 + y * 78.233 + z * 37.719) * 43758.5453);
function vnoise(x, y, z) {
  const i = Math.floor(x), j = Math.floor(y), k = Math.floor(z);
  let u = x - i, v = y - j, w = z - k;
  u = u * u * (3 - 2 * u); v = v * v * (3 - 2 * v); w = w * w * (3 - 2 * w);
  const L = (a, b, t) => a + (b - a) * t;
  return L(L(L(hash(i, j, k), hash(i + 1, j, k), u), L(hash(i, j + 1, k), hash(i + 1, j + 1, k), u), v),
           L(L(hash(i, j, k + 1), hash(i + 1, j, k + 1), u), L(hash(i, j + 1, k + 1), hash(i + 1, j + 1, k + 1), u), v), w);
}

export const STRENGTHS = { calm: 0.3, breeze: 1.0, strong: 1.9 };

export class Wind {
  constructor(uniforms) {
    this.uniforms = uniforms;
    // Shaders bind these uniform objects when they compile; keep them, update in place.
    uniforms.uWind ??= { value: new THREE.Vector2() };
    uniforms.uPuffA ??= { value: Array.from({ length: MAX_PUFFS }, () => new THREE.Vector4()) };
    uniforms.uPuffB ??= { value: Array.from({ length: MAX_PUFFS }, () => new THREE.Vector4()) };
    this.level = "breeze";
    this.strength = STRENGTHS.breeze;
    this.puffs = [];
    this.t = 0;
    this.focus = new THREE.Vector2();
  }

  cycle() {
    const names = Object.keys(STRENGTHS);
    this.level = names[(names.indexOf(this.level) + 1) % names.length];
    return this.level;
  }

  // focus: b-space XY around which gusts are spawned (the hero).
  update(dt, t, focus) {
    this.t = t;
    if (focus) this.focus.set(focus.x, focus.y);
    this.strength += (STRENGTHS[this.level] - this.strength) * (1 - Math.exp(-dt * 0.6));
    const S = this.strength;
    // Prevailing drift: weak, veering, breathing.
    const angle = 0.45 + 0.6 * Math.sin(t * 0.023) + 0.25 * Math.sin(t * 0.071 + 1.3);
    this.angle = angle;
    const base = S * 0.3 * (0.6 + 0.4 * vnoise(t * 0.05, 3.1, 0));
    this.uniforms.uWind.value.set(Math.cos(angle) * base, Math.sin(angle) * base);

    // Cadence: flurries and lulls.
    const mood = vnoise(t * 0.06, 7.7, 1.3);
    const rate = S * (0.15 + 2.6 * Math.pow(mood, 2.2));           // puffs per second
    if (Math.random() < rate * dt && this.puffs.length < MAX_PUFFS) this.spawn(angle, S);

    for (const p of this.puffs) {
      p.age += dt;
      p.x += p.vx * dt; p.y += p.vy * dt;
      const k = p.age / p.life;
      // Quick swell, long fade.
      p.env = k < 0.25 ? Math.sin((k / 0.25) * Math.PI / 2) ** 2 : Math.cos(((k - 0.25) / 0.75) * Math.PI / 2) ** 2;
    }
    this.puffs = this.puffs.filter((p) => p.age < p.life);
    const A = this.uniforms.uPuffA.value, B = this.uniforms.uPuffB.value;
    for (let i = 0; i < MAX_PUFFS; i++) {
      const p = this.puffs[i];
      if (!p) { A[i].set(0, 0, 1, 0); continue; }
      A[i].set(p.x, p.y, p.r, p.peak * p.env);
      B[i].set(p.dx, p.dy, p.swirl, 0);
    }
  }

  spawn(angle, S) {
    const heading = angle + (Math.random() - 0.5) * 1.6;          // each gust its own way
    const dx = Math.cos(heading), dy = Math.sin(heading);
    const speed = 2 + Math.random() * 5;
    // Start upwind of somewhere near the focus, so gusts sweep through it.
    const a = Math.random() * 6.283, d = Math.random() * 18;
    const life = 2.5 + Math.random() * 6;
    this.puffs.push({
      x: this.focus.x + Math.cos(a) * d - dx * speed * life * 0.4,
      y: this.focus.y + Math.sin(a) * d - dy * speed * life * 0.4,
      vx: dx * speed, vy: dy * speed, dx, dy,
      r: 2 + Math.random() * 6,
      peak: S * (0.6 + Math.random() * 1.6),
      swirl: (Math.random() - 0.5) * 1.6,
      life, age: 0, env: 0,
    });
  }

  eddy(x, y) {
    const e = 0.07, qx = x * 0.11, qy = y * 0.11, qz = this.t * 0.13;
    const a = vnoise(qx, qy + e, qz), b = vnoise(qx, qy - e, qz);
    const c = vnoise(qx + e, qy, qz), d = vnoise(qx - e, qy, qz);
    return [(a - b) / (2 * e) * 0.35, (d - c) / (2 * e) * 0.35];
  }

  // b-space wind vector at a point (the same field the shaders see).
  at(x, y) {
    const W = this.uniforms.uWind.value, lw = Math.hypot(W.x, W.y);
    const [ex, ey] = this.eddy(x, y);
    let wx = W.x + ex * (0.25 + 0.5 * lw), wy = W.y + ey * (0.25 + 0.5 * lw);
    for (const p of this.puffs) {
      const ddx = x - p.x, ddy = y - p.y;
      const f = Math.exp(-(ddx * ddx + ddy * ddy) / (p.r * p.r)) * p.peak * p.env;
      wx += f * (p.dx + p.swirl * -ddy / p.r);
      wy += f * (p.dy + p.swirl * ddx / p.r);
    }
    return new THREE.Vector2(wx, wy);
  }
}
