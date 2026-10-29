// The constellation: 36,000 real neurons at their FlyWire positions and the
// 26,000 strongest synapses among them. Each neuron's activity arrives from
// the live brain in units of its own variability (σ). Active neurons spark at
// a rate set by that activity; silenced ones sink to a cool, dim blue; pulses
// run along synapses from active presynaptic cells, gold for excitatory,
// violet for inhibitory.
import * as THREE from "three";

const TEX_W = 256;

// Regions view: [colour, label, which points place the label]. Paired regions
// get a label each; bilateral ones are labelled on one hemisphere ("+x" or
// "-x") so labels spread out instead of piling up on the midline.
const REGION_STYLE = {
  "retina L": ["#f3efe6", null], "retina R": ["#f3efe6", null],
  "optic lobe L": ["#e9c46a", "Optic lobe"], "optic lobe R": ["#e9c46a", "Optic lobe"],
  "visual projection L": ["#f2a65a", "Visual projection"], "visual projection R": ["#f2a65a", "Visual projection"],
  "mushroom body": ["#e5839b", "Mushroom body", "-x"], "central complex": ["#a98cf0", "Central complex"],
  "lateral horn": ["#5fc6b8", "Lateral horn", "+x"], "antennal lobe": ["#8fcf7a", "Antennal lobe", "+x"],
  "central brain": ["#8aa0c8", null], "descending": ["#ef7f6a", "Descending"],
  "neuroendocrine": ["#f0b37e", "Neuroendocrine"], "sensory": ["#7d8696", null],
};

function b64(str, Type) {
  const bin = atob(str);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
}

export class Constellation {
  constructor(el, data) {
    this.el = el;
    this.n = data.n;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(devicePixelRatio);
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20);
    this.dist = 3.25;
    this.yaw = 0.4; this.pitch = 0.12;

    // Activity texture: smoothed σ per neuron.
    const rows = Math.ceil(this.n / TEX_W);
    this.target = new Float32Array(TEX_W * rows);
    this.smooth = new Float32Array(TEX_W * rows);
    this.tex = new THREE.DataTexture(this.smooth, TEX_W, rows, THREE.RedFormat, THREE.FloatType);
    this.tex.needsUpdate = true;
    this.regions = data.regions;
    this.uniforms = {
      uAct: { value: this.tex }, uTime: { value: 0 }, uPR: { value: devicePixelRatio }, uTexW: { value: TEX_W },
      uMode: { value: 0 },                                   // eased: 0 = neurons, 1 = regions
      uRegionColor: { value: Array.from({ length: 16 }, (_, i) =>
        new THREE.Color(REGION_STYLE[this.regions[i]]?.[0] ?? "#888888")) },
      uRegionLevel: { value: new Array(16).fill(0.2) },
    };
    this.mode = "regions";
    this.modeT = 1;

    const lookup = /* glsl */ `
      uniform sampler2D uAct; uniform float uTexW;
      float act(float id) {
        ivec2 c = ivec2(int(mod(id, uTexW)), int(floor(id / uTexW)));
        return texelFetch(uAct, c, 0).r;
      }
      float hash1(float x) { return fract(sin(x * 91.3458) * 47453.5453); }`;

    // Neurons
    const pos = b64(data.pos, Float32Array);
    const judge = b64(data.judge, Int8Array);
    const ids = new Float32Array(this.n).map((_, i) => i);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("nid", new THREE.BufferAttribute(ids, 1));
    const rid = b64(data.region, Uint8Array);
    g.setAttribute("rid", new THREE.BufferAttribute(Float32Array.from(rid), 1));
    // Judges: +1 for neurons whose firing means good, -1 for bad, 0 otherwise.
    const weights = { reward: 1, approach: 1, satiety: 1, punishment: -1, avoidance: -1, escape: -1 };
    g.setAttribute("judge", new THREE.BufferAttribute(Float32Array.from(judge, (j) =>
      j < 0 ? 0 : weights[data.judge_names[j]]), 1));
    this.points = new THREE.Points(g, new THREE.ShaderMaterial({
      uniforms: this.uniforms, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      vertexShader: lookup + /* glsl */ `
        uniform float uTime, uPR, uMode;
        uniform vec3 uRegionColor[16];
        uniform float uRegionLevel[16];
        attribute float nid, judge, rid;
        varying float vA, vNeg, vSpike, vJudge, vLevel;
        varying vec3 vRegion;
        void main() {
          int r = int(rid + 0.5);
          vRegion = uRegionColor[r];
          vLevel = uRegionLevel[r];
          float z = act(nid);
          vA = clamp(z / 3.0, 0.0, 1.0);
          vNeg = clamp(-z / 3.0, 0.0, 1.0);
          // Sparks: a rate set by activity, a random phase per neuron.
          float rate = 0.5 + 13.0 * vA;
          float ph = fract(uTime * rate + hash1(nid));
          vSpike = vA > 0.04 ? exp(-ph * 9.0) : 0.0;
          vJudge = judge;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mv;
          float size = mix(1.0 + 1.8 * vA + 3.2 * vSpike * sqrt(vA) + (judge != 0.0 ? 1.4 : 0.0),
                           1.2 + 1.3 * vLevel, uMode);
          gl_PointSize = uPR * size * (5.5 / -mv.z);
        }`,
      fragmentShader: /* glsl */ `
        uniform float uMode;
        varying float vA, vNeg, vSpike, vJudge, vLevel;
        varying vec3 vRegion;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          if (d > 0.5) discard;
          float core = smoothstep(0.5, 0.0, d);
          vec3 rest = vec3(0.55, 0.57, 0.63);
          vec3 fire = vJudge < 0.0 ? vec3(1.0, 0.45, 0.4) : vec3(1.0, 0.8, 0.45);
          vec3 c = mix(rest, fire, smoothstep(0.1, 0.6, vA));
          c = mix(c, vec3(1.0), vSpike * vA);
          c = mix(c, vec3(0.45, 0.55, 0.95), vNeg);
          float a = core * (0.12 + 0.4 * vA + vSpike * sqrt(vA) + 0.25 * vNeg);
          // Regions: each neuron wears its region's hue, lit by the region's activity.
          vec3 rc = mix(vRegion * 0.55, mix(vRegion, vec3(1.0), 0.25 * vLevel), vLevel);
          // Points add up, and the optic lobes are dense: keep each faint so
          // a busy region glows in its colour instead of saturating to white.
          float ra = core * (0.1 + 0.28 * vLevel * vLevel);
          c = mix(c, rc, uMode);
          a = mix(a, ra, uMode);
          gl_FragColor = vec4(c * a, a);
        }`,
    }));

    // Synapses: one segment per synapse, a pulse travelling pre -> post.
    const pairs = b64(data.edges, Uint32Array);
    const sign = b64(data.edge_sign, Int8Array);
    const w = b64(data.edge_w, Uint8Array);
    const E = pairs.length / 2;
    const lp = new Float32Array(E * 6), pre = new Float32Array(E * 2), end = new Float32Array(E * 2),
      sg = new Float32Array(E * 2), ew = new Float32Array(E * 2), eh = new Float32Array(E * 2);
    for (let e = 0; e < E; e++) {
      const a = pairs[2 * e], b = pairs[2 * e + 1];
      lp.set(pos.subarray(a * 3, a * 3 + 3), e * 6);
      lp.set(pos.subarray(b * 3, b * 3 + 3), e * 6 + 3);
      pre[2 * e] = pre[2 * e + 1] = a;
      end[2 * e] = 0; end[2 * e + 1] = 1;
      sg[2 * e] = sg[2 * e + 1] = sign[e];
      ew[2 * e] = ew[2 * e + 1] = w[e] / 11;
      eh[2 * e] = eh[2 * e + 1] = Math.random();
    }
    const lg = new THREE.BufferGeometry();
    lg.setAttribute("position", new THREE.BufferAttribute(lp, 3));
    lg.setAttribute("pre", new THREE.BufferAttribute(pre, 1));
    lg.setAttribute("endp", new THREE.BufferAttribute(end, 1));
    lg.setAttribute("esign", new THREE.BufferAttribute(sg, 1));
    lg.setAttribute("ew", new THREE.BufferAttribute(ew, 1));
    lg.setAttribute("eh", new THREE.BufferAttribute(eh, 1));
    this.lines = new THREE.LineSegments(lg, new THREE.ShaderMaterial({
      uniforms: this.uniforms, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      vertexShader: lookup + /* glsl */ `
        attribute float pre, endp, esign, ew, eh;
        varying float vE, vA, vS, vW, vH;
        void main() {
          vA = clamp(act(pre) / 3.0, 0.0, 1.0);
          vE = endp; vS = esign; vW = ew; vH = eh;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform float uTime, uMode;
        varying float vE, vA, vS, vW, vH;
        void main() {
          float speed = 0.6 + 2.2 * vA;
          float p = fract(uTime * speed + vH * 7.0);
          float pulse = exp(-abs(vE - p) * 22.0) * vA;
          vec3 c = vS > 0.0 ? vec3(1.0, 0.78, 0.42) : vec3(0.62, 0.5, 1.0);
          float a = (0.022 + 0.05 * vW * vA + 0.9 * pulse) * (1.0 - 0.85 * uMode);
          gl_FragColor = vec4(c * a, a);
        }`,
    }));
    this.buildLabels(pos, rid);
    this.group = new THREE.Group();
    this.group.add(this.lines, this.points);
    this.scene.add(this.group);

    let drag = null;
    el.addEventListener("pointerdown", (e) => { drag = [e.clientX, e.clientY, this.yaw, this.pitch]; el.setPointerCapture(e.pointerId); });
    el.addEventListener("pointermove", (e) => {
      if (!drag) return;
      this.yaw = drag[2] + (e.clientX - drag[0]) * 0.008;
      this.pitch = Math.max(-1.3, Math.min(1.3, drag[3] + (e.clientY - drag[1]) * 0.008));
    });
    el.addEventListener("pointerup", () => (drag = null));
    el.addEventListener("wheel", (e) => { this.dist = Math.max(1.0, Math.min(5, this.dist * (1 + e.deltaY * 0.001))); e.preventDefault(); }, { passive: false });
    this.dragging = () => !!drag;
  }

  // Floating labels at each labelled region's centre.
  buildLabels(pos, rid) {
    const sums = {};
    for (let i = 0; i < this.n; i++) {
      const style = REGION_STYLE[this.regions[rid[i]]];
      if (!style?.[1]) continue;
      if (style[2] === "+x" && pos[3 * i] < 0) continue;
      if (style[2] === "-x" && pos[3 * i] > 0) continue;
      const s = (sums[rid[i]] ??= { p: new THREE.Vector3(), n: 0 });
      s.p.x += pos[3 * i]; s.p.y += pos[3 * i + 1]; s.p.z += pos[3 * i + 2]; s.n++;
    }
    this.labels = Object.entries(sums).map(([r, s]) => {
      const [col, name] = REGION_STYLE[this.regions[r]];
      const el = document.createElement("div");
      el.className = "region-label";
      el.innerHTML = `<i style="background:${col}"></i><span>${name}</span><b><u style="background:${col}"></u></b>`;
      this.el.appendChild(el);
      return { el, bar: el.querySelector("u"), centre: s.p.divideScalar(s.n), ids: [Number(r)] };
    });
  }

  setMode(m) { this.mode = m; }

  // levels: { regionName: 0..1 } from the live brain.
  setRegionLevels(levels) {
    this.regions.forEach((r, i) => { if (r in levels) this.uniforms.uRegionLevel.value[i] = levels[r]; });
  }

  // z: Int8Array of activity in σ x 40, one per neuron.
  setActivity(z) {
    for (let i = 0; i < this.n; i++) this.target[i] = z[i] / 40;
  }

  render(t, dt) {
    const k = 1 - Math.exp(-dt * 8);
    for (let i = 0; i < this.n; i++) this.smooth[i] += (this.target[i] - this.smooth[i]) * k;
    this.tex.needsUpdate = true;
    this.uniforms.uTime.value = t;
    if (!this.dragging()) this.yaw += dt * 0.06;
    this.group.rotation.set(this.pitch, this.yaw, 0);
    const w = this.el.clientWidth, h = this.el.clientHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.position.set(0, 0, this.dist);
    this.camera.updateProjectionMatrix();
    this.renderer.render(this.scene, this.camera);
    // Ease between views; place labels over their regions.
    this.modeT += ((this.mode === "regions" ? 1 : 0) - this.modeT) * (1 - Math.exp(-dt * 5));
    this.uniforms.uMode.value = this.modeT;
    const v = new THREE.Vector3();
    for (const L of this.labels ?? []) {
      v.copy(L.centre).applyMatrix4(this.group.matrixWorld).project(this.camera);
      L.el.style.opacity = (this.modeT * (v.z < 1 ? 1 : 0)).toFixed(2);
      L.el.style.transform = `translate(${((v.x + 1) / 2 * w).toFixed(1)}px, ${((1 - v.y) / 2 * h).toFixed(1)}px)`;
      const lv = Math.max(...L.ids.map((i) => this.uniforms.uRegionLevel.value[i]));
      L.bar.style.width = `${Math.round(lv * 100)}%`;
    }
  }
}
