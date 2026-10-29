// Paradise, live. The meadow renders in real time; several times a second the
// hero's view is sent to the brain server (python -m paradise.live serve),
// whose reply drives the meadow's light, the brain panels, and
// the hero's own steering.
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { loadMeadow, KuwaharaShader, WATER_LEVEL } from "./meadow.js";
import { Flight } from "./flight.js";
import { Eye } from "./eye.js";
import { Constellation } from "./constellation.js";
import { Wind } from "./wind.js";
import { Raster, Traces, drawArch } from "./mind.js";
import { Director } from "./camera.js";

// The brain server (python -m paradise.live serve). Where it is: ?brain=wss://...,
// else the public address in brain.json (written when it is served through a
// tunnel, see paradise/tunnel.py), else this machine. On failure, try the next.
const LOCAL_BRAIN = "ws://127.0.0.1:8766";
const published = await fetch("brain.json", { cache: "no-store" }).then((r) => (r.ok ? r.json() : {})).catch(() => ({}));
const BRAINS = [new URLSearchParams(location.search).get("brain"), published.url, LOCAL_BRAIN].filter(Boolean);
const BRAIN_URL = BRAINS[0];
let brainTry = 0;
const $ = (id) => document.getElementById(id);
const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"];
const params = new URLSearchParams(location.search);

// ---------------------------------------------------------------- renderer & scene
const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance", preserveDrawingBuffer: params.has("shot") });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
// AgX, as in the Blender renders.
renderer.toneMapping = THREE.AgXToneMapping ?? THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.95;
$("view").appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.02, 1500);

const { world, uniforms, activity, groundHeight, mirror, spots } = await loadMeadow(scene);
const sunB = uniforms.uSun.value;
const sun = new THREE.DirectionalLight(0xffdcaa, 2.4);
sun.position.set(sunB.x, sunB.z, -sunB.y).multiplyScalar(50);
scene.add(sun, new THREE.HemisphereLight(0xfff4e0, 0x8a7a60, 1.1));

const flight = new Flight(world, spots, groundHeight);
const wind = new Wind(uniforms);
const eye = new Eye(renderer);
const constData = await fetch("data/constellation.json").then((r) => r.json());
const constellation = new Constellation($("cons"), constData);
const judge = Int8Array.from(atob(constData.judge), (c) => (c.charCodeAt(0) << 24) >> 24);
const raster = new Raster($("raster"), judge, constData.judge_names);
const traces = new Traces($("traces"));
let latestZ = null, latestLevels = null;

// Heaven through an arched window; the fly's mind in the margins either side.
function frame() {
  const { x0 } = drawArch($("arch"));
  document.documentElement.style.setProperty("--mw", `${x0}px`);
}
addEventListener("resize", frame);
frame();
const N = constellation.n;

// Regions view: brightness per region on a log scale (the eye is loud, the
// deep brain quiet), plus each region's own swings around its recent mean.
const regionStats = {};
function regionLevels(regions) {
  const out = {};
  for (const [name, x] of Object.entries(regions)) {
    const lg = Math.log10(x + 1e-9);
    const s = (regionStats[name] ??= { m: lg, v: 0.01 });
    s.m += (lg - s.m) * 0.05;
    s.v += ((lg - s.m) ** 2 - s.v) * 0.05;
    const base = (lg + 6) / (Math.log10(3e-2) + 6);
    out[name] = Math.min(1, Math.max(0.04, 0.2 + 0.55 * base + 0.25 * (lg - s.m) / Math.sqrt(s.v + 1e-6)));
  }
  return out;
}

// The hero's search path, drawn faintly.
const trailGeo = new THREE.BufferGeometry();
trailGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(600 * 3), 3));
const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.22 }));
trail.frustumCulled = false;
world.add(trail);

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.5, 0.7, 0.85);
composer.addPass(bloom);
const paint = new ShaderPass(KuwaharaShader);
composer.addPass(paint);
composer.addPass(new OutputPass());

function resize() {
  const w = $("view").clientWidth || innerWidth, h = $("view").clientHeight || innerHeight;
  renderer.setSize(w, h);
  composer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  const pr = renderer.getPixelRatio();
  paint.uniforms.resolution.value.set(w * pr, h * pr);
}
addEventListener("resize", resize);
resize();

// ---------------------------------------------------------------- camera
// A director frames the hero (camera.js). Drag to take the camera; it hands
// back to the director a few seconds after you let go. Scroll to zoom.
const director = new Director(groundHeight, WATER_LEVEL);
let lookYaw = 0, lookPitch = 0.3, drag = null, manualUntil = -1;
const view = $("view");
view.addEventListener("pointerdown", (e) => {
  // Take over from wherever the director has the camera.
  const off = new THREE.Vector3().subVectors(director.pos, flight.pos);
  lookYaw = Math.atan2(off.x, -off.y) - flight.yaw;
  lookPitch = Math.atan2(off.z, Math.hypot(off.x, off.y));
  drag = [e.clientX, e.clientY, lookYaw, lookPitch];
  view.setPointerCapture(e.pointerId);
});
view.addEventListener("pointermove", (e) => {
  if (!drag) return;
  lookYaw = drag[2] - (e.clientX - drag[0]) * 0.006;
  lookPitch = Math.max(-0.2, Math.min(1.35, drag[3] + (e.clientY - drag[1]) * 0.004));
  manualUntil = performance.now() + 8000;
});
view.addEventListener("pointerup", () => (drag = null));
view.addEventListener("wheel", (e) => {
  director.zoom = Math.max(0.4, Math.min(4, director.zoom * (1 + e.deltaY * 0.001)));
  e.preventDefault();
}, { passive: false });
const b2t = (v) => new THREE.Vector3(v.x, v.z, -v.y);      // b-space -> three
function updateCamera(dt, t) {
  const shot = director.update(dt, t, flight.pos, flight.yaw, flight.mode === "guide" ? "wander" : flight.state);
  if (performance.now() < manualUntil) {
    // Orbit the hero where the viewer dragged it.
    const d = 4.2 * director.zoom, a = flight.yaw + lookYaw;
    director.pos.set(flight.pos.x + Math.sin(a) * d * Math.cos(lookPitch), flight.pos.y - Math.cos(a) * d * Math.cos(lookPitch),
      flight.pos.z + d * Math.sin(lookPitch));
    director.vel.set(0, 0, 0);
  }
  camera.position.copy(b2t(shot.pos));
  camera.lookAt(b2t(shot.look));
  if (Math.abs(camera.fov - shot.fov) > 0.01) { camera.fov = shot.fov; camera.updateProjectionMatrix(); }
  // ?cam=x,y,z&at=x,y,z (b-space): a fixed camera, for checking the scene.
  if (params.has("cam")) {
    const [cx, cy, cz] = params.get("cam").split(",").map(Number);
    const [ax, ay, az] = (params.get("at") || "0,0,0").split(",").map(Number);
    camera.position.copy(b2t(new THREE.Vector3(cx, cy, cz)));
    camera.lookAt(b2t(new THREE.Vector3(ax, ay, az)));
  }
}

// ---------------------------------------------------------------- brain link
let ws, awaiting = false, meta = null, lastSent = 0, glow = 2, pleasure = 0;
const FEELINGS = [[6, "bliss"], [3, "pleased"], [1, "content"], [-1, "indifferent"], [-3, "uneasy"], [-Infinity, "afraid"]];
function connect() {
  const url = BRAINS[brainTry % BRAINS.length];
  let opened = false;
  ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { opened = true; awaiting = false; };
  ws.onclose = () => {
    if (!opened) brainTry++;          // unreachable: try the next address
    setTimeout(connect, 1500);
  };
  ws.onmessage = (e) => {
    if (typeof e.data === "string") { meta = JSON.parse(e.data); return; }
    const z = new Int8Array(e.data, 0, N);
    activity.image.data.set(new Uint8Array(e.data, N, 128 * 64));
    activity.needsUpdate = true;
    constellation.setActivity(z);
    latestLevels = regionLevels(meta.regions);
    constellation.setRegionLevels(latestLevels);
    latestZ = z;
    flight.sense(meta.valence);
    awaiting = false;
  };
}
connect();

function sendView(now) {
  if (!ws || ws.readyState !== 1 || awaiting || now - lastSent < 40) return;
  const eyeB = flight.eye();
  uniforms.uFlyView.value = 1;
  uniforms.uCam.value.copy(eyeB);
  flight.flyView(true);
  trail.visible = false;
  const rgb = eye.capture(scene, world.localToWorld(eyeB.clone()), flight.bodyRotation());
  flight.flyView(false);
  trail.visible = true;
  uniforms.uFlyView.value = 0;
  ws.send(rgb);
  awaiting = true;
  lastSent = now;
}

// ---------------------------------------------------------------- controls
function setMode(m) {
  flight.guide(m === "guide");
  $("seek").classList.toggle("on", m === "seek");
  $("guide").classList.toggle("on", m === "guide");
}
$("seek").onclick = () => setMode("seek");
$("guide").onclick = () => setMode("guide");
addEventListener("keydown", (e) => {
  const k = e.key.toLowerCase();
  if (k === "g") setMode(flight.mode === "guide" ? "seek" : "guide");
  else if (k === "r") { setMode("seek"); const a = Math.random() * 6.28; flight.pos.set(9 * Math.cos(a), 9 * Math.sin(a), 1.6); flight.trail.length = 0; }
  else if (k === "h") document.body.classList.toggle("hide-hud");
  else if (k === "b") wind.cycle();
});
if (params.get("mode") === "guide") setMode("guide");

// ?validate: put the hero's eye exactly on the heaven pose and log the brain's
// verdict. The server saves the view it receives to out/live_eye.png, to be
// compared with out/heaven_current.png (r = 0.988 when last checked).
if (params.has("validate")) {
  setMode("guide");
  flight.yaw = 0;
  flight.pos.set(0, 0, 1.2).sub(flight.eye().sub(flight.pos));   // eye = pos + forward * offset
  const eyeB = flight.eye();
  uniforms.uFlyView.value = 1; uniforms.uCam.value.copy(eyeB); flight.flyView(true); trail.visible = false;
  const rgb = eye.capture(scene, world.localToWorld(eyeB.clone()), flight.bodyRotation());
  flight.flyView(false); trail.visible = true; uniforms.uFlyView.value = 0;
  console.log(`VALIDATE eye=${eyeB.toArray().map((x) => x.toFixed(3))}`);
  const v = new WebSocket(BRAIN_URL); v.binaryType = "arraybuffer";
  v.onopen = () => v.send(rgb);
  v.onmessage = (e) => { if (typeof e.data === "string") console.log(`VALIDATE valence=${JSON.parse(e.data).valence}`); };
}

// ---------------------------------------------------------------- loop
let last = performance.now();
const t0 = last;
function tick(now) {
  const dt = Math.min(0.1, Math.max(0, (now - last) / 1000));
  last = now;
  const t = (now - t0) / 1000;
  wind.update(dt, t, flight.pos);
  flight.update(dt, t, wind);
  // Trail
  const arr = trailGeo.attributes.position.array;
  flight.trail.forEach((p, i) => arr.set([p.x, p.y, p.z], i * 3));
  trailGeo.setDrawRange(0, flight.trail.length);
  trailGeo.attributes.position.needsUpdate = true;

  sendView(now);
  updateCamera(dt, t);
  // Art view uniforms.
  uniforms.uCam.value.copy(world.worldToLocal(camera.position.clone()));
  // Camera axes in b-space, for camera-facing foliage.
  const t2b = (v) => new THREE.Vector3(v.x, -v.z, v.y);
  uniforms.uCamRight.value.copy(t2b(new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0)));
  uniforms.uCamUp.value.copy(t2b(new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1)));
  uniforms.uHeroPos.value.copy(flight.eye());
  uniforms.uHeroInv.value.copy(flight.bodyRotation()).transpose();
  const wantGlow = 1.6 + 0.15 * Math.min(Math.max(meta ? meta.valence : 0, 0), 8);
  glow += (wantGlow - glow) * (1 - Math.exp(-dt * 2));
  uniforms.uGlow.value = glow;
  uniforms.uTime.value = t;
  mirror.render(renderer, scene, camera);
  composer.render();
  constellation.render(t, dt);
  // The animal mind, as instruments.
  // Pleasure: the brain's valence (good minus bad populations), shown as one
  // meter from aversion to pleasure and a word for it.
  if (meta) {
    pleasure += (meta.valence - pleasure) * (1 - Math.exp(-dt * 3));
    const p = Math.max(-1, Math.min(1, pleasure / 8));
    const fill = $("pFill").style;
    fill.left = `${50 + Math.min(0, p) * 50}%`;
    fill.width = `${Math.abs(p) * 50}%`;
    fill.background = p >= 0 ? "#e9c46a" : "#829bd7";
    $("pWord").textContent = FEELINGS.find(([v]) => pleasure >= v)[1];
  }
  raster.update(dt, latestZ);
  traces.update(dt, latestLevels);
  const DOING = { wander: "wandering", toEat: "going to feed", eat: "feeding", toRest: "going to rest", rest: "resting" };
  $("day").textContent = `Day ${ROMAN[Math.floor(t / 60) % 7]} · ${flight.mode === "guide" ? "guided" : DOING[flight.state]}`;
  requestAnimationFrame(tick);
}
if (params.has("shot")) $("loading").remove(); else $("loading").classList.add("gone");
requestAnimationFrame(tick);
