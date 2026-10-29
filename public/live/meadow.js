// The painterly Eden, drawn in real time. Geometry comes from Blender
// (blender/export_meadow.py) in Blender's Z-up world; everything lives inside
// a group rotated into three.js's Y-up frame, and shaders work in Blender
// coordinates ("b-space") throughout.
//
// Colour comes only from the extracted heaven panorama, sampled in the
// direction of each blade's root as seen from the fly's eye. With
// uFlyView = 1 every surface instead shows the raw panorama at the shading
// point, unlit: what the hero's brain is shown.
import * as THREE from "three";
import { terrainWarp, shoreRocks, neuronTrees } from "./shapes.js";
import { WIND_GLSL, MAX_PUFFS } from "./wind.js";

export const EYE = new THREE.Vector3(0, 0, 1.2);   // b-space

const KIND = { ground: 0, grass: 1, flower: 2, leaf: 3, fruit: 4, mote: 5, rock: 6, canopy: 7, branch: 8, foliage: 9 };

// Matches eden.py: sun rotation (80°, 0°, 165°) applied to Blender's -Z.
function sunDirection() {
  const e = new THREE.Euler(THREE.MathUtils.degToRad(80), 0, THREE.MathUtils.degToRad(165), "XYZ");
  return new THREE.Vector3(0, 0, 1).applyEuler(e).normalize();   // toward the sun
}

const common = /* glsl */ `
  uniform sampler2D uHeaven, uActivity;
  uniform vec3 uEye, uSun, uCam, uHeroPos;
  uniform mat3 uHeroInv;
  uniform float uFlyView, uGlow, uContrast, uSaturation, uTime;
  uniform vec3 uCamRight, uCamUp;
  const float PI = 3.14159265;
  vec2 equirect(vec3 d) {
    d = normalize(d);
    return vec2(atan(d.x, d.y) / (2.0 * PI) + 0.5, asin(clamp(d.z, -1.0, 1.0)) / PI + 0.5);
  }
  vec3 grade(vec3 raw) {
    vec3 c = 0.5 + uContrast * (raw - 0.5);
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    c = clamp(mix(vec3(l), c, uSaturation), 0.0, 1.0);
    return pow(c, vec3(2.2));
  }
  float hash(vec3 p) { return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453); }
  float vnoise(vec3 p) {
    vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash(i), hash(i + vec3(1,0,0)), f.x), mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
               mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x), mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
  }
` + WIND_GLSL;

function heavenMaterial(kind, uniforms) {
  return new THREE.ShaderMaterial({
    uniforms,
    defines: { KIND: KIND[kind] },
    side: kind === "ground" || kind === "fruit" ? THREE.FrontSide : THREE.DoubleSide,
    vertexShader: common + /* glsl */ `
      attribute vec3 root; attribute float tip; attribute vec2 card;
      varying vec3 vPos, vNormal, vRoot; varying float vTip; varying vec2 vCard;
      void main() {
        vec3 p = position;
        // The breeze. Things don't just lean: each is pushed, then swings
        // back past rest at its own natural frequency, in waves that travel
        // with the gust; a smaller wobble runs crosswise. Motion grows toward
        // the free end of each thing.
        #if KIND == 1 || KIND == 2 || KIND == 5 || KIND == 8 || KIND == 9
          vec2 w = windAt(root.xy);
          float lw = length(w);
          vec2 wd = w / max(lw, 1e-4), side = vec2(-wd.y, wd.x);
          float h = hash(root);
        #endif
        #if KIND == 1 || KIND == 2
          float bend = tip * tip;
          float freq = 1.3 + 1.8 * h;                          // stiffness varies blade to blade
          float phase = h * 6.283 + dot(root.xy, wd) * 1.1;     // sway travels downwind
          float sway = sin(uTime * freq - phase);
          float wobble = sin(uTime * freq * 2.37 + h * 17.0);
          vec2 off = w * 0.32 + wd * lw * 0.26 * sway + side * lw * 0.12 * wobble
                   + vec2(sin(uTime * 0.8 + h * 9.0), cos(uTime * 0.7 + h * 5.0)) * 0.02;   // never quite still
          p.xy += off * bend;
          p.z -= dot(off, off) * 0.3 * bend;                   // tips bow as they bend
        #elif KIND == 8
          float sway = sin(uTime * (0.7 + 0.5 * h) - dot(root.xy, wd) * 0.6);
          float bend = pow(tip, 1.6);                              // the whole crown leans; twigs most
          p.xy += (w * 0.38 + wd * lw * 0.28 * sway) * bend;
          p.z -= dot(w, w) * 0.04 * bend;
        #elif KIND == 9
          float sway = sin(uTime * (0.7 + 0.5 * h) - dot(root.xy, wd) * 0.6);
          vec2 flutter = vec2(sin(uTime * (3.0 + 2.0 * h) + h * 6.3), cos(uTime * (2.4 + 1.7 * h) + h * 4.1));
          p.xy += w * 0.38 + wd * lw * 0.28 * sway + flutter * 0.15 * lw;   // matches the twig tips
          p.z += sin(uTime * (2.2 + h) + h * 5.0) * 0.1 * lw - dot(w, w) * 0.04;
        #elif KIND == 5
          // Pollen rides the air, looping in the eddies.
          p.xy += w * 1.1 + vec2(sin(uTime * 0.7 + h * 9.0), cos(uTime * 0.6 + h * 7.0)) * 0.3;
          p.z += sin(uTime * 0.9 + h * 11.0) * 0.15 + lw * 0.1;
        #endif
        #if KIND == 9
          // Foliage billboards face the camera; tip holds the card size.
          p += ((card.x - 0.5) * uCamRight + (card.y - 0.5) * uCamUp) * tip;
        #endif
        vPos = p; vNormal = normal; vRoot = root; vTip = tip; vCard = card;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
      }`,
    fragmentShader: common + /* glsl */ `
      varying vec3 vPos, vNormal, vRoot; varying float vTip; varying vec2 vCard;
      void main() {
        #if KIND == 3
          vec2 q = (vCard - 0.5) * vec2(1.0, 1.9);
          if (length(q) > 0.32 + 0.25 * vnoise(vec3(vCard * 7.0, 0.0))) discard;
        #endif
        #if KIND == 9
          // A soft, ragged dab, dithered at its edge (painterly, no sorting).
          float rr = length((vCard - 0.5) * 2.0) + (vnoise(vec3(vCard * 5.0, vRoot.x * 3.1)) - 0.5) * 0.55;
          float alpha = 1.0 - smoothstep(0.45, 1.0, rr);
          if (alpha < hash(vec3(gl_FragCoord.xy, 0.0))) discard;
        #endif
        if (uFlyView > 0.5) {
          gl_FragColor = vec4(texture2D(uHeaven, equirect(vPos - uEye)).rgb, 1.0);
          return;
        }
        #if KIND == 0 || KIND == 4
          vec3 point = vPos;
        #else
          vec3 point = vRoot;
        #endif
        vec3 albedo = grade(texture2D(uHeaven, equirect(point - uEye)).rgb);
        // Painterly variation per dab.
        float b = vnoise(point * 1.6);
        albedo = mix(albedo, albedo * (0.75 + 0.5 * b), 0.5);
        #if KIND == 1 || KIND == 2 || KIND == 3
          albedo = 1.0 - (1.0 - albedo) * (1.0 - 0.35 * vTip * vec3(1.0, 0.95, 0.85));
        #endif
        vec3 n = normalize(vNormal);
        vec3 V = normalize(uCam - vPos);
        vec3 sunCol = vec3(1.0, 0.86, 0.66) * 1.3;
        #if KIND == 6 || KIND == 7
          // Faceted: flat normals from screen derivatives.
          n = normalize(cross(dFdx(vPos), dFdy(vPos)));
          if (dot(n, V) < 0.0) n = -n;
        #endif
        #if KIND == 6
          // Stone: heaven's colour, greyed and darkened; wet and glossy at the waterline.
          albedo = mix(albedo, vec3(dot(albedo, vec3(0.299, 0.587, 0.114))), 0.55) * 0.7;
          float wet = 1.0 - smoothstep(${WATER_LEVEL.toFixed(2)}, ${WATER_LEVEL.toFixed(2)} + 0.1, vPos.z);
          albedo *= 1.0 - 0.45 * wet;
          float diff = max(dot(n, uSun), 0.0);
          vec3 col = albedo * (0.45 + diff * sunCol * 0.65);
          col += wet * pow(max(dot(reflect(-uSun, n), V), 0.0), 30.0) * 0.5;
        #elif KIND == 7
          // Glass canopy: translucent body, bright facet rims, sun through it.
          float diff = abs(dot(n, uSun));
          float back = pow(max(dot(V, -uSun), 0.0), 2.0);
          float rim = pow(1.0 - abs(dot(n, V)), 2.5);
          vec3 col = albedo * (0.5 + diff * sunCol * 0.4) + albedo * back * sunCol * 0.6
                   + (albedo * 0.6 + 0.25) * rim * 0.9;
        #elif KIND == 8
          // Glass branches, rim-lit like the grass.
          albedo *= 0.8;
          float diff = abs(dot(n, uSun));
          float rim = pow(1.0 - abs(dot(n, V)), 2.5);
          float back = pow(max(dot(V, -uSun), 0.0), 3.0);
          vec3 col = albedo * (0.5 + diff * sunCol * 0.4) + (albedo * 0.6 + 0.25) * rim * 0.8 + albedo * back * sunCol * 0.4;
        #elif KIND == 9
          // Foliage mass: sphere normals give one soft roll of light across the
          // crown; sunlight glows through it when you look toward the sun.
          float wrap = dot(n, uSun) * 0.5 + 0.5;
          float lit = smoothstep(0.2, 0.85, wrap);
          float back = pow(max(dot(V, -uSun), 0.0), 2.0);
          float rim = pow(1.0 - max(dot(n, V), 0.0), 2.0);
          vec3 col = albedo * (0.42 + lit * sunCol * 0.55) + albedo * back * sunCol * 0.7 + albedo * rim * 0.35;
        #elif KIND == 0 || KIND == 4
          float diff = max(dot(n, uSun), 0.0);
          vec3 col = albedo * (0.5 + diff * sunCol * 0.5);
          #if KIND == 4
            col += pow(max(dot(reflect(-uSun, n), V), 0.0), 40.0) * 0.6;
          #endif
        #else
          float diff = abs(dot(n, uSun));
          float back = pow(max(dot(V, -uSun), 0.0), 3.0);      // sunlight through the blade
          vec3 col = albedo * (0.55 + diff * sunCol * 0.35) + albedo * back * sunCol * 0.45;
        #endif
        // Neural light: activity of the hero's visual neurons looking this way.
        vec3 body = uHeroInv * (vPos - uHeroPos);
        float act = texture2D(uActivity, equirect(body)).r;
        #if KIND == 5
          float where = 1.0;
          col = albedo * 0.3;
        #elif KIND == 2
          float where = 1.0;
        #elif KIND == 1 || KIND == 3
          float where = vTip * vTip * vTip;
        #elif KIND == 7
          float where = 0.25 + pow(1.0 - abs(dot(n, V)), 2.0);
        #elif KIND == 9
          float where = 0.3 + pow(1.0 - max(dot(n, V), 0.0), 2.0);
        #else
          float where = 0.0;
        #endif
        vec3 lightCol = mix(vec3(1.0, 0.62, 0.25), vec3(1.0, 0.95, 0.85), act);
        // Saturating, so strong activity glows rather than blowing out to white.
        col += lightCol * 0.7 * (1.0 - exp(-act * act * where * uGlow));
        #if KIND == 8
          // Light gathers in the finest twigs, as on a backlit crown (and as a
          // neuron's activity lives at its fine branches). It appears only
          // where the fly's visual neurons are active, with a slow, faint drift.
          float twig = pow(vTip, 3.0);
          float drift = 0.8 + 0.2 * vnoise(vRoot * 1.5 + vec3(0.0, 0.0, uTime * 0.12));
          col += lightCol * 0.6 * (1.0 - exp(-act * act * twig * drift * uGlow));
        #endif
        #if KIND == 5
          col += vec3(1.0, 0.92, 0.75) * 0.3;
          float nearCam = length(vPos - uCam);
          if (hash(vec3(gl_FragCoord.xy, 1.0)) > smoothstep(0.5, 3.0, nearCam)) discard;
        #endif
        // Ground mist: denser near the ground.
        float d = length(vPos - uCam);
        float zAvg = 0.5 * (vPos.z + uCam.z);
        float fog = 1.0 - exp(-d * (0.002 + 0.009 * exp(-0.28 * max(zAvg, 0.0))));
        col = mix(col, vec3(1.0, 0.93, 0.84), fog);
        gl_FragColor = vec4(col, 1.0);
      }`,
  });
}

function skyMaterial(uniforms) {
  return new THREE.ShaderMaterial({
    uniforms, side: THREE.BackSide, depthWrite: false,
    vertexShader: /* glsl */ `
      varying vec3 vPos;
      void main() { vPos = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: common + /* glsl */ `
      varying vec3 vPos;
      void main() {
        vec3 dir = vPos - uCam;
        vec3 raw = texture2D(uHeaven, equirect(dir)).rgb;
        if (uFlyView > 0.5) { gl_FragColor = vec4(raw, 1.0); return; }
        vec3 c = grade(raw);
        c = 1.0 - (1.0 - c) * (1.0 - 0.35 * vec3(1.0, 0.93, 0.82));
        gl_FragColor = vec4(c * 1.05, 1.0);
      }`,
  });
}

// "And a river went out of Eden to water the garden" (Gen. 2:10). One water
// surface at a fixed level floods the meadow's hollows. It mirrors the meadow
// around it (itself coloured by heaven), and through it you see the heaven-
// coloured bed, tinted by depth. In the fly's own view it shows the raw
// panorama, so the eye point still sees exactly its heaven.
export const WATER_LEVEL = -0.4;   // b-space metres

function waterMaterial(uniforms) {
  return new THREE.ShaderMaterial({
    uniforms,
    vertexShader: /* glsl */ `
      uniform mat4 uReflectMatrix;
      varying vec3 vPos;
      varying vec4 vRefl;
      void main() {
        vPos = position;
        vRefl = uReflectMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: common + /* glsl */ `
      uniform sampler2D uHeight, uReflect;
      varying vec3 vPos;
      varying vec4 vRefl;
      void main() {
        if (uFlyView > 0.5) {
          gl_FragColor = vec4(texture2D(uHeaven, equirect(vPos - uEye)).rgb, 1.0);
          return;
        }
        // Water depth from the terrain height field (220 x 220 over +-70 m).
        // Interpolated by hand: float textures are not always filterable,
        // and nearest samples terrace the depth into contour rings.
        vec2 hc = clamp((vPos.xy + 70.0) / 140.0 * 219.0, 0.0, 218.999);
        ivec2 hi = ivec2(floor(hc));
        vec2 hf = hc - floor(hc);
        float bedZ = mix(mix(texelFetch(uHeight, hi, 0).r, texelFetch(uHeight, hi + ivec2(1, 0), 0).r, hf.x),
                         mix(texelFetch(uHeight, hi + ivec2(0, 1), 0).r, texelFetch(uHeight, hi + ivec2(1, 1), 0).r, hf.x), hf.y);
        float depth = ${WATER_LEVEL.toFixed(2)} - bedZ;
        // A still pond. Long, slow undulations keep it alive; the breeze adds
        // finer ripples only where gusts touch down (cat's paws), and these
        // scatter the reflection rather than carving the surface.
        // Local gusts set how rough the water is; the ripples are laid out
        // along the prevailing wind (following the local, eddying direction
        // would wind the pattern into spirals).
        vec2 wl = windAt(vPos.xy);
        float lw = length(wl);
        vec2 wd = normalize(uWind + vec2(1e-4, 0.0)), side = vec2(-wd.y, wd.x);
        float calm = smoothstep(0.0, 0.25, depth);              // waves die out in the shallows
        float windy = smoothstep(0.08, 1.0, lw);
        float px = length(fwidth(vPos.xy));
        // Ripples as slowly evolving noise (not wave trains, which band and
        // ring), drifting downwind, crests stretched across the wind. The two
        // broad octaves always move; the fine ones come with the gusts.
        vec2 p = vec2(dot(vPos.xy, wd) * 1.5, dot(vPos.xy, side));
        vec2 g = vec2(0.0);
        float f = 0.7, amp = 0.045;
        for (int o = 0; o < 4; o++) {
          float fo = float(o);
          vec3 q = vec3((p - vec2(uTime * (0.12 + 0.1 * fo), 0.0)) * f, uTime * (0.1 + 0.06 * fo) + fo * 7.3);
          float h0 = vnoise(q);
          vec2 dh = vec2(vnoise(q + vec3(0.12, 0.0, 0.0)), vnoise(q + vec3(0.0, 0.12, 0.0))) - h0;
          float gust = o < 2 ? 0.55 + 0.45 * windy : 0.08 + 0.92 * windy;
          g += dh / 0.12 * amp * gust * smoothstep(3.0 * px, 8.0 * px, 1.0 / f);
          f *= 2.4; amp *= 0.6;
        }
        vec2 slope = (wd * g.x * 1.5 + side * g.y) * calm;     // back to world axes
        vec3 n = normalize(vec3(-slope, 1.0));
        vec3 V = normalize(uCam - vPos);
        // The meadow mirrored, rippled and softened. Gusts blur it further.
        float soft = 0.002 + 0.006 * windy * calm;
        vec4 ruv = vRefl;
        ruv.xy += slope * 0.35 * ruv.w;
        vec3 refl = vec3(0.0);
        for (int i = 0; i < 5; i++) {
          float a = float(i) * 2.513 + hash(vPos * 40.0) * 6.283;
          vec2 o = i == 0 ? vec2(0.0) : vec2(cos(a), sin(a)) * soft;
          refl += texture2DProj(uReflect, ruv + vec4(o * ruv.w, 0.0, 0.0)).rgb;
        }
        refl /= 5.0;
        // Ruffled patches catch less of the bright scene: a soft darkening.
        refl *= 1.0 - 0.18 * windy * calm;
        // Beneath: the bed seen through the water (refracted a little by the
        // ripples), coloured by heaven like the ground, red light absorbed
        // first, and a cool in-scattered haze gathering with depth.
        vec3 bedPt = vPos + vec3(slope * depth * 2.0, -depth);
        // Heaven's bands, projected from the eye onto flat ground, would ring
        // the pond bed; a soft, wide sample keeps only its colour.
        vec2 bu = equirect(bedPt - uEye);
        vec3 bed = vec3(0.0);
        for (int i = 0; i < 6; i++) {
          float a = float(i) * 1.047;
          bed += texture2D(uHeaven, bu + vec2(cos(a) * 0.03, sin(a) * 0.05)).rgb;
        }
        bed = grade(bed / 6.0);
        bed = mix(vec3(dot(bed, vec3(0.299, 0.587, 0.114))), bed, 0.5) * 0.5;   // silt: muted, darker
        bed *= 0.55 + 0.45 * max(dot(normalize(vec3(-slope * 3.0, 1.0)), uSun), 0.0);
        vec3 deep = grade(texture2D(uHeaven, equirect(vPos - uEye)).rgb);
        float dl = dot(deep, vec3(0.299, 0.587, 0.114));
        vec3 haze = mix(vec3(dl), deep, 0.35) * vec3(0.13, 0.19, 0.22);
        vec3 absorb = exp(-depth * vec3(5.5, 2.8, 2.0));
        vec3 under = bed * absorb + haze * (1.0 - exp(-depth * 2.5));
        // Schlick's Fresnel for water; at the very edge the water thins to a
        // wet film over the bank, so the shore line dissolves.
        // (Lifted a little above the physical 2% so the pond holds a soft
        // sheen of sky even from above, as painted water does.)
        float fres = mix(0.12, 1.0, pow(1.0 - max(dot(n, V), 0.0), 5.0));
        fres *= smoothstep(0.0, 0.03, depth);
        vec3 col = mix(under, refl, fres);
        // A faint bright meniscus where water meets the bank.
        float edge = 1.0 - smoothstep(0.0, 0.02, depth);
        col += vec3(1.0, 0.97, 0.9) * edge * 0.06;
        // The sun's glitter path, broken up by the ripples.
        vec3 R = reflect(-V, n);
        col += vec3(1.0, 0.9, 0.72) * pow(max(dot(R, uSun), 0.0), 600.0) * 0.7 * calm;
        float d = length(vPos - uCam);
        float zAvg = 0.5 * (vPos.z + uCam.z);
        float fog = 1.0 - exp(-d * (0.002 + 0.009 * exp(-0.28 * max(zAvg, 0.0))));
        col = mix(col, vec3(1.0, 0.93, 0.84), fog);
        gl_FragColor = vec4(col, 1.0);
      }`,
  });
}

// Planar reflection for the water (after three.js's Reflector): the scene is
// rendered from the camera mirrored in the water plane, with an oblique near
// plane so nothing below the surface shows, into a half-resolution target
// that the water samples in projection.
export class Mirror {
  constructor(water, uniforms, scale = 0.5) {
    this.water = water;
    this.uniforms = uniforms;
    this.scale = scale;
    this.rt = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType });
    this.cam = new THREE.PerspectiveCamera();
    this.size = new THREE.Vector2();
    uniforms.uReflect.value = this.rt.texture;
  }

  render(renderer, scene, camera) {
    const v = this.cam, water = this.water, u = this.uniforms;
    renderer.getDrawingBufferSize(this.size);
    const w = Math.max(1, Math.round(this.size.x * this.scale)), h = Math.max(1, Math.round(this.size.y * this.scale));
    if (this.rt.width !== w || this.rt.height !== h) this.rt.setSize(w, h);
    const P = new THREE.Vector3().setFromMatrixPosition(water.matrixWorld);
    const C = new THREE.Vector3().setFromMatrixPosition(camera.matrixWorld);
    const rot = new THREE.Matrix4().extractRotation(water.matrixWorld);
    const normal = new THREE.Vector3(0, 0, 1).applyMatrix4(rot);
    const view = new THREE.Vector3().subVectors(P, C);
    if (view.dot(normal) > 0) return;                      // camera below the water
    view.reflect(normal).negate().add(P);
    rot.extractRotation(camera.matrixWorld);
    const look = new THREE.Vector3(0, 0, -1).applyMatrix4(rot).add(C);
    const target = new THREE.Vector3().subVectors(P, look).reflect(normal).negate().add(P);
    v.position.copy(view);
    v.up.set(0, 1, 0).applyMatrix4(rot).reflect(normal);
    v.lookAt(target);
    v.far = camera.far;
    v.updateMatrixWorld();
    v.projectionMatrix.copy(camera.projectionMatrix);
    u.uReflectMatrix.value.set(0.5, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 1)
      .multiply(v.projectionMatrix).multiply(v.matrixWorldInverse).multiply(water.matrixWorld);
    // Oblique clip at the water plane.
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, P).applyMatrix4(v.matrixWorldInverse);
    const clip = new THREE.Vector4(plane.normal.x, plane.normal.y, plane.normal.z, plane.constant);
    const e = v.projectionMatrix.elements;
    const q = new THREE.Vector4((Math.sign(clip.x) + e[8]) / e[0], (Math.sign(clip.y) + e[9]) / e[5], -1, (1 + e[10]) / e[14]);
    clip.multiplyScalar(2 / clip.dot(q));
    e[2] = clip.x; e[6] = clip.y; e[10] = clip.z + 1 - 0.003; e[14] = clip.w;
    // View-dependent shading follows the mirrored camera (b-space).
    const world = water.parent;
    const t2b = (x) => new THREE.Vector3(x.x, -x.z, x.y);
    const saved = [u.uCam.value.clone(), u.uCamRight.value.clone(), u.uCamUp.value.clone()];
    u.uCam.value.copy(world.worldToLocal(v.position.clone()));
    u.uCamRight.value.copy(t2b(new THREE.Vector3().setFromMatrixColumn(v.matrixWorld, 0)));
    u.uCamUp.value.copy(t2b(new THREE.Vector3().setFromMatrixColumn(v.matrixWorld, 1)));
    water.visible = false;
    const prev = renderer.getRenderTarget();
    renderer.setRenderTarget(this.rt);
    renderer.clear();
    renderer.render(scene, v);
    renderer.setRenderTarget(prev);
    water.visible = true;
    [u.uCam.value, u.uCamRight.value, u.uCamUp.value].forEach((x, i) => x.copy(saved[i]));
  }
}

// Ground height lookup on the exported terrain grid (220 x 220 over +-70 m).
function heightField(pos) {
  const N = 220, L = 70;
  return (x, y) => {
    const fx = ((x + L) / (2 * L)) * (N - 1), fy = ((y + L) / (2 * L)) * (N - 1);
    const i = Math.max(0, Math.min(N - 2, Math.floor(fx))), j = Math.max(0, Math.min(N - 2, Math.floor(fy)));
    const u = fx - i, v = fy - j, z = (a, b) => pos[(b * N + a) * 3 + 2];
    return (z(i, j) * (1 - u) + z(i + 1, j) * u) * (1 - v) + (z(i, j + 1) * (1 - u) + z(i + 1, j + 1) * u) * v;
  };
}

export async function loadMeadow(scene) {
  const meta = await fetch("data/meadow/meadow.json").then((r) => r.json());
  const heaven = await new THREE.TextureLoader().loadAsync("data/heaven.png");
  heaven.colorSpace = THREE.NoColorSpace;       // raw values, exactly as the brain saw them
  heaven.wrapS = THREE.RepeatWrapping;
  heaven.minFilter = THREE.LinearFilter;
  const activity = new THREE.DataTexture(new Uint8Array(128 * 64), 128, 64, THREE.RedFormat);
  activity.wrapS = THREE.RepeatWrapping;
  activity.magFilter = activity.minFilter = THREE.LinearFilter;
  activity.flipY = true;                        // row 0 of the brain's map is the top of the sky
  activity.needsUpdate = true;
  const uniforms = {
    uHeaven: { value: heaven }, uActivity: { value: activity },
    uEye: { value: EYE.clone() }, uSun: { value: sunDirection() }, uCam: { value: new THREE.Vector3() },
    uHeroPos: { value: new THREE.Vector3() }, uHeroInv: { value: new THREE.Matrix3() },
    uFlyView: { value: 0 }, uGlow: { value: 2.0 }, uContrast: { value: 1.9 }, uSaturation: { value: 1.9 },
    uTime: { value: 0 },
    uCamRight: { value: new THREE.Vector3(1, 0, 0) }, uCamUp: { value: new THREE.Vector3(0, 0, 1) },
    // Wind field, updated in place by Wind (wind.js).
    uWind: { value: new THREE.Vector2(0.6, 0.3) },
    uPuffA: { value: Array.from({ length: MAX_PUFFS }, () => new THREE.Vector4(0, 0, 1, 0)) },
    uPuffB: { value: Array.from({ length: MAX_PUFFS }, () => new THREE.Vector4()) },
  };
  let groundPos = null, trunkPos = null;
  const world = new THREE.Group();
  world.rotation.x = -Math.PI / 2;              // Blender Z-up -> three Y-up
  scene.add(world);
  // A sphere looks the same any way up; its vertices are simply b-space points.
  const sky = new THREE.Mesh(new THREE.SphereGeometry(600, 48, 24), skyMaterial(uniforms));
  sky.frustumCulled = false;
  world.add(sky);
  await Promise.all(meta.parts.map(async (p) => {
    const buf = await fetch(`data/meadow/${p.name}.bin`).then((r) => r.arrayBuffer());
    let o = 0;
    const take = (n, T) => { const a = new T(buf, o, n); o += n * T.BYTES_PER_ELEMENT; return a; };
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(take(p.verts * 3, Float32Array), 3));
    g.setAttribute("normal", new THREE.BufferAttribute(take(p.verts * 3, Float32Array), 3));
    g.setAttribute("root", new THREE.BufferAttribute(take(p.verts * 3, Float32Array), 3));
    g.setAttribute("tip", new THREE.BufferAttribute(take(p.verts, Float32Array), 1));
    g.setAttribute("card", p.card ? new THREE.BufferAttribute(take(p.verts * 2, Float32Array), 2)
                                  : new THREE.BufferAttribute(new Float32Array(p.verts * 2), 2));
    g.setIndex(new THREE.BufferAttribute(take(p.indices, Uint32Array), 1));
    g.computeBoundingSphere();
    const m = new THREE.Mesh(g, heavenMaterial(p.kind, uniforms));
    m.name = p.name;
    m.frustumCulled = false;
    world.add(m);
    if (p.name === "ground") groundPos = g.attributes.position.array;
  }));
  // Warp the terrain so pools take irregular, lobed shapes; everything rooted
  // in the ground moves with it.
  world.traverse((m) => {
    const kind = m.material?.defines?.KIND;
    if (kind === undefined || kind === KIND.mote || kind === KIND.leaf) return;
    const P = m.geometry.attributes.position.array, R = m.geometry.attributes.root.array;
    const byRoot = kind === KIND.grass || kind === KIND.flower;
    for (let i = 0; i < P.length; i += 3) {
      const dz = byRoot ? terrainWarp(R[i], R[i + 1]) : terrainWarp(P[i], P[i + 1]);
      P[i + 2] += dz;
      R[i + 2] += dz;
    }
    m.geometry.attributes.position.needsUpdate = true;
    m.geometry.attributes.root.needsUpdate = true;
    if (kind === KIND.ground) m.geometry.computeVertexNormals();
    if (m.name === "trunks") trunkPos = P;
  });

  // Open water: blades and flowers rooted in the water, or right at its edge
  // (where they would lean out over it), are removed (collapsed to a point).
  const groundHeight = heightField(groundPos);
  const SHORE_MARGIN = 0.08;   // metres of dry bank kept clear above the waterline
  world.traverse((m) => {
    const kind = m.material?.defines?.KIND;
    if (kind !== KIND.grass && kind !== KIND.flower) return;
    const P = m.geometry.attributes.position.array, R = m.geometry.attributes.root.array;
    for (let i = 0; i < P.length; i += 3) {
      const x = R[i], y = R[i + 1];
      if (groundHeight(x, y) > WATER_LEVEL + SHORE_MARGIN) continue;
      P[i] = x; P[i + 1] = y; P[i + 2] = WATER_LEVEL - 1;
    }
    m.geometry.attributes.position.needsUpdate = true;
  });
  // Stones along every shoreline.
  const rocks = shoreRocks(groundHeight, WATER_LEVEL);
  // Fruit: one landing point on top of each (vertices clustered by proximity).
  const fruits = [];
  world.traverse((m) => {
    if (m.material?.defines?.KIND !== KIND.fruit) return;
    const P = m.geometry.attributes.position.array;
    const groups = [];
    for (let i = 0; i < P.length; i += 3) {
      const v = new THREE.Vector3(P[i], P[i + 1], P[i + 2]);
      let g = groups.find((g) => g.c.distanceTo(v) < 0.5);
      if (!g) groups.push(g = { c: v.clone(), top: v.z, n: 0, sum: new THREE.Vector3() });
      g.sum.add(v); g.n++; g.top = Math.max(g.top, v.z);
    }
    for (const g of groups) { const c = g.sum.divideScalar(g.n); fruits.push(new THREE.Vector3(c.x, c.y, g.top)); }
  });
  world.add(new THREE.Mesh(rocks.geometry, heavenMaterial("rock", uniforms)));
  // Trees grown like neurons replace the exported trunks and leaf cards.
  world.children.filter((m) => m.material?.defines?.KIND === KIND.leaf || m.name === "trunks")
    .forEach((m) => world.remove(m));
  const trees = neuronTrees(trunkPos);
  world.add(new THREE.Mesh(trees.branches, heavenMaterial("branch", uniforms)));
  const foliage = new THREE.Mesh(trees.foliage, heavenMaterial("foliage", uniforms));
  foliage.frustumCulled = false;
  world.add(foliage);
  console.log(`meadow: ${rocks.count} shore stones, ${trees.trees} trees, ${fruits.length} fruit, ${trees.perches.length} perches`);

  const heightTex = new THREE.DataTexture(Float32Array.from({ length: groundPos.length / 3 }, (_, i) => groundPos[i * 3 + 2]),
    220, 220, THREE.RedFormat, THREE.FloatType);
  heightTex.magFilter = heightTex.minFilter = THREE.LinearFilter;
  heightTex.needsUpdate = true;
  uniforms.uHeight = { value: heightTex };
  uniforms.uReflect = { value: null };
  uniforms.uReflectMatrix = { value: new THREE.Matrix4() };
  const water = new THREE.Mesh(new THREE.PlaneGeometry(400, 400, 1, 1), waterMaterial(uniforms));
  water.position.z = WATER_LEVEL;           // PlaneGeometry lies in b-space XY already
  world.add(water);
  world.updateMatrixWorld();
  const mirror = new Mirror(water, uniforms);
  return { world, uniforms, activity, groundHeight, mirror, spots: { fruits, perches: trees.perches } };
}

// Painterly post: a four-sector Kuwahara filter (oil-paint brushwork).
export const KuwaharaShader = {
  uniforms: { tDiffuse: { value: null }, resolution: { value: new THREE.Vector2(1, 1) }, radius: { value: 4 } },
  vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse; uniform vec2 resolution; uniform int radius; varying vec2 vUv;
    void main() {
      vec2 px = 1.0 / resolution;
      vec3 m[4]; vec3 s[4];
      for (int k = 0; k < 4; k++) { m[k] = vec3(0.0); s[k] = vec3(0.0); }
      ivec2 off[4] = ivec2[4](ivec2(-1, -1), ivec2(1, -1), ivec2(-1, 1), ivec2(1, 1));
      float n = float((radius + 1) * (radius + 1));
      for (int k = 0; k < 4; k++) {
        for (int j = 0; j <= 4; j++) for (int i = 0; i <= 4; i++) {
          if (i > radius || j > radius) continue;
          vec3 c = texture2D(tDiffuse, vUv + vec2(float(i * off[k].x), float(j * off[k].y)) * px * 1.5).rgb;
          m[k] += c; s[k] += c * c;
        }
      }
      float best = 1e9; vec3 outc = vec3(0.0);
      for (int k = 0; k < 4; k++) {
        vec3 mu = m[k] / n; vec3 var = abs(s[k] / n - mu * mu);
        float v = var.r + var.g + var.b;
        if (v < best) { best = v; outc = mu; }
      }
      gl_FragColor = vec4(outc, 1.0);
    }`,
};
