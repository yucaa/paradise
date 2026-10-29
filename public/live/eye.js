// The hero's eye: renders a cube map from the front face of the hero cube and
// resamples it to the brain's input, a 128x64 equirectangular view in the
// fly's body frame (forward at the centre, right to the right, sky at the
// top), in raw panorama values.
import * as THREE from "three";

export const W = 128, H = 64;

export class Eye {
  constructor(renderer) {
    this.renderer = renderer;
    this.cubeRT = new THREE.WebGLCubeRenderTarget(192, { type: THREE.UnsignedByteType, colorSpace: THREE.NoColorSpace });
    this.cubeCam = new THREE.CubeCamera(0.02, 2000, this.cubeRT);
    this.rt = new THREE.WebGLRenderTarget(W, H, { type: THREE.UnsignedByteType, colorSpace: THREE.NoColorSpace });
    this.quadScene = new THREE.Scene();
    this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.material = new THREE.ShaderMaterial({
      uniforms: { cube: { value: this.cubeRT.texture }, bodyToB: { value: new THREE.Matrix3() } },
      vertexShader: `void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }`,
      fragmentShader: /* glsl */ `
        uniform samplerCube cube; uniform mat3 bodyToB;
        const float PI = 3.14159265;
        void main() {
          float az = (gl_FragCoord.x / ${W}.0 - 0.5) * 2.0 * PI;
          float el = (gl_FragCoord.y / ${H}.0 - 0.5) * PI;
          vec3 body = vec3(sin(az) * cos(el), cos(az) * cos(el), sin(el));
          vec3 b = bodyToB * body;
          gl_FragColor = vec4(textureCube(cube, vec3(b.x, b.z, -b.y)).rgb, 1.0);  // b-space -> three
        }`,
    });
    this.quadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.material));
    this.pixels = new Uint8Array(W * H * 4);
    this.rgb = new Uint8Array(W * H * 3);
  }

  // eyeWorld: three-space position; bodyToB: Matrix3 (body -> b-space).
  capture(scene, eyeWorld, bodyToB) {
    const r = this.renderer;
    const tm = r.toneMapping;
    r.toneMapping = THREE.NoToneMapping;
    this.cubeCam.position.copy(eyeWorld);
    this.cubeCam.update(r, scene);
    this.material.uniforms.bodyToB.value.copy(bodyToB);
    r.setRenderTarget(this.rt);
    r.render(this.quadScene, this.quadCam);
    r.readRenderTargetPixels(this.rt, 0, 0, W, H, this.pixels);
    r.setRenderTarget(null);
    r.toneMapping = tm;
    // GL rows run bottom-up; the brain expects the top of the sky first.
    for (let y = 0; y < H; y++) {
      const src = (H - 1 - y) * W * 4, dst = y * W * 3;
      for (let x = 0; x < W; x++) {
        this.rgb[dst + x * 3] = this.pixels[src + x * 4];
        this.rgb[dst + x * 3 + 1] = this.pixels[src + x * 4 + 1];
        this.rgb[dst + x * 3 + 2] = this.pixels[src + x * 4 + 2];
      }
    }
    return this.rgb;
  }
}
