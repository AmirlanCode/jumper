// Crab Rescue 3D engine: builds the burning room, drives the real Jumper crab rig (URDF tree,
// STL meshes), renders lit frames (what a camera sees) and a flat "segmentation" frame.
// Everything is a pure function of the state handed to setSim(); nothing here decides anything.
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

export const PALETTE = {
  floor: [232, 228, 216], smoke: [186, 184, 182], wall: [52, 54, 66], obstacle: [126, 84, 48],
  exit: [40, 190, 90], fire: [238, 78, 24], figurine: [36, 112, 255], crab: [200, 40, 190],
};
const FREE = 0, WALL = 1, OBST = 2, EXIT = 3;
const WALL_H = 1.25;
const TMP = new THREE.Vector3();

export const s2w = (x, y, z = 0) => new THREE.Vector3(x, -y, z);
const wrapPi = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const smooth = (a, b, v) => { const t = clamp((v - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const rgb01 = (c) => new THREE.Vector3(c[0] / 255, c[1] / 255, c[2] / 255);

// ---------------------------------------------------------------- textures
function canvasTex(w, h, draw, srgb = true) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8; t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}
function floorTexture(room) {
  const px = 64, W = room.w * px, H = room.h * px, r = rng(room.seed * 31 + 7);
  const t = canvasTex(W, H, (g) => {
    for (let y = 0; y < room.h; y++) for (let x = 0; x < room.w; x++) {
      const v = 30 + Math.floor(r() * 12);
      g.fillStyle = `rgb(${v},${v + 3},${v + 10})`; g.fillRect(x * px, y * px, px, px);
      g.fillStyle = 'rgba(255,255,255,0.025)'; g.fillRect(x * px + 4, y * px + 4, px - 8, 2);
    }
    g.strokeStyle = 'rgba(0,0,0,0.55)'; g.lineWidth = 2;
    for (let x = 0; x <= room.w; x++) { g.beginPath(); g.moveTo(x * px, 0); g.lineTo(x * px, H); g.stroke(); }
    for (let y = 0; y <= room.h; y++) { g.beginPath(); g.moveTo(0, y * px); g.lineTo(W, y * px); g.stroke(); }
    for (let i = 0; i < 1800; i++) { g.fillStyle = `rgba(255,255,255,${r() * 0.05})`; g.fillRect(r() * W, r() * H, 2, 2); }
  });
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  return t;
}
function labelTexture(text, fg, bg, w = 512, h = 128) {
  return canvasTex(w, h, (g) => {
    g.fillStyle = bg; g.fillRect(0, 0, w, h);
    g.fillStyle = fg; g.font = `bold ${h * 0.6}px system-ui, sans-serif`; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText(text, w / 2, h / 2 + 4);
  });
}
function stripeTexture() {
  return canvasTex(64, 64, (g, w, h) => {
    g.fillStyle = '#f2c200'; g.fillRect(0, 0, w, h); g.fillStyle = '#161616';
    for (let i = -2; i < 6; i++) { g.beginPath(); g.moveTo(i * 16, 0); g.lineTo(i * 16 + 8, 0); g.lineTo(i * 16 + 8 + 64, 64); g.lineTo(i * 16 + 64, 64); g.fill(); }
  });
}
function woodTexture() {
  return canvasTex(128, 128, (g, w, h) => {
    g.fillStyle = '#8a5b30'; g.fillRect(0, 0, w, h);
    for (let i = 0; i < 9; i++) { g.fillStyle = i % 2 ? 'rgba(0,0,0,.12)' : 'rgba(255,255,255,.05)'; g.fillRect(0, i * 14, w, 7); }
    g.strokeStyle = 'rgba(0,0,0,.5)'; g.lineWidth = 6; g.strokeRect(3, 3, w - 6, h - 6);
  });
}

// ---------------------------------------------------------------- shaders
const GLSL_NOISE = `
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.-2.*f);
  return mix(mix(hash12(i), hash12(i+vec2(1,0)), f.x), mix(hash12(i+vec2(0,1)), hash12(i+vec2(1,1)), f.x), f.y); }
float fbm(vec2 p){ float a = .5, s = 0.; for(int i=0;i<4;i++){ s += a*vnoise(p); p *= 2.03; a *= .5; } return s; }
`;
const BILLBOARD_VS = `
attribute vec3 aCenter; attribute float aIgnite; attribute float aSeed;
uniform float uSimT; uniform float uAnim; uniform float uMode; // 0 flame, 1 smoke, 2 ember
varying vec2 vUv; varying float vA; varying float vSeed; varying float vLife;
void main(){
  float age = uSimT - aIgnite;
  vec3 right = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
  vec3 up = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);
  vec3 c = aCenter; float size = 0.0; vA = 0.0; vLife = 0.0;
  if (uMode < 0.5) {           // flame
    float g = smoothstep(0.0, 2.2, age) * (0.78 + 0.3 * aSeed);
    size = 0.95 * g; c.z += size * 0.42; vA = step(0.0, age);
    c.xy += 0.10 * vec2(sin(uAnim*3.1+aSeed*40.), cos(uAnim*2.7+aSeed*23.));
  } else {                     // smoke / ember rise from a burning cell
    float rate = uMode < 1.5 ? 0.11 : 0.55;
    float u = fract(uAnim * rate + aSeed * 7.31);
    float on = step(0.6, age) * smoothstep(0.6, 3.0, age);
    if (uMode < 1.5) { size = (0.5 + 1.5 * u) * 1.0; c.z = 0.35 + u * 2.1; vA = on * smoothstep(0.0, 0.12, u) * (1.0 - u) * 0.34; }
    else { size = 0.035 + 0.02 * aSeed; c.z = 0.2 + u * 1.5; vA = on * (1.0 - u) * step(0.35, aSeed); }
    c.xy += vec2(0.35 + 0.7*u, 0.2 + 0.4*u) * (0.4 + aSeed) * (uMode < 1.5 ? 0.9 : 0.5) + 0.22*u*vec2(sin(aSeed*91.+uAnim*.7), cos(aSeed*53.+uAnim*.6));
    vLife = u;
  }
  vUv = uv; vSeed = aSeed;
  vec3 world = c + right * position.x * size + up * position.y * size;
  gl_Position = projectionMatrix * viewMatrix * vec4(world, 1.0);
}`;
const FLAME_FS = `
precision highp float; varying vec2 vUv; varying float vA; varying float vSeed; uniform float uAnim;
${GLSL_NOISE}
void main(){
  if (vA < .5) discard;
  vec2 p = vUv * 2.0 - 1.0; float y = vUv.y;
  float t = uAnim * 2.4 + vSeed * 20.;
  float n = fbm(vec2(p.x * 2.4 + vSeed * 9., y * 3.2 - t));
  float width = pow(1.0 - y, 0.75) * 0.62 + 0.04;
  float body = smoothstep(width, width * 0.35, abs(p.x + (n - .5) * 0.55 * y));
  float fade = smoothstep(1.0, 0.25, y + (n - .5) * 0.5) * smoothstep(0.0, 0.08, y);
  float I = clamp(body * fade * (0.75 + 0.5 * n), 0., 1.);
  vec3 col = mix(vec3(.75,.06,.01), vec3(1.,.42,.06), smoothstep(0.1, .5, I));
  col = mix(col, vec3(1.,.9,.5), smoothstep(.6, .95, I));
  gl_FragColor = vec4(col * I * 1.5, I);
}`;
const SMOKE_FS = `
precision highp float; varying vec2 vUv; varying float vA; varying float vSeed; varying float vLife; uniform float uAnim;
${GLSL_NOISE}
void main(){
  if (vA < .004) discard;
  vec2 p = vUv * 2.0 - 1.0; float r = length(p);
  float n = fbm(p * 2.2 + vSeed * 17. + uAnim * 0.05);
  float a = smoothstep(1.0, 0.15, r) * (0.55 + 0.7 * n) * vA;
  vec3 col = mix(vec3(.42,.2,.09), vec3(.2,.2,.22), smoothstep(0., .5, vLife));
  gl_FragColor = vec4(col, a);
}`;
const EMBER_FS = `
precision highp float; varying vec2 vUv; varying float vA; varying float vLife;
void main(){ if (vA < .02) discard; float r = length(vUv * 2. - 1.); float a = smoothstep(1., 0., r) * vA;
  gl_FragColor = vec4(vec3(1., .55, .15) * a * 1.8, a); }`;
const GLOW_VS = `
attribute vec3 aCenter; attribute float aIgnite; attribute float aSeed; uniform float uSimT; varying vec2 vUv; varying float vG;
void main(){ float age = uSimT - aIgnite; float g = smoothstep(0.0, 2.0, age) * step(0.0, age); vG = g; vUv = uv;
  vec3 w = aCenter + vec3(position.xy * 1.9 * g, 0.012); gl_Position = projectionMatrix * viewMatrix * vec4(w, 1.0); }`;
const GLOW_FS = `
precision highp float; varying vec2 vUv; varying float vG; uniform float uAnim;
void main(){ if (vG < .01) discard; float r = length(vUv * 2. - 1.); float a = smoothstep(1., 0., r); a = a * a;
  float fl = 0.85 + 0.15 * sin(uAnim * 9. + vUv.x * 7.);
  gl_FragColor = vec4(vec3(1., .33, .05) * a * 0.9 * fl, a * 0.8); }`;
const CHAR_FS = `
precision highp float; varying vec2 vUv; varying float vG;
void main(){ if (vG < .01) discard; float r = length(vUv * 2. - 1.); float a = smoothstep(1., .25, r) * 0.55; gl_FragColor = vec4(0., 0., 0., a * vG); }`;
const FLAT_VS = `void main(){ gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const FLAT_FS = `uniform vec3 uColor; void main(){ gl_FragColor = vec4(uColor, 1.0); }`;
const flat = (c) => new THREE.ShaderMaterial({ vertexShader: FLAT_VS, fragmentShader: FLAT_FS, uniforms: { uColor: { value: rgb01(c) } } });

function ribbon(points, width, color, dashed) {
  const pos = [], uv = [], idx = []; let dist = 0;
  for (let i = 0; i < points.length; i++) {
    const p = points[i], a = points[Math.max(0, i - 1)], b = points[Math.min(points.length - 1, i + 1)];
    let dx = b.x - a.x, dy = b.y - a.y; const l = Math.hypot(dx, dy) || 1; dx /= l; dy /= l;
    if (i) dist += Math.hypot(p.x - points[i - 1].x, p.y - points[i - 1].y);
    pos.push(p.x - dy * width, p.y + dx * width, 0.02, p.x + dy * width, p.y - dx * width, 0.02);
    uv.push(dist, 0, dist, 1);
    if (i) { const k = i * 2; idx.push(k - 2, k - 1, k, k - 1, k + 1, k); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2)); g.setIndex(idx);
  const m = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uColor: { value: new THREE.Color(color) }, uAnim: { value: 0 }, uDash: { value: dashed ? 1 : 0 } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }',
    fragmentShader: `precision highp float; varying vec2 vUv; uniform vec3 uColor; uniform float uAnim; uniform float uDash;
      void main(){ float edge = smoothstep(0., .25, vUv.y) * smoothstep(1., .75, vUv.y);
        float d = mix(1., smoothstep(.25, .5, abs(fract(vUv.x * 5. - uAnim * 1.2) - .5) * 2. ), uDash);
        float flow = .55 + .45 * smoothstep(.0, 1., fract(vUv.x * 1.5 - uAnim * .8));
        gl_FragColor = vec4(uColor * edge * d * flow, edge * d); }`,
  });
  const mesh = new THREE.Mesh(g, m); mesh.frustumCulled = false; mesh.renderOrder = 5;
  return mesh;
}

// ---------------------------------------------------------------- crab rig
const LEGS = ['LM', 'RM', 'LR', 'RR'];
const GROUP_A = new Set(['LM', 'RR']);

class CrabRig {
  constructor(kin, gltf) {
    this.kin = kin; this.root = new THREE.Group(); this.links = {}; this.joints = {}; this.q = {};
    const meshes = {};
    gltf.scene.traverse((o) => { if (o.isMesh) { let n = o; while (n && !(n.name in kin.links)) n = n.parent; meshes[(n || o).name] = o; } });
    for (const [name, l] of Object.entries(kin.links)) {
      const g = new THREE.Group(); g.name = name; this.links[name] = g;
      if (l.mesh && meshes[l.mesh]) {
        const m = meshes[l.mesh].clone(); m.position.set(0, 0, 0); m.rotation.set(0, 0, 0); m.scale.set(1, 1, 1);
        m.material = meshes[l.mesh].material.clone(); m.material.roughness = 0.45; m.material.metalness = 0.25;
        m.castShadow = false; g.add(m);
      }
    }
    for (const j of kin.joints) {
      const parent = this.links[j.parent], child = this.links[j.child];
      if (!parent || !child) continue;
      const origin = new THREE.Group(); origin.position.set(...j.xyz);
      origin.quaternion.setFromEuler(new THREE.Euler(j.rpy[0], j.rpy[1], j.rpy[2], 'ZYX')); // placeholder, fixed below
      // URDF rpy = fixed-axis roll(x), pitch(y), yaw(z): R = Rz*Ry*Rx
      origin.quaternion.copy(new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(j.rpy[0], j.rpy[1], j.rpy[2], 'ZYX'))));
      const rot = new THREE.Group();
      parent.add(origin); origin.add(rot); rot.add(child);
      this.joints[j.name] = { def: j, rot, axis: j.axis ? new THREE.Vector3(...j.axis).normalize() : null };
      if (j.type === 'revolute') this.q[j.name] = 0;
    }
    this.root.add(this.links[kin.root]);
    this.home = kin.home;
    this.pose = {}; this.setHome();
    this._calibrate();
  }
  setJoint(name, v) {
    const j = this.joints[name]; if (!j || !j.axis) return;
    const d = j.def; v = clamp(v, d.lower ?? -9, d.upper ?? 9);
    this.q[name] = v; j.rot.quaternion.setFromAxisAngle(j.axis, v);
  }
  setHome() { for (const n of Object.keys(this.q)) this.setJoint(n, this.home[n] ?? 0); }
  footPos(leg, out = new THREE.Vector3()) {
    this.root.updateMatrixWorld(true);
    const l = this.links[`${leg}_foot_tip_link`]; return l.getWorldPosition(out);
  }
  // find the sign of J0 that swings the foot forward (+x) and of J1 that lifts it, per leg
  _calibrate() {
    this.sw = {}; this.lift = {};
    this.root.position.set(0, 0, 0); this.root.quaternion.identity();
    for (const leg of LEGS) {
      this.setHome();
      const p0 = this.footPos(leg).clone();
      this.setJoint(`${leg}_J0_joint`, this.home[`${leg}_J0_joint`] + 0.2);
      const p1 = this.footPos(leg).clone();
      this.sw[leg] = p1.x > p0.x ? 1 : -1;
      this.setHome();
      let best = null;
      for (const s1 of [1, -1]) for (const s2 of [1, -1]) {
        this.setHome();
        this.setJoint(`${leg}_J1_joint`, this.home[`${leg}_J1_joint`] + s1 * 0.35);
        this.setJoint(`${leg}_J2_joint`, this.home[`${leg}_J2_joint`] + s2 * 0.35);
        const z = this.footPos(leg).z;
        if (!best || z > best.z) best = { z, s1, s2 };
      }
      this.lift[leg] = best; this.setHome();
    }
    this.setHome();
  }
  // gait: phase (rad), forward speed vn in [-1,1], turn wn in [-1,1] (ccw positive)
  applyGait(phase, vn, wn, moving) {
    for (const leg of LEGS) {
      const side = leg[0] === 'R' ? 1 : -1;       // right legs speed up when turning ccw
      const sPhase = phase + (GROUP_A.has(leg) ? 0 : Math.PI);
      const amp = 0.5 * clamp(vn + side * wn * 0.8, -1.2, 1.2);
      const swing = -Math.cos(sPhase) * amp;
      const lift = moving ? Math.max(0, Math.sin(sPhase)) : 0;
      const L = this.lift[leg];
      this.setJoint(`${leg}_J0_joint`, this.home[`${leg}_J0_joint`] + this.sw[leg] * swing);
      this.setJoint(`${leg}_J1_joint`, this.home[`${leg}_J1_joint`] + L.s1 * 0.5 * lift);
      this.setJoint(`${leg}_J2_joint`, this.home[`${leg}_J2_joint`] + L.s2 * 0.5 * lift);
    }
  }
}

// ---------------------------------------------------------------- engine
export class Engine {
  constructor(canvas, opts = {}) {
    this.opts = opts; this.driver = !!opts.driver;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: !this.driver, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(opts.pixelRatio ?? Math.min(2, window.devicePixelRatio || 1));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping; this.renderer.toneMappingExposure = 0.92;
    this.scene = new THREE.Scene(); this.scene.background = new THREE.Color(0x070a10);
    this.scene.fog = new THREE.FogExp2(0x080b12, 0.02);
    this.sem = new THREE.Scene(); this.sem.background = new THREE.Color(...PALETTE.floor.map((v) => v / 255));
    this.camera = new THREE.PerspectiveCamera(45, 16 / 9, 0.03, 80); this.camera.up.set(0, 0, 1);
    this.uni = { uSimT: { value: 0 }, uAnim: { value: 0 } };
    this.room = null; this.crab = null; this.fig = null; this.ignite = {};
    this.dyn = new THREE.Group(); this.scene.add(this.dyn);
    this.animClock = 0; this.deterministic = !!opts.deterministic;
  }
  async init(assetBase) {
    const [kin, gltf] = await Promise.all([
      fetch(`${assetBase}/crab_kin.json`).then((r) => r.json()),
      new GLTFLoader().loadAsync(`${assetBase}/crab.glb`)]);
    this.crab = new CrabRig(kin, gltf);
    this.crabHolder = new THREE.Group(); this.crabHolder.add(this.crab.root); this.scene.add(this.crabHolder);
    this.baseH = kin.base_height_m;
    this._lights();
  }
  _lights() {
    const s = this.scene;
    s.add(new THREE.HemisphereLight(0x86abff, 0x1a1418, 0.7));
    const d = new THREE.DirectionalLight(0xa9c6ff, 0.9); d.position.set(3, -6, 9); s.add(d);
    const d2 = new THREE.DirectionalLight(0x4a68c8, 0.35); d2.position.set(-6, 3, 4); s.add(d2);
    this.fireLights = [];
    for (let i = 0; i < 4; i++) { const l = new THREE.PointLight(0xff6a1f, 0, 6.5, 1.6); s.add(l); this.fireLights.push(l); }
    this.exitLight = new THREE.PointLight(0x35e07a, 5, 5, 1.7); s.add(this.exitLight);
    this.coldLights = [0, 1, 2].map(() => { const l = new THREE.PointLight(0x9cc2ff, 7, 11, 1.4); s.add(l); return l; });
    this.crabLight = new THREE.PointLight(0xbfe0ff, 0.55, 2.0, 1.8); s.add(this.crabLight);
  }

  // ------------------------------------------------------------ room
  loadRoom(room) {
    if (this.roomGroup) { this.scene.remove(this.roomGroup); this.sem.clear(); }
    this.room = room; const { w, h, cell } = room; const W = w * cell, H = h * cell;
    const g = new THREE.Group(); this.roomGroup = g; this.scene.add(g);
    // floor
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(W, H), new THREE.MeshStandardMaterial({ map: floorTexture(room), roughness: 0.62, metalness: 0.12 }));
    floor.position.set(W / 2, -H / 2, 0); g.add(floor);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.MeshStandardMaterial({ color: 0x0b0e14, roughness: 1 }));
    ground.position.set(W / 2, -H / 2, -0.02); g.add(ground);
    // walls / obstacles / exits
    const cells = { wall: [], obst: [], exit: [] };
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const v = room.grid[y][x];
      if (v === WALL) cells.wall.push([x, y]); else if (v === OBST) cells.obst.push([x, y]); else if (v === EXIT) cells.exit.push([x, y]);
    }
    const mkInst = (geom, mat, list, zFn, sFn) => {
      const m = new THREE.InstancedMesh(geom, mat, Math.max(1, list.length)); const o = new THREE.Object3D(); const r = rng(room.seed + 5);
      list.forEach(([x, y], i) => { o.position.set((x + 0.5) * cell, -(y + 0.5) * cell, zFn(i, r)); const s = sFn ? sFn(i, r) : 1; o.scale.set(1, 1, s); o.rotation.set(0, 0, 0); o.updateMatrix(); m.setMatrixAt(i, o.matrix); });
      m.count = list.length; return m;
    };
    g.add(mkInst(new THREE.BoxGeometry(cell, cell, WALL_H), new THREE.MeshStandardMaterial({ color: 0x45526c, roughness: 0.7, metalness: 0.15 }), cells.wall, () => WALL_H / 2));
    g.add(mkInst(new THREE.BoxGeometry(cell * 0.9, cell * 0.9, 0.03), new THREE.MeshBasicMaterial({ color: 0x1d5f9c }), cells.wall, () => WALL_H + 0.016));
    const hs = [0.5, 0.62, 0.4, 0.55, 0.7, 0.45, 0.58];
    const crates = mkInst(new THREE.BoxGeometry(cell * 0.9, cell * 0.9, 1), new THREE.MeshStandardMaterial({ map: woodTexture(), roughness: 0.85 }), cells.obst, (i) => hs[i % hs.length] / 2, (i) => hs[i % hs.length]);
    g.add(crates);
    // exit: glowing tiles, arch, sign, safe zone outside the west wall
    const ex = cells.exit; const ey = ex.length ? (ex[0][1] + ex.length / 2) * cell : H / 2;
    for (const [x, y] of ex) {
      const t = new THREE.Mesh(new THREE.BoxGeometry(cell, cell, 0.03), new THREE.MeshStandardMaterial({ color: 0x1c8f4c, emissive: 0x28be5a, emissiveIntensity: 0.9, roughness: 0.4 }));
      t.position.set((x + 0.5) * cell, -(y + 0.5) * cell, 0.015); g.add(t);
    }
    const top = ex.length ? ex[0][1] * cell : 0, bot = top + ex.length * cell;
    const glow = new THREE.MeshBasicMaterial({ color: 0x35e07a });
    for (const yy of [top, bot]) { const p = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.08, 1.15), glow); p.position.set(0.04, -yy, 0.575); g.add(p); }
    const lint = new THREE.Mesh(new THREE.BoxGeometry(0.08, bot - top + 0.16, 0.08), glow); lint.position.set(0.04, -(top + bot) / 2, 1.15); g.add(lint);
    const sign = new THREE.Mesh(new THREE.PlaneGeometry(0.62, 0.2), new THREE.MeshBasicMaterial({ map: labelTexture('EXIT', '#eafff1', '#12924a', 256, 80) }));
    sign.position.set(0.1, -(top + bot) / 2, 1.3); sign.rotation.order = 'ZXY'; sign.rotation.set(Math.PI / 2, 0, Math.PI / 2); g.add(sign);
    const pad = new THREE.Mesh(new THREE.PlaneGeometry(2.4, 1.6), new THREE.MeshStandardMaterial({ map: labelTexture('SAFE ZONE', '#d6ffe6', '#0e5a30', 768, 512), emissive: 0x0f7a3c, emissiveIntensity: 0.55, roughness: 0.6 }));
    pad.position.set(-1.25, -(top + bot) / 2, 0.004); g.add(pad);
    this.exitLight.position.set(0.6, -(top + bot) / 2, 0.9);
    this.coldLights.forEach((l, i) => l.position.set(W * (0.28 + 0.28 * i), -H / 2 + (i % 2 ? 1.2 : -1.2), 2.6));
    this.exitXY = [0.25, (top + bot) / 2];
    // hazard tape around the door in the interior wall
    const stripe = stripeTexture();
    const door = room.door_cells || [];
    if (door.length) {
      const dx = door[0][0], dy0 = Math.min(...door.map((d) => d[1])), dy1 = Math.max(...door.map((d) => d[1])) + 1;
      const mk = (x0, y0, x1, y1) => {
        const t = stripe.clone(); t.needsUpdate = true; t.repeat.set(Math.max(1, Math.round(Math.hypot(x1 - x0, y1 - y0) / 0.25)), 1);
        const m = new THREE.Mesh(new THREE.PlaneGeometry(Math.abs(x1 - x0) || 0.07, Math.abs(y1 - y0) || 0.07), new THREE.MeshBasicMaterial({ map: t }));
        m.position.set((x0 + x1) / 2, -(y0 + y1) / 2, 0.006); g.add(m);
      };
      for (const side of [dx * cell - 0.04, (dx + 1) * cell + 0.04]) mk(side, dy0 * cell, side, dy1 * cell);
    }
    // fire cells (all passable cells can ignite; instances stay scaled to zero until ignition)
    this.fireCells = []; for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (room.grid[y][x] === FREE) this.fireCells.push([x, y]);
    this.cellIndex = (x, y) => y * w + x; this._buildFire(g);
    this._buildFigurine(g); this._buildSemantic(room);
    this.setIgnite({});
  }
  _buildFire(g) {
    const n = this.fireCells.length, cell = this.room.cell, r = rng(this.room.seed * 13 + 1);
    const center = new Float32Array(n * 3), seed = new Float32Array(n), ign = new Float32Array(n).fill(1e9);
    this.fireCells.forEach(([x, y], i) => { center[i * 3] = (x + 0.5) * cell; center[i * 3 + 1] = -(y + 0.5) * cell; center[i * 3 + 2] = 0; seed[i] = r(); });
    this.cellIdOf = new Map(this.fireCells.map(([x, y], i) => [y * this.room.w + x, i]));
    this._ign = ign; this._cellCenter = center; this._seed = seed;
    const mk = (geom, extra) => { const ig = new THREE.InstancedBufferGeometry(); ig.index = geom.index; ig.attributes.position = geom.attributes.position; ig.attributes.uv = geom.attributes.uv; ig.instanceCount = n; for (const [k, v] of Object.entries(extra)) ig.setAttribute(k, v); return ig; };
    const igAttr = new THREE.InstancedBufferAttribute(ign, 1); igAttr.setUsage(THREE.DynamicDrawUsage); this._igAttr = igAttr;
    const base = new THREE.PlaneGeometry(1, 1);
    const baseAttrs = { aCenter: new THREE.InstancedBufferAttribute(center, 3), aIgnite: igAttr, aSeed: new THREE.InstancedBufferAttribute(seed, 1) };
    const U = (mode) => ({ uSimT: this.uni.uSimT, uAnim: this.uni.uAnim, uMode: { value: mode } });
    const mat = (fs, mode, blending) => new THREE.ShaderMaterial({ vertexShader: BILLBOARD_VS, fragmentShader: fs, uniforms: U(mode), transparent: true, depthWrite: false, blending, side: THREE.DoubleSide });
    const flames = new THREE.Mesh(mk(base, baseAttrs), mat(FLAME_FS, 0, THREE.AdditiveBlending)); flames.frustumCulled = false; flames.renderOrder = 4; g.add(flames);
    // smoke / embers: many particles, each tied to one cell (fixed assignment, so no popping)
    const mkParticles = (count, mode, fs, blending, order) => {
      const c = new Float32Array(count * 3), ig = new Float32Array(count), sd = new Float32Array(count), pr = rng(77 + mode);
      const cellOf = new Uint16Array(count);
      for (let i = 0; i < count; i++) { cellOf[i] = Math.floor(pr() * n); sd[i] = pr(); }
      const a = new THREE.InstancedBufferAttribute(ig, 1); a.setUsage(THREE.DynamicDrawUsage);
      const attrs = { aCenter: new THREE.InstancedBufferAttribute(c, 3), aIgnite: a, aSeed: new THREE.InstancedBufferAttribute(sd, 1) };
      for (let i = 0; i < count; i++) { c[i * 3] = center[cellOf[i] * 3]; c[i * 3 + 1] = center[cellOf[i] * 3 + 1]; }
      const geom = mk(base, attrs); geom.instanceCount = count;
      const m = new THREE.Mesh(geom, mat(fs, mode, blending)); m.frustumCulled = false; m.renderOrder = order; g.add(m);
      return { igAttr: a, ig, cellOf };
    };
    this.smoke = mkParticles(520, 1, SMOKE_FS, THREE.NormalBlending, 6);
    this.embers = mkParticles(260, 2, EMBER_FS, THREE.AdditiveBlending, 7);
    // floor scorch + glow
    const gl = new THREE.Mesh(mk(base, baseAttrs), new THREE.ShaderMaterial({ vertexShader: GLOW_VS, fragmentShader: GLOW_FS, uniforms: { uSimT: this.uni.uSimT, uAnim: this.uni.uAnim }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
    gl.frustumCulled = false; gl.renderOrder = 2; g.add(gl);
    const ch = new THREE.Mesh(mk(base, baseAttrs), new THREE.ShaderMaterial({ vertexShader: GLOW_VS, fragmentShader: CHAR_FS, uniforms: { uSimT: this.uni.uSimT }, transparent: true, depthWrite: false }));
    ch.frustumCulled = false; ch.renderOrder = 1; g.add(ch);
  }
  setIgnite(map) {   // map: {cellIndex(y*w+x): igniteTime}
    this.ignite = map; const ign = this._ign; ign.fill(1e9);
    for (const [k, t] of Object.entries(map)) { const i = this.cellIdOf.get(+k); if (i !== undefined) ign[i] = t; }
    this._igAttr.needsUpdate = true;
    for (const p of [this.smoke, this.embers]) { for (let i = 0; i < p.cellOf.length; i++) p.ig[i] = ign[p.cellOf[i]]; p.igAttr.needsUpdate = true; }
    this._updateSem();
  }
  // ------------------------------------------------------------ figurine
  _buildFigurine(g) {
    const f = new THREE.Group(); const blue = new THREE.MeshStandardMaterial({ color: 0x2470ff, roughness: 0.4, emissive: 0x0a2a80, emissiveIntensity: 0.55 });
    const white = new THREE.MeshStandardMaterial({ color: 0xf2f4ff, roughness: 0.5, emissive: 0x333344, emissiveIntensity: 0.3 });
    const cap = (r, l, m, x, y, z, rx, ry, rz) => { const mesh = new THREE.Mesh(new THREE.CapsuleGeometry(r, l, 6, 12), m); mesh.position.set(x, y, z); mesh.rotation.set(rx, ry, rz); f.add(mesh); };
    // lying on its back, long axis = local +x, head at +x (total length ~0.30 m)
    cap(0.032, 0.07, blue, 0, 0, 0.032, 0, 0, Math.PI / 2);               // torso
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.03, 16, 12), white); head.position.set(0.095, 0, 0.034); f.add(head);
    cap(0.011, 0.07, blue, 0.045, 0.058, 0.026, 0, 0, Math.PI / 2 + 0.45);  // arms
    cap(0.011, 0.07, blue, 0.045, -0.058, 0.026, 0, 0, Math.PI / 2 - 0.45);
    cap(0.013, 0.085, blue, -0.105, 0.022, 0.026, 0, 0, Math.PI / 2 + 0.12); // legs
    cap(0.013, 0.085, blue, -0.105, -0.022, 0.026, 0, 0, Math.PI / 2 - 0.12);
    const halo = new THREE.Mesh(new THREE.RingGeometry(0.2, 0.235, 40), new THREE.MeshBasicMaterial({ color: 0x59b0ff, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false }));
    halo.position.z = 0.008; this.figHalo = halo; f.add(halo);
    const l = new THREE.PointLight(0x4f8fff, 0.9, 1.4, 2); l.position.z = 0.25; f.add(l);
    this.fig = f; g.add(f);
  }
  // ------------------------------------------------------------ segmentation proxy scene
  _buildSemantic(room) {
    const { w, h, cell } = room, W = w * cell, H = h * cell, s = this.sem;
    const quad = (x, y, c, z, ww = cell, hh = cell) => { const m = new THREE.Mesh(new THREE.PlaneGeometry(ww, hh), flat(c)); m.position.set(x, y, z); return m; };
    const box = (x, y, c, hgt) => { const m = new THREE.Mesh(new THREE.BoxGeometry(cell, cell, hgt), flat(c)); m.position.set(x, y, hgt / 2); return m; };
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const v = room.grid[y][x], cx = (x + 0.5) * cell, cy = -(y + 0.5) * cell;
      if (v === WALL) s.add(box(cx, cy, PALETTE.wall, WALL_H));
      else if (v === OBST) s.add(box(cx, cy, PALETTE.obstacle, 0.5));
      else if (v === EXIT) s.add(quad(cx, cy, PALETTE.exit, 0.02));
    }
    this.semSmoke = []; this.semFire = [];
    this.fireCells.forEach(([x, y]) => {
      const cx = (x + 0.5) * cell, cy = -(y + 0.5) * cell;
      const sm = quad(cx, cy, PALETTE.smoke, 0.004); sm.visible = false; s.add(sm); this.semSmoke.push(sm);
      const fr = quad(cx, cy, PALETTE.fire, 0.01); fr.visible = false; s.add(fr); this.semFire.push(fr);
    });
    this.semFig = new THREE.Mesh(new THREE.CircleGeometry(0.32 * cell, 24), flat(PALETTE.figurine)); this.semFig.position.z = 0.03; s.add(this.semFig);
    const tri = new THREE.Shape(); const R = 0.42 * cell; // same triangle as the 2D renderer, heading along +x
    tri.moveTo(R, 0); tri.lineTo(R * Math.cos(2.5), R * Math.sin(2.5)); tri.lineTo(R * Math.cos(-2.5), R * Math.sin(-2.5));
    this.semCrab = new THREE.Mesh(new THREE.ShapeGeometry(tri), flat(PALETTE.crab)); this.semCrab.position.z = 0.04; s.add(this.semCrab);
  }
  _updateSem() {
    if (!this.semFire) return; const t = this.uni.uSimT.value, { w, h } = this.room;
    const burning = new Uint8Array(w * h);
    this.fireCells.forEach(([x, y], i) => { if (this._ign[i] <= t) burning[y * w + x] = 1; });
    this.fireCells.forEach(([x, y], i) => {
      let near = false;
      for (let dy = -2; dy <= 2 && !near; dy++) for (let dx = -2; dx <= 2; dx++) { const xx = x + dx, yy = y + dy; if (xx >= 0 && yy >= 0 && xx < w && yy < h && burning[yy * w + xx]) { near = true; break; } }
      this.semFire[i].visible = !!burning[y * w + x]; this.semSmoke[i].visible = near && !burning[y * w + x];
    });
  }
  burningList() { const t = this.uni.uSimT.value, out = []; this.fireCells.forEach(([x, y], i) => { if (this._ign[i] <= t) out.push([x, y, this._ign[i]]); }); return out; }

  // ------------------------------------------------------------ per-frame state
  // st: {t, x, y, th, fx, fy, carried, gait:{phase,vn,wn,moving}, anim:{name,p}|null, figZ}
  setSim(st) {
    this.state = st; this.uni.uSimT.value = st.t; this.uni.uAnim.value = this.deterministic ? st.t : this.animClock;
    this._updateSem();
    const c = this.crabHolder, yaw = -st.th, g = st.gait || { phase: 0, vn: 0, wn: 0, moving: false };
    const bob = g.moving ? 0.0035 * Math.sin(g.phase * 2) : 0;
    c.position.set(st.x, -st.y, this.baseH + bob); c.rotation.set(0, 0, yaw);
    this.crab.applyGait(g.phase, g.vn, g.wn, g.moving);
    this._arms(st);
    // figurine
    const carried = st.carried || (st.anim && (st.anim.name === 'release'));
    const f = this.fig; f.position.set(st.fx, -st.fy, st.figZ ?? 0);
    f.rotation.z = st.figYaw ?? 0; this.figHalo.visible = !st.carried && !st.rescued; this.figHalo.material.opacity = 0.35 + 0.2 * Math.sin(this.uni.uAnim.value * 4);
    this.crabLight.position.set(st.x, -st.y, 0.5);
    // fire lights: farthest-first sample of burning cells, flicker by animation clock
    const burn = this.burningList(); const picks = []; const cell = this.room.cell;
    for (const L of this.fireLights) { L.intensity = 0; }
    if (burn.length) {
      const age = (b) => clamp((st.t - b[2]) / 2.0, 0, 1);
      let first = burn[0]; picks.push(first);
      while (picks.length < this.fireLights.length) {
        let best = null, bd = -1;
        for (const b of burn) { const d = Math.min(...picks.map((p) => Math.hypot(p[0] - b[0], p[1] - b[1]))); if (d > bd) { bd = d; best = b; } }
        if (!best || bd < 1.5) break; picks.push(best);
      }
      picks.forEach((p, i) => { const L = this.fireLights[i]; L.position.set((p[0] + 0.5) * cell, -(p[1] + 0.5) * cell, 0.6);
        const n = burn.filter((b) => Math.hypot(b[0] - p[0], b[1] - p[1]) < 3).length;
        L.intensity = (4 + Math.min(n, 14) * 0.8) * age(p) * (0.88 + 0.12 * Math.sin(this.uni.uAnim.value * 11 + i * 2)); });
    }
  }
  _arms(st) {
    const rig = this.crab, a = st.arm;   // a: {L:{q:[j0..j4]}, R:{...}} absolute joint targets, or null for home
    for (const side of ['L', 'R']) for (let i = 0; i < 5; i++) {
      const n = `${side}F_J${i}_joint`; rig.setJoint(n, a && a[side] ? a[side][i] : (rig.home[n] ?? 0));
    }
  }

  // ------------------------------------------------------------ rendering
  resize(w, h) { this.renderer.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); }
  renderMain() { this.renderer.render(this.scene, this.camera); }
  _topCam() {
    const W = this.room.w * this.room.cell, H = this.room.h * this.room.cell;
    const c = new THREE.OrthographicCamera(-W / 2, W / 2, H / 2, -H / 2, 0.1, 40); c.up.set(0, 1, 0);
    c.position.set(W / 2, -H / 2, 20); c.lookAt(W / 2, -H / 2, 0); return c;
  }
  // kind: 'lit' (what a camera sees) or 'semantic' (flat class colours, exact palette)
  renderObs(kind, wpx, hpx) {
    const r = this.renderer, prev = r.getPixelRatio(), size = r.getSize(new THREE.Vector2());
    r.setPixelRatio(1); r.setSize(wpx, hpx, false);
    const cam = this._topCam();
    if (kind === 'semantic') {
      const st = this.state; this.semFig.position.set(st.fx, -st.fy, 0.03);
      this.semCrab.position.set(st.x, -st.y, 0.04); this.semCrab.rotation.z = -st.th;
      const tm = r.toneMapping; r.toneMapping = THREE.NoToneMapping; r.render(this.sem, cam); r.toneMapping = tm;
    } else { const ex = r.toneMappingExposure; r.toneMappingExposure = 1.7; r.render(this.scene, cam); r.toneMappingExposure = ex; }
    const url = r.domElement.toDataURL('image/png');
    r.setPixelRatio(prev); r.setSize(size.x, size.y, false);
    return url;
  }
  project(v3, out = { x: 0, y: 0, z: 0 }) {
    TMP.copy(v3).project(this.camera); out.x = (TMP.x * 0.5 + 0.5); out.y = (-TMP.y * 0.5 + 0.5); out.z = TMP.z; return out;
  }
  // ------------------------------------------------------------ overlays
  clearDyn() { this.dyn.clear(); }
  addRoute(points, color, dashed) {
    const m = ribbon(points.map((p) => ({ x: p[0], y: -p[1] })), 0.028, color, dashed); m.material.uniforms.uAnim = this.uni.uAnim; this.dyn.add(m); return m;
  }
  addRing(x, y, color, r = 0.12, label) {
    const m = new THREE.Mesh(new THREE.RingGeometry(r * 0.78, r, 36), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false }));
    m.position.set(x, -y, 0.02); this.dyn.add(m); return m;
  }
  addScanline(t01, color = 0x59d0ff) {
    const W = this.room.w * this.room.cell, H = this.room.h * this.room.cell;
    const m = new THREE.Mesh(new THREE.PlaneGeometry(0.06, H), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
    m.position.set(W * t01, -H / 2, 0.03); this.dyn.add(m);
    const tail = new THREE.Mesh(new THREE.PlaneGeometry(Math.max(0.01, W * t01), H), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.07, blending: THREE.AdditiveBlending, depthWrite: false }));
    tail.position.set(W * t01 / 2, -H / 2, 0.025); this.dyn.add(tail);
  }
  // body-frame point -> world (three) coords for the current crab pose
  crabPointToWorld(p) { this.crabHolder.updateMatrixWorld(true); return this.crabHolder.localToWorld(new THREE.Vector3(...p)); }
}

// ---------------------------------------------------------------- arm IK (damped least squares)
export function solveArm(engine, side, target /* base-frame THREE.Vector3 */, seed, iters = 60) {
  const rig = engine.crab, names = [0, 1, 2, 3].map((i) => `${side}F_J${i}_joint`), tip = rig.links[`${side}F_finger_tip_link`];
  const holder = engine.crabHolder; holder.position.set(0, 0, 0); holder.rotation.set(0, 0, 0);
  const q = seed.slice(0, 4); const out = new THREE.Vector3();
  const tipPos = () => { rig.root.updateMatrixWorld(true); return tip.getWorldPosition(out).clone(); };
  const apply = (v) => v.forEach((x, i) => rig.setJoint(names[i], x));
  for (let it = 0; it < iters; it++) {
    apply(q); const p = tipPos(); const e = target.clone().sub(p); if (e.length() < 0.004) break;
    const J = [];
    for (let i = 0; i < 4; i++) { const q2 = q.slice(); q2[i] += 1e-3; apply(q2); const p2 = tipPos(); J.push(p2.sub(p).multiplyScalar(1e3)); }
    // J is 4 columns of 3-vectors: dq = J^T (J J^T + l I)^-1 e
    const lam = 0.002, A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (const c of J) { const a = [c.x, c.y, c.z]; for (let r = 0; r < 3; r++) for (let k = 0; k < 3; k++) A[r][k] += a[r] * a[k]; }
    for (let r = 0; r < 3; r++) A[r][r] += lam;
    const det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    if (Math.abs(det) < 1e-12) break;
    const inv = (m) => { const d = det; return [[(m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d, (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / d, (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d], [(m[1][2] * m[2][0] - m[1][0] * m[2][2]) / d, (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d, (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / d], [(m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d, (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / d, (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d]]; };
    const Ai = inv(A), ev = [e.x, e.y, e.z], y = Ai.map((row) => row[0] * ev[0] + row[1] * ev[1] + row[2] * ev[2]);
    for (let i = 0; i < 4; i++) { const dq = J[i].x * y[0] + J[i].y * y[1] + J[i].z * y[2]; q[i] += clamp(dq, -0.25, 0.25); const d = rig.joints[names[i]].def; q[i] = clamp(q[i], d.lower, d.upper); }
  }
  apply(q); const err = target.distanceTo(tipPos());
  return { q, err };
}
export { wrapPi, clamp, smooth, rng };
