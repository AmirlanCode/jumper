// Crab Rescue 3D viewer: replays a recorded mission (mission.json) on the real Jumper crab rig.
// The viewer decides nothing: every pose, route, fire cell and model call comes from the log.
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { Engine, solveArm, clamp, smooth, wrapPi, PALETTE } from './engine.js';

const $ = (s, r = document) => r.querySelector(s);
const params = new URLSearchParams(location.search);
const REC = params.has('rec');
if (REC) document.body.classList.add('rec');

const PHASES = [['scan', 'SCAN'], ['route', 'ROUTE'], ['grasp', 'GRASP'], ['carry', 'CARRY'], ['release', 'RELEASE'], ['saved', 'SAVED']];
const PH_NAME = { scan: 'Scanning the room', to_figurine: 'Route to the figurine', grasp: 'Claw grasp', to_exit: 'Carrying to the exit', release: 'Handover to the safe zone', rescued: 'Rescue confirmed' };
const PH_STEP = { scan: 0, to_figurine: 1, grasp: 2, to_exit: 3, release: 4, rescued: 5 };
const SRC = {
  nvidia_model: ['NVIDIA MODEL', 'src-nv'], nebius_model: ['NEBIUS MODEL', 'src-nv'],
  offline_standin: ['OFFLINE STAND-IN', 'src-off'], safety_fallback: ['SAFETY FALLBACK · LOCAL A*', 'src-fb'], none: ['—', 'src-none'],
};

let eng, mission, F = [], derived = null, camMode = 'cinema', playing = !REC, speed = 1, T = 0, last = 0, controls, armSol = {};
const canvas = $('#c');

async function boot() {
  const base = params.get('base') || '.';
  eng = new Engine(canvas, { deterministic: REC, pixelRatio: REC ? 1 : Math.min(2, window.devicePixelRatio || 1) });
  await eng.init(`${base}/assets`);
  const url = params.get('mission') || `${base}/mission.json`;
  mission = await fetch(url).then((r) => r.json());
  window.__base = base;
  controls = new OrbitControls(eng.camera, canvas); controls.enableDamping = true; controls.enabled = false;
  loadMission(mission);
  window.addEventListener('resize', onResize); onResize();
  bindUI();
  if (REC) {
    window.__rec = { duration: derived.duration, seek: (t) => { T = t; frame(0, true); return true; }, ready: true };
  } else { requestAnimationFrame(tick); }
  document.body.classList.add('ready');
}

function onResize() {
  const w = canvas.clientWidth, h = canvas.clientHeight; eng.resize(w, h);
  if (!REC) frame(0, true);
}

// ---------------------------------------------------------------- mission -> timeline
function loadMission(m) {
  mission = m; const room = m.room;
  eng.loadRoom(room);
  const ign = {}; (m.ignite || []).forEach(([k, t]) => { ign[k] = t; }); eng.setIgnite(ign);
  // frames: [T, t, x, y, th, fx, fy, carried, anim, p]
  F = m.frames.map((a) => ({ T: a[0], t: a[1], x: a[2], y: a[3], th: a[4], fx: a[5], fy: a[6], carried: !!a[7], anim: a[8] || 0, p: a[9] || 0 }));
  const n = F.length; let phase = 0;
  for (let i = 0; i < n; i++) {
    const f = F[i], a = F[Math.max(0, i - 3)], b = F[Math.min(n - 1, i + 3)], dt = b.T - a.T || 1e-6;
    const dx = b.x - a.x, dy = b.y - a.y, hd = f.th;
    const v = (dx * Math.cos(hd) + dy * Math.sin(hd)) / dt;
    const w = -wrapPi(b.th - a.th) / dt;                // ccw positive in the (y-up) world
    f.vn = clamp(v / 0.55, -1.2, 1.2); f.wn = clamp(w / 2.5, -1.2, 1.2);
    f.moving = f.anim === 0 && (Math.abs(v) > 0.04 || Math.abs(w) > 0.2);
    const sp = Math.max(Math.abs(f.vn), Math.abs(f.wn) * 0.7);
    if (i) { const dtt = f.T - F[i - 1].T; phase += (f.moving ? 2 * Math.PI * (0.9 + 2.6 * sp) : 0) * dtt; }
    f.phase = phase;
  }
  // phases by hold / carried state
  const pickI = F.findIndex((f) => f.anim === 1), relI = F.findIndex((f) => f.anim === 2);
  F.forEach((f, i) => {
    f.ph = f.anim === 3 ? 'scan' : f.anim === 4 ? 'rescued' : f.anim === 1 ? 'grasp' : f.anim === 2 ? 'release'
      : (pickI < 0 || i < pickI) ? 'to_figurine' : (relI < 0 || i < relI) ? 'to_exit' : 'rescued';
  });
  // figurine: where it lay before the grasp hold, and how it is oriented
  const rg = rng32(room.seed * 17 + 3); derived = { pickI, relI, duration: F[n - 1].T, figYaw0: (rg() - 0.5) * Math.PI * 1.2 };
  derived.fig0 = pickI > 0 ? [F[pickI - 1].fx, F[pickI - 1].fy] : [F[0].fx, F[0].fy];
  // IK for the claws (once): home -> pre-grasp -> grasp -> carry
  armSol = solveGrasp();
  // calls + cumulative counters (by replay time)
  derived.calls = (m.calls || []).slice().sort((a, b) => a.T - b.T);
  let cum = { n: 0, tin: 0, tout: 0, usd: 0 }; derived.calls.forEach((c) => { cum = { n: cum.n + (c.counted === false ? 0 : 1), tin: cum.tin + (c.tokens_in || 0), tout: cum.tout + (c.tokens_out || 0), usd: cum.usd + (c.cost_usd || 0) }; c.cum = cum; });
  derived.plans = (m.plans || []).slice().sort((a, b) => a.T - b.T);
  buildCallList(); buildTimelineMarks(); fillMeta(); buildChapters(); T = 0; frame(0, true);
}
function rng32(seed) { let a = seed >>> 0; return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

function solveGrasp() {
  const rig = eng.crab, bh = eng.baseH, out = { err: {} };
  const home = (s) => [0, 1, 2, 3, 4].map((i) => rig.home[`${s}F_J${i}_joint`] ?? 0);
  const stage = (name, z, off) => {
    out[name] = {};
    for (const s of ['L', 'R']) {
      const y = (s === 'L' ? 1 : -1) * off;                   // body frame: +y is the crab's left
      const seed = name === 'pre' ? home(s) : out[name === 'grasp' ? 'pre' : 'grasp'][s];
      const r = solveArm(eng, s, new THREE.Vector3(0.30, y, z - bh), seed, 90);
      out[name][s] = [...r.q, 0]; out.err[`${name}${s}`] = +r.err.toFixed(4);
    }
  };
  stage('pre', 0.10, 0.11); stage('grasp', 0.045, 0.085); stage('carry', 0.115, 0.085);
  eng.crab.setHome();
  const closed = { L: -1.05, R: 1.05 };
  for (const k of ['grasp', 'carry']) for (const s of ['L', 'R']) out[k][s + 'c'] = out[k][s].slice(0, 4).concat([closed[s]]);
  out.homeQ = { L: home('L'), R: home('R') };
  console.log('IK errors (m)', out.err);
  return out;
}

const lerpArr = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
function armPose(f) {
  const S = armSol; if (!S.pre) return null; const H = S.homeQ; const e = (t) => smooth(0, 1, t);
  const mix = (A, B, t) => ({ L: lerpArr(A.L, B.L, e(t)), R: lerpArr(A.R, B.R, e(t)) });
  const Gc = { L: S.grasp.Lc, R: S.grasp.Rc }, Cc = { L: S.carry.Lc, R: S.carry.Rc }, G = S.grasp, P = S.pre;
  if (f.anim === 1) {
    const p = f.p;
    if (p < 0.25) return mix(H, P, p / 0.25);
    if (p < 0.5) return mix(P, G, (p - 0.25) / 0.25);
    if (p < 0.65) return mix(G, Gc, (p - 0.5) / 0.15);
    return mix(Gc, Cc, (p - 0.65) / 0.35);
  }
  if (f.anim === 2) {
    const p = f.p;
    if (p < 0.35) return mix(Cc, Gc, p / 0.35);
    if (p < 0.5) return mix(Gc, G, (p - 0.35) / 0.15);
    return mix(G, H, (p - 0.5) / 0.5);
  }
  if (f.carried) return Cc;
  if (f.anim === 3) { // scan: a small "raise claws" greeting
    const p = f.p, w = Math.sin(clamp(p * 1.4, 0, 1) * Math.PI);
    return mix(H, P, 0.35 * w);
  }
  return null;
}

// ---------------------------------------------------------------- state at time T
function sample(Tq) {
  const n = F.length; let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (F[mid].T <= Tq) lo = mid; else hi = mid; }
  const a = F[lo], b = F[hi], u = b.T > a.T ? clamp((Tq - a.T) / (b.T - a.T), 0, 1) : 0;
  const L = (p, q) => p + (q - p) * u;
  return { a, b, u, T: Tq, t: L(a.t, b.t), x: L(a.x, b.x), y: L(a.y, b.y), th: a.th + wrapPi(b.th - a.th) * u, fx: L(a.fx, b.fx), fy: L(a.fy, b.fy), carried: a.carried, anim: a.anim, p: L(a.p, b.p), ph: a.ph, vn: L(a.vn, b.vn), wn: L(a.wn, b.wn), moving: a.moving || b.moving, phase: L(a.phase, b.phase), i: lo };
}
function simState(s) {
  const st = { t: s.t, x: s.x, y: s.y, th: s.th, fx: s.fx, fy: s.fy, carried: s.carried, anim: s.anim ? { name: s.anim === 1 ? 'pickup' : s.anim === 2 ? 'release' : 'x', p: s.p } : null,
    gait: { phase: s.phase, vn: s.vn, wn: s.wn, moving: s.moving }, figZ: 0, figYaw: derived.figYaw0, rescued: s.ph === 'rescued' };
  const heading = s.th; const cy = -s.th + Math.PI / 2;      // lying across the claws (three yaw)
  if (s.anim === 1) {
    const e = smooth(0.45, 0.9, s.p);
    st.fx = derived.fig0[0] + (s.fx - derived.fig0[0]) * e; st.fy = derived.fig0[1] + (s.fy - derived.fig0[1]) * e;
    st.figZ = 0.115 * smooth(0.65, 1, s.p); st.figYaw = derived.figYaw0 + wrapPi(cy - derived.figYaw0) * smooth(0.3, 0.7, s.p); st.carried = false;
  } else if (s.carried) { st.figZ = 0.115; st.figYaw = cy; }
  else if (s.anim === 2) { st.figZ = 0.115 * (1 - smooth(0, 0.35, s.p)); st.figYaw = cy; }
  else if (derived.relI >= 0 && s.i >= derived.relI) { st.figYaw = -F[derived.relI].th + Math.PI / 2; }
  const ap = armPose(s); st.arm = ap;
  return st;
}

// ---------------------------------------------------------------- overlays: plan route, trail, perception
let ovKey = '';
function activePlan(Tq) { let p = null; for (const q of derived.plans) { if (q.T <= Tq) p = q; else break; } return p; }
function updateOverlays(s) {
  const plan = activePlan(s.T); const pk = `${plan ? plan.T : -1}|${Math.floor(s.T * 2)}`;
  const rem = plan ? plan.waypoints.filter((w) => !w.done_by || w.done_by > s.T) : [];
  const key = `${plan ? plan.T : -1}|${rem.length}|${s.T < 3.2 ? Math.floor(s.T * 20) : 'x'}|${recentPerception(s.T) ? 1 : 0}`;
  if (key === ovKey && !(s.T < 3.2)) return; ovKey = key; eng.clearDyn();
  if (plan) {
    const col = plan.source === 'safety_fallback' ? 0xffb02e : plan.source === 'offline_standin' ? 0x59d0ff : 0x76e000;
    const pts = [[s.x, s.y], ...rem.map((w) => [w.x, w.y])];
    if (pts.length > 1) eng.addRoute(pts, col, plan.source === 'safety_fallback');
    rem.forEach((w) => { if (w.action !== 'move') eng.addRing(w.x, w.y, w.action === 'pickup' ? 0x59b0ff : 0x35e07a, 0.17); else eng.addRing(w.x, w.y, col, 0.045); });
  }
  if (s.ph === 'scan') eng.addScanline(clamp(s.p * 1.25, 0, 1));
}
function recentPerception(Tq) { return derived.calls.some((c) => c.kind === 'perception' && c.T <= Tq && Tq - c.T < 4); }

// ---------------------------------------------------------------- camera director
function chaseTarget(Tq) { // smoothed crab position (moving average over +-1.5 s, deterministic)
  let sx = 0, sy = 0, sc = 0;
  for (let k = -15; k <= 15; k++) { const q = sample(clamp(Tq + k * 0.1, 0, derived.duration)); sx += q.x; sy += q.y; sc++; }
  return { x: sx / sc, y: sy / sc };
}
const W3 = () => mission.room.w * mission.room.cell, H3 = () => mission.room.h * mission.room.cell;
function shotWide(Tq) { const W = W3(), H = H3(), a = -Math.PI / 2 + 0.35 * Math.sin(Tq * 0.12); return { pos: [W / 2 + Math.cos(a) * 3, -H / 2 + Math.sin(a) * (H * 1.15), 8.5], tgt: [W / 2, -H / 2 + 0.6, 0], fov: 42 }; }
// The chase camera keeps one world bearing instead of following the crab's heading: a crab that
// turns in place before every step (model-driven runs) swung a heading-locked camera around it.
const CHASE_AZ = -Math.PI / 2 - 0.45;
function shotChase(Tq, dist, h) {
  const c = chaseTarget(Tq), wx = c.x, wy = -c.y;
  return { pos: [wx + Math.cos(CHASE_AZ) * dist, wy + Math.sin(CHASE_AZ) * dist, h], tgt: [wx, wy, 0.1], fov: 48 };
}
function shotClose(Tq, s0, dur, side) {
  const q = sample(Tq), hd3 = -q.th, a = hd3 + Math.PI / 2 * side + (Tq - s0) * 0.22 * side;
  const tx = q.x + Math.cos(q.th) * 0.28, ty = q.y + Math.sin(q.th) * 0.28;
  return { pos: [tx + Math.cos(a) * 0.78, -ty + Math.sin(a) * 0.78, 0.42], tgt: [tx, -ty, 0.1], fov: 42 };
}
function shotFor(ph, Tq) {
  const tp = phaseStart(ph);
  switch (ph) {
    case 'scan': return shotWide(Tq);
    case 'to_figurine': return shotChase(Tq, 2.4, 1.6);
    case 'grasp': return shotClose(Tq, tp, 2.4, 1);
    case 'to_exit': return shotChase(Tq, 2.3, 1.55);
    case 'release': return shotClose(Tq, tp, 2.4, -1);
    default: { const w = shotWide(Tq); const e = smooth(0, 2.2, Tq - tp); const c = shotClose(Tq, tp, 1, -1); return mixShot(c, w, e); }
  }
}
const phCache = {};
function phaseStart(ph) { if (phCache[ph] === undefined) { const f = F.find((x) => x.ph === ph); phCache[ph] = f ? f.T : 0; } return phCache[ph]; }
const mixShot = (a, b, t) => ({ pos: a.pos.map((v, i) => v + (b.pos[i] - v) * t), tgt: a.tgt.map((v, i) => v + (b.tgt[i] - v) * t), fov: a.fov + (b.fov - a.fov) * t });
const ORDER = ['scan', 'to_figurine', 'grasp', 'to_exit', 'release', 'rescued'];
// keep the camera out of walls and crates by lifting it on the same bearing: hopping to another
// side of the target (the old fallback) jumped the view whenever a wall came between them
function clearCam(c) {
  const R = mission.room, cell = R.cell, [tx, ty, tz] = c.tgt, [px, py, pz] = c.pos;
  const blocked = (x, y, z) => { const cx = Math.floor(x / cell), cy = Math.floor(-y / cell); if (cx < 0 || cy < 0 || cx >= R.w || cy >= R.h) return z < 3; const g = R.grid[cy][cx]; return (g === 1 && z < 1.35) || (g === 2 && z < 0.75); };
  const clear = (x, y, z) => { for (let i = 1; i <= 20; i++) { const u = i / 20; if (blocked(tx + (x - tx) * u, ty + (y - ty) * u, tz + (z - tz) * u)) return false; } return true; };
  const dx = px - tx, dy = py - ty;
  for (let k = 0; k <= 8; k++) {
    const pull = 1 - 0.06 * k, z = pz + 0.35 * k;
    if (clear(tx + dx * pull, ty + dy * pull, z)) return k ? { ...c, pos: [tx + dx * pull, ty + dy * pull, z] } : c;
  }
  return { ...c, pos: [tx + dx * 0.5, ty + dy * 0.5, pz + 3.0] };
}
// The rig eases toward each frame's shot with a 0.7 s time constant of replay time, so a shot
// that changes abruptly still moves the view smoothly. A jump in replay time (seek, chapter,
// first frame) snaps instead. Sequential video frames make it deterministic.
let rig = null;
function camera(Tq, s) {
  const want = clearCam(camera0(Tq, s));
  const dt = rig ? Tq - rig.T : Infinity;
  if (!rig || dt < 0 || dt > 0.5) { rig = { ...want, T: Tq }; return want; }
  const a = 1 - Math.exp(-dt / 0.7), mix = (p, q) => p.map((v, i) => v + (q[i] - v) * a);
  rig = { pos: mix(rig.pos, want.pos), tgt: mix(rig.tgt, want.tgt), fov: rig.fov + (want.fov - rig.fov) * a, T: Tq };
  return rig;
}
function camera0(Tq, s) {
  const ph = s.ph, cur = shotFor(ph, Tq), tp = phaseStart(ph), idx = ORDER.indexOf(ph);
  if (idx > 0 && Tq - tp < 1.1) { const prev = shotFor(ORDER[idx - 1], Tq); return mixShot(prev, cur, smooth(0, 1.1, Tq - tp)); }
  return cur;
}

// ---------------------------------------------------------------- frame
let sMem = null;
function frame(dtReal, force) {
  if (!derived) return;
  if (!REC) eng.animClock += dtReal;
  const s = sample(clamp(T, 0, derived.duration)); sMem = s;
  const st = simState(s); eng.setSim(st); updateOverlays(s);
  if (camMode === 'cinema' || REC) { const c = camera(s.T, s); eng.camera.position.set(...c.pos); eng.camera.fov = c.fov; eng.camera.lookAt(...c.tgt); eng.camera.updateProjectionMatrix(); controls.enabled = false; }
  else if (camMode === 'top') { const W = W3(), H = H3(); eng.camera.position.set(W / 2, -H / 2 - 0.01, 15); eng.camera.fov = 48; eng.camera.lookAt(W / 2, -H / 2, 0); eng.camera.updateProjectionMatrix(); controls.enabled = false; }
  else { if (!controls.enabled) { const c = sample(T); controls.target.set(c.x, -c.y, 0.1); controls.enabled = true; } controls.update(); }
  eng.renderMain(); hud(s); labels(s);
}
function tick(now) {
  const dt = Math.min(0.1, (now - last) / 1000 || 0); last = now;
  if (playing) { T += dt * speed; if (T >= derived.duration) { T = derived.duration; playing = false; syncPlay(); } }
  frame(dt); requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- HUD
const fmt = (v, d = 1) => (+v).toFixed(d);
function srcOf(plan) { return plan ? (SRC[plan.source] || SRC.none) : SRC.none; }
function hud(s) {
  const step = PH_STEP[s.ph] ?? 0;
  $('#phase-name').textContent = PH_NAME[s.ph] || '';
  const ph = $('#phases'); if (ph) ph.innerHTML = PHASES.map(([k, l], i) => `<span class="${i < step ? 'done' : i === step ? 'on' : ''}">${l}</span>`).join('');
  $('#clock').textContent = `${fmt(s.t)} s`;
  const burning = eng.burningList().length; $('#fire').textContent = `${burning} cells · ${fmt(burning * 0.25, 1)} m²`;
  const plan = activePlan(s.T), [sl, sc] = srcOf(plan);
  const el = $('#src'); el.textContent = sl; el.className = `badge ${sc}`;
  const cs = derived.calls.filter((c) => c.T <= s.T), cum = cs.length ? cs[cs.length - 1].cum : { n: 0, tin: 0, tout: 0, usd: 0 };
  const put = (id, f) => { const e = $(id); if (e) f(e); };
  put('#n-calls', (e) => { e.textContent = cum.n; }); put('#n-tok', (e) => { e.textContent = `${cum.tin} / ${cum.tout}`; }); put('#n-usd', (e) => { e.textContent = mission.meta.price_known === false ? 'price not set' : `$${cum.usd.toFixed(5)}`; });
  const callEls = $('#calls')?.children; if (callEls) { let li = -1; derived.calls.forEach((c, i) => { if (c.T <= s.T) li = i; }); if (li !== hud.li) { hud.li = li; [...callEls].forEach((el, i) => { el.style.opacity = i <= li ? 1 : 0.55; }); if (li >= 0 && callEls[li]) { const box = $('#calls'); box.scrollTop = Math.max(0, callEls[li].offsetTop - box.clientHeight + callEls[li].offsetHeight + 4); } } }
  const lastPerc = [...cs].reverse().find((c) => c.kind === 'perception'), lastPlan = [...cs].reverse().find((c) => c.kind !== 'perception');
  $('#last-action').innerHTML = lastActionText(plan, lastPerc, lastPlan, s);
  put('#perc', (e) => { e.innerHTML = lastPerc ? `${esc(lastPerc.model)} <i class="chip ${lastPerc.status === 'accepted' ? 'ok' : 'bad'}">${esc(lastPerc.status)}</i>` : '<span class="dim">waiting</span>'; });
  put('#plnr', (e) => { e.innerHTML = lastPlan ? `${esc(lastPlan.model || 'local A*')} <i class="chip ${lastPlan.status === 'accepted' ? 'ok' : 'bad'}">${esc(lastPlan.status)}</i>` : '<span class="dim">waiting</span>'; });
  if (!REC) { $('#seek').value = (s.T / derived.duration) * 1000; $('#tcur').textContent = `${fmt(s.T)} / ${fmt(derived.duration)} s`; }
  // title cards (video): fade in/out
  const tin = $('#title-in'), tout = $('#title-out');
  if (tin) { const a = s.T < 0.4 ? 1 : 1 - smooth(2.6, 3.4, s.T); tin.style.opacity = REC ? a : 0; }
  if (tout) { const a = smooth(derived.duration - 3.2, derived.duration - 2.4, s.T); tout.style.opacity = REC ? a : 0; }
}
function lastActionText(plan, lp, lpl, s) {
  if (s.ph === 'rescued') return `<b>Figurine in the safe zone.</b> The simulator confirmed the rescue at ${fmt(mission.meta.rescued_t ?? s.t)} s.`;
  if (s.ph === 'grasp') return '<b>pickup</b> · claws close on the figurine (visual grasp, not contact physics).';
  if (s.ph === 'release') return '<b>release</b> · figurine set down in the exit zone.';
  if (s.ph === 'scan') return '<b>Scanning the room.</b> The frame goes to the vision model.';
  const r = plan && plan.rationale ? esc(plan.rationale) : (lpl ? esc(lpl.summary || '') : '');
  return `<b>${plan ? plan.waypoints.length : 0} waypoints.</b> ${r}`;
}
const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// 3D labels (DOM) over figurine / exit / crab
function labels(s) {
  const box = $('#labels'); if (!box) return; const W = canvas.clientWidth, H = canvas.clientHeight; const out = [];
  const add = (x, y, z, text, cls) => { const p = eng.project(new THREE.Vector3(x, y, z)); if (p.z < 1 && p.x > 0 && p.x < 1 && p.y > 0 && p.y < 1) out.push(`<div class="lab ${cls}" style="left:${p.x * W}px;top:${p.y * H}px">${text}</div>`); };
  const fy = eng.fig.position;
  if (!sMem.carried && s.ph !== 'rescued') add(fy.x, fy.y, 0.5, 'FIGURINE', 'l-fig');
  add(eng.exitXY[0] + 0.2, -eng.exitXY[1], 1.6, 'EXIT · SAFE ZONE', 'l-exit');
  const rp = derived.calls.find((c) => c.kind === 'perception' && c.T <= s.T && s.T - c.T < 5);
  if (rp && rp.parsed) { const R = mission.room, Wm = R.w * R.cell, Hm = R.h * R.cell; const g = rp.grounding || {};
    if (rp.parsed.figurine) add(rp.parsed.figurine.x * Wm, -rp.parsed.figurine.y * Hm, 0.9, `${tag(rp)} figurine (${fmt(rp.parsed.figurine.x, 2)}, ${fmt(rp.parsed.figurine.y, 2)})${g.figurine ? ' · ' + g.figurine : ''}`, 'l-det');
    if (rp.parsed.exit) add(rp.parsed.exit.x * Wm, -rp.parsed.exit.y * Hm, 0.5, `${tag(rp)} exit (${fmt(rp.parsed.exit.x, 2)}, ${fmt(rp.parsed.exit.y, 2)})${g.exit ? ' · ' + g.exit : ''}`, 'l-det'); }
  box.innerHTML = out.join('');
}
const tag = (c) => (c.source === 'offline_standin' ? 'STAND-IN' : (c.model || '').split('/')[0].toUpperCase());

// ---------------------------------------------------------------- panels
function fillMeta() {
  const m = mission.meta, live = m.mode === 'nebius_live';
  const rib = $('#ribbon');
  rib.className = live ? 'notice live' : 'notice';
  rib.innerHTML = live ? '<b>Live run</b> via Nebius Token Factory' : '<b>Offline stand-in run.</b> No NVIDIA API calls were made. Local stand-ins with the same JSON contract made every decision.';
  const sd = $('#seed'); if (sd) sd.textContent = `seed ${m.seed}`;
  const pid = m.models.perception?.id || '—', lid = m.models.planner?.id || '—';
  const tk = $('#ticket-model'); if (tk) tk.textContent = live ? (pid === lid ? pid : `${pid} + ${lid}`) : 'offline stand-ins';
  $('#t-in-sub').textContent = live ? (pid === lid ? `Seeing and deciding: ${pid}` : `Vision: ${pid} · Planner: ${lid}`) : 'Offline run: model stand-ins, not NVIDIA';
  // run_note: what this particular run did differently, written by whoever produced the log
  const ci = $('#capsule-img'), fc = derived.calls.find((c) => c.kind === 'perception' && c.frame);
  if (ci && fc) { ci.src = `${window.__base}/${fc.frame}`; $('#capsule').hidden = false; }
  // video page: one player and a grid of the other recorded videos, listed in the log by whoever
  // assembled this page; a card plays its video in the big player
  const gs = $('#gallery'), vids = Array.isArray(m.gallery) ? m.gallery : [];
  if (gs) {
    gs.hidden = !vids.length;
    const DRV = { model: ['Model drove', 'src-nv'], astar: ['A* drove', 'src-fb'], none: ['Nobody moved', 'src-none'] };
    const player = $('#watch-player');
    const show = (i, play) => {
      const v = vids[i]; if (!v) return; const [dl, dc] = DRV[v.driver] || DRV.none;
      player.poster = v.poster || ''; player.src = v.src; if (play) player.play().catch(() => {});
      $('#watch-title').textContent = v.title; $('#watch-desc').textContent = v.result;
      const b = $('#watch-badge'); b.textContent = dl; b.className = `badge ${dc}`;
      gs.querySelectorAll('.card').forEach((c) => c.classList.toggle('now', +c.dataset.i === i));
    };
    $('#gallery-grid').innerHTML = vids.map((v, i) => { const [dl, dc] = DRV[v.driver] || DRV.none; return `<button class="card" data-i="${i}"><span class="thumb"><img src="${esc(v.preview || v.poster || '')}" alt="" loading="lazy" decoding="async"><i class="dur">${esc(v.duration || '')}</i><i class="playing">Now playing</i></span><span class="badge ${dc}">${dl}</span><b>${esc(v.title)}</b><span class="res">${esc(v.result)}</span></button>`; }).join('');
    gs.querySelectorAll('.card').forEach((c) => c.addEventListener('click', () => { show(+c.dataset.i, true); player.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }));
    show(0, false);
  }
  const rn = $('#this-run'), notes = Array.isArray(m.run_note) ? m.run_note : [];
  if (rn) { rn.hidden = !notes.length; $('#this-run-body').innerHTML = notes.map((n) => `<p><b>${esc(n.title || '')}</b> ${esc(n.text || '')}</p>`).join(''); }
  $('#t-out-sub').textContent = m.outcome === 'rescued' ? `Rescued in ${fmt(m.rescued_t)} s of simulation · ${m.budget.calls} model calls · ${live ? '$' + m.budget.usd.toFixed(5) : '$0 (offline)'}` : `Outcome: ${m.outcome}`;
  $('#t-out-note').textContent = 'Simulation only. It does not show that a real robot is ready for rescue work.';
  const dl = $('#dl-json'); if (dl) { dl.href = `data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(m))}`; dl.download = `mission_seed${m.seed}.json`; }
  const ir = $('#issue-run'); if (ir) ir.innerHTML = live ? '<b>Live run.</b> Model replies, parsed actions and any fallback are in the call log; a fallback is labelled wherever it was used.' : '<b>No verified NVIDIA run in this log.</b> This episode is an offline stand-in run: the environment that produced it could not reach Nebius.';
  const kv = (id, v) => { const e = $('#' + id); if (e) e.textContent = v; };
  kv('kv-mode', live ? 'Live (Nebius)' : 'Offline stand-in'); kv('kv-outcome', m.outcome === 'rescued' ? 'Rescued' : m.outcome);
  kv('kv-time', m.rescued_t ? `${fmt(m.rescued_t)} s (sim)` : '—'); kv('kv-calls', `${m.budget.calls}`);
  kv('kv-tokens', `${m.budget.tokens_in} / ${m.budget.tokens_out}`); kv('kv-cost', live ? `$${m.budget.usd.toFixed(4)}` : '$0');
  // one sentence on who routed the crab, read from the log rather than written into the page
  const fb = derived.calls.some((c) => c.source === 'safety_fallback'), mdl = derived.calls.some((c) => c.kind !== 'perception' && c.source === 'nvidia_model' && c.status === 'accepted');
  kv('honest-route', fb ? 'Some routes in this run came from the local A* fallback; the scoreboard shows when.' : mdl ? 'There is no path planner: the model chooses every move.' : 'No model route was accepted in this run.');
  const tb = $('#meta-table'); if (tb) tb.innerHTML = [
    ['Mode', live ? 'nebius_live' : 'offline_standin'], ['Vision model', `${m.models.perception?.id || '—'} (${m.models.perception?.vendor || ''})`],
    ['Planner', `${m.models.planner?.id || '—'} (${m.models.planner?.vendor || ''})`], ['Model calls', m.budget.calls], ['Tokens in / out', `${m.budget.tokens_in} / ${m.budget.tokens_out}`],
    ['Estimated cost', live ? `$${m.budget.usd.toFixed(5)} of a $${m.budget.cap_usd} cap` : '$0 (local stand-ins)'], ['Outcome', `${m.outcome}${m.rescued_t ? ` · ${fmt(m.rescued_t)} s` : ''}`],
    ['Walking', m.locomotion || ''], ['Grasp', m.grasp || ''], ['Fire', m.fire || ''],
  ].map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join('');
}
function buildCallList() {
  const box = $('#calls'); if (!box) return;
  box.innerHTML = derived.calls.map((c, i) => {
    const [sl, sc] = SRC[c.source] || SRC.none;
    return `<button class="call" data-i="${i}"><span class="ct">${fmt(c.t)}s</span><span class="ck">${c.kind === 'perception' ? 'Vision' : c.kind === 'plan' ? 'Plan' : 'Safe route'}</span><span class="cm">${esc(c.model || 'local A*')}</span><span class="badge ${sc}">${sl}</span><i class="chip ${c.status === 'accepted' ? 'ok' : 'bad'}">${esc(c.status)}</i><span class="cl">${c.latency_s ? fmt(c.latency_s, 2) + ' s' : ''}</span></button>`;
  }).join('') || '<div class="dim pad">The log is empty.</div>';
  box.querySelectorAll('.call').forEach((b) => b.addEventListener('click', () => openCall(derived.calls[+b.dataset.i])));
}
function openCall(c) {
  const d = $('#modal'); const img = c.frame ? `<figure><img src="${window.__base}/${c.frame}" alt="frame the model saw"><figcaption>Frame from the 3D scene sent to the model</figcaption></figure>` : '';
  $('#modal-body').innerHTML = `<h3>${c.kind} · t=${fmt(c.t)} s · ${esc(c.model || 'local A*')}</h3><p><span class="badge ${(SRC[c.source] || SRC.none)[1]}">${(SRC[c.source] || SRC.none)[0]}</span> <i class="chip ${c.status === 'accepted' ? 'ok' : 'bad'}">${esc(c.status)}</i> · ${c.latency_s ? fmt(c.latency_s, 2) + ' s' : 'no network'} · ${c.tokens_in || 0}/${c.tokens_out || 0} tokens · $${(c.cost_usd || 0).toFixed(6)}</p>${c.reasons?.length ? `<p class="bad-t">Rejected because: ${esc(c.reasons.join('; '))}</p>` : ''}${img}<h4>Parsed action</h4><pre>${esc(JSON.stringify(c.parsed ?? null, null, 1))}</pre><h4>Raw model reply (no secrets)</h4><pre>${esc(c.raw || '')}</pre>`;
  d.showModal();
}
function buildTimelineMarks() {
  const box = $('#marks'); if (!box) return; const D = derived.duration;
  const mk = (t, cls, title) => `<i class="mk ${cls}" style="left:${(t / D) * 100}%" title="${esc(title)}"></i>`;
  box.innerHTML = derived.calls.map((c) => mk(c.T, c.source === 'safety_fallback' ? 'fb' : c.source === 'offline_standin' ? 'off' : 'nv', `${c.kind} · ${c.model || 'local A*'} · ${c.status}`)).join('')
    + (derived.pickI >= 0 ? mk(F[derived.pickI].T, 'ev', 'pickup') : '') + (derived.relI >= 0 ? mk(F[derived.relI].T, 'ev', 'release') : '');
}
function buildChapters() {
  const box = $('#chapters'); if (!box) return;
  const items = [['scan', 'Scan'], ['to_figurine', 'Route'], ['grasp', 'Grasp'], ['to_exit', 'Carry'], ['release', 'Release'], ['rescued', 'Rescued']];
  box.innerHTML = items.map(([k, l], i) => `<button class="chap" data-ph="${k}"><b>${i + 1}</b><span>${l}</span></button>`).join('');
  box.querySelectorAll('.chap').forEach((b) => b.addEventListener('click', () => { T = phaseStart(b.dataset.ph) + 0.02; playing = true; syncPlay(); frame(0, true); }));
}
function syncPlay() { const b = $('#play'); if (b) b.textContent = playing ? '⏸' : '▶'; }
function bindUI() {
  if (REC) return;
  $('#play').addEventListener('click', () => { if (T >= derived.duration - 0.01) T = 0; playing = !playing; syncPlay(); });
  $('#seek').addEventListener('input', (e) => { T = (e.target.value / 1000) * derived.duration; playing = false; syncPlay(); frame(0, true); });
  $('#speed').addEventListener('change', (e) => { speed = +e.target.value; });
  document.querySelectorAll('[data-cam]').forEach((b) => b.addEventListener('click', () => { camMode = b.dataset.cam; document.querySelectorAll('[data-cam]').forEach((x) => x.classList.toggle('on', x === b)); controls.enabled = false; frame(0, true); }));
  $('#replay')?.addEventListener('click', () => { T = 0; playing = true; syncPlay(); frame(0, true); document.querySelector('#stage').scrollIntoView({ block: 'nearest', behavior: 'smooth' }); });
  $('#modal-close')?.addEventListener('click', () => $('#modal').close());
  $('#load').addEventListener('change', async (e) => { const f = e.target.files[0]; if (!f) return; try { const m = JSON.parse(await f.text()); if (m.schema !== 'crab-rescue-mission/1') throw new Error('not a crab-rescue mission log'); const le0 = $('#loaderr'); if (le0) le0.hidden = true; window.__base = '.'; Object.keys(phCache).forEach((k) => delete phCache[k]); loadMission(m); T = 0; playing = true; syncPlay(); } catch (err) { const le = $('#loaderr'); if (le) { le.hidden = false; le.textContent = `Could not open the log: ${err.message}`; } } });
  syncPlay();
}
boot().catch((e) => { console.error(e); const d = document.createElement('pre'); d.style.cssText = 'color:#f88;padding:16px;position:fixed;top:0;left:0;z-index:99'; d.textContent = `boot failed: ${e.message}`; document.body.appendChild(d); });
