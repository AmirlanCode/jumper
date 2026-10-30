// Headless driver: lets crab_rescue.mission3d push simulator state in and read camera frames out.
import { Engine, PALETTE } from './engine.js';
(async () => {
  const eng = new Engine(document.getElementById('c'), { driver: true, pixelRatio: 1, deterministic: true });
  await eng.init('assets');
  eng.resize(640, 512);
  window.driver = {
    ready: true, palette: PALETTE,
    loadRoom(room) { eng.loadRoom(room); return true; },
    // st: {t,x,y,th,fx,fy,carried}, ignite: [[cellIndex, t], ...]
    setState(st, ignite) {
      const m = {}; (ignite || []).forEach(([k, t]) => { m[k] = t; }); eng.setIgnite(m);
      eng.setSim({ ...st, gait: { phase: 0, vn: 0, wn: 0, moving: false }, anim: null, arm: null, figZ: st.carried ? 0.115 : 0, figYaw: st.carried ? -st.th + Math.PI / 2 : 0.3 });
      return true;
    },
    obs(kind, w, h) { return eng.renderObs(kind, w, h); },
  };
})();
