/* The hero's background: a few hundred faint dots drifting as one loose swarm that every so often
   pools toward the middle and lets go again. Low contrast, ~30 fps, paused off screen or in a
   hidden tab, a single still frame for reduced motion. */
(() => {
  "use strict";
  const c = document.getElementById("swarm");
  if (!c || !c.getContext) return;
  const ctx = c.getContext("2d");
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const TAU = Math.PI * 2;
  let W = 0, H = 0, P = [], t = 0, raf = 0, last = 0, visible = true;

  // each dot has a home on a slowly turning ellipse; the swarm's spread breathes between loose and pooled
  function seed() {
    const n = W < 640 ? 150 : W < 1100 ? 240 : 320;
    let s = 7;
    const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
    P = [];
    for (let i = 0; i < n; i++) {
      const r = Math.sqrt(rnd()) * (.25 + rnd() * .85);
      P.push({
        a: rnd() * TAU, r, w: (.03 + rnd() * .05) * (rnd() < .5 ? -1 : 1) * (1.2 - r * .6),
        ph: rnd() * TAU, f: .4 + rnd() * .8, z: .7 + rnd() * .9,
        blue: i % 19 === 0, x: 0, y: 0, vx: 0, vy: 0
      });
    }
    P.forEach(p => { const h = home(p, 0); p.x = h[0]; p.y = h[1]; });
  }
  // 0 = loose, 1 = pooled: a slow gather every ~14 s, held briefly, then released
  const pool = tt => { const k = (tt % 14) / 14; return k < .55 ? 0 : k < .72 ? ease((k - .55) / .17) : k < .8 ? 1 : 1 - ease((k - .8) / .2); };
  const ease = x => x * x * (3 - 2 * x);
  function home(p, tt) {
    const g = pool(tt), spread = 1 - g * .62;
    const a = p.a + p.w * tt * (1 + g * 1.6);
    const cx = W / 2 + Math.sin(tt * .07) * W * .04, cy = H * .5 + Math.cos(tt * .09) * H * .05;
    const wob = 10 + 8 * (1 - g);
    return [
      cx + Math.cos(a) * p.r * W * .5 * spread + Math.sin(tt * p.f + p.ph) * wob,
      cy + Math.sin(a) * p.r * H * .52 * spread + Math.cos(tt * p.f * .8 + p.ph) * wob
    ];
  }
  function size() {
    const r = c.getBoundingClientRect();
    if (!r.width || !r.height) return false;
    const d = Math.min(devicePixelRatio || 1, 2);
    const nw = Math.round(r.width), nh = Math.round(r.height);
    if (nw !== W || nh !== H) {
      W = nw; H = nh; c.width = Math.round(W * d); c.height = Math.round(H * d);
      ctx.setTransform(d, 0, 0, d, 0, 0); seed();
    }
    return true;
  }
  function step(dt) {
    t += dt;
    const k = 1 - Math.pow(.02, dt), damp = Math.pow(.2, dt);
    for (const p of P) {
      const h = home(p, t);
      p.vx = (p.vx + (h[0] - p.x) * k * 2.2) * damp;
      p.vy = (p.vy + (h[1] - p.y) * k * 2.2) * damp;
      p.x += p.vx * dt * 6; p.y += p.vy * dt * 6;
    }
  }
  function draw() {
    ctx.clearRect(0, 0, W, H);
    const g = pool(t), a = .2 - g * .05;   // denser when pooled, so each dot gets fainter
    ctx.fillStyle = `rgba(20,22,29,${a.toFixed(3)})`;
    ctx.beginPath();
    for (const p of P) if (!p.blue) { ctx.moveTo(p.x + p.z, p.y); ctx.arc(p.x, p.y, p.z, 0, TAU); }
    ctx.fill();
    ctx.fillStyle = `rgba(49,82,255,${(.42 - g * .1).toFixed(3)})`;
    ctx.beginPath();
    for (const p of P) if (p.blue) { ctx.moveTo(p.x + p.z * 1.3, p.y); ctx.arc(p.x, p.y, p.z * 1.3, 0, TAU); }
    ctx.fill();
  }
  function frame(now) {
    raf = 0;
    const dt = last ? Math.min(.1, (now - last) / 1000) : 1 / 30;
    if (now - last >= 31 || !last) { last = now; step(dt); draw(); }
    wake();
  }
  const wake = () => { if (!RM && visible && !document.hidden && !raf) raf = requestAnimationFrame(frame); };
  const still = () => { if (!size()) return; t = 3; P.forEach(p => { const h = home(p, t); p.x = h[0]; p.y = h[1]; }); draw(); };

  if (RM) { still(); addEventListener("resize", still); return; }
  size(); draw();
  addEventListener("resize", () => { size(); draw(); });
  if ("IntersectionObserver" in window) new IntersectionObserver(es => {
    visible = es[0].isIntersecting;
    if (visible) { last = 0; wake(); } else if (raf) { cancelAnimationFrame(raf); raf = 0; }
  }).observe(c);
  document.addEventListener("visibilitychange", () => { last = 0; wake(); });
  wake();
  window.__swarm = { get t() { return t; }, set t(v) { t = v; P.forEach(p => { const h = home(p, t); p.x = h[0]; p.y = h[1]; }); draw(); } };
})();
