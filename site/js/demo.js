/* The demo: one story in seven steps. Two browser tabs (a laptop starts a room, a desktop joins it
   with the code) merge into one room; the room picks a model and loads it across both; chat; a
   phone joins in the middle; Code mode writes a game, asks for approval, serves it; you play it.
   A step bar under the window says where we are and jumps to any step.
   The HTML holds the finished state (readable without JS). This script rewinds and replays it. */
(() => {
  "use strict";
  document.documentElement.classList.add("js"); // also set early by boot.js
  const $ = id => document.getElementById(id);
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- closer: the dots leave the three devices and become the logo ---------- */
  const closer = $("closer");
  if (closer && !RM && "IntersectionObserver" in window) {
    closer.classList.add("arm");
    const io = new IntersectionObserver(es => {
      if (!es[0].isIntersecting) return;
      closer.classList.add("go"); io.disconnect();
    }, { threshold: .45 });
    io.observe($("gather"));
  }

  const win = $("win");
  if (!win || !window.PooledTetris) return;
  const restart = (el, cls) => { el.classList.remove(cls); void el.offsetWidth; el.classList.add(cls); };
  const press = (el, t) => { restart(el, "press"); setTimeout(() => el.classList.remove("press"), t || 260); };
  const scene = s => { win.dataset.scene = s; };
  const flag = (cls, on) => win.classList.toggle(cls, on);

  /* ---------- steps and the bar under the window ---------- */
  const C = 26.0;                                   // Code mode starts
  const STEPS = [0, 3.6, 8.0, 13.2, 20.4, C, C + 12.2];
  const END = C + 15.0;
  const dotBtns = [...win.querySelectorAll(".sb-dots button")];
  const CAPS = dotBtns.map(b => b.querySelector(".lbl").textContent);
  const sbN = $("sbN"), sbT = $("sbT");
  let shownStep = -1;
  const stepAt = t => { let k = 0; STEPS.forEach((s, i) => { if (t >= s - 1e-6) k = i; }); return k; };
  const paintBar = (t, animate) => {
    const k = stepAt(t);
    if (k !== shownStep) {
      shownStep = k;
      sbN.textContent = `${k + 1}/${STEPS.length}`;
      sbT.textContent = CAPS[k];
      if (animate && !RM) restart(sbT, "in");
      dotBtns.forEach((b, i) => {
        b.classList.toggle("done", i < k); b.classList.toggle("cur", i === k);
        if (i === k) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current");
      });
    }
    const span = (STEPS[k + 1] ?? END) - STEPS[k];
    const p = RM ? 100 : Math.max(0, Math.min(100, (t - STEPS[k]) / span * 100));
    dotBtns[k].style.setProperty("--p", p.toFixed(1) + "%");
  };
  dotBtns.forEach((b, i) => b.setAttribute("aria-label", `Step ${i + 1} of ${STEPS.length}: ${CAPS[i]}`));

  /* ---------- 1, 2: the two tabs ---------- */
  const tabA = $("tabA"), tabB = $("tabB"), aGo = $("aGo"), bBtn = $("bBtn");
  const codeL = [...tabA.querySelectorAll(".mcode span")], slots = [...tabB.querySelectorAll(".slots i")];
  const CODE = "K7QX";
  const setSlots = (n, cur) => slots.forEach((s, i) => { s.textContent = i < n ? CODE[i] : ""; s.classList.toggle("cur", i === cur); });

  /* ---------- the room: devices, the pool, the layers ---------- */
  const chips = [...$("chips").children];
  const GB = [12, 12, 2];
  const gbSum = $("gbSum"), tbSum = gbSum.parentNode;
  const setDevices = n => {
    chips.forEach((c, i) => c.classList.toggle("out", i >= n));
    gbSum.textContent = GB.slice(0, n).reduce((a, b) => a + b, 0) + " GB";
  };
  const pick = $("pick"), pkGo = $("pkGo"), pkStat = $("pkStat");
  const lb = [$("lb0"), $("lb1")], lp = [$("lp0"), $("lp1")];
  const strips = [...win.querySelectorAll(".strip")].map(s => [...s.children]);
  const cells = strips.flat();
  const owner = strips[0].map(c => +c.className.slice(1));   // c0 / c1: whose layers
  const LAYERS = [0, 1].map(d => owner.filter(o => o === d).length);
  // both devices download their own layers at the same time: k of 40 is overall progress
  const START = [0, LAYERS[0]];
  const hasLayer = (i, k) => { const d = owner[i]; return i - START[d] < Math.round(k / 40 * LAYERS[d]); };
  let filled = -1;
  const fill = k => {
    if (k === filled) return;
    strips.forEach(st => st.forEach((c, i) => {
      const on = hasLayer(i, k), was = filled >= 0 && hasLayer(i, filled);
      c.className = on ? "c" + owner[i] : "";
      if (on && !was && filled >= 0) c.classList.add("f");
    }));
    filled = k;
    pick.classList.toggle("loaded", k >= 40);
    pkStat.textContent = k < 40 ? `${k} of 40 layers` : "40 layers, split 2 ways";
    [0, 1].forEach(d => {
      const n = Math.round(k / 40 * LAYERS[d]);
      lb[d].style.width = (n / LAYERS[d] * 100) + "%";
      lp[d].textContent = n === LAYERS[d] ? `${n} layers` : `${Math.round(n / LAYERS[d] * 100)}%`;
    });
  };
  // a word travels the MacBook's layers, then the desktop's
  let hot = -1;
  const setHot = i => {
    if (i === hot) return;
    cells.forEach((c, j) => { const x = j % 40; c.classList.toggle("hot", i >= 0 && Math.abs(x - i) <= 1); });
    chips.forEach((c, j) => c.classList.toggle("tok", i >= 0 && owner[i] === j));
    hot = i;
  };

  /* ---------- tiny timeline engine ---------- */
  const says = [...win.querySelectorAll(".say")].map(el => ({ el, text: el.textContent }));
  const restore = s => { s.el.textContent = s.text; s.el.classList.remove("cursor"); };
  const tl = { t: 0, fired: 0, streams: [], done: false, started: false };
  const show = k => {
    const el = win.querySelector(`[data-at="${k}"]`); if (!el) return null;
    el.classList.remove("pending"); restart(el, "enter"); return el;
  };
  const stream = (el, rate, onDone) => {
    const s = says.find(x => x.el === el || el.contains(x.el)); if (!s) return null;
    s.el.textContent = ""; s.el.classList.add("cursor");
    const st = { s, words: s.text.split(" "), t0: tl.t, rate, onDone };
    tl.streams.push(st); return st;
  };
  const typeInto = (el, text, t0, t1, t) => {
    const n = Math.max(0, Math.min(text.length, Math.ceil((t - t0) / (t1 - t0) * text.length)));
    if (el.textContent.length !== n) el.textContent = text.slice(0, n);
  };

  /* ---------- 4, 5: chat ---------- */
  const msgs = $("msgs"), chatTyped = $("chatTyped"), chatComposer = $("chatComposer"), chatWho = $("chatWho");
  const Q1 = "what is Pooled?", Q2 = "can it write code?";
  const toBottom = () => { msgs.scrollTop = msgs.scrollHeight; };
  const typer = icon => { chatWho.querySelector("use").setAttribute("href", icon); chatComposer.classList.add("hot"); };
  const sent = () => { chatTyped.textContent = ""; chatComposer.classList.remove("hot"); };
  let pulse = null;
  const answer = (k, rate) => {
    const li = show(k); li.classList.add("live-share"); flag("pulsing", true);
    pulse = stream(li, rate, () => { li.classList.remove("live-share"); pulse = null; setHot(-1); flag("pulsing", false); });
    toBottom();
  };

  /* ---------- 6, 7: Code mode, then the game it built ---------- */
  const log = $("log"), segCode = win.querySelector(".sg-code");
  const t0 = $("t0"), t1 = $("t1"), t2 = $("t2"), lv = $("lv"), lvPre = $("lvPre"), lvF = $("lvF"), lvN = $("lvN");
  const LINES = [...$("dRows").children].map(r => r.textContent);
  const INDEX = ['<!doctype html>', '<html lang="en">', '<head>', '  <meta charset="utf-8">', '  <title>Tetris</title>', '  <link rel="stylesheet" href="style.css">',
    '</head>', '<body>', '  <canvas id="board" width="200" height="400"></canvas>', '  <p id="score">0</p>', '  <script src="game.js"></script>', '</body>'];
  const pv = $("pv"), app = $("app"), brLoad = $("brLoad"), pvSt = $("pvSt"), game = $("game"), play = $("play"), paused = $("paused");
  const overT = $("overT"), overScore = $("overScore");
  const codeTyped = $("codeTyped"), codeComposer = $("codeComposer");
  const PROMPT = "build me a tetris game";
  const logBottom = () => { log.scrollTop = log.scrollHeight; };
  const fmt = n => n.toLocaleString("en-US");
  const tet = window.PooledTetris($("tetris"), {
    seed: 11, tick: 60,
    onScore: (s, l) => { $("score").textContent = fmt(s); $("lines").textContent = l; overScore.textContent = fmt(s); },
    onState: st => {
      app.classList.toggle("human", st === "play" || st === "paused");
      paused.hidden = st !== "paused";
      overT.hidden = st !== "over";
      play.textContent = st === "over" ? "Play again" : "Play";
    }
  });
  // the live card: the file as the model types it
  let liveSrc = LINES, shown = -1, writing = null;
  const liveTo = k => {
    if (k === shown) return;
    if (k < shown || shown < 0) { lvPre.textContent = ""; shown = 0; }
    for (let i = shown; i < k; i++) { const sp = document.createElement("span"); sp.textContent = liveSrc[i] || " "; lvPre.append(sp); }
    while (lvPre.children.length > 12) lvPre.firstChild.remove();
    shown = k; lvN.textContent = `${k} line${k === 1 ? "" : "s"}`;
  };
  const liveStart = (name, src) => { liveSrc = src; lvF.textContent = name; shown = -1; liveTo(0); lv.classList.remove("pending"); restart(lv, "new"); flag("pulsing", true); };
  const liveEnd = () => { lv.classList.add("pending"); writing = null; setHot(-1); flag("pulsing", false); };
  const chip = (card, state) => {
    const c = card.querySelector(".chip");
    c.className = "chip " + state;
    c.textContent = state === "pending" ? "needs approval" : state;
    card.classList.toggle("ask", state === "pending");
  };
  const toolShow = (card, state) => { card.classList.remove("pending", "shut"); chip(card, state); restart(card, "new"); logBottom(); };
  const approve = (card, t) => { press(card.querySelector(".yes"), t); };
  const openPreview = () => { pv.classList.remove("shut"); restart(brLoad, "go"); flag("app-on", true); };
  const showApp = () => {
    app.classList.remove("blank");
    if (!tet.human) { tet.auto(11); tet.warm(14); }
    if (!RM || tet.human) tet.start(); else tet.draw();
  };
  const L0 = C + 3.0, L1 = C + 3.6;       // index.html streams
  const G0 = C + 5.0, G1 = C + 8.4;       // game.js streams

  const EVENTS = [
    // 1: the laptop starts a room
    [.9, () => press(aGo, 300)],
    [1.25, () => { tabA.classList.add("done"); }],
    ...codeL.map((s, i) => [1.35 + i * .15, () => { s.classList.remove("off"); restart(s, "lit"); }]),
    [2.2, () => tabA.classList.remove("linkless")],
    // 2: the desktop types the code and joins; the tabs merge into one room
    [STEPS[1] + .2, () => setSlots(0, 0)],
    [4.1, () => setSlots(1, 1)], [4.4, () => setSlots(2, 2)], [4.7, () => setSlots(3, 3)],
    [5.0, () => { setSlots(4, -1); bBtn.classList.add("ready"); }],
    [5.5, () => press(bBtn, 300)],
    [5.85, () => { tabB.classList.add("done"); tabA.classList.add("met"); }],
    [6.9, () => { scene("pick"); setDevices(2); }],
    // 3: pick a model, the layers load on both devices
    [9.0, () => press(pkGo, 300)],
    [9.35, () => { pick.classList.add("loading"); fill(0); }],
    [12.4, () => flag("model-on", true)],
    // 4: chat
    [STEPS[3], () => { scene("chat"); show("rdy"); toBottom(); }],
    [13.6, () => typer("#i-laptop")],
    [14.7, () => { sent(); show("q1"); toBottom(); }],
    [15.0, () => answer("a1", .13)],
    // 5: a phone joins in the middle
    [STEPS[4], () => { setDevices(3); restart(chips[2], "in"); restart(tbSum, "bump"); show("ph"); toBottom(); }],
    [21.4, () => typer("#i-phone")],
    [22.4, () => { sent(); show("q2"); toBottom(); }],
    [22.7, () => answer("a2", .13)],
    // 6: Code mode: the agent writes, you approve
    [C, () => { press(segCode, 300); }],
    [C + .25, () => { scene("code"); codeComposer.classList.add("hot"); }],
    [C + 1.4, () => { codeTyped.textContent = ""; codeComposer.classList.remove("hot"); show("c-q"); logBottom(); }],
    [C + 1.7, () => { stream(show("c-s1"), .06); logBottom(); }],
    [L0, () => { liveStart("index.html", INDEX); writing = [L0, L1, INDEX.length]; logBottom(); }],
    [L1 + .15, () => { liveEnd(); toolShow(t0, "pending"); t0.classList.add("shut"); }],
    [C + 4.35, () => approve(t0, 300)],
    [C + 4.6, () => chip(t0, "done")],
    [G0, () => { liveStart("game.js", LINES); writing = [G0, G1, LINES.length]; logBottom(); }],
    [G1 + .2, () => { liveEnd(); toolShow(t1, "pending"); }],
    [C + 9.7, () => approve(t1, 350)],
    [C + 10.0, () => { chip(t1, "done"); t1.classList.add("shut"); logBottom(); }],
    [C + 10.2, () => toolShow(t2, "running")],
    [C + 10.8, () => { chip(t2, "done"); openPreview(); }],
    [C + 11.1, showApp],
    [C + 11.4, () => { stream(show("c-s2"), .06); logBottom(); }],
    // 7: your turn
    [STEPS[6], () => flag("turn", true)],
  ];

  const frame = t => {
    if (t > 9.4 && t < 12.2) fill(Math.min(40, Math.ceil((t - 9.4) / 2.6 * 40)));
    if (t > 13.6 && t < 14.7) typeInto(chatTyped, Q1, 13.75, 14.5, t);
    if (t > 21.4 && t < 22.4) typeInto(chatTyped, Q2, 21.55, 22.2, t);
    if (t > C + .25 && t < C + 1.4) typeInto(codeTyped, PROMPT, C + .4, C + 1.2, t);
    if (pulse) setHot(Math.floor(((t - pulse.t0) % pulse.rate) / pulse.rate * 40));
    if (writing) {
      const [a, b, n] = writing;
      if (t > a) { liveTo(Math.min(n, Math.ceil((t - a) / (b - a) * n))); logBottom(); }
      setHot(Math.floor(((t - a) % .1) / .1 * 40));
    }
  };

  const clearTools = () => [t0, t1, t2].forEach(c => { c.classList.add("pending"); c.classList.remove("ask", "shut"); });
  tl.reset = () => {
    tl.t = 0; tl.fired = 0; tl.streams = []; tl.done = false; pulse = null; writing = null;
    scene("tabs"); ["app-on", "turn", "pulsing", "model-on"].forEach(c => flag(c, false));
    tabA.classList.add("linkless"); tabA.classList.remove("done", "met");
    codeL.forEach(s => s.classList.add("off"));
    tabB.classList.remove("done"); setSlots(0, -1); bBtn.classList.remove("ready", "press");
    setDevices(0); chips.forEach(c => c.classList.remove("in"));
    pick.classList.remove("loading"); filled = -1; fill(0); setHot(-1);
    win.querySelectorAll("[data-at]").forEach(el => el.classList.add("pending"));
    win.querySelectorAll(".live-share").forEach(el => el.classList.remove("live-share"));
    says.forEach(restore); sent(); msgs.scrollTop = 0;
    codeTyped.textContent = ""; codeComposer.classList.remove("hot");
    clearTools(); lv.classList.add("pending"); shown = -1; liveTo(0);
    pv.classList.add("shut"); app.classList.add("blank"); brLoad.classList.remove("go");
    if (!tet.human) { tet.stop(); tet.auto(11); }
    log.scrollTop = 0;
    paintBar(0, true);
  };
  tl.final = () => {
    tl.streams = []; tl.done = true; tl.started = true; tl.t = END; tl.fired = EVENTS.length; pulse = null; writing = null;
    scene("code"); flag("app-on", true); flag("model-on", true); flag("turn", false); flag("pulsing", false);
    tabA.classList.add("done", "met"); tabA.classList.remove("linkless"); codeL.forEach(s => s.classList.remove("off"));
    tabB.classList.add("done"); setSlots(4, -1);
    setDevices(3); pick.classList.add("loading"); fill(40); setHot(-1);
    win.querySelectorAll("[data-at]").forEach(el => el.classList.remove("pending", "enter"));
    win.querySelectorAll(".live-share").forEach(el => el.classList.remove("live-share"));
    says.forEach(restore); sent(); codeTyped.textContent = ""; codeComposer.classList.remove("hot");
    lv.classList.add("pending");
    [t0, t1, t2].forEach(c => { chip(c, "done"); c.classList.add("shut"); });
    pv.classList.remove("shut"); app.classList.remove("blank");
    if (!tet.human) { tet.auto(11); tet.warm(18); }
    if (!RM) tet.start(); else tet.draw();
    toBottom(); logBottom();
    paintBar(END);
  };
  tl.advance = dt => {
    tl.t += dt;
    while (tl.fired < EVENTS.length && EVENTS[tl.fired][0] <= tl.t) { EVENTS[tl.fired][1](); tl.fired++; }
    tl.streams = tl.streams.filter(st => {
      const n = Math.min(st.words.length, Math.floor((tl.t - st.t0) / st.rate));
      if (n >= st.words.length) { restore(st.s); st.onDone && st.onDone(); toBottom(); logBottom(); return false; }
      const txt = st.words.slice(0, n).join(" ");
      if (st.s.el.textContent !== txt) { st.s.el.textContent = txt; toBottom(); logBottom(); }
      return true;
    });
    frame(tl.t);
    paintBar(tl.t, true);
    if (tl.t >= END && !tl.streams.length && tl.fired >= EVENTS.length) tl.done = true;
  };
  EVENTS.sort((a, b) => a[0] - b[0]);

  /* ---------- driver: one rAF, only while the window is on screen and the page is visible ---------- */
  let raf = 0, last = 0, visible = false, frozen = false;
  const needs = () => !frozen && visible && !document.hidden && tl.started && !tl.done;
  function loop(now) {
    const dt = last ? Math.min(.1, (now - last) / 1000) : .016; last = now;
    tl.advance(dt);
    raf = needs() ? requestAnimationFrame(loop) : 0;
  }
  const wake = () => { if (needs() && !raf) { last = 0; raf = requestAnimationFrame(loop); } };
  const halt = () => { if (raf) { cancelAnimationFrame(raf); raf = 0; } };
  const begin = () => { if (!tl.started) { tl.started = true; if (RM) tl.final(); else tl.reset(); } wake(); };

  // jump to step k (0-based). With reduced motion: that step, finished, and no motion.
  const seek = (s, freeze) => {
    halt(); tl.started = true; frozen = !!freeze;
    if (tet.human) tet.auto(11);
    tl.reset();
    while (tl.t < s - 1e-6) tl.advance(Math.min(1 / 30, s - tl.t));
    tet.draw(); wake();
  };
  const goStep = k => {
    if (RM) {
      if (k >= STEPS.length - 1) { tl.final(); return; }
      seek(k === 1 ? 6.5 : STEPS[k + 1] - .35, true); tl.done = true; return;   // step 2 ends on the two joined tabs, before the merge
    }
    seek(STEPS[k]);
  };
  dotBtns.forEach((b, i) => b.addEventListener("click", () => goStep(i)));
  $("replay").addEventListener("click", () => goStep(0));

  /* ---------- the game: keys are captured only while it has focus ---------- */
  const KEYMAP = { ArrowLeft: "left", ArrowRight: "right", ArrowUp: "rot", ArrowDown: "down", " ": "drop", Spacebar: "drop", x: "rot", z: "rot" };
  const takeOver = () => { flag("turn", false); tet.play(); };
  play.addEventListener("click", e => { e.stopPropagation(); game.focus({ preventScroll: true }); takeOver(); });
  game.addEventListener("click", () => {
    if (!tet.human || tet.over) { game.focus({ preventScroll: true }); takeOver(); }
    else if (tet.paused) tet.pause(false);
  });
  game.addEventListener("keydown", e => {
    if (e.key === "Escape") { game.blur(); return; }
    const a = KEYMAP[e.key];
    if (!a) { if (e.key === "Enter" && (!tet.human || tet.over)) { e.preventDefault(); takeOver(); } return; }
    e.preventDefault();
    if (!tet.human || tet.over) { takeOver(); return; }
    if (tet.paused) tet.pause(false);
    tet.act(a);
  });
  game.addEventListener("focus", () => { if (tet.human && tet.paused) tet.pause(false); });
  game.addEventListener("blur", e => { if (tet.human && !tet.over && !(e.relatedTarget && $("pad").contains(e.relatedTarget))) tet.pause(true); });
  $("pad").querySelectorAll("button").forEach(b => {
    const k = b.dataset.k; let rep = 0, hold = 0;
    const fire = () => { if (document.activeElement !== game) game.focus({ preventScroll: true }); if (!tet.human || tet.over) { takeOver(); return; } if (tet.paused) tet.pause(false); tet.act(k); };
    const stopRep = () => { clearTimeout(hold); clearInterval(rep); };
    b.addEventListener("pointerdown", e => {
      e.preventDefault(); fire();
      if (k === "left" || k === "right" || k === "down") hold = setTimeout(() => { rep = setInterval(() => tet.act(k), 90); }, 230);
    });
    ["pointerup", "pointerleave", "pointercancel"].forEach(ev => b.addEventListener(ev, stopRep));
    b.addEventListener("click", e => { if (e.detail === 0) fire(); });
  });

  /* ---------- start when 30% visible ---------- */
  new IntersectionObserver(es => {
    visible = es[0].isIntersecting;
    if (visible) begin();
    wake();
  }, { threshold: .3 }).observe(win);
  document.addEventListener("visibilitychange", wake);

  if (RM) tl.final(); else tl.reset();

  // tests and screenshots: jump to a moment
  window.__demo = {
    seek(s, freeze) { seek(s, freeze); },
    step: goStep,
    set frozen(v) { frozen = v; wake(); }, get t() { return tl.t; }, STEPS, END, tet
  };
})();
