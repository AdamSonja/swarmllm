/* The demo window: one story in six steps (start a room, join it, the model fits, chat, build,
   play), a step bar under it that says where we are and jumps to any step, a playable Tetris,
   and the closing logo that assembles from three devices.
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
  const scene = s => { win.dataset.scene = s; };
  const flag = (cls, on) => win.classList.toggle(cls, on);

  /* ---------- steps and the bar under the window ---------- */
  const T5 = 24.8;
  const STEPS = [0, 4.0, 8.4, 15.2, T5, T5 + 8.6];
  const END = T5 + 10.2;
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

  /* ---------- 1, 2: start, join ---------- */
  const jStart = $("jStart"), jJoin = $("jJoin"), jGo = $("jGo"), jBtn = $("jBtn"), jWire = win.querySelector(".jwire");
  const codeL = [...jStart.querySelectorAll(".jcode span")], slots = [...jJoin.querySelectorAll(".slots i")];
  const CODE = "NEST";
  const setSlots = (n, cur) => slots.forEach((s, i) => { s.textContent = i < n ? CODE[i] : ""; s.classList.toggle("cur", i === cur); });

  /* ---------- devices, the pool, the layers ---------- */
  const chips = [...$("chips").children], rows = [...$("prow").children];
  const GB = [10, 12, 2], SCALE = 26, NEED = 22.5;
  const meter = $("meter"), mst = $("mst"), gbs = [...win.querySelectorAll(".js-gb")];
  const model = $("model"), mstat = $("mstat");
  const strips = [...win.querySelectorAll(".strip")].map(s => [...s.children]);
  const cells = strips.flat();
  const owner = strips[0].map(c => +c.className.slice(1));   // c0 / c1 / c2: whose layers
  const LAYERS = [0, 1, 2].map(d => owner.filter(o => o === d).length);
  const layerTx = rows.map(r => r.querySelector(".pl"));
  const setPool = n => {
    chips.forEach((c, i) => c.classList.toggle("out", i >= n));
    rows.forEach((r, i) => r.classList.toggle("out", i >= n));
    GB.forEach((g, i) => { $("seg" + i).style.width = i < n ? (g / SCALE * 100) + "%" : "0%"; });
    const sum = GB.slice(0, n).reduce((a, b) => a + b, 0);
    gbs.forEach(g => { g.textContent = sum + " GB"; });
    meter.classList.toggle("ok", sum >= NEED);
    mst.textContent = !n ? "Waiting for devices" : sum >= NEED ? "Enough to run it" : (NEED - sum).toFixed(1) + " GB short";
  };
  const arrive = i => { setPool(i + 1); restart(chips[i], "in"); restart(rows[i], "in"); if (GB.slice(0, i + 1).reduce((a, b) => a + b, 0) >= NEED) restart(meter, "hit"); };
  let filled = -1;
  const fill = k => {
    if (k === filled) return;
    strips.forEach(st => st.forEach((c, i) => {
      const on = i < k;
      c.className = on ? "c" + owner[i] : "";
      if (on && i >= filled && filled >= 0) c.classList.add("f");
    }));
    filled = k;
    model.classList.toggle("loaded", k >= 40);
    mstat.textContent = k <= 0 ? "waiting for 22.5 GB" : k < 40 ? `loading ${k} of 40` : "40 layers, split 3 ways";
    layerTx.forEach((el, d) => {
      const n = owner.slice(0, Math.max(0, k)).filter(o => o === d).length;
      el.textContent = k <= 0 ? "" : n === LAYERS[d] ? `${n} layers` : `${n} / ${LAYERS[d]}`;
    });
  };
  // a word travels the MacBook's layers, then the desktop's, then the phone's
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

  /* ---------- 4: chat ---------- */
  const msgs = $("msgs"), chatTyped = $("chatTyped"), chatComposer = $("chatComposer"), chatWho = $("chatWho");
  const Q1 = "what is pooled?", Q2 = "can it write code?";
  const toBottom = () => { msgs.scrollTop = msgs.scrollHeight; };
  const typer = icon => { chatWho.querySelector("use").setAttribute("href", icon); chatComposer.classList.add("hot"); };
  const sent = () => { chatTyped.textContent = ""; chatComposer.classList.remove("hot"); };
  let pulse = null;
  const answer = (k, rate) => {
    const li = show(k); li.classList.add("live-share"); flag("pulsing", true);
    pulse = stream(li, rate, () => { li.classList.remove("live-share"); pulse = null; setHot(-1); flag("pulsing", false); });
    toBottom();
  };

  /* ---------- 5, 6: build, play ---------- */
  const steps = $("steps"), files = [...win.querySelectorAll("[data-f]")];
  const wPre = $("wPre"), wN = $("wN"), wTn = files[1].querySelector(".tn");
  const LINES = wPre.textContent.split("\n");
  const browser = $("browser"), app = $("app"), brLoad = $("brLoad"), game = $("game"), play = $("play"), paused = $("paused");
  const overT = $("overT"), overScore = $("overScore");
  const codeTyped = $("codeTyped"), codeComposer = $("codeComposer");
  const PROMPT = "build me a tetris game";
  const stepsBottom = () => { steps.scrollTop = steps.scrollHeight; };
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
  const pfs = [...win.querySelectorAll("[data-pf]")], pfN = pfs[1] && pfs[1].querySelector(".n");
  let shown = -1, writing = false;
  const writeTo = k => {
    if (k === shown) return;
    if (k < shown || shown < 0) { wPre.textContent = ""; shown = 0; }
    for (let i = shown; i < k; i++) {
      const sp = document.createElement("span"); sp.textContent = LINES[i];
      wPre.querySelectorAll(".nw").forEach(e => e.classList.remove("nw"));
      sp.className = "nw"; wPre.append(sp);
    }
    while (wPre.children.length > (innerWidth <= 760 ? 9 : 11)) wPre.firstChild.remove();
    wPre.style.setProperty("--ln0", String(Math.max(0, k - wPre.children.length)));
    shown = k; wN.textContent = "+" + k;
    if (pfN) pfN.textContent = String(Math.max(k, 1));
  };
  const file = (i, state) => {
    const pf = pfs[i];
    if (pf) {
      pf.classList.toggle("pending", state === "hide");
      pf.classList.toggle("busy", state === "busy");
      if (state === "busy") restart(pf, "new");
    }
    const li = files[i];
    li.classList.toggle("pending", state === "hide");
    li.classList.toggle("busy", state === "busy");
    if (state === "busy") restart(li, "new");
  };
  const openBrowser = () => { browser.classList.remove("shut"); restart(brLoad, "go"); flag("app-on", true); };
  const showApp = () => {
    app.classList.remove("blank");
    if (!tet.human) { tet.auto(11); tet.warm(14); }
    if (!RM || tet.human) tet.start(); else tet.draw();
  };
  const W0 = T5 + 2.8, W1 = T5 + 6.0;

  const EVENTS = [
    // 1: start a room on the MacBook
    [.9, () => jGo.classList.add("press")],
    [1.2, () => { jGo.classList.remove("press"); jStart.classList.add("done"); }],
    [1.35, () => { codeL[0].classList.remove("off"); restart(codeL[0], "lit"); }],
    [1.5, () => { codeL[1].classList.remove("off"); restart(codeL[1], "lit"); flag("has-room", true); arrive(0); }],
    [1.65, () => { codeL[2].classList.remove("off"); restart(codeL[2], "lit"); }],
    [1.8, () => { codeL[3].classList.remove("off"); restart(codeL[3], "lit"); }],
    [2.3, () => jStart.classList.remove("linkless")],
    // 2: the desktop types the code
    [STEPS[1], () => { jJoin.classList.add("on"); setSlots(0, 0); }],
    [4.4, () => setSlots(1, 1)], [4.75, () => setSlots(2, 2)], [5.1, () => setSlots(3, 3)],
    [5.45, () => { setSlots(4, -1); jBtn.classList.add("ready"); }],
    [6.1, () => jBtn.classList.add("press")],
    [6.4, () => { jBtn.classList.remove("press"); jJoin.classList.add("done"); jWire.classList.add("on"); arrive(1); }],
    // 3: the phone joins, the pool crosses the line, the layers load
    [STEPS[2], () => { scene("pool"); fill(0); }],
    [9.8, () => arrive(2)],
    // 4: chat
    [STEPS[3], () => { scene("chat"); show("rdy"); toBottom(); }],
    [15.8, () => typer("#i-phone")],
    [16.9, () => { sent(); show("q1"); toBottom(); }],
    [17.2, () => answer("a1", .19)],
    [20.8, () => typer("#i-laptop")],
    [21.8, () => { sent(); show("q2"); toBottom(); }],
    [22.0, () => answer("a2", .16)],
    // 5: build
    [T5, () => { scene("build"); codeComposer.classList.add("hot"); }],
    [T5 + 1.3, () => { codeTyped.textContent = ""; codeComposer.classList.remove("hot"); show("c-q"); stepsBottom(); }],
    [T5 + 1.6, () => { stream(show("c-s1"), .07); stepsBottom(); }],
    [T5 + 2.3, () => { show("c-t"); file(0, "busy"); stepsBottom(); }],
    [T5 + 2.7, () => { file(0, "done"); file(1, "busy"); wTn.textContent = "writing"; writing = true; flag("pulsing", true); stepsBottom(); }],
    [W1 + .2, () => { file(1, "done"); wTn.textContent = "write_file"; writing = false; setHot(-1); flag("pulsing", false); file(2, "busy"); stepsBottom(); }],
    [W1 + .5, () => { file(2, "done"); file(3, "busy"); stepsBottom(); }],
    [W1 + 1.0, () => { file(3, "done"); openBrowser(); }],
    [W1 + 1.3, showApp],
    // 6: your turn
    [STEPS[5], () => flag("turn", true)],
  ];

  const frame = t => {
    if (t > 11.0 && t < 13.6) fill(Math.min(40, Math.ceil((t - 11.0) / 2.4 * 40)));
    if (t > 15.8 && t < 16.9) typeInto(chatTyped, Q1, 15.95, 16.7, t);
    if (t > 20.8 && t < 21.8) typeInto(chatTyped, Q2, 20.95, 21.6, t);
    if (t > T5 && t < T5 + 1.3) typeInto(codeTyped, PROMPT, T5 + .15, T5 + 1.1, t);
    if (pulse) setHot(Math.floor(((t - pulse.t0) % pulse.rate) / pulse.rate * 40));
    if (writing) {
      if (t > W0) { writeTo(Math.min(LINES.length, Math.ceil((t - W0) / (W1 - W0) * LINES.length))); stepsBottom(); }
      setHot(Math.floor(((t - W0) % .1) / .1 * 40));
    }
  };

  tl.reset = () => {
    tl.t = 0; tl.fired = 0; tl.streams = []; tl.done = false; pulse = null; writing = false;
    scene("join"); ["has-room", "app-on", "turn", "pulsing"].forEach(c => flag(c, false));
    jStart.classList.add("on", "linkless"); jStart.classList.remove("done"); jGo.classList.remove("press");
    codeL.forEach(s => s.classList.add("off"));
    jJoin.classList.remove("on", "done"); setSlots(0, -1); jBtn.classList.remove("ready", "press"); jWire.classList.remove("on");
    setPool(0); chips.forEach(c => c.classList.remove("in")); rows.forEach(r => r.classList.remove("in")); meter.classList.remove("hit");
    filled = -1; fill(0); setHot(-1);
    win.querySelectorAll("[data-at]").forEach(el => el.classList.add("pending"));
    win.querySelectorAll(".live-share").forEach(el => el.classList.remove("live-share"));
    says.forEach(restore); sent(); msgs.scrollTop = 0;
    codeTyped.textContent = ""; codeComposer.classList.remove("hot");
    shown = -1; writeTo(0); wTn.textContent = "write_file";
    files.forEach((_, i) => file(i, "hide"));
    browser.classList.add("shut"); app.classList.add("blank"); brLoad.classList.remove("go");
    if (!tet.human) { tet.stop(); tet.auto(11); }
    steps.scrollTop = 0;
    paintBar(0, true);
  };
  tl.final = () => {
    tl.streams = []; tl.done = true; tl.started = true; tl.t = END; tl.fired = EVENTS.length; pulse = null; writing = false;
    scene("build"); flag("has-room", true); flag("app-on", true); flag("turn", false); flag("pulsing", false);
    jStart.classList.add("on", "done"); jStart.classList.remove("linkless"); codeL.forEach(s => s.classList.remove("off"));
    jJoin.classList.add("on", "done"); setSlots(4, -1); jWire.classList.add("on");
    setPool(3); fill(40); setHot(-1);
    win.querySelectorAll("[data-at]").forEach(el => el.classList.remove("pending", "enter"));
    win.querySelectorAll(".live-share").forEach(el => el.classList.remove("live-share"));
    says.forEach(restore); sent(); codeTyped.textContent = ""; codeComposer.classList.remove("hot");
    shown = -1; writeTo(LINES.length); wTn.textContent = "write_file";
    files.forEach((_, i) => file(i, "done"));
    browser.classList.remove("shut"); app.classList.remove("blank");
    if (!tet.human) { tet.auto(11); tet.warm(18); }
    if (!RM) tet.start(); else tet.draw();
    toBottom(); stepsBottom();
    paintBar(END);
  };
  tl.advance = dt => {
    tl.t += dt;
    while (tl.fired < EVENTS.length && EVENTS[tl.fired][0] <= tl.t) { EVENTS[tl.fired][1](); tl.fired++; }
    tl.streams = tl.streams.filter(st => {
      const n = Math.min(st.words.length, Math.floor((tl.t - st.t0) / st.rate));
      if (n >= st.words.length) { restore(st.s); st.onDone && st.onDone(); toBottom(); stepsBottom(); return false; }
      const txt = st.words.slice(0, n).join(" ");
      if (st.s.el.textContent !== txt) { st.s.el.textContent = txt; toBottom(); stepsBottom(); }
      return true;
    });
    frame(tl.t);
    paintBar(tl.t, true);
    if (tl.t >= END && !tl.streams.length && tl.fired >= EVENTS.length) tl.done = true;
  };

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
      seek(STEPS[k + 1] - .35, true); tl.done = true; return;
    }
    seek(STEPS[k]);
  };
  dotBtns.forEach((b, i) => b.addEventListener("click", () => goStep(i)));
  $("replay").addEventListener("click", () => goStep(0));
  document.addEventListener("click", e => {
    const a = e.target.closest && e.target.closest("a[data-step]"); if (!a) return;
    goStep(+a.dataset.step - 1);
  });

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
