/* The demo window: Chat | Code tabs, two short story timelines with a one-line caption per beat,
   a playable Tetris, and the closing logo that assembles from three devices.
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

  /* ---------- caption: one line per beat ---------- */
  const capT = $("capT");
  const showCap = (text, anim) => {
    if (capT.textContent === text) return;
    capT.textContent = text;
    if (anim && !RM) restart(capT, "in");
  };

  /* ---------- tiny timeline engine ---------- */
  function Timeline(root, { events, end, reset, final, frame, lastCap }) {
    const says = [...root.querySelectorAll(".say")].map(el => ({ el, text: el.textContent, html: el.innerHTML }));
    const tl = { t: 0, fired: 0, streams: [], done: false, started: false, cap: lastCap, lastCap };
    const restore = s => { s.el.innerHTML = s.html; s.el.classList.remove("cursor"); };
    tl.caption = text => { tl.cap = text; if (TL[tab] === tl) showCap(text, true); };
    tl.show = k => {
      const el = root.querySelector(`[data-at="${k}"]`); if (!el) return null;
      el.classList.remove("pending"); restart(el, "enter"); return el;
    };
    tl.stream = (el, rate, onDone) => {
      const s = says.find(x => x.el === el || el.contains(x.el)); if (!s) return null;
      s.el.textContent = ""; s.el.classList.add("cursor");
      const st = { s, words: s.text.split(" "), t0: tl.t, rate, onDone };
      tl.streams.push(st); return st;
    };
    tl.reset = () => {
      tl.t = 0; tl.fired = 0; tl.streams = []; tl.done = false;
      root.querySelectorAll("[data-at]").forEach(el => el.classList.add("pending"));
      says.forEach(restore);
      reset();
    };
    tl.final = () => {
      tl.streams = []; tl.done = true; tl.started = true; tl.cap = lastCap;
      root.querySelectorAll("[data-at]").forEach(el => el.classList.remove("pending", "enter"));
      says.forEach(restore);
      final();
      if (TL[tab] === tl) showCap(lastCap);
    };
    tl.advance = dt => {
      tl.t += dt;
      while (tl.fired < events.length && events[tl.fired][0] <= tl.t) { events[tl.fired][1](); tl.fired++; }
      tl.streams = tl.streams.filter(st => {
        const n = Math.min(st.words.length, Math.floor((tl.t - st.t0) / st.rate));
        if (n >= st.words.length) { restore(st.s); st.onDone && st.onDone(); frame("text"); return false; }
        const txt = st.words.slice(0, n).join(" ");
        if (st.s.el.textContent !== txt) { st.s.el.textContent = txt; frame("text"); }
        return true;
      });
      frame("tick", tl.t);
      if (tl.t >= end && !tl.streams.length && tl.fired >= events.length) tl.done = true;
    };
    return tl;
  }
  const typeInto = (el, text, t0, t1, t) => {
    const n = Math.max(0, Math.min(text.length, Math.ceil((t - t0) / (t1 - t0) * text.length)));
    if (el.textContent.length !== n) el.textContent = text.slice(0, n);
  };

  /* ---------- CHAT: devices join until the model fits, it loads across them, they talk ---------- */
  const pc = $("p-chat"), msgs = $("msgs"), devs = [...pc.querySelectorAll(".dev")];
  const GB = [10, 12, 2], SCALE = 26, NEED = 22.5;
  const meter = $("meter"), mst = $("mst"), gbs = [...pc.querySelectorAll(".js-gb")], cnt = $("cnt"), chatEl = pc.querySelector(".chat");
  const stage = on => { chatEl.classList.toggle("boot", on); chatEl.classList.toggle("run", !on); };
  const model = $("model"), mstat = $("mstat"), cells = [...$("strip").children];
  const owner = cells.map(c => +c.className.slice(1));      // c0 / c1 / c2: whose layers
  const chatTyped = $("chatTyped"), chatComposer = $("chatComposer"), chatWho = $("chatWho");
  const Q1 = "what is pooled?", Q2 = "can it write code?";
  const toBottom = () => { msgs.scrollTop = msgs.scrollHeight; };
  const setPool = n => {
    devs.forEach((d, i) => d.classList.toggle("out", i >= n));
    GB.forEach((g, i) => { $("seg" + i).style.width = i < n ? (g / SCALE * 100) + "%" : "0%"; });
    const sum = GB.slice(0, n).reduce((a, b) => a + b, 0);
    gbs.forEach(g => { g.textContent = sum + " GB"; });
    meter.classList.toggle("ok", sum >= NEED);
    mst.textContent = !n ? "Waiting for devices" : sum >= NEED ? "Enough to run it" : (NEED - sum).toFixed(1) + " GB short";
    cnt.textContent = n ? (n === 1 ? "1 device" : n + " devices") : "";
  };
  let filled = -1;
  const fill = k => {
    if (k === filled) return;
    cells.forEach((c, i) => {
      const on = i < k;
      c.className = on ? "c" + owner[i] : "";
      if (on && i >= filled && filled >= 0) c.classList.add("f");
    });
    filled = k;
    model.classList.toggle("loaded", k >= cells.length);
    mstat.textContent = k <= 0 ? "waiting" : k < cells.length ? `loading ${k} of 40 layers` : "40 layers, split 3 ways";
  };
  // a word travels Maya's layers, then Sam's, then Leo's; the token appears when it reaches the end
  let pulse = null, hot = -1;
  const setHot = i => {
    if (i === hot) return;
    cells.forEach((c, j) => c.classList.toggle("hot", i >= 0 && Math.abs(j - i) <= 1));
    devs.forEach((d, j) => d.classList.toggle("tok", i >= 0 && owner[i] === j));
    hot = i;
  };
  const typer = (who, cls) => { chatWho.textContent = who; chatWho.className = "av typer " + cls; chatComposer.classList.add("hot"); };
  const sent = () => { chatTyped.textContent = ""; chatComposer.classList.remove("hot"); };
  const join = i => { setPool(i + 1); restart(devs[i], "in"); if (GB.slice(0, i + 1).reduce((a, b) => a + b, 0) >= NEED) restart(meter, "hit"); };
  const answer = k => {
    const li = chat.show(k); li.classList.add("live-share");
    pulse = chat.stream(li, .2, () => { li.classList.remove("live-share"); pulse = null; setHot(-1); });
    toBottom();
  };
  let seeking = false, handedOff = false;
  const handOff = () => { if (seeking || handedOff || tab !== "chat") return; handedOff = true; setTimeout(() => select("code"), 0); };
  const LOAD0 = 6.3, LOAD1 = 7.6;

  const chat = Timeline(pc, {
    end: 16.2, lastCap: "Three devices, one model, one conversation",
    events: [
      [0, () => chat.caption("Maya opens a room")],
      [.4, () => join(0)],
      [2.4, () => chat.caption("Sam joins from his PC")],
      [2.6, () => join(1)],
      [4.5, () => chat.caption("Leo joins from his phone")],
      [4.7, () => join(2)],
      [6.1, () => chat.caption("The model loads across all three")],
      [8.8, () => { stage(false); chat.show("rdy"); toBottom(); }],
      [9.2, () => { chat.caption("Leo asks a question"); typer("L", "a3"); }],
      [10.1, () => { sent(); chat.show("q1"); toBottom(); }],
      [10.4, () => { chat.caption("Every word flows through all three devices"); answer("a1"); }],
      [12.9, () => { chat.caption("Maya asks if it can code"); typer("M", "a1"); }],
      [13.9, () => { sent(); chat.show("q2"); toBottom(); }],
      [14.1, () => answer("a2")],
      [16.4, handOff],
    ],
    frame(kind, t) {
      if (kind === "text") return toBottom();
      if (t > LOAD0 && t < LOAD1 + .2) fill(Math.min(40, Math.ceil((t - LOAD0) / (LOAD1 - LOAD0) * 40)));
      if (t > 9.2 && t < 10.1) typeInto(chatTyped, Q1, 9.3, 9.95, t);
      if (t > 12.9 && t < 13.9) typeInto(chatTyped, Q2, 13.0, 13.75, t);
      if (pulse) setHot(Math.floor(((t - pulse.t0) % pulse.rate) / pulse.rate * 40));
    },
    reset() {
      setPool(0); fill(0); setHot(-1); pulse = null; sent(); stage(true); chatEl.classList.remove("run");
      pc.querySelectorAll(".live-share").forEach(el => el.classList.remove("live-share"));
      devs.forEach(d => d.classList.remove("in")); meter.classList.remove("hit");
      msgs.scrollTop = 0;
    },
    final() {
      setPool(3); fill(40); setHot(-1); pulse = null; sent(); chatEl.classList.remove("boot", "run");
      devs.forEach(d => d.classList.remove("in")); meter.classList.remove("hit");
      pc.querySelectorAll(".live-share").forEach(el => el.classList.remove("live-share")); toBottom();
    }
  });

  /* ---------- CODE: they are all there; Maya asks, code streams, it runs, everyone sees it ---------- */
  const pk = $("p-code"), steps = $("steps"), files = [...pk.querySelectorAll("[data-f]")];
  const pres = [...pk.querySelectorAll(".pres li")], ust0 = $("ust0");
  const wcard = $("wcard"), wPre = $("wPre"), wN = $("wN"), wNm = wcard.querySelector(".nm");
  const LINES = wPre.textContent.split("\n"), wf = wcard.querySelector(".wf .t");
  const DONE = "written on Maya's, Sam's and Leo's devices", LIVE = "running on Maya's, Sam's and Leo's devices";
  const peers = [...pk.querySelectorAll(".peer")], ghosts = peers.map(p => p.querySelector(".ghost")), mirs = peers.map(p => p.querySelector(".mir"));
  const browser = $("browser"), app = $("app"), brLoad = $("brLoad"), game = $("game"), play = $("play"), paused = $("paused");
  const overT = $("overT"), overScore = $("overScore");
  const codeTyped = $("codeTyped"), codeComposer = $("codeComposer");
  const PROMPT = "build me a tetris game";
  const stepsBottom = () => { steps.scrollTop = steps.scrollHeight; };
  const fmt = n => n.toLocaleString("en-US");

  // Sam's and Leo's screens mirror the game (each peer runs the preview itself in the real room)
  const mirror = src => {
    peers.forEach((p, i) => {
      if (!p.classList.contains("on")) return;
      const c = mirs[i], r = c.getBoundingClientRect(); if (!r.width) return;
      const d = Math.min(devicePixelRatio || 1, 2), w = Math.round(r.width * d), h = Math.round(r.height * d);
      if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
      const x = c.getContext("2d"), s = Math.min(w / src.width, h / src.height);
      x.fillStyle = "#0B0F1F"; x.fillRect(0, 0, w, h);
      x.drawImage(src, (w - src.width * s) / 2, (h - src.height * s) / 2, src.width * s, src.height * s);
    });
  };
  const tet = window.PooledTetris($("tetris"), {
    seed: 11, tick: 60,
    onScore: (s, l) => { $("score").textContent = fmt(s); $("lines").textContent = l; overScore.textContent = fmt(s); },
    onState: st => {
      app.classList.toggle("human", st === "play" || st === "paused");
      paused.hidden = st !== "paused";
      overT.hidden = st !== "over";
      play.textContent = st === "over" ? "Play again" : "Play";
    },
    onDraw: mirror
  });

  const presence = n => pres.forEach((li, i) => li.classList.toggle("out", i >= n));
  const arrive = i => { presence(i + 1); restart(pres[i], "in"); peers[i - 1] && peerState(i - 1, "in"); };
  const peerState = (i, st) => {
    const p = peers[i];
    p.classList.toggle("wait", st === "out");
    p.classList.toggle("on", st === "live");
    if (st === "live") restart(p, "hit");
  };
  const ghostLine = line => ghosts.forEach(g => {
    const i = document.createElement("i");
    i.style.width = line.trim() ? Math.min(100, 12 + line.length * 2.4) + "%" : "0";
    if (!line.trim()) i.style.background = "none";
    g.append(i);
    while (g.children.length > 24) g.firstChild.remove();
  });
  const clearGhosts = () => ghosts.forEach(g => { g.textContent = ""; });
  let shown = -1;
  const writeTo = k => {
    if (k === shown) return;
    if (k < shown || shown < 0) { wPre.textContent = ""; shown = 0; }
    for (let i = shown; i < k; i++) {
      const sp = document.createElement("span"); sp.textContent = LINES[i];
      wPre.querySelectorAll(".nw").forEach(e => e.classList.remove("nw"));
      sp.className = "nw"; wPre.append(sp); ghostLine(LINES[i]);
    }
    while (wPre.children.length > 12) wPre.firstChild.remove();
    shown = k; wN.textContent = k + (k === 1 ? " line" : " lines");
  };
  const file = (i, state) => {
    const li = files[i];
    li.classList.toggle("pending", state === "hide");
    li.classList.toggle("busy", state === "busy");
    if (state === "busy") restart(li, "new");
  };
  const openBrowser = () => { browser.classList.remove("shut"); restart(brLoad, "go"); };
  const showApp = () => {
    app.classList.remove("blank");
    if (!tet.human) { tet.auto(11); tet.warm(14); }
    if (!RM || tet.human) tet.start(); else tet.draw();
  };
  const W0 = 4.4, W1 = 7.4;

  const code = Timeline(pk, {
    end: 12.4, lastCap: "Built together, seen on every device",
    events: [
      [0, () => code.caption("Everyone is in the room")],
      [.2, () => arrive(0)],
      [.8, () => { arrive(1); code.show("c-w1"); ghostLine("  Sam is watching"); stepsBottom(); }],
      [1.5, () => { arrive(2); code.show("c-w2"); ghostLine("  Leo is watching on his phone"); stepsBottom(); }],
      [2.5, () => { code.caption("Maya asks for a game"); codeComposer.classList.add("hot"); ust0.textContent = "typing"; pres[0].classList.add("act"); }],
      [3.5, () => { codeTyped.textContent = ""; codeComposer.classList.remove("hot"); code.show("c-q"); ghostLine(""); ghostLine("Maya: build me a tetris game"); ust0.textContent = "host"; pres[0].classList.remove("act"); stepsBottom(); }],
      [3.7, () => { code.stream(code.show("c-s1"), .06); ghostLine("Building a canvas Tetris."); ghostLine(""); stepsBottom(); }],
      [4.2, () => { code.caption("The model writes it on all three devices"); code.show("c-w"); wcard.classList.add("streaming"); stepsBottom(); }],
      [7.6, () => { wcard.classList.remove("streaming"); wcard.classList.add("fold"); wNm.textContent = "wrote"; wf.textContent = DONE; code.show("c-files"); file(0, "busy"); stepsBottom(); }],
      [7.75, () => { file(0, "done"); file(1, "busy"); }],
      [7.9, () => { file(1, "done"); file(2, "busy"); }],
      [8.05, () => { file(2, "done"); file(3, "busy"); stepsBottom(); }],
      [8.6, () => { code.caption("It runs"); file(3, "done"); openBrowser(); }],
      [8.9, () => showApp()],
      [10.0, () => { code.caption("Sam and Leo see it on their own devices"); peerState(0, "live"); }],
      [10.4, () => peerState(1, "live")],
      [12.2, () => code.caption("Your turn. Press Play.")],
    ],
    frame(kind, t) {
      if (kind === "text") return stepsBottom();
      if (t > 2.5 && t < 3.5) typeInto(codeTyped, PROMPT, 2.6, 3.35, t);
      if (t > W0 && t < W1 + .2) { writeTo(Math.min(LINES.length, Math.ceil((t - W0) / (W1 - W0) * LINES.length))); stepsBottom(); }
    },
    reset() {
      codeTyped.textContent = ""; codeComposer.classList.remove("hot");
      presence(0); pres.forEach(li => li.classList.remove("in", "act")); ust0.textContent = "host";
      peers.forEach((_, i) => peerState(i, "out")); clearGhosts();
      shown = -1; writeTo(0); wcard.classList.remove("streaming", "fold"); wNm.textContent = "writing"; wf.textContent = LIVE;
      files.forEach((_, i) => file(i, "hide"));
      browser.classList.add("shut"); app.classList.add("blank"); brLoad.classList.remove("go");
      if (!tet.human) { tet.stop(); tet.auto(11); }
      steps.scrollTop = 0;
    },
    final() {
      codeTyped.textContent = ""; codeComposer.classList.remove("hot");
      presence(3); pres.forEach(li => li.classList.remove("in", "act")); ust0.textContent = "host";
      clearGhosts();
      shown = -1; writeTo(LINES.length); wcard.classList.remove("streaming"); wcard.classList.add("fold"); wNm.textContent = "wrote"; wf.textContent = DONE;
      files.forEach((_, i) => file(i, "done"));
      browser.classList.remove("shut"); app.classList.remove("blank");
      peers.forEach((_, i) => peerState(i, "live")); peers.forEach(p => p.classList.remove("hit"));
      if (!tet.human) { tet.auto(11); tet.warm(18); }
      if (!RM) tet.start(); else tet.draw();
      stepsBottom();
    }
  });

  /* ---------- driver: one rAF, only while the window is on screen and the page is visible ---------- */
  const TL = { chat, code };
  let tab = "chat", raf = 0, last = 0, visible = false, frozen = false;
  const cur = () => TL[tab];
  const needs = () => !frozen && visible && !document.hidden && cur().started && !cur().done;
  const syncEnded = () => win.classList.toggle("ended", cur().done);
  function loop(now) {
    const dt = last ? Math.min(.1, (now - last) / 1000) : .016; last = now;
    cur().advance(dt); syncEnded();
    raf = needs() ? requestAnimationFrame(loop) : 0;
  }
  const wake = () => { if (needs() && !raf) { last = 0; raf = requestAnimationFrame(loop); } };
  const begin = name => { const tl = TL[name]; if (!tl.started) { tl.started = true; if (RM) tl.final(); else tl.reset(); } syncEnded(); wake(); };

  /* ---------- tabs ---------- */
  const tabs = [...win.querySelectorAll('[role="tab"]')];
  const select = (name, focus) => {
    if (name === tab) { if (visible) begin(name); return; }
    if (tab === "code" && tet.human && !tet.over && !tet.paused) tet.pause(true);
    tab = name; win.dataset.tab = name;
    tabs.forEach(b => { const on = b.id === "tab-" + name; b.setAttribute("aria-selected", String(on)); b.tabIndex = on ? 0 : -1; if (on && focus) b.focus(); });
    if (raf) { cancelAnimationFrame(raf); raf = 0; }
    requestAnimationFrame(() => { if (name === "code") { tet.draw(); stepsBottom(); } else toBottom(); });
    showCap(TL[name].cap || TL[name].lastCap);
    begin(name); wake();
  };
  tabs.forEach((b, i) => {
    b.addEventListener("click", () => { handedOff = true; select(b.id.slice(4)); });
    b.addEventListener("keydown", e => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault(); handedOff = true; select(tabs[(i + 1) % 2].id.slice(4), true);
    });
  });
  document.addEventListener("click", e => {
    const a = e.target.closest && e.target.closest("[data-go]"); if (!a) return;
    if (a.closest(".demo")) e.preventDefault();
    handedOff = true; select(a.dataset.go);
  });
  $("replay").addEventListener("click", () => {
    const tl = cur(); tl.started = true;
    if (raf) { cancelAnimationFrame(raf); raf = 0; }
    if (RM) { tl.final(); syncEnded(); return; }
    if (tab === "code" && tet.human) { tet.auto(11); }
    tl.reset(); syncEnded(); wake();
  });

  /* ---------- the game: keys are captured only while it has focus ---------- */
  const KEYMAP = { ArrowLeft: "left", ArrowRight: "right", ArrowUp: "rot", ArrowDown: "down", " ": "drop", Spacebar: "drop", x: "rot", z: "rot" };
  const takeOver = () => tet.play();
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
    if (visible) begin(tab);
    wake();
  }, { threshold: .3 }).observe(win);
  document.addEventListener("visibilitychange", wake);

  if (RM) { chat.final(); code.final(); chat.started = code.started = true; }
  else { chat.reset(); code.reset(); chat.cap = "Maya opens a room"; showCap(chat.cap); }

  // tests and screenshots: jump to a moment
  window.__demo = {
    seek(name, s, freeze) {
      seeking = true; frozen = !!freeze; select(name); const tl = TL[name];
      if (raf) { cancelAnimationFrame(raf); raf = 0; }
      tl.started = true; tl.reset(); while (tl.t < s) tl.advance(1 / 30);
      tet.draw(); syncEnded(); wake();
    },
    set frozen(v) { frozen = v; wake(); }, get t() { return cur().t; }, get tab() { return tab; }, tet
  };
})();
