/* The demo window: Chat | Code tabs, two short example timelines, and a playable Tetris.
   The HTML holds the finished state (readable without JS). This script rewinds and replays it. */
(() => {
  "use strict";
  document.documentElement.classList.add("js"); // also set early by boot.js
  const $ = id => document.getElementById(id);
  const win = $("win");
  if (!win || !window.PooledTetris) return;
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const restart = (el, cls) => { el.classList.remove(cls); void el.offsetWidth; el.classList.add(cls); };

  /* ---------- tiny timeline engine ---------- */
  function Timeline(root, { events, end, reset, final, frame }) {
    const says = [...root.querySelectorAll(".say")].map(el => ({ el, text: el.textContent, html: el.innerHTML }));
    const tl = { t: 0, fired: 0, streams: [], done: false, started: false };
    const restore = s => { s.el.innerHTML = s.html; s.el.classList.remove("cursor"); };
    tl.show = k => {
      const el = root.querySelector(`[data-at="${k}"]`); if (!el) return null;
      el.classList.remove("pending"); restart(el, "enter"); return el;
    };
    tl.stream = (el, rate, onDone) => {
      const s = says.find(x => x.el === el || el.contains(x.el)); if (!s) return;
      s.el.textContent = ""; s.el.classList.add("cursor");
      tl.streams.push({ s, words: s.text.split(" "), t0: tl.t, rate, onDone });
    };
    tl.reset = () => {
      tl.t = 0; tl.fired = 0; tl.streams = []; tl.done = false;
      root.querySelectorAll("[data-at]").forEach(el => el.classList.add("pending"));
      says.forEach(restore);
      reset();
    };
    tl.final = () => {
      tl.streams = []; tl.done = true; tl.started = true;
      root.querySelectorAll("[data-at]").forEach(el => el.classList.remove("pending", "enter"));
      says.forEach(restore);
      final();
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

  /* ---------- CHAT: three devices join, then the room talks ---------- */
  const pc = $("p-chat"), msgs = $("msgs"), devs = [...pc.querySelectorAll(".dev")];
  const GB = [10, 12, 2], SCALE = 24;
  const pool = $("pool"), mstatT = $("mstatT"), poolGb = $("poolGb"), cnt = $("cnt");
  const chatTyped = $("chatTyped"), chatComposer = $("chatComposer"), chatWho = $("chatWho");
  const Q1 = "what is pooled?", Q2 = "can it write code?";
  const toBottom = () => { msgs.scrollTop = msgs.scrollHeight; };
  const setPool = n => {
    devs.forEach((d, i) => d.classList.toggle("out", i >= n));
    GB.forEach((g, i) => { $("seg" + i).style.width = i < n ? (g / SCALE * 100) + "%" : "0%"; });
    poolGb.textContent = GB.slice(0, n).reduce((a, b) => a + b, 0) + " GB";
    cnt.textContent = n ? (n === 1 ? "1 device" : n + " devices") : "";
  };
  const setStat = (txt, ok) => { mstatT.textContent = txt; pool.classList.toggle("ok", !!ok); };
  const typer = (who, cls) => { chatWho.textContent = who; chatWho.className = "av typer " + cls; chatComposer.classList.add("hot"); };
  const sent = () => { chatTyped.textContent = ""; chatComposer.classList.remove("hot"); };
  const join = i => { setPool(i + 1); restart(devs[i], "in"); };
  const answer = k => {
    const li = chat.show(k); li.classList.add("live-share");
    chat.stream(li, .07, () => li.classList.remove("live-share"));
    toBottom();
  };
  let seeking = false, handedOff = false;
  const handOff = () => { if (seeking || handedOff || tab !== "chat") return; handedOff = true; setTimeout(() => select("code"), 0); };

  const chat = Timeline(pc, {
    end: 8.1,
    events: [
      [.3, () => join(0)],
      [.8, () => join(1)],
      [1.3, () => { join(2); setStat("Loading"); }],
      [1.8, () => { setStat("Ready", true); msgs.classList.remove("empty"); chat.show("rdy"); toBottom(); }],
      [2.1, () => typer("L", "a3")],
      [3.0, () => { sent(); chat.show("q1"); toBottom(); }],
      [3.3, () => answer("a1")],
      [4.6, () => typer("M", "a1")],
      [5.6, () => { sent(); chat.show("q2"); toBottom(); }],
      [5.9, () => answer("a2")],
      [8.0, handOff],
    ],
    frame(kind, t) {
      if (kind === "text") return toBottom();
      if (t > 2.1 && t < 3.0) typeInto(chatTyped, Q1, 2.2, 2.85, t);
      if (t > 4.6 && t < 5.6) typeInto(chatTyped, Q2, 4.7, 5.45, t);
    },
    reset() {
      setPool(0); setStat("Waiting"); sent(); msgs.classList.add("empty");
      pc.querySelectorAll(".live-share").forEach(el => el.classList.remove("live-share"));
      devs.forEach(d => d.classList.remove("in"));
      msgs.scrollTop = 0;
    },
    final() { setPool(3); setStat("Ready", true); sent(); msgs.classList.remove("empty"); devs.forEach(d => d.classList.remove("in")); pc.querySelectorAll(".live-share").forEach(el => el.classList.remove("live-share")); toBottom(); }
  });

  /* ---------- CODE: ask, files stream, the browser opens, the game plays ---------- */
  const pk = $("p-code"), steps = $("steps"), files = [...pk.querySelectorAll("[data-f]")];
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
  const file = (i, state) => {
    const li = files[i];
    li.classList.toggle("wait-f", state === "hide");
    li.classList.toggle("busy", state === "busy");
    if (state === "busy") restart(li, "new");
  };
  const openBrowser = () => { browser.classList.remove("shut"); restart(brLoad, "go"); };
  const showApp = () => {
    app.classList.remove("blank");
    if (!tet.human) { tet.auto(11); tet.warm(14); }
    if (!RM || tet.human) tet.start(); else tet.draw();
  };

  const code = Timeline(pk, {
    end: 2.4,
    events: [
      [.1, () => codeComposer.classList.add("hot")],
      [.95, () => { codeTyped.textContent = ""; codeComposer.classList.remove("hot"); code.show("c-q"); }],
      [1.05, () => code.stream(code.show("c-s1"), .05)],
      [1.15, () => { code.show("c-files"); file(0, "busy"); stepsBottom(); }],
      [1.3, () => { file(0, "done"); file(1, "busy"); }],
      [1.45, () => { file(1, "done"); file(2, "busy"); }],
      [1.6, () => { file(2, "done"); file(3, "busy"); }],
      [1.75, () => { file(3, "done"); file(4, "busy"); code.show("c-sum"); stepsBottom(); }],
      [1.95, () => { file(4, "done"); openBrowser(); }],
      [2.25, () => showApp()],
    ],
    frame(kind, t) {
      if (kind === "text") return stepsBottom();
      if (t > .1 && t < .95) typeInto(codeTyped, PROMPT, .15, .8, t);
    },
    reset() {
      codeTyped.textContent = ""; codeComposer.classList.remove("hot");
      files.forEach((_, i) => file(i, "hide"));
      browser.classList.add("shut"); app.classList.add("blank"); brLoad.classList.remove("go");
      if (!tet.human) { tet.stop(); tet.auto(11); }
      steps.scrollTop = 0;
    },
    final() {
      codeTyped.textContent = ""; codeComposer.classList.remove("hot");
      files.forEach((_, i) => file(i, "done"));
      browser.classList.remove("shut"); app.classList.remove("blank");
      if (!tet.human) { tet.auto(11); tet.warm(18); }
      if (!RM) tet.start(); else tet.draw();
      stepsBottom();
    }
  });

  /* ---------- driver: one rAF, only while the window is on screen ---------- */
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
    if (name === "code") requestAnimationFrame(() => tet.draw());
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
  else { chat.reset(); code.reset(); }

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
