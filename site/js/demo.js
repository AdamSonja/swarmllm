/* The demo: one window, two modes, eight steps.
   Chat: a laptop starts a room, a desktop joins with the code; together they hold enough memory, so the
   room presses Download model and each device fetches its own 20 of the 40 layers; "what is Pooled?", and while the answer streams a hidden state travels through every layer,
   laptop to desktop and back, once per word.
   Code: the tab switches itself; "build me a tetris game"; the files appear on the left as the agent
   writes them; it serves the game on :5173; then a change ("make the pieces blue and add a
   next-piece preview"), an edit, a reload.
   The HTML holds the finished state (readable without JS). This script rewinds and replays it. */
(() => {
  "use strict";
  document.documentElement.classList.add("js"); // also set early by boot.js
  const $ = id => document.getElementById(id);
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const demo = $("demo");
  if (!demo || !window.PooledTetris) return;

  const restart = (el, cls) => { el.classList.remove(cls); void el.offsetWidth; el.classList.add(cls); };
  const press = (el, t) => { restart(el, "press"); setTimeout(() => el.classList.remove("press"), t || 260); };
  const scene = s => { demo.dataset.scene = s; };
  const flag = (cls, on) => demo.classList.toggle(cls, on);
  const clamp01 = x => x < 0 ? 0 : x > 1 ? 1 : x;

  /* ---------- Chat | Code ---------- */
  const mChat = $("mChat"), mCode = $("mCode"), modes = mChat.parentNode, win = $("win");
  const mode = m => {
    if (demo.dataset.mode === m) return;
    demo.dataset.mode = m;
    [[mChat, "chat"], [mCode, "code"]].forEach(([b, k]) => { b.setAttribute("aria-selected", k === m); b.tabIndex = k === m ? 0 : -1; });
    win.setAttribute("aria-labelledby", m === "code" ? "mCode" : "mChat");
  };

  /* ---------- the timeline's shape ---------- */
  const ANSWER = $("a1").textContent;
  const WORDS = ANSWER.split(" ");
  const DL = 4.3, SPLIT = 6.45, FILL0 = 6.9, FILL1 = 8.9;  // the model card; Download pressed; the halves fill
  const CH = 9.55;                                             // chat
  const A0 = CH + 1.55;                                         // the answer starts
  const DUR = i => [1.6, .85, .5, .34][i] || .17;             // each word's trip; the first is slow enough to follow
  const WT = [A0]; WORDS.forEach((_, i) => WT.push(WT[i] + DUR(i)));
  const A1 = WT[WORDS.length];                                // the answer ends
  const C = Math.ceil((A1 + .9) * 10) / 10;                   // Code starts
  const c = x => C + x;                                       // Code's clock
  const STEPS = [0, 1.9, DL, SPLIT, CH, C, c(1.75), c(7.05)];
  const END = c(12.4);

  /* ---------- the caption bar ---------- */
  const dotBtns = [...demo.querySelectorAll(".sb-dots button")];
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
    const p = RM ? 100 : clamp01((t - STEPS[k]) / span) * 100;
    dotBtns[k].style.setProperty("--p", p.toFixed(1) + "%");
  };
  dotBtns.forEach((b, i) => b.setAttribute("aria-label", `Step ${i + 1} of ${STEPS.length}: ${CAPS[i]}`));

  /* ---------- 1, 2: the room forms ---------- */
  const CODE = "K7QX";
  const tabA = $("tabA"), tabB = $("tabB"), aGo = $("aGo"), bBtn = $("bBtn");
  const aCode = [...$("aCode").children], hdCode = [...$("hdCode").children];
  const slots = [...tabB.querySelectorAll(".slots i")];
  const setSlots = (n, cur) => slots.forEach((s, i) => {
    const ch = i < n ? CODE[i] : "";
    if (s.textContent !== ch) { s.textContent = ch; if (ch) restart(s, "lit"); }
    s.classList.toggle("cur", i === cur);
  });
  const letters = n => {
    aCode.forEach((s, i) => s.classList.toggle("off", i >= n));
    hdCode.forEach((s, i) => { s.textContent = i < n ? CODE[i] : "-"; s.classList.toggle("off", i >= n); });
  };
  const chips = [...$("chips").children];
  const gbSum = $("gbSum"), hdSum = gbSum.parentNode;
  const setDevices = (n, animate) => {
    chips.forEach((c, i) => { c.classList.toggle("out", i >= n); if (animate && i === n - 1) restart(c, "in"); });
    gbSum.textContent = 12 * n + " GB";
    if (animate) restart(hdSum, "bump");
  };

  /* ---------- 3: the model splits and loads ---------- */
  const model = $("model"), lane = $("lane"), mdS = $("mdS"), dlBtn = $("dlBtn");
  const cells = [...$("cells0").children, ...$("cells1").children];
  const hg = [$("hg0"), $("hg1")];
  let filled = -1;
  const fill = k => {                          // k of 20: each device loads its own 20 layers at once
    if (k === filled) return;
    cells.forEach((c, i) => {
      const on = (i % 20) < k, was = filled >= 0 && (i % 20) < filled;
      if (on !== c.classList.contains("f")) { c.classList.toggle("f", on); if (on && !was && !RM) restart(c, "f"); }
    });
    filled = k;
    const gb = (11.2 * k / 20).toFixed(1);
    hg.forEach(h => { h.textContent = k >= 20 ? "11.2 GB, ready" : `${gb} of 11.2 GB`; h.classList.toggle("ok", k >= 20); });
    mdS.textContent = k <= 0 ? "22.5 GB · 40 layers" : k < 20 ? `Downloading · ${(22.5 * k / 20).toFixed(1)} of 22.5 GB` : "Ready on 2 devices";
  };

  /* ---------- 4: a hidden state, through every layer, once per word ---------- */
  const pkt = $("pkt"), retPath = $("retPath"), wire = lane.querySelector(".lk-w i"), a1 = $("a1");
  let geo = null;
  const measure = () => {
    const L = lane.getBoundingClientRect(), k = (L.width / lane.offsetWidth) || 1;   // the stage may be scaled
    const r = i => { const b = cells[i].getBoundingClientRect(); return { x: (b.left - L.left + b.width / 2) / k, y: (b.top - L.top + b.height / 2) / k, b: (b.bottom - L.top) / k }; };
    const f0 = r(0), l0 = r(19), f1 = r(20), l1 = r(39);
    const by = f0.b + 8, dip = 12;
    geo = { f0, l0, f1, l1, by, dip };
    retPath.setAttribute("d", `M${l1.x.toFixed(1)} ${by.toFixed(1)} Q${((f0.x + l1.x) / 2).toFixed(1)} ${(by + dip * 2).toFixed(1)} ${f0.x.toFixed(1)} ${by.toFixed(1)}`);
  };
  let hotI = -1;
  const setHot = i => {
    if (i === hotI) return;
    if (hotI >= 0) cells[hotI].classList.remove("hot");
    if (i >= 0) cells[i].classList.add("hot");
    hotI = i;
  };
  const tok = d => chips.forEach((c, j) => c.classList.toggle("tok", j === d));
  const trip = u => {                          // u in [0,1): where the hidden state is on this word's trip
    measure();
    const { f0, l0, f1, l1, by, dip } = geo;
    if (u < .4) { const k = u / .4, i = Math.min(19, Math.floor(k * 20)); return { x: f0.x + (l0.x - f0.x) * k, y: f0.y, i, d: 0 }; }
    if (u < .5) { const k = (u - .4) / .1; return { x: l0.x + (f1.x - l0.x) * k, y: f0.y, i: -1, d: -1, hop: k }; }
    if (u < .9) { const k = (u - .5) / .4, i = 20 + Math.min(19, Math.floor(k * 20)); return { x: f1.x + (l1.x - f1.x) * k, y: f0.y, i, d: 1 }; }
    const k = (u - .9) / .1, cx = (f0.x + l1.x) / 2, cy = by + dip * 2;
    const x = (1 - k) * (1 - k) * l1.x + 2 * (1 - k) * k * cx + k * k * f0.x, y = (1 - k) * (1 - k) * by + 2 * (1 - k) * k * cy + k * k * by;
    return { x, y, i: -1, d: 0, back: true };
  };
  let words = -1, flowing = false;
  const flow = t => {
    if (t < A0 || t >= A1) {
      if (flowing) { flowing = false; pkt.classList.remove("on", "back"); lane.classList.remove("hop"); wire.style.transform = ""; setHot(-1); tok(-1); a1.classList.remove("cursor"); }
      const n = t < A0 ? 0 : WORDS.length;
      if (n !== words) { words = n; a1.textContent = n ? ANSWER : ""; }
      return;
    }
    flowing = true;
    let w = 0; while (w < WORDS.length - 1 && t >= WT[w + 1]) w++;
    if (w !== words) { words = w; a1.textContent = WORDS.slice(0, w).join(" "); a1.classList.add("cursor"); toBottom(); }
    const p = trip((t - WT[w]) / DUR(w));
    pkt.style.transform = `translate(${p.x.toFixed(1)}px,${p.y.toFixed(1)}px)`;
    pkt.classList.add("on"); pkt.classList.toggle("back", !!p.back);
    lane.classList.toggle("hop", p.hop != null);
    wire.style.transform = p.hop != null ? `scaleX(${p.hop.toFixed(3)})` : "";
    setHot(p.i); tok(p.back ? -1 : p.d);
  };

  /* ---------- chat ---------- */
  const msgs = $("msgs"), chatTyped = $("chatTyped"), chatComposer = $("chatComposer");
  const Q1 = "what is Pooled?";
  const toBottom = () => { msgs.scrollTop = msgs.scrollHeight; };
  const typeInto = (el, text, t0, t1, t) => {
    const n = Math.max(0, Math.min(text.length, Math.ceil((t - t0) / (t1 - t0) * text.length)));
    if (el.textContent.length !== n) el.textContent = text.slice(0, n);
  };
  const show = k => {
    const el = demo.querySelector(`[data-at="${k}"]`); if (!el) return null;
    el.classList.remove("pending"); if (!RM) restart(el, "enter"); return el;
  };

  /* ---------- 5 to 7: Code ---------- */
  const log = $("log"), codeTyped = $("codeTyped"), codeComposer = $("codeComposer");
  const files = demo.querySelector(".files"), ftree = $("ftree");
  const fItems = Object.fromEntries([...ftree.children].map(li => [li.dataset.f, li]));
  const lv = $("lv"), lvPre = $("lvPre"), lvF = $("lvF"), lvN = $("lvN");
  const pv = $("pv"), app = $("app"), brLoad = $("brLoad"), game = $("game");
  const PROMPT = "build me a tetris game", PROMPT2 = "make the pieces blue and add a next-piece preview";
  const WARM = ["#F08A6C", "#F2C14E", "#6CC5A1", "#B18CF0", "#5EB8E8", "#F28DB2", "#9BD16B"];
  const BLUE = ["#3152FF", "#6E86FF", "#A5B4FC", "#C9D1F7", "#8EA2FF", "#4A5FD0", "#DCE2FF"];
  const SRC = {
    "index.html": ['<!doctype html>', '<html lang="en">', '<head>', '  <meta charset="utf-8">', '  <title>Tetris</title>', '  <link rel="stylesheet" href="style.css">',
      '</head>', '<body>', '  <canvas id="board" width="200" height="400"></canvas>', '  <p id="score">0</p>', '  <script src="game.js"></script>', '</body>'],
    "style.css": ['body {', '  margin: 0; display: grid; place-items: center;', '  min-height: 100vh; background: #0B0F1F;', '}', '#board { border-radius: 4px; }', '#score {',
      '  font: 500 16px ui-monospace, monospace;', '  color: #EEF0F6;', '}'],
    "game.js": ['const COLS = 10, ROWS = 20, SIZE = 20;', 'const COLORS = ["#F08A6C", "#F2C14E", "#6CC5A1", ...];', 'const ctx = board.getContext("2d");', 'let grid = empty(), queue = bag();',
      'let piece = spawn(), score = 0, last = 0;', '', 'function empty() {', '  return Array.from({ length: ROWS }, () => Array(COLS).fill(0));', '}', '',
      'function spawn() {', '  if (queue.length < 2) queue.push(...bag());', '  const k = queue.shift();', '  return { k, x: 3, y: -1, r: 0 };', '}', '',
      'function fits(p) {', '  return cells(p).every(([x, y]) =>', '    x >= 0 && x < COLS && y < ROWS && !grid[y]?.[x]);', '}', '',
      'function clearLines() {', '  for (let y = ROWS - 1; y >= 0; y--) {', '    if (!grid[y].every(Boolean)) continue;', '    grid.splice(y, 1); grid.unshift(Array(COLS).fill(0));',
      '    score += 100; y++;', '  }', '}', '', 'function loop(t) {', '  if (t - last > speed()) { drop(); last = t; }', '  draw(grid, piece);', '  requestAnimationFrame(loop);', '}', '',
      'addEventListener("keydown", e => move(e.key));', 'requestAnimationFrame(loop);'],
    edit: ['const COLORS = ["#3152FF", "#6E86FF", "#A5B4FC", ...];', 'const next = document.getElementById("next");', 'function drawNext() {', '  paint(next, queue[0]);', '}']
  };
  const LINES_OUT = { "index.html": 12, "style.css": 9, "game.js": 38, edit: 41 };
  const tet = window.PooledTetris($("tetris"), {
    seed: 11, tick: 70, colors: WARM, next: $("nextc"),
    onScore: (s, l) => { $("score").textContent = s.toLocaleString("en-US"); $("lines").textContent = l; }
  });
  const logBottom = () => { log.scrollTop = log.scrollHeight; };
  let liveSrc = [], liveShown = -1, writing = null;
  const liveTo = k => {
    if (k === liveShown) return;
    if (k < liveShown || liveShown < 0) { lvPre.textContent = ""; liveShown = 0; }
    for (let i = liveShown; i < k; i++) { const sp = document.createElement("span"); sp.textContent = liveSrc[i] || " "; lvPre.append(sp); }
    while (lvPre.children.length > 5) lvPre.firstChild.remove();
    liveShown = k; lvN.textContent = `${k} line${k === 1 ? "" : "s"}`;
  };
  const fileState = (name, st, lines) => {
    const li = fItems[name];
    if (st === "hide") { li.classList.add("pending"); li.classList.remove("w", "mod"); return; }
    if (li.classList.contains("pending")) { li.classList.remove("pending"); if (!RM) restart(li, "new"); }
    li.classList.toggle("w", st === "w"); if (st === "mod") li.classList.add("mod");
    if (lines != null) li.querySelector(".fm").textContent = lines;
    files.classList.toggle("none", ftree.querySelectorAll("li:not(.pending)").length === 0);
  };
  // write: [t0, t1, file, lines, the file row's final line count]
  const liveStart = (name, key, t0, t1) => {
    liveSrc = SRC[key]; lvF.textContent = name; liveShown = -1; liveTo(0);
    lv.classList.remove("pending"); if (!RM) restart(lv, "enter");
    writing = [t0, t1, liveSrc.length, name, key];
    fileState(name, "w", key === "edit" ? null : 0);
    lv.parentNode.append(lv); logBottom();
  };
  const liveEnd = () => { const w = writing; lv.classList.add("pending"); writing = null; tok(-1); if (w) fileState(w[3], "done", LINES_OUT[w[4]]); };
  const tool = k => { const el = show(k); if (el) { log.append(el); logBottom(); } return el; };
  const saysText = new Map([...log.querySelectorAll("[data-at]")].map(el => [el, (el.querySelector(".say") || {}).textContent]));
  let streams = [];
  const reveal = (el, rate) => streams.push({ el: el.querySelector(".say"), text: saysText.get(el), t0: tl.t, rate });
  const setRun = (k, run) => { const st = demo.querySelector(`[data-at="${k}"] .st`); st.textContent = run ? "running" : "done"; st.classList.toggle("run", run); };
  const openPreview = () => { flag("served", true); flag("app-on", true); };
  const reload = () => { if (!RM) restart(brLoad, "go"); };
  const v1 = () => { app.classList.remove("v2"); tet.setColors(WARM); tet.showNext(false); };
  const v2 = () => { app.classList.add("v2"); tet.setColors(BLUE); tet.showNext(true); };
  const runGame = warmN => { if (!tet.human) { tet.auto(11); tet.warm(warmN); } if (!RM) tet.start(); else tet.draw(); };

  const tl = { t: 0, fired: 0, done: false, started: false };
  const EVENTS = [
    // 1: the laptop starts a room
    [.45, () => press(aGo, 260)],
    [.7, () => { tabA.classList.add("done"); setDevices(1, true); }],
    ...[0, 1, 2, 3].map(i => [.78 + i * .1, () => letters(i + 1)]),
    // 2: the desktop types the code and joins; the two tabs fold into one room
    [1.9, () => setSlots(0, 0)],
    [2.08, () => setSlots(1, 1)], [2.26, () => setSlots(2, 2)], [2.44, () => setSlots(3, 3)],
    [2.62, () => { setSlots(4, -1); bBtn.classList.add("ready"); }],
    [2.9, () => press(bBtn, 260)],
    [3.15, () => { tabB.classList.add("done"); tabA.classList.add("met"); setDevices(2, true); }],
    [3.7, () => flag("merge", true)],
    // 3: the room holds enough for the model; Download model
    [DL, () => scene("split")],
    [5.65, () => dlBtn.classList.add("hover")],
    [6.05, () => { dlBtn.classList.remove("hover"); press(dlBtn, 240); }],
    // 4: each device fetches its own half
    [SPLIT, () => model.classList.add("dl")],
    [SPLIT + .2, () => model.classList.add("split")],
    // 5: chat
    [CH, () => scene("chat")],
    [CH + .45, () => { measure(); chatComposer.classList.add("hot"); }],
    [CH + 1.4, () => { chatTyped.textContent = ""; chatComposer.classList.remove("hot"); show("q1"); show("a1"); a1.textContent = ""; toBottom(); }],
    // 6: Code: the tab switches itself
    [C, () => { press(modes, 300); mode("code"); }],
    [c(.2), () => scene("code")],
    [c(.4), () => codeComposer.classList.add("hot")],
    [c(1.3), () => { codeTyped.textContent = ""; codeComposer.classList.remove("hot"); tool("c-q"); }],
    [c(1.5), () => reveal(tool("c-s1"), .035)],
    // 7: it writes three files and serves them
    [c(1.75), () => liveStart("index.html", "index.html", c(1.75), c(2.2))],
    [c(2.3), () => { liveEnd(); tool("c-t0"); }],
    [c(2.4), () => liveStart("style.css", "style.css", c(2.4), c(2.75))],
    [c(2.85), () => { liveEnd(); tool("c-t1"); }],
    [c(2.95), () => liveStart("game.js", "game.js", c(2.95), c(4.5))],
    [c(4.6), () => { liveEnd(); tool("c-t2"); }],
    [c(4.75), () => { setRun("c-t3", true); tool("c-t3"); }],
    [c(5.1), () => { setRun("c-t3", false); openPreview(); }],
    [c(5.3), reload],
    [c(5.45), () => { app.classList.remove("blank"); runGame(10); }],
    [c(5.7), () => reveal(tool("c-s2"), .035)],
    // 8: a change, an edit, a reload
    [c(6.9), () => flag("app-on", false)],
    [c(7.05), () => codeComposer.classList.add("hot")],
    [c(8.25), () => { codeTyped.textContent = ""; codeComposer.classList.remove("hot"); tool("c-q2"); }],
    [c(8.45), () => liveStart("game.js", "edit", c(8.45), c(8.9))],
    [c(9.0), () => { liveEnd(); fileState("game.js", "mod"); tool("c-t4"); }],
    [c(9.25), () => { flag("app-on", true); reload(); app.classList.add("blank"); }],
    [c(9.5), () => { v2(); app.classList.remove("blank"); }],
    [c(9.75), () => reveal(tool("c-s3"), .035)],
  ];
  EVENTS.sort((a, b) => a[0] - b[0]);

  const frame = t => {
    if (t > DL && t < FILL0) fill(0);
    if (t >= FILL0 && t <= FILL1 + .1) fill(Math.min(20, Math.ceil((t - FILL0) / (FILL1 - FILL0) * 20)));
    if (t > CH + .5 && t < CH + 1.4) typeInto(chatTyped, Q1, CH + .55, CH + 1.15, t);
    if (t >= CH) flow(t);
    if (t > c(.4) && t < c(1.3)) typeInto(codeTyped, PROMPT, c(.5), c(1.12), t);
    if (t > c(7.05) && t < c(8.25)) typeInto(codeTyped, PROMPT2, c(7.15), c(8.05), t);
    if (writing) {
      const [a, b, n] = writing;
      liveTo(Math.max(0, Math.min(n, Math.ceil((t - a) / (b - a) * n))));
      fileState(writing[3], "w", writing[4] === "edit" ? null : liveShown);
      tok(Math.floor(t * 7) % 2);
      logBottom();
    }
    streams = streams.filter(s => {
      const w = s.text.split(" "), n = Math.min(w.length, Math.floor((t - s.t0) / s.rate) + 1);
      s.el.textContent = w.slice(0, n).join(" "); s.el.classList.toggle("cursor", n < w.length);
      logBottom();
      return n < w.length;
    });
  };

  tl.reset = () => {
    tl.t = 0; tl.fired = 0; tl.done = false; streams = []; writing = null;
    scene("tabs"); mode("chat"); ["merge", "served", "app-on"].forEach(c => flag(c, false));
    tabA.classList.remove("done", "met"); tabB.classList.remove("done"); letters(0);
    setSlots(0, -1); bBtn.classList.remove("ready", "press"); aGo.classList.remove("press");
    setDevices(0); chips.forEach(c => c.classList.remove("in"));
    model.classList.remove("split", "dl"); dlBtn.classList.remove("hover", "press"); filled = -1; fill(0); geo = null;
    words = -1; flowing = true; flow(0);
    demo.querySelectorAll("[data-at]").forEach(el => el.classList.add("pending"));
    chatTyped.textContent = ""; chatComposer.classList.remove("hot");
    codeTyped.textContent = ""; codeComposer.classList.remove("hot");
    Object.keys(fItems).forEach(n => fileState(n, "hide")); files.classList.add("none");
    lv.classList.add("pending"); liveShown = -1; liveSrc = []; liveTo(0);
    saysText.forEach((txt, el) => { const s = el.querySelector(".say"); if (s) { s.textContent = txt; s.classList.remove("cursor"); } });
    ["c-q", "c-s1", "c-t0", "c-t1", "c-t2", "c-t3", "c-s2", "c-q2", "c-t4", "c-s3"].forEach(k => log.append(demo.querySelector(`[data-at="${k}"]`)));
    app.classList.add("blank"); brLoad.classList.remove("go");
    if (!tet.human) { tet.stop(); v1(); tet.auto(11); }
    msgs.scrollTop = 0; log.scrollTop = 0;
    paintBar(0, true);
  };
  tl.final = () => {
    streams = []; writing = null; tl.done = true; tl.started = true; tl.t = END; tl.fired = EVENTS.length;
    scene("code"); mode("code"); flag("merge", true); flag("served", true); flag("app-on", true);
    tabA.classList.add("done", "met"); tabB.classList.add("done"); letters(4); setSlots(4, -1);
    setDevices(2); model.classList.add("dl", "split"); fill(20);
    words = -1; flowing = true; flow(END);
    demo.querySelectorAll("[data-at]").forEach(el => el.classList.remove("pending", "enter"));
    saysText.forEach((txt, el) => { const s = el.querySelector(".say"); if (s) { s.textContent = txt; s.classList.remove("cursor"); } });
    chatTyped.textContent = ""; codeTyped.textContent = ""; chatComposer.classList.remove("hot"); codeComposer.classList.remove("hot");
    Object.keys(fItems).forEach(n => fileState(n, "done", LINES_OUT[n])); fileState("game.js", "mod", LINES_OUT.edit);
    lv.classList.add("pending"); tok(-1);
    app.classList.remove("blank"); v2(); runGame(18);
    toBottom(); logBottom();
    paintBar(END);
  };
  tl.advance = dt => {
    tl.t += dt;
    while (tl.fired < EVENTS.length && EVENTS[tl.fired][0] <= tl.t) { EVENTS[tl.fired][1](); tl.fired++; }
    frame(tl.t);
    paintBar(tl.t, true);
    if (tl.t >= END && tl.fired >= EVENTS.length) tl.done = true;
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

  // jump to a moment; with reduced motion, to a step's finished state, frozen
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
      const ends = [1.6, 3.6, 6.0, CH - .3, A1 + .5, c(1.6), c(6.6)];
      seek(ends[k], true); tl.done = true; flow(A1 + 1); paintBar(STEPS[k]); return;
    }
    seek(STEPS[k]);
  };
  dotBtns.forEach((b, i) => b.addEventListener("click", () => goStep(i)));
  $("replay").addEventListener("click", () => goStep(0));
  mChat.addEventListener("click", () => { if (demo.dataset.mode !== "chat" || tl.done) goStep(0); });
  mCode.addEventListener("click", () => { if (demo.dataset.mode !== "code" || tl.done) goStep(5); });
  modes.addEventListener("keydown", e => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const b = e.key === "ArrowLeft" ? mChat : mCode; b.focus(); b.click();
  });
  addEventListener("resize", () => { if (demo.dataset.scene === "chat") measure(); });

  /* ---------- the game: it plays itself; arrow keys take over while it has focus ---------- */
  const KEYMAP = { ArrowLeft: "left", ArrowRight: "right", ArrowUp: "rot", ArrowDown: "down", " ": "drop", Spacebar: "drop", x: "rot", z: "rot" };
  game.addEventListener("keydown", e => {
    if (e.key === "Escape") { game.blur(); return; }
    const a = KEYMAP[e.key]; if (!a) return;
    e.preventDefault();
    if (!tet.human || tet.over) tet.play();
    tet.act(a);
  });
  game.addEventListener("pointerdown", () => game.focus({ preventScroll: true }));
  game.addEventListener("blur", () => { if (tet.human) { tet.auto(11); tet.warm(10); if (!RM) tet.start(); } });

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
    set frozen(v) { frozen = v; wake(); }, get t() { return tl.t; }, STEPS, END, C, A1, tet
  };
})();
