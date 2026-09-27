/* The demo: one window, two modes, eight steps, looping.
   First load: the finished app plays for a moment (the hook), then the story starts from the room.
   Chat: a laptop starts a room (it already has a friendly name), a desktop joins with the code; the room
   picks a model, a friend's laptop joins, Download; each device fetches only its own layers (one from
   its cache), with a time estimate; "what is Pooled?", and while the answer streams a hidden state
   travels through every layer on every device, once per word.
   Code: the tab switches itself; the visitor asks for an app (a different one each loop: Tetris, 2048,
   a space shooter, Snake, Breakout); the files appear as the agent writes them; it serves the app on
   :5173; then a change request, an edit, a reload.
   The HTML holds the finished state (readable without JS). This script rewinds and replays it. */
(() => {
  "use strict";
  document.documentElement.classList.add("js"); // also set early by boot.js
  const $ = id => document.getElementById(id);
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const demo = $("demo");
  if (!demo || !window.PooledApps) return;
  const SPEED = 1.33;              // the story runs 33% faster than its timeline's seconds

  const restart = (el, cls) => { el.classList.remove(cls); void el.offsetWidth; el.classList.add(cls); };
  const press = (el, t) => { restart(el, "press"); setTimeout(() => el.classList.remove("press"), t || 260); };
  const scene = s => { demo.dataset.scene = s; };
  const flag = (cls, on) => demo.classList.toggle(cls, on);
  const clamp01 = x => x < 0 ? 0 : x > 1 ? 1 : x;
  const pickOne = a => a[(Math.random() * a.length) | 0];

  /* ---------- friendly names: every device gets one, and you can change it ---------- */
  const ADJ = ["Amber", "Calm", "Quiet", "Swift", "Bright", "Lucky", "Misty", "Sunny", "Cozy", "Brave", "Mellow", "Nimble"];
  const ANI = ["Fox", "Heron", "Lynx", "Otter", "Panda", "Robin", "Koala", "Falcon", "Badger", "Owl", "Wren", "Moose"];
  let NAMES = ["Amber Fox", "Calm Heron", "Quiet Lynx"];
  const newNames = () => {
    const a = ADJ.slice().sort(() => Math.random() - .5), b = ANI.slice().sort(() => Math.random() - .5);
    NAMES = [0, 1, 2].map(i => `${a[i]} ${b[i]}`);
    demo.querySelectorAll("[data-name]").forEach(el => { el.textContent = NAMES[+el.dataset.name]; });
    ANSWER = `Pooled splits one open model across this room. ${NAMES[0]} runs layers 1 to 15, ${NAMES[1]} 16 to 30, ${NAMES[2]} 31 to 40. Every word passes through all three.`;
    WORDS = ANSWER.split(" "); timeWords();
  };

  /* ---------- the apps the agent builds ---------- */
  const html = (title, extra) => ['<!doctype html>', '<html lang="en">', '<head>', '  <meta charset="utf-8">', `  <title>${title}</title>`, '  <link rel="stylesheet" href="style.css">', '</head>', '<body>', '  <canvas id="game"></canvas>', ...(extra || []), '  <script src="game.js"></script>', '</body>'];
  const css = bg => ['* { box-sizing: border-box; }', 'body {', '  margin: 0; display: grid; place-items: center;', `  min-height: 100vh; background: ${bg};`, '}', '#game { width: min(92vw, 440px); aspect-ratio: 5 / 6; }', 'body { font: 500 14px ui-monospace, monospace; color: #EEF0F6; }'];
  const APPS = {
    tetris: {
      dir: "tetris", title: "Tetris", ask: "a tetris game", prompt: "build me a tetris game",
      src: ['const COLS = 10, ROWS = 20;', 'const COLORS = ["#F08A6C", "#F2C14E", "#6CC5A1", ...];', 'const SHAPES = { I: [[0,1],[1,1],[2,1],[3,1]], O: [[1,0],[2,0],[1,1],[2,1]], ... };', 'let grid = empty(), queue = bag(), piece = spawn();', 'let score = 0, lines = 0, last = 0;', '',
        'function spawn() {', '  if (queue.length < 2) queue.push(...bag());', '  return { k: queue.shift(), x: 3, y: -1, r: 0 };', '}', '', 'function fits(p) {', '  return cells(p).every(([x, y]) =>', '    x >= 0 && x < COLS && y < ROWS && !grid[y]?.[x]);', '}', '',
        'function clearLines() {', '  const full = grid.filter(row => row.every(Boolean));', '  grid = grid.filter(row => !row.every(Boolean));', '  while (grid.length < ROWS) grid.unshift(Array(COLS).fill(0));', '  lines += full.length;', '  score += [0, 100, 300, 500, 800][full.length];', '}', '',
        'function loop(t) {', '  if (t - last > speed()) { drop(); last = t; }', '  draw(grid, piece);', '  requestAnimationFrame(loop);', '}', '', 'addEventListener("keydown", e => move(e.key));', 'requestAnimationFrame(loop);'],
      done: "Tetris is running on :5173. Click the preview to play.",
      prompt2: "make the pieces blue and add a next-piece preview",
      diff: [["del", 'const COLORS = ["#F08A6C", "#F2C14E", "#6CC5A1", ...];'], ["add", 'const COLORS = ["#2A45E0", "#6E86FF", "#A5B4FC", ...];'], ["add", 'const next = document.getElementById("next");'], ["add", 'function drawNext() { paint(next, queue[0]); }'], ["add", '// called after each spawn()']],
      done2: "Done. The pieces are blue, and the next one shows beside the board.",
      aria: "The Tetris game the agent wrote, playing itself. Focus it and use the arrow keys and space to take over; Escape hands it back."
    },
    "2048": {
      dir: "2048", title: "2048", ask: "a 2048 game", prompt: "build me a 2048 game",
      src: ['const N = 4;', 'const TILE = { 2: "#EEE4DA", 4: "#EDE0C8", 8: "#F2B179", ... };', 'let board = Array.from({ length: N }, () => Array(N).fill(0));', 'let score = 0;', '',
        'function slide(row) {', '  const v = row.filter(Boolean), out = [];', '  for (let i = 0; i < v.length; i++) {', '    if (v[i] === v[i + 1]) { out.push(v[i] * 2); score += v[i] * 2; i++; }', '    else out.push(v[i]);', '  }', '  while (out.length < N) out.push(0);', '  return out;', '}', '',
        'function move(dir) {', '  const before = JSON.stringify(board);', '  board = rotate(board, dir).map(slide);', '  board = rotate(board, -dir);', '  if (JSON.stringify(board) !== before) spawn();', '  animate(); draw();', '}', '',
        'function spawn() {', '  const empty = cellsWhere(v => v === 0);', '  const [x, y] = empty[Math.floor(Math.random() * empty.length)];', '  board[y][x] = Math.random() < 0.9 ? 2 : 4;', '}', '',
        'const KEYS = { ArrowLeft: 0, ArrowUp: 1, ArrowRight: 2, ArrowDown: 3 };', 'addEventListener("keydown", e => e.key in KEYS && move(KEYS[e.key]));', 'spawn(); spawn(); draw();'],
      done: "2048 is running on :5173. Click the preview and use the arrow keys.",
      prompt2: "make the tiles blue and keep a best score",
      diff: [["del", 'const TILE = { 2: "#EEE4DA", 4: "#EDE0C8", 8: "#F2B179", ... };'], ["add", 'const TILE = { 2: "#DCE2FF", 4: "#C9D1F7", 8: "#A5B4FC", ... };'], ["add", 'let best = Number(localStorage.best) || 0;'], ["add", 'best = Math.max(best, score);'], ["add", 'localStorage.best = best;']],
      done2: "Done. The tiles are blue, and your best score sticks around.",
      aria: "The 2048 game the agent wrote, playing itself. Focus it and use the arrow keys to take over; Escape hands it back."
    },
    shooter: {
      dir: "space-shooter", title: "Space shooter", ask: "a space shooter", prompt: "build me a space shooter",
      src: ['const ship = { x: 0.5, lives: 3 };', 'const shots = [], bombs = [], sparks = [];', 'let enemies = wave(4, 7), score = 0;', '',
        'function wave(rows, cols) {', '  const out = [];', '  for (let y = 0; y < rows; y++)', '    for (let x = 0; x < cols; x++) out.push({ x, y, alive: true });', '  return out;', '}', '',
        'function update(dt) {', '  ship.x = clamp(ship.x + input.dx * dt, 0.05, 0.95);', '  for (const s of shots) s.y -= 520 * dt;', '  for (const b of bombs) b.y += 170 * dt;', '  for (const e of enemies) if (e.alive && hit(e, shots)) {', '    e.alive = false; score += 10 * (4 - e.y);', '  }', '  if (Math.random() < dt * 1.3) enemyFires();', '  if (!enemies.some(e => e.alive)) enemies = wave(4, 7);', '}', '',
        'function draw() {', '  ctx.fillStyle = "#0B0F1F"; ctx.fillRect(0, 0, W, H);', '  enemies.forEach(drawInvader);', '  drawShip(ship.x * W, H - 34);', '  hud(score, ship.lives);', '}', '', 'loop(update, draw);'],
      done: "The shooter is running on :5173. Click it, then arrows to move and space to fire.",
      prompt2: "add a starfield and make the enemies explode",
      diff: [["add", 'const stars = makeStars(90);'], ["add", 'function drawStars(dt) { for (const s of stars) s.y += s.z * dt; }'], ["add", 'function explode(x, y, color) {'], ["add", '  for (let i = 0; i < 18; i++) sparks.push(spark(x, y, color));'], ["add", '}']],
      done2: "Done. Stars drift past, and every hit bursts into sparks.",
      aria: "The space shooter the agent wrote, playing itself. Focus it, move with the arrow keys and fire with space; Escape hands it back."
    },
    snake: {
      dir: "snake", title: "Snake", ask: "a snake game", prompt: "build me a snake game",
      src: ['const COLS = 16, ROWS = 18, TICK = 95;', 'let snake = [[5, 9], [4, 9], [3, 9]];', 'let dir = [1, 0], food = place(), score = 0;', '',
        'function step() {', '  const [hx, hy] = snake[0];', '  const head = [hx + dir[0], hy + dir[1]];', '  if (hitsWall(head) || hitsSelf(head)) return reset();', '  snake.unshift(head);', '  if (same(head, food)) { score += 10; food = place(); }', '  else snake.pop();', '}', '',
        'function draw() {', '  ctx.fillStyle = "#0E1430"; ctx.fillRect(0, 0, W, H);', '  ctx.fillStyle = "#6CC5A1"; dot(food);', '  snake.forEach((p, i) => {', '    ctx.fillStyle = "#F08A6C";', '    cell(p);', '  });', '}', '',
        'addEventListener("keydown", e => turn(e.key));', 'setInterval(() => { step(); draw(); }, TICK);'],
      done: "Snake is running on :5173. Click the preview and steer with the arrow keys.",
      prompt2: "make the snake a blue gradient and speed up as it grows",
      diff: [["del", '    ctx.fillStyle = "#F08A6C";'], ["add", '    ctx.fillStyle = mix("#2A45E0", "#A5B4FC", i / snake.length);'], ["del", 'setInterval(() => { step(); draw(); }, TICK);'], ["add", 'const tick = () => Math.max(55, 100 - snake.length);'], ["add", 'loop(() => { step(); draw(); }, tick);']],
      done2: "Done. The snake fades from blue to lavender and speeds up as it eats.",
      aria: "The Snake game the agent wrote, playing itself. Focus it and steer with the arrow keys; Escape hands it back."
    },
    breakout: {
      dir: "breakout", title: "Breakout", ask: "a breakout game", prompt: "build me a breakout game",
      src: ['const ROWS = 6, COLS = 9;', 'const COLORS = ["#B9C6FF", "#7C8FFF", "#2A45E0", "#1C33B8", "#2A45E0", "#7C8FFF"];', 'let bricks = grid(ROWS, COLS), paddle = 0.5;', 'let ball = { x: 0.5, y: 0.7, vx: 0.32, vy: -0.62 };', 'let score = 0, lives = 3;', '',
        'function update(dt) {', '  ball.x += ball.vx * dt; ball.y += ball.vy * dt;', '  if (ball.x < 0 || ball.x > 1) ball.vx *= -1;', '  if (ball.y < 0.1) ball.vy *= -1;', '  if (onPaddle(ball)) bounce(ball, paddle);', '  const b = bricks.find(b => b.alive && inside(ball, b));', '  if (b) { b.alive = false; ball.vy *= -1; score += 10; }', '  if (ball.y > 1) { lives--; serve(); }', '}', '',
        'function draw() {', '  bricks.filter(b => b.alive).forEach(drawBrick);', '  drawPaddle(paddle);', '  drawBall(ball);', '}', '',
        'addEventListener("pointermove", e => paddle = e.clientX / innerWidth);', 'loop(update, draw);'],
      done: "Breakout is running on :5173. Click the preview, then arrows or the pointer to play.",
      prompt2: "add a glowing trail to the ball and sparks when bricks break",
      diff: [["add", 'trail.push([ball.x, ball.y]); if (trail.length > 12) trail.shift();'], ["add", 'ctx.shadowColor = "#6E86FF"; ctx.shadowBlur = 16;'], ["add", 'function shatter(b) {'], ["add", '  burst(b.x, b.y, COLORS[b.row], 14);'], ["add", '}']],
      done2: "Done. The ball leaves a glowing trail, and bricks shatter into sparks.",
      aria: "The Breakout game the agent wrote, playing itself. Focus it and use the arrow keys or the pointer to take over; Escape hands it back."
    }
  };
  Object.values(APPS).forEach(a => { a.files = { "index.html": html(a.title), "style.css": css("#0B0F1F"), "game.js": a.src }; });

  /* ---------- Chat | Code ---------- */
  const mChat = $("mChat"), mCode = $("mCode"), modes = mChat.parentNode, win = $("win");
  const mode = m => {
    if (demo.dataset.mode === m) return;
    demo.dataset.mode = m;
    [[mChat, "chat"], [mCode, "code"]].forEach(([b, k]) => { b.setAttribute("aria-selected", k === m); b.tabIndex = k === m ? 0 : -1; });
    win.setAttribute("aria-labelledby", m === "code" ? "mCode" : "mChat");
  };

  /* ---------- the timeline's shape (timeline seconds; played at SPEED) ---------- */
  // every step lasts at least 2.5 s on screen (3.4 timeline seconds at SPEED); step 5 about 5 s
  const S1 = 3.4, S2 = S1 + 3.4;
  const HOV1 = S2 + .6, HOV2 = S2 + .95, SEL = S2 + 1.3, TOAST = S2 + 1.7, JOIN3 = S2 + 1.95, TOAST_OFF = S2 + 3.05, DLH = S2 + 3.25, DLP = S2 + 3.6;
  const S3 = S2 + 3.85, FILL0 = S3 + .55, CACHE1 = S3 + 1.25, FILL1 = S3 + 3.05, ETA0 = S3 + 1.2;
  const CH = S3 + 3.75, A0 = CH + 1.25;
  let ANSWER = $("a1").textContent, WORDS = ANSWER.split(" ");
  const DUR = i => [1.5, .75, .45, .3][i] || .08;    // each word's trip; the first slow enough to follow
  let WT = [], A1 = 0, C = 0, STEPS = [], END = 0;
  const K = 1.65;                                      // step 6 (the ask) runs this much longer than steps 7, 8 assume
  const HOLD = 4;                                      // after the last step, before the next loop
  function timeWords() {
    WT = [A0]; WORDS.forEach((_, i) => WT.push(WT[i] + DUR(i)));
    A1 = WT[WORDS.length];
    C = Math.ceil((A1 + .6) * 10) / 10;
    STEPS = [0, S1, S2, S3, CH, C, C + K + 1.75, C + K + 7.05];
    END = C + K + 12.4;
  }
  timeWords();
  const c = x => C + x;                               // step 6
  const d = x => C + K + x;                           // steps 7, 8

  /* ---------- the caption bar ---------- */
  const dotBtns = [...demo.querySelectorAll(".sb-dots button")];
  const CAPS = dotBtns.map(b => b.querySelector(".lbl").textContent);
  const HOOKCAP = "An agent built this, on a big open model (35B) pooled across 3 devices. Here's how.";
  const sbN = $("sbN"), sbT = $("sbT");
  let shownStep = -2;
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
  const paintHook = () => {
    shownStep = -1; sbN.textContent = "Demo"; sbT.textContent = HOOKCAP; if (!RM) restart(sbT, "in");
    dotBtns.forEach(b => { b.classList.remove("done", "cur"); b.removeAttribute("aria-current"); });
  };
  const labelDots = () => dotBtns.forEach((b, i) => b.setAttribute("aria-label", `Step ${i + 1} of ${STEPS.length}: ${CAPS[i]}`));

  /* ---------- 1, 2: the room forms ---------- */
  const CODE = "K7QX";
  const tabA = $("tabA"), tabB = $("tabB"), aGo = $("aGo"), bBtn = $("bBtn"), nmA = $("nmA"), nmB = $("nmB");
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
  const GB = [0, 12, 24, 32];
  const setDevices = (n, animate) => {
    chips.forEach((ch, i) => { ch.classList.toggle("out", i >= n); if (animate && i === n - 1) restart(ch, "in"); });
    gbSum.textContent = GB[n] + " GB";
    if (animate) restart(hdSum, "bump");
    pkRoom(GB[Math.max(2, n)], animate);
  };

  /* ---------- 3: pick a model; a friend joins ---------- */
  const model = $("model"), lane = $("lane"), mdS = $("mdS"), dlBtn = $("dlBtn"), toast = $("toast");
  const pkRows = [...$("pkList").children], pkGB = $("pkGB");
  const pkRoom = (gb, animate) => {
    pkGB.textContent = gb + " GB"; if (animate) restart(pkGB.parentNode, "bump");
    pkRows.forEach(li => li.querySelector(".pk-bar i").style.setProperty("--w", (100 * li.dataset.need / gb).toFixed(1) + "%"));
  };
  const pkHover = k => pkRows.forEach((li, i) => li.classList.toggle("hov", i === k));
  const pkSelect = k => pkRows.forEach((li, i) => li.classList.toggle("sel", i === k));

  /* ---------- 4: each device downloads only its own layers ---------- */
  const segs = [0, 1, 2].map(k => [...$("cells" + k).children]);
  const cells = segs.flat();
  const hg = [$("hg0"), $("hg1"), $("hg2")];
  const SEG_GB = [8.4, 8.4, 5.6];
  let filled = [-1, -1, -1], mdKey = "";
  const fillSeg = (k, n) => {
    if (n === filled[k]) return;
    segs[k].forEach((cel, i) => {
      const on = i < n;
      if (on !== cel.classList.contains("f")) { cel.classList.toggle("f", on); if (on && !RM) restart(cel, "f"); }
    });
    filled[k] = n;
    const all = segs[k].length, gb = (SEG_GB[k] * n / all).toFixed(1);
    const done = n >= all;
    hg[k].classList.toggle("ok", done);
    // .hx and .of drop out on phones, where "3.9/5.6 GB" replaces "3.9 of 5.6 GB"
    if (k === 0) hg[k].innerHTML = done ? `${SEG_GB[0]} GB<span class="hx">, <span class="cache">from cache</span></span>` : n > 0 ? `<span class="hx"><span class="cache">loading from cache</span> · </span>${gb} GB` : `${SEG_GB[0]} GB`;
    else hg[k].innerHTML = done ? `${SEG_GB[k]} GB<span class="hx">, ready</span>` : n > 0 ? `${gb}<span class="of"> of </span><span class="sl">/</span>${SEG_GB[k]} GB` : `${SEG_GB[k]} GB`;
  };
  const setMd = (txt, eta) => {
    const key = txt + "|" + (eta || "");
    if (key === mdKey) return;
    const fresh = eta && !mdKey.includes("left");
    mdKey = key;
    // on phones the "Downloading · " prefix drops out and "16.9 of 22.5 GB" becomes "16.9/22.5 GB"
    const m = /^Downloading · ([\d.]+) of (.*)$/.exec(txt);
    if (m) mdS.innerHTML = `<span class="hx">Downloading · </span>${m[1]}<span class="of"> of </span><span class="sl">/</span>${m[2]}`;
    else mdS.textContent = txt;
    if (eta) { const e = document.createElement("span"); e.className = "eta" + (fresh && !RM ? " in" : ""); e.textContent = " · " + eta; mdS.append(e); }
  };
  const loadAt = t => {
    if (t < FILL0) { [0, 1, 2].forEach(k => fillSeg(k, 0)); setMd("22.5 GB · 40 layers · split 3 ways by memory"); return; }
    const p0 = clamp01((t - FILL0) / (CACHE1 - FILL0)), p = clamp01((t - FILL0) / (FILL1 - FILL0));
    fillSeg(0, Math.ceil(15 * p0)); fillSeg(1, Math.ceil(15 * p)); fillSeg(2, Math.ceil(10 * clamp01(p * 1.08)));
    if (p >= 1) { setMd("Ready on 3 devices · 40 layers, split by memory"); return; }
    const gb = (8.4 * p0 + 8.4 * p + 5.6 * clamp01(p * 1.08)).toFixed(1);
    // what is left, at about 50 MB/s: a believable home connection, not the demo's own pace
    const left = (8.4 + 5.6) * (1 - clamp01((t - ETA0) / (FILL1 - ETA0)) * .75) * .85, mins = Math.round(left / .05 / 60);
    setMd(`Downloading · ${gb} of 22.5 GB`, t < ETA0 ? "estimating time" : mins >= 1 ? `about ${mins} min left` : "under a minute left");
  };

  /* ---------- 5: a hidden state, through every layer, once per word ---------- */
  const pkt = $("pkt"), retPath = $("retPath"), wires = [...lane.querySelectorAll(".lk-w i")], a1 = $("a1");
  let geo = null;
  const measure = () => {
    const L = lane.getBoundingClientRect(), k = (L.width / lane.offsetWidth) || 1;
    const r = cel => { const b = cel.getBoundingClientRect(); return { x: (b.left - L.left + b.width / 2) / k, y: (b.top - L.top + b.height / 2) / k, b: (b.bottom - L.top) / k }; };
    const ends = segs.map(s => [r(s[0]), r(s[s.length - 1])]);
    const by = ends[0][0].b + 8, dip = 12;
    geo = { ends, by, dip };
    const f0 = ends[0][0], lz = ends[2][1];
    retPath.setAttribute("d", `M${lz.x.toFixed(1)} ${by.toFixed(1)} Q${((f0.x + lz.x) / 2).toFixed(1)} ${(by + dip * 2).toFixed(1)} ${f0.x.toFixed(1)} ${by.toFixed(1)}`);
  };
  // the trip, in legs weighted by how far they go
  const LEGS = [["seg", 0, 15], ["hop", 0, 3], ["seg", 1, 15], ["hop", 1, 3], ["seg", 2, 10], ["ret", 0, 6]];
  const LW = LEGS.reduce((a, l) => a + l[2], 0);
  let hotI = -1;
  const setHot = i => { if (i === hotI) return; if (hotI >= 0) cells[hotI].classList.remove("hot"); if (i >= 0) cells[i].classList.add("hot"); hotI = i; };
  const tok = d => chips.forEach((ch, j) => ch.classList.toggle("tok", j === d));
  const trip = u => {
    if (!geo) measure();
    const { ends, by, dip } = geo;
    let acc = 0, leg = LEGS[0], k = 0;
    for (const l of LEGS) { if (u * LW < acc + l[2]) { leg = l; k = (u * LW - acc) / l[2]; break; } acc += l[2]; leg = l; k = 1; }
    const [type, s] = leg, y = ends[0][0].y;
    if (type === "seg") { const [a, b] = ends[s], n = segs[s].length, base = s === 0 ? 0 : s === 1 ? 15 : 30; return { x: a.x + (b.x - a.x) * k, y, i: base + Math.min(n - 1, Math.floor(k * n)), d: s }; }
    if (type === "hop") { const a = ends[s][1], b = ends[s + 1][0]; return { x: a.x + (b.x - a.x) * k, y, i: -1, d: -1, hop: k, w: s }; }
    const a = ends[2][1], b = ends[0][0], cx = (a.x + b.x) / 2, cy = by + dip * 2;
    return { x: (1 - k) * (1 - k) * a.x + 2 * (1 - k) * k * cx + k * k * b.x, y: (1 - k) * (1 - k) * by + 2 * (1 - k) * k * cy + k * k * by, i: -1, d: -1, back: true };
  };
  let words = -1, flowing = false;
  const clearWires = () => wires.forEach(w => { w.style.transform = ""; });
  const flow = t => {
    if (t < A0 || t >= A1) {
      if (flowing) { flowing = false; pkt.classList.remove("on", "back"); lane.classList.remove("hop"); clearWires(); setHot(-1); tok(-1); a1.classList.remove("cursor"); }
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
    wires.forEach((wi, j) => { wi.style.transform = p.hop != null && p.w === j ? `scaleX(${p.hop.toFixed(3)})` : p.hop != null && j < p.w ? "scaleX(1)" : ""; });
    setHot(p.i); tok(p.d);
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

  /* ---------- 6 to 8: Code ---------- */
  const log = $("log"), codeTyped = $("codeTyped"), codeComposer = $("codeComposer");
  const files = demo.querySelector(".files"), ftree = $("ftree");
  const fItems = Object.fromEntries([...ftree.children].map(li => [li.dataset.f, li]));
  const lv = $("lv"), lvPre = $("lvPre"), lvF = $("lvF"), lvN = $("lvN");
  const app = $("app"), brLoad = $("brLoad"), game = $("game"), canvas = $("appc"), pvHint = $("pvHint");
  const at = k => demo.querySelector(`[data-at="${k}"]`);
  const saysText = new Map();
  let APP = "tetris", A = APPS.tetris, inst = null, lineOut = {};
  // what a real app of each kind runs to; the live pane shows a few of those lines, the counts are the real size
  const LINES = { tetris: 213, "2048": 148, shooter: 236, snake: 122, breakout: 184 };
  const useApp = name => {
    APP = name; A = APPS[name];
    $("fDir").textContent = A.dir;
    lineOut = { "index.html": 24, "style.css": 60, "game.js": LINES[name] };
    const add = A.diff.filter(d => d[0] === "add").length, del = A.diff.length - add;
    lineOut.edit = lineOut["game.js"] + add - del;
    at("c-q").querySelector("p").textContent = A.prompt;
    at("c-q2").querySelector("p").textContent = A.prompt2;
    saysText.set(at("c-s1"), "I'll write three files and serve them on :5173.");
    saysText.set(at("c-s2"), A.done); saysText.set(at("c-s3"), A.done2);
    const t4 = at("c-t4");
    t4.querySelector(".add").textContent = "+" + add; t4.querySelector(".del").textContent = "-" + del; t4.querySelector(".del").hidden = !del;
    const diff = $("diff"); diff.textContent = "";
    A.diff.forEach(([k, txt]) => { const sp = document.createElement("span"); sp.className = "r " + k; sp.textContent = txt; diff.append(sp); });
    game.setAttribute("aria-label", A.aria);
    CAPS[5] = `Switch to Code and ask for ${A.ask}`; labelDots();
    if (inst) inst.destroy();
    inst = window.PooledApps.make(name, canvas, { seed: 11 });
  };
  const logBottom = () => { log.scrollTop = log.scrollHeight; };
  let liveSrc = [], liveShown = -1, writing = null;
  const liveTo = k => {
    if (k === liveShown) return;
    if (k < liveShown || liveShown < 0) { lvPre.textContent = ""; liveShown = 0; }
    for (let i = liveShown; i < k; i++) { const sp = document.createElement("span"); sp.textContent = liveSrc[i] || " "; lvPre.append(sp); }
    while (lvPre.children.length > 6) lvPre.firstChild.remove();
    liveShown = k;
    const n = writing && liveSrc.length ? Math.round(writing[5] * k / liveSrc.length) : k;
    lvN.textContent = `${n} line${n === 1 ? "" : "s"}`;
  };
  const fileState = (name, st, lines) => {
    const li = fItems[name];
    if (st === "hide") { li.classList.add("pending"); li.classList.remove("w", "mod"); return; }
    if (li.classList.contains("pending")) { li.classList.remove("pending"); if (!RM) restart(li, "new"); }
    li.classList.toggle("w", st === "w"); if (st === "mod") li.classList.add("mod");
    if (lines != null) li.querySelector(".fm").textContent = lines;
    files.classList.toggle("none", ftree.querySelectorAll("li:not(.pending)").length === 0);
  };
  const liveStart = (name, key, t0, t1) => {
    liveSrc = key === "edit" ? A.diff.filter(d => d[0] === "add").map(d => d[1]) : A.files[key];
    lvF.textContent = name; liveShown = -1; liveTo(0);
    lv.classList.remove("pending"); if (!RM) restart(lv, "enter");
    writing = [t0, t1, liveSrc.length, name, key, key === "edit" ? liveSrc.length : lineOut[key]];
    fileState(name, "w", key === "edit" ? null : 0);
    lv.parentNode.append(lv); logBottom();
  };
  const liveEnd = () => { const w = writing; lv.classList.add("pending"); writing = null; tok(-1); if (w) fileState(w[3], "done", lineOut[w[4]]); };
  const tool = k => { const el = show(k); if (el) { log.append(el); logBottom(); } return el; };
  let streams = [];
  const reveal = (el, rate) => streams.push({ el: el.querySelector(".say"), text: saysText.get(el), t0: tl.t, rate });
  const setRun = (k, run) => { const st = at(k).querySelector(".st"); st.textContent = run ? "running" : "done"; st.classList.toggle("run", run); };
  const openPreview = () => { flag("served", true); flag("app-on", true); };
  const reload = () => { if (!RM) restart(brLoad, "go"); };
  const WARM_S = { tetris: 16, "2048": 40, shooter: 7, snake: 22, breakout: 9 };
  const runGame = (warm, v2) => { if (!inst.human) { inst.auto(); inst.reset(11); inst.v2(v2); inst.warm(WARM_S[APP] * warm); } else inst.v2(v2); if (!RM) inst.start(); else inst.draw(); };

  const tl = { t: 0, fired: 0, done: false, started: false };
  const EVENTS = () => [
    // 1: the laptop has a name already; it starts a room
    [.3, () => nmA.classList.add("fresh")],
    [1.3, () => nmA.classList.remove("fresh")],
    [1.55, () => press(aGo, 260)],
    [1.8, () => { tabA.classList.add("done"); setDevices(1, true); }],
    ...[0, 1, 2, 3].map(i => [1.9 + i * .12, () => letters(i + 1)]),
    // 2: the desktop (named too) wakes, types the code and joins; the two tabs fold into one room
    [S1, () => tabB.classList.remove("idle")],
    [S1 + .15, () => nmB.classList.add("fresh")],
    [S1 + .75, () => nmB.classList.remove("fresh")],
    [S1 + .8, () => setSlots(0, 0)],
    [S1 + .98, () => setSlots(1, 1)], [S1 + 1.16, () => setSlots(2, 2)], [S1 + 1.34, () => setSlots(3, 3)],
    [S1 + 1.52, () => { setSlots(4, -1); bBtn.classList.add("ready"); }],
    [S1 + 1.8, () => press(bBtn, 260)],
    [S1 + 2.05, () => { tabB.classList.add("done"); tabA.classList.add("met"); setDevices(2, true); }],
    [S1 + 2.6, () => flag("merge", true)],
    // 3: pick a model; a friend's laptop joins; Download
    [S2, () => scene("split")],
    [HOV1, () => pkHover(1)], [HOV2, () => pkHover(2)],
    [SEL, () => { pkHover(-1); pkSelect(2); }],
    [TOAST, () => { if (!RM) toast.classList.add("on"); }],
    [JOIN3, () => setDevices(3, true)],
    [TOAST_OFF, () => toast.classList.remove("on")],
    [DLH, () => dlBtn.classList.add("hover")],
    [DLP, () => { dlBtn.classList.remove("hover"); press(dlBtn, 240); }],
    // 4: each device fetches only its own layers
    [S3, () => model.classList.add("dl")],
    [S3 + .2, () => model.classList.add("split")],
    // 5: chat
    [CH, () => scene("chat")],
    [CH + .45, () => { measure(); chatComposer.classList.add("hot"); }],
    [CH + 1.2, () => { chatTyped.textContent = ""; chatComposer.classList.remove("hot"); show("q1"); show("a1"); a1.textContent = ""; toBottom(); }],
    // 6: Code: the tab switches itself
    [C, () => { press(modes, 300); mode("code"); }],
    [c(.2), () => scene("code")],
    [c(.6), () => codeComposer.classList.add("hot")],
    [c(2.6), () => { codeTyped.textContent = ""; codeComposer.classList.remove("hot"); tool("c-q"); }],
    [c(2.85), () => reveal(tool("c-s1"), .035)],
    // 7: it writes three files and serves them
    [d(1.75), () => liveStart("index.html", "index.html", d(1.75), d(2.2))],
    [d(2.3), () => { liveEnd(); tool("c-t0"); }],
    [d(2.4), () => liveStart("style.css", "style.css", d(2.4), d(2.75))],
    [d(2.85), () => { liveEnd(); tool("c-t1"); }],
    [d(2.95), () => liveStart("game.js", "game.js", d(2.95), d(4.5))],
    [d(4.6), () => { liveEnd(); tool("c-t2"); }],
    [d(4.75), () => { setRun("c-t3", true); tool("c-t3"); }],
    [d(5.1), () => { setRun("c-t3", false); openPreview(); }],
    [d(5.3), reload],
    [d(5.45), () => { app.classList.remove("blank"); runGame(.6, false); }],
    [d(5.7), () => reveal(tool("c-s2"), .035)],
    // 8: a change, an edit, a reload
    [d(6.9), () => flag("app-on", false)],
    [d(7.05), () => codeComposer.classList.add("hot")],
    [d(8.25), () => { codeTyped.textContent = ""; codeComposer.classList.remove("hot"); tool("c-q2"); }],
    [d(8.45), () => liveStart("game.js", "edit", d(8.45), d(8.9))],
    [d(9.0), () => { liveEnd(); fileState("game.js", "mod"); tool("c-t4"); }],
    [d(9.25), () => { flag("app-on", true); reload(); app.classList.add("blank"); }],
    [d(9.5), () => { inst.v2(true); app.classList.remove("blank"); }],
    [d(9.75), () => reveal(tool("c-s3"), .035)],
  ].sort((a, b) => a[0] - b[0]);
  let EV = EVENTS();

  const frame = t => {
    if (t >= S3) loadAt(t);
    if (t > CH + .5 && t < CH + 1.2) typeInto(chatTyped, Q1, CH + .55, CH + 1.05, t);
    if (t >= CH) flow(t);
    if (t > c(.6) && t < c(2.6)) typeInto(codeTyped, A.prompt, c(.8), c(2.3), t);
    if (t > d(7.05) && t < d(8.25)) typeInto(codeTyped, A.prompt2, d(7.15), d(8.05), t);
    if (writing) {
      const [a, b, n] = writing;
      liveTo(Math.max(0, Math.min(n, Math.ceil((t - a) / (b - a) * n))));
      fileState(writing[3], "w", writing[4] === "edit" ? null : Math.round(writing[5] * liveShown / Math.max(1, writing[2])));
      tok(Math.floor(t * 7) % 3);
      logBottom();
    }
    streams = streams.filter(s => {
      const w = s.text.split(" "), n = Math.min(w.length, Math.floor((t - s.t0) / s.rate) + 1);
      s.el.textContent = w.slice(0, n).join(" "); s.el.classList.toggle("cursor", n < w.length);
      logBottom();
      return n < w.length;
    });
  };

  const LOGKEYS = ["c-q", "c-s1", "c-t0", "c-t1", "c-t2", "c-t3", "c-s2", "c-q2", "c-t4", "c-s3"];
  tl.reset = () => {
    tl.t = 0; tl.fired = 0; tl.done = false; streams = []; writing = null;
    flag("hook", false);
    scene("tabs"); mode("chat"); ["merge", "served", "app-on"].forEach(k => flag(k, false));
    tabA.classList.remove("done", "met"); tabB.classList.remove("done"); tabB.classList.add("idle"); letters(0);
    nmA.classList.remove("fresh"); nmB.classList.remove("fresh");
    setSlots(0, -1); bBtn.classList.remove("ready", "press"); aGo.classList.remove("press");
    setDevices(0); chips.forEach(ch => ch.classList.remove("in")); pkRoom(24);
    pkHover(-1); pkSelect(-1); toast.classList.remove("on");
    model.classList.remove("split", "dl"); dlBtn.classList.remove("hover", "press"); filled = [-1, -1, -1]; mdKey = ""; loadAt(0); geo = null;
    words = -1; flowing = true; flow(0);
    demo.querySelectorAll("[data-at]").forEach(el => el.classList.add("pending"));
    chatTyped.textContent = ""; chatComposer.classList.remove("hot");
    codeTyped.textContent = ""; codeComposer.classList.remove("hot");
    Object.keys(fItems).forEach(n => fileState(n, "hide")); files.classList.add("none");
    lv.classList.add("pending"); liveShown = -1; liveSrc = []; liveTo(0);
    saysText.forEach((txt, el) => { const s = el.querySelector(".say"); if (s) { s.textContent = txt; s.classList.remove("cursor"); } });
    LOGKEYS.forEach(k => log.append(at(k)));
    app.classList.add("blank"); brLoad.classList.remove("go");
    if (!inst.human) { inst.stop(); inst.v2(false); }
    msgs.scrollTop = 0; log.scrollTop = 0;
    paintBar(0, true);
  };
  tl.final = () => {
    streams = []; writing = null; tl.done = true; tl.t = END; tl.fired = EV.length;
    scene("code"); mode("code"); flag("merge", true); flag("served", true); flag("app-on", true);
    tabA.classList.add("done", "met"); tabB.classList.add("done"); tabB.classList.remove("idle"); letters(4); setSlots(4, -1);
    setDevices(3); pkSelect(2); pkHover(-1); toast.classList.remove("on");
    model.classList.add("dl", "split"); loadAt(FILL1 + 1);
    words = -1; flowing = true; flow(END);
    demo.querySelectorAll("[data-at]").forEach(el => el.classList.remove("pending", "enter"));
    saysText.forEach((txt, el) => { const s = el.querySelector(".say"); if (s) { s.textContent = txt; s.classList.remove("cursor"); } });
    chatTyped.textContent = ""; codeTyped.textContent = ""; chatComposer.classList.remove("hot"); codeComposer.classList.remove("hot");
    Object.keys(fItems).forEach(n => fileState(n, "done", lineOut[n])); fileState("game.js", "mod", lineOut.edit);
    lv.classList.add("pending"); tok(-1);
    app.classList.remove("blank"); runGame(1, true);
    toBottom(); logBottom();
    paintBar(END);
  };
  tl.advance = dt => {
    tl.t += dt;
    while (tl.fired < EV.length && EV[tl.fired][0] <= tl.t) { EV[tl.fired][1](); tl.fired++; }
    frame(tl.t);
    paintBar(Math.min(tl.t, END), true);
    if (tl.t >= END + HOLD && tl.fired >= EV.length) nextLoop();
  };
  // a new loop: new names, a different app
  const nextLoop = () => {
    newNames(); EV = EVENTS();
    useApp(pickOne(Object.keys(APPS).filter(n => n !== APP)));
    tl.reset();
  };

  /* ---------- driver: one rAF, only while the window is on screen and the page is visible ---------- */
  let raf = 0, last = 0, visible = false, frozen = false, hook = 0;
  const needs = () => !frozen && visible && !document.hidden && tl.started && (hook > 0 || !tl.done);
  function loop(now) {
    const dt = last ? Math.min(.1, (now - last) / 1000) : .016; last = now;
    if (hook > 0) { hook -= dt; if (hook <= 0) endHook(); }
    else tl.advance(dt * SPEED);
    raf = needs() ? requestAnimationFrame(loop) : 0;
  }
  const wake = () => { if (needs() && !raf) { last = 0; raf = requestAnimationFrame(loop); } };
  const halt = () => { if (raf) { cancelAnimationFrame(raf); raf = 0; } };
  const endHook = () => { hook = 0; nextLoop(); };
  const begin = () => {
    if (tl.started) { wake(); return; }
    tl.started = true;
    if (RM) { tl.final(); return; }
    // the hook: the finished app, playing, big; then the story from the room
    tl.final(); tl.done = false; flag("hook", true); paintHook(); hook = 2.6;
    wake();
  };

  // jump to a moment; with reduced motion, to a step's finished state, frozen
  const seek = (s, freeze) => {
    halt(); tl.started = true; frozen = !!freeze; hook = 0;
    if (inst.human) { inst.auto(); game.blur(); }
    tl.reset();
    while (tl.t < s - 1e-6) tl.advance(Math.min(1 / 30, s - tl.t));
    inst.draw(); wake();
  };
  const goStep = k => {
    if (RM) {
      if (k >= STEPS.length - 1) { tl.final(); return; }
      const ends = [S1 - .3, S2 - .3, DLP + .1, CH - .3, A1 + .5, c(3.2), d(6.6)];
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
  addEventListener("resize", () => { geo = null; if (demo.dataset.scene === "chat") measure(); });

  /* ---------- the app: it plays itself; click in and the keys take over ---------- */
  const hint = on => { pvHint.textContent = on ? "Esc to stop" : "Click to play"; pvHint.classList.toggle("on", on); };
  game.addEventListener("keydown", e => {
    if (e.key === "Escape") { game.blur(); return; }
    if (inst.key(e, true)) e.preventDefault();
  });
  game.addEventListener("keyup", e => { inst.key(e, false); });
  game.addEventListener("pointerdown", e => { if (e.pointerType !== "mouse" && APP !== "breakout") return; game.focus({ preventScroll: true }); if (!inst.human) inst.play(); inst.start(); });
  game.addEventListener("pointermove", e => { if (document.activeElement !== game || !inst.human) return; const r = game.getBoundingClientRect(); inst.pointer((e.clientX - r.left) / r.width); });
  game.addEventListener("focus", () => hint(true));
  game.addEventListener("blur", () => { hint(false); if (inst.human) { inst.auto(); inst.reset(11); inst.warm(WARM_S[APP]); if (!RM) inst.start(); } });

  /* ---------- start when 30% visible ---------- */
  new IntersectionObserver(es => {
    visible = es[0].isIntersecting;
    if (visible) begin();
    wake();
  }, { threshold: .3 }).observe(win);
  document.addEventListener("visibilitychange", wake);

  useApp("tetris"); labelDots();
  if (RM) tl.final(); else { tl.reset(); tl.final(); }

  // tests and screenshots: jump to a moment, pick an app
  window.__demo = {
    seek(s, freeze) { seek(s, freeze); },
    step: goStep,
    setApp(n) { useApp(n); seek(0); },
    set frozen(v) { frozen = v; wake(); }, get t() { return tl.t; }, get hook() { return hook; }, get app() { return APP; },
    get STEPS() { return STEPS; }, get END() { return END; }, get C() { return C; }, get A1() { return A1; }, APPS: Object.keys(APPS)
  };
})();
