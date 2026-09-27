/* Tetris for the Code demo (the game the agent "wrote"). Plays itself until the visitor takes over.
   PooledTetris(canvas, { seed, tick, onScore, onState }) */
(() => {
  "use strict";
  const COLS = 10, ROWS = 20;
  const SHAPES = {
    I: [[0, 1], [1, 1], [2, 1], [3, 1]], O: [[1, 0], [2, 0], [1, 1], [2, 1]],
    T: [[1, 0], [0, 1], [1, 1], [2, 1]], S: [[1, 0], [2, 0], [0, 1], [1, 1]],
    Z: [[0, 0], [1, 0], [1, 1], [2, 1]], J: [[0, 0], [0, 1], [1, 1], [2, 1]],
    L: [[2, 0], [0, 1], [1, 1], [2, 1]]
  };
  const KEYS = Object.keys(SHAPES);
  const COLORS = ["#3152FF", "#6E86FF", "#A5B4FC", "#C9D1F7", "#8EA2FF", "#4A5FD0", "#DCE2FF"];
  const rot = (cells, n) => {
    let c = cells;
    for (let k = 0; k < n; k++) c = c.map(([x, y]) => [-y, x]);
    const mx = Math.min(...c.map(p => p[0])), my = Math.min(...c.map(p => p[1]));
    return c.map(([x, y]) => [x - mx, y - my]);
  };
  const ROTS = {};
  KEYS.forEach(k => { ROTS[k] = [0, 1, 2, 3].map(n => rot(SHAPES[k], n)); });
  const UNIQUE = { I: 2, O: 1, S: 2, Z: 2, T: 4, J: 4, L: 4 };
  function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  function create(canvas, opts = {}) {
    const ctx = canvas.getContext("2d");
    const onScore = opts.onScore || (() => {}), onState = opts.onState || (() => {}), TK = opts.tick || 55;
    let rand, bag, board, cur, plan, score, lines, flash, over = false, human = false, paused = false;
    let acc = 0, raf = 0, last = 0, playing = false, visible = true;

    const empty = () => Array.from({ length: ROWS }, () => Array(COLS).fill(0));
    const fits = (cells, px, py, b = board) => cells.every(([x, y]) => { const X = px + x, Y = py + y; return X >= 0 && X < COLS && Y < ROWS && (Y < 0 || !b[Y][X]); });
    const cells = () => ROTS[cur.k][cur.r];

    function next() {
      if (!bag.length) { bag = KEYS.slice(); for (let i = bag.length - 1; i > 0; i--) { const j = (rand() * (i + 1)) | 0; [bag[i], bag[j]] = [bag[j], bag[i]]; } }
      const k = bag.pop();
      cur = { k, r: 0, x: 3, y: -1, c: KEYS.indexOf(k) };
      if (!fits(ROTS[k][0], cur.x, cur.y)) {
        if (human) { over = true; onState("over"); return; }
        board = empty();
      }
      if (!human) plan = think(k);
    }
    function think(k) {
      let best = null;
      for (let r = 0; r < UNIQUE[k]; r++) {
        const cs = ROTS[k][r], w = Math.max(...cs.map(p => p[0])) + 1;
        for (let x = 0; x <= COLS - w; x++) {
          let y = -1; if (!fits(cs, x, y)) continue;
          while (fits(cs, x, y + 1)) y++;
          const b = board.map(row => row.slice());
          cs.forEach(([cx, cy]) => { if (y + cy >= 0) b[y + cy][x + cx] = 1; });
          let cleared = 0; for (let yy = 0; yy < ROWS; yy++) if (b[yy].every(v => v)) cleared++;
          const h = []; let holes = 0;
          for (let cx = 0; cx < COLS; cx++) {
            let top = ROWS; for (let yy = 0; yy < ROWS; yy++) if (b[yy][cx]) { top = yy; break; }
            h.push(ROWS - top);
            for (let yy = top + 1; yy < ROWS; yy++) if (!b[yy][cx]) holes++;
          }
          let bump = 0; for (let cx = 0; cx < COLS - 1; cx++) bump += Math.abs(h[cx] - h[cx + 1]);
          const s = -0.51 * h.reduce((a, v) => a + v, 0) + 0.76 * cleared - 0.36 * holes - 0.18 * bump + rand() * .01;
          if (!best || s > best.s) best = { s, r, x };
        }
      }
      return best || { r: 0, x: 3 };
    }
    function lock() {
      cells().forEach(([x, y]) => { if (cur.y + y >= 0) board[cur.y + y][cur.x + x] = cur.c + 1; });
      const full = []; for (let y = 0; y < ROWS; y++) if (board[y].every(v => v)) full.push(y);
      if (full.length) flash = { rows: full, t: 0 }; else next();
    }
    function flashStep() {
      flash.t++;
      if (flash.t > 4) {
        board = board.filter((_, y) => !flash.rows.includes(y));
        while (board.length < ROWS) board.unshift(Array(COLS).fill(0));
        lines += flash.rows.length; score += [0, 100, 300, 500, 800][flash.rows.length];
        onScore(score, lines); flash = null; next();
      }
    }
    function aiTick() {
      if (flash) return flashStep();
      if (cur.r !== plan.r) { const r = (cur.r + 1) % 4; if (fits(ROTS[cur.k][r], cur.x, cur.y)) { cur.r = r; return; } }
      if (cur.x !== plan.x) { const x = cur.x + Math.sign(plan.x - cur.x); if (fits(cells(), x, cur.y)) { cur.x = x; if (cur.r === plan.r) return; } }
      if (fits(cells(), cur.x, cur.y + 1)) cur.y++; else lock();
    }
    function fall() { if (fits(cells(), cur.x, cur.y + 1)) { cur.y++; return true; } lock(); return false; }
    const gravity = () => Math.max(110, 560 - lines * 22);

    function act(a) {
      if (!human || paused || over || flash || !cur) return;
      if (a === "left" || a === "right") { const x = cur.x + (a === "left" ? -1 : 1); if (fits(cells(), x, cur.y)) cur.x = x; }
      else if (a === "rot") {
        const r = (cur.r + 1) % 4;
        for (const dx of [0, -1, 1, -2, 2]) if (fits(ROTS[cur.k][r], cur.x + dx, cur.y)) { cur.r = r; cur.x += dx; break; }
      }
      else if (a === "down") { if (fall()) { score += 1; onScore(score, lines); } acc = 0; }
      else if (a === "drop") { let n = 0; while (fits(cells(), cur.x, cur.y + 1)) { cur.y++; n++; } score += n * 2; onScore(score, lines); lock(); acc = 0; }
      draw();
    }
    function reset(seed) {
      rand = rng(seed == null ? (opts.seed || 7) : seed); bag = []; board = empty(); score = 0; lines = 0; flash = null; over = false; acc = 0;
      next(); onScore(score, lines);
    }
    function warm(n) { let pieces = 0; while (pieces < n) { const c = cur; aiTick(); if (cur !== c) pieces++; } }

    let W = 0, H = 0, dpr = 1;
    function size() {
      const r = canvas.getBoundingClientRect();
      if (!r.width) return false;
      dpr = Math.min(devicePixelRatio || 1, 2); W = r.width; H = r.height;
      const cw = Math.round(W * dpr), ch = Math.round(H * dpr);
      if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
      return true;
    }
    function draw() {
      if (!size()) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const s = Math.min(W / COLS, H / ROWS), ox = (W - s * COLS) / 2, oy = (H - s * ROWS) / 2;
      ctx.fillStyle = "#0E1328"; ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = "rgba(147,166,255,.09)";
      for (let y = 0; y < ROWS; y++) for (let x = 0; x < COLS; x++) ctx.fillRect(ox + x * s + s / 2 - .75, oy + y * s + s / 2 - .75, 1.5, 1.5);
      const cell = (x, y, col, a = 1) => { ctx.globalAlpha = a; ctx.fillStyle = col; ctx.fillRect(ox + x * s + 1, oy + y * s + 1, s - 2, s - 2); ctx.globalAlpha = 1; };
      for (let y = 0; y < ROWS; y++) for (let x = 0; x < COLS; x++) {
        const v = board[y][x]; if (!v) continue;
        const f = flash && flash.rows.includes(y);
        cell(x, y, f ? "#FFFFFF" : COLORS[v - 1], f ? (flash.t % 2 ? .95 : .55) : 1);
      }
      if (cur && !flash && !over) {
        const cs = cells(); let gy = cur.y; while (fits(cs, cur.x, gy + 1)) gy++;
        ctx.strokeStyle = "rgba(147,166,255,.32)"; ctx.lineWidth = 1;
        cs.forEach(([x, y]) => { if (gy + y >= 0) ctx.strokeRect(ox + (cur.x + x) * s + 1.5, oy + (gy + y) * s + 1.5, s - 3, s - 3); });
        cs.forEach(([x, y]) => { if (cur.y + y >= 0) cell(cur.x + x, cur.y + y, COLORS[cur.c]); });
      }
      if (over) {
        ctx.fillStyle = "rgba(11,15,31,.72)"; ctx.fillRect(0, 0, W, H);
      }
    }
    function frame(now) {
      const dt = last ? Math.min(100, now - last) : 16; last = now;
      acc += dt;
      if (flash) { while (acc >= 55 && flash) { acc -= 55; flashStep(); } }
      else if (!human) { while (acc >= TK) { acc -= TK; aiTick(); } }
      else if (!paused && !over) { const g = gravity(); while (acc >= g && !over && !flash) { acc -= g; fall(); } }
      else acc = 0;
      draw();
      raf = (playing && visible && !document.hidden) ? requestAnimationFrame(frame) : 0;
    }
    const wake = () => { if (playing && visible && !document.hidden && !raf) { last = 0; raf = requestAnimationFrame(frame); } };
    if ("IntersectionObserver" in window) new IntersectionObserver(es => { visible = es[0].isIntersecting; wake(); }).observe(canvas);
    document.addEventListener("visibilitychange", wake);
    addEventListener("resize", () => { if (!raf) draw(); });

    reset();
    return {
      start() { playing = true; wake(); },
      stop() { playing = false; if (raf) cancelAnimationFrame(raf); raf = 0; draw(); },
      reset(seed) { reset(seed); draw(); },
      warm(n) { warm(n); draw(); },
      play(seed) { human = true; paused = false; reset(seed == null ? (Date.now() % 9973) : seed); onState("play"); playing = true; wake(); },
      auto(seed) { human = false; paused = false; reset(seed); onState("auto"); },
      pause(on) { if (!human || over) return; paused = !!on; acc = 0; onState(paused ? "paused" : "play"); draw(); },
      act,
      draw,
      get human() { return human; }, get over() { return over; }, get paused() { return paused; }, get score() { return score; }
    };
  }
  window.PooledTetris = create;
})();
