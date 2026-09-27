/* Switching to Code: a shine sweeps across the tab and a few small stars burst out of it.
   Shared by the landing demo and the room (window.pooledSparkle(el)). Nothing under reduced motion. */
(function () {
  const STAR = "M8 0C8.6 4.6 11.4 7.4 16 8 11.4 8.6 8.6 11.4 8 16 7.4 11.4 4.6 8.6 0 8 4.6 7.4 7.4 4.6 8 0Z";
  const COLORS = ["#2A45E0", "#6E86FF", "#A5B4FC", "#FFFFFF"];
  function sparkle(el) {
    if (!el || !el.animate) return;
    try { if (matchMedia("(prefers-reduced-motion: reduce)").matches) return; } catch (e) { /* no matchMedia: animate */ }
    shine(el);
    // the stars: fixed to the page so the tab's rounded clip does not cut them
    const r = el.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    burst(el, r, cx, cy);
  }
  // the shine: a soft light band crossing the tab, clipped to its shape
  function shine(el) {
    if (!el || !el.animate) return;
    try { if (matchMedia("(prefers-reduced-motion: reduce)").matches) return; } catch (e) { /* animate */ }
    const clip = document.createElement("span");
    clip.setAttribute("aria-hidden", "true");
    clip.style.cssText = "position:absolute;inset:0;border-radius:inherit;overflow:hidden;pointer-events:none;z-index:2";
    const band = document.createElement("span");
    band.style.cssText = "position:absolute;top:0;bottom:0;left:0;width:60%;background:linear-gradient(100deg,transparent,rgba(165,180,252,.55),rgba(255,255,255,.9),rgba(165,180,252,.55),transparent)";
    clip.appendChild(band);
    el.appendChild(clip);
    band.animate([{ transform: "translateX(-120%)" }, { transform: "translateX(190%)" }], { duration: 650, easing: "cubic-bezier(.3,.7,.3,1)" })
      .finished.then(() => clip.remove(), () => clip.remove());
  }
  function burst(el, r, cx, cy) {
    const N = 9;
    for (let i = 0; i < N; i++) {
      const a = (i / N) * Math.PI * 2 + (i % 2 ? .35 : 0), d = 26 + (i % 3) * 12, s = 7 + (i % 3) * 3;
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", "0 0 16 16");
      svg.setAttribute("aria-hidden", "true");
      svg.style.cssText = `position:fixed;left:${cx - s / 2}px;top:${cy - s / 2}px;width:${s}px;height:${s}px;pointer-events:none;z-index:1000;overflow:visible`;
      const color = COLORS[i % COLORS.length];
      svg.innerHTML = `<path d="${STAR}" fill="${color}"${color === "#FFFFFF" ? ' stroke="#6E86FF" stroke-width="1"' : ""}/>`;
      document.body.appendChild(svg);
      const dx = Math.cos(a) * d * (r.width / r.height > 2 ? 1.5 : 1), dy = Math.sin(a) * d;
      svg.animate([
        { transform: "translate(0,0) scale(0) rotate(0deg)", opacity: 0 },
        { transform: `translate(${dx * .6}px,${dy * .6}px) scale(1) rotate(90deg)`, opacity: 1, offset: .45 },
        { transform: `translate(${dx}px,${dy}px) scale(0) rotate(180deg)`, opacity: 0 },
      ], { duration: 720 + (i % 3) * 90, delay: i * 18, easing: "cubic-bezier(.2,.7,.3,1)" })
        .finished.then(() => svg.remove(), () => svg.remove());
    }
  }
  window.pooledSparkle = sparkle;

  // [data-twinkle] tabs: small stars that twinkle inside the tab while it is selected (the page's CSS
  // turns the selected Code pill black). Kept to the edges so they never sit on the word.
  const SPOTS = [[9, 28, 5], [17, 72, 4], [30, 18, 3], [74, 20, 4], [86, 66, 5], [91, 30, 3], [66, 80, 3]];
  const css = document.createElement("style");
  css.textContent = ".pooled-tw{position:absolute;inset:0;border-radius:inherit;overflow:hidden;pointer-events:none;opacity:0;transition:opacity .35s ease}"
    + "[data-twinkle][aria-selected=\"true\"]>.pooled-tw{opacity:1}"
    + ".pooled-tw svg{position:absolute;animation:pooledtw 1.9s ease-in-out infinite}"
    + "@keyframes pooledtw{0%,100%{opacity:0;transform:scale(.3) rotate(0)}50%{opacity:1;transform:scale(1) rotate(45deg)}}"
    // hovering the tab before it is chosen: the stars show in blue on the light tab, as an invitation
    + "[data-twinkle]:not([aria-selected=\"true\"]):hover>.pooled-tw{opacity:1}"
    + "[data-twinkle]:not([aria-selected=\"true\"]) .pooled-tw path{fill:#6E86FF}"
    + "@media (prefers-reduced-motion:reduce){.pooled-tw svg{animation:none;opacity:.8}}";
  function twinkles(el) {
    if (el.querySelector(".pooled-tw")) return;
    const box = document.createElement("span");
    box.className = "pooled-tw";
    box.setAttribute("aria-hidden", "true");
    SPOTS.forEach(([x, y, s], i) => {
      box.insertAdjacentHTML("beforeend", `<svg viewBox="0 0 16 16" style="left:calc(${x}% - ${s / 2}px);top:calc(${y}% - ${s / 2}px);width:${s}px;height:${s}px;animation-delay:${(i * .27).toFixed(2)}s"><path d="${STAR}" fill="${i % 3 ? "#FFFFFF" : "#A5B4FC"}"/></svg>`);
    });
    el.appendChild(box);
    // and a shine across it when the pointer arrives, while it is not chosen yet
    let at = 0;
    el.addEventListener("pointerenter", (e) => {
      if (e.pointerType === "touch" || el.getAttribute("aria-selected") === "true" || performance.now() - at < 900) return;
      at = performance.now(); shine(el);
    });
  }
  const init = () => { document.head.appendChild(css); document.querySelectorAll("[data-twinkle]").forEach(twinkles); };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
