/* Switching to Code: a shine sweeps across the tab and a few small stars burst out of it.
   Shared by the landing demo and the room (window.pooledSparkle(el)). Nothing under reduced motion. */
(function () {
  const STAR = "M8 0C8.6 4.6 11.4 7.4 16 8 11.4 8.6 8.6 11.4 8 16 7.4 11.4 4.6 8.6 0 8 4.6 7.4 7.4 4.6 8 0Z";
  const COLORS = ["#2A45E0", "#6E86FF", "#A5B4FC", "#FFFFFF"];
  function sparkle(el) {
    if (!el || !el.animate) return;
    try { if (matchMedia("(prefers-reduced-motion: reduce)").matches) return; } catch (e) { /* no matchMedia: animate */ }
    // the shine: a soft light band crossing the tab, clipped to its shape
    const clip = document.createElement("span");
    clip.setAttribute("aria-hidden", "true");
    clip.style.cssText = "position:absolute;inset:0;border-radius:inherit;overflow:hidden;pointer-events:none;z-index:2";
    const band = document.createElement("span");
    band.style.cssText = "position:absolute;top:0;bottom:0;left:0;width:60%;background:linear-gradient(100deg,transparent,rgba(165,180,252,.55),rgba(255,255,255,.9),rgba(165,180,252,.55),transparent)";
    clip.appendChild(band);
    el.appendChild(clip);
    band.animate([{ transform: "translateX(-120%)" }, { transform: "translateX(190%)" }], { duration: 650, easing: "cubic-bezier(.3,.7,.3,1)" })
      .finished.then(() => clip.remove(), () => clip.remove());
    // the stars: fixed to the page so the tab's rounded clip does not cut them
    const r = el.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
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
})();
