/* The stage: on first view the demo is a big product shot right under the headline; as the page
   scrolls it settles into its framed size. Transform only (scale on the demo, a matching lift on what
   follows), one rAF per scroll burst, layout read only on load and resize. Off on narrow screens and
   with reduced motion: the demo is simply its final size. */
(() => {
  "use strict";
  const demo = document.getElementById("demo");
  if (!demo || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const hero = document.querySelector(".hero-in");
  const after = [...document.querySelectorAll(".closer, .footer")];
  const clamp01 = x => x < 0 ? 0 : x > 1 ? 1 : x;
  const ease = p => p * p * (3 - 2 * p);                  // smoothstep: soft in, soft out

  let S0 = 1, R = 1, H = 0, cur = -1, raf = 0;
  const pageTop = el => { let y = 0; for (; el; el = el.offsetParent) y += el.offsetTop; return y; };

  const measure = () => {                                 // layout sizes; transforms don't touch these
    const vw = document.documentElement.clientWidth, vh = innerHeight;
    const W = demo.offsetWidth; H = demo.offsetHeight;
    const top = pageTop(demo);
    // big: nearly the full width, never wider than the viewport (no horizontal scroll)
    S0 = vw < 760 ? 1 : Math.max(1, Math.min(1.42, (vw - Math.max(32, vw * .045)) / W));
    // settled once the framed demo would sit comfortably in view
    R = Math.max(220, Math.min(top - 20, vh * .62));
    cur = -1;
  };

  const apply = () => {
    raf = 0;
    const e = S0 > 1 ? ease(clamp01(scrollY / R)) : 1;
    const s = S0 + (1 - S0) * e;
    if (Math.abs(s - cur) < 1e-4) return;
    cur = s;
    const big = s > 1.0005;
    demo.style.transform = big ? `scale(${s.toFixed(4)})` : "";
    demo.classList.toggle("big", s > 1 + (S0 - 1) * .5);
    const dy = big ? ((s - 1) * H).toFixed(1) : 0;
    after.forEach(el => { el.style.transform = big ? `translate3d(0,${dy}px,0)` : ""; });
    if (hero) {
      const f = S0 > 1 ? e : 0;
      hero.style.opacity = f ? (1 - f * .55).toFixed(3) : "";
      hero.style.transform = f ? `translate3d(0,${(-f * 18).toFixed(1)}px,0)` : "";
    }
  };
  const tick = () => { if (!raf) raf = requestAnimationFrame(apply); };

  measure(); apply();
  addEventListener("scroll", tick, { passive: true });
  addEventListener("resize", () => { measure(); tick(); });
  addEventListener("load", () => { measure(); tick(); });
  if (document.fonts) document.fonts.ready.then(() => { measure(); tick(); });
})();
