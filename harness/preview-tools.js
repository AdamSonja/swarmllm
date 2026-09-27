// serve / preview_logs / stop_serve: the agent's side of the preview server
// (docs/design/harness-app.md C.2). serve answers with what the page did in its first 500 ms,
// so the common "write, serve, see the error" loop needs no separate preview_logs step.
import { DEFAULT_PORT } from "./preview.js";
import { buildPreviewDoc } from "./preview-build.js";

export const LOG_LINES = 40, LOG_CHARS = 3000, SERVE_LINES = 8;

const kb = (n) => (n < 1024 ? `${n} B` : `${(n / 1024).toFixed(1)} KB`);
const secs = (ms) => `${(ms / 1000).toFixed(1)}s`;
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
const where = (e) => (e.src ? `${e.src}${e.line ? `:${e.line}${e.col ? `:${e.col}` : ""}` : ""} ` : "");
// one log entry as the model reads it; stacks keep 3 frames
export function logLine(e) {
  const [first, ...rest] = e.text.split("\n");
  const more = rest.map((l) => l.trim()).filter(Boolean).slice(0, 3);
  const text = [first.slice(0, 300), ...more.map((l) => "  " + l.slice(0, 160))].join("\n");
  return `[${secs(e.t)}] ${e.level} ${where(e)}${text}`;
}
// consecutive identical entries fold into one line with ×N
function fold(lines) {
  const out = [];
  for (const e of lines) {
    const k = e.level + "\u0000" + e.src + e.line + "\u0000" + e.text, p = out[out.length - 1];
    if (p && p.k === k && p.e.rev === e.rev) p.n++;
    else out.push({ k, e, n: 1 });
  }
  return out.map(({ e, n }) => ({ e, n, text: logLine(e) + (n > 1 ? ` ×${n}` : "") }));
}

export function previewTools(server) {
  const portOf = (p) => (p == null || p === "" ? DEFAULT_PORT : Number(p));
  const nothing = (port) => {
    const others = server.ports().map((s) => ":" + s.port);
    return `error: nothing is served on :${port}${others.length ? ` (served: ${others.join(" ")})` : ""}; call serve first`;
  };
  return [
    {
      name: "serve", mutates: false,
      description: `Serve a folder as a static site on a preview port and report the page's first errors. Serving again reloads it.`,
      parameters: { type: "object", properties: { dir: { type: "string", description: "folder, default project root" }, port: { type: "integer", description: `default ${DEFAULT_PORT}` }, entry: { type: "string", description: "default index.html" } } },
      async run({ dir = "", port, entry = "index.html" } = {}) {
        port = portOf(port);
        const since = server.cursor(port);
        const snap = await server.serve({ dir, port, entry });
        const { missing } = buildPreviewDoc(snap, {});
        const head = `serving ${snap.dir || "."} on :${port} (${snap.entry}, ${plural(snap.files.size, "file")}, ${kb(snap.bytes)})`;
        const miss = missing.length ? `\nmissing: ${missing.slice(0, 10).join(", ")}${missing.length > 10 ? ` (+${missing.length - 10} more)` : ""}` : "";
        const idle = await server.whenIdle(port, 2000);
        if (!idle && !server.hasFrame(port)) return `${head} · no preview open, logs appear when it is${miss}`;
        const lines = server.logs(port, since).lines.filter((e) => e.rev === snap.rev);
        const errs = lines.filter((e) => e.level === "error"), warns = lines.filter((e) => e.level === "warn");
        const counts = errs.length || warns.length ? [errs.length && plural(errs.length, "error"), warns.length && plural(warns.length, "warning")].filter(Boolean).join(", ") : "no errors";
        const state = idle ? `loaded in ${Math.round(idle.loadedMs)} ms` : "still loading after 2 s";
        const shown = fold([...errs, ...warns].sort((a, b) => a.seq - b.seq)).slice(0, SERVE_LINES);
        const out = [`${head}`, `${state} · ${counts}${shown.length ? ":" : ""}`, ...shown.map((s) => s.text)];
        if (lines.length > shown.reduce((k, s) => k + s.n, 0)) out.push(`more: preview_logs since=${since}`);
        return out.join("\n") + miss;
      },
    },
    {
      name: "preview_logs", mutates: false,
      description: "Console output and errors of a served page since a cursor (each result ends with next: since=N).",
      parameters: { type: "object", properties: { port: { type: "integer" }, since: { type: "integer" } } },
      async run({ port, since = 0 } = {}) {
        port = portOf(port);
        const snap = server.snapshot(port);
        if (!snap) return nothing(port);
        const { lines, next, dropped } = server.logs(port, Number(since) || 0);
        if (!lines.length) {
          const at = server.loadedAt(port);
          return `no new logs on :${port} (rev ${snap.rev}, ${at ? `loaded ${secs(Date.now() - at)} ago` : server.hasFrame(port) ? "loading" : "no preview open"})\nnext: since=${next}`;
        }
        const rows = [];
        let rev = null;
        for (const r of fold(lines)) {
          if (r.e.rev !== rev) { rev = r.e.rev; rows.push({ e: r.e, text: `(rev ${rev}${rev === snap.rev ? ", current" : ""})` }); }
          rows.push(r);
        }
        // newest last; keep the tail within the caps and summarize what was cut
        let size = 0, keep = 0;
        for (let i = rows.length - 1; i >= 0 && keep < LOG_LINES; i--) {
          if (size + rows[i].text.length + 1 > LOG_CHARS && keep) break;
          size += rows[i].text.length + 1; keep++;
        }
        const cut = rows.length - keep, out = [];
        if (dropped) out.push(`(${dropped} older lines no longer kept)`);
        if (cut) out.push(`(${cut} earlier lines, since=${Number(since) || 0}; newest shown)`);
        out.push(...rows.slice(cut).map((r) => r.text), `next: since=${next}`);
        return out.join("\n");
      },
    },
    {
      name: "stop_serve", mutates: false,
      description: "Stop serving a preview port.",
      parameters: { type: "object", properties: { port: { type: "integer" } }, required: ["port"] },
      async run({ port } = {}) {
        port = portOf(port);
        return server.stop(port) ? `stopped :${port}` : nothing(port);
      },
    },
  ];
}
