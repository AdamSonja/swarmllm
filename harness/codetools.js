// Coding tools for a Tabby agent over a workspace (harness/workspace.js). Each tool is
// { name, description, parameters (JSON schema), mutates, run(args, ctx) -> string,
//   preview?(args) -> { path, before, after } }.
// Results are plain text written for the model: short, with line numbers, capped so one call
// never eats the context (a 16k window holds ~11k of conversation), and with errors that say how
// to recover. Tools that change files are marked `mutates` so the agent can ask the user first;
// preview() gives the approval card its before/after without touching the file.
//
// codingTools(ws, { server }): with a PreviewServer, a write under a served folder says the
// preview reloads, so the model knows to check preview_logs rather than serve again.

export const MAX_LINES = 200, MAX_CHARS = 8000, MAX_LINE = 1000, MAX_HITS = 30, HIT_CHARS = 160, MAX_ENTRIES = 200;

const kb = (n) => (n < 1024 ? `${n} B` : `${(n / 1024).toFixed(1)} KB`);
const cut = (s, n) => (s.length > n ? s.slice(0, n) + `…(${s.length - n} chars cut)` : s);
const lineCount = (t) => (t === "" ? 0 : t.split("\n").length - (t.endsWith("\n") ? 1 : 0));

// edit_file's search/replace; also used by its preview so the card shows exactly what run does
function applyEdit(text, path, o, n) {
  if (!o) return { error: "error: old is empty; use write_file to create or replace a whole file" };
  if (o === n) return { error: "error: old and new are the same" };
  const k = text.split(o).length - 1;
  if (k === 0) return { error: `error: old not found in ${path}; read_file it again and copy the text exactly (whitespace included)` };
  if (k > 1) return { error: `error: old appears ${k} times in ${path}; include more surrounding lines so it is unique` };
  const at = text.indexOf(o);
  return { after: text.slice(0, at) + n + text.slice(at + o.length), line: text.slice(0, at).split("\n").length };
}

// the xml tool format trims one newline at each end of a parameter; either name is accepted
const oldOf = (a) => a.old ?? a.old_string ?? "", newOf = (a) => a.new ?? a.new_string ?? "";

export function codingTools(ws, { server = null } = {}) {
  const reloadNote = (path) => {
    const ports = server?.servedPorts?.(path) || [];
    return ports.length ? ` · preview ${ports.map((p) => ":" + p).join(" ")} reloaded` : "";
  };
  const readOr = async (p) => ((await ws.exists(p)) ? ws.read(p) : null);
  return [
    {
      name: "list_dir", mutates: false,
      description: "List a directory (default: project root). Folders end with /.",
      parameters: { type: "object", properties: { path: { type: "string" } } },
      async run({ path = "" } = {}) {
        const es = await ws.list(path);
        if (!es.length) return "(empty)";
        const out = es.slice(0, MAX_ENTRIES).map((e) => e.name + (e.dir ? "/" : ""));
        if (es.length > MAX_ENTRIES) out.push(`(+${es.length - MAX_ENTRIES} more)`);
        return out.join("\n");
      },
    },
    {
      name: "read_file", mutates: false,
      description: `Read a file with line numbers, at most ${MAX_LINES} lines per call; use start_line/end_line for more.`,
      parameters: { type: "object", properties: { path: { type: "string" }, start_line: { type: "integer" }, end_line: { type: "integer" } }, required: ["path"] },
      async run({ path, start_line, end_line }) {
        const text = await ws.read(path);
        if (text === "") return `(${path} is empty)`;
        const lines = text.split("\n");
        if (lines.length > 1 && lines[lines.length - 1] === "") lines.pop();   // a final newline is not a line
        const from = Math.max(1, Math.floor(start_line) || 1);
        if (from > lines.length) return `error: ${path} has ${lines.length} lines`;
        const last = Math.min(lines.length, Math.floor(end_line) || Infinity, from + MAX_LINES - 1);
        const out = [];
        let size = 0, to = from - 1;
        for (let i = from; i <= last; i++) {
          const l = `${i}|${cut(lines[i - 1], MAX_LINE)}`;
          if (out.length && size + l.length + 1 > MAX_CHARS) break;
          out.push(l); size += l.length + 1; to = i;
        }
        if (to < lines.length && !(end_line && to >= end_line)) out.push(`(lines ${from}-${to} of ${lines.length}; read on with start_line=${to + 1})`);
        return out.join("\n");
      },
    },
    {
      name: "search", mutates: false,
      description: "Search the project for a JavaScript regular expression; returns path:line: text.",
      parameters: { type: "object", properties: { pattern: { type: "string" }, path: { type: "string", description: "only under this folder" }, ignore_case: { type: "boolean" } }, required: ["pattern"] },
      async run({ pattern, path = "", ignore_case = false }) {
        let re;
        try { re = new RegExp(pattern, ignore_case ? "i" : ""); } catch (e) { return `error: bad pattern: ${e.message}`; }
        const pre = path && path !== "." ? path.replace(/\/+$/, "") + "/" : "";
        const hits = [];
        let total = 0;
        for (const f of await ws.walk()) {
          if (pre && !f.startsWith(pre)) continue;
          let text;
          try { text = await ws.read(f); } catch { continue; }
          if (text.includes("\u0000") || text.includes("�")) continue;   // binary
          const lines = text.split("\n");
          for (let i = 0; i < lines.length; i++) {
            if (!re.test(lines[i])) continue;
            if (++total <= MAX_HITS) hits.push(`${f}:${i + 1}: ${cut(lines[i].trim(), HIT_CHARS)}`);
          }
          if (total > 2000) break;   // enough to say "(+N more)"; a runaway pattern stops here
        }
        if (!hits.length) return "no matches";
        return hits.join("\n") + (total > MAX_HITS ? `\n(+${total - MAX_HITS}${total > 2000 ? "+" : ""} more; narrow the pattern or path)` : "");
      },
    },
    {
      name: "edit_file", mutates: true,
      description: "Replace one exact piece of text in a file. old must appear exactly once (add surrounding lines to make it unique).",
      parameters: { type: "object", properties: { path: { type: "string" }, old: { type: "string" }, new: { type: "string" } }, required: ["path", "old", "new"] },
      async preview(args) {
        const before = await ws.read(args.path), r = applyEdit(before, args.path, oldOf(args), newOf(args));
        return { path: args.path, before, after: r.error ? before : r.after, ...(r.error ? { error: r.error } : {}) };
      },
      async run(args) {
        const { path } = args, o = oldOf(args), n = newOf(args);
        if (!(await ws.exists(path))) return `error: no such file: ${path}; use write_file to create it`;
        const r = applyEdit(await ws.read(path), path, o, n);
        if (r.error) return r.error;
        await ws.write(path, r.after);
        const a = o.split("\n").length, b = n.split("\n").length, end = r.line + Math.max(b, 1) - 1;
        return `edited ${path} ${end > r.line ? `lines ${r.line}-${end}` : `line ${r.line}`} (${a} -> ${b} lines)` + reloadNote(path);
      },
    },
    {
      name: "write_file", mutates: true,
      description: "Create or overwrite a file. append: true adds to its end (write long files in parts).",
      parameters: { type: "object", properties: { path: { type: "string" }, content: { type: "string" }, append: { type: "boolean" } }, required: ["path", "content"] },
      async preview({ path, content = "", append = false }) {
        const before = await readOr(path);
        return { path, before, after: append && before != null ? before + content : content };
      },
      async run({ path, content = "", append = false }) {
        const before = append ? await readOr(path) : null;
        const text = before != null ? before + content : String(content);
        await ws.write(path, text);
        const size = `${lineCount(text)} lines, ${kb(new TextEncoder().encode(text).length)}`;
        return (before != null ? `appended to ${path} (now ${size})` : `wrote ${path} (${size})`) + reloadNote(path);
      },
    },
  ];
}
