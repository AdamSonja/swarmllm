// The files a Tabby agent works on. Two implementations with one interface:
//   MemoryWorkspace   - a Map of path -> text (tests, scratch projects)
//   DirWorkspace      - a folder the user picked with showDirectoryPicker() (File System Access
//                       API): reads and writes go to the real files on their disk, nothing leaves
//                       the machine
// Paths are relative, "/"-separated, and may not climb out of the root ("..").
//
// interface: list(dir) -> [{ name, dir: bool }], read(path) -> string, write(path, text),
//            readBytes(path) -> Uint8Array, writeBytes(path, u8), remove(path) (file or folder),
//            exists(path) -> bool, walk() -> [path] (every file, for search)
// watch(ws) adds onChange(fn) so the preview server hears about every write made through it.

const enc = new TextEncoder(), dec = new TextDecoder();

export function normPath(p) {
  const parts = [];
  for (const s of String(p || "").replace(/\\/g, "/").split("/")) {
    if (!s || s === ".") continue;
    if (s === "..") throw new Error(`path leaves the workspace: ${p}`);
    parts.push(s);
  }
  return parts.join("/");
}

// Values are text or Uint8Array (images for the preview); each read converts as needed.
export class MemoryWorkspace {
  constructor(files = {}) { this.files = new Map(Object.entries(files).map(([k, v]) => [normPath(k), v])); }
  _get(p) {
    const k = normPath(p);
    if (!this.files.has(k)) throw new Error(`no such file: ${k}`);
    return this.files.get(k);
  }
  async read(p) { const v = this._get(p); return typeof v === "string" ? v : dec.decode(v); }
  async readBytes(p) { const v = this._get(p); return typeof v === "string" ? enc.encode(v) : v; }
  async write(p, text) { this.files.set(normPath(p), String(text)); }
  async writeBytes(p, u8) { this.files.set(normPath(p), new Uint8Array(u8)); }
  async remove(p) {
    const k = normPath(p), pre = k + "/";
    let n = 0;
    for (const f of [...this.files.keys()]) if (f === k || f.startsWith(pre)) { this.files.delete(f); n++; }
    if (!n) throw new Error(`no such file: ${k}`);
  }
  async exists(p) { const k = normPath(p); return this.files.has(k) || [...this.files.keys()].some((f) => f.startsWith(k + "/")); }
  async list(dir = "") {
    const d = normPath(dir), pre = d ? d + "/" : "", seen = new Map();
    for (const f of this.files.keys()) {
      if (!f.startsWith(pre)) continue;
      const rest = f.slice(pre.length), i = rest.indexOf("/");
      if (i < 0) seen.set(rest, false); else seen.set(rest.slice(0, i), true);
    }
    if (d && !seen.size) throw new Error(`no such directory: ${d}`);
    return [...seen].map(([name, dir]) => ({ name, dir })).sort((a, b) => a.name.localeCompare(b.name));
  }
  async walk() { return [...this.files.keys()].sort(); }
}

// Folders nobody wants an agent to read through.
export const SKIP_DIRS = new Set([".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next", "target"]);

export class DirWorkspace {
  constructor(handle) { this.root = handle; }   // a FileSystemDirectoryHandle
  async _dir(parts, create = false) {
    let h = this.root;
    for (const s of parts) h = await h.getDirectoryHandle(s, { create });
    return h;
  }
  async _file(p) {
    const parts = normPath(p).split("/");
    return (await (await this._dir(parts.slice(0, -1))).getFileHandle(parts[parts.length - 1])).getFile();
  }
  async read(p) { return (await this._file(p)).text(); }
  async readBytes(p) { return new Uint8Array(await (await this._file(p)).arrayBuffer()); }
  async write(p, text) {
    const parts = normPath(p).split("/");
    const f = await (await this._dir(parts.slice(0, -1), true)).getFileHandle(parts[parts.length - 1], { create: true });
    const w = await f.createWritable(); await w.write(text); await w.close();
  }
  async writeBytes(p, u8) { return this.write(p, u8); }   // createWritable takes a BufferSource too
  async remove(p) {
    const parts = normPath(p).split("/");
    await (await this._dir(parts.slice(0, -1))).removeEntry(parts[parts.length - 1], { recursive: true });
  }
  async exists(p) {
    const parts = normPath(p).split("/");
    try {
      const d = await this._dir(parts.slice(0, -1));
      try { await d.getFileHandle(parts[parts.length - 1]); } catch { await d.getDirectoryHandle(parts[parts.length - 1]); }
      return true;
    } catch { return false; }
  }
  async list(dir = "") {
    const d = await this._dir(normPath(dir) ? normPath(dir).split("/") : []), out = [];
    for await (const [name, h] of d.entries()) out.push({ name, dir: h.kind === "directory" });
    return out.sort((a, b) => a.name.localeCompare(b.name));
  }
  async walk(limit = 5000) {
    const out = [];
    const go = async (h, pre) => {
      for await (const [name, c] of h.entries()) {
        if (out.length >= limit) return;
        if (c.kind === "directory") { if (!SKIP_DIRS.has(name)) await go(c, pre + name + "/"); }
        else out.push(pre + name);
      }
    };
    await go(this.root, "");
    return out.sort();
  }
}

// A view of ws whose writes and removes are reported to onChange listeners after they land
// ({ path, kind: "write" | "remove" }). Reads go straight through. Watching a watched workspace
// returns it as is, so the tools and the preview server can each call watch() safely.
export function watch(ws) {
  if (ws.onChange) return ws;
  const fns = new Set(), w = Object.create(ws);
  const emit = (path, kind) => { for (const f of [...fns]) { try { f({ path, kind }); } catch (e) { console.error(e); } } };
  w.write = async (p, t) => { await ws.write(p, t); emit(normPath(p), "write"); };
  w.writeBytes = async (p, u8) => { await ws.writeBytes(p, u8); emit(normPath(p), "write"); };
  w.remove = async (p) => { await ws.remove(p); emit(normPath(p), "remove"); };
  w.onChange = (fn) => { fns.add(fn); return () => fns.delete(fn); };
  return w;
}
