// Code mode (docs/design/harness-app.md): the host's agent controller and the peers' read-only
// view. room.js imports this lazily and calls initCode(roomApi, { mock }) once. Stub until the
// UI lands: it registers nothing, so code messages are dropped as before.
export function initCode(api, opts = {}) {}
