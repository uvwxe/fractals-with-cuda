// Fractal explorer frontend — WebGL2 view + FastAPI CUDA render backend.

const MIN_SCALE = 1e-14;
const MAX_SCALE = 10;
const FADE_MS = 150;

const state = {
  centerRe: -0.6, centerIm: 0, scale: 3.4, mode: 0,
  juliaRe: -0.8, juliaIm: 0.156, maxIter: 400,
  palette: 0, interior: 0, ssaa: 1, precision: 0
};

const canvas = document.getElementById('view');
const gpuBadge = document.getElementById('gpu-badge');
const hudMag = document.getElementById('hud-mag');
const hudCoords = document.getElementById('hud-coords');
const hudRender = document.getElementById('hud-render');
const modeBtns = [...document.querySelectorAll('#mode-seg button[data-mode]')];
const paletteDots = [...document.querySelectorAll('.palette-dot')];
const presetSelect = document.getElementById('preset-select');
const iterSlider = document.getElementById('iter-slider');
const iterValue = document.getElementById('iter-value');
const ssaaBtn = document.getElementById('ssaa-btn');
const precisionBtn = document.getElementById('precision-btn');
const resetBtn = document.getElementById('reset-btn');
const hint = document.getElementById('hint');
const toast = document.getElementById('toast');

let defaultView = null;

const gl = canvas.getContext('webgl2');
if (!gl) {
  toast.textContent = 'WebGL2 not supported in this browser';
  toast.classList.add('show');
  throw new Error('WebGL2 not supported');
}

gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
gl.enable(gl.BLEND);
gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

function compileShader(type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
  return sh;
}

const program = gl.createProgram();
gl.attachShader(program, compileShader(gl.VERTEX_SHADER, `#version 300 es
out vec2 v_uv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 0.5;
  gl_Position = vec4(p * 4.0 - 1.0, 0.0, 1.0);
  v_uv = vec2(p.x, 1.0 - p.y);
}`));
gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, `#version 300 es
precision highp float;
uniform sampler2D u_tex;
uniform vec4 u_xf;
uniform float u_alpha;
in vec2 v_uv;
out vec4 fragColor;
void main() {
  vec2 uv = vec2(u_xf.x * v_uv.x + u_xf.z, u_xf.y * v_uv.y + u_xf.w);
  fragColor = vec4(texture(u_tex, uv).rgb * u_alpha, u_alpha);
}`));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
gl.useProgram(program);
const uXfLoc = gl.getUniformLocation(program, 'u_xf');
const uAlphaLoc = gl.getUniformLocation(program, 'u_alpha');

function createTexture() {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, 1, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0]));
  return t;
}

let curTex = createTexture();
let spareTex = createTexture();
let hasFrame = false;
let fading = false;
let fadeStart = 0;
let oldTex = null;
let oldXf = null;
let displayTransform = { sx: 1, sy: 1, tx: 0, ty: 0 };

function draw(tex, xf, alpha) {
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.uniform4f(uXfLoc, xf.sx, xf.sy, xf.tx, xf.ty);
  gl.uniform1f(uAlphaLoc, alpha);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

function frame(now) {
  requestAnimationFrame(frame);
  if (anim) stepAnim(now);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  if (!hasFrame) return;
  if (fading) {
    const t = Math.min(1, (now - fadeStart) / FADE_MS);
    draw(oldTex, oldXf, 1);
    draw(curTex, displayTransform, t);
    if (t >= 1) fading = false;
  } else {
    draw(curTex, displayTransform, 1);
  }
}
requestAnimationFrame(frame);

function showToast(msg) {
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => toast.classList.remove('show'), 3000);
}

function uploadFrame(px) {
  const w = canvas.width;
  const h = canvas.height;
  const old = hasFrame ? { tex: curTex, xf: { ...displayTransform } } : null;
  const swap = curTex;
  curTex = spareTex;
  spareTex = swap;
  gl.bindTexture(gl.TEXTURE_2D, curTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, w, h, 0, gl.RGB, gl.UNSIGNED_BYTE, px);
  displayTransform = { sx: 1, sy: 1, tx: 0, ty: 0 };
  if (old) {
    oldTex = old.tex;
    oldXf = old.xf;
    fadeStart = performance.now();
    fading = true;
  } else {
    hasFrame = true;
  }
}

let inFlight = false;
let dirty = false;

async function requestRender() {
  if (inFlight) { dirty = true; return; }
  inFlight = true;
  try {
    const res = await fetch('/render', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        w: canvas.width, h: canvas.height,
        centerRe: state.centerRe, centerIm: state.centerIm, scale: state.scale,
        juliaRe: state.juliaRe, juliaIm: state.juliaIm,
        maxIter: state.maxIter, mode: state.mode, palette: state.palette,
        interior: state.interior, ssaa: state.ssaa, precision: state.precision
      })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const buf = await res.arrayBuffer();
    if (buf.byteLength === canvas.width * canvas.height * 3) {
      uploadFrame(new Uint8Array(buf));
    }
    hudRender.textContent = `render ${res.headers.get('X-Render-Ms')} ms`;
  } catch (err) {
    showToast('GPU server unreachable — is uvicorn running?');
  } finally {
    inFlight = false;
    if (dirty) { dirty = false; requestRender(); }
  }
}

let renderTimer = null;
function scheduleRender() {
  clearTimeout(renderTimer);
  renderTimer = setTimeout(requestRender, 120);
}

function formatMag(n) {
  const fmt = (v) => v >= 100 ? String(Math.round(v)) : v.toPrecision(2);
  if (n >= 1e6) return fmt(n / 1e6) + 'M';
  if (n >= 1000) return fmt(n / 1000) + 'k';
  if (n >= 100) return String(Math.round(n));
  if (n >= 10) return n.toFixed(1);
  return n.toPrecision(2);
}

function updateHud() {
  hudMag.textContent = '×' + formatMag(3.4 / state.scale);
  const d = state.scale < 1e-4 ? 10 : state.scale < 1 ? 7 : 5;
  const re = state.centerRe.toFixed(d);
  const sign = state.centerIm >= 0 ? '+' : '−';
  const im = Math.abs(state.centerIm).toFixed(d);
  hudCoords.textContent = `${re} ${sign} ${im}i`;
}

function updateModeUI() {
  modeBtns.forEach(b => b.classList.toggle('active', +b.dataset.mode === state.mode));
}

function updatePaletteUI() {
  paletteDots.forEach(b => b.classList.toggle('active', +b.dataset.palette === state.palette));
}

function updateIterUI() {
  iterSlider.value = state.maxIter;
  iterValue.textContent = state.maxIter;
}

function updateSsaaUI() {
  ssaaBtn.classList.toggle('active', state.ssaa === 2);
}

function setPrecision(v) {
  state.precision = v;
  precisionBtn.classList.toggle('active', v === 1);
  if (v === 1) {
    state.ssaa = 1;
    ssaaBtn.disabled = true;
  } else {
    ssaaBtn.disabled = false;
  }
  updateSsaaUI();
}

function setIter(n) {
  state.maxIter = n;
  updateIterUI();
  scheduleRender();
}

let anim = null;

const EASE = t => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

function animateTo(target, dur = 1400) {
  const from = { centerRe: state.centerRe, centerIm: state.centerIm, scale: state.scale };
  const to = {
    centerRe: target.centerRe ?? state.centerRe,
    centerIm: target.centerIm ?? state.centerIm,
    scale: target.scale ?? state.scale
  };
  state.mode = target.mode ?? state.mode;
  state.juliaRe = target.juliaRe ?? state.juliaRe;
  state.juliaIm = target.juliaIm ?? state.juliaIm;
  state.maxIter = target.maxIter ?? state.maxIter;
  state.palette = target.palette ?? state.palette;
  state.ssaa = target.ssaa ?? state.ssaa;
  setPrecision(target.precision ?? state.precision);
  updateModeUI();
  updatePaletteUI();
  updateIterUI();
  updateSsaaUI();
  anim = { from, to, t0: performance.now(), dur, lastRender: 0 };
}

function stepAnim(now) {
  const t = Math.min(1, (now - anim.t0) / anim.dur);
  const e = EASE(t);
  const logFrom = Math.log(anim.from.scale);
  const logTo = Math.log(anim.to.scale);
  state.scale = Math.exp(logFrom + (logTo - logFrom) * e);
  state.centerRe = anim.from.centerRe + (anim.to.centerRe - anim.from.centerRe) * e;
  state.centerIm = anim.from.centerIm + (anim.to.centerIm - anim.from.centerIm) * e;
  displayTransform = { sx: 1, sy: 1, tx: 0, ty: 0 };
  updateHud();
  if (now - anim.lastRender > 40) {
    anim.lastRender = now;
    requestRender();
  }
  if (t >= 1) {
    const exact = { ...anim.to };
    anim = null;
    state.scale = exact.scale;
    state.centerRe = exact.centerRe;
    state.centerIm = exact.centerIm;
    updateHud();
    requestRender();
  }
}

function cancelAnim() {
  anim = null;
}

function resetAll() {
  if (!defaultView) return;
  animateTo(defaultView);
}

function localPoint(e) {
  const r = canvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top, w: r.width, h: r.height };
}

function complexAt(px, py) {
  const p = state.scale / canvas.clientWidth;
  return {
    re: state.centerRe + (px - canvas.clientWidth / 2) * p,
    im: state.centerIm - (py - canvas.clientHeight / 2) * p
  };
}

function panBy(dx, dy) {
  cancelAnim();
  displayTransform.tx += dx / canvas.clientWidth;
  displayTransform.ty += dy / canvas.clientHeight;
  const p = state.scale / canvas.clientWidth;
  state.centerRe -= dx * p;
  state.centerIm += dy * p;
}

function zoomAround(factor, cx, cy, fx, fy) {
  cancelAnim();
  const c = complexAt(fx, fy);
  const old = state.scale;
  state.scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, old / factor));
  const mag = old / state.scale;
  displayTransform.sx *= mag;
  displayTransform.sy *= mag;
  displayTransform.tx = displayTransform.tx * mag + cx * (1 - mag);
  displayTransform.ty = displayTransform.ty * mag + cy * (1 - mag);
  const p = state.scale / canvas.clientWidth;
  state.centerRe = c.re - (fx - canvas.clientWidth / 2) * p;
  state.centerIm = c.im + (fy - canvas.clientHeight / 2) * p;
}

const pointers = new Map();
let clickCandidate = null;
let dragging = false;
let pinchDist = 0;
canvas.style.cursor = 'grab';

canvas.addEventListener('pointerdown', e => {
  e.preventDefault();
  canvas.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (pointers.size === 1) {
    clickCandidate = { x: e.clientX, y: e.clientY, moved: 0 };
    dragging = false;
  } else if (pointers.size === 2) {
    clickCandidate = null;
    dragging = false;
    const pts = [...pointers.values()];
    pinchDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
  }
});

canvas.addEventListener('pointermove', e => {
  if (!pointers.has(e.pointerId)) return;
  const prev = pointers.get(e.pointerId);
  const dx = e.clientX - prev.x;
  const dy = e.clientY - prev.y;
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (pointers.size === 2) {
    clickCandidate = null;
    const pts = [...pointers.values()];
    const d = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
    const factor = d / pinchDist;
    pinchDist = d;
    const r = canvas.getBoundingClientRect();
    const mx = (pts[0].x + pts[1].x) / 2 - r.left;
    const my = (pts[0].y + pts[1].y) / 2 - r.top;
    zoomAround(factor, mx / r.width, my / r.height, mx, my);
    updateHud();
    scheduleRender();
  } else if (clickCandidate && !dragging) {
    clickCandidate.moved = Math.hypot(e.clientX - clickCandidate.x, e.clientY - clickCandidate.y);
    if (clickCandidate.moved > 4) {
      dragging = true;
      canvas.style.cursor = 'grabbing';
      panBy(e.clientX - clickCandidate.x, e.clientY - clickCandidate.y);
      clickCandidate = null;
      updateHud();
      scheduleRender();
    }
  } else if (dragging) {
    panBy(dx, dy);
    updateHud();
    scheduleRender();
  }
});

function endPointer(e, fireClick) {
  pointers.delete(e.pointerId);
  if (pointers.size === 1) {
    const p = [...pointers.values()][0];
    clickCandidate = { x: p.x, y: p.y, moved: 0 };
    dragging = false;
    canvas.style.cursor = 'grab';
  } else if (pointers.size === 0) {
    if (fireClick && clickCandidate && clickCandidate.moved < 4) handleClick(e.clientX, e.clientY);
    clickCandidate = null;
    dragging = false;
    canvas.style.cursor = 'grab';
  }
}

canvas.addEventListener('pointerup', e => endPointer(e, true));
canvas.addEventListener('pointercancel', e => endPointer(e, false));

function handleClick(cx, cy) {
  if (state.mode !== 0) return;
  cancelAnim();
  const r = canvas.getBoundingClientRect();
  const c = complexAt(cx - r.left, cy - r.top);
  state.juliaRe = c.re;
  state.juliaIm = c.im;
  state.mode = 1;
  state.centerRe = 0;
  state.centerIm = 0;
  state.scale = 3.4;
  displayTransform = { sx: 1, sy: 1, tx: 0, ty: 0 };
  updateModeUI();
  updateHud();
  requestRender();
}

canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const pt = localPoint(e);
  zoomAround(Math.exp(-e.deltaY * 0.0015), pt.x / pt.w, pt.y / pt.h, pt.x, pt.y);
  updateHud();
  scheduleRender();
}, { passive: false });

modeBtns.forEach(b => b.addEventListener('click', () => {
  state.mode = +b.dataset.mode;
  updateModeUI();
  requestRender();
}));

paletteDots.forEach(b => b.addEventListener('click', () => {
  state.palette = +b.dataset.palette;
  updatePaletteUI();
  requestRender();
}));

presetSelect.addEventListener('change', () => {
  const p = JSON.parse(presetSelect.value);
  cancelAnim();
  animateTo(p);
});

iterSlider.addEventListener('input', () => {
  state.maxIter = +iterSlider.value;
  iterValue.textContent = iterSlider.value;
  scheduleRender();
});

ssaaBtn.addEventListener('click', () => {
  if (state.precision === 1) return;
  state.ssaa = state.ssaa === 1 ? 2 : 1;
  updateSsaaUI();
  requestRender();
});

precisionBtn.addEventListener('click', () => {
  setPrecision(1 - state.precision);
  requestRender();
});

resetBtn.addEventListener('click', resetAll);

window.addEventListener('keydown', e => {
  const t = e.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
  switch (e.key.toLowerCase()) {
    case 'r': resetAll(); break;
    case 'm':
      state.mode = 1 - state.mode;
      updateModeUI();
      requestRender();
      break;
    case 'f':
      setPrecision(1 - state.precision);
      requestRender();
      break;
    case 'q':
      if (state.precision === 0) {
        state.ssaa = state.ssaa === 1 ? 2 : 1;
        updateSsaaUI();
        requestRender();
      }
      break;
    case '[': setIter(Math.max(+iterSlider.min, Math.floor(state.maxIter / 2))); break;
    case ']': setIter(Math.min(+iterSlider.max, state.maxIter * 2)); break;
  }
  if (e.key >= '1' && e.key <= '4') {
    state.palette = +e.key - 1;
    updatePaletteUI();
    requestRender();
  }
});

function onResize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.round(canvas.clientWidth * dpr);
  const h = Math.round(canvas.clientHeight * dpr);
  if (w !== canvas.width || h !== canvas.height) {
    canvas.width = w;
    canvas.height = h;
    displayTransform = { sx: 1, sy: 1, tx: 0, ty: 0 };
    scheduleRender();
  }
}
window.addEventListener('resize', onResize);
onResize();

fetch('/presets')
  .then(r => { if (!r.ok) throw new Error(); return r.json(); })
  .then(p => {
    const deep = document.createElement('optgroup');
    deep.label = 'Deep Zoom';
    p.deepZoom.forEach(x => {
      const o = document.createElement('option');
      o.value = JSON.stringify({ ...x, mode: 0 });
      o.textContent = x.name;
      deep.appendChild(o);
    });
    const jul = document.createElement('optgroup');
    jul.label = 'Julia';
    p.julia.forEach(x => {
      const o = document.createElement('option');
      o.value = JSON.stringify({ mode: 1, centerRe: 0, centerIm: 0, scale: 3.4, juliaRe: x.re, juliaIm: x.im, maxIter: 400 });
      o.textContent = x.name;
      jul.appendChild(o);
    });
    presetSelect.appendChild(deep);
    presetSelect.appendChild(jul);
    defaultView = { ...p.defaultView };
  })
  .catch(() => showToast('GPU server unreachable — is uvicorn running?'));

fetch('/health')
  .then(r => { if (!r.ok) throw new Error(); return r.json(); })
  .then(h => {
    const name = h.device.replace(/^NVIDIA (GeForce )?/, '').replace(/ Laptop GPU$/, '');
    gpuBadge.textContent = `${name} · ${Math.round(h.vramTotalMb / 1024)} GB · CUDA ${(h.cudaRuntime / 1000).toFixed(1)}`;
  })
  .catch(() => showToast('GPU server unreachable — is uvicorn running?'));

setTimeout(() => hint.classList.add('hidden'), 6000);

requestRender();
