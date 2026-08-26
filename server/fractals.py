"""Raw CUDA fractal kernels, compiled at runtime via NVRTC.

One thread per pixel. fp32 for interactive speed, fp64 for deep zoom.
Both variants come from the same source with USE_DOUBLE toggled.
Supersampling (2x2) happens in-kernel to avoid slow host-side reductions.
"""

import cupy as cp

_KERNEL_SRC = r"""
#if USE_DOUBLE
#define REAL double
#define RLOG log
#define RLOG2 log2
#else
#define REAL float
#define RLOG logf
#define RLOG2 log2f
#endif

__device__ inline REAL escape_time(
    REAL zr0, REAL zi0, REAL cr, REAL ci, int max_iter)
{
    REAL zr = zr0, zi = zi0;
    int n = 0;
    while (n < max_iter && (zr * zr + zi * zi) <= 4.0) {
        REAL tmp = zr * zr - zi * zi + cr;
        zi = 2.0 * zr * zi + ci;
        zr = tmp;
        n++;
    }
    if (n >= max_iter) return (REAL)-1.0;
    REAL logz = 0.5 * RLOG(zr * zr + zi * zi);
    REAL smooth = (REAL)n - RLOG2(logz / RLOG(2.0));
    // Normalize by a fixed iteration period (not max_iter) so every
    // palette cycle spans ~20 iterations. max_iter compression washes the
    // exterior into its first stop; a fixed period keeps vivid repeating
    // bands that scale naturally with zoom depth.
    REAL t = smooth / 60.0;
    if (t < 0.0) t = 0.0;
    return t;
}

__device__ inline void palette_color(
    REAL t, int palette, unsigned char& r, unsigned char& g, unsigned char& b)
{
    const REAL P_THERMAL[] = {0.30,0.80,0.79,  0.35,0.45,1.00,  1.00,0.42,0.42};
    const REAL P_MONO[]    = {0.00,0.00,0.00,  1.00,1.00,1.00};
    const REAL P_CLASSIC[] = {0.01,0.01,0.08,  0.05,0.25,0.85,  1.00,0.96,0.88,  0.85,0.55,0.10};
    const REAL P_SUNSET[]  = {0.02,0.01,0.06,  0.80,0.18,0.30,  1.00,0.68,0.20,  0.97,0.95,0.80};

    const REAL* stops = P_THERMAL;
    int n = 3;
    if (palette == 1) { stops = P_MONO;    n = 2; }
    else if (palette == 2) { stops = P_CLASSIC; n = 4; }
    else if (palette == 3) { stops = P_SUNSET;  n = 4; }

    const REAL cycles = 3.0;
    REAL tt = fmod(t * cycles * (REAL)(n - 1), (REAL)(n - 1));
    int i = (int)tt;
    if (i < 0) i = 0;
    if (i >= n - 1) i = n - 2;
    REAL f = tt - (REAL)i;

    REAL ar = stops[3 * i],     ag = stops[3 * i + 1],     ab = stops[3 * i + 2];
    REAL br = stops[3 * (i+1)], bg = stops[3 * (i+1) + 1], bb = stops[3 * (i+1) + 2];

    r = (unsigned char)(255.0 * (ar + (br - ar) * f));
    g = (unsigned char)(255.0 * (ag + (bg - ag) * f));
    b = (unsigned char)(255.0 * (ab + (bb - ab) * f));
}

extern "C" __global__
void fractal_kernel(
    unsigned char* out,
    const int width, const int height,
    const double min_re, const double min_im,
    const double step_re, const double step_im,
    const double julia_re, const double julia_im,
    const int max_iter,
    const int mode,
    const int palette,
    const int interior,
    const int ssaa)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Plane scalars arrive as double (cupy marshals Python floats as float64);
    // cast to REAL once here. fp32 still uses REAL=float math below.
    const REAL r_min_re = (REAL)min_re;
    const REAL r_min_im = (REAL)min_im;
    const REAL r_step_re = (REAL)step_re;
    const REAL r_step_im = (REAL)step_im;
    const REAL r_julia_re = (REAL)julia_re;
    const REAL r_julia_im = (REAL)julia_im;

    const int idx = 3 * (y * width + x);

    if (ssaa >= 2) {
        unsigned int sr = 0, sg = 0, sb = 0;
        for (int sy = 0; sy < 2; sy++) {
            for (int sx = 0; sx < 2; sx++) {
                REAL px = (REAL)x + (REAL)0.25 + (REAL)0.5 * (REAL)sx;
                REAL py = (REAL)y + (REAL)0.25 + (REAL)0.5 * (REAL)sy;
                REAL t;
                unsigned char r, g, b;
                if (mode == 0) {
                    t = escape_time(0.0, 0.0,
                        r_min_re + px * r_step_re, r_min_im + py * r_step_im, max_iter);
                } else {
                    t = escape_time(
                        r_min_re + px * r_step_re, r_min_im + py * r_step_im,
                        r_julia_re, r_julia_im, max_iter);
                }
                if (t < 0.0) {
                    unsigned char v = (interior == 1) ? (unsigned char)255 : (unsigned char)0;
                    r = v; g = v; b = v;
                } else {
                    palette_color(t, palette, r, g, b);
                }
                sr += r; sg += g; sb += b;
            }
        }
        out[idx] = (unsigned char)(sr >> 2);
        out[idx + 1] = (unsigned char)(sg >> 2);
        out[idx + 2] = (unsigned char)(sb >> 2);
        return;
    }

    REAL t;
    if (mode == 0) {
        t = escape_time(0.0, 0.0,
            r_min_re + (REAL)x * r_step_re, r_min_im + (REAL)y * r_step_im, max_iter);
    } else {
        t = escape_time(
            r_min_re + (REAL)x * r_step_re, r_min_im + (REAL)y * r_step_im,
            r_julia_re, r_julia_im, max_iter);
    }

    if (t < 0.0) {
        unsigned char v = (interior == 1) ? (unsigned char)255 : (unsigned char)0;
        out[idx] = v; out[idx + 1] = v; out[idx + 2] = v;
        return;
    }
    palette_color(t, palette, out[idx], out[idx + 1], out[idx + 2]);
}
"""

_kernels: dict[str, cp.RawKernel] = {}


def _get_kernel(use_double: bool) -> cp.RawKernel:
    key = "f64" if use_double else "f32"
    if key not in _kernels:
        opts = ("-DUSE_DOUBLE=1",) if use_double else ("-DUSE_DOUBLE=0",)
        _kernels[key] = cp.RawKernel(_KERNEL_SRC, "fractal_kernel", options=opts)
    return _kernels[key]


def _launch(kernel: cp.RawKernel, out, view: dict) -> None:
    w = int(view["w"])
    h = int(view["h"])
    step = view["scale"] / w
    min_re = view["centerRe"] - view["scale"] / 2.0
    min_im = view["centerIm"] - (view["scale"] * h / w) / 2.0

    # Kernel takes double for plane scalars (cupy marshals Python floats as
    # float64) and int for integer params. Passing pure Python scalars is
    # ~170x faster than wrapping in np.int32/np.float32 (43ms -> 0.25ms/call).
    grid = ((w + 15) // 16, (h + 15) // 16)
    kernel(
        grid,
        (16, 16),
        (
            out,
            w, h,
            min_re, min_im,
            step, step,
            view["juliaRe"], view["juliaIm"],
            int(view["maxIter"]),
            int(view["mode"]),
            int(view["palette"]),
            int(view["interior"]),
            int(view.get("ssaa", 1)),
        ),
    )


def render_rgba(view: dict) -> bytes:
    """Render a fractal view on the GPU, return raw RGB bytes."""
    w = int(view["w"])
    h = int(view["h"])
    use_double = bool(view.get("precision", 0))
    kernel = _get_kernel(use_double)

    out = cp.empty((h, w, 3), dtype=cp.uint8)
    _launch(kernel, out, view)
    # No explicit Stream.null.synchronize(): cp.asnumpy() blocks on the device
    # transfer, and a separate null-stream sync costs ~50ms on Windows WDDM.
    return cp.asnumpy(out).tobytes()


# ---- Pipelined async rendering for the WebSocket path ----------------------
# The browser's WebGL compositing contends with CUDA syncs on the same WDDM
# GPU (~90ms/frame instead of ~10ms). Pipelining: launch render N+1 while the
# async D2H copy of frame N is still in flight, so the GPU never idles behind
# a host sync. Uses a non-blocking stream + rotating pinned host buffers.

_stream = cp.cuda.Stream(non_blocking=True)
_pinned: dict[tuple[int, int], list] = {}


def _pinned_bufs(w: int, h: int) -> list:
    """Two rotating pinned host buffers per (w,h) so frame N's copy in flight
    never aliases frame N+1's target."""
    import numpy as np
    key = (w, h)
    if key not in _pinned:
        n = w * h * 3
        _pinned[key] = []
        for _ in range(2):
            mem = cp.cuda.alloc_pinned_memory(n)
            host = np.frombuffer(mem, dtype=np.uint8, count=n)
            _pinned[key].append((mem, host))
    return _pinned[key]


class PendingFrame:
    """A frame whose render+copy is in flight; call .bytes() to block for it."""

    def __init__(self, host, w, h):
        self._host = host
        self._w = w
        self._h = h

    def bytes(self):
        _stream.synchronize()
        return self._host[: self._w * self._h * 3].tobytes()


def render_async(view: dict) -> PendingFrame:
    """Launch kernel + async D2H copy on the pipeline stream. Returns a handle
    whose .bytes() waits for THIS frame only (earlier frames may still be
    rendering — they pipeline)."""
    w = int(view["w"])
    h = int(view["h"])
    use_double = bool(view.get("precision", 0))
    kernel = _get_kernel(use_double)

    out = cp.empty((h, w, 3), dtype=cp.uint8)
    bufs = _pinned_bufs(w, h)
    mem, host = bufs[0]
    bufs.append(bufs.pop(0))

    with _stream:
        _launch(kernel, out, view)
        cp.cuda.runtime.memcpyAsync(
            mem.ptr, out.data.ptr, w * h * 3, cp.cuda.runtime.memcpyDeviceToHost, _stream.ptr
        )
    return PendingFrame(host, w, h)


def warmup() -> None:
    """Compile both kernels on server start so first render is instant."""
    for use_double in (False, True):
        k = _get_kernel(use_double)
        out = cp.empty((8, 8, 3), dtype=cp.uint8)
        k((1, 1), (16, 16), (
            out,
            8, 8,
            -2.5, -1.0, 0.2, 0.2, -0.8, 0.156,
            64, 0, 0, 0, 1,
        ))
    cp.cuda.Stream.null.synchronize()


# --------------------------------------------------------------------------
# Presets — every view empirically verified via interior-ratio + gradient
# probes (scripts/tune_presets.py, scripts/minibrot_final.py) and
# cross-checked against published landmark coordinates.

FULL_SET = {"centerRe": -0.6, "centerIm": 0.0, "scale": 3.4, "maxIter": 400}

DEEP_ZOOM_PRESETS = [
    {"name": "Seahorse Valley", "centerRe": -0.745, "centerIm": 0.113, "scale": 0.05, "maxIter": 600},
    {"name": "Seahorse Deep", "centerRe": -0.743643887037151, "centerIm": 0.13182590420533, "scale": 0.0002, "maxIter": 1200, "precision": 1},
    {"name": "Elephant Valley", "centerRe": 0.2825, "centerIm": 0.005, "scale": 0.045, "maxIter": 600},
    {"name": "Triple Spiral", "centerRe": -0.088, "centerIm": 0.654, "scale": 0.04, "maxIter": 800},
    {"name": "Satellite Minibrot", "centerRe": -1.755, "centerIm": 0.0, "scale": 0.06, "maxIter": 3000},
    {"name": "Satellite Edge", "centerRe": -1.755, "centerIm": 0.012, "scale": 0.002, "maxIter": 4000},
    {"name": "Scepter Valley", "centerRe": -1.36, "centerIm": 0.035, "scale": 0.025, "maxIter": 600},
]

JULIA_PRESETS = [
    {"name": "Classic", "re": -0.8, "im": 0.156},
    {"name": "Swirl", "re": -0.4, "im": 0.6},
    {"name": "Douady Rabbit", "re": -0.123, "im": 0.745},
    {"name": "Basilica", "re": -1.0, "im": 0.0},
    {"name": "Dendrite", "re": 0.0, "im": 1.0},
    {"name": "Siegel Disk", "re": -0.39054, "im": -0.58679},
    {"name": "Feathers", "re": -0.835, "im": -0.2321},
]
