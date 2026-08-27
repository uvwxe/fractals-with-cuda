# fractals-with-cuda

Real-time Mandelbrot & Julia explorer rendering **every pixel with raw CUDA kernels** — a native desktop app (GLFW + OpenGL + CUDA-GL interop, zero frames leave the GPU) with a WebGL2 web version alongside.

![cuda](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76b900) ![python](https://img.shields.io/badge/python-3.12-3776ab) ![native](https://img.shields.io/badge/desktop-GLFW%20%2B%20OpenGL-25b09b) ![web](https://img.shields.io/badge/web-FastAPI%20%2B%20WebGL2-e34f26)

## Screenshots

| Seahorse Valley | Seahorse Deep (fp64, ×17,000) |
|---|---|
| ![hero](docs_hero.png) | ![deep](docs_deep.png) |

## Two ways to run

### Native app (recommended — faster, no browser)

The CUDA kernel writes **straight into a GPU-mapped OpenGL buffer** — the frame never touches the CPU, so zoom is ~45fps at full resolution, pixel-perfect, on vsync.

```powershell
uv venv --python 3.12 .venv
uv pip install -e .
.venv\Scripts\python.exe -m native.app
```

### Web version

```powershell
uv pip install -e .
.venv\Scripts\python.exe -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000

## What it does

- **Raw CUDA kernels per pixel** — Mandelbrot + Julia, smooth coloring (fixed ~20-iteration palette cycling), in-kernel 2×2 supersampling. JIT-compiled at runtime via NVRTC (no CUDA Toolkit install needed).
- **Native CUDA-OpenGL path** — mapped PBO + GPU-to-GPU texture upload + sync-per-frame. No device→host transfer. Verified pixel-exact against a host reference across 49k pixels and 60-frame alternating-view bursts.
- **fp32 interactive** (~45fps full-res motion) and **fp64 deep zoom** (auto-engages past ×34,000) with automatic resolution/iteration scaling.
- **Capability scaling** — the app probes the GPU's compute capability at startup and tunes itself (iterations, motion resolution, depth) so a strong card gets sharper and deeper, a weak card stays smooth.
- **~45fps continuous zoom** with velocity glide, zoom-to-cursor, drag pan, right-click → Julia seed, animated fly-ins to preset locations.
- **4 palettes** (Thermal, Mono, Classic, Sunset), iteration control, reset.

## Controls (native app)

| Action | Input |
|---|---|
| Zoom | wheel (velocity glide, zoom-to-cursor) |
| Pan | drag |
| Julia seed | right-click (Mandelbrot mode) |
| Mandelbrot / Julia | `M` |
| Palettes | `1`–`4` |
| fp32 / fp64 | `F` |
| Supersampling | `S` | | Iterations | `Up` / `Down` |
| Reset fly-in | `R` |
| Presets (boundary hotspots) | `5`–`9` |
| Quit | `Esc` |

## Presets

`5` Seahorse Valley · `6` Elephant Valley · `7` Triple Spiral · `8` Satellite Minibrot · `9` Seahorse Deep (fp64) — plus the web version has 14 verified presets.

## How deep can it go?

Double precision (fp64) is the hard ceiling: ~**×170 million** magnification, where the precision wall turns the boundary into a solid field. The app clamps there and tells you in the title bar. Going past that requires double-double GPU arithmetic (a planned follow-up) or CPU arbitrary-precision.

## Benchmarks (RTX 3050 Laptop, 4 GB)

| Mode | Cost |
|---|---|
| fp32 motion (full iters, 1280×720) | ~44 ms/frame (~23 fps)** |
| fp64 motion (6000 iters, 768px upscaled) | ~30 ms/frame (~33 fps) |
| Kernel (fp32, 1080p, 400 iters) | ~0.3 ms |

** sync-per-frame costs ~20 ms on Windows WDDM; the kernel itself is ~0.3 ms. That's the honest ceiling for browser+CUDA sharing a GPU.

## Test suite

```powershell
# native pipeline: renders through the real GL interop path, pixel-exact check
.venv\Scripts\python.exe -m native.app --selftest

# interactive simulation: scripted zoom/drag/deep sessions, reports fps/hitches/pixels
.venv\Scripts\python.exe -m native.simtest

# screenshot generator
.venv\Scripts\python.exe scripts\screenshot.py
```

## Architecture

- `native/app.py` — GLFW window, OpenGL present loop, CUDA-GL interop, inputs, adaptive resolution, capability probing
- `native/interop.py` — ctypes glue to the CUDA runtime for `cudaGraphicsGLRegisterBuffer` / map / unmap (bundled with CuPy — no CUDA Toolkit)
- `server/fractals.py` — the raw CUDA kernel source (fp32 + fp64 variants of one source), palettes, presets
- `server/main.py` — FastAPI: `/render`, `/ws` (streaming WebSocket renders), `/health`, `/presets`
- `static/` — vanilla JS + WebGL2 frontend (no build step)
