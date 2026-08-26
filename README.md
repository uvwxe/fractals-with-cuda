# fractals-with-cuda

Real-time Mandelbrot and Julia set explorer. Every pixel is computed by a **raw CUDA kernel** running on an NVIDIA GPU (CuPy + NVRTC runtime compilation — no CUDA Toolkit install needed), streamed to a clean web UI rendered with WebGL2.

![fractals with cuda](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76b900) ![python](https://img.shields.io/badge/python-3.12-3776ab)

## Requirements

- Windows or Linux with any CUDA-capable NVIDIA GPU
- Python 3.12+
- [uv](https://github.com/astral-sh/uv)

## Run

```powershell
uv venv --python 3.12 .venv
uv pip install -e .
.venv\Scripts\python.exe -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

## Features

- **Raw CUDA kernels** — one thread per pixel, compiled at runtime via NVRTC
- **fp32 interactive mode** (~0.3 ms per 1080p frame) and **fp64 deep-zoom mode** (beyond 100,000× magnification)
- **Smooth zoom** — instant canvas transforms while zooming, debounced GPU re-renders, 150 ms crossfade between frames
- **Click a Mandelbrot point → jump to its Julia set**
- **14 verified presets** — Seahorse Valley, Elephant Valley, Triple Spiral, Satellite Minibrot, Douady Rabbit, Siegel Disk, and more (every view empirically verified)
- **4 palettes** — Thermal, Mono, Classic, Sunset
- **2×2 supersampling** (in-kernel) for crisp frames
- **Live HUD** — magnification, center coordinates, render time, GPU badge
- **Animated preset fly-ins** — cinematic eased zoom between views

## Controls

| Action | Input |
|--------|-------|
| Zoom | Scroll wheel / pinch |
| Pan | Drag |
| Jump to Julia | Click a point (Mandelbrot mode) |
| Switch mode | `M` or segmented control |
| Palettes | `1`–`4` or palette dots |
| Iterations | `[` / `]` or slider |
| fp64 toggle | `F` |
| Supersampling | `Q` or 2× button |
| Reset | `R` |

## How it works

- `server/fractals.py` — CUDA kernel source (Mandelbrot + Julia, smooth coloring, in-kernel SSAA), compiled twice: fp32 and fp64 variants of the same source
- `server/main.py` — FastAPI app: `POST /render` (view state → raw RGB bytes), `GET /health`, `GET /presets`
- `static/` — vanilla JS + WebGL2 frontend, no build step

## Benchmarks (RTX 3050 Laptop, 4 GB VRAM)

| Resolution | fp32 | fp64 |
|---|---|---|
| 1080p (512 iter) | ~0.3 ms | ~42 ms |
| 4K (512 iter) | ~0.8 ms | ~163 ms |

Deep-zoom presets marked `precision: 1` render in fp64; everything else stays in fp32.
