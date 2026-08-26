# fractals-with-cuda

Real-time Mandelbrot and Julia set explorer. Every pixel is computed by a raw CUDA kernel running on an NVIDIA GPU (via CuPy + NVRTC), streamed to a clean web UI.

## Requirements

- Windows or Linux with an NVIDIA GPU (any CUDA-capable card)
- Python 3.12+
- [uv](https://github.com/astral-sh/uv)

No CUDA Toolkit install needed — CuPy bundles the CUDA runtime and compiles kernels at runtime.

## Run

```powershell
uv venv --python 3.12 .venv
uv pip install -e .
.venv\Scripts\python.exe -m uvicorn server.main:app --port 8000
```

Then open http://127.0.0.1:8000

## How it works

- `server/fractals.py` — raw CUDA kernels (Mandelbrot + Julia, smooth coloring) compiled via NVRTC
- `server/main.py` — FastAPI app, `/render` endpoint streams raw RGBA bytes
- `static/` — WebGL2 canvas UI, smooth zoom/pan with debounced GPU re-renders

## Benchmarks (RTX 3050 Laptop, 4GB VRAM)

| Resolution | fp32 | fp64 |
|---|---|---|
| 1080p (512 iter) | ~0.3 ms | ~42 ms |
| 4K (512 iter) | ~0.8 ms | ~163 ms |
