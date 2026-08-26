"""FastAPI app: GPU render endpoint + static frontend.

Run: .venv\\Scripts\\python.exe -m uvicorn server.main:app --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path
import time

import cupy as cp
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import fractals, palettes

MAX_ITER_FP32 = 20000
MAX_ITER_FP64 = 6000  # keep fp64 kernels well under the Windows TDR timeout


class RenderView(BaseModel):
    w: int = Field(1280, ge=16, le=4096)
    h: int = Field(720, ge=16, le=4096)
    centerRe: float = -0.6
    centerIm: float = 0.0
    scale: float = Field(3.4, gt=0.0)
    juliaRe: float = -0.8
    juliaIm: float = 0.156
    maxIter: int = Field(400, ge=32, le=MAX_ITER_FP32)
    mode: int = Field(0, ge=0, le=1)          # 0 mandelbrot, 1 julia
    palette: int = Field(0, ge=0, le=3)
    interior: int = Field(0, ge=0, le=1)      # 0 black, 1 white
    ssaa: int = Field(1, ge=1, le=2)          # 1 native, 2 = 2x2 supersample
    precision: int = Field(0, ge=0, le=1)     # 0 fp32, 1 fp64


@asynccontextmanager
async def lifespan(_: FastAPI):
    fractals.warmup()
    yield


app = FastAPI(title="fractals-with-cuda", lifespan=lifespan)


@app.post("/render")
def render(view: RenderView):
    if view.precision == 1 and view.maxIter > MAX_ITER_FP64:
        view.maxIter = MAX_ITER_FP64
    if view.ssaa == 2 and view.precision == 1:
        view.ssaa = 1  # fp64 supersampling is too slow for interactive use

    t0 = time.perf_counter()
    raw = fractals.render_rgba(view.model_dump())
    ms = (time.perf_counter() - t0) * 1000

    return Response(
        content=raw,
        media_type="application/octet-stream",
        headers={"X-Render-Ms": f"{ms:.1f}"},
    )


@app.get("/health")
def health():
    props = cp.cuda.runtime.getDeviceProperties(0)
    mem = cp.cuda.runtime.memGetInfo()
    return {
        "device": props["name"].decode(),
        "computeCapability": f"{props['major']}.{props['minor']}",
        "vramTotalMb": mem[1] // (1024 * 1024),
        "vramFreeMb": mem[0] // (1024 * 1024),
        "cupyVersion": cp.__version__,
        "cudaRuntime": cp.cuda.runtime.runtimeGetVersion(),
        "driver": cp.cuda.runtime.driverGetVersion(),
    }


@app.get("/presets")
def presets():
    return {
        "fullSet": fractals.FULL_SET,
        "deepZoom": fractals.DEEP_ZOOM_PRESETS,
        "julia": fractals.JULIA_PRESETS,
        "palettes": palettes.PALETTES,
        "defaultView": {
            "mode": 0,
            "palette": 0,
            "interior": 0,
            "ssaa": 1,
            "precision": 0,
            "maxIter": fractals.FULL_SET["maxIter"],
            "centerRe": fractals.FULL_SET["centerRe"],
            "centerIm": fractals.FULL_SET["centerIm"],
            "scale": fractals.FULL_SET["scale"],
        },
    }


_static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
