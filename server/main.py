"""FastAPI app: GPU render endpoint + static frontend.

Run: .venv\\Scripts\\python.exe -m uvicorn server.main:app --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import json
import struct
import time

import cupy as cp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import fractals, palettes

MAX_ITER_FP32 = 20000
MAX_ITER_FP64 = 6000  # keep fp64 kernels well under the Windows TDR timeout

# Preview cap: buffers wider than ~320px hit the ~50ms Windows WDDM slow-sync
# path; at <=320px a render+transfer is ~10ms. Keep previews small: they're
# upscaled with GL_LINEAR and replaced by a full-res pass on settle.
PREVIEW_MAX_W = 192

# Per-frame kernel-work guard (px * samples * iters). A request like
# 4096x4096 + ssaa2 + 20000 iters is ~2.8s of continuous GPU load, past the
# 2s Windows TDR default — the driver resets and the CUDA context dies,
# taking the server down. Caps keep any single render under ~1s on the
# calibration card (fp64 is ~4.4x costlier per px-iter, so it gets less).
WORK_CAP_FP32 = 4.0e11
WORK_CAP_FP64 = 8.0e10


def clamp_work(view: dict) -> None:
    pixels = int(view.get("w", 16)) * int(view.get("h", 16))
    samples = max(1, int(view.get("ssaa", 1))) ** 2
    cap = WORK_CAP_FP64 if int(view.get("precision", 0)) == 1 else WORK_CAP_FP32
    max_iter = int(cap / max(pixels * samples, 1))
    view["maxIter"] = max(32, min(int(view.get("maxIter", 400)), max_iter))


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

    v = view.model_dump()
    clamp_work(v)
    t0 = time.perf_counter()
    raw = fractals.render_rgba(v)
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


@app.websocket("/ws")
async def ws_render(ws: WebSocket):
    """Stream rendered frames back-to-back.

    Client sends JSON view updates (text). Server continuously renders the
    latest view and pushes frames as binary: 4-byte LE width + 4-byte LE height
    + RGB bytes. Rendering back-to-back pipelines the GPU so zoom is smooth
    instead of a request/response slideshow.
    """

    def _render_now(view: dict) -> bytes:
        pending = fractals.render_async(view)
        return pending.bytes()

    await ws.accept()
    view: dict | None = None
    seq = 0
    settle = False
    view_lock = asyncio.Lock()
    closed = False

    async def receiver():
        nonlocal view, seq, settle
        while True:
            try:
                msg = await ws.receive_text()
            except WebSocketDisconnect:
                break
            try:
                v = json.loads(msg)
            except (ValueError, TypeError):
                continue
            async with view_lock:
                view = v
                seq = v.get("seq", 0)
                settle = bool(v.get("settle", False))

    async def sender():
        nonlocal closed
        last_seq = -1
        last_settle = None
        while not closed:
            async with view_lock:
                v = view
                s = seq
                st = settle
            if v is None or (s == last_seq and st == last_settle):
                await asyncio.sleep(0.003)
                continue
            last_seq, last_settle = s, st
            # Preview downscale while moving; full-res on settle.
            w = int(v.get("w", 320))
            h = int(v.get("h", 200))
            if not st and w > PREVIEW_MAX_W:
                h = max(1, round(h * PREVIEW_MAX_W / w))
                w = PREVIEW_MAX_W
            rv = dict(v)
            rv["w"], rv["h"] = w, h
            # While moving, cap iterations so previews stay ~10ms/frame
            # (smooth motion). Full iteration depth only on settle.
            it = min(int(rv.get("maxIter", 400)), MAX_ITER_FP32)
            if not st:
                it = min(it, 900)
            rv["maxIter"] = it
            clamp_work(rv)  # TDR guard: settle frames are full-res, unvalidated
            try:
                # Render directly on the event loop: previews are ~3ms of
                # blocking, and skipping the executor removes per-frame
                # scheduling latency (~10ms+ on Windows).
                raw = _render_now(rv)
            except Exception:
                break
            try:
                await ws.send_bytes(struct.pack("<II", w, h) + raw)
            except Exception:
                closed = True
                break

    try:
        await asyncio.gather(receiver(), sender())
    finally:
        closed = True
        try:
            await ws.close()
        except Exception:
            pass


_static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")


def serve() -> None:
    """Console entry point (fractals-web): run the web server on port 8000."""
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
