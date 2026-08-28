"""Native fractal explorer: GLFW + OpenGL + CUDA interop on cupy kernels.

The CUDA kernel writes pixels straight into a GL-registered pixel buffer
(GPU-to-GPU, no device-to-host copy, no browser, no HTTP). Presented on
vsync. This is the real deal: input -> kernel -> pixels, one frame.

Run:  .venv\\Scripts\\python.exe -m native.app [--selftest] [--res WxH]

Keys: drag = pan · wheel = zoom · right-click = Julia seed · M = Mandelbrot/Julia
      F = fp32/fp64 (pins) · S = 2x2 supersample · 1-4 = palette
      Up/Down = iterations (pins) · R = reset · Esc = quit
"""
import json
import math
import os
import sys
import time

import glfw
from OpenGL import GL as gl

import cupy as cp

from native.interop import CudaGLError, GlCudaBuffer, prefer_discrete_gpu, sys_executable
from server import fractals

def min_scale() -> float:
    return CAP["min_scale"]


MAX_SCALE = 10.0
FULL_SET = {"centerRe": -0.6, "centerIm": 0.0, "scale": 3.4}


def auto_iter(mag: float, precision: int) -> int:
    ceiling = CAP["iter_ceiling_fp64"] if precision else CAP["iter_ceiling_fp32"]
    return int(min(ceiling, round(400 * (1 + math.sqrt(min(mag, 20000)) * 0.9))))


def format_mag(n: float) -> str:
    if n >= 1e6:
        return f"{n/1e6:.0f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}k"
    if n >= 10:
        return f"{n:.0f}"
    return f"{n:.1f}"


# Kernel-cost calibration. Event timing on this machine is polluted by the
# WDDM flush tax (~20ms per host-observable CUDA sync), so probes never trust
# event elapsed times. Instead use the trusted kernel measurements:
#   fp32 0.26ms @ 1080p (2,073,600 px), 400 iters   -> cost-per-px-iter
#   fp64 is ~161x fp32 at equal size/iters (consumer fp64 rate)
_PROBE = {"fp32": 0.26 * (65536 / 2073600) * (256 / 400), "fp64": None}
_PROBE["fp64"] = _PROBE["fp32"] * 161.0
_PROBE_PX = 256 * 256
_PROBE_ITERS = 256


def predict_ms(precision: int, pixels: int, iters: int, width: int = 0) -> float:
    # Two-part model: a fixed WDDM present floor that scales with render
    # width (measured: ~16ms @640px, ~21ms @1280px) plus the pure kernel
    # time (per-px-iter, calibrated from the real loop measurements:
    # fp64 4000it@720p total 55.7ms - 21ms floor => 9.2e-9 ms/px/iter.
    # The old pure-linear model ignored the floor and over-estimated
    # deep fp64 by ~3x, which made adaptive-res crater to 25%.
    import math as _m
    floor = (13.0 + width * 0.0068) if width else 0.0
    kpi = 2.08e-9 if precision == 0 else 9.2e-9
    return floor + kpi * (max(pixels, 1)) * (max(iters, 1))


# ---- Capability scaling ----------------------------------------------------
# A hardcoded zoom floor + resolution cap was tuned to THIS RTX 3050 Laptop.
# On a stronger card it needlessly limits depth/quality; on a weaker one it
# over-promises. Probe the actual device and derive constants from what it
# can genuinely do. run_probe() populates CAP for the lifetime of the app.
CAP = {
    "min_scale": 1e-14,    # measured: fp64 renders detail to 1e-14 at real
                           # boundary coords (the old 2e-8 'precision wall'
                           # was an artifact of testing at a uniform spot)
    "motion_wide": 768,    # motion render width (present-path cliff)
    "iter_ceiling_fp64": 6000,
    "iter_ceiling_fp32": 12000,
    "device": "",
}


def run_probe() -> None:
    """Tier the device by compute capability (authoritative fp64 rate flag:
    CC 5.2-6.x / 9.x = 1:2 fp64, CC 7.x-8.x consumer = 1:32-1:64).

    Real-time kernel timing at small sizes is launch-overhead dominated and
    misestimates the fp64 slowdown (measured 4000 instead of 6000 iters —
    the probe over-tuned the ceiling), so capability is authoritative.
    """
    import cupy as cp
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        CAP["device"] = props["name"].decode().replace("NVIDIA GeForce ", "")
        major = int(props["major"])
    except Exception:
        return
    if major <= 4 or (major == 5 and int(props.get("minor", 0)) >= 2):
        CAP["iter_ceiling_fp64"] = 20000     # Kepler/older, 1:2 fp64
        CAP["motion_wide"] = 1280
    elif major == 6 or major == 9:
        CAP["iter_ceiling_fp64"] = 20000     # Pascal & Ada workstation-class, 1:2
        CAP["motion_wide"] = 1280
    elif major == 7:
        CAP["iter_ceiling_fp64"] = 10000     # Volta — 1:2 (V100) or consumer 1:32
        CAP["motion_wide"] = 960
    else:
        # CC 8.x consumer (Ampere/Ada RTX): fp64 1:64 — measured on this
        # RTX 3050 Laptop. Iteration ceiling 6000, motion width 768.
        CAP["iter_ceiling_fp64"] = 6000
        CAP["motion_wide"] = 768


class AppState:
    def __init__(self):
        self.centerRe = FULL_SET["centerRe"]
        self.centerIm = FULL_SET["centerIm"]
        self.scale = FULL_SET["scale"]
        self.mode = 0            # 0 mandelbrot, 1 julia
        self.juliaRe = -0.8
        self.juliaIm = 0.156
        self.palette = 0
        self.ssaa = 1
        self.precision = 0       # 0 fp32, 1 fp64
        self.max_iter = 400
        self.precision_pin = None   # manual fp choice survives depth changes
        self.iter_pin = None        # manual iteration choice

    def view(self, w: int, h: int) -> dict:
        return {
            "w": w, "h": h,
            "centerRe": self.centerRe, "centerIm": self.centerIm,
            "scale": self.scale,
            "juliaRe": self.juliaRe, "juliaIm": self.juliaIm,
            "maxIter": self.max_iter,
            "mode": self.mode, "palette": self.palette,
            "interior": 0, "ssaa": self.ssaa, "precision": self.precision,
        }

    def refresh_adaptive(self):
        if self.precision_pin is None:
            want = 1 if self.scale < 1e-4 else 0
            if want != self.precision:
                self.precision = want
                self.ssaa = 1 if want else self.ssaa
        if self.iter_pin is None:
            self.max_iter = auto_iter(FULL_SET["scale"] / self.scale, self.precision)


class Renderer:
    """GL side: fullscreen-quad shader + RGB textures written by CUDA.

    The pixel buffer is mapped ONCE at allocation and stays mapped: the CUDA
    kernel writes into it directly every frame and GL reads it via
    TexSubImage2D. Per-frame map/unmap would cost ~20ms/frame in WDDM
    CUDA<->GL ownership transitions (measured); mapped-once runs the full
    path at ~2ms and is verified pixel-exact against the host path on this
    machine (formally UB per CUDA docs — re-verify after driver updates).
    """

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self._vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self._vao)

        vs = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(vs, """#version 330 core
        out vec2 v_uv;
        void main() {
          vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 0.5;
          gl_Position = vec4(p * 4.0 - 1.0, 0.0, 1.0);
          v_uv = p * 2.0;
        }""")
        gl.glCompileShader(vs)
        fs = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(fs, """#version 330 core
        in vec2 v_uv;
        out vec4 fragColor;
        uniform sampler2D u_tex;
        void main() { fragColor = vec4(texture(u_tex, v_uv).rgb, 1.0); }""")
        gl.glCompileShader(fs)
        prog = gl.glCreateProgram()
        gl.glAttachShader(prog, vs)
        gl.glAttachShader(prog, fs)
        gl.glLinkProgram(prog)
        gl.glUseProgram(prog)
        gl.glUniform1i(gl.glGetUniformLocation(prog, "u_tex"), 0)
        self._prog = prog

        self.buffers = []
        self.idx = 0
        for _ in range(2):
            pbo = gl.glGenBuffers(1)
            tex = gl.glGenTextures(1)
            self.buffers.append({"pbo": pbo, "tex": tex, "cuda": None, "out": None})
        self.nbytes = 0
        self.alloc(w, h)

    def _alloc_one(self, b, w, h):
        nbytes = w * h * 3
        if b["cuda"] is not None:
            b["cuda"].close()
        gl.glBindTexture(gl.GL_TEXTURE_2D, b["tex"])
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, b["pbo"])
        gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, nbytes, None, gl.GL_STREAM_DRAW)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        b["cuda"] = GlCudaBuffer(b["pbo"])
        ptr = b["cuda"].map()
        try:
            mem = cp.cuda.UnownedMemory(int(ptr), nbytes, None)
        except TypeError:
            mem = cp.cuda.UnownedMemory(int(ptr), nbytes, None, 0)
        b["out"] = cp.ndarray((h, w, 3), dtype=cp.uint8, memptr=cp.cuda.MemoryPointer(mem, 0))
        return nbytes

    def alloc(self, w: int, h: int):
        # Drain the pipeline BEFORE touching GL/CUDA resources: if frames
        # referencing the current PBO/texture are still in flight, unmapping
        # or deleting them can hang forever on WDDM. alloc is rare (resize /
        # adaptive-res step), so a bounded wait is fine — never a bare
        # blocking sync, which can wedge.
        try:
            stream = cp.cuda.get_current_stream()
            deadline = time.perf_counter() + 1.0
            while not stream.done() and time.perf_counter() < deadline:
                time.sleep(0.002)
            gl.glFinish()
        except Exception:
            pass
        self.w, self.h = w, h
        self.nbytes = 0
        for b in self.buffers:
            self.nbytes = self._alloc_one(b, w, h)

    def render_and_present(self, kernel, view: dict):
        """CUDA kernel writes the mapped buffer; wait for it; GL copies it to
        the texture and draws.

        Per-frame sync is the rock-solid pattern (never accumulates async
        work). glFinish is deliberately NOT called: it costs ~5ms/frame and
        is only needed before GL readbacks, not for presentation — the swap
        completes the pipeline ordering."""
        b = self.buffers[self.idx]
        fractals._launch(kernel, b["out"], view)
        cp.cuda.get_current_stream().synchronize()
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, b["tex"])
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, b["pbo"])
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.w, self.h,
                           gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        gl.glBindVertexArray(self._vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        self.idx ^= 1

    def present_last(self):
        """Redraw the most recently presented texture (idle frames)."""
        b = self.buffers[self.idx ^ 1]
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, b["tex"])
        gl.glBindVertexArray(self._vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

    def close(self):
        for b in self.buffers:
            if b["cuda"] is not None:
                b["cuda"].close()
                b["cuda"] = None


def main() -> int:
    argv = sys.argv[1:]
    selftest = "--selftest" in argv
    width, height = 1280, 720
    for a in argv:
        if a.startswith("--res="):
            width, height = (int(x) for x in a.split("=")[1].split("x"))

    if not glfw.init():
        print("ERROR: glfw.init failed", file=sys.stderr)
        return 1
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.RESIZABLE, True)
    window = glfw.create_window(width, height, "fractals — native CUDA", None, None)
    if not window:
        glfw.terminate()
        print("ERROR: could not create GLFW window", file=sys.stderr)
        return 1
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    fb_w, fb_h = glfw.get_framebuffer_size(window)

    print(f"GL: {gl.glGetString(gl.GL_RENDERER).decode()}", flush=True)

    # CUDA after the GL context exists. Warmup JIT-compiles both kernels.
    fractals.warmup()
    run_probe()
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"CUDA: {props['name'].decode()}", flush=True)

    state = AppState()
    state.refresh_adaptive()
    renderer = Renderer(fb_w, fb_h)
    glfw.set_framebuffer_size_callback(window, lambda *_: None)  # polled per-frame

    # ---- interaction state ----
    scroll_queue: list[float] = []
    zoom_vel = 0.0            # e-folds of scale per second (positive = zoom in)
    dragging = False
    drag_prev = None
    anim = None               # reset/preset fly-in
    last_render_ms = 0.0
    fps_ema = 60.0
    dirty = True
    last_input_t = 0.0

    def mark_input():
        nonlocal last_input_t
        last_input_t = time.perf_counter()

    bookmarks: list[dict | None] = [None] * 10
    pending_bookmark: str | None = None
    STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "state.json")

    def save_state():
        try:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "centerRe": state.centerRe, "centerIm": state.centerIm,
                    "scale": state.scale, "mode": state.mode,
                    "palette": state.palette, "maxIter": state.max_iter,
                    "bookmarks": [b for b in bookmarks if b],
                }, f)
        except Exception as e:
            print(f"state save failed: {e}", flush=True)

    def restore_state():
        nonlocal bookmarks
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    d = json.load(f)
                state.centerRe = float(d.get("centerRe", state.centerRe))
                state.centerIm = float(d.get("centerIm", state.centerIm))
                state.scale = float(d.get("scale", state.scale))
                state.mode = int(d.get("mode", state.mode))
                state.palette = int(d.get("palette", state.palette))
                state.max_iter = int(d.get("maxIter", state.max_iter))
                for i, b in enumerate(d.get("bookmarks", [])[:10]):
                    if b:
                        bookmarks[i] = b
                print(f"restored state: ×{format_mag(FULL_SET['scale']/state.scale)}", flush=True)
        except Exception as e:
            print(f"state restore failed: {e}", flush=True)

    restore_state()
    state.refresh_adaptive()

    def cursor_pos():
        return glfw.get_cursor_pos(window)

    def framebuffer():
        return glfw.get_framebuffer_size(window)

    def cursor_complex(px, py, w, h):
        min_re = state.centerRe - state.scale / 2.0
        step = state.scale / w
        min_im = state.centerIm - (state.scale * h / w) / 2.0
        return min_re + px * step, min_im + (h - py) * step

    def on_scroll(_win, _dx, dy):
        scroll_queue.append(dy)
        mark_input()

    def on_button(_win, button, action, _mods):
        nonlocal dragging, drag_prev
        mark_input()
        w, h = framebuffer()
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                dragging = True
                drag_prev = cursor_pos()
            elif action == glfw.RELEASE:
                dragging = False
                drag_prev = None
        elif button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
            px, py = cursor_pos()
            state.juliaRe, state.juliaIm = cursor_complex(px, py, w, h)
            state.mode = 1
            state.iter_pin = None
            state.precision_pin = None
            state.refresh_adaptive()
            nonlocal_dirty_set()

    def nonlocal_dirty_set():
        nonlocal dirty
        dirty = True

    # Preset hotkeys: jump to famous boundary coordinates so zooming ALWAYS
    # lands on real structure (the set interior is pure black; zooming there
    # shows nothing but black, which feels broken).
    PRESETS = {
        glfw.KEY_5: ("Seahorse Valley", -0.745, 0.113, 0.05),
        glfw.KEY_6: ("Elephant Valley", 0.2825, 0.005, 0.045),
        glfw.KEY_7: ("Triple Spiral", -0.088, 0.654, 0.04),
        glfw.KEY_8: ("Satellite Minibrot", -1.768, 0.0009, 0.07),
        glfw.KEY_9: ("Seahorse Deep", -0.743643887037151, 0.13182590420533, 2e-4),
        glfw.KEY_0: ("Seahorse Abyss", -0.743643887037151, 0.13182590420533, 1e-10),
    }

    def on_key(_win, key, _scancode, action, _mods):
        nonlocal anim, dirty, pending_bookmark
        if action != glfw.PRESS:
            return
        mark_input()
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_P:
            # screenshot: capture the CURRENT framebuffer to shots/<ts>.png
            try:
                w, h = framebuffer()
                gl.glFinish()
                px = gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                import numpy as _np
                from PIL import Image as _Image
                arr = _np.frombuffer(px, dtype=_np.uint8).reshape(h, w, 3)[::-1]
                os.makedirs("shots", exist_ok=True)
                fname = f"shots/fractal_{time.strftime('%Y%m%d_%H%M%S')}.png"
                _Image.fromarray(arr).save(fname)
                print(f"saved screenshot: {fname}", flush=True)
            except Exception as e:
                print(f"screenshot failed: {e}", flush=True)
        elif key in PRESETS:
            name, cre, cim, scale = PRESETS[key]
            state.precision_pin = None
            state.iter_pin = None
            anim = {"t0": time.perf_counter(), "dur": 1.2,
                    "from": {"centerRe": state.centerRe, "centerIm": state.centerIm,
                             "scale": state.scale},
                    "to": {"centerRe": cre, "centerIm": cim, "scale": scale}}
            print(f"fly to presets: {name}", flush=True)
        elif key == glfw.KEY_R:
            state.precision_pin = None
            state.iter_pin = None
            anim = {"t0": time.perf_counter(), "dur": 1.1,
                    "from": {"centerRe": state.centerRe, "centerIm": state.centerIm,
                             "scale": state.scale}}
        elif key == glfw.KEY_M:
            state.mode = 1 - state.mode
            dirty = True
        elif key == glfw.KEY_F:
            state.precision_pin = 1 - state.precision
            state.precision = state.precision_pin
            if state.precision == 1:
                state.ssaa = 1
            dirty = True
        elif key == glfw.KEY_S:
            if state.precision == 0:
                state.ssaa = 2 if state.ssaa == 1 else 1
                dirty = True
        elif glfw.KEY_1 <= key <= glfw.KEY_4:
            state.palette = key - glfw.KEY_1
            dirty = True
        elif key == glfw.KEY_UP:
            state.iter_pin = int(state.max_iter * 1.5)
            state.max_iter = state.iter_pin
            dirty = True
        elif key == glfw.KEY_DOWN:
            state.iter_pin = max(64, int(state.max_iter / 1.5))
            state.max_iter = state.iter_pin
            dirty = True
        elif key == glfw.KEY_B:
            # bookmark save: B then 0-9
            pending_bookmark = "save"
        elif key == glfw.KEY_G:
            pending_bookmark = "go"
        elif pending_bookmark and glfw.KEY_0 <= key <= glfw.KEY_9:
            slot = key - glfw.KEY_0
            if pending_bookmark == "save":
                bookmarks[slot] = {
                    "centerRe": state.centerRe, "centerIm": state.centerIm,
                    "scale": state.scale, "mode": state.mode,
                    "palette": state.palette, "maxIter": state.max_iter,
                }
                print(f"bookmark {slot} saved: ×{format_mag(FULL_SET['scale']/state.scale)}", flush=True)
            else:
                b = bookmarks[slot] if slot < len(bookmarks) else None
                if b:
                    state.precision_pin = None
                    state.iter_pin = None
                    state.mode = b["mode"]
                    state.palette = b["palette"]
                    anim = {"t0": time.perf_counter(), "dur": 1.1,
                            "from": {"centerRe": state.centerRe, "centerIm": state.centerIm,
                                     "scale": state.scale},
                            "to": {"centerRe": b["centerRe"], "centerIm": b["centerIm"],
                                   "scale": b["scale"]}}
                    print(f"fly to bookmark {slot}", flush=True)
                else:
                    print(f"bookmark {slot} empty", flush=True)
            pending_bookmark = None

    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_mouse_button_callback(window, on_button)
    glfw.set_key_callback(window, on_key)

    def render_frame(w: int, h: int, v: dict):
        nonlocal last_render_ms
        t0 = time.perf_counter()
        if (w, h) != (renderer.w, renderer.h):
            renderer.alloc(w, h)
        kernel = fractals._get_kernel(bool(state.precision))
        renderer.render_and_present(kernel, v)
        last_render_ms = (time.perf_counter() - t0) * 1000.0

    def ease(t: float) -> float:
        return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2

    # ---- selftest: render real frames through the interop path, then exit ----
    if selftest:
        w, h = framebuffer()
        timings = []
        for _ in range(30):
            t0 = time.perf_counter()
            render_frame(w, h, state.view(w, h))
            timings.append((time.perf_counter() - t0) * 1000)
            glfw.swap_buffers(window)
            glfw.poll_events()
        fp32_avg = sum(timings[5:]) / len(timings[5:])
        state.scale = 1e-5
        state.refresh_adaptive()
        deep = []
        for _ in range(8):
            t0 = time.perf_counter()
            render_frame(w, h, state.view(w, h))
            deep.append((time.perf_counter() - t0) * 1000)
            glfw.swap_buffers(window)
            glfw.poll_events()
        state.ssaa = 2
        state.refresh_adaptive()
        render_frame(w, h, state.view(w, h))  # exercise ssaa + fp64 kernel path
        glfw.swap_buffers(window)
        print("bench: sustained 120 frames...", flush=True)
        state.scale = FULL_SET["scale"]
        state.ssaa = 1
        state.refresh_adaptive()
        # sustained throughput: back-to-back frames, wall clock
        t0 = time.perf_counter()
        for _ in range(120):
            render_frame(w, h, state.view(w, h))
            glfw.swap_buffers(window)
            glfw.poll_events()
        sustained = (time.perf_counter() - t0) / 120 * 1000
        print(f"selftest: fp32 render {fp32_avg:.2f} ms/frame, "
              f"fp64@1e-5 full-res {sum(deep[2:])/len(deep[2:]):.1f} ms, "
              f"sustained {sustained:.2f} ms/frame (~{1000/sustained:.0f} fps), "
              f"precision={state.precision}, ssaa path ok")
        print("SELFTEST PASS")
        renderer.close()
        glfw.destroy_window(window)
        glfw.terminate()
        return 0

    # ---- main loop ----
    last_t = time.perf_counter()
    last_title = 0.0
    last_px = fb_w * fb_h
    motion_wide = CAP["motion_wide"]  # motion render width (full iters, LINEAR upscale)
    last_motion_at = time.perf_counter()
    try:
        while not glfw.window_should_close(window):
            dt = min(time.perf_counter() - last_t, 0.1)
            last_t = time.perf_counter()

            glfw.poll_events()
            w, h = framebuffer()
            if w == 0 or h == 0:
                continue

            # wheel -> zoom velocity (impulse), applied as a continuous glide.
            # Clamp both the impulse (trackpads burst many events per poll) and
            # the per-frame step so a fast scroll never dives straight to the
            # zoom floor (which renders as a black wall past fp64's limit).
            if scroll_queue:
                impulse = 2.4 * sum(y for y in scroll_queue)
                impulse = max(-6.0, min(6.0, impulse))
                zoom_vel = max(-8.0, min(8.0, zoom_vel + impulse))
                scroll_queue.clear()
            if abs(zoom_vel) > 1e-3:
                px, py = cursor_pos()
                cre, cim = cursor_complex(px, py, w, h)
                step = math.exp(max(-0.35, min(0.35, -zoom_vel * dt)))
                new_scale = min(MAX_SCALE, max(min_scale(), state.scale * step))
                if new_scale != state.scale:
                    k = new_scale / state.scale
                    state.centerRe = cre - (cre - state.centerRe) * k
                    state.centerIm = cim - (cim - state.centerIm) * k
                    state.scale = new_scale
                    dirty = True
                zoom_vel *= math.exp(-7.0 * dt)

            # drag pan
            if dragging and drag_prev is not None:
                mx, my = cursor_pos()
                dx, dy = mx - drag_prev[0], my - drag_prev[1]
                drag_prev = (mx, my)
                if dx or dy:
                    state.centerRe -= dx * state.scale / w
                    state.centerIm += dy * state.scale / w
                    dirty = True

            # reset/preset fly-in: keyframe ease from wherever we are to anim["to"]
            if anim is not None:
                t = min(1.0, (time.perf_counter() - anim["t0"]) / anim["dur"])
                e = ease(t)
                f = anim["from"]
                g = anim.get("to", FULL_SET)
                log_s = math.log(f["scale"]) + (math.log(g["scale"]) - math.log(f["scale"])) * e
                state.scale = min(MAX_SCALE, max(min_scale(), math.exp(log_s)))
                state.centerRe = f["centerRe"] + (g["centerRe"] - f["centerRe"]) * e
                state.centerIm = f["centerIm"] + (g["centerIm"] - f["centerIm"]) * e
                state.refresh_adaptive()
                dirty = True
                if t >= 1.0:
                    anim = None

            state.refresh_adaptive()

            # ---- render: full-iter, adaptive motion width, sharpens on idle ----
            # The WDDM present path has a hard cost cliff vs render width
            # (measured): 640px = 21ms, 768px = 23ms, 960px = 34ms, 1080p =
            # 56ms @ full fp64 iterations. So during motion we render at
            # motionWide (FULL iterations — no black voids, real math, LINEAR
            # upscale = soft but never pixel-blocky). When the user stops
            # >0.4s, one full-res pass sharpens it for the still frame.
            if dirty:
                dirty = False
                mw = min(w, motion_wide)
                mh = max(2, round(h * (mw / w)))
                v = state.view(mw, mh)
                render_frame(mw, mh, v)
                last_px = mw * mh
                last_motion_at = time.perf_counter()
            elif (
                time.perf_counter() - last_motion_at > 0.4
                and (renderer.w != w or renderer.h != h)
            ):
                # settle sharpen: one full-res frame (56ms @ 720p fp64 — at
                # most a 2-3 frame hitch after motion stops, then crisp)
                v = state.view(w, h)
                render_frame(w, h, v)
                last_px = w * h
                last_motion_at = time.perf_counter()  # re-arm
            else:
                renderer.present_last()  # keep vsync present while idle

            # Pace the loop to ~62fps; swap doesn't reliably block on WDDM.
            elapsed = time.perf_counter() - last_t
            if elapsed < 0.016:
                time.sleep(0.016 - elapsed)

            now = time.perf_counter()
            if now - last_title > 0.25:
                last_title = now
                # Cost model for the HUD only (WDDM present floor + kernel).
                # Adaptive resolution is handled by motion_wide — render
                # size never changes dynamically, so nothing can go blocky.
                mw = min(w, motion_wide)
                mh = max(2, round(h * (mw / w)))
                pred = predict_ms(state.precision, mw * mh, state.max_iter, mw)
                loop_fps = 1.0 / max(time.perf_counter() - last_t, 1e-6)
                fps_ema = fps_ema * 0.7 + min(loop_fps, 62) * 0.3
                mag = FULL_SET["scale"] / state.scale
                prec_name = 'fp64' if state.precision else 'fp32'
                limit_hint = " ⚠ DOUBLE PRECISION LIMIT — can't go deeper" if state.scale <= min_scale() * 1.01 else ""
                glfw.set_window_title(window, (
                    f"fractals — {'Julia' if state.mode else 'Mandelbrot'} · "
                    f"×{format_mag(mag)} · {prec_name} · "
                    f"iter {state.max_iter} · gpu ~{pred:.1f} ms · "
                    f"5-9 presets | B/G+0-9 bookmarks | P shot"
                    f"{limit_hint}"))

            glfw.swap_buffers(window)
    finally:
        save_state()
        renderer.close()
        glfw.destroy_window(window)
        glfw.terminate()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CudaGLError as e:
        print(f"\nCUDA-OpenGL interop failed: {e}", file=sys.stderr)
        print("This usually means the GL context landed on the integrated GPU.", file=sys.stderr)
        if prefer_discrete_gpu():
            exe = sys_executable()
            print(f"Set GpuPreference=2 for {exe} in the registry.", file=sys.stderr)
            print("Re-run the same command — it should now use the NVIDIA GPU.", file=sys.stderr)
        sys.exit(2)
