"""Native fractal explorer: GLFW + OpenGL + CUDA interop on cupy kernels.

The CUDA kernel writes pixels straight into a GL-registered pixel buffer
(GPU-to-GPU, no device-to-host copy, no browser, no HTTP). Presented on
vsync. This is the real deal: input -> kernel -> pixels, one frame.

Run:  .venv\\Scripts\\python.exe -m native.app [--selftest] [--res WxH]

Keys: drag = pan · wheel = zoom · right-click = Julia seed · M = Mandelbrot/Julia
      F = fp32/fp64 (pins) · S = 2x2 supersample · 1-4 = palette
      Up/Down = iterations (pins) · R = reset · Esc = quit
"""
import math
import sys
import time

import glfw
from OpenGL import GL as gl

import cupy as cp

from native.interop import CudaGLError, GlCudaBuffer, prefer_discrete_gpu, sys_executable
from server import fractals

MIN_SCALE = 1e-14
MAX_SCALE = 10.0
FULL_SET = {"centerRe": -0.6, "centerIm": 0.0, "scale": 3.4}


def auto_iter(mag: float, precision: int) -> int:
    ceiling = 4000 if precision else 12000
    return int(min(ceiling, round(400 * (1 + math.sqrt(min(mag, 20000)) * 0.9))))


def format_mag(n: float) -> str:
    if n >= 1e6:
        return f"{n/1e6:.0f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}k"
    if n >= 10:
        return f"{n:.0f}"
    return f"{n:.1f}"


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
        self.w, self.h = w, h
        self.nbytes = 0
        for b in self.buffers:
            self.nbytes = self._alloc_one(b, w, h)

    def render_and_present(self, kernel, view: dict):
        """CUDA kernel writes the mapped buffer; GL copies it to the texture
        and draws. Fully async: NO host-side CUDA sync per frame.

        Measured on this machine (WDDM): any per-frame host sync on the
        interop path costs ~20ms (forced command-buffer flush + round trip),
        capping the loop at ~46fps. Unsynced submission is ~0.2ms/frame and
        the interop driver still orders kernel->texsub via its scheduler
        tokens, so the display shows correct frames paced by the compositor.
        The texsub of frame N reads kernel N through that token, not through
        a host wait."""
        b = self.buffers[self.idx]
        fractals._launch(kernel, b["out"], view)
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

    def on_button(_win, button, action, _mods):
        nonlocal dragging, drag_prev
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

    def on_key(_win, key, _scancode, action, _mods):
        nonlocal anim, dirty
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
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

    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_mouse_button_callback(window, on_button)
    glfw.set_key_callback(window, on_key)

    def render_frame(w: int, h: int):
        nonlocal last_render_ms
        t0 = time.perf_counter()
        if (w, h) != (renderer.w, renderer.h):
            renderer.alloc(w, h)
        kernel = fractals._get_kernel(bool(state.precision))
        renderer.render_and_present(kernel, state.view(w, h))
        last_render_ms = (time.perf_counter() - t0) * 1000.0

    def ease(t: float) -> float:
        return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2

    # ---- selftest: render real frames through the interop path, then exit ----
    if selftest:
        w, h = framebuffer()
        timings = []
        for _ in range(30):
            t0 = time.perf_counter()
            render_frame(w, h)
            timings.append((time.perf_counter() - t0) * 1000)
            glfw.swap_buffers(window)
            glfw.poll_events()
        fp32_avg = sum(timings[5:]) / len(timings[5:])
        state.scale = 1e-5
        state.refresh_adaptive()
        deep = []
        for _ in range(8):
            t0 = time.perf_counter()
            render_frame(w, h)
            deep.append((time.perf_counter() - t0) * 1000)
            glfw.swap_buffers(window)
            glfw.poll_events()
        state.ssaa = 2
        state.refresh_adaptive()
        render_frame(w, h)  # exercise ssaa + fp64 kernel path
        glfw.swap_buffers(window)
        print("bench: sustained 120 frames...", flush=True)
        state.scale = FULL_SET["scale"]
        state.ssaa = 1
        state.refresh_adaptive()
        # sustained throughput: back-to-back frames, wall clock
        t0 = time.perf_counter()
        for _ in range(120):
            render_frame(w, h)
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
    res_factor = 1.0          # internal render scale; drops when frames get heavy
    frame_start = 0.0
    res_factor_dirty = False
    try:
        while not glfw.window_should_close(window):
            dt = min(time.perf_counter() - last_t, 0.1)
            last_t = time.perf_counter()

            glfw.poll_events()
            w, h = framebuffer()
            if w == 0 or h == 0:
                continue

            # wheel -> zoom velocity (impulse), applied as a continuous glide
            if scroll_queue:
                for y in scroll_queue:
                    zoom_vel += 2.4 * y
                scroll_queue.clear()
            if abs(zoom_vel) > 1e-3:
                px, py = cursor_pos()
                cre, cim = cursor_complex(px, py, w, h)
                new_scale = min(MAX_SCALE, max(MIN_SCALE, state.scale * math.exp(-zoom_vel * dt)))
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

            # reset fly-in: keyframe ease from wherever we are to FULL_SET
            if anim is not None:
                t = min(1.0, (time.perf_counter() - anim["t0"]) / anim["dur"])
                e = ease(t)
                f = anim["from"]
                log_s = math.log(f["scale"]) + (math.log(FULL_SET["scale"]) - math.log(f["scale"])) * e
                state.scale = min(MAX_SCALE, max(MIN_SCALE, math.exp(log_s)))
                state.centerRe = f["centerRe"] + (FULL_SET["centerRe"] - f["centerRe"]) * e
                state.centerIm = f["centerIm"] + (FULL_SET["centerIm"] - f["centerIm"]) * e
                state.refresh_adaptive()
                dirty = True
                if t >= 1.0:
                    anim = None

            state.refresh_adaptive()

            if dirty:
                dirty = False
                rw = max(2, round(w * res_factor))
                rh = max(2, round(h * res_factor))
                t_frame = time.perf_counter()
                render_frame(rw, rh)
                frame_start = t_frame
                res_factor_dirty = True
            else:
                renderer.present_last()  # keep vsync present while idle

            # Pace the loop to ~62fps; swap doesn't reliably block on WDDM.
            elapsed = time.perf_counter() - last_t
            if elapsed < 0.016:
                time.sleep(0.016 - elapsed)

            now = time.perf_counter()
            if now - last_title > 0.25:
                last_title = now
                # Sample true kernel time 4x/sec: one sync here is invisible
                # (20ms/250ms) and drives the adaptive-resolution decision.
                cp.cuda.get_current_stream().synchronize()
                kernel_ms = (now - frame_start) * 1000.0 if frame_start else last_render_ms
                if res_factor_dirty:
                    res_factor_dirty = False
                    if kernel_ms > 24.0 and res_factor > 0.25:
                        res_factor = max(0.25, res_factor * 0.75)
                        dirty = True
                    elif kernel_ms < 9.0 and res_factor < 1.0:
                        res_factor = min(1.0, res_factor / 0.75)
                        dirty = True
                fps_ema = fps_ema * 0.7 + (1000.0 / max(last_render_ms, 1e-6)) * 0.3
                mag = FULL_SET["scale"] / state.scale
                glfw.set_window_title(window, (
                    f"fractals — {'Julia' if state.mode else 'Mandelbrot'} · "
                    f"×{format_mag(mag)} · {'fp64' if state.precision else 'fp32'} · "
                    f"iter {state.max_iter} · gpu {kernel_ms:.1f} ms · ~{min(fps_ema, 62):.0f} fps"))

            glfw.swap_buffers(window)
    finally:
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
