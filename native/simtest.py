"""Interactive simulation of the real native render loop.

Same GL context / Renderer / kernel path as native.app, but input is scripted
(wheel zoom glides, drag pans, deep crossing into fp64) and the loop is
instrumented for hitches and adaptive-res decisions. Ends by reading the
texture back and comparing to the host-reference of the final view.

Run: .venv\\Scripts\\python.exe -m native.simtest
"""
import math
import os
import subprocess
import sys
import time

import numpy as np
import glfw
from OpenGL import GL as gl

import cupy as cp

from native.app import (AppState, Renderer, FULL_SET, MIN_SCALE, MAX_SCALE,
                        auto_iter, calibrate_probes, predict_ms, format_mag)
from native.interop import GlCudaBuffer
from server import fractals

W, H = 1280, 720


import os
DRY_INTERVAL = float(os.environ.get("DRY_INTERVAL", "1.0"))


class Sim:
    def __init__(self):
        self.state = AppState()
        self.zoom_vel = 0.0
        self.dragging = False
        self.drag_prev = None
        self.scroll_queue = []

    def cursor_pos(self):
        return W / 2.0, H / 2.0

    def cursor_complex(self, px, py):
        min_re = self.state.centerRe - self.state.scale / 2.0
        step = self.state.scale / W
        min_im = self.state.centerIm - (self.state.scale * H / W) / 2.0
        return min_re + px * step, min_im + (H - py) * step

    def on_scroll(self, _win, _dx, dy):
        self.scroll_queue.append(dy)

    def on_button(self, _win, button, action, _mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.dragging = action == glfw.PRESS
            self.drag_prev = None if action == glfw.RELEASE else self.cursor_pos()
        elif button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
            px, py = self.cursor_pos()
            self.state.juliaRe, self.state.juliaIm = self.cursor_complex(px, py)
            self.state.mode = 1

    def step(self, dt, dirty_was):
        """One loop iteration (same math as main()). Returns dirty."""
        dirty = dirty_was
        if self.scroll_queue:
            for y in self.scroll_queue:
                self.zoom_vel += 2.4 * y
            self.scroll_queue.clear()
        if abs(self.zoom_vel) > 1e-3:
            px, py = self.cursor_pos()
            cre, cim = self.cursor_complex(px, py)
            new_scale = min(MAX_SCALE, max(MIN_SCALE, self.state.scale * math.exp(-self.zoom_vel * dt)))
            if new_scale != self.state.scale:
                k = new_scale / self.state.scale
                self.state.centerRe = cre - (cre - self.state.centerRe) * k
                self.state.centerIm = cim - (cim - self.state.centerIm) * k
                self.state.scale = new_scale
                dirty = True
            self.zoom_vel *= math.exp(-7.0 * dt)
        if self.dragging and self.drag_prev is not None:
            mx, my = self.cursor_pos()
            dx, dy = mx - self.drag_prev[0], my - self.drag_prev[1]
            self.drag_prev = (mx, my)
            if dx or dy:
                self.state.centerRe -= dx * self.state.scale / W
                self.state.centerIm += dy * self.state.scale / W
                dirty = True
        self.state.refresh_adaptive()
        return dirty


def make_gl():
    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    win = glfw.create_window(W, H, "simtest", None, None)
    glfw.make_context_current(win)
    glfw.swap_interval(1)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    return win


def main():
    win = make_gl()
    fractals.warmup()
    calibrate_probes()
    renderer = Renderer(W, H)
    sim = Sim()
    print(f"probes: fp32 {predict_ms(0, W*H, 400):.2f} ms @ full-res/400it, "
          f"fp64 {predict_ms(1, W*H, 400):.2f} ms", flush=True)

    res_factor = 1.0
    dirty = False
    last_px = W * H
    last_title = time.perf_counter()
    last_t = time.perf_counter()
    last_dog_purge = time.perf_counter()
    timeline = []
    hitches = []
    dirty_count = 0
    drag_x = 0.0

    def decide_res():
        nonlocal res_factor, dirty, last_px
        pred = predict_ms(sim.state.precision, last_px, sim.state.max_iter)
        prev = res_factor
        if pred > 24.0 and res_factor > 0.25:
            res_factor = max(0.25, res_factor * 0.75)
            dirty = True
        elif pred < 8.0 and res_factor < 1.0 and not sim_moving:
            res_factor = min(1.0, res_factor / 0.75)
            dirty = True
        if res_factor != prev:
            print(f"  res {prev:.2f}->{res_factor:.2f} (pred {pred:.1f}ms, {sim.state.max_iter} iter, prec {sim.state.precision})", flush=True)

    def queued_guard():
        # count of renders since last adapt happened implicitly through pacing
        pass

    sim_moving = False

    phases = [
        ("idle", 1.0),
        ("zoom-x60", 2.6),
        ("drag", 1.2),
        ("deep-fp64", 2.8),
    ]
    t0_real = time.perf_counter()
    elapsed_real = 0.0
    phase_elapsed = 0.0
    phase_i = 0
    phase_name = phases[0][0]
    phase_dur = phases[0][1]
    wheel_ticks = 0

    try:
        started = False
        while phase_i < len(phases):
            now_r = time.perf_counter()
            dt = min(now_r - last_t, 0.1)
            last_t = now_r
            glfw.poll_events()
            elapsed_real = now_r - t0_real

            if phase_name == "idle" and not started:
                started = True
                print(f"sim start: probe fp32={predict_ms(0, W*H, 400):.2f}ms fp64={predict_ms(1, W*H, 400):.2f}ms", flush=True)
            if phase_name == "zoom-x60" and wheel_ticks == 0:
                print("phase: zoom-x60 (60 wheel ticks)", flush=True)
            if phase_name == "drag" and sim.drag_prev is None:
                print("phase: drag", flush=True)
            if phase_name == "deep-fp64" and phase_elapsed == 0.0:
                print("phase: deep-fp64 (checking stream after jump)", flush=True)
                # bounded probe: is the stream already wedged right after jump?
                import threading
                done = threading.Event()
                q = {}
                def probe():
                    try:
                        cp.cuda.get_current_stream().synchronize()
                        q["ok"] = True
                    except Exception as e:
                        q["err"] = e
                    done.set()
                th = threading.Thread(target=probe, daemon=True); th.start()
                done.wait(6.0)
                print(f"  stream after jump: {'OK' if q.get('ok') else 'WEDGED'}", flush=True)

            if phase_name == "zoom-x60" and wheel_ticks < 60:
                if wheel_ticks % 3 == 0:
                    sim.on_scroll(None, 0, 1)
                wheel_ticks += 1
            if phase_name == "drag" and sim.drag_prev is None:
                sim.on_button(None, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0)
            if phase_name == "drag":
                drag_x += 6.0
                sim.drag_prev = (W/2 + drag_x, H/2)
                sim.dragging = True
                dirty = True

            # main-loop math
            w, h = W, H
            dirty = sim.step(dt, dirty)

            if dirty:
                dirty = False
                rw = max(2, round(w * res_factor))
                rh = max(2, round(h * res_factor))
                renderer.render_and_present(fractals._get_kernel(bool(sim.state.precision)),
                                            sim.state.view(rw, rh))
                last_px = rw * rh
                dirty_count += 1
            else:
                renderer.present_last()
            glfw.swap_buffers(win)

            # pacing
            el = time.perf_counter() - last_t
            if el < 0.016:
                time.sleep(0.016 - el)

            # title-time adaptation logic (no syncs!)
            now_t = time.perf_counter()
            if now_t - last_title > 0.25:
                last_title = now_t
                sim_moving = sim.zoom_vel > 1e-3 or sim.dragging
                decide_res()
                mag = FULL_SET["scale"] / sim.state.scale
                timeline.append({
                    "t": round(elapsed_real, 2),
                    "phase": phase_name,
                    "mag": format_mag(mag),
                    "prec": sim.state.precision,
                    "res": round(res_factor, 2),
                    "iter": sim.state.max_iter,
                })

            # hitch detection: any loop iteration way over pacing budget
            if dt > 0.03:
                hitches.append((round(elapsed_real, 2), round(dt * 1000, 1), phase_name))

            # MAIN-THREAD periodic drain (blocking, not watchdogged): the
            # WDDM interop queue wedges after ~2s of never-synced frames;
            # this retires it. Sync cost ~20ms on this machine.
            if phase_name != "deep-fp64" and now_r - last_dog_purge > DRY_INTERVAL:
                last_dog_purge = now_r
                cp.cuda.get_current_stream().synchronize()
                gl.glFinish()

            phase_elapsed += dt
            if phase_elapsed >= phase_dur:
                phase_i += 1
                if phase_i < len(phases):
                    phase_name, phase_dur = phases[phase_i]
                    phase_elapsed = 0.0
                    if phase_name == "deep-fp64":
                        # jump deep so fp64 kicks in without 60 slow zoom steps
                        sim.state.scale = 1e-4 * 2
                        if "deep-fp64" == phase_name:
                            sim.state.centerRe = -0.745
                            sim.state.centerIm = 0.113
                        sim.state.precision_pin = None
                        sim.state.iter_pin = None
                        sim.state.refresh_adaptive()
                        dirty = True

        # ---- final correctness: subprocess verification ----
        # Re-verifying in a second context in-process poisons the driver
        # (glfw re-init after terminate leaves CUDA-GL state wedged), so the
        # fresh check runs in a clean subprocess.
        final_view = dict(sim.state.view(W, H))
        import json

        def watchdog_sync(timeout=8.0):
            import threading
            done = threading.Event()
            out = {}
            def work():
                try:
                    cp.cuda.get_current_stream().synchronize()
                    out["ok"] = True
                except Exception as e:
                    out["err"] = e
                finally:
                    done.set()
            t = threading.Thread(target=work, daemon=True)
            t.start()
            done.wait(timeout)
            return out.get("ok", False)

        print("final-verify: sim-context wedged-stream check (watchdog)", flush=True)
        stream_ok = watchdog_sync()

        w, h = W, H
        renderer.close()
        glfw.destroy_window(win)
        glfw.terminate()

        if not stream_ok:
            print("final-verify: sim context WEDGED despite 1s drains", flush=True)
            print("SIMTEST FAIL (wedged)", flush=True)
            return 1

        print("final-verify: subprocess pixel check...", flush=True)
        payload = json.dumps({
            "view": final_view,
        })
        sub = subprocess.Popen([sys.executable, "-m", "native.verifyframe", payload],
                               cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        rc = sub.wait(timeout=60)
        sub_ok = (rc == 0)
        diff = "n/a (subprocess)"
        print("timeline:", flush=True)
        for row in timeline:
            print(" ", row, flush=True)
        print(f"dirty renders: {dirty_count}", flush=True)
        print(f"sim-context sync {'OK' if stream_ok else 'WEDGED'}", flush=True)
        # PASS rules: no wedge, pixels exact, res stayed full during shallow
        # (no unjustified pixelation). Decided hitches (>150ms) count; the
        # accepted 1Hz drains land well under 150ms in normal use.
        big_hitches = [hh for hh in hitches if hh[1] > 150.0]
        shallow_res_ok = all(
            r["res"] == 1.0 for r in timeline
            if r["phase"] in ("idle", "zoom-x60", "drag")
        )
        print(f"final res_factor={res_factor}, dirty renders={dirty_count}, "
              f"hitches={len(hitches)} (big={len(big_hitches)}), "
              f"pixel check {'PASS' if sub_ok else 'FAIL'}, "
              f"shallow res full: {shallow_res_ok}", flush=True)
        for hh in big_hitches[:10]:
            print("  big hitch:", hh, flush=True)
        ok = stream_ok and sub_ok and shallow_res_ok and len(big_hitches) == 0
        print("SIMTEST PASS" if ok else "SIMTEST FAIL", flush=True)
    except Exception:
        raise
    finally:
        try:
            renderer.close()
        except Exception:
            pass
        try:
            glfw.terminate()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
