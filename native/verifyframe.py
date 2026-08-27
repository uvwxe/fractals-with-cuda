"""Subprocess pixel verification: render a view through the real GL interop
path in a clean process, read the texture back, compare against the host
render. Exit 0 = pixel-exact, 1 = mismatch. Used by native.simtest.

Run: python -m native.verifyframe '<view-json>'
"""
import json
import sys

import numpy as np
import glfw
from OpenGL import GL as gl

import cupy as cp

from native.app import Renderer, calibrate_probes
from server import fractals

W, H = 320, 200


def main() -> int:
    view = json.loads(sys.argv[1])
    view = view.get("view", view)  # allow {"view": {...}} envelopes or raw dict
    view["w"], view["h"] = W, H

    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    win = glfw.create_window(W, H, "verifyframe", None, None)
    glfw.make_context_current(win)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

    fractals.warmup()
    calibrate_probes()
    r = Renderer(W, H)
    r.render_and_present(fractals._get_kernel(bool(view["precision"])), view)
    cp.cuda.get_current_stream().synchronize()
    gl.glFinish()

    got = np.frombuffer(
        gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE),
        dtype=np.uint8).reshape(H, W, 3)
    ref = np.frombuffer(fractals.render_rgba(view), dtype=np.uint8).reshape(H, W, 3)
    diff = int(np.abs(got.astype(int) - ref.astype(int)).max())
    print(f"verifyframe: max diff = {diff}")
    r.close()
    glfw.destroy_window(win)
    glfw.terminate()
    return 0 if diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
