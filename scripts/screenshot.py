import sys
import numpy as np
import glfw
from OpenGL import GL as gl
sys.path.insert(0, '.')
from native.app import AppState, Renderer, CAP
from server import fractals
from PIL import Image

glfw.init()
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
W, H = 1280, 720
win = glfw.create_window(W, H, 'shot', None, None)
glfw.make_context_current(win)
gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
fractals.warmup()
r = Renderer(W, H)

# hero shot: Seahorse Valley with palette 0
st = AppState()
st.centerRe = -0.745
st.centerIm = 0.113
st.scale = 0.05
st.palette = 0
st.refresh_adaptive()
v = st.view(W, H)
r.render_and_present(fractals._get_kernel(bool(st.precision)), v)
gl.glFinish()
p = gl.glReadPixels(0, 0, W, H, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
a = np.frombuffer(p, dtype=np.uint8).reshape(H, W, 3)[::-1]
Image.fromarray(a).save(r'C:/Users/hamme/fractals-with-cuda/docs_hero.png')
print('hero saved', CAP['device'], flush=True)

# deep zoom shot: Seahorse Deep fp64
st2 = AppState()
st2.centerRe = -0.743643887037151
st2.centerIm = 0.13182590420533
st2.scale = 0.0002
st2.palette = 0
st2.precision = 1
st2.refresh_adaptive()
v2 = st2.view(W, H)
r.render_and_present(fractals._get_kernel(True), v2)
gl.glFinish()
p2 = gl.glReadPixels(0, 0, W, H, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
a2 = np.frombuffer(p2, dtype=np.uint8).reshape(H, W, 3)[::-1]
Image.fromarray(a2).save(r'C:/Users/hamme/fractals-with-cuda/docs_deep.png')
print('deep saved', flush=True)

r.close()
glfw.destroy_window(win)
glfw.terminate()
