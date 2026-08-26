"""Render every preset to PNG for visual verification (dev tool)."""
import glob
import os
import sys
import time
from PIL import Image

for stale in glob.glob("preview_*.png"):
    os.remove(stale)

sys.path.insert(0, ".")
from server.fractals import render_rgba, warmup, FULL_SET, DEEP_ZOOM_PRESETS, JULIA_PRESETS
from server.palettes import PALETTES

warmup()

W, H = 1280, 720

def save(name, view):
    t0 = time.perf_counter()
    raw = render_rgba(view)
    ms = (time.perf_counter() - t0) * 1000
    img = Image.frombytes("RGB", (W, H), raw)
    img.save(f"preview_{name}.png")
    print(f"{name}: {ms:.1f} ms")

base = dict(w=W, h=H, mode=0, juliaRe=-0.8, juliaIm=0.156, palette=0, interior=0, quality=1, precision=0, ssaa=1)
save("fullset_thermal", {**base, **FULL_SET})

for i, p in enumerate(DEEP_ZOOM_PRESETS):
    v = dict(base, **p)
    v.setdefault("precision", 0)
    save(f"deep_{i}_{p['name'].replace(' ','')}", v)

for i, p in enumerate(JULIA_PRESETS):
    save(f"julia_{i}_{p['name'].replace(' ','')}", {**base, "mode": 1, "centerRe": 0.0, "centerIm": 0.0, "scale": 3.4, "juliaRe": p["re"], "juliaIm": p["im"], "maxIter": 400})

for pal in PALETTES:
    save(f"fullset_pal{pal['id']}", {**base, **FULL_SET, "palette": pal["id"]})

# fp64 sanity + 2x2 in-kernel SSAA timing
save("fullset_fp64", {**base, **FULL_SET, "precision": 1, "maxIter": 800})
save("fullset_2xaa", {**base, **FULL_SET, "ssaa": 2})
print("DONE")
