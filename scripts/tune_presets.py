"""Tune suspicious presets: sweep scale/iter, report interior% and detail."""
import sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, ".")
from server.fractals import render_rgba, warmup

warmup()
W, H = 800, 450

def probe(name, view):
    t0 = time.perf_counter()
    raw = render_rgba(view)
    ms = (time.perf_counter() - t0) * 1000
    img = np.asarray(Image.frombytes("RGB", (W, H), raw)).astype(np.int16)
    lum = img.sum(axis=2) / 3
    black = (lum == 0).mean() * 100
    grad = np.abs(np.diff(lum, axis=1)).mean() + np.abs(np.diff(lum, axis=0)).mean()
    print(f"{name:48s} black={black:5.1f}% grad={grad:4.1f} {ms:6.1f}ms")

base = dict(w=W, h=H, mode=0, juliaRe=0.0, juliaIm=0.0, palette=0, interior=0, ssaa=1, precision=1)

# Mini Mandelbrot: is the minibrot in frame? sweep scale
for scale in (0.00002, 0.000005, 0.000003):
    for it in (2000, 4000):
        probe(f"MiniM scale={scale} iter={it}",
              {**base, "centerRe": -1.768778833, "centerIm": 0.00173894, "scale": scale, "maxIter": it})

# Double Spiral: sweep scale
for scale in (0.00004, 0.00001, 0.000004):
    probe(f"DoubleSpiral scale={scale}",
          {**base, "centerRe": -0.7454, "centerIm": 0.113, "scale": scale, "maxIter": 2000})

# Scepter: less interior
for scale in (0.04, 0.01, 0.004):
    probe(f"Scepter scale={scale}",
          {**base, "centerRe": -1.36, "centerIm": 0.0, "scale": scale, "maxIter": 600})
