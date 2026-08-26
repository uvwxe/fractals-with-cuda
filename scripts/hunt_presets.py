"""Round 2: satellite minibrot hunt + scepter refinement."""
import sys
import numpy as np
from PIL import Image
sys.path.insert(0, ".")
from server.fractals import render_rgba, warmup

warmup()
W, H = 240, 150

def black_pct(view):
    raw = render_rgba(view)
    img = np.asarray(Image.frombytes("RGB", (W, H), raw)).astype(np.int16)
    lum = img.sum(axis=2) / 3
    return (lum == 0).mean() * 100

base = dict(w=W, h=H, mode=0, juliaRe=0.0, juliaIm=0.0, palette=0, interior=0, ssaa=1, precision=1, maxIter=1500)

print("== Satellite Minibrot (period-3, real axis) ==")
for re, scale, it, prec, label in [
    (-1.754877666, 8.5e-4, 1500, 0, "span 8.5e-4 fp32"),
    (-1.754877666, 2e-4, 1500, 0, "span 2e-4 fp32"),
    (-1.754877666, 5e-5, 2000, 0, "span 5e-5 fp32"),
    (-1.754877666, 2e-5, 2000, 0, "span 2e-5 fp32"),
]:
    b = black_pct({**base, "centerRe": re, "centerIm": 0.0, "scale": scale, "maxIter": it, "precision": prec})
    print(f"  (-1.754877666, 0) {label}: black {b:5.1f}%")

print("== Scepter refinement ==")
for im, scale in [(0.03, 0.02), (0.035, 0.025), (0.04, 0.02), (0.045, 0.03)]:
    b = black_pct({**base, "centerRe": -1.36, "centerIm": im, "scale": scale, "maxIter": 600})
    print(f"  (-1.36, {im}) span {scale}: black {b:5.1f}%")
