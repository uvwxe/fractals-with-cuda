"""Objective render verification: stats that catch degenerate/broken renders."""
import glob
import numpy as np
from PIL import Image

for path in sorted(glob.glob("preview_*.png")):
    img = np.asarray(Image.open(path)).astype(np.int16)
    h, w, _ = img.shape
    lum = img.sum(axis=2) / 3
    interior = (lum == 0).mean() * 100
    ncolors = len(np.unique(img.reshape(-1, 3), axis=0))
    mean_lum = lum.mean()
    # detail: mean absolute gradient
    gx = np.abs(np.diff(lum, axis=1)).mean()
    gy = np.abs(np.diff(lum, axis=0)).mean()
    # color spread
    rng = (img.max(axis=(0, 1)) - img.min(axis=(0, 1))).tolist()
    verdict = "OK"
    if mean_lum < 1: verdict = "DEGENERATE: all black"
    elif ncolors < 50: verdict = "SUSPECT: banded/flat"
    print(f"{path.split('preview_')[1]:38s} black={interior:5.1f}% colors={ncolors:6d} meanlum={mean_lum:5.1f} grad={gx+gy:5.1f} range={rng} {verdict}")
