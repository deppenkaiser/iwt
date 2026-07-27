#!/usr/bin/env python3
import os
import sys
import glob
import numpy as np
from PIL import Image

# ============================================================
# PFADE
# ============================================================
BIN_DIR = os.path.expanduser("~/Dokumente/daten/sandbox/apps/iwt/build/bin/")

def analyze_gif(gif_path):
    """Analysiert ein GIF und gibt Statistiken aus."""
    print(f"\n=== {os.path.basename(gif_path)} ===")

    # GIF laden
    img = Image.open(gif_path)
    frames = []
    while True:
        frame = np.array(img.convert('L'))
        frames.append(frame)
        try:
            img.seek(img.tell() + 1)
        except EOFError:
            break

    print(f"  Frames: {len(frames)}")
    print(f"  Größe: {frames[0].shape}")

    # Statistiken über alle Frames
    all_pixels = np.concatenate([f.flatten() for f in frames])
    print(f"  Min: {all_pixels.min()}")
    print(f"  Max: {all_pixels.max()}")
    print(f"  Mittelwert: {all_pixels.mean():.2f}")
    print(f"  Std: {all_pixels.std():.2f}")

    # Hotspots pro Frame
    hotspots_per_frame = []
    for f in frames:
        hotspots = np.sum(f > 200)
        hotspots_per_frame.append(hotspots)

    print(f"  Hotspots pro Frame (Mittelwert): {np.mean(hotspots_per_frame):.2f}")
    print(f"  Hotspots pro Frame (Std): {np.std(hotspots_per_frame):.2f}")
    print(f"  Hotspots pro Frame (Min): {np.min(hotspots_per_frame)}")
    print(f"  Hotspots pro Frame (Max): {np.max(hotspots_per_frame)}")

    # Stabilität: Überlappung zwischen erstem und letztem Frame
    first = frames[0] > 200
    last = frames[-1] > 200
    overlap = np.sum(first & last)
    print(f"  Überlappung (erster ↔ letzter Frame): {overlap}")

    # Bewegung: Schwerpunkt über die Zeit
    centers = []
    for f in frames:
        y, x = np.where(f > 200)
        if len(x) > 0:
            cx = np.mean(x)
            cy = np.mean(y)
            centers.append((cx, cy))
        else:
            centers.append((None, None))

    valid_centers = [c for c in centers if c[0] is not None]
    if len(valid_centers) > 1:
        # Durchschnittliche Bewegung
        movements = []
        for i in range(1, len(valid_centers)):
            dx = valid_centers[i][0] - valid_centers[i-1][0]
            dy = valid_centers[i][1] - valid_centers[i-1][1]
            movements.append(np.sqrt(dx*dx + dy*dy))
        print(f"  Mittlere Bewegung pro Frame: {np.mean(movements):.2f} Pixel")
        print(f"  Std der Bewegung: {np.std(movements):.2f} Pixel")
    else:
        print("  Keine Bewegung messbar (keine Hotspots)")

def main():
    print("GIF-Analyse")
    print("=" * 70)

    gifs = sorted(glob.glob(os.path.join(BIN_DIR, "lauf_*.gif")))
    if not gifs:
        print("Keine GIFs gefunden in:", BIN_DIR)
        return

    print(f"Gefundene GIFs: {len(gifs)}")
    for g in gifs:
        analyze_gif(g)

if __name__ == "__main__":
    main()