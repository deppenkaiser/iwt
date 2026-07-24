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
OUTPUT_DIR = BIN_DIR  # GIFs werden im bin-Ordner gespeichert

def read_pgm(filename):
    """Liest eine PGM-Datei und gibt ein PIL-Image zurück."""
    with open(filename, 'rb') as f:
        magic = f.readline().strip()
        if magic != b'P5':
            raise ValueError(f"Ungültiges PGM-Format: {magic}")
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        width, height = map(int, line.split())
        maxval = int(f.readline().strip())
        data = f.read()
        image = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
        return Image.fromarray(image, mode='L')  # 'L' = Graustufen

def create_gif(pattern, output_filename, duration=0.2):
    """
    Erstellt ein GIF aus allen Dateien, die auf das Muster passen.
    pattern: z.B. "heatmap_mass_*.pgm"
    output_filename: z.B. "animation_mass.gif"
    duration: Zeit pro Frame in Sekunden
    """
    files = sorted(glob.glob(os.path.join(BIN_DIR, pattern)))
    if not files:
        print(f"Keine Dateien gefunden für Muster: {pattern}")
        return False

    print(f"Erstelle GIF aus {len(files)} Bildern: {output_filename}")
    images = []
    for f in files:
        img = read_pgm(f)
        images.append(img)

    # GIF speichern
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=int(duration * 1000),  # PIL erwartet Millisekunden
        loop=0,
        optimize=True
    )
    print(f"GIF gespeichert: {output_path}")
    return True

def main():
    print("Erstelle GIFs aus Heatmaps")
    print("=" * 70)

    # GIF für Mass-Heatmaps
    create_gif("heatmap_mass_*.pgm", "animation_mass.gif", duration=0.1)

    # GIF für Charge-Heatmaps
    create_gif("heatmap_charge_*.pgm", "animation_charge.gif", duration=0.1)

if __name__ == "__main__":
    main()