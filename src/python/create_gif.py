#!/usr/bin/env python3
import os
import glob
import numpy as np
from PIL import Image

BIN_DIR = os.path.expanduser("~/Dokumente/daten/sandbox/apps/iwt/build/bin/")
OUTPUT_DIR = BIN_DIR

def read_ppm(filename):
    """Liest eine PPM-Datei (P6) und gibt ein numpy-Array (H x W x 3) zurück."""
    with open(filename, 'rb') as f:
        magic = f.readline().strip()
        if magic != b'P6':
            raise ValueError(f"Kein PPM: {magic}")
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        width, height = map(int, line.split())
        maxval = int(f.readline().strip())
        data = f.read()
        rgb = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        return rgb

def overlay_mass_charge(mass_rgb, charge_rgb):
    """
    Überlagert Masse (Graustufen) und Ladung (Rot/Blau) - vektorisiert.
    - Helligkeit kommt von der Masse
    - Farbton kommt von der Ladung
    """
    # Masse als Helligkeit (1. Kanal, auf 0..1 skaliert)
    mass_gray = mass_rgb[:, :, 0].astype(np.float32) / 255.0

    # Ladung als Farben (auf 0..1 skaliert)
    charge_norm = charge_rgb.astype(np.float32) / 255.0

    # Erkenne neutrale Ladung (weiß = 1,1,1)
    is_neutral = np.all(charge_norm > 0.99, axis=2)

    # Erkenne rote Ladung (rot = 1,0,0)
    is_red = (charge_norm[:, :, 0] > 0.99) & (charge_norm[:, :, 1] < 0.01) & (charge_norm[:, :, 2] < 0.01)

    # Erkenne blaue Ladung (blau = 0,0,1)
    is_blue = (charge_norm[:, :, 0] < 0.01) & (charge_norm[:, :, 1] < 0.01) & (charge_norm[:, :, 2] > 0.99)

    # Ergebnis-Array (RGB, uint8)
    result = np.zeros_like(mass_rgb, dtype=np.uint8)

    # Fall 1: Neutrale Ladung -> Graustufen (Masse)
    gray_vals = (mass_gray * 255).astype(np.uint8)
    result[is_neutral, 0] = gray_vals[is_neutral]
    result[is_neutral, 1] = gray_vals[is_neutral]
    result[is_neutral, 2] = gray_vals[is_neutral]

    # Fall 2: Rote Ladung -> Rot, mit Helligkeit von der Masse
    red_vals = (mass_gray * 255).astype(np.uint8)
    result[is_red, 0] = red_vals[is_red]
    result[is_red, 1] = 0
    result[is_red, 2] = 0

    # Fall 3: Blaue Ladung -> Blau, mit Helligkeit von der Masse
    blue_vals = (mass_gray * 255).astype(np.uint8)
    result[is_blue, 0] = 0
    result[is_blue, 1] = 0
    result[is_blue, 2] = blue_vals[is_blue]

    # Fall 4: Alles andere (gemischte Ladung) -> Mischung aus Rot und Blau mit Helligkeit von der Masse
    is_mixed = ~(is_neutral | is_red | is_blue)
    if np.any(is_mixed):
        # Für gemischte Ladung: wir mischen Rot und Blau anteilig
        r_frac = charge_norm[:, :, 0]
        b_frac = charge_norm[:, :, 2]
        total = r_frac + b_frac + 0.001
        r_norm = r_frac / total
        b_norm = b_frac / total
        mix_vals = (mass_gray * 255).astype(np.uint8)
        result[is_mixed, 0] = (mix_vals[is_mixed] * r_norm[is_mixed]).astype(np.uint8)
        result[is_mixed, 1] = 0
        result[is_mixed, 2] = (mix_vals[is_mixed] * b_norm[is_mixed]).astype(np.uint8)

    return Image.fromarray(result, mode='RGB')

def create_gif(pattern, output_filename, duration=0.2, mode='mass'):
    files = sorted(glob.glob(os.path.join(BIN_DIR, pattern)))
    if not files:
        print(f"Keine Dateien für {pattern}")
        return False
    print(f"Erstelle {output_filename} (mode={mode})")
    images = []
    for f in files:
        if mode == 'mass':
            img = Image.open(f)
        elif mode == 'charge':
            img = Image.open(f)
        elif mode == 'overlay':
            base = os.path.basename(f)
            idx = base.split('_')[2].split('.')[0]
            mass_file = os.path.join(BIN_DIR, f"heatmap_mass_{idx}.ppm")
            charge_file = os.path.join(BIN_DIR, f"heatmap_charge_{idx}.ppm")
            if not os.path.exists(mass_file) or not os.path.exists(charge_file):
                print(f"Fehlende Dateien für {idx}, überspringe")
                continue
            mass_rgb = read_ppm(mass_file)
            charge_rgb = read_ppm(charge_file)
            img = overlay_mass_charge(mass_rgb, charge_rgb)
        else:
            img = Image.open(f)
        images.append(img)
    if not images:
        return False
    path = os.path.join(OUTPUT_DIR, output_filename)
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(duration * 1000), loop=0, optimize=True)
    print(f"GIF gespeichert: {path}")
    return True

def main():
    print("Erstelle GIFs aus Heatmaps")
    print("=" * 70)
    create_gif("heatmap_mass_*.ppm", "animation_mass.gif", duration=0.1, mode='mass')
    create_gif("heatmap_charge_*.ppm", "animation_charge_rgb.gif", duration=0.1, mode='charge')
    create_gif("heatmap_mass_*.ppm", "animation_overlay.gif", duration=0.1, mode='overlay')

if __name__ == "__main__":
    main()