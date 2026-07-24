#!/usr/bin/env python3
import os
import sys
import subprocess
import glob
import shutil
import time
from PIL import Image
import numpy as np

# ============================================================
# PFADE
# ============================================================
PROJECT_DIR = os.path.expanduser("~/Dokumente/daten/sandbox/apps/iwt")
BIN_DIR = os.path.join(PROJECT_DIR, "build/bin")
EXECUTABLE = os.path.join(BIN_DIR, "iwt")
SRC_PYTHON = os.path.join(PROJECT_DIR, "src/python")

def read_pgm(filename):
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
        return Image.fromarray(image, mode='L')

def create_gif_from_pattern(pattern, output_filename, duration=0.1):
    files = sorted(glob.glob(os.path.join(BIN_DIR, pattern)))
    if not files:
        print(f"  Keine Dateien für Muster: {pattern}")
        return False
    images = [read_pgm(f) for f in files]
    output_path = os.path.join(BIN_DIR, output_filename)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=int(duration * 1000),
        loop=0,
        optimize=True
    )
    print(f"  GIF gespeichert: {output_path}")
    return True

def run_simulation(seed):
    """Führt die Simulation mit einem bestimmten Seed aus."""
    print(f"\n--- Lauf mit Seed: {seed} ---")
    
    # Alte Heatmaps löschen (nur für diesen Lauf)
    for f in glob.glob(os.path.join(BIN_DIR, "heatmap_*.pgm")):
        os.remove(f)
    
    # Simulation starten
    cmd = [EXECUTABLE, "--seed", str(seed), "--no-fluctuations"]
    # Hinweis: --no-fluctuations ist optional – wenn Du Fluktuationen behalten willst, lösche es
    # cmd = [EXECUTABLE, "--seed", str(seed)]
    
    print(f"  Starte: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=BIN_DIR, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  FEHLER: Simulation mit Seed {seed} fehlgeschlagen")
        print(result.stderr)
        return False
    
    print(f"  Simulation beendet (RC={result.returncode})")
    return True

def analyze_and_create_gifs(run_number):
    """Analysiert die Heatmaps und erstellt GIFs für einen Lauf."""
    print(f"\n  Analysiere Lauf {run_number}...")
    
    # GIFs erstellen
    mass_gif = f"lauf_{run_number:03d}_mass.gif"
    charge_gif = f"lauf_{run_number:03d}_charge.gif"
    
    success_mass = create_gif_from_pattern("heatmap_mass_*.pgm", mass_gif)
    success_charge = create_gif_from_pattern("heatmap_charge_*.pgm", charge_gif)
    
    return success_mass and success_charge

def main():
    print("=" * 70)
    print("AUTOMATISIERTE TESTS MIT VERSCHIEDENEN SEEDS")
    print("=" * 70)
    
    # Prüfen, ob die ausführbare Datei existiert
    if not os.path.exists(EXECUTABLE):
        print(f"FEHLER: Ausführbare Datei nicht gefunden: {EXECUTABLE}")
        print("Bitte zuerst 'make' in build/ ausführen.")
        return
    
    # Seeds für die Tests
    seeds = [12345, 67890, 11111, 22222, 33333]
    # seeds = [42, 43, 44]  # alternative kleine Seeds
    
    print(f"\nGeplante Läufe: {len(seeds)}")
    print(f"Seeds: {seeds}")
    
    for idx, seed in enumerate(seeds, start=1):
        if not run_simulation(seed):
            continue
        
        if not analyze_and_create_gifs(idx):
            continue
        
        print(f"  Lauf {idx} abgeschlossen.")
        time.sleep(0.5)  # Kurze Pause zwischen den Läufen
    
    print("\n" + "=" * 70)
    print("ALLE LÄUFE ABGESCHLOSSEN")
    print("=" * 70)
    
    # Auflistung der erstellten GIFs
    gifs = sorted(glob.glob(os.path.join(BIN_DIR, "lauf_*.gif")))
    if gifs:
        print("\nErstellte GIFs:")
        for g in gifs:
            print(f"  {os.path.basename(g)}")
    else:
        print("\nKeine GIFs erstellt.")

if __name__ == "__main__":
    main()
    