#!/usr/bin/env python3
import os
import sys
import glob
import numpy as np
from collections import Counter

# ============================================================
# PFADE
# ============================================================
BIN_DIR = os.path.expanduser("~/Dokumente/daten/sandbox/apps/iwt/build/bin/")

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
        return image

def find_hotspots(image, threshold=200):
    """Findet helle Regionen (Hotspots) im Bild."""
    hotspots = []
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            if image[y, x] > threshold:
                hotspots.append((x, y))
    return hotspots

def analyze_movement(mass_files):
    """Analysiert die Bewegung der Massen-Hotspots über die Zeit."""
    print("Bewegungsanalyse der Massen-Hotspots")
    print("=" * 70)
    
    # Positionen über die Zeit sammeln
    positions_by_iter = {}
    all_hotspots = []
    
    for f in mass_files:
        # Iteration aus Dateiname extrahieren
        basename = os.path.basename(f)
        iter_str = basename.replace("heatmap_mass_", "").replace(".pgm", "")
        iter_num = int(iter_str)
        
        image = read_pgm(f)
        hotspots = find_hotspots(image, threshold=200)
        
        positions_by_iter[iter_num] = hotspots
        all_hotspots.extend(hotspots)
        
        print(f"  Iteration {iter_num}: {len(hotspots)} Hotspots")
    
    print("\n" + "=" * 70)
    print("Statistik:")
    print(f"  Anzahl Bilder: {len(mass_files)}")
    print(f"  Iterationen: {min(positions_by_iter.keys())} - {max(positions_by_iter.keys())}")
    print(f"  Hotspots pro Bild: {np.mean([len(h) for h in positions_by_iter.values()]):.2f} (±{np.std([len(h) for h in positions_by_iter.values()]):.2f})")
    
    # Bewegung der Hotspots analysieren
    print("\nBewegung der Hotspots (Änderung der Position pro Iteration):")
    prev_hotspots = None
    total_movement = []
    
    for iter_num in sorted(positions_by_iter.keys()):
        current = positions_by_iter[iter_num]
        if prev_hotspots is not None and len(current) > 0 and len(prev_hotspots) > 0:
            # Durchschnittliche Bewegung berechnen
            movement = 0
            for h in current:
                # Nächsten Hotspot in vorherigem Bild finden
                min_dist = float('inf')
                for ph in prev_hotspots:
                    dist = np.sqrt((h[0] - ph[0])**2 + (h[1] - ph[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                movement += min_dist
            movement /= len(current)
            total_movement.append(movement)
        prev_hotspots = current
    
    if total_movement:
        print(f"  Mittlere Bewegung pro Iteration: {np.mean(total_movement):.2f} Pixel")
        print(f"  Std der Bewegung: {np.std(total_movement):.2f} Pixel")
        
        # Bewegungstrend
        if len(total_movement) > 10:
            trend = np.polyfit(range(len(total_movement)), total_movement, 1)[0]
            print(f"  Bewegungstrend: {trend:.4f} Pixel/Iteration (positiv = driftet auseinander)")
    
    # Hotspot-Kontinuität: Verfolge einzelne Hotspots
    print("\nHotspot-Kontinuität (Verfolgung über 10 Iterationen):")
    
    # Hotspots in erste und letzte 10 Bilder vergleichen
    first_10 = []
    last_10 = []
    for i, iter_num in enumerate(sorted(positions_by_iter.keys())):
        if i < 10:
            first_10.extend(positions_by_iter[iter_num])
        if i >= len(positions_by_iter) - 10:
            last_10.extend(positions_by_iter[iter_num])
    
    # Überlappung prüfen
    if first_10 and last_10:
        first_set = set(first_10)
        last_set = set(last_10)
        overlap = first_set.intersection(last_set)
        print(f"  Hotspots in ersten 10 Bildern: {len(first_set)}")
        print(f"  Hotspots in letzten 10 Bildern: {len(last_set)}")
        print(f"  Überlappung: {len(overlap)}")
        if len(first_set) > 0:
            print(f"  Stabilität (Überlappung / erste 10): {len(overlap)/len(first_set)*100:.1f}%")
    
    return positions_by_iter

def main():
    print("Heatmap Analyse – Bewegungsanalyse")
    print("=" * 70)
    
    mass_files = sorted(glob.glob(os.path.join(BIN_DIR, "heatmap_mass_*.pgm")))
    charge_files = sorted(glob.glob(os.path.join(BIN_DIR, "heatmap_charge_*.pgm")))
    
    if not mass_files:
        print("Keine Mass-Heatmaps gefunden in:", BIN_DIR)
        return
    
    print(f"Mass-Heatmaps: {len(mass_files)}")
    print(f"Charge-Heatmaps: {len(charge_files)}")
    print()
    
    positions_by_iter = analyze_movement(mass_files)

if __name__ == "__main__":
    main()