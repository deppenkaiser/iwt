#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import glob

CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../build/bin')

def get_dodecahedron_vertices():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    raw = np.array([
        [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1],
        [-1,  1,  1], [-1,  1, -1], [-1, -1,  1], [-1, -1, -1],
        [ phi,  1/phi, 0], [ phi, -1/phi, 0], [-phi,  1/phi, 0], [-phi, -1/phi, 0],
        [0,  phi,  1/phi], [0,  phi, -1/phi], [0, -phi,  1/phi], [0, -phi, -1/phi],
        [ 1/phi, 0,  phi], [ 1/phi, 0, -phi], [-1/phi, 0,  phi], [-1/phi, 0, -phi]
    ])
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms

def main():
    files = sorted(glob.glob(os.path.join(CSV_DIR, 'positions_step_*.csv')))
    if not files:
        print("Keine CSV-Dateien gefunden.")
        return
    
    data = {}
    for f in files:
        step = int(f.split('_')[-1].split('.')[0])
        data[step] = pd.read_csv(f)
        print(f"Geladen: Schritt {step}")
    
    steps = sorted(data.keys())
    final_step = steps[-1]
    df_final = data[final_step]
    
    # 1. Die 20 Knoten mit der höchsten Dichte im FINALEN Schritt finden
    if 'I' in df_final.columns:
        df_sorted = df_final.sort_values('I', ascending=False)
    else:
        # Fallback: Radius als Proxy für Dichte
        df_final['r2'] = df_final['x']**2 + df_final['y']**2 + df_final['z']**2
        df_sorted = df_final.sort_values('r2', ascending=True)
    
    top20_indices = df_sorted.head(20)['index'].values
    print(f"Die 20 dichtesten Knoten (final): {top20_indices}")
    
    # 2. Diese 20 Knoten über die Zeit verfolgen
    dodeca_vertices = get_dodecahedron_vertices()
    
    # Abweichung der 20 Knoten vom Dodekaeder über die Zeit
    deviations = []
    
    for step in steps:
        df = data[step]
        positions = []
        for idx in top20_indices:
            row = df[df['index'] == idx]
            if not row.empty:
                positions.append([row['x'].values[0], row['y'].values[0], row['z'].values[0]])
            else:
                positions.append([0, 0, 0])
        positions = np.array(positions)
        
        # Normieren auf Einheitskugel
        norms = np.linalg.norm(positions, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        positions_norm = positions / norms
        
        # Mittlere Abweichung von den Dodekaeder-Ecken
        dev = np.mean(np.linalg.norm(positions_norm - dodeca_vertices, axis=1))
        deviations.append(dev)
    
    # 3. Plot: Konvergenz der 20 festen Knoten
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, deviations, 'b-o', linewidth=2, markersize=6)
    ax.axhline(y=0.02, color='red', linestyle='--', linewidth=1.5, label='Schwelle (0.02)')
    ax.set_xlabel('Zeitschritt')
    ax.set_ylabel('Mittlere Abweichung vom Dodekaeder')
    ax.set_title('Konvergenz der 20 dichtesten Knoten (final) zum Dodekaeder')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CSV_DIR, 'convergence_fixed_nodes.png'), dpi=150)
    print("Plot gespeichert: convergence_fixed_nodes.png")
    
    # 4. Statistik
    print(f"\nStart-Abweichung: {deviations[0]:.4f}")
    print(f"End-Abweichung: {deviations[-1]:.4f}")
    print(f"Verbesserung: {deviations[0] - deviations[-1]:.4f}")

if __name__ == "__main__":
    main()