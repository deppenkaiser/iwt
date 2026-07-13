#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os

# Pfad zur CSV-Datei (relativ zum Skript)
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../build/bin/drift.csv')

df = pd.read_csv(csv_path)

plt.figure(figsize=(12, 6))
colors = ['red' if d < 0 else 'green' for d in df['drift_per_step']]
plt.bar(df['index'], df['drift_per_step'], color=colors, alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-')
plt.xlabel('Knoten-Index')
plt.ylabel('Drift pro Schritt')
plt.title('DSTT-Drift: Information fließt von Knoten 0–3 zu Knoten 4–5')
plt.grid(True, alpha=0.3)
plt.savefig('drift_plot.png', dpi=150)
print("Plot gespeichert: drift_plot.png")