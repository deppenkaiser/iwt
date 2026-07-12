#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob

# Alle Positions-CSVs einlesen
files = sorted(glob.glob('positions_step_*.csv'))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

for f in files:
    step = int(f.split('_')[-1].split('.')[0])
    df = pd.read_csv(f)
    
    ax.clear()
    ax.scatter(df['x'], df['y'], df['z'], c='blue', s=5, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Schritt {step}')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    
    plt.pause(0.1)
    plt.savefig(f'frame_{step:03d}.png')