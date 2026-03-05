#!/usr/bin/env python3
"""
gatherAndPlot.py - Load benchmark timestamp files and generate FPS analysis plots.

Expects timestamp files: cpu.txt, gpu.txt, rvc2.txt (one float per line).
Computes FPS from consecutive timestamp deltas and generates:
  1. FPS over time (line plot)
  2. Average FPS with min/max error bars (bar chart)
  3. FPS variance comparison (bar chart)

Usage:
    python gatherAndPlot.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def load_timestamps(filepath: str) -> np.ndarray:
    """Load timestamps from a file (one float per line)."""
    with open(filepath, 'r') as f:
        timestamps = [float(line.strip()) for line in f if line.strip()]
    return np.array(timestamps)


def compute_fps(timestamps: np.ndarray) -> np.ndarray:
    """Compute FPS from consecutive timestamp deltas."""
    if len(timestamps) < 2:
        return np.array([])
    deltas = np.diff(timestamps)
    deltas = np.where(deltas > 0, deltas, np.finfo(float).eps)
    return 1.0 / deltas


def plot_fps_over_time(fps_data: dict[str, np.ndarray]):
    """Plot FPS over time for each benchmark run."""
    plt.figure(figsize=(12, 6))
    
    for label, fps in fps_data.items():
        frame_numbers = np.arange(len(fps))
        plt.plot(frame_numbers, fps, label=label, alpha=0.8, linewidth=2)
    
    plt.xlabel('Frame Number')
    plt.ylabel('FPS')
    plt.title('FPS Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/fps_over_time.png', dpi=150)
    plt.close()


def plot_average_fps_with_minmax(fps_data: dict[str, np.ndarray]):
    """Plot average FPS as bar chart with min/max error bars."""
    labels = list(fps_data.keys())
    means = [np.mean(fps) for fps in fps_data.values()]
    mins = [np.min(fps) for fps in fps_data.values()]
    maxs = [np.max(fps) for fps in fps_data.values()]
    
    means = np.array(means)
    mins = np.array(mins)
    maxs = np.array(maxs)
    errors = np.array([means - mins, maxs - means])
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    bars = plt.bar(x, means, color='steelblue', alpha=0.8)
    
    # Draw error bars with colored markers for min/max
    for i, (bar, mean, mn, mx) in enumerate(zip(bars, means, mins, maxs)):
        cx = bar.get_x() + bar.get_width() / 2
        # Vertical line from min to max
        plt.plot([cx, cx], [mn, mx], color='#333333', linewidth=1.5, zorder=3)
        # Min marker + label
        plt.scatter(cx, mn, color='crimson', s=60, zorder=4, marker='v')
        plt.text(cx + 0.15, mn, f'{mn:.1f}', ha='left', va='center', fontsize=9, color='crimson')
        # Max marker + label
        plt.scatter(cx, mx, color='forestgreen', s=60, zorder=4, marker='^')
        plt.text(cx + 0.15, mx, f'{mx:.1f}', ha='left', va='center', fontsize=9, color='forestgreen')
        # Mean label on bar
        plt.text(cx, mean / 2, f'avg {mean:.1f}', ha='center', va='center',
                 fontsize=11, fontweight='bold', color='white')
    
    # Custom legend for min/max markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='forestgreen', markersize=10, label='Max'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='crimson', markersize=10, label='Min'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.xlabel('Benchmark')
    plt.ylabel('FPS')
    plt.title('Average FPS (with Min/Max Range)')
    plt.xticks(x, labels)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('plots/fps_average.png', dpi=150)
    plt.close()


def plot_fps_distribution(fps_data: dict[str, np.ndarray]):
    """Plot FPS histogram with fitted normal curve overlay for each benchmark."""
    from scipy.stats import norm
    
    plt.figure(figsize=(12, 6))
    colors = ['steelblue', 'coral', 'mediumseagreen']
    
    for i, (label, fps) in enumerate(fps_data.items()):
        color = colors[i % len(colors)]
        mean, std = np.mean(fps), np.std(fps)
        
        # Histogram (normalized to density)
        plt.hist(fps, bins=40, alpha=0.3, color=color, density=True, label=f'{label} hist')
        
        # Fitted normal curve
        x = np.linspace(fps.min(), fps.max(), 200)
        plt.plot(x, norm.pdf(x, mean, std), color=color, linewidth=2,
                 label=f'{label} (μ={mean:.1f}, σ={std:.2f})')
    
    plt.xlabel('FPS')
    plt.ylabel('Density')
    plt.title('FPS Distribution (Histogram + Fitted Normal Curve)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/fps_distribution.png', dpi=150)
    plt.close()


def print_statistics(fps_data: dict[str, np.ndarray]):
    """Print summary statistics for each benchmark."""
    print("\n" + "=" * 60)
    print("FPS Statistics Summary")
    print("=" * 60)
    
    for label, fps in fps_data.items():
        print(f"\n{label}:")
        print(f"  Frames analyzed: {len(fps)}")
        print(f"  Average FPS:     {np.mean(fps):.2f}")
        print(f"  Min FPS:         {np.min(fps):.2f}")
        print(f"  Max FPS:         {np.max(fps):.2f}")
        print(f"  Std Dev:         {np.std(fps):.2f}")
        print(f"  Variance:        {np.var(fps):.2f}")
        print(f"  Median FPS:      {np.median(fps):.2f}")
    
    print("\n" + "=" * 60)


def main():
    # Hardcoded input files (rvc2 is optional)
    files = {'CPU': 'CPU.txt', 'GPU': 'GPU.txt'}
    if os.path.isfile('RVC2.txt'):
        files['RVC2'] = 'RVC2.txt'
    
    # Load and process each file
    fps_data = {}
    for label, filepath in files.items():
        try:
            timestamps = load_timestamps(filepath)
            fps = compute_fps(timestamps)
            fps_data[label] = fps
            print(f"Loaded {len(timestamps)} timestamps from {filepath} -> {len(fps)} FPS values")
        except FileNotFoundError:
            print(f"Warning: {filepath} not found, skipping")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    if not fps_data:
        print("Error: No valid data loaded")
        return
    
    # Print statistics
    print_statistics(fps_data)
    
    # Generate plots
    os.makedirs('plots', exist_ok=True)
    plot_fps_over_time(fps_data)
    plot_average_fps_with_minmax(fps_data)
    plot_fps_distribution(fps_data)
    
    print('\nPlots saved to plots/')


if __name__ == '__main__':
    main()
