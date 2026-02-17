#!/usr/bin/env python3
"""
analyze_geant4.py — Analyze Geant4 simulation outputs and compare with Highland.

Reads CSV event files from Geant4, fits scattering distributions,
and generates comparison plots against Highland predictions.

Usage:
  python analyze_geant4.py results/geant4/ --highland-dir results/highland/ --output-dir results/comparison/
"""

import argparse
import csv
import math
import os
import sys
import glob
from pathlib import Path

try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.stats import norm
except ImportError:
    print("Install: pip install numpy matplotlib scipy")
    sys.exit(1)


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


def analyze_event_file(filepath):
    """Read Geant4 CSV output and extract scattering distribution."""
    theta_x = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            theta_x.append(float(row['theta_x_mrad']))

    theta_x = np.array(theta_x)

    # Fit Gaussian to core (central 95%)
    q025, q975 = np.percentile(theta_x, [2.5, 97.5])
    core = theta_x[(theta_x > q025) & (theta_x < q975)]

    mu_fit = np.mean(core)
    sigma_fit = np.std(core)

    return {
        'theta_x': theta_x,
        'n_events': len(theta_x),
        'mean': np.mean(theta_x),
        'rms': np.std(theta_x),
        'sigma_core': sigma_fit,
        'mu_core': mu_fit,
    }


def load_highland_results(highland_dir):
    """Load Highland predictions from CSV."""
    results = {}
    csv_path = os.path.join(highland_dir, "predictions.csv")
    if not os.path.exists(csv_path):
        return results

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['material']}_{row['p_GeV']}GeV"
            results[key] = {
                'theta0_mrad': float(row['theta0_mrad']),
                'material': row['material'],
                'p_GeV': float(row['p_GeV']),
            }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("geant4_dir", help="Directory with Geant4 event CSVs")
    parser.add_argument("--highland-dir", default=None, help="Highland prediction directory")
    parser.add_argument("--output-dir", default="results/comparison", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Highland predictions for comparison
    highland = {}
    if args.highland_dir:
        highland = load_highland_results(args.highland_dir)
        print(f"📊 Loaded {len(highland)} Highland predictions")

    # Find and analyze all Geant4 event files
    event_files = sorted(glob.glob(os.path.join(args.geant4_dir, "*/events.csv")))
    if not event_files:
        event_files = sorted(glob.glob(os.path.join(args.geant4_dir, "**/events.csv"), recursive=True))

    print(f"⚛️  Found {len(event_files)} Geant4 event files")

    geant4_results = {}
    for ef in event_files:
        dirname = os.path.basename(os.path.dirname(ef))
        print(f"   Analyzing: {dirname}")
        geant4_results[dirname] = analyze_event_file(ef)

    # ─── Generate comparison plot ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # LEFT: Bar chart comparing Geant4 σ vs Highland θ₀
    ax1 = axes[0]
    materials = sorted(geant4_results.keys())
    g4_sigmas = [geant4_results[m]['sigma_core'] for m in materials]
    labels = [m.replace('_', '\n') for m in materials]

    x_pos = np.arange(len(materials))
    bars1 = ax1.bar(x_pos - 0.15, g4_sigmas, 0.3, label='Geant4 (σ core)',
                    color='#1E88E5', edgecolor='black', linewidth=0.5)

    # Overlay Highland predictions if available
    highland_vals = []
    for m in materials:
        # Try to match material name
        h_match = None
        for hk, hv in highland.items():
            if hv['material'] in m:
                h_match = hv
                break
        highland_vals.append(h_match['theta0_mrad'] if h_match else None)

    h_vals_plot = [v for v in highland_vals if v is not None]
    h_positions = [x_pos[i] for i, v in enumerate(highland_vals) if v is not None]
    if h_vals_plot:
        ax1.bar([p + 0.15 for p in h_positions], h_vals_plot, 0.3,
                label='Highland formula', color='#FFA726', edgecolor='black', linewidth=0.5)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("σ / θ₀ (mrad)", fontsize=12, fontweight='bold')
    ax1.set_title("Geant4 vs Highland Comparison", fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.15)

    # RIGHT: Overlaid scattering distributions from Geant4
    ax2 = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(materials)))
    for i, m in enumerate(materials):
        data = geant4_results[m]['theta_x']
        ax2.hist(data, bins=80, density=True, histtype='step', linewidth=2,
                color=colors[i], label=f"{m.split('_')[0]} (σ={geant4_results[m]['sigma_core']:.3f})")

    ax2.set_xlabel("Projected scattering angle θₓ (mrad)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Probability density", fontsize=12)
    ax2.set_title("Geant4 Scattering Distributions", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.15)

    plt.tight_layout()
    outpath = os.path.join(args.output_dir, "geant4_vs_highland.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Comparison plot: {outpath}")

    # ─── Summary table ───
    summary_path = os.path.join(args.output_dir, "COMPARISON.md")
    with open(summary_path, 'w') as f:
        f.write("# Geant4 vs Highland Comparison\n\n")
        f.write("| Configuration | Geant4 σ (mrad) | Highland θ₀ (mrad) | Ratio | N events |\n")
        f.write("|--------------|-----------------|-------------------|-------|----------|\n")
        for m in materials:
            g4_s = geant4_results[m]['sigma_core']
            h_match = None
            for hk, hv in highland.items():
                if hv['material'] in m:
                    h_match = hv
                    break
            h_val = h_match['theta0_mrad'] if h_match else None
            ratio = f"{g4_s / h_val:.3f}" if h_val else "—"
            h_str = f"{h_val:.4f}" if h_val else "—"
            f.write(f"| {m} | {g4_s:.4f} | {h_str} | {ratio} | {geant4_results[m]['n_events']:,} |\n")

    print(f"✅ Summary: {summary_path}")


if __name__ == "__main__":
    main()
