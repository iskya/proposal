#!/usr/bin/env python3
"""
highland_calculator.py — Instant MCS predictions from student request files.

Reads a YAML request file and produces:
  1. A scattering angle distribution overlay plot
  2. A classification plot (if multiple momenta)
  3. A comparison table (CSV)
  4. A Markdown summary (for PR comments)

Usage:
  python highland_calculator.py requests/maria_pvc_vs_pet.yaml --output-dir results/maria_pvc_vs_pet/
"""

import argparse
import math
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Try importing optional dependencies
try:
    import yaml
except ImportError:
    print("Installing PyYAML...")
    os.system(f"{sys.executable} -m pip install pyyaml -q")
    import yaml

try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    HAS_PLOTTING = True
except ImportError:
    print("Warning: matplotlib/numpy not available. Will produce tables only.")
    HAS_PLOTTING = False


# ═══════════════════════════════════════════════════════════════
# MATERIALS DATABASE
# ═══════════════════════════════════════════════════════════════

MATERIALS_DB = {
    # Plastics
    "G4_POLYETHYLENE":        {"name": "PE",       "X0_cm": 47.9,  "rho": 0.94, "category": "plastic"},
    "G4_POLYPROPYLENE":       {"name": "PP",       "X0_cm": 47.4,  "rho": 0.90, "category": "plastic"},
    "G4_POLYSTYRENE":         {"name": "PS",       "X0_cm": 42.5,  "rho": 1.05, "category": "plastic"},
    "G4_PLEXIGLASS":          {"name": "PMMA",     "X0_cm": 34.4,  "rho": 1.18, "category": "plastic"},
    "G4_NYLON-6-6":           {"name": "Nylon",    "X0_cm": 36.7,  "rho": 1.14, "category": "plastic"},
    "G4_MYLAR":               {"name": "PET",      "X0_cm": 28.7,  "rho": 1.38, "category": "plastic"},
    "G4_POLYVINYL_CHLORIDE":  {"name": "PVC",      "X0_cm": 15.3,  "rho": 1.40, "category": "plastic"},
    # Heritage / Geological
    "Obsidian":               {"name": "Obsidian", "X0_cm": 12.3,  "rho": 2.40, "category": "heritage"},
    "G4_SILICON_DIOXIDE":     {"name": "SiO₂",    "X0_cm": 12.29, "rho": 2.20, "category": "heritage"},
    "G4_CALCIUM_CARBONATE":   {"name": "CaCO₃",   "X0_cm": 10.7,  "rho": 2.71, "category": "heritage"},
    "G4_ALUMINUM_OXIDE":      {"name": "Al₂O₃",   "X0_cm": 10.2,  "rho": 3.95, "category": "heritage"},
    "G4_FERRIC_OXIDE":        {"name": "Fe₂O₃",   "X0_cm": 5.5,   "rho": 5.24, "category": "heritage"},
    # Metals
    "G4_Al":                  {"name": "Al",       "X0_cm": 8.90,  "rho": 2.70, "category": "metal"},
    "G4_Fe":                  {"name": "Fe",       "X0_cm": 1.757, "rho": 7.87, "category": "metal"},
    "G4_Pb":                  {"name": "Pb",       "X0_cm": 0.561, "rho": 11.35,"category": "metal"},
}

COLORS = {
    "plastic":  ["#1E88E5", "#42A5F5", "#43A047", "#66BB6A", "#FFA726", "#EF5350", "#E53935"],
    "heritage": ["#78909C", "#C0CA33", "#00BCD4", "#AB47BC", "#6D4C41"],
    "metal":    ["#B0BEC5", "#607D8B", "#37474F"],
    "custom":   ["#FF6F00", "#F9A825", "#E65100", "#BF360C"],
    "fallback": ["#9E9E9E", "#757575"],
}


# ═══════════════════════════════════════════════════════════════
# PHYSICS
# ═══════════════════════════════════════════════════════════════

def highland_theta0_mrad(p_GeV, x_cm, X0_cm):
    """Highland formula: RMS projected scattering angle in mrad."""
    p_MeV = p_GeV * 1000.0
    beta = 1.0  # relativistic approximation (valid for GeV beams)
    ratio = x_cm / X0_cm
    if ratio <= 0:
        return 0.0
    theta0 = (13.6 / (p_MeV * beta)) * math.sqrt(ratio) * (1 + 0.038 * math.log(ratio))
    return theta0 * 1000.0  # convert to mrad

def bethe_bloch_dE_MeV(x_cm, rho_gcc):
    """Approximate MIP energy loss: ~2 MeV/(g/cm²) for most materials."""
    return 2.0 * rho_gcc * x_cm  # MeV

def events_for_3sigma(theta1, theta2):
    """Events needed to separate two materials at 3σ (Gaussian means)."""
    diff = abs(theta1 - theta2)
    if diff == 0:
        return float('inf')
    avg_sigma = (theta1 + theta2) / 2.0
    return int(math.ceil(2 * (3 * avg_sigma / diff) ** 2))


# ═══════════════════════════════════════════════════════════════
# REQUEST PROCESSING
# ═══════════════════════════════════════════════════════════════

def load_request(filepath):
    """Load and validate a student request YAML file."""
    with open(filepath) as f:
        req = yaml.safe_load(f)

    # Defaults
    req.setdefault("author", "Anonymous")
    req.setdefault("description", "")
    req["beam"].setdefault("particle", "e-")
    req["beam"].setdefault("num_events", 10000)
    req["outputs"].setdefault("highland_prediction", True)
    req["outputs"].setdefault("geant4_simulation", False)
    req["outputs"].setdefault("classification_plot", True)
    req["outputs"].setdefault("distribution_overlay", True)
    req["outputs"].setdefault("comparison_table", True)

    return req

def compute_predictions(req):
    """Compute Highland formula predictions for all materials × momenta."""
    results = []
    for mat in req["materials"]:
        g4name = mat["geant4_name"]
        x_mm = mat["thickness_mm"]
        x_cm = x_mm / 10.0

        db = MATERIALS_DB.get(g4name, None)

        # Allow fully user-specified material properties in the YAML
        # (useful for student exploration without touching code)
        if db is None:
            if ("X0_cm" in mat or "x0_cm" in mat) and \
               ("rho" in mat or "rho_gcc" in mat or "density_gcm3" in mat):
                X0_val = float(mat.get("X0_cm", mat.get("x0_cm")))
                rho_val = float(mat.get("rho", mat.get("rho_gcc", mat.get("density_gcm3"))))
                db = {
                    "name": mat.get("name", g4name),
                    "X0_cm": X0_val,
                    "rho": rho_val,
                    "category": mat.get("category", "custom"),
                }
                print(f"  📐 Custom material: {mat['name']} (X₀={X0_val} cm, ρ={rho_val} g/cm³)")
            else:
                print(f"  ⚠️  Unknown material: {g4name} (and no X0_cm/rho provided). Skipping.")
                continue

        for p in req["beam"]["momenta_GeV"]:
            theta0 = highland_theta0_mrad(p, x_cm, db["X0_cm"])
            dE = bethe_bloch_dE_MeV(x_cm, db["rho"])

            results.append({
                "name": mat["name"],
                "geant4_name": g4name,
                "category": db["category"],
                "X0_cm": db["X0_cm"],
                "rho": db["rho"],
                "thickness_mm": x_mm,
                "p_GeV": p,
                "theta0_mrad": theta0,
                "dE_MeV": dE,
            })

    return results


# ═══════════════════════════════════════════════════════════════
# OUTPUT GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_table_csv(results, outdir):
    """Write results as CSV."""
    filepath = os.path.join(outdir, "predictions.csv")
    with open(filepath, 'w') as f:
        f.write("material,geant4_name,category,X0_cm,rho_gcc,thickness_mm,p_GeV,theta0_mrad,dE_MeV\n")
        for r in results:
            f.write(f"{r['name']},{r['geant4_name']},{r['category']},{r['X0_cm']},{r['rho']},"
                    f"{r['thickness_mm']},{r['p_GeV']},{r['theta0_mrad']:.4f},{r['dE_MeV']:.2f}\n")
    return filepath

def generate_markdown_summary(req, results, outdir, plots_generated):
    """Write a Markdown summary suitable for PR comments."""
    filepath = os.path.join(outdir, "SUMMARY.md")
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    with open(filepath, 'w') as f:
        f.write(f"# 🔬 BeamScan Simulation Results\n\n")
        f.write(f"**Author:** {req['author']}  \n")
        f.write(f"**Description:** {req['description']}  \n")
        f.write(f"**Generated:** {now}  \n")
        f.write(f"**Method:** Highland formula (analytical)\n\n")

        f.write(f"## Beam Settings\n")
        f.write(f"- Particle: `{req['beam']['particle']}`\n")
        f.write(f"- Momenta: {req['beam']['momenta_GeV']} GeV/c\n")
        f.write(f"- Events requested: {req['beam']['num_events']:,}\n\n")

        # Results table
        f.write(f"## Predictions\n\n")
        f.write(f"| Material | p (GeV/c) | θ₀ (mrad) | ΔE (MeV) | X₀ (cm) | Thickness |\n")
        f.write(f"|----------|-----------|-----------|----------|---------|----------|\n")
        for r in results:
            f.write(f"| {r['name']} | {r['p_GeV']} | **{r['theta0_mrad']:.3f}** | "
                    f"{r['dE_MeV']:.1f} | {r['X0_cm']} | {r['thickness_mm']} mm |\n")

        # Discrimination matrix
        momenta = sorted(set(r['p_GeV'] for r in results))
        p_ref = momenta[0]  # Use lowest momentum for discrimination
        ref_results = [r for r in results if r['p_GeV'] == p_ref]

        if len(ref_results) > 1:
            f.write(f"\n## Discrimination Power (at {p_ref} GeV/c)\n\n")
            f.write(f"Events needed for 3σ separation:\n\n")
            names = [r['name'] for r in ref_results]
            f.write(f"| | " + " | ".join(names) + " |\n")
            f.write(f"|---" * (len(names) + 1) + "|\n")
            for i, r1 in enumerate(ref_results):
                row = f"| **{r1['name']}** |"
                for j, r2 in enumerate(ref_results):
                    if i == j:
                        row += " — |"
                    else:
                        n = events_for_3sigma(r1['theta0_mrad'], r2['theta0_mrad'])
                        if n > 100000:
                            row += f" ❌ {n:,.0f} |"
                        elif n > 5000:
                            row += f" ⚠️ {n:,} |"
                        else:
                            row += f" ✅ {n:,} |"
                f.write(row + "\n")

            f.write(f"\n✅ Easy (<5k events) | ⚠️ Moderate (5k–100k) | ❌ Impractical (>100k)\n")

        # Plots
        if plots_generated:
            f.write(f"\n## Figures\n\n")
            for plot in plots_generated:
                fname = os.path.basename(plot)
                f.write(f"![{fname}]({fname})\n\n")

        f.write(f"\n---\n*Generated automatically by BeamScan Highland Calculator*\n")

    return filepath


def generate_distribution_plot(req, results, outdir):
    """Overlaid Gaussian scattering distributions for each material at each momentum."""
    if not HAS_PLOTTING:
        return None

    momenta = sorted(set(r['p_GeV'] for r in results))

    fig, axes = plt.subplots(1, len(momenta), figsize=(7 * len(momenta), 5), squeeze=False)

    for idx, p in enumerate(momenta):
        ax = axes[0, idx]
        p_results = [r for r in results if r['p_GeV'] == p]

        max_theta = max(r['theta0_mrad'] for r in p_results) * 3.5
        x = np.linspace(-max_theta, max_theta, 500)

        color_idx = {cat: 0 for cat in COLORS}
        for r in p_results:
            cat = r['category']
            cat_colors = COLORS.get(cat, COLORS.get('custom', ['#FF6F00']))
            ci = color_idx.get(cat, 0) % len(cat_colors)
            color = cat_colors[ci]
            color_idx[cat] = color_idx.get(cat, 0) + 1

            sigma = r['theta0_mrad']
            y = np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
            ax.plot(x, y, linewidth=2.2, color=color,
                    label=f"{r['name']} (σ={sigma:.3f} mrad)")

        ax.set_xlabel("Projected scattering angle θₓ (mrad)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Probability density" if idx == 0 else "", fontsize=12)
        ax.set_title(f"p = {p} GeV/c, {r['thickness_mm']:.0f} mm targets", fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.15)

    fig.suptitle(f"BeamScan — Expected Scattering Distributions\n{req['author']}: {req['description']}",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = os.path.join(outdir, "distributions.png")
    plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    return filepath


def generate_classification_plot(req, results, outdir):
    """2D classification plot using θ₀ at two momenta (or θ₀ vs dE)."""
    if not HAS_PLOTTING:
        return None

    momenta = sorted(set(r['p_GeV'] for r in results))
    materials_in_request = sorted(set(r['name'] for r in results))

    if len(momenta) >= 2:
        # Two-momentum classification plot
        p1, p2 = momenta[0], momenta[1]
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        color_idx = {cat: 0 for cat in COLORS}
        for mat_name in materials_in_request:
            r1 = next((r for r in results if r['name'] == mat_name and r['p_GeV'] == p1), None)
            r2 = next((r for r in results if r['name'] == mat_name and r['p_GeV'] == p2), None)
            if not r1 or not r2:
                continue

            cat = r1['category']
            cat_colors = COLORS.get(cat, COLORS.get('custom', ['#FF6F00']))
            ci = color_idx.get(cat, 0) % len(cat_colors)
            color = cat_colors[ci]
            color_idx[cat] = color_idx.get(cat, 0) + 1

            marker = 'o' if cat == 'plastic' else ('D' if cat == 'heritage' else 's')
            ax.scatter(r1['theta0_mrad'], r2['theta0_mrad'], c=color, s=150,
                      marker=marker, edgecolors='black', linewidth=0.8, zorder=5, label=mat_name)

            # Error ellipse (10k events)
            N = req['beam']['num_events']
            e1 = r1['theta0_mrad'] / math.sqrt(2 * N) * 5
            e2 = r2['theta0_mrad'] / math.sqrt(2 * N) * 5
            ell = Ellipse((r1['theta0_mrad'], r2['theta0_mrad']),
                         width=2*e1, height=2*e2, facecolor=color, alpha=0.15,
                         edgecolor=color, linewidth=1)
            ax.add_patch(ell)

        ax.set_xlabel(f"θ₀ at {p1} GeV/c (mrad)", fontsize=13, fontweight='bold')
        ax.set_ylabel(f"θ₀ at {p2} GeV/c (mrad)", fontsize=13, fontweight='bold')
        ax.set_title(f"BeamScan Classification Plot\n{req['author']}: {req['description']}",
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.15)

    else:
        # Single momentum: use θ₀ vs ΔE
        p = momenta[0]
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        color_idx = {cat: 0 for cat in COLORS}
        for r in [r for r in results if r['p_GeV'] == p]:
            cat = r['category']
            cat_colors = COLORS.get(cat, COLORS.get('custom', ['#FF6F00']))
            ci = color_idx.get(cat, 0) % len(cat_colors)
            color = cat_colors[ci]
            color_idx[cat] = color_idx.get(cat, 0) + 1

            marker = 'o' if cat == 'plastic' else ('D' if cat == 'heritage' else 's')
            ax.scatter(r['theta0_mrad'], r['dE_MeV'], c=color, s=150,
                      marker=marker, edgecolors='black', linewidth=0.8, zorder=5, label=r['name'])

        ax.set_xlabel(f"θ₀ at {p} GeV/c (mrad)", fontsize=13, fontweight='bold')
        ax.set_ylabel("ΔE (MeV)", fontsize=13, fontweight='bold')
        ax.set_title(f"BeamScan Classification Plot\n{req['author']}: {req['description']}",
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.15)

    plt.tight_layout()
    filepath = os.path.join(outdir, "classification.png")
    plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    return filepath


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BeamScan Highland Calculator")
    parser.add_argument("request_file", help="Path to YAML request file")
    parser.add_argument("--output-dir", default=None, help="Output directory (auto-generated if not set)")
    args = parser.parse_args()

    # Load request
    print(f"📄 Loading request: {args.request_file}")
    req = load_request(args.request_file)
    print(f"   Author: {req['author']}")
    print(f"   Description: {req['description']}")
    print(f"   Materials: {[m['name'] for m in req['materials']]}")
    print(f"   Momenta: {req['beam']['momenta_GeV']} GeV/c")

    # Create output directory
    if args.output_dir:
        outdir = args.output_dir
    else:
        stem = Path(args.request_file).stem
        outdir = f"results/{stem}"
    os.makedirs(outdir, exist_ok=True)

    # Compute predictions
    print(f"\n🔢 Computing Highland formula predictions...")
    results = compute_predictions(req)

    for r in results:
        print(f"   {r['name']:12s} @ {r['p_GeV']:4.1f} GeV/c → θ₀ = {r['theta0_mrad']:.3f} mrad, "
              f"ΔE = {r['dE_MeV']:.1f} MeV")

    # Generate outputs
    plots = []

    print(f"\n📊 Generating outputs in {outdir}/")

    # CSV table
    csv_path = generate_table_csv(results, outdir)
    print(f"   ✅ {csv_path}")

    # Distribution plot
    if req['outputs'].get('distribution_overlay', True):
        p = generate_distribution_plot(req, results, outdir)
        if p:
            plots.append(p)
            print(f"   ✅ {p}")

    # Classification plot
    if req['outputs'].get('classification_plot', True):
        p = generate_classification_plot(req, results, outdir)
        if p:
            plots.append(p)
            print(f"   ✅ {p}")

    # Markdown summary
    md_path = generate_markdown_summary(req, results, outdir, plots)
    print(f"   ✅ {md_path}")

    print(f"\n🎉 Done! Results in {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
