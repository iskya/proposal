#!/usr/bin/env python3
"""
pseudo_mc.py — Generate Highland-based pseudo-events for CI testing.

When Geant4 is not available (e.g. on GitHub Actions free tier),
this script generates Gaussian-sampled events matching Highland
formula predictions. This lets the full analysis pipeline run
end-to-end for testing, while clearly marking results as
"Highland pseudo-MC" rather than true Geant4.

Usage:
  python scripts/pseudo_mc.py requests/myfile.yaml results/geant4/
"""

import csv
import math
import os
import random
import sys

try:
    import yaml
except ImportError:
    os.system(f"{sys.executable} -m pip install pyyaml -q")
    import yaml

X0_DB = {
    "G4_POLYETHYLENE": 47.9, "G4_POLYPROPYLENE": 47.4,
    "G4_POLYSTYRENE": 42.5, "G4_PLEXIGLASS": 34.4,
    "G4_NYLON-6-6": 36.7, "G4_MYLAR": 28.7,
    "G4_POLYVINYL_CHLORIDE": 15.3,
    "Obsidian": 12.3, "G4_SILICON_DIOXIDE": 12.29,
    "G4_CALCIUM_CARBONATE": 10.7, "G4_ALUMINUM_OXIDE": 10.2,
    "G4_FERRIC_OXIDE": 5.5,
    "G4_Al": 8.90, "G4_Fe": 1.757, "G4_Pb": 0.561,
}

def highland_theta0(p_GeV, x_cm, X0_cm):
    p_MeV = p_GeV * 1000.0
    ratio = x_cm / X0_cm
    if ratio <= 0:
        return 0.0
    return (13.6 / p_MeV) * math.sqrt(ratio) * (1 + 0.038 * math.log(ratio))

def main():
    request_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results/geant4"

    with open(request_file) as f:
        req = yaml.safe_load(f)

    print(f"🎲 Generating Highland pseudo-MC events")
    print(f"   Request: {request_file}")

    for mat in req["materials"]:
        X0 = X0_DB.get(mat["geant4_name"], 30.0)
        x_cm = mat["thickness_mm"] / 10.0

        for p_GeV in req["beam"]["momenta_GeV"]:
            theta0 = highland_theta0(p_GeV, x_cm, X0)

            dirname = os.path.join(output_dir, f"{mat['name']}_{p_GeV}GeV_{mat['thickness_mm']}mm")
            os.makedirs(dirname, exist_ok=True)

            N = req["beam"].get("num_events", 10000)
            filepath = os.path.join(dirname, "events.csv")

            with open(filepath, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["event_id", "theta_x_mrad", "theta_y_mrad", "theta_3d_mrad"])
                for i in range(N):
                    tx = random.gauss(0, theta0) * 1000  # mrad
                    ty = random.gauss(0, theta0) * 1000
                    t3d = math.sqrt(tx**2 + ty**2)
                    w.writerow([i, f"{tx:.6f}", f"{ty:.6f}", f"{t3d:.6f}"])

            print(f"   ✅ {mat['name']:12s} @ {p_GeV} GeV/c → {N} events (θ₀={theta0*1000:.4f} mrad)")

    print(f"\n🎲 Pseudo-MC complete. Note: these are Gaussian (no Molière tails).")
    print(f"   For true non-Gaussian tails, run with actual Geant4.")

if __name__ == "__main__":
    main()
