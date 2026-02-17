<div align="center">

# 🔬 BeamScan

### A Particle-Beam Material Classifier
### From Recycling to Heritage Science

**BL4S 2026 Proposal — Córdoba, Argentina 🇦🇷**

*Facility-agnostic: works at CERN T9 · DESY · ELSA*

---

</div>

## Motivation

We are a team of high-school students from Córdoba, Argentina. In our country, plastic waste clogs rivers from the Pampas to Patagonia, and pre-Columbian obsidian tools still emerge from Andean soil. These challenges — sorting plastic for recycling and identifying archaeological materials — share a surprising connection: both can be addressed by measuring how a particle beam scatters inside matter.

We want to show that the same physics probing the structure of the universe can serve society. Participating in BL4S would let us test this idea with a real accelerator beam and bring the results home to our communities.

---

## The Experiment

### Can we build a "BeamScan Atlas"?

A classification chart that identifies materials by measuring how charged particles scatter through them — demonstrated for two real-world domains in one experiment: **recycling quality control** and **heritage science**.

### The Physics: Multiple Coulomb Scattering

When a GeV-scale charged particle traverses matter, it undergoes many small deflections from atomic nuclei — **multiple Coulomb scattering (MCS)**. The resulting angular spread θ₀ depends on the material's **radiation length X₀**, a fundamental property set by elemental composition:

$$\theta_0 \approx \frac{13.6 \text{ MeV}}{p\beta c}\sqrt{\frac{x}{X_0}}\left[1 + 0.038\ln\left(\frac{x}{X_0}\right)\right]$$

Materials with heavier atoms have shorter X₀ and scatter the beam more. This is the fingerprint.

### Predicted Separation (Highland formula, PDG X₀ values)

All predictions are for **10 mm targets at 3 GeV/c**:

| Material | X₀ (cm) | θ₀ (mrad) | Category | Application |
|----------|---------|-----------|----------|-------------|
| PE | 47.9 | 0.559 | Plastic | Most common packaging |
| PP | 47.4 | 0.562 | Plastic | Food containers |
| PS | 42.5 | 0.596 | Plastic | Disposable cups, foam |
| Nylon | 36.7 | 0.646 | Plastic | Textiles, fishing nets |
| PMMA | 34.4 | 0.669 | Plastic | Plexiglass, lenses |
| PET | 28.7 | 0.738 | Plastic | Beverage bottles |
| **PVC** | **15.3** | **1.039** | **Plastic** | ⚠️ **Problem plastic — contaminates recycling** |
| | | *— gap —* | | *Organic ↔ inorganic divide* |
| Obsidian | 12.3 | 1.169 | Heritage | Pre-Columbian tools |
| SiO₂ | 12.3 | 1.170 | Heritage | Pottery, geological ref. |
| CaCO₃ | 10.7 | 1.261 | Heritage | Marble, mortar, limestone |
| Al₂O₃ | 10.2 | 1.294 | Heritage | Ceramics, gemstones |
| Fe₂O₃ | 5.5 | 1.808 | Heritage | Red ochre pigment |

**PVC jumps out** because chlorine (Z = 17) is much heavier than C/H/O/N. Detecting PVC contamination in recycling streams is our **hero result**.

### Why the Gap Is a Feature, Not a Bug

<div align="center">
<img src="docs/figures/physics_explanation_gap.png" width="800" alt="Why the gap exists between plastics and minerals">

*Common matter splits into organic (C/H/O/N → long X₀) and inorganic (Si/Ca/Al/Fe → short X₀). No everyday natural material bridges this divide.*
</div>

The gap proves **MCS naturally sorts materials into compositional families**. Two clear clusters + one bridge (PVC) is more memorable, more classifiable, and a stronger scientific result than a smooth continuum.

### Discrimination Power

<div align="center">
<img src="docs/figures/discrimination_matrix.png" width="700" alt="Events needed for 3σ separation">

*Green = trivial (<100 events). Yellow = feasible (100–10k). Red = impractical (>100k). PE vs PP is honestly impossible. PVC vs anything is trivial.*
</div>

Our Geant4 simulations confirm that even PS vs PMMA needs fewer than 2,000 events (seconds of beam time). **PVC versus PE needs only ~50 events.** The full 12-material atlas requires under one hour.

---

## Experimental Setup

<div align="center">
<img src="docs/figures/setup_schematic.png" width="800" alt="Facility-agnostic BeamScan layout">

*Beam → Scintillator → Tracker₁,₂ → **TARGET** → Tracker₃,₄ → Scintillator. Magnet and calorimeter are optional extensions.*
</div>

This is **facility-agnostic** — works with DWCs at CERN T9, beam telescopes at DESY, or tracking detectors at ELSA. No magnetic spectrometer required for the core measurement.

### Targets and Facility Considerations

BL4S rules: CERN and DESY accept only **non-combustible** targets; ELSA also permits combustible materials.

**At ELSA (preferred for full programme):** Plastics (PE, PP, PS, PMMA, PET, Nylon, PVC) + geological references (obsidian, quartz, calcite, alumina, iron oxide). This gives us the PVC "hero result" and the recycling application.

**At CERN or DESY:** Non-combustible set only — geological references + graphite + metal foils (Al, Fe) as low-Z anchors. Still spans a wide X₀ range and demonstrates the full classifier concept.

All targets are solid samples in an aluminium holder. Safety data sheets provided.

### Measurement Program

| Phase | What | Events | Time |
|-------|------|--------|------|
| **1. Calibration** | No-target runs at both momenta | 10⁴ | ~30 min |
| **2. Core Atlas** | All materials × 2 momenta (3 & 6 GeV/c) × 10 mm | ≥10⁴ each | ~6 hours |
| **3. Systematics** | Thickness variation (5/10/20 mm) for PE, PVC, CaCO₃ | ≥10⁴ each | ~3 hours |
| **4. Extensions** | Contaminant sensitivity, composites, particle species | as available | remaining |

---

## The Deliverable: BeamScan Atlas

<div align="center">
<img src="docs/figures/classification_plot_final.png" width="800" alt="BeamScan Atlas — 2D classification plot">

*Each material is a point in (θ₀ at 3 GeV/c, θ₀ at 6 GeV/c) space. Plastics cluster bottom-left, minerals top-right, PVC bridges the gap. Both axes are MCS-only — no magnet needed.*
</div>

This single figure shows all materials on one plot. But note an important subtlety: because θ₀ ∝ 1/p exactly (the log correction has no momentum dependence), **all materials fall on the same line** with slope = p₁/p₂. The two-momentum plot is a **consistency check and systematics control**, not an independent second discriminator. Classification power comes from the extracted X₀ values. Where the facility offers additional observables (dE/dx, calorimeter response), we add a genuinely independent second axis.

---

## Simulation & Open Science

We built a complete simulation pipeline in this repository:

```
Student edits YAML → git push → GitHub Actions → plots appear in PR comment (~30 sec)
```

**Students don't need C++ or Geant4.** They copy a template, choose materials and beam energies, and push. The Highland calculator does the rest. See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/guides/STUDENT_GUIDE.md](docs/guides/STUDENT_GUIDE.md).

### Repository Structure

```
├── requests/                ← 👈 Students edit YAML files here
│   ├── template.yaml         ← Copy this to start
│   └── examples/             ← Named examples (Valentina, Tomás, Lucía, Sofía)
├── analysis/                ← Highland calculator + Geant4 analyzer
├── simulation/              ← Full Geant4 C++ source code
├── results/                 ← Auto-generated by CI
├── docs/                    ← Figures, guides, physics deep-dive
├── .github/workflows/       ← Automation (Highland + Geant4 + Pages)
└── schemas/                 ← YAML validation schema
```

### Automation

| Workflow | Trigger | Speed | What it produces |
|----------|---------|-------|-----------------|
| 🔬 Highland Prediction | Every push/PR to `requests/` | ~30 sec | Plots + tables + PR comment |
| ⚛️ Geant4 Simulation | Manual (self-hosted runner) | ~15 min | Full MC event data |
| 📖 Publish Atlas | After Highland runs on main | ~1 min | GitHub Pages site |

---

## What We Hope to Take Away

We want to return to Córdoba with three things:

1. A **validated BeamScan Atlas** proving that particle beams can classify materials non-destructively
2. The **experience** of designing, running, and analysing a real experiment at a world-class facility
3. A **story to share** — if students from Argentina can use a CERN beamline to help solve recycling challenges and study their country's archaeological heritage, it shows that fundamental physics belongs to everyone

We will share our results with local recycling cooperatives, schools, and museums — and publish everything openly so others can build on our work.

---

## Outreach

Before submitting this proposal, we organised a **"Physics Meets the Street"** event at a recycling cooperative in Córdoba. We brought samples of different plastics and demonstrated how difficult it is to distinguish PE from PP or detect PVC contamination by eye. We explained our BL4S idea: that accelerator particles can "see" the atoms inside a material, not just the surface.

If selected, we will expand this into a bilingual workshop series: **"Del acelerador al reciclaje"** (From the Accelerator to Recycling). We will bring our BeamScan Atlas to schools and cooperatives across Córdoba, and share our Geant4 simulation as an open educational resource with Spanish-language tutorials for students across Latin America.

---

## For Teachers & Collaborators

### Quick Start

```bash
# Clone
git clone https://github.com/artfisica/beam4school-proposal.git
cd beam4school-proposal

# Run a prediction locally (optional — CI does this automatically)
pip install -r analysis/requirements.txt
python analysis/highland_calculator.py requests/full_classification.yaml --output-dir results/full_classification

# Or use Docker
docker build -t beamscan .
docker run --rm -v $PWD:/work beamscan
```

### Student Workflow

1. Fork → copy `requests/template.yaml` → edit materials/momenta → push → open PR
2. GitHub Actions validates the YAML, runs predictions, posts results as PR comment
3. Team reviews and discusses in the PR thread
4. Merge to main → results committed to `results/` → Pages updated

### Key Documents

| Document | Purpose |
|----------|---------|
| [PROPOSAL.md](PROPOSAL.md) | This document — the canonical proposal |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How students contribute |
| [docs/guides/STUDENT_GUIDE.md](docs/guides/STUDENT_GUIDE.md) | Bilingual (ES/EN) step-by-step guide |
| [docs/SIMULATION_PLAN.md](docs/SIMULATION_PLAN.md) | Full simulation architecture |
| [docs/CI_GEANT4.md](docs/CI_GEANT4.md) | Setting up the Geant4 self-hosted runner |
| [docs/BeamScan_Physics_Deep_Dive.md](docs/BeamScan_Physics_Deep_Dive.md) | Comprehensive physics explanation |

### Geant4 Self-Hosted Runner

The Geant4 workflow runs on `[self-hosted, geant4]`. See [docs/CI_GEANT4.md](docs/CI_GEANT4.md) for setup instructions. The recommended approach is a dedicated Linux machine in Córdoba with Geant4 installed as a GitHub Actions runner.

---

## References

- [BL4S 2026 — Beams & Detectors](https://beamlineforschools.cern)
- [PDG — Passage of Particles Through Matter](https://pdg.lbl.gov/)
- Highland, V.L. — *"Some practical remarks on multiple scattering"* (NIM 1975)
- [Geant4 Documentation](https://geant4.web.cern.ch/)

---

<div align="center">

**Built with ❤️ in Córdoba, Argentina for CERN Beamline for Schools 2026**

🇦🇷 → 🔬 → 🌍

*¡La física fundamental es para todos!*

</div>
