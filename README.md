# 🔬 BeamScan — Particle-Beam Material Classification

**Los Topos Cósmicos · BL4S 2026 · Santa Rosa de Calamuchita, Córdoba, Argentina 🇦🇷**

[![GitHub Pages](https://img.shields.io/badge/🌐_Website-BeamScan_Atlas-2E75B6?style=for-the-badge)](https://los-topos-cosmicos.github.io/beam4school-proposal/)
[![Highland CI](https://img.shields.io/github/actions/workflow/status/los-topos-cosmicos/beam4school-proposal/highland-predict.yml?style=flat-square&label=Highland%20CI&logo=githubactions&logoColor=white)](https://github.com/los-topos-cosmicos/beam4school-proposal/actions/workflows/highland-predict.yml)
[![Geant4 CI](https://img.shields.io/github/actions/workflow/status/los-topos-cosmicos/beam4school-proposal/geant4-simulate.yml?style=flat-square&label=Geant4%20CI&logo=githubactions&logoColor=white)](https://github.com/los-topos-cosmicos/beam4school-proposal/actions/workflows/geant4-simulate.yml)

> *Can a beam of particles "read" what a material is made of?*

BeamScan measures how charged particles scatter inside different materials (Multiple Coulomb Scattering) to create a **fingerprint library** that classifies plastics for recycling and geological materials for heritage science.

---

## 🚀 Quick Start — For Students

**You don't need to know C++ or Geant4 to contribute.** The simulation system is designed so you can run predictions by editing a simple YAML text file.

### Step 1: Create a branch (team members) or fork (external)

**Team members** (added as collaborators): create a branch directly.
```bash
git checkout -b mi-experimento
```

**External contributors**: click the **Fork** button (top-right on GitHub). Note: automatic PR comments may not appear on fork PRs due to GitHub permissions -- check the Actions tab for your results instead.

### Step 2: Create your request file
Copy the template and give it your name:
```bash
cp requests/examples/template.yaml requests/yourname_experiment.yaml
```

### Step 3: Edit your request
Open your file and change:
- **Your name** (the `author:` field)
- **What you want to investigate** (the `description:` field)
- **Which materials** to simulate (pick from the list in the template)
- **Beam momenta** (e.g., `[3.0, 6.0]` for 3 and 6 GeV/c)

### Step 4: Push and watch the magic
```bash
git add requests/yourname_experiment.yaml
git commit -m "My first BeamScan simulation!"
git push
```

Go to the **Actions** tab on GitHub — you'll see your simulation running. In ~30 seconds, it produces:
- 📊 Scattering distribution plots
- 📈 Classification plot (2D fingerprint)
- 📋 Numerical comparison table
- 📝 Markdown summary with discrimination power matrix

### Step 5: Open a Pull Request
This posts your results as a comment on the PR so the whole team can see and discuss them.

---

## 🏗 Architecture — How It Works

```
┌───────────────┐     ┌─────────────────────┐     ┌───────────────┐
│   STUDENTS    │     │   GITHUB ACTIONS    │     │    RESULTS    │
│               │     │                     │     │               │
│ Edit YAML ────┼────►│ 1. Highland Calc ───┼────►│ predictions/  │
│ in requests/  │     │    (Python, ~30s)   │     │   *.csv       │
│               │     │                     │     │   *.png       │
│               │     │ 2. Geant4 Sim ──────┼────►│ geant4/       │
│               │     │    (Docker, ~15min) │     │   events.csv  │
│               │     │    [manual trigger] │     │               │
│               │     │ 3. Analysis ────────┼────►│ comparison/   │
│               │     │    (Python)         │     │   *.png       │
│               │     │                     │     │   SUMMARY.md  │
│ View results  │◄────┼─ 4. PR Comment ─────┤     │               │
│ on PR / Pages │     │    or commit back   │     │               │
└───────────────┘     └─────────────────────┘     └───────────────┘
```

### Two Modes

| Mode | Speed | What it does | When to use |
|------|-------|-------------|-------------|
| **Highland** (default) | ~30 seconds | Analytical formula prediction | Quick exploration, "what-if" studies |
| **Geant4** (manual) | ~15–60 min | Full Monte Carlo simulation | Final results, tail studies, validation |

Highland mode runs automatically on every push. Geant4 mode is **manual only**: a maintainer triggers it from the Actions tab (see `docs/CI_GEANT4.md`). It runs inside a Docker container on GitHub-hosted runners, no local installation needed.

---

## 📁 Repository Structure

```
bl4s-beamscan/
├── .github/workflows/
│   ├── highland-predict.yml    ← Runs instantly on every push to requests/
│   └── geant4-simulate.yml     ← Full Monte Carlo (manual trigger)
│
├── requests/                    ← 👈 STUDENTS EDIT FILES HERE
│   ├── examples/
│   │   └── template.yaml        ← Copy this to start
│   ├── full_classification.yaml ← All 12 materials
│   └── yourname_experiment.yaml ← Your custom study
│
├── results/                     ← 📊 AUTO-GENERATED (don't edit)
│   ├── full_classification/
│   │   ├── predictions.csv
│   │   ├── distributions.png
│   │   ├── classification.png
│   │   └── SUMMARY.md
│   └── geant4/                  ← Monte Carlo event data
│
├── simulation/                  ← Geant4 C++ source code
│   ├── CMakeLists.txt
│   ├── beamscan.cc              ← Main program
│   ├── src/                     ← Detector geometry, physics
│   ├── include/                 ← Headers
│   └── macros/                  ← Geant4 macro files
│
├── analysis/                    ← Python analysis scripts
│   ├── highland_calculator.py   ← Highland formula engine
│   └── analyze_geant4.py        ← Geant4 output analysis
│
├── data/
│   └── materials_database.json  ← Material properties reference
│
└── docs/
    └── guides/
        └── STUDENT_GUIDE.md     ← Detailed instructions
```

---

## 🧪 The Physics — What Are We Measuring?

When a fast particle (electron, pion, proton) passes through a material, it gets deflected by atomic nuclei. The total deflection follows the **Highland formula**:

```
θ₀ ≈ (13.6 MeV / p) · √(x/X₀) · [1 + 0.038 · ln(x/X₀)]
```

**X₀** (radiation length) is the material's fingerprint — it depends on which atoms the material is made of. Heavy atoms (like Cl in PVC, or Fe in iron oxide) give small X₀ and large scattering. Light atoms (like C and H in polyethylene) give large X₀ and small scattering.

### Our 12 Materials

| Material | X₀ (cm) | θ₀ at 3 GeV/c (mrad) | Category |
|----------|---------|----------------------|----------|
| PE | 47.9 | 0.559 | Plastic |
| PP | 47.4 | 0.562 | Plastic |
| PS | 42.5 | 0.596 | Plastic |
| Nylon | 36.7 | 0.646 | Plastic |
| PMMA | 34.4 | 0.669 | Plastic |
| PET | 28.7 | 0.738 | Plastic |
| **PVC** | **19.9** | **0.901** | **Plastic** ⚠️ |
| SiO₂ | 12.3 | 1.170 | Heritage |
| CaCO₃ | 8.7 | 1.411 | Heritage |
| Al₂O₃ | 7.1 | 1.575 | Heritage |
| Fe₂O₃ | 3.3 | 2.382 | Heritage |

PVC stands out from the other plastics because of its chlorine (Z=17). Detecting PVC contamination in recycling streams is the primary industrial application.

---

## 💡 Investigation Ideas for Students

Here are things you can explore by creating request files:

### Beginner
- **"Which plastics can we tell apart?"** — Simulate PE, PS, PET, PVC at 3 GeV/c and look at the distribution overlay
- **"Does beam energy matter?"** — Same materials at 1, 3, 6 GeV/c — how does separation change?
- **"Thick vs thin targets"** — Same material at 5, 10, 20 mm — does more material help?

### Intermediate
- **"Can we detect a metal contaminant?"** — Simulate PE, then PE + thin Al or Fe layer
- **"Pions vs electrons"** — Same material, different beam particles
- **"The PE vs PP challenge"** — Can you find ANY setting that separates them?

### Advanced
- **"Optimize the experiment"** — What momentum + thickness gives the best classification?
- **"Heritage material identification"** — Can you tell quartz from calcite?
- **"Build the classification plot"** — Use two momenta to create a 2D fingerprint

---

## 🛠 For Developers — Building Geant4 Locally

If you want to run the full Monte Carlo simulation on your computer:

### Prerequisites
- Geant4 11.x ([installation guide](https://geant4.web.cern.ch/download/))
- CMake ≥ 3.16
- A C++17 compiler (gcc/clang)
- Python 3.8+ with numpy, matplotlib, scipy, pyyaml

### Build
```bash
cd simulation
mkdir build && cd build
cmake .. -DGeant4_DIR=/path/to/geant4/lib/cmake/Geant4
make -j$(nproc)
```

### Run
```bash
./beamscan -m macros/scan_plastics.mac
```

### Or use Docker
```bash
docker run -v $(pwd):/work -w /work/simulation/build \
  jeffersonlab/geant4:g4v11.3.2-fedora40 \
  bash -c "cmake .. && make -j4 && ./beamscan -m ../macros/scan_plastics.mac"
```

---

## 📆 Timeline

| Week | Target |
|------|--------|
| Feb 17–23 | ✅ Repo initialized, Highland calculator working, first student requests |
| Feb 24–Mar 2 | Full simulation matrix running in CI, proposal figures generated |
| Mar 3–9 | Proposal text finalized, video filmed |
| Mar 10–13 | Final review and submission (**deadline: Mar 13, 23:59 CEST**) |

---

## 🇦🇷 About Our Team

We are high-school students and teachers from Santa Rosa de Calamuchita, Córdoba, Argentina. Our city sits on the Río Santa Rosa at the foot of the Sierras, where Comechingón peoples carved morteros into the rock two thousand years ago. Today, Córdoba's growing recycling cooperatives face the challenge of identifying mixed plastics without expensive lab equipment.

BeamScan bridges these two worlds: the same physics that can fingerprint an ancient quartz mortero can also detect PVC contamination in a recycling stream.

---

## 📚 References

- [BL4S 2026 — Beams & Detectors](https://beamlineforschools.cern/wp-content/uploads/2026/02/Beams_Detectors_BL4S_2026.pdf)
- [BL4S 2026 — Example Experiments](https://beamlineforschools.cern/wp-content/uploads/2026/02/BL4S_Example_Experiments_2026.pdf)
- [PDG — Passage of Particles Through Matter](https://pdg.lbl.gov/)
- [Highland, V.L. — "Some practical remarks on multiple scattering" (NIM 1975)](https://doi.org/10.1016/0029-554X(75)90743-0)
- [Geant4 Documentation](https://geant4.web.cern.ch/)

---

*Los Topos Cósmicos · Instituto San Francisco de Asís, Santa Rosa de Calamuchita, Córdoba 🇦🇷*
*Built with ❤️ for [CERN Beamline for Schools 2026](https://beamlineforschools.cern) · ¡La física fundamental es para todos!*
