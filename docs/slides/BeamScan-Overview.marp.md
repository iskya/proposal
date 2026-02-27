---
marp: true
theme: default
paginate: true
math: mathjax
style: |
  /* BeamScan site palette: #1A237E → #283593 → #1565C0 */
  :root {
    --indigo: #1A237E;
    --indigo-mid: #283593;
    --blue: #1565C0;
    --indigo-light: #E8EAF6;
    --indigo-muted: #C5CAE9;
    --text: #1a1a2e;
    --muted: #666;
    --green: #2E7D32;
    --orange: #E65100;
    --red: #C62828;
    --gold: #F57F17;
  }
  section {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: var(--text);
    background: #ffffff;
    padding: 40px 50px;
  }
  section::after {
    content: attr(data-marpit-pagination) ' / ' attr(data-marpit-pagination-total);
    font-size: 11px; color: var(--muted);
  }
  h1 { color: var(--indigo); font-size: 1.7em; border-bottom: 2px solid var(--indigo-light); padding-bottom: 8px; }
  h2 { color: var(--indigo-mid); font-size: 1.25em; margin-top: 0.3em; }
  h3 { color: var(--blue); font-size: 1.05em; }
  strong { color: var(--indigo); }
  em { color: var(--muted); }
  code { background: var(--indigo-light); color: var(--indigo); padding: 1px 5px; border-radius: 3px; font-size: 0.88em; }
  a { color: var(--blue); }
  table { font-size: 0.78em; border-collapse: collapse; width: 100%; }
  th { background: var(--indigo); color: white; padding: 6px 10px; text-align: left; }
  td { padding: 5px 10px; border-bottom: 1px solid #E0E0E0; }
  tr:nth-child(even) { background: #F5F5F5; }
  blockquote { border-left: 4px solid var(--blue); background: var(--indigo-light); padding: 12px 16px; margin: 12px 0; font-size: 0.9em; }
  /* Title slide */
  section.title { background: linear-gradient(135deg, #1A237E 0%, #283593 50%, #1565C0 100%); color: white; text-align: center; }
  section.title h1 { color: white; border: none; font-size: 2.4em; }
  section.title h2 { color: #C5CAE9; font-size: 1.1em; font-weight: 300; }
  section.title p { color: #E8EAF6; }
  section.title a { color: #90CAF9; }
  /* Section divider */
  section.divider { background: var(--indigo); color: white; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; }
  section.divider h1 { color: white; border: none; font-size: 2.2em; }
  section.divider p { color: var(--indigo-muted); }
  /* Two-column layout */
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  .cols-60-40 { display: grid; grid-template-columns: 3fr 2fr; gap: 24px; }
  .cols-40-60 { display: grid; grid-template-columns: 2fr 3fr; gap: 24px; }
  /* Cards */
  .card { background: #FAFAFA; border: 1px solid #E0E0E0; border-radius: 6px; padding: 14px; }
  .card-blue { border-left: 4px solid var(--blue); }
  .card-green { border-left: 4px solid var(--green); }
  .card-orange { border-left: 4px solid var(--orange); }
  .card-red { border-left: 4px solid var(--red); }
  /* Badge */
  .badge { display: inline-block; background: var(--indigo-light); color: var(--indigo); padding: 2px 10px; border-radius: 12px; font-size: 0.75em; font-weight: 600; }
  .badge-green { background: #E8F5E9; color: var(--green); }
  .badge-red { background: #FFEBEE; color: var(--red); }
  .badge-orange { background: #FFF3E0; color: var(--orange); }
  /* Footer */
  .footer { position: absolute; bottom: 20px; left: 50px; right: 50px; text-align: center; font-size: 0.65em; color: var(--muted); border-top: 1px solid #E0E0E0; padding-top: 6px; }
  /* Smaller text util */
  .sm { font-size: 0.82em; }
  .xs { font-size: 0.72em; }
---

<!-- _class: title -->
<!-- _paginate: false -->

# 🔬 BeamScan

## A Particle-Beam Material Classifier — From Recycling to Heritage Science

<br>

**Los Topos Cósmicos** · Instituto San Francisco de Asís
Santa Rosa de Calamuchita, Córdoba, Argentina 🇦🇷

*BL4S 2026 · Project Briefing for Teachers*

---

# What is BeamScan?

<div class="cols">
<div class="card card-blue">

### The Idea

Shoot a beam of particles (protons, pions, electrons) through a thin slab of material.

**Measure how much the beam spreads out** — Multiple Coulomb Scattering (MCS).

Different materials scatter differently → each gets a unique "fingerprint."

🟢 **Non-destructive:** the sample is untouched.

</div>
<div class="card card-green">

### Two Real-World Applications

**♻️ Recycling**
Detect PVC contamination in mixed plastic waste — a critical quality-control problem for cooperatives in Córdoba.

**🏛️ Heritage Science**
Classify geological materials in Comechingón *morteros* — 2,000-year-old stone tools that cannot be destroyed for analysis.

</div>
</div>

<div class="footer">Los Topos Cósmicos · BL4S 2026</div>

---

# The Physics — Multiple Coulomb Scattering

<div class="cols-60-40">
<div>

When a fast particle passes through matter, it bounces off atomic nuclei many times.

**Heavier atoms (higher Z) → more scattering → wider beam spread.**

The Highland formula gives the expected spread:

$$\theta_0 = \frac{13.6 \text{ MeV}}{p\beta c}\sqrt{\frac{x}{X_0}}\left[1 + 0.038\ln\frac{x}{X_0}\right]$$

Where: $x$ = thickness, $X_0$ = radiation length (material property), $p$ = momentum.

</div>
<div>

![w:420](../figures/physics_explanation_gap.png)

*The natural gap between plastics (C,H,O) and minerals (Si,Ca,Al,Fe) is the scientific result.*

</div>
</div>

<div class="footer">Los Topos Cósmicos · BL4S 2026</div>

---

# The BeamScan Atlas — 11 Materials, 2 Momenta

<div class="cols-60-40">
<div>

![w:520](../figures/classification_plot_final.png)

</div>
<div class="sm">

### Key Results

✅ 7 plastics + 4 minerals fully separable

✅ PVC detection needs only ~50 events (seconds of beam time)

✅ Full atlas in under 1 hour

✅ No magnet required — works at CERN, DESY, ELSA

✅ Two momenta provide built-in consistency check ($1/p$ scaling)

<div class="card card-green">

**Geant4 validated:** G4/Highland = **1.12 ± 0.03** across all light materials

</div>
</div>
</div>

<div class="footer">Los Topos Cósmicos · BL4S 2026</div>

---

# Geant4 Validates Highland — And Reveals Nuclear Scattering

<div class="cols">
<div>

| Material | 3 GeV/c | 6 GeV/c | Note |
|----------|---------|---------|------|
| PET | 1.097 | 1.090 | Closest to Highland |
| PVC | 1.101 | 1.108 | |
| Nylon | 1.105 | 1.113 | |
| PMMA | 1.112 | 1.111 | |
| PP | 1.120 | 1.122 | |
| CaCO₃ | 1.136 | 1.135 | |
| SiO₂ | 1.147 | 1.152 | |
| Al₂O₃ | 1.168 | 1.168 | |
| **Fe₂O₃** | **1.397** | **1.493** | ⚠️ Nuclear scattering |

</div>
<div class="card card-blue">

### What this means

**Highland underestimates by a consistent 12%** — because it only models electromagnetic scattering.

Geant4 adds nuclear elastic + inelastic scattering, which grows with Z.

**Fe₂O₃ is the proof:** iron nuclei (Z=26) have much larger hadronic cross-sections → 40–50% excess.

> *This deviation is itself a publishable result from BeamScan.*

</div>
</div>

<div class="footer">Los Topos Cósmicos · BL4S 2026</div>

---

# How It's Built — Technical Architecture

<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; font-size: 0.78em;">

<div class="card card-blue">

### 📝 Student Input
Students write YAML request files: materials, momenta, thickness.
No code needed — just a text editor.

</div>
<div class="card" style="border-left: 4px solid #00838F;">

### ⚡ Highland CI
GitHub Actions runs Highland predictions in ~30 seconds.
Produces plots + CSV + PR comment.
Triggered automatically on PR.

</div>
<div class="card card-green">

### ⚛️ Geant4 CI
Full Monte Carlo simulation.
Geant4 11.3.2 via Conda.
22 sims × 2,000 events.
Runs ~20 min on GitHub Actions.

</div>
<div class="card card-orange">

### 📊 Results
Auto-committed to repo:
`results/full_classification/`
`results/geant4_comparison/`
Predictions, plots, SUMMARY.md

</div>
<div class="card" style="border-left: 4px solid #6A1B9A;">

### 🌐 GitHub Pages
Live website with atlas plot, setup schematic, physics gap, discrimination matrix.
[los-topos-cosmicos.github.io](https://los-topos-cosmicos.github.io/beam4school-proposal/)

</div>
<div class="card" style="border-left: 4px solid #BF360C;">

### 📄 Proposal Docs
`PROPOSAL.md` + `BL4S_Proposal_v4.md`
Physics Deep Dive (2,200 words)
Student Guide (ES/EN)
Contributing guide

</div>
</div>

<div class="footer">Los Topos Cósmicos · BL4S 2026</div>

---

# Repository — 66 files, 100% open source

<div class="sm">

| Directory | Contents | Who uses it |
|-----------|----------|-------------|
| `requests/` | Student YAML files — material choices | 👩‍🎓 Students |
| `requests/examples/` | 4 named examples (Lucía, Valentina, Tomás, Sofía) | 👩‍🎓 Students |
| `analysis/` | `highland_calculator.py`, `analyze_geant4.py` | 🤖 CI / Teachers |
| `simulation/` | Geant4 C++ (`beamscan.cc`, 6 source files) | 🤖 CI |
| `scripts/` | `validate_requests.py`, `generate_macros.py`, `pseudo_mc.py` | 🤖 CI |
| `docs/` | GitHub Pages site, Physics Deep Dive, Student Guide | 📖 Everyone |
| `results/` | Committed predictions, plots, G4 comparison | 📊 Everyone |
| `.github/workflows/` | 2 CI pipelines (Highland + Geant4) | 🤖 CI |
| `materials/` | `materials.yaml` — 11 materials + Graphite fallback | 📖 Reference |
| `schemas/` | `request.schema.json` — validates YAML structure | 🤖 CI |

</div>

<div class="footer">Los Topos Cósmicos · BL4S 2026</div>

---

# How Students Participate

| Step | Action | Details |
|------|--------|---------|
| **1** | **Choose Materials** | Pick materials to study. *"I want to see if BeamScan can tell PVC apart from PE at 3 GeV."* |
| **2** | **Write a YAML File** | Copy a template, change materials, momenta, thickness. No programming — it's a structured text file. |
| **3** | **Open a Pull Request** | Push to a branch → open PR → GitHub Actions runs Highland predictions in 30 seconds. |
| **4** | **Interpret Results** | CI comments with plots + CSV. The student analyzes: *can these materials be distinguished? How many events are needed?* |

<br>

> 🟢 **Key:** Students never need to install Geant4, C++, or Python. Just a GitHub account and a text editor.

<div class="footer">Los Topos Cósmicos · BL4S 2026</div>

---

# What a Student Request Looks Like

<div class="cols">
<div>

```yaml
# valentina_pvc_detection.yaml
author: "Valentina García"
description: >
  Can BeamScan detect PVC contamination
  in a mixed plastic recycling stream?

materials:
  - name: PE
    geant4_name: G4_POLYETHYLENE
    thickness_mm: 10
  - name: PVC
    geant4_name: G4_POLYVINYL_CHLORIDE
    thickness_mm: 10

beam:
  particle: e-
  momenta_GeV: [3.0, 6.0]

num_events: 10000
```

</div>
<div class="sm">

### What happens next

1. Student commits this file
2. Opens a Pull Request
3. CI validates the YAML
4. Highland runs in **30 seconds**
5. Bot comments with plots:
   - `distributions.png`
   - `classification.png`
   - `predictions.csv`
   - `SUMMARY.md`
6. **Student writes analysis** in their PR description

</div>
</div>

<div class="footer">Los Topos Cósmicos · BL4S 2026</div>

---

# What's Missing — Next Steps

<div class="cols">
<div>

<div class="card card-red">

### 🔴 Before Submission (March deadline)
- Video (1 min): required by BL4S — team intro + experiment explanation
- Cost estimate: target materials, shipping to facility
- Team photo + teacher signatures on application form

</div>

<div class="card card-orange">

### 🟠 Technical Hardening
- Schema caps: limit `num_events`, momenta, materials per request
- Increase Geant4 statistics: 2,000 → 10,000+ events
- Add safety data sheets (SDS) for all target materials

</div>
</div>
<div>

<div class="card card-green">

### 🟢 Student Activities (can start now)
- Each student picks a research question → writes YAML → opens PR
- 4 named examples already exist: Valentina, Tomás, Lucía, Sofía
- Students write analysis in PR descriptions — builds proposal portfolio

</div>

<div class="card card-blue">

### 🔵 If Selected (summer/fall)
- Prepare physical target samples (aluminum holder, 5/10/20 mm slabs)
- Practice data analysis pipeline with simulated data
- Outreach: *"Del acelerador al reciclaje"* workshop at cooperatives

</div>
</div>
</div>

<div class="footer">Los Topos Cósmicos · BL4S 2026</div>

---

# Why This Proposal Stands Out

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; font-size: 0.82em;">

<div class="card card-blue">

**🔬 Real Science** — Not a textbook repeat. Original classification method with validated predictions. The G4/Highland ratio analysis is genuinely publishable.

</div>
<div class="card card-green">

**🏫 Student-First Design** — Students participate without installing C++ or Geant4. YAML → PR → results in 30 seconds. The barrier to entry is a text editor.

</div>
<div class="card card-blue">

**🌍 Social Impact** — Connects CERN physics to recycling cooperatives and indigenous heritage in Córdoba. Reviewers love proposals that matter beyond the lab.

</div>
<div class="card card-green">

**🔧 Facility-Agnostic** — Works at CERN T9, DESY, or ELSA. Adapts target list to safety rules. No magnet needed. Easy for BL4S organizers to schedule.

</div>
<div class="card card-blue">

**💻 Professional Infrastructure** — Public GitHub repo, CI/CD pipelines, automated Geant4. Most BL4S proposals don't have this. Shows the team can run the experiment.

</div>
<div class="card card-green">

**🇦🇷 Underrepresented Region** — Argentina teams are rare at BL4S. The Comechingón cultural connection and bilingual outreach plan are unique differentiators.

</div>
</div>

<div class="footer">Los Topos Cósmicos · BL4S 2026</div>

---

<!-- _class: title -->
<!-- _paginate: false -->

# 🔗 Ready to Explore

<br>

| | |
|---|---|
| 🌐 **Website** | [los-topos-cosmicos.github.io/beam4school-proposal](https://los-topos-cosmicos.github.io/beam4school-proposal/) |
| 📂 **Repository** | [github.com/los-topos-cosmicos/beam4school-proposal](https://github.com/los-topos-cosmicos/beam4school-proposal) |
| 📄 **Proposal** | `PROPOSAL.md` |
| 🤝 **Contributing** | `CONTRIBUTING.md` |
| 📖 **Student Guide** | `docs/guides/STUDENT_GUIDE.md` |

<br>

Los Topos Cósmicos · Instituto San Francisco de Asís
Santa Rosa de Calamuchita, Córdoba, Argentina 🇦🇷
*¡La física fundamental es para todos!*
