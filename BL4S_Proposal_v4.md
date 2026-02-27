# BeamScan: A Particle-Beam Material Classifier — From Recycling to Heritage Science

## 1. Motivation to Participate (~100 words)

We are students and teachers from Instituto San Francisco de Asís in Santa Rosa de Calamuchita, Córdoba, Argentina. Two thousand years ago, the Comechingón people carved and shaped the rock of our region — and the artefacts preserved in our local museum still cannot be studied without risking damage. Today, mixed plastic waste arrives at recycling facilities where a single piece of PVC, invisible to the naked eye, can contaminate and ruin the entire batch. Two problems, two thousand years apart, the same challenge: characterising a material without destroying it. Particle physics offers one answer for both: multiple Coulomb scattering reveals atomic composition through deflection angles alone. Participating in BL4S means we can finally prove it works.

---

## 2. Experiment Idea (~800 words)

### The Question

Can we build a "BeamScan Atlas" — a classification chart that identifies materials by measuring how charged particles scatter through them? We aim to demonstrate this for two real-world domains in one experiment: identifying plastics for recycling quality control and classifying geological reference materials relevant to heritage science.

### The Physics

When a charged particle travels through matter at GeV-scale energies, it does not go straight — it deflects slightly each time it passes near an atomic nucleus. The cumulative effect of thousands of these tiny deflections is called multiple Coulomb scattering (MCS). The resulting angular spread θ₀ follows the Highland formula:
<img width="940" height="538" alt="θ₀ ≈ (13 6 MeV  p) · √(xX₀) ·  1 + 0 038 · ln(xX₀)" src="https://github.com/user-attachments/assets/a235afa4-a6a9-4aad-b639-27e31292b99e" />

### Predicted Separation

Using the Highland formula with PDG radiation lengths, we calculated expected scattering angles at 3 GeV/c through 10 mm targets. The results split naturally into two families: 
Plastics (C, H, O, N — light atoms, long X₀): PE and PP sit at θ₀ ≈ 0.56 mrad, a pure carbon-hydrogen baseline. PS, PMMA and PET follow at 0.60–0.74 mrad. PVC stands apart at θ₀ ≈ 0.90 mrad — its chlorine atom (Z = 17) dramatically increases scattering. Detecting that contamination in recycling streams is our primary industrial application.
Geological materials (Si, Ca, Al, Fe — heavier atoms, short X₀): quartz, calcite, alumina and iron oxide scatter at θ₀ = 1.17–2.38 mrad, well separated from the plastics cluster.
The gap between both families is itself the scientific result — it proves MCS naturally sorts materials into compositional families. Our Geant4 simulations confirm that even the closest pairs (PS vs PMMA) need fewer than 2,000 events at 3σ — seconds of beam time. PVC versus PE needs only ~50 events. The full atlas requires under one hour of data.

### Experimental Setup (facility-agnostic)

Our core measurement requires only tracking detectors — standard equipment at all BL4S facilities:
Beam → trackers 1, 2 → TARGET → trackers 3, 4
Two trackers upstream measure the incoming particle direction. Two downstream measure the direction after the target. Subtracting the beam's natural divergence — measured in dedicated no-target runs — we extract the scattering signal from the material alone. The layout works with Delay Wire Chambers at CERN, beam telescopes at DESY, and the available tracking detectors at ELSA.

### Targets and Facility Considerations

CERN and DESY accept only non-combustible targets; ELSA also permits combustible materials. Our plan adapts accordingly: **at ELSA** we run the full set — plastics (PE, PP, PS, PMMA, PET, Nylon, PVC) plus geological references (quartz, calcite, alumina, iron oxide), including the PVC "hero result." **At CERN or DESY** we run the non-combustible set — geological references plus graphite and metal foils as low-Z anchors — still spanning a wide X₀ range and demonstrating the full classifier concept.

All targets are solid samples in an aluminium holder. We will provide safety data sheets and coordinate with facility safety officers.

### Measurement Program

**Phase 1 — Calibration:** No-target runs at both momenta to characterise beam divergence and detector resolution.
**Phase 2 — Core atlas:** All permitted materials at two momenta (3 and 6 GeV/c), 10 mm thickness, ≥10⁴ events each. The second momentum validates the 1/p scaling predicted by Highland and provides a built-in consistency check: systematic effects like alignment shifts affect both settings similarly.
**Phase 3 — Systematics:** Thickness variation (5, 10, 20 mm) for selected materials to validate the √(x/X₀) dependence and extract X₀ directly from data.
**Phase 4 — Extensions:** Contaminant sensitivity (a thin metal foil hidden inside a plastic sample), composite materials, and — where the facility offers additional observables such as dE/dx or calorimeter response — a second independent classification axis.

### The Deliverable: BeamScan Atlas

A measured classification table and plot of θ₀ (and extracted X₀) for each material. Every data point has a physical interpretation, a real-world application, and a simulation prediction to compare against. Where the facility offers additional observables, we add a second independent axis (e.g. dE/dx or calorimeter response). The atlas is simultaneously a scientific result, a practical reference, and a memorable visualisation.

### Simulation and Open Science

We have built a Geant4 Monte Carlo simulation of the full experiment, published in a public GitHub repository. Students contribute by editing simple YAML files and instantly seeing updated predictions — no C++ or Geant4 needed. Every figure in this proposal can be reproduced from the repository.
Our simulations (Geant4 11.3.2, FTFP_BERT, 2,000 events per configuration) validate the Highland predictions: across 10 of 11 materials at two momenta, Geant4 consistently exceeds Highland by 12 ± 3%, attributable to nuclear elastic scattering that the analytic formula omits. The exception is Fe₂O₃, where the ratio rises to ~1.4–1.5 — directly revealing the larger hadronic cross-section of iron nuclei, itself a measurable result from the same setup

---

## 3. What We Hope to Take Away (~100 words)

We want to return to Córdoba with three things: a validated BeamScan Atlas proving that particle beams can classify materials non-destructively; the experience of designing, running, and analysing a real experiment at a world-class facility; and a story to share. If students from Argentina can use a CERN beamline to help solve recycling challenges and study their country's archaeological heritage, it shows that fundamental physics belongs to everyone. We will share our results with local recycling cooperatives, schools, and museums — and publish everything openly so others can build on our work.

---

## Outreach Activity (optional, ~200 words)

Before submitting this proposal, our team organised a "Physics Meets the Street" event at a recycling cooperative in Córdoba. We brought samples of different plastics and demonstrated — using simple density-sorting and light-transmission tests — how difficult it is to distinguish PE from PP or detect PVC contamination by eye. We explained our BL4S idea: that accelerator particles can "see" the atoms inside a material, not just the surface. Workers were fascinated that CERN science could connect to their daily sorting challenges.

If selected, we will expand this into a bilingual workshop series: "Del acelerador al reciclaje" (From the Accelerator to Recycling). We will bring our BeamScan Atlas to schools and cooperatives across Córdoba, explaining the physics behind each data point. We will publish the Geant4 simulation and Spanish-language tutorials so students across Latin America can run their own predictions and extend the atlas with new materials.
