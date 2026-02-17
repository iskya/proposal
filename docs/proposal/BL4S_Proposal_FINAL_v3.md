# BeamScan: A Particle-Beam Material Classifier — From Recycling to Heritage Science

## 1. Motivation to Participate (~100 words)

We are a team of high-school students from Córdoba, Argentina. In our country, plastic waste clogs rivers from the Pampas to Patagonia, and pre-Columbian obsidian tools still emerge from Andean soil. These challenges — sorting plastic for recycling and identifying archaeological materials — share a surprising connection: both can be addressed by measuring how a particle beam scatters inside matter. We want to show that the same physics probing the structure of the universe can serve society. Participating in BL4S would let us test this idea with a real accelerator beam and bring the results home to our communities.

---

## 2. Experiment Idea (~800 words)

### The Question

Can we build a "BeamScan Atlas" — a classification chart that identifies materials by measuring how charged particles scatter through them? We aim to demonstrate this for two real-world domains in one experiment: identifying plastics for recycling quality control and classifying geological reference materials relevant to heritage science.

### The Physics

When a GeV-scale charged particle traverses matter, it undergoes many small deflections from atomic nuclei — multiple Coulomb scattering (MCS). The resulting angular spread θ₀ depends on the material's radiation length X₀, a fundamental property set by elemental composition. From the Highland formula, θ₀ ∝ √(x/X₀) / p, where x is target thickness and p is beam momentum. Materials with heavier atoms scatter the beam more.

### Predicted Separation

Using the Highland formula with PDG radiation lengths, we calculated expected scattering angles at 3 GeV/c through 10 mm targets:

**Polyolefins** PE and PP (θ₀ ≈ 0.56 mrad): pure carbon–hydrogen baseline. **Oxygen-containing plastics** PS, PMMA, PET (θ₀ = 0.60–0.74 mrad): progressively heavier effective composition. **PVC** (θ₀ ≈ 1.04 mrad): the chlorine atom (Z = 17) dramatically increases scattering — detecting PVC contamination in recycling streams is our primary industrial application. **Geological materials** — obsidian, quartz, calcite, alumina, iron oxide (θ₀ = 1.17–1.81 mrad): minerals built from silicon, calcium, aluminium, and iron scatter even more.

A natural gap separates the plastics cluster from the minerals cluster, reflecting a fundamental divide in chemistry: organic matter (C, H, O, N) versus inorganic matter (Si, Ca, Al, Fe). This gap is itself the scientific result — it proves MCS naturally sorts materials into compositional families.

Our Geant4 simulations confirm that even the closest useful pairs (PS vs PMMA) need fewer than 2,000 events to distinguish at 3σ — seconds of beam time. PVC versus PE needs only ~50 events. The full 12-material atlas requires under one hour of data.

### Experimental Setup (facility-agnostic)

Our core measurement requires only tracking detectors and trigger scintillators — standard equipment at all BL4S facilities:

Beam (1–6 GeV/c) → scintillator S₁ → trackers 1, 2 → TARGET → trackers 3, 4 → scintillator S₂.

Two trackers upstream record the incoming direction; two downstream record the direction after the target. The difference gives the scattering angle per particle. This layout works at CERN T9 (Delay Wire Chambers), at DESY (beam telescope), and at ELSA.

### Targets

**Plastics (primary):** PE, PP, PS, PMMA, PET, Nylon, PVC — commercial flat sheets, 5/10/20 mm thick.

**Heritage/geology references (extension):** obsidian, fused quartz, calcite, alumina, iron oxide — commercially available certified mineral standards.

All targets are non-biological, solid samples mounted in a rigid aluminium holder. No cutting, heating, or friction occurs at the beamline. We will provide safety data sheets and coordinate with BL4S safety officers before the experiment. If any polymer is not permitted at the assigned facility, we will substitute non-combustible low-Z references (e.g. graphite) and proceed with the geological set, which is inherently non-combustible and still demonstrates the full classifier concept.

### Measurement Program

**Phase 1 — Calibration:** No-target runs at both momenta to measure beam divergence and detector angular resolution.

**Phase 2 — Core atlas:** Measure θ₀ for all materials at two momenta (3 and 6 GeV/c), 10 mm thickness, ≥10⁴ events each. Extract θ₀ from Gaussian fits. Compare with Highland and Geant4 predictions.

**Phase 3 — Systematics:** Vary thickness (5, 10, 20 mm) for selected materials to validate the √(x/X₀) dependence and extract X₀.

**Phase 4 — Extensions (if time allows):** Contaminant sensitivity (thin metal foil inside plastic), composite sample scanning, and particle species comparison at CERN T9.

### The Deliverable: BeamScan Atlas

A 2D classification plot: x-axis = θ₀ at 3 GeV/c, y-axis = θ₀ at 6 GeV/c. Each material forms a cluster with uncertainty ellipses. This "two-momentum fingerprint" requires no magnetic spectrometer, works at any BL4S facility, and cancels many systematic effects while amplifying material-dependent differences. The atlas is simultaneously a scientific result, a practical lookup table, and a memorable visualization.

### Simulation and Open Science

We have built a Geant4 Monte Carlo simulation of the full experiment, published in a public GitHub repository with automated CI workflows that build, simulate, and generate the atlas plots on every code change. Students contribute by adding materials via pull requests and instantly seeing updated predictions. The repository is already public — anyone can reproduce every figure in this proposal.

---

## 3. What We Hope to Take Away (~100 words)

We want to return to Córdoba with three things: a validated BeamScan Atlas proving that particle beams can classify materials non-destructively; the experience of designing, running, and analysing a real experiment at a world-class facility; and a story to share. If students from Argentina can use a CERN beamline to help solve recycling challenges and study their country's archaeological heritage, it shows that fundamental physics belongs to everyone. We will share our results with local recycling cooperatives, schools, and museums — and publish everything openly so others can build on our work.

---

## Outreach Activity (optional, ~200 words)

Before submitting this proposal, our team organised a "Physics Meets the Street" event at a recycling cooperative in Córdoba. We brought samples of different plastics and demonstrated — using simple density-sorting and light-transmission tests — how difficult it is to distinguish PE from PP or detect PVC contamination by eye. We explained our BL4S idea: that accelerator particles can "see" the atoms inside a material, not just the surface. Workers were fascinated that CERN science could connect to their daily sorting challenges.

If selected, we will expand this into a bilingual workshop series: "Del acelerador al reciclaje" (From the Accelerator to Recycling). We will bring our BeamScan Atlas to schools and cooperatives across Córdoba, explaining the physics behind each data point. We will share our Geant4 simulation as an open educational resource with Spanish-language tutorials, so students across Latin America can run their own predictions. Our GitHub repository is already public — anyone can reproduce every figure in this proposal.
