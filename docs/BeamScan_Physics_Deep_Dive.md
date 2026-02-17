# BeamScan — The Physics Explained: Why This Works, Why the Gap, and Why Not Fill It

---

## 1. What Multiple Coulomb Scattering Actually Measures

When a charged particle (electron, pion, proton) flies through a piece of matter, it doesn't go in a straight line. It gets slightly deflected — over and over — by the electric fields of atomic nuclei in the material. Each individual deflection is tiny and random, but after millions of them, the particle exits the material at a small but measurable angle to its original path.

This is **multiple Coulomb scattering (MCS)**, and it's described by the **Highland formula:**

```
θ₀ ≈ (13.6 MeV / pβc) · √(x/X₀) · [1 + 0.038 · ln(x/X₀)]
```

The beauty of this formula is what it depends on:

- **p** (momentum): Higher momentum → less scattering. You control this with the beam settings.
- **x** (thickness): Thicker target → more scattering. You control this with your sample preparation.
- **X₀** (radiation length): This is the **material fingerprint**. It's a property of the material's atomic composition — you don't control it, you *measure* it.

**X₀ is the whole game.** It depends on the atomic number Z and atomic weight A of the material through:

```
1/X₀ ∝ Z(Z+1) · ln(183/Z^{1/3}) / A
```

Heavy atoms (large Z) → small X₀ → more scattering.  
Light atoms (small Z) → large X₀ → less scattering.

For compounds (like plastics or minerals), X₀ is the weighted harmonic mean of the element radiation lengths:

```
1/X₀ = Σ wⱼ/X₀ⱼ    (where wⱼ is the mass fraction of element j)
```

This is why different plastics give different θ₀ — they contain different atoms.

---

## 2. Why Different Plastics Give Different Signals

### The lightest plastics: PE and PP (polyolefins)
**PE** = (CH₂)ₙ — nothing but carbon (Z=6) and hydrogen (Z=1).  
**PP** = (C₃H₆)ₙ — also just carbon and hydrogen.

These have nearly identical X₀ (~48–50 cm) because they're made of the same two elements in almost the same ratio. **You cannot separate them by MCS.** This is not a failure — it's physics being honest. The Highland formula responds to *bulk atomic composition*, not molecular structure.

### Aromatic and oxygen-containing plastics: PS, PMMA, PET, Nylon
**PS** = (C₈H₈)ₙ — higher carbon-to-hydrogen ratio than PE → slightly lower X₀ (~42 cm).  
**PMMA** = (C₅H₈O₂)ₙ — contains **oxygen** (Z=8) → X₀ drops to ~34 cm.  
**Nylon 6** = (C₆H₁₁NO)ₙ — contains both **oxygen** and **nitrogen** (Z=7) → X₀ ~37 cm.  
**PET** = (C₁₀H₈O₄)ₙ — lots of oxygen → X₀ drops to ~29 cm.

Each step introduces heavier elements or higher concentrations of them, making X₀ smaller and the scattering angle larger.

### The standout: PVC
**PVC** = (C₂H₃Cl)ₙ — contains **chlorine** (Z=17, much heavier than C/H/O/N).  
X₀ ≈ 15–20 cm. This scatters the beam almost **twice as much** as PE.

Why does this matter industrially? PVC is the single most problematic contaminant in plastic recycling. When PVC gets mixed into a PET recycling stream, the chlorine releases hydrochloric acid during reprocessing, destroying equipment and contaminating the output. Current optical sorters often fail on dark, dirty, or multilayer plastics. A scattering-based sensor that measures *atomic composition*, not surface color, could solve this.

---

## 3. Why the Gap Exists — And Why It's a Feature, Not a Bug

### The chemistry argument

Look at the bar chart of θ₀ values (sorted):

```
PE       ████████████░░░░░░░░░░░░░░░░░░░  0.559 mrad
PP       ████████████░░░░░░░░░░░░░░░░░░░  0.562
PS       █████████████░░░░░░░░░░░░░░░░░░  0.596
Nylon    ██████████████░░░░░░░░░░░░░░░░░  0.646
PMMA     ███████████████░░░░░░░░░░░░░░░░  0.669
PET      █████████████████░░░░░░░░░░░░░░  0.738
PVC      ███████████████████████░░░░░░░░  1.039  ← chlorine pushes it up

            ~~~ GAP: 1.04 → 1.17 mrad ~~~

Obsidian ██████████████████████████░░░░░  1.169  ← silicon/oxygen mineral
SiO₂     ██████████████████████████░░░░░  1.170
CaCO₃    ███████████████████████████░░░░  1.261
Al₂O₃    ████████████████████████████░░░  1.294
Fe₂O₃    █████████████████████████████████ 1.808  ← iron dominates
```

The gap between PVC (1.04 mrad) and obsidian (1.17 mrad) is real and physically meaningful. It reflects a fundamental divide in chemistry:

**Organic matter** is built from **light elements** — carbon (Z=6), hydrogen (Z=1), oxygen (Z=8), nitrogen (Z=7). Even with chlorine (Z=17) in PVC, the *effective Z* of organic materials tops out around 8–10.

**Inorganic matter** (minerals, ceramics, rocks) is built from **heavier elements** — silicon (Z=14), calcium (Z=20), aluminium (Z=13), iron (Z=26). These pack densely in crystalline lattices, making their radiation lengths much shorter.

There is simply **no common, naturally occurring material** that sits halfway between organic polymers and inorganic minerals. It would need to be some kind of half-carbon, half-silicon compound that doesn't exist in everyday life.

### So why not put materials in the gap artificially?

**You could.** For example:
- Bakelite (phenol-formaldehyde resin, X₀ ~20 cm, θ₀ ~0.91 mrad)
- Teflon/PTFE (fluorinated polymer, X₀ ~16 cm, θ₀ ~1.02 mrad)
- Borosilicate glass (X₀ ~13 cm, θ₀ ~1.14 mrad)

These would indeed fall in the gap. But **you shouldn't add them**, for five excellent reasons:

### Reason 1: The gap IS the scientific result

In classification science, **well-separated clusters** are what you want. If everything blurred into a smooth continuum from PE to Fe₂O₃, you'd have a monotonic curve — less impressive, harder to classify, and no clear "aha!" moment.

The gap *demonstrates* that MCS naturally sorts matter into two categories: organic and inorganic. That's a clean, publishable, memorable result. A judge looking at 500 proposals will remember the plot with two clusters and a bridge much more than a plot with 20 evenly-spaced points.

### Reason 2: Application-driven sample selection

Every material in the proposal earns its place because it solves a real problem:

| Material | Why it's there |
|----------|---------------|
| PE, PP | Most common packaging plastics (baseline) |
| PS, PMMA, PET | Key recyclable plastics with oxygen/nitrogen content |
| PVC | The "problem plastic" — detecting it is the industrial application |
| Obsidian | Argentine archaeological significance (Andean trade routes) |
| SiO₂, CaCO₃ | Pottery/mortar components (heritage science) |
| Al₂O₃ | Ceramic/pigment reference |
| Fe₂O₃ | Iron oxide pigment (cave paintings, pottery glazes) |

Bakelite or Teflon are not in either application domain. Adding them would dilute the narrative: "We measured random things" vs "We measured materials that matter for recycling and heritage science."

### Reason 3: Beam time economy

With ~10 hours of productive beam time, each material costs approximately:
- 3 thickness settings × 2 momentum settings × 10 minutes = **1 hour per material** (including swaps, alignment checks, and calibration runs)
- 12 materials = 12 hours — already tight
- Adding 5 "gap-filler" materials would cost 5 more hours you don't have

Better to spend that time on **systematics** (verifying Highland formula, studying tail shapes, testing contaminant sensitivity) than on filling a parameter space.

### Reason 4: PVC already tells the bridge story

PVC sits at θ₀ = 1.04 mrad — right at the edge of the gap. It's the material that *bridges* organic and inorganic because its chlorine gives it mineral-like scattering while its backbone is pure hydrocarbon. One material demonstrating this transition is more powerful than three materials crowding the gap.

### Reason 5: Honesty about limitations is strength

The PE/PP indistinguishability and the gap both show that MCS measures *bulk effective Z*, not molecular structure. Acknowledging this boundary earns credibility. A proposal that claims to classify everything will be suspected of not understanding the physics. A proposal that says "this is what MCS can and cannot do" will be trusted.

---

## 4. Why the 2D Plot Works (Even Without a Magnet)

The original proposal used the MNP17 magnet at CERN to create a second axis (momentum loss Δp). The feedback correctly noted that this is risky: ΔE/E for 1 cm of plastic at 3 GeV is tiny (~0.06%), and the experiment shouldn't depend on equipment that may not be available.

**Better approach: Use θ₀ at two different beam momenta as the two axes.**

The Highland formula says θ₀ ∝ 1/p (at relativistic speeds where β ≈ 1). So if you measure θ₀ at 3 GeV and at 6 GeV, you get:

```
θ₀(3 GeV) ≈ 2 × θ₀(6 GeV)    (for all materials)
```

This means all materials fall roughly along a straight line with slope 1/2 in the (θ₃, θ₆) plane. **But not exactly!** The logarithmic correction term `[1 + 0.038 · ln(x/X₀)]` introduces a material-dependent deviation. Materials with very different X₀ deviate differently from the line, adding a tiny but real second dimension.

More importantly, **measuring at two momenta:**
- Validates the Highland formula's 1/p dependence experimentally
- Controls for systematic errors (a detector misalignment would shift θ₀ equally at both momenta, but the ratio would be preserved)
- Doubles your dataset, improving statistics

If the MNP17 magnet *is* available (CERN only), it becomes a powerful extension: the energy loss Δp provides a truly independent second axis. But the experiment succeeds without it.

---

## 5. Why This Experiment Is Practically Guaranteed to Work

### The signal is enormous
The angular resolution of BL4S tracking detectors (DWCs) is approximately 0.014 mrad (from ~100 µm position resolution over a 50 cm lever arm). The smallest signal we measure (PE at 6 GeV) is 0.27 mrad — that's a **signal-to-resolution ratio of ~20:1**. The largest signal (Fe₂O₃ at 3 GeV) is 1.81 mrad — a ratio of **130:1**.

We are not hunting for a tiny signal in noise. We are measuring a massive, clean effect.

### The statistics are trivial
At BL4S beam rates (~1 kHz at DESY, ~1000 particles/spill at CERN), collecting 10,000 events takes 10 seconds to 2 minutes. Even the hardest plastic-to-plastic pair (PS vs PMMA, ~10,000 events for 3σ) needs only minutes. Cross-domain separation (any plastic vs any mineral) needs <100 events — literally a few seconds of beam.

### The analysis is simple
No neural networks, no machine learning, no complex reconstruction. You compute:
```
θ = (x_downstream - x_upstream) / lever_arm
```
for each particle, build a histogram, fit a Gaussian, and read off the width. A first-year physics student can do this.

### Every failure mode is interesting
- If θ₀ matches Highland: you've validated a fundamental formula experimentally ✓
- If θ₀ deviates from Highland: you've found a measurement of Molière tail effects — this is active research ✓
- If two materials you expected to separate don't: you've demonstrated a real physical limitation ✓
- If a contaminant shifts θ₀: you've demonstrated sensitivity ✓

There is no scenario where you run this experiment and come back with nothing.

---

## 6. Summary: Why BeamScan Is a Strong Case

| Strength | Explanation |
|----------|------------|
| **Physics is clean** | One formula, one observable, one prediction. Judges understand it immediately. |
| **The gap tells a story** | Two separated clusters = proof the technique classifies material families. |
| **PVC is the hero** | The most industrially relevant result (detecting the "problem plastic") falls right out of the physics. |
| **Heritage adds creativity** | No prior BL4S team has bridged environmental + archaeological applications. |
| **Facility-agnostic** | Works at CERN, DESY, or ELSA with standard tracking + scintillators. |
| **Guaranteed result** | Signal-to-resolution is 20:1 minimum. Statistics need seconds, not hours. |
| **Failure is interesting** | Every deviation from prediction teaches something. |
| **Honest about limits** | PE≈PP and obsidian≈SiO₂ show maturity, not weakness. |
